"""
Training loop for the Phase 1 pilot.

Trains one instance of the 150M Llama-style model from scratch, with:
  - bfloat16 mixed-precision autocast (recommended on Ada/Blackwell)
  - Gradient accumulation to reach the effective batch size
  - Linear warmup followed by cosine decay to a floor
  - AdamW with standard Llama-style hyperparameters
  - Dense log-spaced checkpoints (from TrainingConfig.checkpoint_schedule())
  - Periodic held-out evaluation
  - Informative per-step console output (loss, lr, throughput, ETA)
  - CSV logging of every optimizer step

The console output is designed to be informative without being noisy.
Per-step lines are logged every TrainingConfig.log_every steps (default
50), eval lines roughly every eval_every steps (default 500), and
checkpoint lines on the log-spaced schedule.

Single-GPU only. No DDP, no FSDP — Phase 1 is a single-workstation
project. If multi-GPU is needed later, this loop should be rewritten.

Public entry points:
  - train_one_run(model_cfg, train_cfg, seed, run_dir, num_proc=None):
        Runs one complete training pass. Saves checkpoints, metadata,
        and CSV log into run_dir.
"""

import os
import time
import math
import json
import csv
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW

from config import ModelConfig, TrainingConfig, save_config_pair
from models import (
    LlamaStyleTransformer, count_parameters, estimate_training_memory_gb,
)
from data import prepare_dataset, make_dataloaders


# ----------------------------------------------------------------------
# Learning rate schedule.
# ----------------------------------------------------------------------
def compute_lr(step: int, train_cfg: TrainingConfig) -> float:
    """
    Linear warmup from 0 to learning_rate over warmup_steps, then cosine
    decay from learning_rate down to learning_rate × lr_floor_ratio over
    the remaining (total_steps - warmup_steps) steps.
    """
    if step < train_cfg.warmup_steps:
        return train_cfg.learning_rate * (step + 1) / max(1, train_cfg.warmup_steps)
    progress = (step - train_cfg.warmup_steps) / max(
        1, train_cfg.total_steps - train_cfg.warmup_steps
    )
    progress = min(max(progress, 0.0), 1.0)
    lr_min = train_cfg.learning_rate * train_cfg.lr_floor_ratio
    return lr_min + 0.5 * (train_cfg.learning_rate - lr_min) * (
        1.0 + math.cos(math.pi * progress)
    )


# ----------------------------------------------------------------------
# Held-out evaluation.
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_held_out(model, eval_loader, device, autocast_dtype):
    """
    Compute the mean held-out cross-entropy loss over the eval set.
    Returns (mean_loss, n_tokens).

    Uses autocast for consistency with training. Token count is the
    total number of next-token predictions (excludes the first token
    of each sequence, which has no prediction target).
    """
    model.eval()
    total_loss_sum = 0.0
    total_tokens = 0
    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype, enabled=(device == "cuda")):
            _, loss, _ = model(input_ids, labels=input_ids)
        # CE was reduced to mean over (B × (T-1)) tokens. Reconstruct the sum.
        n = input_ids.size(0) * (input_ids.size(1) - 1)
        total_loss_sum += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss_sum / max(1, total_tokens), total_tokens


# ----------------------------------------------------------------------
# Console formatting helpers.
# ----------------------------------------------------------------------
def _format_time(seconds: float) -> str:
    """Format a number of seconds as 'XdYh', 'XhYm', or 'XmYs'."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m{seconds % 60:02.0f}s"
    if seconds < 86400:
        return f"{seconds / 3600:.0f}h{(seconds % 3600) / 60:02.0f}m"
    return f"{seconds / 86400:.1f}d"


def _format_throughput(tokens_per_sec: float) -> str:
    if tokens_per_sec >= 1e6:
        return f"{tokens_per_sec / 1e6:.2f}M tok/s"
    if tokens_per_sec >= 1e3:
        return f"{tokens_per_sec / 1e3:.1f}k tok/s"
    return f"{tokens_per_sec:.0f} tok/s"


def _print_startup_banner(
    device: str, model_cfg: ModelConfig, train_cfg: TrainingConfig,
    seed: int, run_dir: str, autocast_dtype,
):
    """Print a multi-line banner showing what we're about to train and
    where it'll fit in memory. Mirrors the spirit of the older codebase's
    startup output."""
    total, trainable = count_parameters(LlamaStyleTransformer(model_cfg))
    print()
    print("=" * 78)
    print(f"  Phase 1: Llama-style 150M training run")
    print("=" * 78)
    print(f"  Device:           {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU:              {gpu_name} ({total_vram:.1f} GB VRAM)")
        print(f"  Autocast dtype:   {autocast_dtype}")
    print(f"  Seed:             {seed}")
    print(f"  Run dir:          {run_dir}")
    print()
    print(f"  Model:            H={model_cfg.hidden_size}, "
          f"L={model_cfg.num_hidden_layers}, "
          f"I={model_cfg.intermediate_size}, "
          f"heads={model_cfg.num_attention_heads} (head_dim={model_cfg.head_dim})")
    print(f"  Vocab:            {model_cfg.vocab_size:,} (Mistral tokenizer)")
    print(f"  Trainable params: {trainable / 1e6:.2f}M")
    print(f"  Tied embeddings:  {model_cfg.tie_embeddings}")
    print(f"  Grad checkpoint:  {model_cfg.gradient_checkpointing}")
    print()
    # Memory estimate.
    mem = estimate_training_memory_gb(
        model_cfg,
        micro_batch_size=train_cfg.micro_batch_size,
        seq_len=train_cfg.train_seq_len,
        dtype_bytes=2,  # bfloat16
    )
    print(f"  Memory estimate (lower bound, doesn't include autograd scratch):")
    print(f"    weights:        {mem['weights_gb']:.2f} GB")
    print(f"    gradients:      {mem['gradients_gb']:.2f} GB")
    print(f"    optimizer:      {mem['optimizer_gb']:.2f} GB (AdamW: m, v, fp32 master)")
    print(f"    activations:    {mem['activations_gb']:.2f} GB (with gradient ckpt)")
    print(f"    estimated tot:  {mem['total_gb']:.2f} GB")
    if device == "cuda":
        headroom = total_vram - mem["total_gb"]
        print(f"    headroom:       ~{headroom:.1f} GB of {total_vram:.1f} GB VRAM")
        if headroom < 4.0:
            print(f"    ⚠️  Less than 4 GB headroom — autograd scratch may OOM. "
                  f"Consider reducing micro_batch_size from "
                  f"{train_cfg.micro_batch_size} to {max(1, train_cfg.micro_batch_size // 2)}.")
    print()
    # Training budget.
    eff_batch = train_cfg.micro_batch_size * train_cfg.grad_accum_steps
    tokens_per_step = eff_batch * train_cfg.train_seq_len
    total_tokens = tokens_per_step * train_cfg.total_steps
    print(f"  Training schedule:")
    print(f"    micro batch:    {train_cfg.micro_batch_size}")
    print(f"    grad accum:     {train_cfg.grad_accum_steps}")
    print(f"    effective batch:{eff_batch} sequences × {train_cfg.train_seq_len} tokens "
          f"= {tokens_per_step:,} tokens/step")
    print(f"    total steps:    {train_cfg.total_steps:,}")
    print(f"    total tokens:   {total_tokens / 1e9:.2f}B")
    print(f"    optimizer:      AdamW(lr={train_cfg.learning_rate}, "
          f"wd={train_cfg.weight_decay}, β=({train_cfg.beta1}, {train_cfg.beta2}))")
    print(f"    warmup steps:   {train_cfg.warmup_steps}")
    print(f"    lr floor ratio: {train_cfg.lr_floor_ratio}")
    print(f"    grad clip:      {train_cfg.grad_clip}")
    print()
    sched = train_cfg.checkpoint_schedule()
    print(f"  Checkpoints:      {len(sched)} log-spaced from step "
          f"{sched[0]} to step {sched[-1]}")
    print(f"    first 5: {sched[:5]}")
    print(f"    last 3:  {sched[-3:]}")
    print("=" * 78)
    print()


# ----------------------------------------------------------------------
# Single training run.
# ----------------------------------------------------------------------
def train_one_run(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    seed: int,
    run_dir: str,
    num_proc: Optional[int] = None,
    device: Optional[str] = None,
):
    """
    Train one model from scratch with the given configs and seed, saving
    checkpoints, metadata, and a CSV log into `run_dir`.

    Args:
        model_cfg: Model architecture configuration.
        train_cfg: Training hyperparameters and checkpoint schedule.
        seed: Random seed for parameter init, data shuffle, and any other
            stochastic decisions.
        run_dir: Directory to write checkpoints/, run_metadata.json,
            training_log.csv. Created if it doesn't exist.
        num_proc: Multiprocessing count for the tokenization step (None
            uses min(cpu_count, 32)).
        device: 'cuda' or 'cpu'. If None, auto-detects.
    """
    # ----- setup -----
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mixed precision: bfloat16 on Ada/Blackwell, fp16 on older. We pick
    # bfloat16 by default since the target hardware is RTX 5090 (Blackwell).
    if device == "cuda":
        capability = torch.cuda.get_device_capability(0)
        # Ada is sm_89, Hopper sm_90, Blackwell sm_100+. All support bf16.
        autocast_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    else:
        autocast_dtype = torch.float32  # autocast disabled on CPU anyway

    # Determinism. We seed torch globally; full determinism (CUBLAS, etc.)
    # is opt-in via train_cfg.deterministic because it slows training.
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    if train_cfg.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Enable TF32 and the standard fast path. Performance > exact reproducibility.
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # ----- print startup banner -----
    _print_startup_banner(
        device=device, model_cfg=model_cfg, train_cfg=train_cfg,
        seed=seed, run_dir=run_dir, autocast_dtype=autocast_dtype,
    )

    # ----- data -----
    print(">> Preparing dataset ...")
    train_dataset, held_out_dataset = prepare_dataset(
        model_cfg=model_cfg, train_cfg=train_cfg, num_proc=num_proc,
    )
    train_loader, eval_loader = make_dataloaders(
        train_dataset, held_out_dataset, train_cfg, seed=seed,
    )
    train_iter = iter(train_loader)

    # ----- model -----
    print(">> Building model ...")
    model = LlamaStyleTransformer(model_cfg).to(device)
    model.train()
    total, trainable = count_parameters(model)
    print(f"   ↳ ✅ Model on {device}. "
          f"Trainable parameters: {trainable / 1e6:.2f}M")
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"   ↳ VRAM after model load: allocated={allocated:.2f} GB, "
              f"reserved={reserved:.2f} GB")

    # ----- optimizer -----
    # Llama convention: don't decay biases or norm weights. Our model has no
    # biases (all linears are bias=False) so we just split norms from rest.
    decay_params, nodecay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # RMSNorm.weight is 1D; everything decayed is 2D+.
        if p.ndim < 2:
            nodecay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        eps=1e-8,
    )

    # ----- run metadata -----
    metadata_path = os.path.join(run_dir, train_cfg.metadata_filename)
    save_config_pair(model_cfg, train_cfg, metadata_path)
    print(f"   ↳ Wrote run metadata to {metadata_path}")

    # ----- CSV log -----
    csv_path = os.path.join(run_dir, train_cfg.csv_log_path)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "step", "lr", "train_loss", "grad_norm", "tokens_per_sec",
        "tokens_seen", "eval_loss", "elapsed_sec",
    ])

    # ----- training -----
    checkpoint_steps = set(train_cfg.checkpoint_schedule())
    last_eval_loss = float("nan")
    last_eval_step = -1
    t_start = time.time()
    t_last_log = t_start
    tokens_since_last_log = 0
    tokens_per_step = (
        train_cfg.micro_batch_size * train_cfg.grad_accum_steps
        * train_cfg.train_seq_len
    )

    print(">> Beginning training ...")
    print()

    for step in range(1, train_cfg.total_steps + 1):
        # ---- learning rate update ----
        lr = compute_lr(step - 1, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- gradient accumulation loop ----
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for accum_idx in range(train_cfg.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                # End of epoch: reshuffle and continue.
                train_iter = iter(train_loader)
                batch = next(train_iter)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=autocast_dtype,
                                     enabled=(device == "cuda")):
                _, loss, _ = model(input_ids, labels=input_ids)
            # Divide by accum so the accumulated grad equals the average.
            scaled_loss = loss / train_cfg.grad_accum_steps
            scaled_loss.backward()
            loss_accum += loss.item()
        loss_accum /= train_cfg.grad_accum_steps

        # ---- gradient clip and optimizer step ----
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), train_cfg.grad_clip
        ).item()
        optimizer.step()
        tokens_since_last_log += tokens_per_step

        # ---- periodic eval ----
        do_eval = (
            step % train_cfg.eval_every == 0
            or step == train_cfg.total_steps
            or step in checkpoint_steps
        )
        if do_eval:
            eval_loss, _ = evaluate_held_out(
                model, eval_loader, device, autocast_dtype,
            )
            last_eval_loss = eval_loss
            last_eval_step = step

        # ---- periodic console log ----
        do_log = (
            step % train_cfg.log_every == 0
            or step == 1
            or step == train_cfg.total_steps
            or step in checkpoint_steps
        )
        if do_log:
            t_now = time.time()
            elapsed_window = t_now - t_last_log
            throughput = tokens_since_last_log / max(elapsed_window, 1e-6)
            tokens_per_step_total = step * tokens_per_step
            # ETA based on average throughput since this run started.
            avg_throughput = tokens_per_step_total / max(t_now - t_start, 1e-6)
            tokens_remaining = (train_cfg.total_steps - step) * tokens_per_step
            eta_sec = tokens_remaining / max(avg_throughput, 1e-6)
            pct = step / train_cfg.total_steps * 100.0

            eval_str = (
                f" | eval {last_eval_loss:.4f}"
                if not math.isnan(last_eval_loss) else ""
            )
            print(
                f"  step {step:>6,}/{train_cfg.total_steps:,} "
                f"({pct:5.1f}%) | loss {loss_accum:.4f}{eval_str} "
                f"| lr {lr:.2e} | grad {grad_norm:6.3f} "
                f"| {_format_throughput(throughput)} "
                f"| eta {_format_time(eta_sec)}"
            )
            t_last_log = t_now
            tokens_since_last_log = 0

        # ---- CSV row (every step) ----
        csv_writer.writerow([
            step, f"{lr:.6e}", f"{loss_accum:.6f}", f"{grad_norm:.6e}",
            "", step * tokens_per_step,
            f"{last_eval_loss:.6f}" if step == last_eval_step else "",
            f"{time.time() - t_start:.2f}",
        ])
        csv_file.flush()

        # ---- checkpoint ----
        if step in checkpoint_steps:
            ckpt_path = os.path.join(
                run_dir, "checkpoints", f"step_{step:08d}.pt"
            )
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_accum,
                "eval_loss": last_eval_loss,
                "lr": lr,
                "elapsed_sec": time.time() - t_start,
                "seed": seed,
            }, ckpt_path)
            print(f"   ↳ 💾 Saved checkpoint: {ckpt_path}")

    csv_file.close()
    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  ✅ Training complete in {_format_time(elapsed)}")
    print(f"     Final loss: {loss_accum:.4f}, final eval: {last_eval_loss:.4f}")
    print(f"     Checkpoints saved to: {os.path.join(run_dir, 'checkpoints')}")
    print(f"     CSV log: {csv_path}")
    print("=" * 78)
    print()


# ----------------------------------------------------------------------
# CLI entry point.
# ----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train one Phase 1 model.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Output directory for checkpoints, metadata, log.")
    parser.add_argument("--total_steps", type=int, default=None,
                        help="Override TrainingConfig.total_steps (useful for "
                             "smoke tests; e.g. --total_steps 200).")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu", None])
    args = parser.parse_args()

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig(seed=args.seed)
    if args.total_steps is not None:
        train_cfg.total_steps = args.total_steps
        # When overriding steps for a smoke test, also reduce warmup/eval
        # cadence so the smoke test exercises those code paths.
        train_cfg.warmup_steps = min(train_cfg.warmup_steps, args.total_steps // 4)
        train_cfg.eval_every = min(train_cfg.eval_every, max(50, args.total_steps // 4))
        # Same idea for checkpoints — cap to total_steps.
        train_cfg.first_checkpoint_step = min(
            train_cfg.first_checkpoint_step, max(10, args.total_steps // 10),
        )

    train_one_run(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        seed=args.seed,
        run_dir=args.run_dir,
        num_proc=args.num_proc,
        device=args.device,
    )


if __name__ == "__main__":
    main()
    