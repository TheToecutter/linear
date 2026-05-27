"""
Phase 2 memory-fit pre-check.

Some Phase 2 variants are substantially larger than the Phase 1
baseline (notably H1792 at ~527M params, ~3.6× baseline; and L24 at
~263M params). The existing training recipe (micro_batch_size=8,
grad_accum_steps=8, seq_len=1024, gradient_checkpointing on) was
sized for the 146M model. Whether it fits at the variant sizes on
a single 5090 (32 GB VRAM) is an open question we want to answer
*before* launching a full 24,000-step run that may OOM at step 2000.

What this script does
---------------------
For each named variant (or all flagged ones):

  1. Build the model on GPU.
  2. Try a short training pass (default 50 steps) at the standard
     micro_batch_size=8.
  3. If OOM at step 1: retry at micro_batch_size=4 with grad_accum_steps=16.
  4. If still OOM: retry at micro_batch_size=2 with grad_accum_steps=32.
  5. If still OOM: report failure and skip.
  6. If ok: report peak VRAM, step time, and the effective batch shape
     that worked.

Outputs
-------
For each variant tested, writes
    phase2_runs/<axis>/<variant>/MEMFIT_NOTES.txt
documenting the working batch shape and observed step time. The
launch script can read this file and apply the overrides when
launching the full run.

Usage
-----
    # Default: check the variants marked as heavyweight.
    python3 phase2_memfit.py

    # Specific variants:
    python3 phase2_memfit.py --variant H1792 --variant L24

    # Try a different number of probe steps:
    python3 phase2_memfit.py --variant H1792 --probe_steps 100

    # Dry-run (no GPU; resolves configs and prints plan):
    python3 phase2_memfit.py --dry_run

Note
----
This script is environment-aware: it skips its actual GPU work when
no GPU is available (and the user passes --dry_run), and reports
that the fit-check needs to be re-run on the target machine. Its
unit tests cover only the dry-run / config-resolution paths.
"""

import argparse
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

from config import ModelConfig, TrainingConfig
from phase2_configs import ALL_TIER1_VARIANTS, find_variant, VariantSpec
from phase2_launch import run_dir_for, PHASE2_ROOT


# ----------------------------------------------------------------------
# Default heavyweight set.
# ----------------------------------------------------------------------
DEFAULT_HEAVYWEIGHT_LABELS = ["H1792", "L24"]


# Fallback batch shapes to try, in order. Each entry is
# (micro_batch_size, grad_accum_steps), such that the product stays
# at 64 (the effective batch the training loop expects).
FALLBACK_BATCH_SHAPES: List[Tuple[int, int]] = [
    (8, 8),
    (4, 16),
    (2, 32),
    (1, 64),
]


# ----------------------------------------------------------------------
# Notes file.
# ----------------------------------------------------------------------
@dataclass
class MemFitNotes:
    variant_label: str
    variant_axis: str
    params_M: float
    micro_batch_size: int
    grad_accum_steps: int
    seq_len: int
    peak_vram_gb: float
    step_time_s: float
    probe_steps: int
    used_gradient_checkpointing: bool
    hostname: str
    timestamp: str


def write_notes(variant: VariantSpec, notes: MemFitNotes,
                root: str = PHASE2_ROOT) -> str:
    """Write a MEMFIT_NOTES.txt under the variant's directory.

    Returns the path written.
    """
    var_dir = os.path.dirname(run_dir_for(variant, seed=0, root=root))
    os.makedirs(var_dir, exist_ok=True)
    path = os.path.join(var_dir, "MEMFIT_NOTES.txt")
    with open(path, "w") as f:
        f.write("Memory-fit pre-check results for Phase 2 variant\n")
        f.write("=================================================\n\n")
        for k, v in asdict(notes).items():
            f.write(f"  {k:35} : {v}\n")
        f.write(
            "\nNotes:\n"
            "  - These values are the effective batch shape that fit on\n"
            "    the target machine and should be applied to the full\n"
            "    training run for this variant.\n"
            "  - The TrainingConfig defaults are micro_batch=8, "
            "grad_accum=8 (effective batch 64 sequences).\n"
            "  - If micro_batch differs from 8, the launcher should\n"
            "    override TrainingConfig.micro_batch_size and\n"
            "    TrainingConfig.grad_accum_steps accordingly so the\n"
            "    effective batch stays at 64.\n"
        )
    return path


def read_notes(variant: VariantSpec,
               root: str = PHASE2_ROOT) -> Optional[dict]:
    """Read a MEMFIT_NOTES.txt back into a dict, or return None.

    Used by the launcher to apply overrides on the full training run.
    """
    var_dir = os.path.dirname(run_dir_for(variant, seed=0, root=root))
    path = os.path.join(var_dir, "MEMFIT_NOTES.txt")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for line in f:
            if " : " not in line:
                continue
            k, v = line.strip().split(" : ", 1)
            k = k.strip()
            v = v.strip()
            # Attempt to coerce types.
            try:
                v_coerced = int(v)
            except ValueError:
                try:
                    v_coerced = float(v)
                except ValueError:
                    v_coerced = v
            out[k] = v_coerced
    return out


# ----------------------------------------------------------------------
# Fit-check core.
# ----------------------------------------------------------------------
def _run_probe(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    probe_steps: int,
    device: str,
) -> Tuple[float, float]:
    """Run probe_steps of training on a tiny synthetic batch stream.

    Returns (peak_vram_gb, mean_step_time_s).

    We use synthetic inputs (random token IDs in the model's vocab range)
    rather than the FineWeb-Edu loader because we want to isolate the
    memory footprint of the model + optimizer, independent of any data-
    loading slowness or HF dataset cache warming. The forward/backward
    memory cost is identical.

    Raises torch.cuda.OutOfMemoryError if the configuration OOMs.
    """
    import torch
    from torch.optim import AdamW
    from models import build_model

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if device == "cuda":
        autocast_dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8
            else torch.float16
        )
    else:
        autocast_dtype = torch.float32

    model = build_model(model_cfg).to(device)
    model.train()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        betas=(train_cfg.beta1, train_cfg.beta2),
        weight_decay=train_cfg.weight_decay,
    )

    B = train_cfg.micro_batch_size
    T = train_cfg.train_seq_len
    V = model_cfg.vocab_size

    step_times = []
    for step in range(probe_steps):
        t0 = time.time()
        for _ in range(train_cfg.grad_accum_steps):
            # Synthetic batch on-device to skip dataloader.
            ids = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
            with torch.amp.autocast(
                "cuda", dtype=autocast_dtype, enabled=(device == "cuda"),
            ):
                logits, loss, _ = model(ids, labels=ids)
            (loss / train_cfg.grad_accum_steps).backward()
        # Mimic the optimizer step / zero_grad cycle.
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.time() - t0)

    peak_vram_gb = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device == "cuda" else 0.0
    )
    mean_step_time = sum(step_times) / len(step_times)

    # Free everything before returning.
    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()

    return peak_vram_gb, mean_step_time


def fit_check_variant(
    variant: VariantSpec,
    probe_steps: int = 50,
    device: str = "cuda",
    write: bool = True,
) -> Optional[MemFitNotes]:
    """Run the fit-check for one variant. Returns notes or None if all
    fallback batch shapes OOMed (in which case the variant is unfittable
    on the target machine with the current training recipe).
    """
    import torch

    cfg = variant.config_factory()
    train_cfg_base = TrainingConfig()
    params_M = cfg.estimate_param_count() / 1e6

    print()
    print("=" * 72)
    print(f"Memory-fit check: variant {variant.label}  ({params_M:.1f}M params)")
    print("=" * 72)

    last_error = None
    for micro_batch, grad_accum in FALLBACK_BATCH_SHAPES:
        train_cfg = TrainingConfig()
        train_cfg.micro_batch_size = micro_batch
        train_cfg.grad_accum_steps = grad_accum

        print(f"  Trying micro_batch={micro_batch}, grad_accum={grad_accum} "
              f"(effective batch = {micro_batch * grad_accum}) ...")

        try:
            peak_vram, step_time = _run_probe(
                model_cfg=cfg, train_cfg=train_cfg,
                probe_steps=probe_steps, device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            print(f"    ↳ OOM at this batch shape. Trying smaller.")
            last_error = e
            # Reset CUDA state for the next attempt.
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            # Non-OOM error -- don't keep trying.
            print(f"    ↳ Unexpected error: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None
        print(f"    ↳ ✅ Fits. Peak VRAM = {peak_vram:.2f} GB, "
              f"mean step = {step_time:.2f}s.")

        from socket import gethostname
        from datetime import datetime, timezone

        notes = MemFitNotes(
            variant_label=variant.label,
            variant_axis=variant.axis,
            params_M=params_M,
            micro_batch_size=micro_batch,
            grad_accum_steps=grad_accum,
            seq_len=train_cfg.train_seq_len,
            peak_vram_gb=peak_vram,
            step_time_s=step_time,
            probe_steps=probe_steps,
            used_gradient_checkpointing=cfg.gradient_checkpointing,
            hostname=gethostname(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if write:
            path = write_notes(variant, notes)
            print(f"    ↳ Wrote: {path}")
        return notes

    # All fallbacks OOMed.
    print(f"  ❌ All fallback batch shapes OOMed for {variant.label}.")
    print(f"     Last error: {last_error}")
    return None


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 memory-fit pre-check.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument(
        "--variant", action="append", default=None,
        help="Variant label to check (repeatable). Default: the "
             "DEFAULT_HEAVYWEIGHT_LABELS set.",
    )
    parser.add_argument(
        "--probe_steps", type=int, default=50,
        help="Number of optimizer steps in the probe. Default 50.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Resolve configs and report planned checks without running.",
    )
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    labels = args.variant or DEFAULT_HEAVYWEIGHT_LABELS
    variants = [find_variant(lab) for lab in labels]

    if args.dry_run:
        print(">> Memory-fit dry-run plan:")
        for v in variants:
            cfg = v.config_factory()
            print(f"  - {v.label} (axis={v.axis}): "
                  f"L={cfg.num_hidden_layers}, H={cfg.hidden_size}, "
                  f"I_stored={cfg.intermediate_size}, "
                  f"params={cfg.estimate_param_count()/1e6:.1f}M")
        print(f"  Probe steps: {args.probe_steps}")
        print(f"  Fallback batch shapes (micro, accum): {FALLBACK_BATCH_SHAPES}")
        return

    # Sanity check: GPU available?
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is required to run the fit-check.",
              file=sys.stderr)
        sys.exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Error: --device=cuda requested but no CUDA device is "
              "available. Re-run on the GPU machine, or pass --dry_run "
              "to validate the config-resolution path.", file=sys.stderr)
        sys.exit(1)

    successes = 0
    failures = []
    for variant in variants:
        notes = fit_check_variant(
            variant=variant,
            probe_steps=args.probe_steps,
            device=args.device,
        )
        if notes is None:
            failures.append(variant.label)
        else:
            successes += 1

    print()
    print("=" * 72)
    print(f"Memory-fit summary: {successes} / {len(variants)} variants fit.")
    if failures:
        print(f"  Failed: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
