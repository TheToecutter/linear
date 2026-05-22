"""
Stage A regeneration with randomized pilot positions per chunk.

This script extends the multiview campaign by generating a second set
of augmented activation files (with the suffix '_random' added to the
filename) using randomized pilot positions instead of the fixed
default_pilot_positions scheme. The randomized files decouple position
from textual context, which makes the next-token-vs-position
identification problem (§5.2 of the investigation writeup) tractable.

The script collects augmented activations at four representative
checkpoints spanning the three training phases identified in §8.2:

  Phase I  representative: step 500   (rapid SVD consolidation)
  Phase II representative: step 2500  (consolidated plateau)
  Phase III mid:           step 10000 (late-training restructuring)
  Phase III final:         step 24000 (final checkpoint)

These can be overridden with --checkpoints. By default the script
processes all four seeds at each step.

The output files are saved as
    run_dir/multiview/seed_S/augmented_step_NNNNNNNN_random.npz
to keep them distinct from the fixed-position files used in the
original campaign. Downstream tools can be pointed at these via a
straightforward filename change.

Important: the script does NOT touch the existing fixed-position files,
nor any other stage of the multiview campaign. It only adds new files.

Usage:
    # Dry run: report what would be done.
    python regenerate_stage_a.py --run-dir ../phase1_runs_gelu --dry-run

    # Default: 4 checkpoints x 4 seeds = 16 inference passes.
    python regenerate_stage_a.py --run-dir ../phase1_runs_gelu

    # Override checkpoints and seeds.
    python regenerate_stage_a.py --run-dir ../phase1_runs_gelu \\
        --checkpoints 500 2500 10000 24000 --seeds 0 1 2 3

    # Reproducible: pin the RNG seed for position draws.
    python regenerate_stage_a.py --run-dir ../phase1_runs_gelu \\
        --position-rng-seed 20260521
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

import torch

from config import ModelConfig, load_config_pair
from data import prepare_dataset, make_dataloaders
from model import LlamaStyleTransformer

from analyze import default_pilot_positions
from multiview import save_augmented_activations
from multiview_campaign import (
    seeds_in_run,
    checkpoints_in_seed,
    multiview_dir,
)


# ----------------------------------------------------------------------
# Filename convention: same as augmented_path() but with '_random' suffix.
# ----------------------------------------------------------------------
def randomized_augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(
        multiview_dir(run_dir), f"seed_{seed}",
        f"augmented_step_{step:08d}_random.npz",
    )


# ----------------------------------------------------------------------
# Inference loop with per-sequence randomized pilot positions.
# This is a near-copy of multiview._collect_activations_with_metadata_body
# with the position handling changed: each sequence in each batch gets
# its own independent set of K random positions, drawn from the valid
# range [0, T-2]. The number K per sequence matches the count of
# default_pilot_positions for the model's seq_len.
# ----------------------------------------------------------------------
def collect_activations_randomized(
    model,
    eval_loader,
    n_positions_per_seq: int,
    device: str,
    autocast_dtype,
    max_pilots: int,
    rng: np.random.Generator,
    compute_predictions: bool = True,
) -> dict:
    model.eval()
    num_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    n_layer_outputs = num_layers + 2  # post-embed + L block outputs + post-final-norm

    state_buffers: List[List[np.ndarray]] = [[] for _ in range(n_layer_outputs)]
    input_id_chunks: List[np.ndarray] = []
    next_id_chunks: List[np.ndarray] = []
    pred_id_chunks: List[np.ndarray] = []
    pos_chunks: List[np.ndarray] = []
    collected = 0

    with torch.no_grad():
        for batch in eval_loader:
            if collected >= max_pilots:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)  # (B, T)
            B, T = input_ids.shape

            # Valid position range: [0, T-2] so that p+1 is a valid
            # successor token within the same chunk.
            valid_upper = T - 2
            if valid_upper < 0 or n_positions_per_seq < 1:
                continue
            # Cannot draw more unique positions than available.
            K = min(n_positions_per_seq, valid_upper + 1)
            # Per-sequence random positions: shape (B, K), independent
            # draws per sequence in this batch. Sample without
            # replacement within each sequence so we don't double-count
            # a position for the same chunk.
            pos_per_seq = np.zeros((B, K), dtype=np.int64)
            for b in range(B):
                pos_per_seq[b] = rng.choice(
                    valid_upper + 1, size=K, replace=False)
            # Sort along K for consistent ordering (cosmetic; tools
            # don't depend on it).
            pos_per_seq = np.sort(pos_per_seq, axis=1)
            pos_per_seq_t = torch.from_numpy(pos_per_seq).to(device)

            with torch.amp.autocast(
                    "cuda", dtype=autocast_dtype,
                    enabled=(device == "cuda")):
                logits, _, hidden = model(
                    input_ids, return_hidden_states=True)

            # Gather hidden states at per-sequence positions. Each
            # hidden tensor h has shape (B, T, H); we need
            # h[b, pos_per_seq[b], :] for each b.
            # torch.gather expects the index tensor to match h's shape
            # along the dim we gather (dim=1, the T axis), so expand
            # pos_per_seq_t to (B, K, H).
            pos_idx_BKH = pos_per_seq_t.unsqueeze(-1).expand(-1, -1, H)
            for layer_idx, h in enumerate(hidden):
                gathered = torch.gather(h, dim=1, index=pos_idx_BKH)
                picked = gathered.float().cpu().numpy()    # (B, K, H)
                state_buffers[layer_idx].append(picked.reshape(-1, H))

            # Input tag: input_ids at pos_per_seq[b].
            input_tag = torch.gather(
                input_ids, dim=1, index=pos_per_seq_t).cpu().numpy().astype(np.int32)
            # Successor tag: input_ids at pos+1.
            next_idx = pos_per_seq_t + 1
            next_tag = torch.gather(
                input_ids, dim=1, index=next_idx).cpu().numpy().astype(np.int32)

            if compute_predictions:
                # logits at the same per-sequence positions; argmax over V.
                V = logits.size(-1)
                pos_idx_BKV = pos_per_seq_t.unsqueeze(-1).expand(-1, -1, V)
                pred_logits = torch.gather(logits, dim=1, index=pos_idx_BKV)
                pred_tag = pred_logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
            else:
                pred_tag = -1 * np.ones((B, K), dtype=np.int32)

            input_id_chunks.append(input_tag.reshape(-1))
            next_id_chunks.append(next_tag.reshape(-1))
            pred_id_chunks.append(pred_tag.reshape(-1))
            pos_chunks.append(pos_per_seq.astype(np.int32).reshape(-1))

            collected += B * K

    N = collected
    states = np.zeros((n_layer_outputs, N, H), dtype=np.float32)
    for layer_idx in range(n_layer_outputs):
        if state_buffers[layer_idx]:
            cat = np.concatenate(state_buffers[layer_idx], axis=0)
            states[layer_idx] = cat[:N]
    input_ids_arr = (
        np.concatenate(input_id_chunks, axis=0)[:N]
        if input_id_chunks else np.zeros((0,), dtype=np.int32))
    next_ids_arr = (
        np.concatenate(next_id_chunks, axis=0)[:N]
        if next_id_chunks else np.zeros((0,), dtype=np.int32))
    pred_ids_arr = (
        np.concatenate(pred_id_chunks, axis=0)[:N]
        if pred_id_chunks else np.zeros((0,), dtype=np.int32))
    pos_arr = (
        np.concatenate(pos_chunks, axis=0)[:N]
        if pos_chunks else np.zeros((0,), dtype=np.int32))

    return {
        "states": states,
        "input_ids": input_ids_arr,
        "next_ids": next_ids_arr,
        "pred_ids": pred_ids_arr,
        "positions": pos_arr,
    }


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
# Default representative checkpoints, one per training phase from §8.2.
DEFAULT_CHECKPOINTS = [500, 2500, 10000, 24000]


def main():
    ap = argparse.ArgumentParser(
        description="Stage A regeneration with randomized pilot positions.")
    ap.add_argument("--run-dir", required=True,
                    help="Top-level run directory (e.g. ../phase1_runs_gelu).")
    ap.add_argument("--config", required=True,
                    help="Path to the JSON config file used in the original "
                         "Phase 1 campaign (loaded by config.load_config_pair). "
                         "Required for model and dataset construction.")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="Seeds to process (default: all seeds in run-dir).")
    ap.add_argument("--checkpoints", type=int, nargs="*",
                    default=DEFAULT_CHECKPOINTS,
                    help="Training steps to regenerate (default: "
                         f"{DEFAULT_CHECKPOINTS}).")
    ap.add_argument("--max-pilots", type=int, default=100_000)
    ap.add_argument("--position-rng-seed", type=int, default=20260521,
                    help="RNG seed for position draws "
                         "(reproducible across runs).")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing",
                    action="store_false")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds_avail = seeds_in_run(args.run_dir)
    if not seeds_avail:
        print(f"No seeds found in {args.run_dir}")
        sys.exit(1)
    seeds = args.seeds if args.seeds is not None else seeds_avail
    bad_seeds = [s for s in seeds if s not in seeds_avail]
    if bad_seeds:
        print(f"Requested seeds not present: {bad_seeds}")
        sys.exit(1)

    # Map requested steps to actual checkpoint files per seed; warn if
    # any requested step is missing.
    steps_per_seed = {}
    for s in seeds:
        avail = dict(checkpoints_in_seed(args.run_dir, s))
        steps_per_seed[s] = avail
    plan: List[Tuple[int, int, str, str]] = []  # (seed, step, ckpt_path, out_path)
    for s in seeds:
        avail = steps_per_seed[s]
        for target_step in args.checkpoints:
            if target_step not in avail:
                # Nearest available step in log space.
                nearest = min(avail.keys(),
                              key=lambda k: abs(np.log(max(k, 1)) -
                                                np.log(max(target_step, 1))))
                print(f"[warn] seed {s}: requested step {target_step} not "
                      f"available; nearest is {nearest} "
                      f"(off by {abs(nearest - target_step)} steps)")
                step = nearest
            else:
                step = target_step
            out_path = randomized_augmented_path(args.run_dir, s, step)
            plan.append((s, step, avail[step], out_path))

    # Dry-run report.
    print()
    print("=" * 72)
    print(f"Plan: {len(plan)} (seed, step) pairs to regenerate")
    print("=" * 72)
    n_to_do = 0
    n_skip = 0
    for seed, step, ckpt, out in plan:
        exists = os.path.exists(out)
        if exists and args.skip_existing:
            status = "skip (exists)"
            n_skip += 1
        else:
            status = "regenerate" + (", overwriting existing" if exists else "")
            n_to_do += 1
        print(f"  seed {seed} step {step:>6d}: {status}")
        print(f"    in:  {ckpt}")
        print(f"    out: {out}")
    print(f"Will process {n_to_do} pairs ({n_skip} skipped)")

    if args.dry_run:
        print("\n[dry run] no work performed.")
        return

    if n_to_do == 0:
        print("\nNothing to do.")
        return

    # Set up model config and eval loader, matching the pattern used by
    # multiview_campaign.py's stage A entry point.
    print()
    print("Loading config and eval loader ...")
    model_cfg, train_cfg = load_config_pair(args.config)
    _, held_out_dataset = prepare_dataset(model_cfg, train_cfg)
    _, eval_loader = make_dataloaders(held_out_dataset, held_out_dataset,
                                       train_cfg)
    print(f"  model_cfg: H={model_cfg.hidden_size} "
          f"L={model_cfg.num_hidden_layers} V={model_cfg.vocab_size} "
          f"T={model_cfg.max_position_embeddings}")

    if args.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif args.device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Number of positions per sequence: match the default scheme.
    default_pos = default_pilot_positions(
        seq_len=model_cfg.max_position_embeddings)
    n_positions = len([
        p for p in default_pos
        if 0 <= p and p + 1 < model_cfg.max_position_embeddings - 1
    ])
    print(f"  n_positions_per_seq = {n_positions} "
          f"(matches default_pilot_positions count)")

    # RNG. Use a single Generator for the whole campaign so that the
    # draws are reproducible given the same position-rng-seed.
    rng_master = np.random.default_rng(args.position_rng_seed)

    print()
    t_start = time.time()
    for i, (seed, step, ckpt_path, out_path) in enumerate(plan):
        if args.skip_existing and os.path.exists(out_path):
            continue
        # Sub-seed: derive a unique seed for this (seed, step) so that
        # different checkpoints of the same seed see different random
        # position sets, and re-running the script gives the same draws.
        sub_seed = int(rng_master.integers(0, 2**31 - 1))
        rng = np.random.default_rng(sub_seed)
        print(f"[{i+1}/{len(plan)}] seed {seed} step {step}: regenerating "
              f"with sub_seed={sub_seed} ...", flush=True)
        t0 = time.time()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Load model.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = LlamaStyleTransformer(model_cfg).to(args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        payload = collect_activations_randomized(
            model=model,
            eval_loader=eval_loader,
            n_positions_per_seq=n_positions,
            device=args.device,
            autocast_dtype=autocast_dtype,
            max_pilots=args.max_pilots,
            rng=rng,
            compute_predictions=True,
        )

        save_augmented_activations(payload, out_path)
        N = payload["states"].shape[1]
        print(f"  saved {N:,} pilots in {time.time() - t0:.1f}s -> "
              f"{out_path}")

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
    