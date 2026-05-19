"""
Quick single-checkpoint analyzer sanity check.

Runs the analyzer on ONE smoke-test checkpoint and reports timing + recovered
values. Useful for verifying the optimized analyzer is working correctly
without disrupting a long-running validation, or for benchmarking
post-optimization performance.

Usage:
    python3 quick_analyzer_check.py --run_dir smoke_test --device cpu
    python3 quick_analyzer_check.py --run_dir smoke_test --device cpu --step 200

Output: timing breakdown (activation collection vs SVD vs pairwise residuals)
plus the recovered λ, log α, etc. so we can verify the values look sensible.
"""

import argparse
import os
import time
import json
import glob

import numpy as np
import torch

from config import ModelConfig, TrainingConfig, load_config_pair
from data import prepare_dataset, make_dataloaders
from analyze import (
    default_pilot_positions, collect_activations, recover_linear_flow,
)
from models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--step", type=int, default=None,
                        help="Specific checkpoint step to analyze. "
                             "Default: latest available.")
    parser.add_argument("--max_pilots", type=int, default=10000)
    args = parser.parse_args()

    # Find the checkpoint.
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "checkpoints",
                                            "step_*.pt")))
    if not ckpts:
        print(f"❌ No checkpoints in {args.run_dir}/checkpoints/")
        return
    if args.step is not None:
        target = f"step_{args.step:08d}.pt"
        ckpt_path = next((p for p in ckpts if p.endswith(target)), None)
        if ckpt_path is None:
            print(f"❌ Checkpoint at step {args.step} not found.")
            print(f"   Available: {[os.path.basename(p) for p in ckpts]}")
            return
    else:
        ckpt_path = ckpts[-1]

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device:     {args.device}")
    print()

    # Load metadata.
    metadata_path = os.path.join(args.run_dir, "run_metadata.json")
    model_cfg, train_cfg = load_config_pair(metadata_path)
    print(f"Model: H={model_cfg.hidden_size}, L={model_cfg.num_hidden_layers}")

    # Build eval loader.
    print("\nBuilding eval loader...")
    t0 = time.time()
    _, held_out_dataset = prepare_dataset(model_cfg=model_cfg, train_cfg=train_cfg)
    _, eval_loader = make_dataloaders(
        train_dataset=held_out_dataset,
        held_out_dataset=held_out_dataset,
        train_cfg=train_cfg, seed=0, num_workers=2,
    )
    print(f"  Eval loader ready [{time.time() - t0:.1f}s]")

    # Determine autocast dtype.
    if args.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif args.device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Load model.
    print("\nLoading model from checkpoint...")
    t0 = time.time()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(model_cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Model loaded [{time.time() - t0:.1f}s]")
    print(f"  Step: {ckpt['step']}, eval loss: {ckpt.get('eval_loss', 'n/a')}")

    # Pilot positions.
    sample = next(iter(eval_loader))
    seq_len = sample["input_ids"].size(1)
    pilot_positions = default_pilot_positions(seq_len=seq_len)

    # Collect activations.
    print("\nCollecting activations...")
    t0 = time.time()
    activations = collect_activations(
        model=model, eval_loader=eval_loader,
        pilot_positions=pilot_positions, device=args.device,
        autocast_dtype=autocast_dtype, max_pilots=args.max_pilots,
    )
    t_collect = time.time() - t0
    print(f"  Collected {activations.shape[1]:,} pilots × "
          f"{activations.shape[0]} layers × {activations.shape[2]} dims "
          f"[{t_collect:.1f}s]")

    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # Recover linear flow.
    print("\nRecovering linear flow...")
    t0 = time.time()
    flow = recover_linear_flow(activations, center=True)
    t_flow = time.time() - t0
    print(f"  Linear flow recovered [{t_flow:.1f}s]")
    print()

    # Report.
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Step:                {int(ckpt['step'])}")
    print(f"Eval loss:           {ckpt.get('eval_loss', 'n/a')}")
    print(f"λ (variance rate):   {flow['lambda']:+.4f}")
    print(f"log α (prefactor):   {flow['log_alpha']:+.4f}")
    print(f"Effective rank by layer:")
    er = flow['effective_rank']
    for t in range(len(er)):
        print(f"  layer {t:>2}: {er[t]:.2f}")
    print(f"Kurtosis mean: {np.nanmean(flow['kurtosis_per_layer']):.3f}  "
          f"(0 = Gaussian)")
    print(f"Isotropy mean: {np.nanmean(flow['isotropy_per_layer']):.3f}  "
          f"(0 = isotropic)")
    print()
    print(f"Timing breakdown:")
    print(f"  Activation collection: {t_collect:>6.1f}s")
    print(f"  Linear flow recovery:  {t_flow:>6.1f}s")
    print(f"  Per-checkpoint total:  {t_collect + t_flow:>6.1f}s")

    # Sanity checks.
    print()
    print("Sanity checks:")
    checks_passed = 0
    checks_total = 0

    checks_total += 1
    if np.isfinite(flow['lambda']):
        print(f"  ✅ λ is finite")
        checks_passed += 1
    else:
        print(f"  ❌ λ is not finite: {flow['lambda']}")

    checks_total += 1
    if flow['lambda'] > 0:
        print(f"  ✅ λ is positive (variance grows with depth)")
        checks_passed += 1
    else:
        print(f"  ⚠️  λ ≤ 0 — variance not growing with depth (could be "
              f"normal for very early training)")

    checks_total += 1
    if not np.any(np.isnan(er)):
        print(f"  ✅ All effective ranks are finite")
        checks_passed += 1
    else:
        print(f"  ❌ Some effective ranks are NaN")

    checks_total += 1
    if er.max() <= model_cfg.hidden_size + 1:
        print(f"  ✅ Effective ranks are ≤ H = {model_cfg.hidden_size}")
        checks_passed += 1
    else:
        print(f"  ❌ Some effective rank exceeds H!")

    print()
    print(f"Sanity: {checks_passed}/{checks_total} passed.")


if __name__ == "__main__":
    main()
    