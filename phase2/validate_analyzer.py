"""
Validate the analyzer on the smoke-test checkpoints.

This script:
  1. Loads the smoke-test run metadata.
  2. Builds the held-out evaluation dataloader.
  3. Runs analyze_run() on every smoke-test checkpoint.
  4. Reports a summary of recovered quantities across checkpoints
     so we can sanity-check the pipeline.

The primary goal is to validate that the analyzer runs end-to-end on
real checkpoints without crashes, OOMs, or numerical issues. The
secondary goal is to see what early-training linear-flow recovery
looks like in practice (the smoke-test checkpoints are only 20-200
steps into training, so they're not converged — but the pipeline
should still produce well-defined output).

Usage:
    python3 validate_analyzer.py --run_dir smoke_test
    python3 validate_analyzer.py --run_dir smoke_test --device cpu

Note on device choice:
  - If a separate training run is using the GPU, pass --device cpu to
    avoid contention. SVD on CPU is slow at H=896 (~30s per checkpoint)
    but safe.
  - If the GPU is free, pass --device cuda (default). Much faster,
    typically a few seconds per checkpoint.
"""

import argparse
import os
import sys
import time

import numpy as np

from config import ModelConfig, TrainingConfig, load_config_pair
from data import prepare_dataset, make_dataloaders
from analyze import analyze_run, load_flow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to a completed training run with "
                             "checkpoints/ and run_metadata.json.")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Device for inference. Default: cuda if "
                             "available else cpu.")
    parser.add_argument("--max_pilots", type=int, default=10000,
                        help="Max pilot activations per checkpoint. "
                             "Smaller = faster, less accurate SVD.")
    parser.add_argument("--summary_only", action="store_true",
                        help="Skip running the analyzer; just summarize "
                             "existing flow .npz files (if any).")
    parser.add_argument("--statistic_mode", type=str, default="both",
                        choices=["ours", "paper", "both"],
                        help="Which variance/kurtosis convention to display. "
                             "'ours' = log(mean_d var_d), mean_d κᵢ. "
                             "'paper' = mean_d log(var_d), mean_d |κᵢ| — "
                             "directly comparable to Sarfati et al.'s "
                             "published log α, λ, <|κ|>. "
                             "Both are always computed and saved; this "
                             "flag controls only what gets printed.")
    args = parser.parse_args()

    # Validate the run dir.
    assert os.path.isdir(args.run_dir), f"run_dir does not exist: {args.run_dir}"
    metadata_path = os.path.join(args.run_dir, "run_metadata.json")
    assert os.path.exists(metadata_path), (
        f"run_metadata.json missing from {args.run_dir}"
    )
    model_cfg, train_cfg = load_config_pair(metadata_path)
    print(f"✅ Loaded metadata from {metadata_path}")
    print(f"   Model: H={model_cfg.hidden_size}, "
          f"L={model_cfg.num_hidden_layers}, "
          f"vocab={model_cfg.vocab_size}")

    if not args.summary_only:
        # ----- build the eval dataloader -----
        # We need the same FineWeb-Edu held-out split that the model was
        # trained against. prepare_dataset() is deterministic — it reads
        # the same cached tokenized dataset and takes the last
        # held_out_chunks as eval.
        print()
        print(">> Building eval dataloader ...")
        print("   (Reusing the cached tokenized FineWeb-Edu from training.)")
        _, held_out_dataset = prepare_dataset(
            model_cfg=model_cfg, train_cfg=train_cfg,
        )
        _, eval_loader = make_dataloaders(
            train_dataset=held_out_dataset,  # not used; we need both args
            held_out_dataset=held_out_dataset,
            train_cfg=train_cfg, seed=0, num_workers=2,
        )

        # ----- run the analyzer -----
        print()
        print(">> Running analyzer on all checkpoints in "
              f"{args.run_dir}/checkpoints/ ...")
        t_start = time.time()
        analyze_run(
            run_dir=args.run_dir,
            eval_loader=eval_loader,
            device=args.device or ("cuda" if _cuda_available() else "cpu"),
            skip_existing=True,
            max_pilots=args.max_pilots,
            statistic_mode=args.statistic_mode,
        )
        print(f"\n>> Total analysis time: {time.time() - t_start:.1f}s")

    # ----- summarize whatever flow .npz files exist -----
    print()
    print(">> Summary of recovered linear flows:")
    print()
    flow_dir = os.path.join(args.run_dir, "flow_analysis")
    if not os.path.isdir(flow_dir):
        print(f"   (No flow_analysis directory yet — nothing to summarize.)")
        return

    import glob
    flow_files = sorted(glob.glob(os.path.join(flow_dir, "flow_step_*.npz")))
    if not flow_files:
        print(f"   (No flow .npz files yet.)")
        return

    # Convention-aware accessors. Older flow .npz files (written before
    # the paper-convention refinement) won't have the *_paper fields; we
    # surface NaN in that case rather than crash, and warn once.
    def _field(flow, key):
        return flow.get(key, float("nan")) if isinstance(flow, dict) else float("nan")

    have_paper = "lambda_paper" in load_flow(flow_files[0])
    if args.statistic_mode in ("paper", "both") and not have_paper:
        print("   ⚠️  These flow .npz files predate the paper-convention "
              "refinement (no 'lambda_paper' field). Showing 'ours' "
              "values only; re-run the analyzer (without --summary_only "
              "and with skip_existing turned off, or delete the .npz "
              "files first) to regenerate with both conventions.")
        print()
        effective_mode = "ours"
    else:
        effective_mode = args.statistic_mode

    # Header.
    if effective_mode == "ours":
        print(f"   {'step':>8}  {'eval':>7}  {'λ':>8}  "
              f"{'log_α':>8}  {'eff_rank(L=0)':>15}  {'eff_rank(mid)':>15}  "
              f"{'<κ>':>10}  {'iso_mean':>10}")
        print(f"   {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*15}  {'-'*15}  "
              f"{'-'*10}  {'-'*10}")
    elif effective_mode == "paper":
        print(f"   {'step':>8}  {'eval':>7}  {'λ_paper':>9}  "
              f"{'logα_paper':>11}  {'eff_rank(L=0)':>15}  "
              f"{'eff_rank(mid)':>15}  {'<|κ|>':>10}  {'iso_mean':>10}")
        print(f"   {'-'*8}  {'-'*7}  {'-'*9}  {'-'*11}  {'-'*15}  "
              f"{'-'*15}  {'-'*10}  {'-'*10}")
    else:  # both
        print(f"   {'step':>8}  {'eval':>7}  "
              f"{'λ(o)':>8}  {'λ(p)':>8}  {'logα(o)':>9}  {'logα(p)':>9}  "
              f"{'eff_rk(0)':>10}  {'eff_rk(mid)':>11}  "
              f"{'<κ>':>8}  {'<|κ|>':>8}")
        print(f"   {'-'*8}  {'-'*7}  "
              f"{'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  "
              f"{'-'*10}  {'-'*11}  {'-'*8}  {'-'*8}")

    for path in flow_files:
        flow = load_flow(path)
        step = int(flow["checkpoint_step"])
        eval_loss = float(flow["checkpoint_eval_loss"])
        L = int(flow["num_layers_total"])
        eff_rank = flow["effective_rank"]
        kurt = flow["kurtosis_per_layer"]
        kurt_abs = flow.get("kurtosis_abs_per_layer")
        iso = flow["isotropy_per_layer"]
        # Layer 0 = post-embedding; layer L//2 = mid-network.
        if effective_mode == "ours":
            print(f"   {step:>8}  {eval_loss:>7.4f}  {flow['lambda']:>+8.4f}  "
                  f"{flow['log_alpha']:>+8.3f}  {eff_rank[0]:>15.2f}  "
                  f"{eff_rank[L // 2]:>15.2f}  "
                  f"{np.nanmean(kurt):>+10.3f}  {np.nanmean(iso):>+10.3f}")
        elif effective_mode == "paper":
            kabs_mean = float(np.nanmean(kurt_abs)) if kurt_abs is not None else float("nan")
            print(f"   {step:>8}  {eval_loss:>7.4f}  "
                  f"{flow['lambda_paper']:>+9.4f}  "
                  f"{flow['log_alpha_paper']:>+11.3f}  {eff_rank[0]:>15.2f}  "
                  f"{eff_rank[L // 2]:>15.2f}  "
                  f"{kabs_mean:>10.3f}  {np.nanmean(iso):>+10.3f}")
        else:  # both
            kabs_mean = float(np.nanmean(kurt_abs)) if kurt_abs is not None else float("nan")
            print(f"   {step:>8}  {eval_loss:>7.4f}  "
                  f"{flow['lambda']:>+8.4f}  {flow['lambda_paper']:>+8.4f}  "
                  f"{flow['log_alpha']:>+9.3f}  {flow['log_alpha_paper']:>+9.3f}  "
                  f"{eff_rank[0]:>10.2f}  {eff_rank[L // 2]:>11.2f}  "
                  f"{np.nanmean(kurt):>+8.3f}  {kabs_mean:>8.3f}")

    # ----- a few qualitative checks -----
    print()
    print(">> Sanity checks across checkpoints:")
    first_flow = load_flow(flow_files[0])
    last_flow = load_flow(flow_files[-1])

    # Check 1: does effective rank change as training progresses?
    eff_first = first_flow["effective_rank"]
    eff_last = last_flow["effective_rank"]
    print(f"   Effective rank, layer 0:  "
          f"first ckpt {eff_first[0]:.2f} → last ckpt {eff_last[0]:.2f}")
    print(f"   Effective rank, mid:      "
          f"first ckpt {eff_first[len(eff_first) // 2]:.2f} → "
          f"last ckpt {eff_last[len(eff_last) // 2]:.2f}")

    # Check 2: variance scaling rate λ across training.
    lambdas = []
    eval_losses = []
    for path in flow_files:
        f = load_flow(path)
        lambdas.append(f["lambda"])
        eval_losses.append(f["checkpoint_eval_loss"])
    lambdas = np.array(lambdas)
    eval_losses = np.array(eval_losses)
    print(f"   λ range across training: "
          f"min={lambdas.min():.4f}, max={lambdas.max():.4f}, "
          f"final={lambdas[-1]:.4f}")
    print(f"   eval loss range:         "
          f"first={eval_losses[0]:.3f}, final={eval_losses[-1]:.3f}, "
          f"reduction={eval_losses[0] - eval_losses[-1]:+.3f}")

    # Check 3: is the linear flow itself stabilizing?
    # Compute distance between consecutive flows in their basis-invariant
    # statistics (just the singular value spectra, in log space).
    print()
    print("   Successive-checkpoint basis-invariant distance:")
    print("     Δ ‖log Σ(t)‖ summed over layers, between consecutive checkpoints")
    print("     (decreasing = converging; rising or stable = not yet settled)")
    sv_distances = []
    for i in range(1, len(flow_files)):
        f_prev = load_flow(flow_files[i - 1])
        f_curr = load_flow(flow_files[i])
        sv_prev = f_prev["singular_values"]  # (L, H)
        sv_curr = f_curr["singular_values"]
        log_diff = np.log(np.maximum(sv_curr, 1e-12)) - np.log(np.maximum(sv_prev, 1e-12))
        d = np.linalg.norm(log_diff, axis=-1).sum()  # sum over layers
        sv_distances.append(d)
    sv_distances = np.array(sv_distances)
    # Print at sparse stride to keep output tidy.
    stride = max(1, len(sv_distances) // 10)
    for i in range(0, len(sv_distances), stride):
        step = int(load_flow(flow_files[i + 1])["checkpoint_step"])
        print(f"     step {step:>6}: Δ = {sv_distances[i]:.3f}")
    print(f"     mean Δ (early half): {sv_distances[:len(sv_distances)//2].mean():.3f}")
    print(f"     mean Δ (late half):  {sv_distances[len(sv_distances)//2:].mean():.3f}")
    if sv_distances[len(sv_distances)//2:].mean() < sv_distances[:len(sv_distances)//2].mean():
        print(f"     → late mean < early mean ✓ (flow is converging)")
    else:
        print(f"     → late mean >= early mean (flow NOT yet converging)")

    print()
    print(">> ✅ Validation complete.")


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
    