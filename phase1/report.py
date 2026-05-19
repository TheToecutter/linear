"""
Generate the full Phase 1 plot set from a completed training run.

Usage:
    python3 report.py --run_dir run_seed0
    python3 report.py --run_dir run_seed0 --output_dir custom_plots

Produces eight figures in `run_dir/plots/` (or the custom output_dir):
    01_loss_curves.png                       - training/eval loss vs step
    02_basis_invariant_trajectories.png      - λ, log α, eff rank, kurt, iso vs step
    03_effective_rank_depth_profile.png      - eff rank vs layer, colored by step
    04_flow_convergence.png                  - distance to final L^(K_final)
    05_variance_scaling_fit.png              - log σ² vs (t+τ) linear fit
    06_singular_value_spectra.png            - spectra by depth + by step
    07_successive_layer_angles.png           - mean angle R(t) vs R(t+1)
    08_pairwise_residual_heatmap.png         - log var(t, t+τ) heatmap

The Phase 1 narrative (per the proposal) is built around Figures 1, 2,
and 4 — the loss vs flow convergence comparison. Figures 5-8 are
diagnostics that validate the framework's assumptions on the trained
model.

Prerequisites:
  - The run must have been analyzed first (analyze_run() or
    validate_analyzer.py). Specifically, `run_dir/flow_analysis/` must
    contain flow .npz files.

Memory note:
  - Figure 7 (successive layer angles) needs the R matrices, which are
    large (~1 GB at H=896, L=14, K=50). They're loaded lazily and freed
    after use; peak RAM during report generation is roughly 2-3 GB.
"""

import argparse
import os
import sys
import time

from flow_series import load_flow_series
from plots import (
    plot_loss_curves,
    plot_basis_invariant_trajectories,
    plot_effective_rank_depth_profile,
    plot_flow_convergence,
    plot_variance_scaling_fit,
    plot_singular_value_spectra,
    plot_successive_layer_angles,
    plot_pairwise_residual_heatmap,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Run directory with flow_analysis/.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write plots. Default: run_dir/plots/.")
    parser.add_argument("--skip_R_plots", action="store_true",
                        help="Skip plots requiring R matrices (saves "
                             "~1 GB RAM during report generation).")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run_dir does not exist: {args.run_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.run_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    print(f">> Loading flow series from {args.run_dir} ...")
    t0 = time.time()
    fs = load_flow_series(args.run_dir)
    print(f"   ↳ Loaded {fs.K} checkpoints, L={fs.L} layer states, "
          f"H={fs.H} hidden, seed={fs.seed} "
          f"[{time.time() - t0:.1f}s]")
    print(f"   ↳ Training steps: {fs.steps[0]} to {fs.steps[-1]}")
    print(f"   ↳ Eval loss: {fs.eval_losses[0]:.4f} → {fs.eval_losses[-1]:.4f}")
    print()

    title_suffix = f" (seed={fs.seed})"

    plots_to_run = [
        ("01_loss_curves.png",
         lambda p: plot_loss_curves(fs, p, title_suffix=title_suffix)),
        ("02_basis_invariant_trajectories.png",
         lambda p: plot_basis_invariant_trajectories(fs, p, title_suffix=title_suffix)),
        ("03_effective_rank_depth_profile.png",
         lambda p: plot_effective_rank_depth_profile(fs, p, title_suffix=title_suffix)),
        ("04_flow_convergence.png",
         lambda p: plot_flow_convergence(fs, p, title_suffix=title_suffix)),
        ("05_variance_scaling_fit.png",
         lambda p: plot_variance_scaling_fit(fs, p, title_suffix=title_suffix)),
        ("06_singular_value_spectra.png",
         lambda p: plot_singular_value_spectra(fs, p, title_suffix=title_suffix)),
        ("08_pairwise_residual_heatmap.png",
         lambda p: plot_pairwise_residual_heatmap(fs, p, title_suffix=title_suffix)),
    ]
    if not args.skip_R_plots:
        plots_to_run.append(
            ("07_successive_layer_angles.png",
             lambda p: plot_successive_layer_angles(fs, p, title_suffix=title_suffix)),
        )

    print(f">> Generating {len(plots_to_run)} plots in {output_dir}/ ...")
    for filename, plot_fn in plots_to_run:
        path = os.path.join(output_dir, filename)
        t = time.time()
        try:
            plot_fn(path)
            print(f"   ↳ ✅ {filename}  [{time.time() - t:.1f}s]")
        except Exception as e:
            print(f"   ↳ ❌ {filename}: {type(e).__name__}: {e}")

    print()
    print(">> ✅ Report complete.")
    print(f"   Plots in: {output_dir}/")


if __name__ == "__main__":
    main()
