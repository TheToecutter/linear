"""
Driver for producing multi-view figures from the trajectory files.

Run after Stage D has completed. Produces:
  1. fig1_triptych_seed0.png       -- single-seed decomposition triptych
  2. fig2_triptych_meanband.png    -- four-seed average with seed-dispersion band
  3. fig3_ratio_profile.png        -- within/between ratio vs layer (raw)
  4. fig4_lambda_trajectory.png    -- log_alpha (all-to-all) + lambda (per view)
                                      through training
  5. fig5_dashboard_seed0.png      -- per-view dashboard at final checkpoint
  5b. fig5b_per_token_spread_seed0.png -- per-token forward spread profile
  6. fig6_ratio_profile_headline.png   -- polished, annotated version of fig3
                                          (the project's headline structure figure)
  7. fig7_colocation.png               -- co-location of all-to-all log_alpha hump
                                          and reverse-view lambda dip (the project's
                                          headline training-dynamic figure)
  8. fig8_ratio_heatmap.png            -- within/between ratio across training and
                                          depth, per view (the layer-by-step view of
                                          the structural and dynamic findings together)
  8b. fig8b_ratio_heatmap_delta.png    -- *change* in within/between ratio relative
                                          to first checkpoint (emphasizes training
                                          dynamics over the static profile)
  8c. fig8c_ratio_heatmap_with_init.png -- same as fig8 but with random-init baseline
                                          strip below the main heatmap; visualizes
                                          how training reshapes the initial profile.
                                          Requires init_check.py to have been run.

Usage:
    python3 make_plots.py [run_dir]

If run_dir is omitted, defaults to ../phase1_runs_gelu/multiview.
Output figures go to <run_dir>/../figures/ (created if missing).
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless; can be removed if running interactively
import matplotlib.pyplot as plt

# Import from the project's multiview modules.
from multiview import load_multi_view_result
from multiview_plots import (
    plot_decomposition_triptych,
    plot_per_view_dashboard,
    plot_per_token_spread,
    plot_log_alpha_trajectory,
    plot_ratio_profile_headline,
    plot_colocation,
    plot_ratio_heatmap,
    plot_ratio_heatmap_delta,
    plot_ratio_heatmap_with_init,
)


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------
def _resolve_paths(run_dir_arg: str | None) -> Tuple[str, str, str]:
    """Return (multiview_dir, trajectories_dir, figures_dir)."""
    if run_dir_arg is None:
        run_dir_arg = "../phase1_runs_gelu/multiview"
    multiview_dir = os.path.abspath(run_dir_arg)
    trajectories_dir = os.path.join(multiview_dir, "trajectories")
    figures_dir = os.path.join(os.path.dirname(multiview_dir), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return multiview_dir, trajectories_dir, figures_dir


# ----------------------------------------------------------------------
# Plot 1 & 5: single-seed plots from one MVR.
# ----------------------------------------------------------------------
def _final_step(trajectories_dir: str) -> int:
    """Read the steps array from any trajectory file and return the last one."""
    with np.load(os.path.join(trajectories_dir, "crossover.npz")) as f:
        return int(f["steps"][-1])


def make_single_seed_plots(multiview_dir: str, trajectories_dir: str,
                           figures_dir: str, seed: int = 0):
    step = _final_step(trajectories_dir)
    mvr_path = os.path.join(multiview_dir, f"seed_{seed}",
                            f"mvr_step_{step:08d}")
    print(f"[1] Loading MVR from seed {seed} step {step} ...")
    # We need decomposition arrays plus per-token effective_rank, kurtosis,
    # and singular_values for the dashboard. We don't need R, which is
    # the expensive one. Skip it to load quickly.
    r = load_multi_view_result(mvr_path, skip_arrays={"R"})

    out1 = os.path.join(figures_dir, "fig1_triptych_seed0.png")
    plot_decomposition_triptych(r, out1)
    print(f"[1] -> {out1}")

    out5 = os.path.join(figures_dir, "fig5_dashboard_seed0.png")
    plot_per_view_dashboard(r, out5)
    print(f"[5] -> {out5}")

    out_pt = os.path.join(figures_dir, "fig5b_per_token_spread_seed0.png")
    plot_per_token_spread(r, out_pt, view="forward")
    print(f"[5b] -> {out_pt}")


# ----------------------------------------------------------------------
# Plot 2: cross-seed averaged triptych with dispersion band.
# ----------------------------------------------------------------------
def make_cross_seed_triptych(trajectories_dir: str, figures_dir: str):
    """Cross-seed average decomposition at the final checkpoint, with
    a band showing the min/max range across seeds."""
    print("[2] Building cross-seed averaged decomposition ...")
    with np.load(os.path.join(trajectories_dir, "decomposition.npz")) as f:
        # Each array is (n_seeds, n_steps, L). Pick the final step.
        w_fwd = f["v_within_forward"][:, -1, :]            # (n_seeds, L)
        b_fwd = f["v_between_forward"][:, -1, :]
        s_fwd = f["v_subset_forward"][:, -1, :]
        w_ra = f["v_within_reverse_actual"][:, -1, :]
        b_ra = f["v_between_reverse_actual"][:, -1, :]
        s_ra = f["v_subset_reverse_actual"][:, -1, :]
        w_rp = f["v_within_reverse_pred"][:, -1, :]
        b_rp = f["v_between_reverse_pred"][:, -1, :]
        s_rp = f["v_subset_reverse_pred"][:, -1, :]

    def _frac(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, num / denom, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, w, b, s, title, direction in [
        (axes[0], w_fwd, b_fwd, s_fwd, "Forward (input-conditioned)", "forward"),
        (axes[1], w_ra, b_ra, s_ra, "Reverse (actual successor)", "reverse"),
        (axes[2], w_rp, b_rp, s_rp, "Reverse (predicted successor)", "reverse"),
    ]:
        L = w.shape[1]
        layers = np.arange(L)
        within_frac = _frac(w, s)  # (n_seeds, L)
        between_frac = _frac(b, s)

        # Mean and min/max across seeds.
        w_mean = np.nanmean(within_frac, axis=0)
        b_mean = np.nanmean(between_frac, axis=0)
        w_lo, w_hi = np.nanmin(within_frac, axis=0), np.nanmax(within_frac, axis=0)
        b_lo, b_hi = np.nanmin(between_frac, axis=0), np.nanmax(between_frac, axis=0)

        # Stacked plot: between on bottom, within on top of it.
        ax.fill_between(layers, 0, b_mean, alpha=0.55, color="C0",
                        label=r"between $\mathrm{Var}_z[\mu_t(z)]$")
        ax.fill_between(layers, b_mean, b_mean + w_mean, alpha=0.55, color="C3",
                        label=r"within $\mathbb{E}_z[V_z(t)]$")
        # Range bands as thin dashed lines.
        ax.plot(layers, b_mean, "-", color="C0", lw=1.2)
        ax.plot(layers, b_lo, ":", color="C0", lw=0.8, alpha=0.6)
        ax.plot(layers, b_hi, ":", color="C0", lw=0.8, alpha=0.6)
        ax.plot(layers, b_mean + w_mean, "-", color="C3", lw=1.2)
        ax.plot(layers, b_lo + w_lo, ":", color="C3", lw=0.8, alpha=0.6)
        ax.plot(layers, b_hi + w_hi, ":", color="C3", lw=0.8, alpha=0.6)

        # Crossover (computed on the mean curves).
        from multiview import crossover_layer
        # Reconstruct absolute variances from the means for crossover.
        # We can equivalently work on the fractions since denominator cancels.
        c, status = crossover_layer(w_mean, b_mean, direction=direction)
        if status == "crossover":
            ax.axvline(c, color="k", ls="--", lw=1)
            ax.text(c + 0.2, 0.05, f"{c:.2f}", fontsize=10, color="k")
        elif status == "always_true":
            ax.text(0.5, 0.92, f"always {direction}-dominant",
                    transform=ax.transAxes, ha="center", fontsize=10)
        elif status == "no_crossover":
            ax.text(0.5, 0.92, f"no {direction} crossover",
                    transform=ax.transAxes, ha="center", fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("layer state index t")
        ax.set_xlim(0, L - 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("variance fraction")
    fig.suptitle(f"Three-view decomposition: cross-seed mean (solid) "
                 f"with seed range (dotted)  —  "
                 f"final checkpoint, 4 seeds")
    fig.tight_layout()
    out = os.path.join(figures_dir, "fig2_triptych_meanband.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[2] -> {out}")


# ----------------------------------------------------------------------
# Plot 3: within/between ratio profiles, per view.
# ----------------------------------------------------------------------
def make_ratio_profile(trajectories_dir: str, figures_dir: str):
    """Within/between ratio vs layer, all three views, all seeds overlaid."""
    print("[3] Building within/between ratio profile ...")
    with np.load(os.path.join(trajectories_dir, "decomposition.npz")) as f:
        w_fwd = f["v_within_forward"][:, -1, :]
        b_fwd = f["v_between_forward"][:, -1, :]
        w_ra = f["v_within_reverse_actual"][:, -1, :]
        b_ra = f["v_between_reverse_actual"][:, -1, :]
        w_rp = f["v_within_reverse_pred"][:, -1, :]
        b_rp = f["v_between_reverse_pred"][:, -1, :]

    def _ratio(w: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, w / b, np.nan)

    r_fwd = _ratio(w_fwd, b_fwd)
    r_ra = _ratio(w_ra, b_ra)
    r_rp = _ratio(w_rp, b_rp)

    L = r_fwd.shape[1]
    layers = np.arange(L)
    fig, ax = plt.subplots(figsize=(10, 6))

    for arr, color, label in [
        (r_fwd, "C0", "forward (within / between by input)"),
        (r_ra, "C3", "reverse (within / between by actual successor)"),
        (r_rp, "C2", "reverse (within / between by predicted successor)"),
    ]:
        # Per-seed faint lines.
        for s in range(arr.shape[0]):
            ax.plot(layers, arr[s], "-", color=color, alpha=0.25, lw=0.8)
        # Mean across seeds.
        ax.plot(layers, np.nanmean(arr, axis=0), "-o", color=color,
                label=label, markersize=4, lw=1.6)

    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("within / between variance ratio")
    ax.set_title("Within/between ratio per view, final checkpoint\n"
                 "(faint = individual seeds; bold = cross-seed mean; dashed line = 1.0)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    out = os.path.join(figures_dir, "fig3_ratio_profile.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[3] -> {out}")


# ----------------------------------------------------------------------
# Plot 4: log_alpha trajectories through training.
# ----------------------------------------------------------------------
def make_log_alpha_trajectory(trajectories_dir: str, figures_dir: str):
    """Two-panel: all-to-all log_alpha (left) + per-view lambda (right)."""
    print("[4] Building log_alpha + lambda trajectory plot ...")
    out = os.path.join(figures_dir, "fig4_lambda_trajectory.png")
    plot_log_alpha_trajectory(
        os.path.join(trajectories_dir, "variance_fit.npz"),
        out,
    )
    print(f"[4] -> {out}")


def make_headline_ratio_plot(trajectories_dir: str, figures_dir: str):
    """Polished, annotated within/between ratio profile."""
    print("[6] Building headline ratio profile ...")
    out = os.path.join(figures_dir, "fig6_ratio_profile_headline.png")
    plot_ratio_profile_headline(
        os.path.join(trajectories_dir, "decomposition.npz"),
        out,
    )
    print(f"[6] -> {out}")


def make_colocation_plot(trajectories_dir: str, figures_dir: str):
    """Co-location of all-to-all log_alpha hump and reverse lambda dip."""
    print("[7] Building co-location plot ...")
    out = os.path.join(figures_dir, "fig7_colocation.png")
    plot_colocation(
        os.path.join(trajectories_dir, "variance_fit.npz"),
        out,
    )
    print(f"[7] -> {out}")


def make_ratio_heatmap(trajectories_dir: str, figures_dir: str):
    """Training-dynamic heatmap of within/between ratio per view."""
    print("[8] Building training-dynamic ratio heatmap ...")
    out = os.path.join(figures_dir, "fig8_ratio_heatmap.png")
    plot_ratio_heatmap(
        os.path.join(trajectories_dir, "decomposition.npz"),
        out,
    )
    print(f"[8] -> {out}")


def make_ratio_heatmap_delta(trajectories_dir: str, figures_dir: str):
    """Training-dynamic heatmap of *change* in within/between ratio,
    relative to first checkpoint."""
    print("[8b] Building training-dynamic ratio delta heatmap ...")
    out = os.path.join(figures_dir, "fig8b_ratio_heatmap_delta.png")
    plot_ratio_heatmap_delta(
        os.path.join(trajectories_dir, "decomposition.npz"),
        out,
    )
    print(f"[8b] -> {out}")


def make_ratio_heatmap_with_init(multiview_dir: str, trajectories_dir: str,
                                 figures_dir: str):
    """Training-dynamic ratio heatmap with random-init baseline strip.
    Requires init_baseline/init_seedNNNN.npz from running init_check.py.

    init_check.py saves to <multiview_dir>/init_baseline/ (sibling of
    the trajectories directory, both inside the multiview dir).
    """
    print("[8c] Building init-augmented ratio heatmap ...")
    init_dir = os.path.join(multiview_dir, "init_baseline")
    if not os.path.isdir(init_dir):
        print(f"[8c] No init_baseline directory found at {init_dir}; "
              f"run init_check.py first. Skipping.")
        return
    candidates = sorted(f for f in os.listdir(init_dir)
                        if f.startswith("init_seed") and f.endswith(".npz"))
    if not candidates:
        print(f"[8c] No init_seedNNNN.npz in {init_dir}; skipping.")
        return
    init_path = os.path.join(init_dir, candidates[0])
    out = os.path.join(figures_dir, "fig8c_ratio_heatmap_with_init.png")
    plot_ratio_heatmap_with_init(
        os.path.join(trajectories_dir, "decomposition.npz"),
        init_path,
        out,
    )
    print(f"[8c] -> {out}  (using {candidates[0]})")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    run_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    multiview_dir, trajectories_dir, figures_dir = _resolve_paths(run_dir_arg)
    print(f"multiview_dir:    {multiview_dir}")
    print(f"trajectories_dir: {trajectories_dir}")
    print(f"figures_dir:      {figures_dir}")
    print()

    make_single_seed_plots(multiview_dir, trajectories_dir, figures_dir)
    make_cross_seed_triptych(trajectories_dir, figures_dir)
    make_ratio_profile(trajectories_dir, figures_dir)
    make_log_alpha_trajectory(trajectories_dir, figures_dir)
    make_headline_ratio_plot(trajectories_dir, figures_dir)
    make_colocation_plot(trajectories_dir, figures_dir)
    make_ratio_heatmap(trajectories_dir, figures_dir)
    make_ratio_heatmap_delta(trajectories_dir, figures_dir)
    make_ratio_heatmap_with_init(multiview_dir, trajectories_dir, figures_dir)
    print()
    print("All figures written.")


if __name__ == "__main__":
    main()
    