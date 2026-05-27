"""
Cross-seed comparison plots and summary statistics for Phase 1.

This script consumes multiple seed runs (each a complete `--run_dir` with
`flow_analysis/` already populated) and produces:

  - Multi-seed overlay plots for the basis-invariant trajectories,
    loss curves, and flow-convergence panels.
  - A summary table (text + CSV) of final-checkpoint values across
    seeds, with mean, std, range, and a "1.5×-std" universality
    threshold useful for Phase 2 planning.

Usage:
    python3 compare_seeds.py \
        --run_dirs ../phase1_runs/seed_0 ../phase1_runs/seed_1 \
                   ../phase1_runs/seed_2 \
        --output_dir ../phase1_runs/cross_seed_plots

If --output_dir is omitted, output goes to a `cross_seed_plots/`
directory alongside the run dirs (or in the cwd if they're not
co-located).

Outputs (relative to --output_dir):
    cross_loss_curves.png
    cross_basis_invariant.png
    cross_flow_convergence.png
    cross_summary_table.txt
    cross_summary_table.csv

This is a Phase-1 cross-seed analysis. It uses only the basis-invariant
fields produced by `analyze.py`; it does not touch R matrices and does
not perform any alignment. (Alignment is a Phase 2 concern.)

Both the "ours" and "paper" statistic conventions are surfaced where
they exist. Older flow .npz files lacking the paper-convention fields
are handled gracefully — paper-convention rows fall back to NaN in
that case.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_series import load_flow_series, FlowSeries
from analyze import load_flow
from distances import singular_value_distance, effective_rank_distance


# ----------------------------------------------------------------------
# Color scheme — distinct per seed, consistent across plots.
# ----------------------------------------------------------------------
# These are picked from matplotlib's tab10 palette but reordered so the
# first few are visually distinct and colorblind-distinguishable.
SEED_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


_setup_style()


# ----------------------------------------------------------------------
# Per-seed data bundle.
# ----------------------------------------------------------------------
class SeedBundle:
    """All the cross-seed-relevant data for one seed, plus paper-convention
    fields pulled in from the .npz files (which FlowSeries doesn't load)."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.fs: FlowSeries = load_flow_series(run_dir)
        self.label = f"seed {self.fs.seed}"
        # Paper-convention fields live on disk but not in FlowSeries.
        # We pull them straight from the .npz files.
        self.lambda_paper = np.full(self.fs.K, np.nan, dtype=np.float32)
        self.log_alpha_paper = np.full(self.fs.K, np.nan, dtype=np.float32)
        # (K, L) — per-checkpoint per-layer mean-of-|κ|.
        self.kurtosis_abs = np.full(
            (self.fs.K, self.fs.L), np.nan, dtype=np.float32,
        )
        for k, path in enumerate(self.fs.flow_paths):
            flow = load_flow(path)
            if "lambda_paper" in flow:
                self.lambda_paper[k] = float(flow["lambda_paper"])
            if "log_alpha_paper" in flow:
                self.log_alpha_paper[k] = float(flow["log_alpha_paper"])
            if "kurtosis_abs_per_layer" in flow:
                self.kurtosis_abs[k] = flow["kurtosis_abs_per_layer"]

    @property
    def has_paper_convention(self) -> bool:
        return not np.all(np.isnan(self.log_alpha_paper))


def load_seeds(run_dirs: List[str]) -> List[SeedBundle]:
    """Load all seed bundles, sorted by seed number for consistent plotting."""
    bundles = []
    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"run_dir not found: {run_dir}")
        bundles.append(SeedBundle(run_dir))
    # Sort by seed number so colors map deterministically.
    bundles.sort(key=lambda b: b.fs.seed)
    # Validate compatibility: all seeds should share L and H.
    L0, H0 = bundles[0].fs.L, bundles[0].fs.H
    for b in bundles[1:]:
        if b.fs.L != L0 or b.fs.H != H0:
            raise ValueError(
                f"{b.run_dir} has L={b.fs.L}, H={b.fs.H}; expected "
                f"L={L0}, H={H0} to match seed {bundles[0].fs.seed}. "
                f"Cross-seed comparison requires matching architecture."
            )
    return bundles


# ----------------------------------------------------------------------
# Convergence diagnostic: H1 criterion.
# ----------------------------------------------------------------------
def compute_sv_distance_trajectory(fs: FlowSeries) -> np.ndarray:
    """Distance from each checkpoint's Σ-spectrum to the final checkpoint's.
    Equivalent to plots.plot_flow_convergence's blue curve."""
    ref_flow = {
        "singular_values": fs.singular_values[-1],
        "effective_rank": fs.effective_ranks[-1],
    }
    dists = np.zeros(fs.K, dtype=np.float32)
    for k in range(fs.K):
        flow_k = {
            "singular_values": fs.singular_values[k],
            "effective_rank": fs.effective_ranks[k],
        }
        dists[k] = singular_value_distance(flow_k, ref_flow)
    return dists


def compute_er_distance_trajectory(fs: FlowSeries) -> np.ndarray:
    """Effective-rank distance to final, per checkpoint."""
    ref_flow = {"effective_rank": fs.effective_ranks[-1]}
    dists = np.zeros(fs.K, dtype=np.float32)
    for k in range(fs.K):
        flow_k = {"effective_rank": fs.effective_ranks[k]}
        dists[k] = effective_rank_distance(flow_k, ref_flow)
    return dists


def compute_h1_criterion(sv_distances: np.ndarray) -> Tuple[float, float, bool]:
    """
    H1 success criterion from the proposal: last 25% of training, the
    distance-to-final std should be ≤ 10% of the total reduction.

    Returns (last_quarter_std, total_reduction, passed).
    """
    K = len(sv_distances)
    total_reduction = sv_distances[0] - sv_distances[-1]
    if total_reduction <= 0:
        return 0.0, total_reduction, False
    last_quarter_start = int(0.75 * K)
    last_quarter = sv_distances[last_quarter_start:]
    last_quarter_std = float(last_quarter.std())
    passed = last_quarter_std <= 0.10 * total_reduction
    return last_quarter_std, total_reduction, passed


# ----------------------------------------------------------------------
# Plot: loss curves overlay.
# ----------------------------------------------------------------------
def plot_cross_loss_curves(bundles: List[SeedBundle], output_path: str):
    """Eval loss vs step, all seeds overlaid. Sanity check that the
    seeds produce nearly identical learning curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for i, b in enumerate(bundles):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(
            b.fs.steps, b.fs.eval_losses,
            marker="o", markersize=2.5, lw=1.5,
            color=color, label=b.label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("held-out eval loss")
    ax.set_title(f"Eval loss across {len(bundles)} seeds")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot: basis-invariant trajectories overlay.
# ----------------------------------------------------------------------
def plot_cross_basis_invariant(bundles: List[SeedBundle], output_path: str):
    """6-panel mirror of plots.plot_basis_invariant_trajectories, with
    one line per seed on each panel.

    Layout:
      (a) λ                          (b) log α
      (c) mean effective rank        (d) mean kurtosis (signed)
      (e) mean isotropy              (f) eval loss
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    for i, b in enumerate(bundles):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        fs = b.fs
        # (a) λ.
        axes[0, 0].plot(
            fs.steps, fs.lambda_values, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )
        # (b) log α.
        axes[0, 1].plot(
            fs.steps, fs.log_alpha_values, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )
        # (c) mean effective rank.
        mean_er = fs.effective_ranks.mean(axis=1)
        axes[1, 0].plot(
            fs.steps, mean_er, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )
        # (d) mean kurtosis (signed).
        mean_kurt = np.nanmean(fs.kurtosis, axis=1)
        axes[1, 1].plot(
            fs.steps, mean_kurt, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )
        # (e) mean isotropy.
        mean_iso = np.nanmean(fs.isotropy, axis=1)
        axes[2, 0].plot(
            fs.steps, mean_iso, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )
        # (f) eval loss.
        axes[2, 1].plot(
            fs.steps, fs.eval_losses, marker="o", markersize=2.5, lw=1.4,
            color=color, label=b.label,
        )

    # Axis labels & titles.
    axes[0, 0].set_ylabel("λ (variance scaling rate)")
    axes[0, 0].set_title("λ")
    axes[0, 1].set_ylabel("log α (variance prefactor)")
    axes[0, 1].set_title("log α")
    axes[1, 0].set_ylabel("mean effective rank")
    axes[1, 0].set_title(f"Mean effective rank (across {bundles[0].fs.L} layer states)")
    axes[1, 0].axhline(bundles[0].fs.H, color="gray", ls=":", lw=1,
                       label=f"H = {bundles[0].fs.H}")
    axes[1, 1].set_ylabel("mean excess kurtosis of residuals")
    axes[1, 1].set_title("Residual kurtosis (0 = Gaussian)")
    axes[1, 1].axhline(0, color="gray", ls=":", lw=1)
    axes[2, 0].set_ylabel("mean isotropy (std of log Σ²_residual)")
    axes[2, 0].set_title("Residual isotropy (0 = perfectly isotropic)")
    axes[2, 1].set_ylabel("held-out eval loss")
    axes[2, 1].set_title("Eval loss (for cross-reference)")

    for ax in axes.flat:
        ax.set_xscale("log")
    for ax in axes[-1, :]:
        ax.set_xlabel("training step (log scale)")

    # One legend on the first panel; the rest share it visually via color.
    axes[0, 0].legend(loc="lower right", fontsize=8)
    # Effective-rank panel has the H=… legend entry too — keep both.
    axes[1, 0].legend(loc="lower right", fontsize=8)

    fig.suptitle(
        f"Basis-invariant convergence diagnostics — "
        f"{len(bundles)}-seed overlay (ours convention)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Plot: flow convergence overlay.
# ----------------------------------------------------------------------
def plot_cross_flow_convergence(
    bundles: List[SeedBundle], output_path: str,
):
    """Mirror of plots.plot_flow_convergence with seeds overlaid.

    Left panel: absolute Σ-spectrum and effective-rank distances to final
    Right panel: normalized distance + normalized loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for i, b in enumerate(bundles):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        fs = b.fs
        sv_dists = compute_sv_distance_trajectory(fs)
        er_dists = compute_er_distance_trajectory(fs)

        # Left panel: Σ-spectrum distance (solid) and effective rank (dashed).
        ax1.plot(
            fs.steps, sv_dists, marker="o", markersize=2.5, lw=1.5,
            color=color, label=f"{b.label} (Σ)",
        )
        ax1.plot(
            fs.steps, er_dists, marker="s", markersize=2.5, lw=1.5,
            color=color, ls="--", label=f"{b.label} (eff rank)",
        )

        # Right panel: normalized Σ-spectrum + normalized eval loss.
        sv_norm = sv_dists / (sv_dists[0] + 1e-30)
        loss_norm = (fs.eval_losses - fs.eval_losses[-1]) / (
            fs.eval_losses[0] - fs.eval_losses[-1] + 1e-30
        )
        ax2.plot(
            fs.steps, sv_norm, marker="o", markersize=2.5, lw=1.5,
            color=color, label=f"{b.label} (Σ)",
        )
        ax2.plot(
            fs.steps, loss_norm, marker="o", markersize=2.5, lw=1.0,
            color=color, ls=":", alpha=0.7, label=f"{b.label} (loss)",
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("training step (log scale)")
    ax1.set_ylabel(f"distance to checkpoint at final step")
    ax1.set_title("Flow-distance convergence (per seed)")
    ax1.legend(loc="upper right", fontsize=7, ncol=2)

    ax2.set_xscale("log")
    ax2.set_xlabel("training step (log scale)")
    ax2.set_ylabel("normalized distance to final (0 = converged)")
    ax2.set_title("Normalized flow convergence vs loss convergence")
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.axhline(1, color="gray", ls=":", lw=1)
    ax2.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle(
        f"Convergence of the linear flow to its final state — "
        f"{len(bundles)}-seed overlay",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary table.
# ----------------------------------------------------------------------
def collect_summary_rows(bundles: List[SeedBundle]) -> List[Dict]:
    """
    Build a per-quantity summary table.

    Each row has:
        name, [per-seed values], mean, std, range, 1.5*std_threshold
    where the per-seed values are at the final checkpoint.

    "1.5*std_threshold" is the threshold a Phase 2 cross-architecture
    comparison would need to clear to claim a non-trivial difference,
    given how much variance exists within a single architecture.

    Quantities included:
        eval loss (final)
        λ (ours, paper)
        log α (ours, paper)
        <κ> (signed mean kurtosis, mean over layers)
        <|κ|> (mean of absolute kurtosis, mean over layers; paper-style)
        mean effective rank (mean over layers)
        eff rank at L=0 (post-embedding)
        eff rank at mid-network (layer L//2)
        H1 last-quarter std (sanity check for the convergence criterion)
    """
    rows = []
    n_seeds = len(bundles)

    def add_row(name: str, values_per_seed: List[float]):
        arr = np.array(values_per_seed, dtype=np.float64)
        if np.all(np.isnan(arr)):
            mean_, std_, range_, threshold_ = (float("nan"),) * 4
        else:
            mean_ = float(np.nanmean(arr))
            # Use ddof=1 (sample std). Need ≥2 non-NaN values to compute it;
            # numpy warns otherwise.
            n_finite = int(np.sum(~np.isnan(arr)))
            if n_finite >= 2:
                with np.errstate(invalid="ignore"):
                    std_ = float(np.nanstd(arr, ddof=1))
            else:
                std_ = float("nan")
            range_ = float(np.nanmax(arr) - np.nanmin(arr))
            threshold_ = 1.5 * std_ if not np.isnan(std_) else float("nan")
        rows.append({
            "name": name,
            "values": values_per_seed,
            "mean": mean_,
            "std": std_,
            "range": range_,
            "threshold_1p5_std": threshold_,
        })

    # Helpers to pull last-checkpoint scalars from a bundle.
    def last_lambda_ours(b):       return float(b.fs.lambda_values[-1])
    def last_lambda_paper(b):      return float(b.lambda_paper[-1])
    def last_log_alpha_ours(b):    return float(b.fs.log_alpha_values[-1])
    def last_log_alpha_paper(b):   return float(b.log_alpha_paper[-1])
    def last_eval_loss(b):         return float(b.fs.eval_losses[-1])
    def last_mean_kurt(b):
        return float(np.nanmean(b.fs.kurtosis[-1]))
    def last_mean_kurt_abs(b):
        return float(np.nanmean(b.kurtosis_abs[-1])) \
            if not np.all(np.isnan(b.kurtosis_abs[-1])) else float("nan")
    def last_mean_eff_rank(b):
        return float(b.fs.effective_ranks[-1].mean())
    def last_eff_rank_L0(b):
        return float(b.fs.effective_ranks[-1, 0])
    def last_eff_rank_mid(b):
        L = b.fs.L
        return float(b.fs.effective_ranks[-1, L // 2])
    def last_quarter_std(b):
        sv_d = compute_sv_distance_trajectory(b.fs)
        s, _, _ = compute_h1_criterion(sv_d)
        return s
    def total_reduction(b):
        sv_d = compute_sv_distance_trajectory(b.fs)
        _, r, _ = compute_h1_criterion(sv_d)
        return r

    add_row("eval loss (final)",         [last_eval_loss(b) for b in bundles])
    add_row("λ (ours)",                  [last_lambda_ours(b) for b in bundles])
    add_row("λ (paper)",                 [last_lambda_paper(b) for b in bundles])
    add_row("log α (ours)",              [last_log_alpha_ours(b) for b in bundles])
    add_row("log α (paper)",             [last_log_alpha_paper(b) for b in bundles])
    add_row("<κ> (signed mean kurt)",    [last_mean_kurt(b) for b in bundles])
    add_row("<|κ|> (paper kurt)",        [last_mean_kurt_abs(b) for b in bundles])
    add_row("mean effective rank",       [last_mean_eff_rank(b) for b in bundles])
    add_row("eff rank L=0",              [last_eff_rank_L0(b) for b in bundles])
    add_row("eff rank mid",              [last_eff_rank_mid(b) for b in bundles])
    add_row("H1: last-quarter std",      [last_quarter_std(b) for b in bundles])
    add_row("H1: total reduction",       [total_reduction(b) for b in bundles])

    return rows


def write_summary_text(rows: List[Dict], bundles: List[SeedBundle],
                       output_path: str):
    """Pretty-print the summary table to a text file."""
    n_seeds = len(bundles)
    seed_labels = [b.label for b in bundles]

    # Compute column widths.
    name_w = max(len(r["name"]) for r in rows)
    name_w = max(name_w, len("quantity"))
    val_w = max(11, max(len(s) for s in seed_labels) + 1)
    # Three summary cols: mean / std / range. Threshold optional.
    summary_w = 11

    def fmt_num(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "       —   "
        # Choose precision based on magnitude.
        ax = abs(x)
        if ax == 0:
            return f"{x:+{summary_w}.4f}"
        if ax >= 100:
            return f"{x:+{summary_w}.2f}"
        if ax >= 1:
            return f"{x:+{summary_w}.4f}"
        return f"{x:+{summary_w}.4f}"

    with open(output_path, "w") as f:
        # Header.
        title = f"Cross-seed summary ({n_seeds} seed{'s' if n_seeds != 1 else ''}: " + \
                ", ".join(seed_labels) + ")"
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write("All values are at the final checkpoint of each seed.\n")
        f.write("std uses ddof=1 (sample std). Threshold = 1.5 × std,\n")
        f.write("a guide for Phase 2 within-variant universality margins.\n\n")

        # Column header.
        header = f"{'quantity':<{name_w}}"
        for lbl in seed_labels:
            header += f"  {lbl:>{val_w}}"
        header += f"  {'mean':>{summary_w}}  {'std':>{summary_w}}  " \
                  f"{'range':>{summary_w}}  {'1.5×std':>{summary_w}}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for r in rows:
            line = f"{r['name']:<{name_w}}"
            for v in r["values"]:
                line += "  " + fmt_num(v).rjust(val_w)
            line += "  " + fmt_num(r["mean"]).rjust(summary_w)
            line += "  " + fmt_num(r["std"]).rjust(summary_w)
            line += "  " + fmt_num(r["range"]).rjust(summary_w)
            line += "  " + fmt_num(r["threshold_1p5_std"]).rjust(summary_w)
            f.write(line + "\n")

        # H1 pass/fail per seed.
        f.write("\n")
        f.write("H1 convergence criterion (last-quarter std ≤ 10% of total "
                "reduction):\n")
        for b in bundles:
            sv_d = compute_sv_distance_trajectory(b.fs)
            s, r_, passed = compute_h1_criterion(sv_d)
            verdict = "PASS" if passed else "FAIL"
            f.write(
                f"  {b.label:<14} last-qtr std = {s:>9.3f}, "
                f"total reduction = {r_:>9.3f}, "
                f"ratio = {s / max(r_, 1e-30):.4f}   "
                f"[{verdict}]\n"
            )


def write_summary_csv(rows: List[Dict], bundles: List[SeedBundle],
                      output_path: str):
    """Write the summary table to CSV for programmatic consumption."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header.
        header = ["quantity"]
        for b in bundles:
            header.append(b.label)
        header.extend(["mean", "std_ddof1", "range", "threshold_1p5_std"])
        writer.writerow(header)
        for r in rows:
            row = [r["name"]] + list(r["values"]) + [
                r["mean"], r["std"], r["range"], r["threshold_1p5_std"],
            ]
            writer.writerow(row)


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-seed Phase 1 comparison plots and summary."
    )
    parser.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="Paths to seed run directories (each must contain "
             "flow_analysis/ already populated by validate_analyzer.py "
             "or analyze_run).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write outputs. Default: cross_seed_plots/ "
             "alongside the first run dir's parent.",
    )
    args = parser.parse_args()

    # Resolve output dir.
    if args.output_dir is None:
        first_parent = os.path.dirname(os.path.abspath(args.run_dirs[0]))
        args.output_dir = os.path.join(first_parent, "cross_seed_plots")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all seeds.
    print(f">> Loading {len(args.run_dirs)} seed run(s) ...")
    bundles = load_seeds(args.run_dirs)
    for b in bundles:
        n_paper = int(b.has_paper_convention)
        print(f"   ↳ {b.label}: K={b.fs.K}, L={b.fs.L}, H={b.fs.H}, "
              f"paper-convention fields: {'yes' if n_paper else 'no'}")
    print()

    print(f">> Writing outputs to {args.output_dir} ...")

    # Plots.
    path = os.path.join(args.output_dir, "cross_loss_curves.png")
    plot_cross_loss_curves(bundles, path)
    print(f"   ↳ {path}")

    path = os.path.join(args.output_dir, "cross_basis_invariant.png")
    plot_cross_basis_invariant(bundles, path)
    print(f"   ↳ {path}")

    path = os.path.join(args.output_dir, "cross_flow_convergence.png")
    plot_cross_flow_convergence(bundles, path)
    print(f"   ↳ {path}")

    # Summary table.
    rows = collect_summary_rows(bundles)
    path_txt = os.path.join(args.output_dir, "cross_summary_table.txt")
    write_summary_text(rows, bundles, path_txt)
    print(f"   ↳ {path_txt}")

    path_csv = os.path.join(args.output_dir, "cross_summary_table.csv")
    write_summary_csv(rows, bundles, path_csv)
    print(f"   ↳ {path_csv}")

    # Echo the text table to stdout for convenience.
    print()
    with open(path_txt) as f:
        print(f.read())

    print(">> ✅ Cross-seed comparison complete.")


if __name__ == "__main__":
    main()
