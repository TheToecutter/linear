"""
Boundary-layer exclusion check for Phase 1.

The paper (Sarfati et al., §2 and §4.3) excludes the embedding state and
the post-final-norm state from its variance-scaling analyses. Our
analyzer includes both. The post-final-norm state in particular sits
~2 log units below the fit line on our converged-checkpoint data (plot
05_variance_scaling_fit.png), pulling log α downward in the global fit.

This script re-fits log α and λ from the saved pairwise residual
variance arrays with the boundary layers excluded, parallel to the
paper's convention. The fit operates entirely on `pairwise_residual_variance`
(ours convention) and `pairwise_mean_log_var` (paper convention)
already stored in the flow .npz files; no re-running of the analyzer
or model is needed.

Outputs:
  - boundary_summary.txt   — per-seed table of included/excluded log α/λ
  - boundary_summary.csv   — same, machine-readable
  - boundary_log_alpha_trajectory.png — log α vs step, with/without exclusion
  - boundary_variance_fit_final.png   — final-checkpoint fit, both versions

Usage:
    python3 boundary_layer_check.py \\
        --run_dirs ../phase1_runs/seed_0 ../phase1_runs/seed_1 \\
                   ../phase1_runs/seed_2 \\
        --output_dir ../phase1_runs/boundary_check

The exclusion is symmetric: both source and target dimensions drop the
specified layers. By default we exclude layer 0 (post-embedding) and
layer L-1 (post-final-norm). This is overridable via --exclude_layers.
"""

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_series import load_flow_series, FlowSeries
from analyze import load_flow


# Colors shared with compare_seeds.py for visual consistency.
SEED_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
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
# Core: refit with arbitrary subset of source/target layers excluded.
# ----------------------------------------------------------------------
def refit_variance_scaling(
    pairwise_var: np.ndarray,             # (L, L), ours-convention residual var
    pairwise_mean_log_var: np.ndarray,    # (L, L), paper-convention mean log var
    exclude_layers: List[int],
) -> Dict:
    """
    Re-marginalize and re-fit the variance scaling line from the pairwise
    residual matrices, dropping the specified layer indices from both the
    source and target axes.

    Args:
        pairwise_var: (L, L). Entry [t, target] = mean_d var_d for that pair.
            Entries with t >= target are NaN.
        pairwise_mean_log_var: (L, L). Entry [t, target] = mean_d log(var_d).
            Same NaN pattern.
        exclude_layers: list of layer indices to exclude as both sources
            and targets. For the paper-style fit on L=14 layer states,
            this is [0, 13].

    Returns:
        Dict with:
          - 'endpoint_indices': (K_kept,) target-layer indices kept in fit
          - 'endpoint_log_var': (K_kept,) "ours" y-values fitted
          - 'endpoint_mean_log_var': (K_kept,) "paper" y-values fitted
          - 'log_alpha', 'lambda': linear fit "ours" convention
          - 'log_alpha_paper', 'lambda_paper': linear fit "paper" convention
          - 'n_endpoints_fitted': K_kept (informational)
    """
    L = pairwise_var.shape[0]
    excl = set(int(e) for e in exclude_layers)

    endpoint_indices = []
    endpoint_log_var = []
    endpoint_mean_log_var = []
    for end in range(1, L):
        if end in excl:
            continue
        # Sources are all t < end with t not in excluded set.
        sources_kept = [t for t in range(end) if t not in excl]
        if not sources_kept:
            continue
        vars_at_end = pairwise_var[sources_kept, end]
        mlv_at_end = pairwise_mean_log_var[sources_kept, end]
        # Drop NaNs (shouldn't be any in the upper triangle by construction,
        # but guard anyway).
        vars_at_end = vars_at_end[~np.isnan(vars_at_end)]
        mlv_at_end = mlv_at_end[~np.isnan(mlv_at_end)]
        if len(vars_at_end) == 0:
            continue
        endpoint_indices.append(end)
        endpoint_log_var.append(float(np.log(vars_at_end.mean())))
        endpoint_mean_log_var.append(float(mlv_at_end.mean()))

    endpoint_indices = np.array(endpoint_indices, dtype=np.float32)
    endpoint_log_var = np.array(endpoint_log_var, dtype=np.float32)
    endpoint_mean_log_var = np.array(endpoint_mean_log_var, dtype=np.float32)

    if len(endpoint_indices) >= 2:
        lam, log_alpha = np.polyfit(endpoint_indices, endpoint_log_var, deg=1)
        lam_paper, log_alpha_paper = np.polyfit(
            endpoint_indices, endpoint_mean_log_var, deg=1,
        )
    else:
        lam = log_alpha = lam_paper = log_alpha_paper = float("nan")

    return {
        "endpoint_indices": endpoint_indices,
        "endpoint_log_var": endpoint_log_var,
        "endpoint_mean_log_var": endpoint_mean_log_var,
        "log_alpha": float(log_alpha),
        "lambda": float(lam),
        "log_alpha_paper": float(log_alpha_paper),
        "lambda_paper": float(lam_paper),
        "n_endpoints_fitted": int(len(endpoint_indices)),
    }


# ----------------------------------------------------------------------
# Per-seed processing.
# ----------------------------------------------------------------------
class SeedRefitBundle:
    """Holds both the as-saved and the boundary-excluded fits for one seed,
    across all checkpoints, plus the final-checkpoint fit details for
    plotting Figure 05-style scatter."""

    def __init__(self, run_dir: str, exclude_layers: List[int]):
        self.run_dir = run_dir
        self.fs = load_flow_series(run_dir)
        self.exclude_layers = exclude_layers
        self.label = f"seed {self.fs.seed}"

        K = self.fs.K
        # As-saved fits (these are already in flow_series for ours; we pull
        # paper from the .npz files since flow_series doesn't load them).
        self.log_alpha_incl = np.array(self.fs.log_alpha_values, dtype=np.float64)
        self.lambda_incl = np.array(self.fs.lambda_values, dtype=np.float64)
        self.log_alpha_paper_incl = np.full(K, np.nan, dtype=np.float64)
        self.lambda_paper_incl = np.full(K, np.nan, dtype=np.float64)

        # Boundary-excluded fits, recomputed.
        self.log_alpha_excl = np.full(K, np.nan, dtype=np.float64)
        self.lambda_excl = np.full(K, np.nan, dtype=np.float64)
        self.log_alpha_paper_excl = np.full(K, np.nan, dtype=np.float64)
        self.lambda_paper_excl = np.full(K, np.nan, dtype=np.float64)

        # Final-checkpoint scatter data for the Figure-05-style plot.
        self.final_fit_data: Optional[Dict] = None

        for k, path in enumerate(self.fs.flow_paths):
            flow = load_flow(path)
            if "lambda_paper" in flow:
                self.lambda_paper_incl[k] = float(flow["lambda_paper"])
            if "log_alpha_paper" in flow:
                self.log_alpha_paper_incl[k] = float(flow["log_alpha_paper"])
            pwv = flow.get("pairwise_residual_variance")
            pmlv = flow.get("pairwise_mean_log_var")
            if pwv is None or pmlv is None:
                # Pre-v3 .npz lacking paper-convention pairwise array;
                # we can't recompute paper-style excluded fit. Skip.
                continue
            refit = refit_variance_scaling(pwv, pmlv, exclude_layers)
            self.log_alpha_excl[k] = refit["log_alpha"]
            self.lambda_excl[k] = refit["lambda"]
            self.log_alpha_paper_excl[k] = refit["log_alpha_paper"]
            self.lambda_paper_excl[k] = refit["lambda_paper"]
            if k == K - 1:
                # Store full scatter data for the final-checkpoint plot.
                # Also build the "included" version for comparison.
                L = self.fs.L
                incl = refit_variance_scaling(pwv, pmlv, exclude_layers=[])
                self.final_fit_data = {
                    "L": L,
                    "step": int(flow["checkpoint_step"]),
                    "eval_loss": float(flow["checkpoint_eval_loss"]),
                    "included_endpoint_indices": incl["endpoint_indices"],
                    "included_endpoint_log_var": incl["endpoint_log_var"],
                    "included_log_alpha": incl["log_alpha"],
                    "included_lambda": incl["lambda"],
                    "excluded_endpoint_indices": refit["endpoint_indices"],
                    "excluded_endpoint_log_var": refit["endpoint_log_var"],
                    "excluded_log_alpha": refit["log_alpha"],
                    "excluded_lambda": refit["lambda"],
                    "excluded_layers": list(exclude_layers),
                }


def load_bundles(run_dirs: List[str], exclude_layers: List[int]) -> List[SeedRefitBundle]:
    bundles = [SeedRefitBundle(rd, exclude_layers) for rd in run_dirs]
    bundles.sort(key=lambda b: b.fs.seed)
    # Validate compat: matching L and H across seeds.
    L0, H0 = bundles[0].fs.L, bundles[0].fs.H
    for b in bundles[1:]:
        if b.fs.L != L0 or b.fs.H != H0:
            raise ValueError(
                f"{b.run_dir} has L={b.fs.L}, H={b.fs.H}; expected "
                f"L={L0}, H={H0} to match seed {bundles[0].fs.seed}."
            )
    # Validate exclude_layers are in range.
    for ex in exclude_layers:
        if not (0 <= ex < L0):
            raise ValueError(
                f"--exclude_layers value {ex} is outside [0, {L0}). "
                f"Layer state indices are 0 (post-embedding) through "
                f"{L0 - 1} (post-final-norm)."
            )
    return bundles


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_log_alpha_trajectory(bundles: List[SeedRefitBundle],
                              output_path: str,
                              exclude_layers: List[int]):
    """log α vs training step, with-and-without boundary exclusion,
    all seeds overlaid.

    Two panels:
      Left:  "ours" convention. Solid = included (as saved), dashed = excluded.
      Right: "paper" convention. Same line styles.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for i, b in enumerate(bundles):
        color = SEED_COLORS[i % len(SEED_COLORS)]
        steps = b.fs.steps
        # Left: ours convention.
        axL.plot(steps, b.log_alpha_incl, marker="o", markersize=2.5, lw=1.4,
                 color=color, ls="-",
                 label=f"{b.label} (all layers)")
        axL.plot(steps, b.log_alpha_excl, marker="s", markersize=2.5, lw=1.4,
                 color=color, ls="--",
                 label=f"{b.label} (excl. layers {exclude_layers})")
        # Right: paper convention.
        axR.plot(steps, b.log_alpha_paper_incl, marker="o", markersize=2.5,
                 lw=1.4, color=color, ls="-",
                 label=f"{b.label} (all layers)")
        axR.plot(steps, b.log_alpha_paper_excl, marker="s", markersize=2.5,
                 lw=1.4, color=color, ls="--",
                 label=f"{b.label} (excl. layers {exclude_layers})")

    axL.set_xscale("log")
    axL.set_xlabel("training step (log scale)")
    axL.set_ylabel("log α (variance prefactor)")
    axL.set_title("log α trajectory — ours convention")
    axL.legend(loc="lower right", fontsize=8, ncol=1)

    axR.set_xscale("log")
    axR.set_xlabel("training step (log scale)")
    axR.set_ylabel("log α (variance prefactor)")
    axR.set_title("log α trajectory — paper convention")
    axR.legend(loc="lower right", fontsize=8, ncol=1)

    fig.suptitle(
        f"Effect of boundary-layer exclusion on log α "
        f"(excluded: layers {exclude_layers})",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_variance_fit_final(bundles: List[SeedRefitBundle],
                            output_path: str):
    """Mirror of plot_variance_scaling_fit (Figure 5) at the final
    checkpoint, but showing BOTH the included and excluded fits.

    One panel per seed (small multiples). On each panel:
      - All endpoint points scattered
      - Included-fit line drawn through all points
      - Excluded-fit line drawn through only the kept subset
      - Dropped points highlighted differently
    """
    n = len(bundles)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows),
        squeeze=False, sharey=True,
    )

    for i, b in enumerate(bundles):
        ax = axes[i // ncols, i % ncols]
        if b.final_fit_data is None:
            ax.text(
                0.5, 0.5,
                f"{b.label}: pairwise data not present\n"
                f"(pre-v3 flow .npz files)",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(b.label)
            continue

        d = b.final_fit_data
        incl_x = d["included_endpoint_indices"]
        incl_y = d["included_endpoint_log_var"]
        excl_x = d["excluded_endpoint_indices"]

        # Plot all points; color the kept-in-excluded-fit ones differently.
        kept_mask = np.isin(incl_x, excl_x)
        dropped_mask = ~kept_mask

        ax.scatter(incl_x[kept_mask], incl_y[kept_mask],
                   s=42, color="tab:blue", zorder=3,
                   label="kept (excluded fit)")
        ax.scatter(incl_x[dropped_mask], incl_y[dropped_mask],
                   s=42, facecolors="none", edgecolors="tab:red",
                   linewidths=1.5, zorder=3,
                   label="dropped boundary layers")

        # Fit lines.
        x_extent = np.array(
            [incl_x.min() - 0.5, incl_x.max() + 0.5], dtype=np.float32,
        )
        y_incl = d["included_log_alpha"] + d["included_lambda"] * x_extent
        y_excl = d["excluded_log_alpha"] + d["excluded_lambda"] * x_extent
        ax.plot(
            x_extent, y_incl, color="tab:gray", lw=1.5, ls="-",
            label=f"all-layer fit: "
                  f"log α = {d['included_log_alpha']:.3f}, "
                  f"λ = {d['included_lambda']:.4f}",
        )
        ax.plot(
            x_extent, y_excl, color="tab:blue", lw=1.8, ls="--",
            label=f"excluded fit: "
                  f"log α = {d['excluded_log_alpha']:.3f}, "
                  f"λ = {d['excluded_lambda']:.4f}",
        )

        delta_la = d["excluded_log_alpha"] - d["included_log_alpha"]
        delta_lam = d["excluded_lambda"] - d["included_lambda"]
        ax.set_title(
            f"{b.label} — step {d['step']}, eval {d['eval_loss']:.3f}\n"
            f"Δlog α = {delta_la:+.3f}, Δλ = {delta_lam:+.4f}",
        )
        ax.set_xlabel("target layer index (t+τ)")
        if i % ncols == 0:
            ax.set_ylabel("log(mean residual variance per coordinate)")
        ax.legend(loc="upper left", fontsize=7)

    # Hide unused axes.
    for j in range(len(bundles), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    fig.suptitle(
        "Variance scaling fit at final checkpoint — "
        "all-layer vs boundary-excluded (ours convention)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary table.
# ----------------------------------------------------------------------
def collect_summary_rows(bundles: List[SeedRefitBundle]) -> List[Dict]:
    """Per-seed final-checkpoint summary, with included/excluded both
    conventions and the deltas."""
    rows = []
    for b in bundles:
        d = {
            "seed": b.label,
            "log_alpha_ours_incl":  float(b.log_alpha_incl[-1]),
            "log_alpha_ours_excl":  float(b.log_alpha_excl[-1]),
            "delta_log_alpha_ours": float(b.log_alpha_excl[-1] - b.log_alpha_incl[-1]),
            "lambda_ours_incl":     float(b.lambda_incl[-1]),
            "lambda_ours_excl":     float(b.lambda_excl[-1]),
            "delta_lambda_ours":    float(b.lambda_excl[-1] - b.lambda_incl[-1]),
            "log_alpha_paper_incl":  float(b.log_alpha_paper_incl[-1]),
            "log_alpha_paper_excl":  float(b.log_alpha_paper_excl[-1]),
            "delta_log_alpha_paper": float(b.log_alpha_paper_excl[-1] - b.log_alpha_paper_incl[-1]),
            "lambda_paper_incl":     float(b.lambda_paper_incl[-1]),
            "lambda_paper_excl":     float(b.lambda_paper_excl[-1]),
            "delta_lambda_paper":    float(b.lambda_paper_excl[-1] - b.lambda_paper_incl[-1]),
        }
        rows.append(d)
    return rows


def write_summary_text(rows: List[Dict], exclude_layers: List[int],
                       paper_reference_log_alpha: Dict[str, float],
                       output_path: str):
    """Pretty text summary."""
    with open(output_path, "w") as f:
        title = (f"Boundary-layer exclusion check "
                 f"(excluded layers: {exclude_layers})")
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(
            "Per-seed final-checkpoint variance-scaling fit, computed\n"
            "with and without the boundary layers. Δ = excluded − included.\n"
            "Positive Δ on log α means excluding boundaries pulls log α\n"
            "upward (toward less-negative values).\n\n"
        )

        # Ours convention table.
        f.write("=== ours convention (log of mean per-coord variance) ===\n\n")
        header = (
            f"{'seed':<10}  {'log α incl':>11}  {'log α excl':>11}  "
            f"{'Δ log α':>10}  {'λ incl':>10}  {'λ excl':>10}  {'Δ λ':>10}"
        )
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            f.write(
                f"{r['seed']:<10}  "
                f"{r['log_alpha_ours_incl']:>+11.4f}  "
                f"{r['log_alpha_ours_excl']:>+11.4f}  "
                f"{r['delta_log_alpha_ours']:>+10.4f}  "
                f"{r['lambda_ours_incl']:>+10.4f}  "
                f"{r['lambda_ours_excl']:>+10.4f}  "
                f"{r['delta_lambda_ours']:>+10.4f}\n"
            )
        # Mean delta across seeds.
        if len(rows) >= 2:
            with np.errstate(invalid="ignore"):
                mean_d_la = float(np.nanmean([r["delta_log_alpha_ours"] for r in rows]))
                mean_d_lam = float(np.nanmean([r["delta_lambda_ours"] for r in rows]))
            f.write("-" * len(header) + "\n")
            f.write(
                f"{'mean Δ':<10}  "
                f"{'':>11}  {'':>11}  "
                f"{mean_d_la:>+10.4f}  "
                f"{'':>10}  {'':>10}  "
                f"{mean_d_lam:>+10.4f}\n"
            )

        # Paper convention table.
        f.write("\n=== paper convention (mean of per-coord log variance) ===\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for r in rows:
            f.write(
                f"{r['seed']:<10}  "
                f"{r['log_alpha_paper_incl']:>+11.4f}  "
                f"{r['log_alpha_paper_excl']:>+11.4f}  "
                f"{r['delta_log_alpha_paper']:>+10.4f}  "
                f"{r['lambda_paper_incl']:>+10.4f}  "
                f"{r['lambda_paper_excl']:>+10.4f}  "
                f"{r['delta_lambda_paper']:>+10.4f}\n"
            )
        if len(rows) >= 2:
            with np.errstate(invalid="ignore"):
                mean_d_la = float(np.nanmean([r["delta_log_alpha_paper"] for r in rows]))
                mean_d_lam = float(np.nanmean([r["delta_lambda_paper"] for r in rows]))
            f.write("-" * len(header) + "\n")
            f.write(
                f"{'mean Δ':<10}  "
                f"{'':>11}  {'':>11}  "
                f"{mean_d_la:>+10.4f}  "
                f"{'':>10}  {'':>10}  "
                f"{mean_d_lam:>+10.4f}\n"
            )

        # Gap-to-paper analysis.
        if paper_reference_log_alpha:
            f.write("\n=== gap to paper's published log α values ===\n\n")
            f.write(
                "After excluding boundary layers, how much of the gap to\n"
                "Sarfati et al.'s published log α values is closed?\n"
                "Reference values are for the paper convention; we compare\n"
                "to our paper-convention values.\n\n"
            )
            with np.errstate(invalid="ignore"):
                mean_la_paper_incl = float(np.nanmean(
                    [r["log_alpha_paper_incl"] for r in rows]
                ))
                mean_la_paper_excl = float(np.nanmean(
                    [r["log_alpha_paper_excl"] for r in rows]
                ))
            f.write(
                f"Our mean log α (paper, all layers):    {mean_la_paper_incl:+.4f}\n"
            )
            f.write(
                f"Our mean log α (paper, excl boundary): {mean_la_paper_excl:+.4f}\n"
            )
            f.write(
                f"Shift from exclusion:                  "
                f"{mean_la_paper_excl - mean_la_paper_incl:+.4f}\n\n"
            )
            for ref_name, ref_val in paper_reference_log_alpha.items():
                gap_incl = mean_la_paper_incl - ref_val
                gap_excl = mean_la_paper_excl - ref_val
                closed = abs(gap_incl) - abs(gap_excl)
                frac_closed = closed / abs(gap_incl) if abs(gap_incl) > 1e-9 else 0.0
                f.write(
                    f"  vs {ref_name} (paper log α = {ref_val:+.4f}):\n"
                    f"    gap (all layers):    {gap_incl:+.4f}\n"
                    f"    gap (excl boundary): {gap_excl:+.4f}\n"
                    f"    fraction closed:     {frac_closed:+.2%}\n\n"
                )


def write_summary_csv(rows: List[Dict], output_path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Re-fit log α and λ from saved flow .npz files with "
                    "the boundary layers (post-embedding and post-final-norm) "
                    "excluded, parallel to the paper's convention.",
    )
    parser.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="Paths to seed run directories (each must contain "
             "flow_analysis/ already populated with v3-format .npz files "
             "that include the pairwise_residual_variance and "
             "pairwise_mean_log_var arrays).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write outputs. Default: boundary_check/ alongside "
             "the first run_dir's parent.",
    )
    parser.add_argument(
        "--exclude_layers", type=int, nargs="*", default=None,
        help="Layer indices to exclude from both source and target axes. "
             "Default: [0, L-1] (post-embedding and post-final-norm), "
             "inferred from the first seed's L. Provide explicitly to "
             "override, e.g. --exclude_layers 0 13 for L=14 layer states.",
    )
    args = parser.parse_args()

    # Resolve output dir.
    if args.output_dir is None:
        first_parent = os.path.dirname(os.path.abspath(args.run_dirs[0]))
        args.output_dir = os.path.join(first_parent, "boundary_check")
    os.makedirs(args.output_dir, exist_ok=True)

    # Peek at the first seed to determine L for the default exclude_layers.
    if args.exclude_layers is None:
        peek_fs = load_flow_series(args.run_dirs[0])
        L = peek_fs.L
        args.exclude_layers = [0, L - 1]
        print(f">> Auto-selected boundary layers: {args.exclude_layers} "
              f"(L={L}; layer 0 = post-embedding, layer {L-1} = post-final-norm)")

    print(f">> Loading {len(args.run_dirs)} seed run(s) and refitting ...")
    bundles = load_bundles(args.run_dirs, args.exclude_layers)
    for b in bundles:
        n_paper = int(not np.all(np.isnan(b.log_alpha_paper_incl)))
        n_excl_paper = int(not np.all(np.isnan(b.log_alpha_paper_excl)))
        print(f"   ↳ {b.label}: K={b.fs.K}, L={b.fs.L}, "
              f"paper-incl: {'yes' if n_paper else 'no'}, "
              f"paper-excl recomputed: {'yes' if n_excl_paper else 'no'}")

    rows = collect_summary_rows(bundles)

    # Paper reference values for the gap-to-paper analysis. These come
    # from PAPER_CODE_REVIEW.md §6.1 — the paper convention values that
    # Sarfati et al. published for their two named models.
    paper_ref = {
        "GPT-2 medium (paper)": -0.45,
        "Llama-2-7B (paper)":   -5.4,
    }

    # Write artifacts.
    print(f"\n>> Writing outputs to {args.output_dir} ...")
    path = os.path.join(args.output_dir, "boundary_summary.txt")
    write_summary_text(rows, args.exclude_layers, paper_ref, path)
    print(f"   ↳ {path}")

    path = os.path.join(args.output_dir, "boundary_summary.csv")
    write_summary_csv(rows, path)
    print(f"   ↳ {path}")

    path = os.path.join(args.output_dir, "boundary_log_alpha_trajectory.png")
    plot_log_alpha_trajectory(bundles, path, args.exclude_layers)
    print(f"   ↳ {path}")

    path = os.path.join(args.output_dir, "boundary_variance_fit_final.png")
    plot_variance_fit_final(bundles, path)
    print(f"   ↳ {path}")

    # Echo the text table.
    path_txt = os.path.join(args.output_dir, "boundary_summary.txt")
    print()
    with open(path_txt) as f:
        print(f.read())

    print(">> ✅ Boundary-layer exclusion check complete.")


if __name__ == "__main__":
    main()
