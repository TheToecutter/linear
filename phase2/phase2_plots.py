"""
Phase 2 plots: attribution heatmap and Tier 1b cross-input-distribution
comparison.

Produces two figures from the same on-disk data the text/CSV outputs
of phase2_attribution.py consume:

  - phase2_attribution_heatmap.png
        Rows = statistics, cols = design axes, cells colored by
        classification (robust / controls↑ / controls↓ / non-monotonic).
        Each cell annotated with the maximum |Δ| over its variants.

  - phase2_tier1b_bars.png
        For each statistic in the baseline summary, a grouped bar
        chart of the statistic value under real / shuffled / random
        input distributions. Read out the FFN-vs-attention
        decomposition by eye.

Both plots are read-only over disk -- they consume the
flow_analysis/ and flow_analysis_<dist>/ outputs from
phase2_analyze.py. They do nothing if no Phase 2 data is present
yet; in that case they print a polite message and exit 0.

Usage
-----
    python3 phase2_plots.py --out_dir phase2_plots/
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from phase2_attribution import (
    build_attribution_matrix, build_tier1b_table,
    ROBUST, CONTROLS_UP, CONTROLS_DOWN, NON_MONOTONIC, INSUFFICIENT,
    AttributionCell,
)


def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,           # heatmap doesn't want gridlines
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
    })


_setup_style()


# ----------------------------------------------------------------------
# Color scheme for classification labels.
# Choose discrete categorical colors (not a perceptual gradient) since
# the classification has no natural ordering.
#
# Selected from the matplotlib tab10 palette for distinctness:
#   ROBUST         → gray
#   CONTROLS↑      → blue (cool, "above")
#   CONTROLS↓      → orange (warm, "below")
#   NON_MONOTONIC  → purple (a third axis: not just up/down)
#   INSUFFICIENT   → very pale gray (data-absent rather than data-saying)
# ----------------------------------------------------------------------
CLASS_TO_INT = {
    ROBUST: 0,
    CONTROLS_UP: 1,
    CONTROLS_DOWN: 2,
    NON_MONOTONIC: 3,
    INSUFFICIENT: 4,
}

CLASS_COLORS = [
    "#9aa0a6",  # ROBUST  — neutral gray
    "#4e79a7",  # CONTROLS_UP — blue
    "#f28e2b",  # CONTROLS_DOWN — orange
    "#76448a",  # NON_MONOTONIC — purple
    "#e6e6e6",  # INSUFFICIENT — very pale gray
]

# Label text drawn inside each cell.
CLASS_GLYPH = {
    ROBUST: "•",
    CONTROLS_UP: "↑",
    CONTROLS_DOWN: "↓",
    NON_MONOTONIC: "?",
    INSUFFICIENT: "—",
}


# ----------------------------------------------------------------------
# Attribution heatmap.
# ----------------------------------------------------------------------
def _cells_to_matrix(cells: List[AttributionCell]):
    """Pivot the flat list of cells into a (statistic × axis) matrix
    plus row/column labels and per-cell max |Δ| annotations."""
    statistics: List[str] = []
    axes_list: List[str] = []
    for c in cells:
        if c.statistic not in statistics:
            statistics.append(c.statistic)
        if c.axis not in axes_list:
            axes_list.append(c.axis)
    # Numeric matrix of classifications.
    M = np.full((len(statistics), len(axes_list)), CLASS_TO_INT[INSUFFICIENT],
                dtype=np.int8)
    # Per-cell maximum |delta| / threshold (None if not measurable).
    annot = np.full((len(statistics), len(axes_list)), np.nan, dtype=np.float64)
    glyph = np.full((len(statistics), len(axes_list)), "", dtype=object)
    cell_lookup = {(c.statistic, c.axis): c for c in cells}
    for i, s in enumerate(statistics):
        for j, a in enumerate(axes_list):
            c = cell_lookup.get((s, a))
            if c is None:
                continue
            M[i, j] = CLASS_TO_INT.get(c.classification,
                                        CLASS_TO_INT[INSUFFICIENT])
            glyph[i, j] = CLASS_GLYPH.get(c.classification, "")
            if c.deltas and not np.isnan(c.baseline_threshold) \
                    and c.baseline_threshold > 0:
                max_abs_d = max(abs(d) for d in c.deltas)
                annot[i, j] = max_abs_d / c.baseline_threshold
    return statistics, axes_list, M, annot, glyph


def plot_attribution_heatmap(
    cells: List[AttributionCell],
    output_path: str,
    title: Optional[str] = None,
):
    """Render the attribution matrix as a categorical heatmap.

    Each cell is colored by classification and annotated with:
      - a glyph indicating the classification (•, ↑, ↓, ?, —),
      - the maximum |Δ| / threshold ratio across the variants on that axis,
        if measurable.
    """
    if not cells:
        print("plot_attribution_heatmap: no cells; skipping.")
        return

    statistics, axes_list, M, annot, glyph = _cells_to_matrix(cells)
    n_rows, n_cols = len(statistics), len(axes_list)

    fig, ax = plt.subplots(
        figsize=(max(5, n_cols * 1.6 + 3), max(3.5, n_rows * 0.6 + 1.5))
    )
    cmap = ListedColormap(CLASS_COLORS)
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=len(CLASS_COLORS) - 1)

    # Axis ticks and labels.
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(axes_list, rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(statistics)

    # Annotate cells.
    for i in range(n_rows):
        for j in range(n_cols):
            g = glyph[i, j]
            r = annot[i, j]
            if not np.isnan(r):
                label = f"{g}\n{r:.2g}×"
            else:
                label = g
            color = "white" if M[i, j] in (1, 2, 3) else "black"
            ax.text(
                j, i, label, ha="center", va="center",
                fontsize=10, color=color, linespacing=1.0,
            )

    # Light gridlines between cells.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)

    # Legend.
    legend_handles = [
        Patch(color=CLASS_COLORS[CLASS_TO_INT[ROBUST]],
              label=f"{CLASS_GLYPH[ROBUST]} robust"),
        Patch(color=CLASS_COLORS[CLASS_TO_INT[CONTROLS_UP]],
              label=f"{CLASS_GLYPH[CONTROLS_UP]} controls↑"),
        Patch(color=CLASS_COLORS[CLASS_TO_INT[CONTROLS_DOWN]],
              label=f"{CLASS_GLYPH[CONTROLS_DOWN]} controls↓"),
        Patch(color=CLASS_COLORS[CLASS_TO_INT[NON_MONOTONIC]],
              label=f"{CLASS_GLYPH[NON_MONOTONIC]} non-monotonic"),
        Patch(color=CLASS_COLORS[CLASS_TO_INT[INSUFFICIENT]],
              label=f"{CLASS_GLYPH[INSUFFICIENT]} insufficient data"),
    ]
    ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.,
    )

    ax.set_title(title or
                  "Phase 2 attribution matrix — design axis vs macro statistic")
    ax.set_xlabel("design axis")
    ax.set_ylabel("basis-invariant statistic")
    # Annotation reads "max |Δ| / threshold" — bigger = more dispositive
    # threshold crossing.
    fig.text(
        0.99, 0.01,
        "Cell annotation: max |Δ| / (1.5×std) across variants on that axis.\n"
        "> 1× = crosses the within-variant noise floor.",
        ha="right", va="bottom", fontsize=8, style="italic", color="#444",
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   ↳ Wrote {output_path}")


# ----------------------------------------------------------------------
# Tier 1b bar chart.
# ----------------------------------------------------------------------
def plot_tier1b_bars(
    rows: List[dict],
    output_path: str,
    title: Optional[str] = None,
):
    """Plot a grouped bar chart of each statistic's value under
    real/shuffled/random inputs.

    Bars show signed relative deviation from real:
        (value_under_dist - value_on_real) / |value_on_real|

    Reading:
      - Bar at 0 ⇒ statistic identical to real ⇒ input-invariant ⇒
        FFN-driven contribution to that statistic.
      - Bar far from 0 ⇒ statistic depends on input structure ⇒
        attention-driven contribution.
      - Sign-preserving: bar above 0 means "more positive than real";
        below 0 means "more negative than real" (regardless of whether
        the baseline value itself is positive or negative). For
        log α ≈ -3.7, a bar at -0.1 means shuffled gave a more negative
        log α (real = -3.7, shuffled ≈ -4.1).
    """
    if not rows:
        print("plot_tier1b_bars: no rows; skipping.")
        return

    # Pivot: statistic → {dist → (mean, std)}.
    by_stat: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for r in rows:
        s = r["statistic"]
        if s not in by_stat:
            by_stat[s] = {}
        by_stat[s][r["input_distribution"]] = (r["mean"], r["std"])

    # Drop statistics where 'real' is absent or zero (denominator).
    statistics = []
    for s in by_stat:
        if "real" in by_stat[s] and abs(by_stat[s]["real"][0]) > 1e-12:
            statistics.append(s)

    if not statistics:
        print("plot_tier1b_bars: no statistics with a real-input baseline; "
              "skipping.")
        return

    # Distributions to show (preserve input order, omit 'real' since it's
    # the reference at 0.0).
    dists_present = []
    for r in rows:
        d = r["input_distribution"]
        if d not in dists_present:
            dists_present.append(d)
    plot_dists = [d for d in dists_present if d != "real"]

    n_stats = len(statistics)
    n_groups = len(plot_dists)
    bar_width = 0.8 / max(1, n_groups)
    x = np.arange(n_stats)

    fig, ax = plt.subplots(figsize=(max(7, n_stats * 0.9), 4.5))
    palette = ["#4e79a7", "#f28e2b", "#76b7b2", "#e15759"]
    for k, dist in enumerate(plot_dists):
        rel_devs = []
        rel_errs = []
        for s in statistics:
            real_mean, _ = by_stat[s]["real"]
            denom = abs(real_mean)
            if dist not in by_stat[s]:
                rel_devs.append(np.nan)
                rel_errs.append(0.0)
                continue
            d_mean, d_std = by_stat[s][dist]
            rel_devs.append((d_mean - real_mean) / denom)
            rel_errs.append(abs(d_std / denom) if not np.isnan(d_std) else 0.0)
        ax.bar(
            x + (k - (n_groups - 1) / 2) * bar_width,
            rel_devs, bar_width, yerr=rel_errs, capsize=2,
            label=dist, color=palette[k % len(palette)],
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8,
               label="real (reference, value 0)")
    ax.set_xticks(x)
    ax.set_xticklabels(statistics, rotation=30, ha="right")
    ax.set_ylabel("(value − real) / |real|")
    ax.set_title(title or
                  "Phase 2 Tier 1b: macro statistics by input distribution")
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    # Place the explanatory caption inside the axes (top, away from the
    # rotated x-tick labels along the bottom).
    ax.text(
        0.01, 0.98,
        "Bars near 0 ⇒ statistic input-invariant ⇒ FFN-driven.\n"
        "Bars far from 0 ⇒ statistic input-sensitive ⇒ attention-driven.\n"
        "Sign-preserving: + = shifted positive vs real, − = shifted negative.",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=7.5, style="italic", color="#444",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
    )
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   ↳ Wrote {output_path}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Phase 2 plots.")
    parser.add_argument(
        "--out_dir", type=str, default="phase2_plots",
        help="Directory to write plots into. Default: phase2_plots/.",
    )
    parser.add_argument(
        "--skip_attribution", action="store_true",
        help="Skip the attribution-matrix heatmap.",
    )
    parser.add_argument(
        "--skip_tier1b", action="store_true",
        help="Skip the Tier 1b cross-input-distribution bar chart.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_attribution:
        try:
            cells = build_attribution_matrix()
        except RuntimeError as e:
            print(f"⚠️  Attribution data not available: {e}")
            cells = []
        if cells:
            plot_attribution_heatmap(
                cells,
                output_path=os.path.join(
                    args.out_dir, "phase2_attribution_heatmap.png",
                ),
            )

    if not args.skip_tier1b:
        rows = build_tier1b_table()
        if rows:
            plot_tier1b_bars(
                rows,
                output_path=os.path.join(
                    args.out_dir, "phase2_tier1b_bars.png",
                ),
            )


if __name__ == "__main__":
    main()
