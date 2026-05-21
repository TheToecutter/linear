"""
Alignment diagnostic: does shared structure exist in cross-seed R matrices?

The pairwise alignment scripts (`alignment_check.py` for embedding-space,
`alignment_check_activations.py` for per-layer activation-space) both
reported that cross-seed R matrices, after the best alignment we could
find, were at the same Frobenius distance as random orthogonal matrices,
with top-10 principal angles at ~85°.

This diagnostic resolves two competing explanations:

  (1) Cross-seed R matrices share no structure. The top principal
      directions are essentially random across seeds, even after best-
      possible alignment.

  (2) The top few directions are shared, but the trailing directions
      are seed-specific noise that swamps any signal at the full-R
      level. The aggregate Frobenius distance would still look like
      random in this case because trailing rows dominate by count.

These produce different diagnostic signatures:

  Test A: subspace alignment vs K. For each (seed pair, layer), sweep
    the subspace dimensionality K and measure principal angles between
    the top-K rows of the *activation-aligned* R_A and the top-K rows of
    R_B. Hypothesis (1): angles stay at ~85° for all K. Hypothesis (2):
    angles are small for small K and rise toward 85° as K grows.

  Test B: within-seed self-consistency. Re-run the SVD-based R
    recovery on *halves* of the same seed's pilot set. Compare the two
    halves' R matrices. This gives the within-seed noise floor — how
    much R varies just from sampling noise on the same model.
    Cross-seed angles must be interpreted relative to this floor; if
    within-seed angles are also large, the analyzer itself is the
    issue.

  Test C: top-1 direction angle. The cleanest single number per
    (pair, layer): the angle between R_A's top-1 direction (after
    alignment) and R_B's top-1 direction. If even the dominant
    direction is at ~85°, no shared structure exists at any level.

The diagnostic reuses the cached activations from
alignment_check_activations.py (aligned_activations.npy in each
run_dir) and the cached R matrices from flow_analysis/. No GPU or
inference needed.

Usage:
    python3 align_topk_diagnostic.py \\
        --run_dirs ../phase1_runs/seed_0 ../phase1_runs/seed_1 \\
                   ../phase1_runs/seed_2 ../phase1_runs/seed_3 \\
        --output_dir ../phase1_runs/align_topk_diag

Outputs (relative to --output_dir):
    topk_diagnostic_summary.txt   — narrative summary with the verdicts
    topk_diagnostic_summary.csv   — per-pair, per-K, per-layer numbers
    angle_vs_k.png                — Test A: subspace angles vs K, aggregated over layers/pairs
    angle_vs_k_per_layer.png      — Test A: same, faceted by layer
    within_vs_cross_angles.png    — Test B: within-seed vs cross-seed comparison
    top1_direction_angles.png     — Test C: top-1 angle per layer, all pairs
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_series import load_flow_series
from analyze import load_flow
from align import (
    align_activations_per_layer, transport_R_per_layer,
)


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
# Per-seed data loading (mirror of alignment_check_activations.py).
# ----------------------------------------------------------------------
ACTIVATION_CACHE_NAME = "aligned_activations.npy"


def get_seed_activations(run_dir: str) -> np.ndarray:
    """Load cached activations from run_dir/aligned_activations.npy.
    Raises if the cache is missing — this diagnostic doesn't trigger
    a fresh inference run; it just reads what alignment_check_activations
    already produced."""
    cache_path = os.path.join(run_dir, ACTIVATION_CACHE_NAME)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"No cached activations at {cache_path}. Run "
            f"alignment_check_activations.py first to populate the cache."
        )
    return np.load(cache_path)


def get_final_R(run_dir: str) -> np.ndarray:
    """Load only the final-checkpoint R matrix."""
    flow_dir = os.path.join(run_dir, "flow_analysis")
    flow_files = sorted([f for f in os.listdir(flow_dir)
                         if f.startswith("flow_step_") and f.endswith(".npz")])
    final_flow = load_flow(os.path.join(flow_dir, flow_files[-1]))
    return final_flow["R"]


class SeedBundle:
    """Mirror of alignment_check_activations.SeedBundle, simpler."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.fs = load_flow_series(run_dir)
        self.label = f"seed {self.fs.seed}"
        self.activations = get_seed_activations(run_dir)  # (L, N, H)
        self.R = get_final_R(run_dir)                     # (L, H, H)
        self.L = self.activations.shape[0]
        self.N = self.activations.shape[1]
        self.H = self.activations.shape[2]
        assert self.R.shape == (self.L, self.H, self.H), (
            f"R shape mismatch: R is {self.R.shape}, "
            f"expected ({self.L}, {self.H}, {self.H})"
        )


# ----------------------------------------------------------------------
# Subspace-angle helper.
# ----------------------------------------------------------------------
def principal_angles_top_k(A: np.ndarray, B: np.ndarray, k: int) -> np.ndarray:
    """
    Principal angles (in degrees) between the row-subspace of A[:k] and
    the row-subspace of B[:k]. Both A and B are (H, H) with orthonormal
    rows (as produced by SVD). Returns (k,) angles in [0°, 90°].

    For k=1: returns (1,) array with the single angle between A[0] and B[0]
    in degrees.
    """
    assert A.shape == B.shape, f"Shape mismatch: {A.shape} vs {B.shape}"
    A_top = A[:k]
    B_top = B[:k]
    M = A_top @ B_top.T  # (k, k)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.degrees(np.arccos(s))


def mean_subspace_angle(A: np.ndarray, B: np.ndarray, k: int) -> float:
    """Mean of the k principal angles between A[:k] and B[:k]."""
    return float(principal_angles_top_k(A, B, k).mean())


# ----------------------------------------------------------------------
# Test A: per-K subspace angles between two R stacks.
# ----------------------------------------------------------------------
def angles_vs_k_per_layer(
    R_A_transported: np.ndarray, R_B: np.ndarray,
    k_values: List[int],
) -> np.ndarray:
    """
    For each layer t and each K in k_values, compute the mean principal
    angle between R_A_transported[t, :K] and R_B[t, :K].

    Returns (L, len(k_values)) array of mean angles in degrees.
    """
    L = R_A_transported.shape[0]
    out = np.zeros((L, len(k_values)), dtype=np.float32)
    for t in range(L):
        for ki, k in enumerate(k_values):
            out[t, ki] = mean_subspace_angle(R_A_transported[t], R_B[t], k)
    return out


# ----------------------------------------------------------------------
# Test B: within-seed self-consistency.
# ----------------------------------------------------------------------
def split_half_R(
    activations: np.ndarray, rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given (L, N, H) activations, split the N pilots into two random
    halves and compute R (via centered SVD) on each half independently.

    The split is deterministic given rng_seed.

    Returns (R_half1, R_half2), each (L, H, H) with orthonormal rows.
    """
    L, N, H = activations.shape
    rng = np.random.default_rng(rng_seed)
    perm = rng.permutation(N)
    half = N // 2
    idx1 = perm[:half]
    idx2 = perm[half:half * 2]  # same size, avoid odd N edge case

    R1 = np.zeros((L, H, H), dtype=np.float32)
    R2 = np.zeros((L, H, H), dtype=np.float32)
    for t in range(L):
        x1 = activations[t, idx1]
        x2 = activations[t, idx2]
        x1c = x1 - x1.mean(axis=0)
        x2c = x2 - x2.mean(axis=0)
        # We only need Vt = R.
        _, _, vt1 = np.linalg.svd(x1c, full_matrices=False)
        _, _, vt2 = np.linalg.svd(x2c, full_matrices=False)
        # vt has shape (min(N_half, H), H); pad to (H, H) with zeros if needed.
        # In our regime N_half >> H so vt is exactly (H, H).
        K = vt1.shape[0]
        R1[t, :K] = vt1
        R2[t, :K] = vt2
    return R1, R2


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_angle_vs_k_aggregated(
    bundles: List[SeedBundle],
    angle_curves: Dict[Tuple[int, int], np.ndarray],
    within_curves: Dict[int, np.ndarray],
    k_values: List[int],
    output_path: str,
):
    """One panel: mean (over layers) angle vs K, one line per ordered
    cross-seed pair, plus the within-seed baseline as a dashed thick line.
    Also a random-baseline guide.

    angle_curves: {(i, j): (L, n_k) array of mean angles}
    within_curves: {seed_idx: (L, n_k) array of within-seed mean angles}
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    pair_colors = SEED_COLORS * 2

    # Cross-seed: one line per pair, mean over layers.
    for k, ((i, j), curves) in enumerate(angle_curves.items()):
        mean_over_layers = curves.mean(axis=0)
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        ax.plot(k_values, mean_over_layers, marker="o", markersize=3, lw=1.0,
                color=col, label=label, alpha=0.6)

    # Within-seed: thick dashed lines.
    for seed_idx, curves in within_curves.items():
        mean_over_layers = curves.mean(axis=0)
        ax.plot(
            k_values, mean_over_layers, marker="s", markersize=5, lw=2.2,
            color="black", ls="--", alpha=0.45,
            label=f"{bundles[seed_idx].label} (within-seed split-half)"
                if seed_idx == sorted(within_curves.keys())[0] else None,
        )

    # Random baseline: mean principal angle between independent random
    # orthonormal subspaces of dimension K in ambient H.
    # For K << H: approximately 90° - O(sqrt(K/H)).
    # For K = H: exactly 0° (they span the full space).
    # The exact expectation under Haar measure is a known formula but
    # we plot it numerically for clarity.
    H = bundles[0].H
    rng = np.random.default_rng(42)
    rand_ang = []
    for k in k_values:
        # Two random orthonormal k-subspaces in H-dim space.
        Q1, _ = np.linalg.qr(rng.standard_normal((H, k)))
        Q2, _ = np.linalg.qr(rng.standard_normal((H, k)))
        ang = principal_angles_top_k(Q1.T, Q2.T, k)
        rand_ang.append(float(ang.mean()))
    ax.plot(
        k_values, rand_ang, marker="^", markersize=5, lw=1.5,
        color="gray", ls=":", alpha=0.7,
        label="Random orthonormal K-subspaces",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Subspace dimensionality K (top-K rows of R)")
    ax.set_ylabel("Mean principal angle of top-K rows (degrees)")
    ax.axhline(90, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylim(0, 95)
    ax.set_title(
        "Subspace alignment of R matrices vs K\n"
        "(after activation-space per-layer alignment, mean over layers)"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_angle_vs_k_per_layer(
    bundles: List[SeedBundle],
    angle_curves: Dict[Tuple[int, int], np.ndarray],
    k_values: List[int],
    output_path: str,
):
    """Faceted version: one subplot per layer. Shows whether different
    layers have different alignment behavior."""
    L = bundles[0].L
    n_cols = 4
    n_rows = (L + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows),
                             sharex=True, sharey=True, squeeze=False)
    pair_colors = SEED_COLORS * 2

    for t in range(L):
        ax = axes[t // n_cols, t % n_cols]
        for (i, j), curves in angle_curves.items():
            color_idx = (i * len(bundles) + j) % len(pair_colors)
            col = pair_colors[color_idx]
            ax.plot(k_values, curves[t], marker="o", markersize=2, lw=0.9,
                    color=col, alpha=0.7)
        ax.set_xscale("log")
        ax.set_title(f"layer {t}", fontsize=10)
        ax.axhline(90, color="gray", ls=":", lw=0.5, alpha=0.4)
        ax.set_ylim(0, 95)

    # Hide unused.
    for t in range(L, n_rows * n_cols):
        axes[t // n_cols, t % n_cols].set_visible(False)

    for col in range(n_cols):
        axes[-1, col].set_xlabel("K")
    for row in range(n_rows):
        axes[row, 0].set_ylabel("angle (°)")

    fig.suptitle(
        "Subspace alignment angle vs K, per layer (after activation-space alignment)",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_within_vs_cross_angles(
    bundles: List[SeedBundle],
    cross_angles_topk: Dict[Tuple[int, int], np.ndarray],
    within_angles_topk: Dict[int, np.ndarray],
    k_for_plot: int,
    output_path: str,
):
    """Per-layer top-K subspace angle, comparing within-seed
    self-consistency to cross-seed alignment. One panel; cross-seed
    pairs as thin colored lines, within-seed as thick black-dashed lines.

    cross_angles_topk: {(i, j): (L,) angle at the chosen K}
    within_angles_topk: {seed_idx: (L,) angle at the chosen K}
    """
    L = bundles[0].L
    layer_indices = np.arange(L)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    pair_colors = SEED_COLORS * 2

    for k, ((i, j), angs) in enumerate(cross_angles_topk.items()):
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        ax.plot(layer_indices, angs, marker="o", markersize=3, lw=1.0,
                color=col, alpha=0.6, label=label)

    for seed_idx, angs in within_angles_topk.items():
        ax.plot(layer_indices, angs, marker="s", markersize=5, lw=2.0,
                color="black", ls="--", alpha=0.55,
                label=f"{bundles[seed_idx].label} (split-half within-seed)"
                    if seed_idx == sorted(within_angles_topk.keys())[0] else None)

    ax.set_xlabel("layer state t")
    ax.set_ylabel(f"Mean principal angle of top-{k_for_plot} subspaces (degrees)")
    ax.axhline(90, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylim(0, 95)
    ax.set_title(
        f"Within-seed vs cross-seed subspace alignment, top-{k_for_plot} subspace\n"
        f"(within-seed = noise floor from finite-sample SVD;\n"
        f"cross-seed = after best activation-space alignment)"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_top1_angles(
    bundles: List[SeedBundle],
    cross_top1_angles: Dict[Tuple[int, int], np.ndarray],
    within_top1_angles: Dict[int, np.ndarray],
    output_path: str,
):
    """Per-layer angle between top-1 directions only. The cleanest
    single number for the diagnostic."""
    L = bundles[0].L
    layer_indices = np.arange(L)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    pair_colors = SEED_COLORS * 2

    for k, ((i, j), angs) in enumerate(cross_top1_angles.items()):
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        ax.plot(layer_indices, angs, marker="o", markersize=4, lw=1.2,
                color=col, alpha=0.7, label=label)

    for seed_idx, angs in within_top1_angles.items():
        ax.plot(layer_indices, angs, marker="s", markersize=5, lw=2.0,
                color="black", ls="--", alpha=0.55,
                label=f"{bundles[seed_idx].label} (split-half within-seed)"
                    if seed_idx == sorted(within_top1_angles.keys())[0] else None)

    ax.set_xlabel("layer state t")
    ax.set_ylabel("Angle between top-1 directions (degrees)")
    ax.axhline(90, color="gray", ls=":", lw=0.7, alpha=0.5)
    ax.set_ylim(0, 95)
    ax.set_title(
        "Top-1 principal direction angle, per layer\n"
        "(0° = same direction up to sign; 90° = orthogonal)"
    )
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------
def write_summary(
    bundles: List[SeedBundle],
    cross_angles_vs_k: Dict[Tuple[int, int], np.ndarray],
    within_angles_vs_k: Dict[int, np.ndarray],
    k_values: List[int],
    cross_top1: Dict[Tuple[int, int], np.ndarray],
    within_top1: Dict[int, np.ndarray],
    output_path: str,
):
    """Pretty narrative summary."""
    n = len(bundles)
    pairs = list(cross_angles_vs_k.keys())
    L = bundles[0].L
    H = bundles[0].H

    with open(output_path, "w") as f:
        title = (f"Top-K alignment diagnostic ({n} seeds: "
                 f"{', '.join(b.label for b in bundles)})")
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(
            "Test A: subspace angle between top-K rows of R, vs K.\n"
            "Test B: within-seed split-half R consistency (noise floor).\n"
            "Test C: top-1 direction angle per layer.\n\n"
            f"H = {H}; L = {L}; N (pilots per layer) = {bundles[0].N}\n\n"
        )

        # Test A summary: mean angle across (layers, pairs) at each K,
        # vs mean within-seed angle at each K.
        f.write("=== Test A: subspace angle vs K ===\n\n")
        f.write(
            "Mean angle in degrees, averaged across all layers and all\n"
            "ordered cross-seed pairs. Compare to the within-seed split-half\n"
            "baseline (averaged across seeds) which gives the analyzer's\n"
            "noise floor at each K.\n\n"
        )
        # Means.
        n_pairs = len(pairs)
        n_seeds = len(within_angles_vs_k)
        cross_stack = np.stack(
            [cross_angles_vs_k[p] for p in pairs], axis=0,
        )  # (n_pairs, L, n_k)
        within_stack = np.stack(
            [within_angles_vs_k[s] for s in within_angles_vs_k], axis=0,
        )  # (n_seeds, L, n_k)
        cross_mean = cross_stack.mean(axis=(0, 1))      # (n_k,)
        within_mean = within_stack.mean(axis=(0, 1))    # (n_k,)
        gap = cross_mean - within_mean
        f.write(
            f"   {'K':>6}  {'cross-seed':>12}  {'within-seed':>12}  "
            f"{'gap':>10}  {'random':>10}\n"
        )
        f.write("   " + "-" * 60 + "\n")
        # Compute random baseline.
        rng = np.random.default_rng(42)
        rand_at_k = []
        for k in k_values:
            Q1, _ = np.linalg.qr(rng.standard_normal((H, k)))
            Q2, _ = np.linalg.qr(rng.standard_normal((H, k)))
            ang = principal_angles_top_k(Q1.T, Q2.T, k)
            rand_at_k.append(float(ang.mean()))
        for ki, k in enumerate(k_values):
            f.write(
                f"   {k:>6}  {cross_mean[ki]:>12.2f}  "
                f"{within_mean[ki]:>12.2f}  {gap[ki]:>+10.2f}  "
                f"{rand_at_k[ki]:>10.2f}\n"
            )
        f.write("\n")
        # Interpretation.
        # Look at K=1 (top-1) — does cross-seed beat random by more than
        # it beats within-seed?
        cross_k1 = cross_mean[0]
        within_k1 = within_mean[0]
        random_k1 = rand_at_k[0]
        if cross_k1 < within_k1 + 5.0:
            f.write(
                "   Interpretation: cross-seed top-1 angle is within 5° of\n"
                "   the within-seed self-consistency floor — the cross-seed\n"
                "   alignment is doing as well as can be expected given\n"
                "   finite-sample noise in the SVD. ✅ Shared structure exists.\n"
            )
        elif cross_k1 < random_k1 - 10.0:
            f.write(
                "   Interpretation: cross-seed top-1 angle is meaningfully\n"
                "   below the random baseline but well above the within-seed\n"
                "   floor. There is *some* shared top-direction structure,\n"
                "   but it's degraded relative to the finite-sample noise\n"
                "   floor. Partial shared structure.\n"
            )
        else:
            f.write(
                "   Interpretation: cross-seed top-1 angle is close to the\n"
                "   random baseline. ⚠️  No shared top-direction structure\n"
                "   even at K=1 — alignment cannot recover meaningful\n"
                "   correspondence between seeds' principal axes.\n"
            )
        f.write("\n")

        # Test B summary: per-layer within-seed top-1 vs top-10 angles.
        f.write("=== Test B: within-seed R reproducibility ===\n\n")
        f.write(
            "Mean per-layer angle when splitting one seed's pilots into two\n"
            "halves and recomputing R on each half. Small angles here mean\n"
            "the analyzer is sample-stable; large angles mean R itself\n"
            "fluctuates from sampling noise.\n\n"
        )
        f.write(f"   {'seed':>10}  {'top-1':>10}  {'top-10':>10}  {'full':>10}\n")
        f.write("   " + "-" * 50 + "\n")
        # Pick K indices that match common values.
        k_top1_idx = k_values.index(1) if 1 in k_values else 0
        k_top10_idx = (
            k_values.index(10) if 10 in k_values
            else min(range(len(k_values)), key=lambda i: abs(k_values[i] - 10))
        )
        k_full_idx = len(k_values) - 1
        for seed_idx, angs_vs_k in within_angles_vs_k.items():
            # Per-layer mean → mean over layers per K.
            per_k = angs_vs_k.mean(axis=0)  # (n_k,)
            f.write(
                f"   {bundles[seed_idx].label:>10}  "
                f"{per_k[k_top1_idx]:>10.2f}  "
                f"{per_k[k_top10_idx]:>10.2f}  "
                f"{per_k[k_full_idx]:>10.2f}\n"
            )
        f.write("\n")

        # Test C: top-1 directions cross-seed, per layer.
        f.write("=== Test C: top-1 direction angles, per layer ===\n\n")
        f.write(
            "Each row = layer t. Each col = mean across all cross-seed\n"
            "ordered pairs, vs the mean within-seed split-half value\n"
            "at the same layer.\n\n"
        )
        f.write(f"   {'layer':>6}  {'cross-seed':>12}  {'within-seed':>12}  "
                f"{'gap':>10}\n")
        f.write("   " + "-" * 50 + "\n")
        cross_top1_mean = np.stack([cross_top1[p] for p in pairs], axis=0).mean(axis=0)
        within_top1_mean = np.stack(
            [within_top1[s] for s in within_top1], axis=0,
        ).mean(axis=0)
        for t in range(L):
            f.write(
                f"   {t:>6}  {cross_top1_mean[t]:>12.2f}  "
                f"{within_top1_mean[t]:>12.2f}  "
                f"{cross_top1_mean[t] - within_top1_mean[t]:>+10.2f}\n"
            )
        f.write("\n")

        # Final verdict.
        f.write("=== Verdict ===\n\n")
        # Three regimes to distinguish:
        gap_at_k1 = cross_mean[0] - within_mean[0]
        if cross_mean[0] < 30.0 and gap_at_k1 < 5.0:
            f.write(
                "  ✅ Cross-seed R matrices DO share top-direction structure.\n"
                "  Top-1 directions align to within ~5° of the within-seed\n"
                "  self-consistency floor. The earlier 85°-angles result\n"
                "  was an artifact of looking at the full R matrix instead\n"
                "  of just the top directions. Phase 2 can compare top-K\n"
                "  subspaces; full-R comparison is unsuitable.\n"
            )
        elif cross_mean[0] < random_k1 - 15.0:
            f.write(
                "  🟡 Cross-seed R matrices share PARTIAL top-direction\n"
                "  structure. Cross-seed top-1 angles are below random but\n"
                "  meaningfully above the within-seed floor. Phase 2 can\n"
                "  use top-K subspace comparisons with caution; some\n"
                "  cross-variant signal is recoverable but it will be\n"
                "  mixed with non-trivial cross-seed noise.\n"
            )
        else:
            f.write(
                "  ⚠️  Cross-seed R matrices share NO recoverable structure,\n"
                "  even in the top-1 direction. Different seeds learn\n"
                "  functionally equivalent models (same eval loss) but\n"
                "  organize their hidden representations along seed-specific\n"
                "  bases that don't translate via any orthogonal map.\n"
                "  Phase 2 must rely entirely on basis-invariant statistics\n"
                "  (λ, log α, effective rank, kurtosis); R-matrix comparison\n"
                "  is unworkable.\n"
            )


def write_summary_csv(
    bundles: List[SeedBundle],
    cross_angles_vs_k: Dict[Tuple[int, int], np.ndarray],
    within_angles_vs_k: Dict[int, np.ndarray],
    k_values: List[int],
    output_path: str,
):
    """Per-(pair, layer, K) numbers in CSV form for downstream plotting."""
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "kind", "seed_A", "seed_B", "layer", "k", "mean_angle_deg",
        ])
        for (i, j), curves in cross_angles_vs_k.items():
            for t in range(curves.shape[0]):
                for ki, k in enumerate(k_values):
                    w.writerow([
                        "cross", bundles[i].label, bundles[j].label,
                        t, k, float(curves[t, ki]),
                    ])
        for seed_idx, curves in within_angles_vs_k.items():
            for t in range(curves.shape[0]):
                for ki, k in enumerate(k_values):
                    w.writerow([
                        "within", bundles[seed_idx].label, "",
                        t, k, float(curves[t, ki]),
                    ])


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Top-K rows + within-seed diagnostic for the "
                    "alignment-failure question.",
    )
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Default: align_topk_diag/ alongside first run_dir's parent.",
    )
    parser.add_argument(
        "--k_values", type=int, nargs="+",
        default=[1, 2, 5, 10, 20, 50, 100, 200, 500, 896],
        help="Subspace dimensionalities K to sweep. Will be clipped to "
             "[1, H]. Default includes K=1 (top-1) through K=H (full).",
    )
    parser.add_argument(
        "--k_topk_plot", type=int, default=10,
        help="Which K to highlight in the within-vs-cross per-layer plot. "
             "Default 10 (matches the principal_angle_profile default).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        first_parent = os.path.dirname(os.path.abspath(args.run_dirs[0]))
        args.output_dir = os.path.join(first_parent, "align_topk_diag")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load bundles.
    print(f">> Loading {len(args.run_dirs)} seed(s) ...")
    bundles = []
    for run_dir in args.run_dirs:
        b = SeedBundle(run_dir)
        bundles.append(b)
        print(f"   {b.label}: activations {b.activations.shape}, R {b.R.shape}")
    bundles.sort(key=lambda b: b.fs.seed)

    H = bundles[0].H
    # Clip K values to [1, H].
    k_values = sorted(set(max(1, min(H, k)) for k in args.k_values))
    print(f"\n>> Sweeping K through {k_values}")
    # Make sure k_topk_plot is in k_values; if not, add it.
    if args.k_topk_plot not in k_values:
        k_values = sorted(k_values + [max(1, min(H, args.k_topk_plot))])

    # Equalize pilot counts.
    N_min = min(b.N for b in bundles)
    for b in bundles:
        if b.N != N_min:
            b.activations = b.activations[:, :N_min, :]
            b.N = N_min
    print(f"   N (pilots per layer): {N_min}")

    # Compute cross-seed angles after activation-space alignment.
    print(f"\n>> Computing per-layer activation alignment and "
          f"cross-seed angles vs K ({len(bundles) * (len(bundles)-1)} pairs) ...")
    cross_angles_vs_k = {}
    cross_top1 = {}
    t_start = time.time()
    for i, b_A in enumerate(bundles):
        for j, b_B in enumerate(bundles):
            if i == j:
                continue
            Qs, _ = align_activations_per_layer(
                b_A.activations, b_B.activations,
            )
            R_A_aligned = transport_R_per_layer(b_A.R, Qs)
            angs = angles_vs_k_per_layer(R_A_aligned, b_B.R, k_values)
            cross_angles_vs_k[(i, j)] = angs
            cross_top1[(i, j)] = angs[:, k_values.index(1)]
            print(f"   {b_A.label}→{b_B.label}: top-1 mean angle = "
                  f"{cross_top1[(i, j)].mean():.1f}°, "
                  f"K={H} mean angle = {angs[:, -1].mean():.1f}°")
    print(f"   Cross-seed sweep done in {time.time() - t_start:.1f}s")

    # Compute within-seed self-consistency angles.
    print(f"\n>> Computing within-seed split-half angles for "
          f"{len(bundles)} seeds ...")
    within_angles_vs_k = {}
    within_top1 = {}
    t_start = time.time()
    for seed_idx, b in enumerate(bundles):
        R1, R2 = split_half_R(b.activations, rng_seed=42 + seed_idx)
        # Compare R1 to R2 — both built from the same seed, just different halves.
        # No alignment needed (same coordinate system).
        angs = angles_vs_k_per_layer(R1, R2, k_values)
        within_angles_vs_k[seed_idx] = angs
        within_top1[seed_idx] = angs[:, k_values.index(1)]
        print(f"   {b.label}: top-1 mean within-seed angle = "
              f"{within_top1[seed_idx].mean():.1f}°, "
              f"K={H} mean within-seed angle = {angs[:, -1].mean():.1f}°")
    print(f"   Within-seed analysis done in {time.time() - t_start:.1f}s")

    # Write outputs.
    print(f"\n>> Writing outputs to {args.output_dir} ...")
    txt = os.path.join(args.output_dir, "topk_diagnostic_summary.txt")
    write_summary(bundles, cross_angles_vs_k, within_angles_vs_k,
                  k_values, cross_top1, within_top1, txt)
    print(f"   ↳ {txt}")
    csv_path = os.path.join(args.output_dir, "topk_diagnostic_summary.csv")
    write_summary_csv(bundles, cross_angles_vs_k, within_angles_vs_k,
                      k_values, csv_path)
    print(f"   ↳ {csv_path}")
    p1 = os.path.join(args.output_dir, "angle_vs_k.png")
    plot_angle_vs_k_aggregated(bundles, cross_angles_vs_k, within_angles_vs_k,
                               k_values, p1)
    print(f"   ↳ {p1}")
    p2 = os.path.join(args.output_dir, "angle_vs_k_per_layer.png")
    plot_angle_vs_k_per_layer(bundles, cross_angles_vs_k, k_values, p2)
    print(f"   ↳ {p2}")
    # Slice at k_topk_plot for the within-vs-cross plot.
    k_idx = k_values.index(args.k_topk_plot)
    cross_at_k = {
        p: cross_angles_vs_k[p][:, k_idx] for p in cross_angles_vs_k
    }
    within_at_k = {
        s: within_angles_vs_k[s][:, k_idx] for s in within_angles_vs_k
    }
    p3 = os.path.join(args.output_dir, "within_vs_cross_angles.png")
    plot_within_vs_cross_angles(bundles, cross_at_k, within_at_k,
                                args.k_topk_plot, p3)
    print(f"   ↳ {p3}")
    p4 = os.path.join(args.output_dir, "top1_direction_angles.png")
    plot_top1_angles(bundles, cross_top1, within_top1, p4)
    print(f"   ↳ {p4}")

    # Echo summary.
    print()
    with open(txt) as f:
        print(f.read())

    print(">> ✅ Top-K diagnostic complete.")


if __name__ == "__main__":
    main()
