"""
Cross-seed Procrustes alignment check — the Phase 1 → Phase 2 gate.

The proposal's vocabulary-anchored Procrustes alignment uses each
model's learned embedding matrix as a "Rosetta stone" — a known
correspondence (vocabulary token → embedding vector) that lets us
construct an orthogonal map between two models' hidden spaces. For
four seeds with identical architecture and tokenizer trained on the
same corpus, cross-seed alignment is the cleanest possible test of
whether this alignment procedure works at all. If it doesn't recover
small residuals here, it won't work for the much harder cross-variant
case in Phase 2.

**Important detail (added after diagnosing a v1 failure):** for a
small model trained on a modest amount of data, the bulk of the
vocabulary's embedding rows are *undertrained* and contribute pure
noise to the Procrustes residual. Empirically: with the full 32,768
Mistral tokens, ρ_E ≈ 0.60 (alignment fails); with the top 1000
tokens by L2 norm (a proxy for "this token was actively trained"),
ρ_E ≈ 0.06 (alignment works). The fix is to filter the anchor set
to the **intersection of each model's top-K tokens by row-norm**
before computing the Procrustes Q. The single Q is then applied to
all R matrices regardless of vocabulary.

This script:
  1. Extracts each seed's final-checkpoint embedding matrix from its
     model state dict (cached as a .npy file in each run_dir for
     reuse).
  2. Loads each seed's final-checkpoint R matrices.
  3. For each ordered seed pair (A, B):
     a. Computes the intersection of A's and B's top-K-by-norm token
        indices.
     b. Calls `align.align_embeddings` on the filtered E_A[idx] and
        E_B[idx] to get the orthogonal Q and residual ρ_E.
     c. Uses Q to transport A's R matrices into B's coordinate frame.
     d. Measures post-alignment Frobenius distance and top-10
        principal angles per layer.
  4. Compares to two baselines: identity (no alignment) and random
     orthogonal (worst-case meaningless alignment).
  5. Optionally sweeps K over a range of values and reports the
     ρ_E-vs-K curve — useful as a Phase 1 diagnostic showing the
     untrained-rare-token effect.

The single headline number is the **embedding-space residual ρ_E**
averaged across the cross-seed pairs at the chosen K. Small (≤ 0.10)
means alignment works; large (≥ 0.30) means it doesn't and Phase 2
needs to think harder before launching.

Usage:
    python3 alignment_check.py \\
        --run_dirs ../phase1_runs/seed_0 ../phase1_runs/seed_1 \\
                   ../phase1_runs/seed_2 ../phase1_runs/seed_3 \\
        --top_k_by_norm 1000 \\
        --rho_e_sweep \\
        --output_dir ../phase1_runs/alignment_check

The first invocation will extract embedding matrices from each seed's
final checkpoint (requires torch + model code to load the state dict).
Subsequent runs read the cached `final_embedding.npy` from each run_dir.

Outputs (relative to --output_dir):
    alignment_summary.txt      — per-pair ρ_E, R-distance, angles,
                                 baselines, and Phase-2 readiness verdict
    alignment_summary.csv      — same, machine-readable
    pairwise_rho_e.png         — heatmap of ρ_E across all seed pairs
    perlayer_angles.png        — per-layer principal angles, aligned
                                 vs baselines
    perlayer_R_distances.png   — per-layer R Frobenius distance, same
                                 layout
    rho_e_vs_k.png             — (if --rho_e_sweep) ρ_E as a function
                                 of K, the anchor-set size
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

from flow_series import load_flow_series
from analyze import load_flow
from align import (
    align_embeddings, transport_R_per_layer, orthogonal_procrustes,
)
from distances import aligned_R_distance, principal_angle_profile


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
# Per-seed data extraction.
# ----------------------------------------------------------------------
EMBEDDING_CACHE_NAME = "final_embedding.npy"


def get_final_embedding(run_dir: str, verbose: bool = True) -> np.ndarray:
    """
    Return the (V, H) embedding matrix from the seed's final checkpoint.

    On first call, loads the state dict from the final checkpoint, extracts
    the embedding weight, and caches it as `final_embedding.npy` in
    run_dir. Subsequent calls load from the cache.

    Cached extraction requires torch only on the first call; the cache
    is plain numpy and is read by future invocations without torch.
    """
    cache_path = os.path.join(run_dir, EMBEDDING_CACHE_NAME)
    if os.path.exists(cache_path):
        if verbose:
            print(f"   ↳ Loading cached embedding from {cache_path}")
        return np.load(cache_path)

    # No cache: extract from final checkpoint.
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(
            f"No checkpoints/ directory in {run_dir} — can't extract embedding."
        )
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"No .pt checkpoints in {ckpt_dir}.")
    # Final checkpoint = highest step number, which lex-sort gives if names
    # are zero-padded. Otherwise pick the one corresponding to the last
    # flow file (safer).
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    if verbose:
        print(f"   ↳ Extracting embedding from {ckpt_path}")

    # torch is imported here lazily so seeds with a cached embedding can
    # be processed without it.
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    # Strip any DDP/compile prefixes.
    candidate_keys = [k for k in sd.keys() if k.endswith("token_embed.weight")]
    if not candidate_keys:
        raise KeyError(
            f"Couldn't find 'token_embed.weight' in checkpoint state_dict. "
            f"Available keys (first 20): {list(sd.keys())[:20]}"
        )
    key = candidate_keys[0]
    E = sd[key].detach().cpu().float().numpy()
    if verbose:
        print(f"      ↳ Embedding shape: {E.shape} (key: {key})")

    # Cache.
    np.save(cache_path, E)
    if verbose:
        print(f"      ↳ Cached to {cache_path}")
    return E


def get_final_R(run_dir: str) -> Tuple[np.ndarray, int, int]:
    """
    Load only the final-checkpoint R matrix from a seed's flow files.
    Returns (R, step, L). Avoids the full FlowSeries.load_R() which would
    load all 50 checkpoints.
    """
    flow_dir = os.path.join(run_dir, "flow_analysis")
    flow_files = sorted([f for f in os.listdir(flow_dir)
                         if f.startswith("flow_step_") and f.endswith(".npz")])
    if not flow_files:
        raise FileNotFoundError(f"No flow files in {flow_dir}.")
    final_flow = load_flow(os.path.join(flow_dir, flow_files[-1]))
    return (
        final_flow["R"],
        int(final_flow["checkpoint_step"]),
        int(final_flow["num_layers_total"]),
    )


class SeedAlignBundle:
    """Holds the data needed for cross-seed alignment for one seed."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.fs = load_flow_series(run_dir)
        self.label = f"seed {self.fs.seed}"
        self.E = get_final_embedding(run_dir)               # (V, H)
        self.R, self.step, self.L = get_final_R(run_dir)    # (L, H, H), int, int

    def __repr__(self):
        return (f"SeedAlignBundle({self.label}, step={self.step}, "
                f"E={self.E.shape}, R={self.R.shape})")


# ----------------------------------------------------------------------
# Pairwise alignment computation.
# ----------------------------------------------------------------------
def random_orthogonal(H: int, seed: int = 0) -> np.ndarray:
    """Random orthogonal H×H matrix from a Haar-uniform distribution.
    Used as a worst-case alignment baseline."""
    rng = np.random.default_rng(seed)
    # QR of a random Gaussian matrix gives a Haar-uniform orthogonal matrix
    # (modulo sign correction, but for our purposes the sign doesn't matter).
    M = rng.standard_normal((H, H))
    Q, _ = np.linalg.qr(M)
    return Q.astype(np.float32)


def per_layer_R_distance(R_A: np.ndarray, R_B: np.ndarray) -> np.ndarray:
    """Per-layer Frobenius distance between R_A and R_B with no alignment.
    Returns (L,) array."""
    L = R_A.shape[0]
    out = np.zeros(L, dtype=np.float32)
    for t in range(L):
        out[t] = float(np.linalg.norm(R_A[t] - R_B[t]))
    return out


def top_k_by_norm_intersection(
    E_A: np.ndarray, E_B: np.ndarray, k: Optional[int],
) -> np.ndarray:
    """
    Indices of tokens that are in both A's and B's top-K-by-norm subsets.

    Per-row L2 norm is a proxy for "this token was actively trained":
    actively-trained embeddings grow in norm from their initialization
    scale, while undertrained embeddings stay near init. The intersection
    of each model's top-K subsets gives a symmetric anchor set
    (same for the (A,B) and (B,A) pairs) consisting of tokens that are
    well-trained in *both* models.

    Args:
        E_A, E_B: (V, H) embedding matrices.
        k: target subset size. If None (or k >= V), returns all indices
           (no filtering — equivalent to using the full vocabulary).

    Returns:
        idx: (n,) sorted int64 array of token indices to use as alignment
            anchors. n ≤ k by construction (typically n is close to k for
            similarly-trained models; for very-different models the
            intersection can be much smaller).
    """
    V = E_A.shape[0]
    if k is None or k >= V:
        return np.arange(V, dtype=np.int64)
    norms_A = np.linalg.norm(E_A, axis=1)
    norms_B = np.linalg.norm(E_B, axis=1)
    # argsort gives ascending order; take the top k by negating.
    top_A = set(np.argsort(-norms_A)[:k].tolist())
    top_B = set(np.argsort(-norms_B)[:k].tolist())
    inter = sorted(top_A & top_B)
    return np.array(inter, dtype=np.int64)


def compute_pairwise_alignment(
    bundle_A: SeedAlignBundle, bundle_B: SeedAlignBundle,
    top_k: Optional[int] = 1000,
    random_q_seed: int = 1234,
) -> Dict:
    """
    Run the full alignment + baseline comparison for one ordered pair (A, B).

    The Procrustes Q is computed from the intersection of A's and B's
    top-K-by-norm tokens (see `top_k_by_norm_intersection`). The
    resulting Q is then applied to the full R-matrix stack regardless
    of vocabulary, since Q is an orthogonal map in H-dim space, not
    vocabulary-dependent.

    Args:
        bundle_A, bundle_B: the two seeds to align.
        top_k: anchor-set size. None or ≥ V means use full vocabulary
            (the v1 behavior; equivalent to the proposal's literal
            spec). Default 1000 chosen because at K=1000 cross-seed
            ρ_E is ~0.06 on Phase 1 data — well within the
            "good alignment" range — and going higher just adds
            undertrained-token noise.

    Returns a dict with:
        rho_E: embedding-space alignment residual on the anchor subset
        Q: (H, H) orthogonal map from A's space to B's
        n_anchor_tokens: int — actual anchor-set size used
        R_aligned_dist_per_layer: (L,) Frobenius distance per layer, post-alignment
        R_identity_dist_per_layer: (L,) distance with no alignment (identity Q)
        R_random_dist_per_layer:   (L,) distance with random orthogonal Q
        angles_aligned: (L,) mean principal angle (top-10) per layer, post-alignment
        angles_identity: (L,) ditto, no alignment
        angles_random:   (L,) ditto, random alignment
    """
    anchor_idx = top_k_by_norm_intersection(bundle_A.E, bundle_B.E, top_k)
    E_A_anchor = bundle_A.E[anchor_idx]
    E_B_anchor = bundle_B.E[anchor_idx]
    Q, rho_E = align_embeddings(E_A_anchor, E_B_anchor)

    # Transport A's R matrices into B's coords using Q.
    # transport_R uses Q.T @ R @ Q (same Q at both input and output sides).
    L = bundle_A.L
    Qs = [Q] * L
    R_A_transported = transport_R_per_layer(bundle_A.R, Qs)

    # Aligned distance.
    R_aligned_dist = aligned_R_distance(R_A_transported, bundle_B.R, per_layer=True)
    angles_aligned = principal_angle_profile(R_A_transported, bundle_B.R, top_k=10)

    # Identity baseline: just compare raw R matrices (Q = I).
    R_identity_dist = per_layer_R_distance(bundle_A.R, bundle_B.R)
    angles_identity = principal_angle_profile(bundle_A.R, bundle_B.R, top_k=10)

    # Random-orthogonal baseline: transport A's R through a meaningless Q.
    H = bundle_A.E.shape[1]
    Q_rand = random_orthogonal(H, seed=random_q_seed)
    R_A_random = transport_R_per_layer(bundle_A.R, [Q_rand] * L)
    R_random_dist = aligned_R_distance(R_A_random, bundle_B.R, per_layer=True)
    angles_random = principal_angle_profile(R_A_random, bundle_B.R, top_k=10)

    return {
        "rho_E": rho_E,
        "Q": Q,
        "n_anchor_tokens": int(len(anchor_idx)),
        "R_aligned_dist_per_layer": R_aligned_dist,
        "R_identity_dist_per_layer": R_identity_dist,
        "R_random_dist_per_layer": R_random_dist,
        "angles_aligned": angles_aligned,
        "angles_identity": angles_identity,
        "angles_random": angles_random,
    }


def compute_rho_e_sweep(
    bundle_A: SeedAlignBundle, bundle_B: SeedAlignBundle,
    k_values: List[Optional[int]],
) -> Dict[Optional[int], Tuple[float, int]]:
    """
    Compute just ρ_E (no R-transport) for each K in k_values.

    Cheap: only does the SVD-based Procrustes on the filtered anchors.
    Used to diagnose how alignment quality depends on anchor-set size —
    the "alignment quality vs vocabulary subset" diagnostic from Phase 1.

    Returns: dict mapping K → (rho_E, n_anchor_tokens_used).
    """
    out = {}
    for k in k_values:
        anchor_idx = top_k_by_norm_intersection(bundle_A.E, bundle_B.E, k)
        E_A = bundle_A.E[anchor_idx]
        E_B = bundle_B.E[anchor_idx]
        _, rho = align_embeddings(E_A, E_B)
        out[k] = (rho, int(len(anchor_idx)))
    return out


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_rho_e_heatmap(bundles: List[SeedAlignBundle],
                       rho_e_matrix: np.ndarray,
                       output_path: str):
    """Heatmap of ρ_E values across all seed pairs. Diagonal is zero
    (self-alignment); off-diagonal cells show pairwise alignment quality."""
    n = len(bundles)
    labels = [b.label for b in bundles]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rho_e_matrix, cmap="viridis_r", vmin=0,
                   vmax=max(rho_e_matrix.max(), 1e-6))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            txt_color = "white" if rho_e_matrix[i, j] > rho_e_matrix.max() * 0.5 else "black"
            ax.text(j, i, f"{rho_e_matrix[i, j]:.3f}",
                    ha="center", va="center", color=txt_color, fontsize=9)
    fig.colorbar(im, ax=ax, label="ρ_E (embedding alignment residual)")
    ax.set_title("Pairwise embedding-space alignment residual (ρ_E)\n"
                 "(lower = better alignment; diagonal trivially 0)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rho_e_sweep(
    bundles: List[SeedAlignBundle],
    sweep_results: Dict[Tuple[int, int], Dict[Optional[int], Tuple[float, int]]],
    k_values: List[Optional[int]],
    chosen_k: Optional[int],
    output_path: str,
):
    """Plot ρ_E as a function of anchor-set size K, one line per
    ordered seed pair. Useful as the Phase 1 diagnostic showing the
    untrained-rare-token effect.

    k_values: list of K values used in the sweep. None = full vocab.
    chosen_k: the K used for the main pairwise analysis (highlighted
        with a vertical line).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Convert None to V (full vocab) for plotting purposes.
    V = bundles[0].E.shape[0]
    x_values = [V if k is None else k for k in k_values]

    pair_colors = SEED_COLORS * 2
    for (i, j), kvals in sweep_results.items():
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        ys = [kvals[k][0] for k in k_values]
        ax.plot(x_values, ys, marker="o", markersize=4, lw=1.4,
                color=col, label=label)

    # Highlight the chosen K.
    x_chosen = V if chosen_k is None else chosen_k
    ax.axvline(x_chosen, color="black", ls="--", lw=1.0, alpha=0.5,
               label=f"chosen K = {x_chosen}")
    # Threshold reference lines.
    ax.axhline(0.10, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(x_values[0], 0.10, " ρ_E = 0.10 (excellent threshold)",
            va="bottom", fontsize=8, color="gray")
    ax.axhline(0.20, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(x_values[0], 0.20, " ρ_E = 0.20 (good threshold)",
            va="bottom", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_xlabel("Anchor set size K (top-K by per-row norm, intersected)")
    ax.set_ylabel("ρ_E (Procrustes residual on the K-subset)")
    ax.set_title(
        "Alignment quality vs anchor-set size\n"
        "(small K = only well-trained tokens; large K = includes "
        "undertrained-token noise)"
    )
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perlayer_angles(
    bundles: List[SeedAlignBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Per-layer principal angles (top-10) for all cross-seed pairs,
    in three panels: aligned / identity / random.

    Each panel has one line per ordered pair (A, B), A != B.
    """
    L = bundles[0].L
    layer_indices = np.arange(L)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    pair_labels = list(pair_results.keys())
    pair_colors = SEED_COLORS * 2  # extend for >6 pairs if needed

    for (i, j), data in pair_results.items():
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        axes[0].plot(layer_indices, data["angles_aligned"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)
        axes[1].plot(layer_indices, data["angles_identity"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)
        axes[2].plot(layer_indices, data["angles_random"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)

    axes[0].set_title("Aligned (via Procrustes on embeddings)")
    axes[1].set_title("Identity baseline (no alignment)")
    axes[2].set_title("Random-orthogonal baseline")
    axes[0].set_ylabel("Mean principal angle of top-10 directions (degrees)")
    for ax in axes:
        ax.set_xlabel("layer state t")
        ax.axhline(90.0, color="gray", ls=":", lw=1, alpha=0.7)
        ax.set_ylim(0, 95)
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    fig.suptitle(
        "Cross-seed R-matrix similarity per layer "
        "(top-10 principal angles; 0° = identical subspace, 90° = orthogonal)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rho_e_per_layer(
    bundles: List[SeedAlignBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Per-layer Frobenius R-distance for all cross-seed pairs,
    aligned vs identity vs random. Log-y scale because identity and
    random baselines may be much larger than aligned values.
    """
    L = bundles[0].L
    layer_indices = np.arange(L)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    pair_colors = SEED_COLORS * 2

    for (i, j), data in pair_results.items():
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        axes[0].plot(layer_indices, data["R_aligned_dist_per_layer"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)
        axes[1].plot(layer_indices, data["R_identity_dist_per_layer"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)
        axes[2].plot(layer_indices, data["R_random_dist_per_layer"],
                     marker="o", markersize=3, lw=1.4, color=col, label=label)

    axes[0].set_title("Aligned (via Procrustes on embeddings)")
    axes[1].set_title("Identity baseline (no alignment)")
    axes[2].set_title("Random-orthogonal baseline")
    axes[0].set_ylabel("Frobenius distance ‖R_A_transported − R_B‖")
    for ax in axes:
        ax.set_xlabel("layer state t")
        ax.set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    fig.suptitle(
        "Cross-seed R-matrix Frobenius distance per layer "
        "(log scale; aligned should be << identity << random)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------
def write_summary(
    bundles: List[SeedAlignBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    rho_e_matrix: np.ndarray,
    output_path: str,
    top_k: Optional[int] = None,
    sweep_results: Optional[Dict] = None,
    sweep_k_values: Optional[List[Optional[int]]] = None,
):
    """Pretty text summary. Includes a Phase-2 readiness verdict.

    Args:
        top_k: the anchor-set size used for the headline pair_results.
            Used in the verdict heading and the comment about anchor
            selection. None = full vocabulary.
        sweep_results, sweep_k_values: optional ρ_E-vs-K sweep data.
            If provided, an additional section is written showing the
            sweep curve as a table.
    """
    n = len(bundles)
    pairs = [(i, j) for i, j in pair_results.keys()]
    aligned_rhos = [pair_results[p]["rho_E"] for p in pairs]
    aligned_R_sums = [
        float(pair_results[p]["R_aligned_dist_per_layer"].sum()) for p in pairs
    ]
    identity_R_sums = [
        float(pair_results[p]["R_identity_dist_per_layer"].sum()) for p in pairs
    ]
    random_R_sums = [
        float(pair_results[p]["R_random_dist_per_layer"].sum()) for p in pairs
    ]
    # Mean across-pair statistics.
    mean_rho = float(np.mean(aligned_rhos))
    mean_aligned_R = float(np.mean(aligned_R_sums))
    mean_identity_R = float(np.mean(identity_R_sums))
    mean_random_R = float(np.mean(random_R_sums))
    # Aligned-to-random ratio: alignment is "working" if << 1.
    aligned_vs_random = mean_aligned_R / max(mean_random_R, 1e-30)
    # Aligned-to-identity ratio: alignment helps if < 1.
    aligned_vs_identity = mean_aligned_R / max(mean_identity_R, 1e-30)

    # Mean angles.
    mean_aligned_angle = float(np.mean([
        float(pair_results[p]["angles_aligned"].mean()) for p in pairs
    ]))
    mean_identity_angle = float(np.mean([
        float(pair_results[p]["angles_identity"].mean()) for p in pairs
    ]))
    mean_random_angle = float(np.mean([
        float(pair_results[p]["angles_random"].mean()) for p in pairs
    ]))

    with open(output_path, "w") as f:
        title = (f"Cross-seed alignment check "
                 f"({n} seeds: {', '.join(b.label for b in bundles)})")
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(
            "Vocabulary-anchored Procrustes alignment using each seed's\n"
            "learned input-embedding matrix as the Rosetta-stone anchor.\n"
            "For each ordered pair (A, B), we find the orthogonal Q\n"
            "minimizing ‖E_A Q − E_B‖_F, then use that single Q to transport\n"
            "A's R(t) matrices into B's coordinate system, and compare to\n"
            "B's R(t) by Frobenius distance and top-10 principal angles.\n\n"
        )
        # Describe anchor selection.
        V = bundles[0].E.shape[0]
        if top_k is None or top_k >= V:
            f.write(
                f"Anchor set: full vocabulary, V = {V} tokens.\n"
                f"⚠️  At our model scale, the bulk of the vocabulary is\n"
                f"   undertrained — its embeddings stay near random init\n"
                f"   and contribute noise to the Procrustes residual.\n"
                f"   Consider running with --top_k_by_norm 1000 for a more\n"
                f"   informative measurement.\n\n"
            )
        else:
            # Pull n_anchor info from the first pair (all pairs have similar
            # intersection sizes for similar-architecture seeds).
            example_pair = next(iter(pair_results.keys()))
            n_anchor_used = pair_results[example_pair].get("n_anchor_tokens", "?")
            f.write(
                f"Anchor set: top {top_k} tokens by per-row L2 norm,\n"
                f"intersected between each pair. Typical anchor-set size\n"
                f"after intersection: {n_anchor_used} (out of K={top_k}).\n"
                f"Rationale: per-row norm proxies for 'this token was actively\n"
                f"trained'. Undertrained-token noise dominates the full-vocab\n"
                f"Procrustes residual at our scale; filtering by norm-rank\n"
                f"isolates the alignable component.\n\n"
            )
        f.write(
            "Two baselines are reported alongside:\n"
            "  identity = no alignment (Q = I)\n"
            "  random   = Haar-uniform random orthogonal Q\n"
            "Alignment is 'working' if aligned R-distance is much smaller\n"
            "than both baselines.\n\n"
        )

        # Optional: ρ_E vs K sweep table.
        if sweep_results is not None and sweep_k_values is not None:
            f.write("=== ρ_E vs anchor-set size (diagnostic sweep) ===\n\n")
            f.write(
                "How alignment quality depends on the size of the\n"
                "anchor-set K. Small K = only well-trained tokens;\n"
                "large K = includes undertrained-token noise.\n"
                "Each row = one ordered pair; each column = one K value.\n\n"
            )
            # Header.
            f.write(f"   {'pair':>16}")
            for k in sweep_k_values:
                label = f"K={V}" if k is None else f"K={k}"
                f.write(f"  {label:>10}")
            f.write("\n")
            f.write("   " + "-" * (16 + 12 * len(sweep_k_values)) + "\n")
            for (i, j), kvals in sweep_results.items():
                label = f"{bundles[i].label}→{bundles[j].label}"
                f.write(f"   {label:>16}")
                for k in sweep_k_values:
                    rho_at_k, _ = kvals[k]
                    f.write(f"  {rho_at_k:>10.4f}")
                f.write("\n")
            # Mean row.
            f.write("   " + "-" * (16 + 12 * len(sweep_k_values)) + "\n")
            f.write(f"   {'mean':>16}")
            for k in sweep_k_values:
                rhos = [sweep_results[p][k][0] for p in sweep_results.keys()]
                f.write(f"  {float(np.mean(rhos)):>10.4f}")
            f.write("\n\n")


        # ρ_E pairwise table.
        f.write("=== Embedding-space alignment residual ρ_E ===\n\n")
        f.write(f"   {'':>12}")
        for b in bundles:
            f.write(f"  {b.label:>12}")
        f.write("\n")
        for i, b in enumerate(bundles):
            f.write(f"   {b.label:>12}")
            for j in range(n):
                if i == j:
                    f.write(f"  {'—':>12}")
                else:
                    f.write(f"  {rho_e_matrix[i, j]:>12.4f}")
            f.write("\n")
        f.write(f"\n   mean off-diagonal ρ_E = {mean_rho:.4f}\n")
        f.write(
            f"   (small = embedding spaces correspond well after rotation;\n"
            f"    typical thresholds: < 0.10 excellent, < 0.20 good,\n"
            f"    > 0.30 questionable)\n\n"
        )

        # Per-pair R-distance summary.
        f.write("=== Per-pair R-matrix distance (summed over layers) ===\n\n")
        header = (
            f"   {'pair':>16}  {'aligned':>10}  {'identity':>10}  "
            f"{'random':>10}  {'aln/rnd':>9}  {'aln/id':>9}"
        )
        f.write(header + "\n")
        f.write("   " + "-" * (len(header) - 3) + "\n")
        for p in pairs:
            i, j = p
            label = f"{bundles[i].label}→{bundles[j].label}"
            aln = float(pair_results[p]["R_aligned_dist_per_layer"].sum())
            idd = float(pair_results[p]["R_identity_dist_per_layer"].sum())
            rnd = float(pair_results[p]["R_random_dist_per_layer"].sum())
            f.write(
                f"   {label:>16}  {aln:>10.3f}  {idd:>10.3f}  {rnd:>10.3f}  "
                f"{aln/max(rnd,1e-30):>9.4f}  {aln/max(idd,1e-30):>9.4f}\n"
            )
        f.write("   " + "-" * (len(header) - 3) + "\n")
        f.write(
            f"   {'mean':>16}  {mean_aligned_R:>10.3f}  "
            f"{mean_identity_R:>10.3f}  {mean_random_R:>10.3f}  "
            f"{aligned_vs_random:>9.4f}  {aligned_vs_identity:>9.4f}\n\n"
        )

        # Angle summary.
        f.write("=== Per-pair mean principal angle (top-10), degrees ===\n\n")
        header = (
            f"   {'pair':>16}  {'aligned':>10}  {'identity':>10}  {'random':>10}"
        )
        f.write(header + "\n")
        f.write("   " + "-" * (len(header) - 3) + "\n")
        for p in pairs:
            i, j = p
            label = f"{bundles[i].label}→{bundles[j].label}"
            aln = float(pair_results[p]["angles_aligned"].mean())
            idd = float(pair_results[p]["angles_identity"].mean())
            rnd = float(pair_results[p]["angles_random"].mean())
            f.write(
                f"   {label:>16}  {aln:>10.2f}  {idd:>10.2f}  {rnd:>10.2f}\n"
            )
        f.write("   " + "-" * (len(header) - 3) + "\n")
        f.write(
            f"   {'mean':>16}  {mean_aligned_angle:>10.2f}  "
            f"{mean_identity_angle:>10.2f}  {mean_random_angle:>10.2f}\n\n"
        )

        # Phase-2 readiness verdict.
        f.write("=== Phase-2 readiness assessment ===\n\n")
        verdicts = []
        # Criterion 1: ρ_E reasonable.
        if mean_rho < 0.10:
            verdicts.append(("ρ_E", "PASS", f"mean ρ_E = {mean_rho:.3f} < 0.10"))
        elif mean_rho < 0.20:
            verdicts.append(("ρ_E", "MARGINAL",
                             f"mean ρ_E = {mean_rho:.3f} in [0.10, 0.20)"))
        else:
            verdicts.append(("ρ_E", "FAIL",
                             f"mean ρ_E = {mean_rho:.3f} ≥ 0.20"))
        # Criterion 2: aligned R-distance much less than random.
        if aligned_vs_random < 0.25:
            verdicts.append(("aligned/random R", "PASS",
                             f"ratio = {aligned_vs_random:.3f} < 0.25"))
        elif aligned_vs_random < 0.50:
            verdicts.append(("aligned/random R", "MARGINAL",
                             f"ratio = {aligned_vs_random:.3f} in [0.25, 0.50)"))
        else:
            verdicts.append(("aligned/random R", "FAIL",
                             f"ratio = {aligned_vs_random:.3f} ≥ 0.50"))
        # Criterion 3: aligned R-distance less than identity.
        # (i.e., alignment helps over no-alignment)
        if aligned_vs_identity < 0.80:
            verdicts.append(("aligned/identity R", "PASS",
                             f"ratio = {aligned_vs_identity:.3f} < 0.80 "
                             f"(alignment beats no-op)"))
        elif aligned_vs_identity < 1.0:
            verdicts.append(("aligned/identity R", "MARGINAL",
                             f"ratio = {aligned_vs_identity:.3f} "
                             f"(alignment barely helps)"))
        else:
            verdicts.append(("aligned/identity R", "FAIL",
                             f"ratio = {aligned_vs_identity:.3f} ≥ 1.0 "
                             f"(alignment makes it worse!)"))
        # Criterion 4: aligned angle reasonable.
        if mean_aligned_angle < 30.0:
            verdicts.append(("aligned angle", "PASS",
                             f"mean angle = {mean_aligned_angle:.1f}° < 30°"))
        elif mean_aligned_angle < 60.0:
            verdicts.append(("aligned angle", "MARGINAL",
                             f"mean angle = {mean_aligned_angle:.1f}° "
                             f"in [30°, 60°)"))
        else:
            verdicts.append(("aligned angle", "FAIL",
                             f"mean angle = {mean_aligned_angle:.1f}° ≥ 60°"))

        for criterion, verdict, detail in verdicts:
            f.write(f"  [{verdict:>8}] {criterion}: {detail}\n")
        f.write("\n")

        if all(v[1] == "PASS" for v in verdicts):
            f.write("  OVERALL: ✅ ALIGNMENT WORKING — Phase 2 alignment\n")
            f.write("  procedure validated on cross-seed (within-variant) data.\n")
            f.write("  Cross-variant alignment in Phase 2 will be harder, but\n")
            f.write("  the procedure itself is sound.\n")
        elif any(v[1] == "FAIL" for v in verdicts):
            f.write("  OVERALL: ⚠️  ALIGNMENT QUESTIONABLE — at least one\n")
            f.write("  criterion failed. Phase 2 may need a different anchor\n")
            f.write("  (e.g., per-layer activation-space alignment instead of\n")
            f.write("  embedding-space) before launching the full campaign.\n")
        else:
            f.write("  OVERALL: 🟡 ALIGNMENT MARGINAL — works but residuals\n")
            f.write("  larger than ideal. Phase 2 can proceed but interpret\n")
            f.write("  cross-variant alignment results cautiously.\n")


def write_summary_csv(
    bundles: List[SeedAlignBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """CSV summary with per-pair scalars."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed_A", "seed_B", "rho_E", "n_anchor_tokens",
            "R_aligned_sum", "R_identity_sum", "R_random_sum",
            "aligned_over_random", "aligned_over_identity",
            "mean_angle_aligned_deg", "mean_angle_identity_deg",
            "mean_angle_random_deg",
        ])
        for (i, j), data in pair_results.items():
            aln_R = float(data["R_aligned_dist_per_layer"].sum())
            id_R = float(data["R_identity_dist_per_layer"].sum())
            rnd_R = float(data["R_random_dist_per_layer"].sum())
            writer.writerow([
                bundles[i].label, bundles[j].label, data["rho_E"],
                data.get("n_anchor_tokens", -1),
                aln_R, id_R, rnd_R,
                aln_R / max(rnd_R, 1e-30),
                aln_R / max(id_R, 1e-30),
                float(data["angles_aligned"].mean()),
                float(data["angles_identity"].mean()),
                float(data["angles_random"].mean()),
            ])


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-seed vocabulary-anchored Procrustes alignment "
                    "check — the Phase 1 → Phase 2 gate.",
    )
    parser.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="Paths to seed run directories.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write outputs. Default: alignment_check/ alongside "
             "the first run_dir's parent.",
    )
    parser.add_argument(
        "--top_k_by_norm", type=int, default=1000,
        help="Anchor-set size: use the top-K tokens (per-row L2 norm, "
             "intersected between each pair) as Procrustes anchors. "
             "Default 1000. Pass 0 or a value ≥ vocab size to disable "
             "filtering and use the full vocabulary (the v1 behavior; "
             "expect ρ_E ≈ 0.6 at our model scale because most rare "
             "tokens are undertrained).",
    )
    parser.add_argument(
        "--rho_e_sweep", action="store_true",
        help="Additionally compute ρ_E at multiple K values "
             "(100, 1000, 5000, full vocab) for each seed pair, and "
             "produce a sweep plot showing alignment quality vs anchor-"
             "set size. Cheap: only re-does the SVD-based Procrustes "
             "step, not the R-transport.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        first_parent = os.path.dirname(os.path.abspath(args.run_dirs[0]))
        args.output_dir = os.path.join(first_parent, "alignment_check")
    os.makedirs(args.output_dir, exist_ok=True)

    # Normalize top_k_by_norm: 0 or negative ⇒ None (full vocab).
    top_k = args.top_k_by_norm if args.top_k_by_norm > 0 else None

    # Load all seeds.
    print(f">> Loading {len(args.run_dirs)} seed(s) ...")
    bundles = []
    for run_dir in args.run_dirs:
        b = SeedAlignBundle(run_dir)
        bundles.append(b)
        print(f"   {b}")
    bundles.sort(key=lambda b: b.fs.seed)

    # Sanity checks.
    H0 = bundles[0].E.shape[1]
    V0 = bundles[0].E.shape[0]
    L0 = bundles[0].L
    for b in bundles[1:]:
        if b.E.shape != (V0, H0):
            raise ValueError(
                f"{b.run_dir} embedding shape {b.E.shape} doesn't match "
                f"{bundles[0].run_dir} shape ({V0}, {H0})."
            )
        if b.L != L0:
            raise ValueError(
                f"{b.run_dir} L={b.L} doesn't match {bundles[0].run_dir} L={L0}."
            )

    if top_k is None:
        print(f"\n>> Anchor selection: full vocabulary ({V0} tokens). "
              f"Likely to fail at our model scale.")
    else:
        print(f"\n>> Anchor selection: top-{top_k} tokens by per-row norm, "
              f"intersected per pair.")

    # Compute pairwise alignments for all ordered pairs (A, B), A != B.
    print(f"\n>> Computing pairwise alignments ({len(bundles)}*{len(bundles)-1} "
          f"= {len(bundles) * (len(bundles) - 1)} ordered pairs) ...")
    pair_results = {}
    rho_e_matrix = np.zeros((len(bundles), len(bundles)), dtype=np.float32)
    for i, b_A in enumerate(bundles):
        for j, b_B in enumerate(bundles):
            if i == j:
                continue
            data = compute_pairwise_alignment(b_A, b_B, top_k=top_k)
            pair_results[(i, j)] = data
            rho_e_matrix[i, j] = data["rho_E"]
            print(f"   {b_A.label} → {b_B.label}: "
                  f"ρ_E = {data['rho_E']:.4f} "
                  f"(K_eff = {data['n_anchor_tokens']}), "
                  f"R-dist (aligned) = "
                  f"{data['R_aligned_dist_per_layer'].sum():.2f}, "
                  f"mean angle = {data['angles_aligned'].mean():.1f}°")

    # Optional sweep.
    sweep_results = None
    sweep_k_values = None
    if args.rho_e_sweep:
        # Pick K values that span the diagnostic range.
        sweep_k_values = [100, 1000, 5000, None]  # None = full vocab
        # Drop any K values that exceed V (no need).
        sweep_k_values = [k for k in sweep_k_values if k is None or k <= V0]
        print(f"\n>> Sweeping ρ_E across K = "
              f"{[V0 if k is None else k for k in sweep_k_values]} ...")
        sweep_results = {}
        for i, b_A in enumerate(bundles):
            for j, b_B in enumerate(bundles):
                if i == j:
                    continue
                kvals = compute_rho_e_sweep(b_A, b_B, sweep_k_values)
                sweep_results[(i, j)] = kvals

    # Write outputs.
    print(f"\n>> Writing outputs to {args.output_dir} ...")
    txt = os.path.join(args.output_dir, "alignment_summary.txt")
    write_summary(
        bundles, pair_results, rho_e_matrix, txt,
        top_k=top_k, sweep_results=sweep_results,
        sweep_k_values=sweep_k_values,
    )
    print(f"   ↳ {txt}")
    csv_path = os.path.join(args.output_dir, "alignment_summary.csv")
    write_summary_csv(bundles, pair_results, csv_path)
    print(f"   ↳ {csv_path}")
    p1 = os.path.join(args.output_dir, "pairwise_rho_e.png")
    plot_rho_e_heatmap(bundles, rho_e_matrix, p1)
    print(f"   ↳ {p1}")
    p2 = os.path.join(args.output_dir, "perlayer_angles.png")
    plot_perlayer_angles(bundles, pair_results, p2)
    print(f"   ↳ {p2}")
    p3 = os.path.join(args.output_dir, "perlayer_R_distances.png")
    plot_rho_e_per_layer(bundles, pair_results, p3)
    print(f"   ↳ {p3}")
    if args.rho_e_sweep:
        p4 = os.path.join(args.output_dir, "rho_e_vs_k.png")
        plot_rho_e_sweep(bundles, sweep_results, sweep_k_values, top_k, p4)
        print(f"   ↳ {p4}")

    # Echo summary.
    print()
    with open(txt) as f:
        print(f.read())

    print(">> ✅ Alignment check complete.")


if __name__ == "__main__":
    main()
    