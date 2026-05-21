"""
Cross-seed activation-space Procrustes alignment check.

The companion `alignment_check.py` script uses each seed's embedding
matrix as the Procrustes anchor. That approach gave ρ_E ≈ 0.10 (with
top-1000-token filtering) but failed to reduce the R-matrix
Frobenius distance or the principal angles below the random-
orthogonal baseline — i.e., the embedding-space Q didn't transport
the R matrices into matching coordinate frames.

This script implements the activation-space alternative
(`align.align_activations_per_layer` from `align.py`). For each
ordered pair (A, B):

  1. Run both models on the *same* eval batches in the *same* order
     to get per-layer activations X_t^(A) and X_t^(B), shape (N, H)
     each, where the i-th row at layer t in A and B corresponds to
     the same (input chunk, position) pair.
  2. For each layer t, find the orthogonal Q_t minimizing
     ‖X_t^(A) Q_t − X_t^(B)‖_F.
  3. Transport A's R(t) into B's coords using *the per-layer Q_t*
     (different Q per layer, not a single global Q).
  4. Measure the residual R-matrix distance and per-layer principal
     angles after this transport.

The headline number is the **mean per-layer ρ_t**, averaged across
the ordered pairs. This activation-space ρ_t is the right anchor for
checking whether the residual stream's coordinate frame can be
mapped between seeds, as opposed to the embedding-space ρ_E which
only checks the vocabulary geometry.

Workflow optimization: each seed's activations are computed once on
a fixed eval loader and cached in run_dir/aligned_activations.npy.
The pairwise alignment loop then reads these cached files and does
pure-numpy alignment + transport + comparison.

Usage:
    python3 alignment_check_activations.py \\
        --run_dirs ../phase1_runs/seed_0 ../phase1_runs/seed_1 \\
                   ../phase1_runs/seed_2 ../phase1_runs/seed_3 \\
        --output_dir ../phase1_runs/alignment_check_act

Outputs (relative to --output_dir):
    alignment_act_summary.txt    — per-pair mean ρ_t, R-distance,
                                   angles, baselines, Phase-2 readiness
    alignment_act_summary.csv    — same, machine-readable
    pairwise_mean_rho_t.png      — heatmap of mean-across-layers ρ_t
    rho_t_per_layer.png          — per-layer ρ_t curves, all pairs
    perlayer_angles.png          — per-layer angles, four panels
                                   (aligned-act / aligned-emb /
                                    identity / random)
    perlayer_R_distances.png     — same layout for Frobenius R distance
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
    align_embeddings, align_activations_per_layer,
    transport_R_per_layer, orthogonal_procrustes,
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
# Activation collection — cached per seed.
# ----------------------------------------------------------------------
ACTIVATION_CACHE_NAME = "aligned_activations.npy"
EMBEDDING_CACHE_NAME = "final_embedding.npy"


def get_seed_activations(
    run_dir: str, max_pilots: int = 9500, verbose: bool = True,
) -> np.ndarray:
    """
    Return (L_total, N, H) activations from this seed's final checkpoint,
    collected on the eval loader. Cached to run_dir/aligned_activations.npy.

    The cache is what makes pairwise alignment cheap: each seed's
    activations are computed once and reused for all pairs involving
    that seed.

    The cached activations must come from the SAME eval batches in the
    SAME order across all seeds. That requirement is met by reproducible
    eval-loader construction in `data.make_dataloaders` (deterministic
    given identical config + seed=0). If you ever change the eval
    construction, delete the existing caches first.
    """
    cache_path = os.path.join(run_dir, ACTIVATION_CACHE_NAME)
    if os.path.exists(cache_path):
        if verbose:
            print(f"   ↳ Loading cached activations from {cache_path}")
        return np.load(cache_path)

    # No cache: extract by running inference on the final checkpoint.
    if verbose:
        print(f"   ↳ Computing activations for {run_dir} (no cache yet) ...")

    # Lazy imports of heavy modules.
    import torch
    from config import load_config_pair
    from data import prepare_dataset, make_dataloaders
    from analyze import collect_activations, default_pilot_positions
    from models import build_model

    # Load model config + checkpoint.
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"run_metadata.json missing from {run_dir} — can't load model config."
        )
    model_cfg, train_cfg = load_config_pair(metadata_path)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, ckpt_files[-1])
    if verbose:
        print(f"      Loading checkpoint: {ckpt_path}")

    # Build eval loader. seed=0 for determinism — critical for cross-seed
    # alignment to use the same input rows.
    _, held_out_dataset = prepare_dataset(model_cfg=model_cfg,
                                          train_cfg=train_cfg)
    _, eval_loader = make_dataloaders(
        train_dataset=held_out_dataset,
        held_out_dataset=held_out_dataset,
        train_cfg=train_cfg, seed=0, num_workers=2,
    )

    # Load model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Pick pilot positions matching the analyzer's defaults. The analyzer
    # uses seq_len=2048 as a "max" and collect_activations filters positions
    # that exceed the actual T per batch. We use train_seq_len when
    # available, otherwise the same 2048 default the analyzer uses.
    seq_len = getattr(train_cfg, "train_seq_len", None)
    if seq_len is None:
        seq_len = 2048
    pilot_positions = default_pilot_positions(seq_len=seq_len)
    if verbose:
        print(f"      Pilot positions: {len(pilot_positions)} per chunk "
              f"(seq_len = {seq_len})")

    # Autocast dtype matches training conventions.
    if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Collect.
    t0 = time.time()
    activations = collect_activations(
        model=model, eval_loader=eval_loader,
        pilot_positions=pilot_positions, device=device,
        autocast_dtype=autocast_dtype, max_pilots=max_pilots,
    )
    if verbose:
        print(f"      Collected {activations.shape[1]:,} pilot activations "
              f"× {activations.shape[0]} layers × {activations.shape[2]} dims "
              f"in {time.time() - t0:.1f}s")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Cache.
    np.save(cache_path, activations)
    if verbose:
        print(f"      Cached to {cache_path} "
              f"({os.path.getsize(cache_path) / 2**20:.1f} MB)")
    return activations


def get_final_R(run_dir: str) -> Tuple[np.ndarray, int, int]:
    """Load only the final-checkpoint R matrix. Same helper as in
    alignment_check.py."""
    flow_dir = os.path.join(run_dir, "flow_analysis")
    flow_files = sorted([f for f in os.listdir(flow_dir)
                         if f.startswith("flow_step_") and f.endswith(".npz")])
    final_flow = load_flow(os.path.join(flow_dir, flow_files[-1]))
    return (
        final_flow["R"],
        int(final_flow["checkpoint_step"]),
        int(final_flow["num_layers_total"]),
    )


def get_final_embedding(run_dir: str) -> Optional[np.ndarray]:
    """Optionally load the cached embedding for the single-Q baseline.
    Returns None if no cache exists — the single-Q baseline is then
    omitted from the report."""
    cache_path = os.path.join(run_dir, EMBEDDING_CACHE_NAME)
    if os.path.exists(cache_path):
        return np.load(cache_path)
    return None


# ----------------------------------------------------------------------
# Per-seed bundle.
# ----------------------------------------------------------------------
class SeedBundle:
    """Holds everything needed for cross-seed activation-space alignment."""

    def __init__(self, run_dir: str, max_pilots: int = 9500):
        self.run_dir = run_dir
        self.fs = load_flow_series(run_dir)
        self.label = f"seed {self.fs.seed}"
        self.activations = get_seed_activations(run_dir, max_pilots=max_pilots)
        # (L_total, N, H).
        self.L_total = self.activations.shape[0]
        self.N = self.activations.shape[1]
        self.H = self.activations.shape[2]
        self.R, self.step, self.L_R = get_final_R(run_dir)  # (L_total, H, H)
        # Sanity-check that activation L matches R's L.
        if self.L_total != self.L_R:
            raise ValueError(
                f"{run_dir}: activations have L_total = {self.L_total} but "
                f"R matrices have L_R = {self.L_R}. These must match — both "
                f"should be num_layers + 2."
            )
        # Optional embedding for single-Q baseline comparison.
        self.E = get_final_embedding(run_dir)

    def __repr__(self):
        return (f"SeedBundle({self.label}, step={self.step}, "
                f"activations={self.activations.shape}, R={self.R.shape})")


# ----------------------------------------------------------------------
# Baselines.
# ----------------------------------------------------------------------
def random_orthogonal(H: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal H×H matrix (Haar-uniform)."""
    M = rng.standard_normal((H, H))
    Q, _ = np.linalg.qr(M)
    return Q.astype(np.float32)


def per_layer_R_distance(R_A: np.ndarray, R_B: np.ndarray) -> np.ndarray:
    """Frobenius distance per layer with no alignment."""
    L = R_A.shape[0]
    out = np.zeros(L, dtype=np.float32)
    for t in range(L):
        out[t] = float(np.linalg.norm(R_A[t] - R_B[t]))
    return out


def top_k_by_norm_intersection(
    E_A: np.ndarray, E_B: np.ndarray, k: int,
) -> np.ndarray:
    """Top-K-by-norm intersection (mirror of `alignment_check.top_k_...`).
    Used here only for the single-Q embedding baseline comparison."""
    V = E_A.shape[0]
    if k >= V:
        return np.arange(V, dtype=np.int64)
    norms_A = np.linalg.norm(E_A, axis=1)
    norms_B = np.linalg.norm(E_B, axis=1)
    top_A = set(np.argsort(-norms_A)[:k].tolist())
    top_B = set(np.argsort(-norms_B)[:k].tolist())
    return np.array(sorted(top_A & top_B), dtype=np.int64)


# ----------------------------------------------------------------------
# Pairwise alignment.
# ----------------------------------------------------------------------
def compute_pairwise_alignment(
    bundle_A: SeedBundle, bundle_B: SeedBundle,
    rng_seed: int = 1234,
    embedding_top_k: int = 1000,
) -> Dict:
    """
    Run activation-space alignment for one ordered pair (A, B), plus
    three baselines (identity, random-per-layer, embedding-single-Q
    if embeddings are available).

    Returns a dict with:
        rho_t_per_layer: (L,) — per-layer activation-space residual ratio
        mean_rho_t: scalar — average across layers (the headline number)
        Qs: list of L (H, H) per-layer orthogonal alignments
        R_aligned_dist_per_layer: (L,) Frobenius distance after per-layer alignment
        R_identity_dist_per_layer: (L,) distance with no alignment
        R_random_dist_per_layer: (L,) distance with random per-layer Q
        R_embedding_dist_per_layer: (L,) distance with single embedding Q
                                       (NaN if no embedding cache)
        angles_aligned, angles_identity, angles_random, angles_embedding:
            (L,) mean principal-angle profiles for the four conditions
    """
    L = bundle_A.L_total
    H = bundle_A.H

    # Per-layer activation alignment.
    Qs, rho_t = align_activations_per_layer(
        bundle_A.activations, bundle_B.activations,
    )
    rho_t_arr = np.array(rho_t, dtype=np.float32)
    R_A_act = transport_R_per_layer(bundle_A.R, Qs)
    R_aligned_dist = aligned_R_distance(R_A_act, bundle_B.R, per_layer=True)
    angles_aligned = principal_angle_profile(R_A_act, bundle_B.R, top_k=10)

    # Identity baseline: no alignment.
    R_identity_dist = per_layer_R_distance(bundle_A.R, bundle_B.R)
    angles_identity = principal_angle_profile(bundle_A.R, bundle_B.R, top_k=10)

    # Random per-layer baseline: independent random Q at each layer.
    # Using a different RNG state per layer matches what a "no learning"
    # alignment would look like.
    rng = np.random.default_rng(rng_seed)
    Q_rand_list = [random_orthogonal(H, rng) for _ in range(L)]
    R_A_rand = transport_R_per_layer(bundle_A.R, Q_rand_list)
    R_random_dist = aligned_R_distance(R_A_rand, bundle_B.R, per_layer=True)
    angles_random = principal_angle_profile(R_A_rand, bundle_B.R, top_k=10)

    # Embedding-single-Q baseline (only if both seeds have cached embeddings).
    if bundle_A.E is not None and bundle_B.E is not None:
        anchor_idx = top_k_by_norm_intersection(
            bundle_A.E, bundle_B.E, embedding_top_k,
        )
        Q_emb, _ = align_embeddings(
            bundle_A.E[anchor_idx], bundle_B.E[anchor_idx],
        )
        R_A_emb = transport_R_per_layer(bundle_A.R, [Q_emb] * L)
        R_embedding_dist = aligned_R_distance(
            R_A_emb, bundle_B.R, per_layer=True,
        )
        angles_embedding = principal_angle_profile(
            R_A_emb, bundle_B.R, top_k=10,
        )
    else:
        R_embedding_dist = np.full(L, np.nan, dtype=np.float32)
        angles_embedding = np.full(L, np.nan, dtype=np.float32)

    return {
        "rho_t_per_layer": rho_t_arr,
        "mean_rho_t": float(rho_t_arr.mean()),
        "Qs": Qs,
        "R_aligned_dist_per_layer": R_aligned_dist,
        "R_identity_dist_per_layer": R_identity_dist,
        "R_random_dist_per_layer": R_random_dist,
        "R_embedding_dist_per_layer": R_embedding_dist,
        "angles_aligned": angles_aligned,
        "angles_identity": angles_identity,
        "angles_random": angles_random,
        "angles_embedding": angles_embedding,
    }


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_pairwise_mean_rho_t(
    bundles: List[SeedBundle],
    rho_matrix: np.ndarray,
    output_path: str,
):
    """Heatmap of mean-across-layers ρ_t for each ordered pair."""
    n = len(bundles)
    labels = [b.label for b in bundles]
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    im = ax.imshow(rho_matrix, cmap="viridis_r", vmin=0,
                   vmax=max(rho_matrix.max(), 1e-6))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            txt_color = "white" if rho_matrix[i, j] > rho_matrix.max() * 0.5 else "black"
            ax.text(j, i, f"{rho_matrix[i, j]:.3f}",
                    ha="center", va="center", color=txt_color, fontsize=9)
    fig.colorbar(im, ax=ax, label="mean ρ_t across layers")
    ax.set_title("Pairwise activation-space alignment residual\n"
                 "(mean of per-layer ρ_t; lower = better)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_rho_t_per_layer(
    bundles: List[SeedBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Per-layer ρ_t for all pairs on a single panel."""
    L = bundles[0].L_total
    layer_indices = np.arange(L)
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    pair_colors = SEED_COLORS * 2
    for (i, j), data in pair_results.items():
        label = f"{bundles[i].label}→{bundles[j].label}"
        color_idx = (i * len(bundles) + j) % len(pair_colors)
        col = pair_colors[color_idx]
        ax.plot(layer_indices, data["rho_t_per_layer"],
                marker="o", markersize=3, lw=1.4, color=col, label=label)
    ax.set_xlabel("layer state t")
    ax.set_ylabel("ρ_t  (Procrustes residual on layer t's activations)")
    ax.set_title("Per-layer activation-space alignment residual")
    ax.axhline(0.10, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(0, 0.10, " ρ_t = 0.10 threshold", va="bottom",
            fontsize=8, color="gray")
    ax.axhline(0.20, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.text(0, 0.20, " ρ_t = 0.20 threshold", va="bottom",
            fontsize=8, color="gray")
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perlayer_angles(
    bundles: List[SeedBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Four panels: aligned-act / aligned-emb / identity / random.
    Same Y-axis range to make the comparison visual."""
    has_emb = not np.all(np.isnan(
        next(iter(pair_results.values()))["angles_embedding"]
    ))
    n_panels = 4 if has_emb else 3
    L = bundles[0].L_total
    layer_indices = np.arange(L)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                             sharey=True)
    pair_colors = SEED_COLORS * 2

    titles = ["Aligned (activation-space, per-layer Q_t)"]
    keys = ["angles_aligned"]
    if has_emb:
        titles.append("Embedding-single-Q baseline")
        keys.append("angles_embedding")
    titles.append("Identity baseline (no alignment)")
    keys.append("angles_identity")
    titles.append("Random-orthogonal baseline (per-layer)")
    keys.append("angles_random")

    for ax_idx, (title, key) in enumerate(zip(titles, keys)):
        ax = axes[ax_idx]
        for (i, j), data in pair_results.items():
            label = f"{bundles[i].label}→{bundles[j].label}"
            color_idx = (i * len(bundles) + j) % len(pair_colors)
            col = pair_colors[color_idx]
            ax.plot(layer_indices, data[key],
                    marker="o", markersize=3, lw=1.4, color=col, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer state t")
        ax.set_ylim(0, 95)
        ax.axhline(90, color="gray", ls=":", lw=0.8, alpha=0.7)
    axes[0].set_ylabel("Mean principal angle of top-10 directions (degrees)")
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    fig.suptitle(
        "Cross-seed R-matrix similarity per layer "
        "(top-10 principal angles after transport)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_perlayer_R_distances(
    bundles: List[SeedBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Mirror of plot_perlayer_angles for Frobenius R-distance."""
    has_emb = not np.all(np.isnan(
        next(iter(pair_results.values()))["R_embedding_dist_per_layer"]
    ))
    n_panels = 4 if has_emb else 3
    L = bundles[0].L_total
    layer_indices = np.arange(L)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5),
                             sharey=True)
    pair_colors = SEED_COLORS * 2

    titles = ["Aligned (activation-space, per-layer Q_t)"]
    keys = ["R_aligned_dist_per_layer"]
    if has_emb:
        titles.append("Embedding-single-Q baseline")
        keys.append("R_embedding_dist_per_layer")
    titles.append("Identity baseline (no alignment)")
    keys.append("R_identity_dist_per_layer")
    titles.append("Random-orthogonal baseline (per-layer)")
    keys.append("R_random_dist_per_layer")

    for ax_idx, (title, key) in enumerate(zip(titles, keys)):
        ax = axes[ax_idx]
        for (i, j), data in pair_results.items():
            label = f"{bundles[i].label}→{bundles[j].label}"
            color_idx = (i * len(bundles) + j) % len(pair_colors)
            col = pair_colors[color_idx]
            ax.plot(layer_indices, data[key],
                    marker="o", markersize=3, lw=1.4, color=col, label=label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer state t")
        ax.set_yscale("log")
    axes[0].set_ylabel("Frobenius distance ‖R_A_transported − R_B‖")
    axes[0].legend(loc="lower right", fontsize=7, ncol=2)

    fig.suptitle(
        "Cross-seed R-matrix Frobenius distance per layer (log scale)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------
def write_summary(
    bundles: List[SeedBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    rho_matrix: np.ndarray,
    output_path: str,
):
    """Pretty text summary with Phase-2 readiness verdict."""
    n = len(bundles)
    pairs = list(pair_results.keys())
    has_emb = not np.all(np.isnan(
        pair_results[pairs[0]]["R_embedding_dist_per_layer"]
    ))

    # Per-pair scalars.
    rhos = [pair_results[p]["mean_rho_t"] for p in pairs]
    aln_R = [float(pair_results[p]["R_aligned_dist_per_layer"].sum()) for p in pairs]
    id_R = [float(pair_results[p]["R_identity_dist_per_layer"].sum()) for p in pairs]
    rnd_R = [float(pair_results[p]["R_random_dist_per_layer"].sum()) for p in pairs]
    if has_emb:
        emb_R = [float(pair_results[p]["R_embedding_dist_per_layer"].sum())
                 for p in pairs]

    aln_angle = [float(pair_results[p]["angles_aligned"].mean()) for p in pairs]
    id_angle = [float(pair_results[p]["angles_identity"].mean()) for p in pairs]
    rnd_angle = [float(pair_results[p]["angles_random"].mean()) for p in pairs]
    if has_emb:
        emb_angle = [float(pair_results[p]["angles_embedding"].mean())
                     for p in pairs]

    mean_rho = float(np.mean(rhos))
    mean_aln_R = float(np.mean(aln_R))
    mean_id_R = float(np.mean(id_R))
    mean_rnd_R = float(np.mean(rnd_R))
    mean_aln_ang = float(np.mean(aln_angle))
    mean_id_ang = float(np.mean(id_angle))
    mean_rnd_ang = float(np.mean(rnd_angle))
    aln_vs_rnd = mean_aln_R / max(mean_rnd_R, 1e-30)
    aln_vs_id = mean_aln_R / max(mean_id_R, 1e-30)

    with open(output_path, "w") as f:
        title = (f"Cross-seed alignment check — activation-space "
                 f"({n} seeds: {', '.join(b.label for b in bundles)})")
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        f.write(
            "Per-layer Procrustes alignment via shared-input activations.\n"
            "For each ordered pair (A, B), each seed processes the same\n"
            "FineWeb-Edu held-out chunks in the same order. For each layer\n"
            "t we then find the orthogonal Q_t minimizing\n"
            "  ‖X_t^(A) Q_t − X_t^(B)‖_F,\n"
            "and use that per-layer Q_t to transport R_A[t] into B's\n"
            "coordinate frame before comparing to R_B[t].\n\n"
        )
        f.write(
            f"Pilot activations: {bundles[0].N:,} per layer × "
            f"{bundles[0].L_total} layers × {bundles[0].H} dim.\n"
            f"Activation cache: run_dir/aligned_activations.npy "
            f"per seed.\n\n"
        )
        f.write("Baselines:\n")
        f.write("  identity      = no alignment (Q_t = I for all t)\n")
        f.write("  random        = Haar-uniform random Q_t, independent per layer\n")
        if has_emb:
            f.write("  embedding-Q   = single Q from embedding-space Procrustes\n"
                    "                  (the previous v1 approach)\n")
        f.write("\n")

        # Per-pair ρ_t pairwise heatmap as a table.
        f.write("=== Mean ρ_t across layers, per pair ===\n\n")
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
                    f.write(f"  {rho_matrix[i, j]:>12.4f}")
            f.write("\n")
        f.write(f"\n   mean off-diagonal ρ_t = {mean_rho:.4f}\n")
        f.write(
            f"   (small = residual streams correspond well after per-layer\n"
            f"    rotation; typical thresholds same as for embedding-space:\n"
            f"    < 0.10 excellent, < 0.20 good, > 0.30 questionable)\n\n"
        )

        # R-matrix distance table.
        f.write("=== Per-pair R-matrix distance (summed over layers) ===\n\n")
        if has_emb:
            header = (
                f"   {'pair':>16}  {'aligned':>10}  {'embedding':>10}  "
                f"{'identity':>10}  {'random':>10}  {'aln/rnd':>9}  "
                f"{'aln/id':>9}  {'aln/emb':>9}"
            )
            f.write(header + "\n")
            f.write("   " + "-" * (len(header) - 3) + "\n")
            for k, p in enumerate(pairs):
                i, j = p
                label = f"{bundles[i].label}→{bundles[j].label}"
                f.write(
                    f"   {label:>16}  "
                    f"{aln_R[k]:>10.3f}  {emb_R[k]:>10.3f}  "
                    f"{id_R[k]:>10.3f}  {rnd_R[k]:>10.3f}  "
                    f"{aln_R[k] / max(rnd_R[k], 1e-30):>9.4f}  "
                    f"{aln_R[k] / max(id_R[k], 1e-30):>9.4f}  "
                    f"{aln_R[k] / max(emb_R[k], 1e-30):>9.4f}\n"
                )
            f.write("   " + "-" * (len(header) - 3) + "\n")
            mean_emb_R = float(np.mean(emb_R))
            f.write(
                f"   {'mean':>16}  {mean_aln_R:>10.3f}  "
                f"{mean_emb_R:>10.3f}  "
                f"{mean_id_R:>10.3f}  {mean_rnd_R:>10.3f}  "
                f"{aln_vs_rnd:>9.4f}  {aln_vs_id:>9.4f}  "
                f"{mean_aln_R / max(mean_emb_R, 1e-30):>9.4f}\n\n"
            )
        else:
            header = (
                f"   {'pair':>16}  {'aligned':>10}  {'identity':>10}  "
                f"{'random':>10}  {'aln/rnd':>9}  {'aln/id':>9}"
            )
            f.write(header + "\n")
            f.write("   " + "-" * (len(header) - 3) + "\n")
            for k, p in enumerate(pairs):
                i, j = p
                label = f"{bundles[i].label}→{bundles[j].label}"
                f.write(
                    f"   {label:>16}  "
                    f"{aln_R[k]:>10.3f}  "
                    f"{id_R[k]:>10.3f}  {rnd_R[k]:>10.3f}  "
                    f"{aln_R[k] / max(rnd_R[k], 1e-30):>9.4f}  "
                    f"{aln_R[k] / max(id_R[k], 1e-30):>9.4f}\n"
                )
            f.write("   " + "-" * (len(header) - 3) + "\n")
            f.write(
                f"   {'mean':>16}  {mean_aln_R:>10.3f}  "
                f"{mean_id_R:>10.3f}  {mean_rnd_R:>10.3f}  "
                f"{aln_vs_rnd:>9.4f}  {aln_vs_id:>9.4f}\n\n"
            )

        # Angle table.
        f.write("=== Per-pair mean principal angle (top-10), degrees ===\n\n")
        if has_emb:
            header = (
                f"   {'pair':>16}  {'aligned':>10}  {'embedding':>10}  "
                f"{'identity':>10}  {'random':>10}"
            )
        else:
            header = (
                f"   {'pair':>16}  {'aligned':>10}  "
                f"{'identity':>10}  {'random':>10}"
            )
        f.write(header + "\n")
        f.write("   " + "-" * (len(header) - 3) + "\n")
        for k, p in enumerate(pairs):
            i, j = p
            label = f"{bundles[i].label}→{bundles[j].label}"
            if has_emb:
                f.write(
                    f"   {label:>16}  {aln_angle[k]:>10.2f}  "
                    f"{emb_angle[k]:>10.2f}  "
                    f"{id_angle[k]:>10.2f}  {rnd_angle[k]:>10.2f}\n"
                )
            else:
                f.write(
                    f"   {label:>16}  {aln_angle[k]:>10.2f}  "
                    f"{id_angle[k]:>10.2f}  {rnd_angle[k]:>10.2f}\n"
                )
        f.write("   " + "-" * (len(header) - 3) + "\n")
        if has_emb:
            mean_emb_ang = float(np.mean(emb_angle))
            f.write(
                f"   {'mean':>16}  {mean_aln_ang:>10.2f}  "
                f"{mean_emb_ang:>10.2f}  "
                f"{mean_id_ang:>10.2f}  {mean_rnd_ang:>10.2f}\n\n"
            )
        else:
            f.write(
                f"   {'mean':>16}  {mean_aln_ang:>10.2f}  "
                f"{mean_id_ang:>10.2f}  {mean_rnd_ang:>10.2f}\n\n"
            )

        # Verdict.
        f.write("=== Phase-2 readiness assessment ===\n\n")
        verdicts = []
        if mean_rho < 0.10:
            verdicts.append(("ρ_t", "PASS", f"mean ρ_t = {mean_rho:.3f} < 0.10"))
        elif mean_rho < 0.20:
            verdicts.append(("ρ_t", "MARGINAL",
                             f"mean ρ_t = {mean_rho:.3f} in [0.10, 0.20)"))
        else:
            verdicts.append(("ρ_t", "FAIL",
                             f"mean ρ_t = {mean_rho:.3f} ≥ 0.20"))
        if aln_vs_rnd < 0.25:
            verdicts.append(("aligned/random R", "PASS",
                             f"ratio = {aln_vs_rnd:.3f} < 0.25"))
        elif aln_vs_rnd < 0.50:
            verdicts.append(("aligned/random R", "MARGINAL",
                             f"ratio = {aln_vs_rnd:.3f} in [0.25, 0.50)"))
        else:
            verdicts.append(("aligned/random R", "FAIL",
                             f"ratio = {aln_vs_rnd:.3f} ≥ 0.50"))
        if aln_vs_id < 0.80:
            verdicts.append(("aligned/identity R", "PASS",
                             f"ratio = {aln_vs_id:.3f} < 0.80"))
        elif aln_vs_id < 1.0:
            verdicts.append(("aligned/identity R", "MARGINAL",
                             f"ratio = {aln_vs_id:.3f} (alignment barely helps)"))
        else:
            verdicts.append(("aligned/identity R", "FAIL",
                             f"ratio = {aln_vs_id:.3f} ≥ 1.0 "
                             f"(alignment makes it worse)"))
        if mean_aln_ang < 30.0:
            verdicts.append(("aligned angle", "PASS",
                             f"mean angle = {mean_aln_ang:.1f}° < 30°"))
        elif mean_aln_ang < 60.0:
            verdicts.append(("aligned angle", "MARGINAL",
                             f"mean angle = {mean_aln_ang:.1f}° in [30°, 60°)"))
        else:
            verdicts.append(("aligned angle", "FAIL",
                             f"mean angle = {mean_aln_ang:.1f}° ≥ 60°"))

        for criterion, verdict, detail in verdicts:
            f.write(f"  [{verdict:>8}] {criterion}: {detail}\n")
        f.write("\n")

        if all(v[1] == "PASS" for v in verdicts):
            f.write("  OVERALL: ✅ ACTIVATION-SPACE ALIGNMENT WORKING.\n"
                    "  Phase 2 alignment procedure validated on cross-seed\n"
                    "  (within-variant) data. Use per-layer activation-space\n"
                    "  alignment for cross-variant Phase 2 comparisons.\n")
        elif any(v[1] == "FAIL" for v in verdicts):
            f.write("  OVERALL: ⚠️  ACTIVATION-SPACE ALIGNMENT STILL\n"
                    "  PROBLEMATIC. At least one criterion failed. Phase 2\n"
                    "  may need an even richer alignment procedure (e.g.,\n"
                    "  joint optimization of Q_t plus a per-layer scaling,\n"
                    "  or alignment in a learned latent space).\n")
        else:
            f.write("  OVERALL: 🟡 ACTIVATION-SPACE ALIGNMENT MARGINAL.\n"
                    "  Phase 2 can proceed but cross-variant alignment\n"
                    "  residuals should be interpreted cautiously.\n")


def write_summary_csv(
    bundles: List[SeedBundle],
    pair_results: Dict[Tuple[int, int], Dict],
    output_path: str,
):
    """Per-pair scalar summary, CSV."""
    pairs = list(pair_results.keys())
    has_emb = not np.all(np.isnan(
        pair_results[pairs[0]]["R_embedding_dist_per_layer"]
    ))
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "seed_A", "seed_B", "mean_rho_t",
            "R_aligned_sum", "R_identity_sum", "R_random_sum",
            "aligned_over_random", "aligned_over_identity",
            "mean_angle_aligned_deg", "mean_angle_identity_deg",
            "mean_angle_random_deg",
        ]
        if has_emb:
            header += [
                "R_embedding_sum", "aligned_over_embedding",
                "mean_angle_embedding_deg",
            ]
        writer.writerow(header)
        for (i, j), data in pair_results.items():
            aln_R = float(data["R_aligned_dist_per_layer"].sum())
            id_R = float(data["R_identity_dist_per_layer"].sum())
            rnd_R = float(data["R_random_dist_per_layer"].sum())
            row = [
                bundles[i].label, bundles[j].label, data["mean_rho_t"],
                aln_R, id_R, rnd_R,
                aln_R / max(rnd_R, 1e-30),
                aln_R / max(id_R, 1e-30),
                float(data["angles_aligned"].mean()),
                float(data["angles_identity"].mean()),
                float(data["angles_random"].mean()),
            ]
            if has_emb:
                emb_R = float(data["R_embedding_dist_per_layer"].sum())
                row += [
                    emb_R, aln_R / max(emb_R, 1e-30),
                    float(data["angles_embedding"].mean()),
                ]
            writer.writerow(row)


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Cross-seed activation-space Procrustes alignment check.",
    )
    parser.add_argument(
        "--run_dirs", nargs="+", required=True,
        help="Paths to seed run directories.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write outputs. Default: alignment_check_act/ "
             "alongside the first run_dir's parent.",
    )
    parser.add_argument(
        "--max_pilots", type=int, default=9500,
        help="Max pilot activations per seed (limited by held-out chunks × "
             "pilot positions). Default 9500 matches the analyzer's default.",
    )
    parser.add_argument(
        "--embedding_top_k", type=int, default=1000,
        help="For the embedding-single-Q baseline (which compares against\n"
             "the v1 approach), filter to the top-K tokens by per-row L2 "
             "norm. Default 1000 matches the v1 alignment_check.py setting.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        first_parent = os.path.dirname(os.path.abspath(args.run_dirs[0]))
        args.output_dir = os.path.join(first_parent, "alignment_check_act")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all seeds. This is where the (cached or fresh) inference happens.
    print(f">> Loading {len(args.run_dirs)} seed(s) and "
          f"collecting/loading activations ...")
    bundles = []
    for run_dir in args.run_dirs:
        print(f"\n   --- {run_dir} ---")
        b = SeedBundle(run_dir, max_pilots=args.max_pilots)
        bundles.append(b)
        print(f"   {b}")
    bundles.sort(key=lambda b: b.fs.seed)

    # Sanity checks.
    L0, H0, N0 = bundles[0].L_total, bundles[0].H, bundles[0].N
    for b in bundles[1:]:
        if b.L_total != L0 or b.H != H0:
            raise ValueError(
                f"{b.run_dir} has L={b.L_total}, H={b.H}; expected "
                f"L={L0}, H={H0}."
            )
        if b.N != N0:
            print(f"⚠️  Pilot count mismatch: {b.label} has N={b.N}, "
                  f"first seed has N={N0}. Per-row alignment requires "
                  f"identical inputs in the same order. Truncating to "
                  f"min(N).")
    # Truncate all to min N.
    N_min = min(b.N for b in bundles)
    if N_min != N0:
        for b in bundles:
            b.activations = b.activations[:, :N_min, :]
            b.N = N_min

    # Compute pairwise alignments.
    print(f"\n>> Computing pairwise activation-space alignments "
          f"({len(bundles)}*{len(bundles)-1} = "
          f"{len(bundles) * (len(bundles) - 1)} ordered pairs) ...")
    pair_results = {}
    rho_matrix = np.zeros((len(bundles), len(bundles)), dtype=np.float32)
    for i, b_A in enumerate(bundles):
        for j, b_B in enumerate(bundles):
            if i == j:
                continue
            t0 = time.time()
            data = compute_pairwise_alignment(
                b_A, b_B, embedding_top_k=args.embedding_top_k,
            )
            elapsed = time.time() - t0
            pair_results[(i, j)] = data
            rho_matrix[i, j] = data["mean_rho_t"]
            aln_R_sum = data["R_aligned_dist_per_layer"].sum()
            mean_ang = data["angles_aligned"].mean()
            print(f"   {b_A.label} → {b_B.label}: "
                  f"mean ρ_t = {data['mean_rho_t']:.4f}, "
                  f"R-dist (aligned) = {aln_R_sum:.2f}, "
                  f"mean angle = {mean_ang:.1f}°  [{elapsed:.1f}s]")

    # Write outputs.
    print(f"\n>> Writing outputs to {args.output_dir} ...")
    txt = os.path.join(args.output_dir, "alignment_act_summary.txt")
    write_summary(bundles, pair_results, rho_matrix, txt)
    print(f"   ↳ {txt}")
    csv_path = os.path.join(args.output_dir, "alignment_act_summary.csv")
    write_summary_csv(bundles, pair_results, csv_path)
    print(f"   ↳ {csv_path}")
    p1 = os.path.join(args.output_dir, "pairwise_mean_rho_t.png")
    plot_pairwise_mean_rho_t(bundles, rho_matrix, p1)
    print(f"   ↳ {p1}")
    p2 = os.path.join(args.output_dir, "rho_t_per_layer.png")
    plot_rho_t_per_layer(bundles, pair_results, p2)
    print(f"   ↳ {p2}")
    p3 = os.path.join(args.output_dir, "perlayer_angles.png")
    plot_perlayer_angles(bundles, pair_results, p3)
    print(f"   ↳ {p3}")
    p4 = os.path.join(args.output_dir, "perlayer_R_distances.png")
    plot_perlayer_R_distances(bundles, pair_results, p4)
    print(f"   ↳ {p4}")

    print()
    with open(txt) as f:
        print(f.read())

    print(">> ✅ Activation-space alignment check complete.")


if __name__ == "__main__":
    main()
    