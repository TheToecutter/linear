"""
Per-token covariance similarity for the linear-Gaussian baseline check.

Question: does the linear-Gaussian framework's prediction that the
within-token noise covariance Sigma_t is token-independent actually
hold in the forward view of the trained residual-stream ensemble?

If yes: the per-token forward bundles all spread in the same dominant
directions at each layer, and the differences in their per-coordinate
spread (fig5b) are just rescalings of the same underlying covariance
structure. The linear-Gaussian model is then a faithful description
of the within-bundle geometry.

If no: different input tokens drive context-injection along different
directions of the residual stream. The model is doing something
genuinely token-specific that goes beyond a token-independent noise
process — a deviation from the linear-Gaussian baseline that has to
be acknowledged.

The test: for every pair of top-20 forward tokens, compute principal
angles between the top-k singular subspaces of their per-token
covariances at each layer. Compare to two null baselines:

  - Self-consistency: split one token's pilots in half, compute the
    principal angle between the two halves' subspaces. This captures
    the floor due to sample noise alone.
  - Random subset: pair up two random pilot subsets (regardless of
    token), compute principal angles. This is the ceiling — what
    "no covariance relationship" looks like.

If trained-pair angles sit near the self-consistency floor, the
linear-Gaussian prediction holds. If they sit near the random-subset
ceiling, it doesn't.

Usage:
    python3 covariance_similarity.py [--run-dir PATH] [--step STEP]
                                     [--seed SEED] [--k K]

Defaults: ../phase1_runs_gelu, step 24000, seed 0, k=20 components.

Output:
    ../phase1_runs_gelu/figures/fig11_per_token_covariance_angles.png
    ../phase1_runs_gelu/multiview/covariance/
        principal_angles_seed<S>_step<STEP>.npz
"""

from __future__ import annotations

import argparse
import itertools
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiview import load_augmented_activations, select_token_set


def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(run_dir, "multiview", f"seed_{seed}",
                        f"augmented_step_{step:08d}.npz")


def top_k_subspace(X: np.ndarray, k: int) -> np.ndarray:
    """Return the top-k left singular vectors of (X - mean(X)).

    Args:
        X: (N, H) array of points.
        k: how many principal directions to keep.

    Returns:
        U_k: (H, k) orthonormal columns, the top-k principal directions
             of the centered point cloud.
    """
    X_centered = X - X.mean(axis=0, keepdims=True)
    # We want the right singular vectors of X_centered (which are the
    # principal directions of the covariance). SVD of (N, H) gives
    # X = U @ diag(s) @ Vt, with Vt of shape (min(N,H), H). The first
    # k rows of Vt are the top-k principal directions.
    _, _, Vt = np.linalg.svd(X_centered.astype(np.float64),
                             full_matrices=False)
    k_use = min(k, Vt.shape[0])
    return Vt[:k_use].T  # (H, k_use)


def principal_angles(U_a: np.ndarray, U_b: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces.

    Args:
        U_a: (H, k_a) orthonormal columns spanning subspace A.
        U_b: (H, k_b) orthonormal columns spanning subspace B.

    Returns:
        angles: 1D array of min(k_a, k_b) angles in radians, sorted
                ascending (smallest = best alignment direction).
    """
    # Singular values of U_a^T U_b are the cosines of the principal
    # angles. Clip for numerical safety (acos of slightly-over-1 is NaN).
    M = U_a.T @ U_b
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.arccos(s)


def compute_per_token_subspaces(states: np.ndarray, input_ids: np.ndarray,
                                token_set: np.ndarray, k: int
                                ) -> Dict[int, List[np.ndarray]]:
    """For each token in token_set, compute its top-k principal subspace
    at every layer."""
    L = states.shape[0]
    out: Dict[int, List[np.ndarray]] = {}
    for tok in token_set:
        mask = input_ids == int(tok)
        n = int(mask.sum())
        if n < k + 5:
            # Skip tokens with too few pilots for a reliable subspace.
            continue
        per_layer = []
        for t in range(L):
            per_layer.append(top_k_subspace(states[t, mask, :], k))
        out[int(tok)] = per_layer
    return out


def compute_pairwise_angles(subspaces: Dict[int, List[np.ndarray]]
                            ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """For each pair of tokens, return their principal angles per layer.

    Returns:
        pairs: list of (tok_a, tok_b) tuples.
        angles: (n_pairs, L, k) array of angles in radians.
    """
    tokens = sorted(subspaces.keys())
    L = len(next(iter(subspaces.values())))
    k = next(iter(subspaces.values()))[0].shape[1]
    pairs: List[Tuple[int, int]] = []
    angles_list = []
    for a, b in itertools.combinations(tokens, 2):
        pair_angles = np.zeros((L, k))
        for t in range(L):
            ang = principal_angles(subspaces[a][t], subspaces[b][t])
            pair_angles[t, :len(ang)] = ang
        pairs.append((a, b))
        angles_list.append(pair_angles)
    return pairs, np.stack(angles_list, axis=0)


def compute_self_consistency_null(states: np.ndarray, input_ids: np.ndarray,
                                   token_set: np.ndarray, k: int,
                                   rng: np.random.Generator,
                                   n_reps: int = 5
                                   ) -> np.ndarray:
    """For each token, split its pilots into two random halves and
    compute principal angles between the half-bundles' subspaces.
    Returns (n_tokens, n_reps, L, k) of angles."""
    L = states.shape[0]
    angles_list = []
    for tok in token_set:
        mask = input_ids == int(tok)
        idx = np.where(mask)[0]
        n = idx.size
        if n < 2 * (k + 5):
            # Too few pilots to split.
            continue
        rep_angles = np.zeros((n_reps, L, k))
        for r in range(n_reps):
            perm = rng.permutation(idx)
            half = n // 2
            i1, i2 = perm[:half], perm[half:2 * half]
            for t in range(L):
                U1 = top_k_subspace(states[t, i1, :], k)
                U2 = top_k_subspace(states[t, i2, :], k)
                ang = principal_angles(U1, U2)
                rep_angles[r, t, :len(ang)] = ang
        angles_list.append(rep_angles)
    if not angles_list:
        return np.zeros((0, n_reps, L, k))
    return np.stack(angles_list, axis=0)


def compute_random_subset_null(states: np.ndarray, k: int,
                               rng: np.random.Generator,
                               subset_size: int, n_reps: int = 10
                               ) -> np.ndarray:
    """Pair up two random pilot subsets (not conditioned on tokens),
    compute principal angles between their subspaces.
    Returns (n_reps, L, k) of angles."""
    L, N, _ = states.shape
    rep_angles = np.zeros((n_reps, L, k))
    for r in range(n_reps):
        perm = rng.permutation(N)
        i1 = perm[:subset_size]
        i2 = perm[subset_size:2 * subset_size]
        for t in range(L):
            U1 = top_k_subspace(states[t, i1, :], k)
            U2 = top_k_subspace(states[t, i2, :], k)
            ang = principal_angles(U1, U2)
            rep_angles[r, t, :len(ang)] = ang
    return rep_angles


def plot_angles(layers: np.ndarray, pair_angles: np.ndarray,
                self_null: np.ndarray, random_null: np.ndarray,
                seed: int, step: int, k: int,
                output_path: str) -> None:
    """Two-panel figure.

    Left: per-layer first principal angle (smallest = best alignment
    direction) with trained-pair mean curve and two reference null bands.

    Right: per-layer median principal angle across the top-k components.

    The key visual claim: if the trained-pair curve overlaps the
    self-consistency null band, the per-token covariances are
    statistically indistinguishable from "same covariance with sample
    noise" — the linear-Gaussian token-independence prediction holds.
    If the trained-pair curve sits clearly above the self-consistency
    band, the per-token covariances are genuinely different beyond
    sample noise — the prediction fails by that much.

    The random-subset null is shown for context: it measures the
    sample-noise scale for the marginal covariance (random pilot subsets
    all estimate the same all-to-all covariance, modulo sampling). It
    should sit near the self-consistency null when both are healthy.
    """
    fig, (ax_first, ax_median) = plt.subplots(1, 2, figsize=(15, 6),
                                              sharey=True)

    deg = lambda a: np.degrees(a)

    pair_first = deg(pair_angles[:, :, 0])
    self_first = deg(self_null[:, :, :, 0]) if self_null.size else None
    rand_first = deg(random_null[:, :, 0])

    pair_med = deg(np.median(pair_angles, axis=-1))
    self_med = deg(np.median(self_null, axis=-1)) if self_null.size else None
    rand_med = deg(np.median(random_null, axis=-1))

    for ax, pair_data, self_data, rand_data, title in [
        (ax_first, pair_first, self_first, rand_first,
         "First principal angle (smallest, best alignment direction)"),
        (ax_median, pair_med, self_med, rand_med,
         f"Median across top-{k} principal angles"),
    ]:
        # Trained pairs.
        for i in range(pair_data.shape[0]):
            ax.plot(layers, pair_data[i], "-", color="C0", alpha=0.18, lw=0.7)
        ax.plot(layers, pair_data.mean(axis=0), "-o", color="C0",
                lw=2.0, markersize=4,
                label=f"trained token pairs (mean of {pair_data.shape[0]})",
                zorder=5)

        # Self-consistency null: how big the principal angle is purely
        # from sample noise when the underlying covariance is identical.
        if self_data is not None and self_data.size:
            self_flat = self_data.reshape(-1, self_data.shape[-1])
            ax.fill_between(layers,
                            np.percentile(self_flat, 5, axis=0),
                            np.percentile(self_flat, 95, axis=0),
                            color="C2", alpha=0.20,
                            label="self-consistency null (sample noise, same cov)")
            ax.plot(layers, self_flat.mean(axis=0), "--", color="C2",
                    lw=1.4)

        # Random-subset null: principal angles between two random pilot
        # subsets. Measures sample noise for the marginal covariance.
        ax.plot(layers, rand_data.mean(axis=0), "--", color="C3", lw=1.4,
                label="random-subset null (marginal cov, sample noise)")
        ax.fill_between(layers,
                        np.percentile(rand_data, 5, axis=0),
                        np.percentile(rand_data, 95, axis=0),
                        color="C3", alpha=0.12)

        ax.set_xlabel("layer state index t")
        ax.set_xticks(layers)
        ax.set_xlim(-0.3, layers[-1] + 0.3)
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)

    ax_first.set_ylabel("principal angle (degrees)")
    # y-range: include enough headroom for whatever the data shows.
    y_max_data = max(
        np.percentile(pair_first, 99),
        np.percentile(pair_med, 99),
    )
    y_max = max(y_max_data * 1.15, 30)
    ax_first.set_ylim(0, y_max)
    # Only draw the "orthogonal" reference if it's within the plot area.
    if y_max >= 90:
        ax_first.axhline(90, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax_first.text(layers[-1] + 0.3, 90, "  90° (orthogonal)", color="gray",
                      fontsize=8, va="center", ha="left")
    ax_first.legend(loc="best", fontsize=8, framealpha=0.92)

    fig.suptitle(f"Per-token covariance subspace alignment in the forward view\n"
                 f"(seed {seed}, step {step}, top-{k} principal directions). "
                 f"Trained-pair line at the self-consistency band = "
                 f"token-independent covariance.")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_summary(layers: np.ndarray, pair_angles: np.ndarray,
                  self_null: np.ndarray, random_null: np.ndarray) -> None:
    """Compact table of per-layer angles."""
    deg = lambda a: np.degrees(a)
    pair_first_mean = deg(pair_angles[:, :, 0].mean(axis=0))
    pair_med_mean = deg(np.median(pair_angles, axis=-1).mean(axis=0))
    if self_null.size:
        self_first_mean = deg(self_null[:, :, :, 0].mean(axis=(0, 1)))
    else:
        self_first_mean = np.full(layers.size, np.nan)
    rand_first_mean = deg(random_null[:, :, 0].mean(axis=0))
    rand_med_mean = deg(np.median(random_null, axis=-1).mean(axis=0))

    print()
    print(f"Per-layer principal angles (degrees):")
    print(f"{'t':>3} {'pair 1st':>10} {'self 1st':>10} {'rand 1st':>10} "
          f"{'pair med':>10} {'rand med':>10}")
    print("-" * 60)
    for i, t in enumerate(layers):
        print(f"{t:>3} {pair_first_mean[i]:>10.2f} {self_first_mean[i]:>10.2f} "
              f"{rand_first_mean[i]:>10.2f} {pair_med_mean[i]:>10.2f} "
              f"{rand_med_mean[i]:>10.2f}")
    print()
    # Compute overall averages used in the interpretation block.
    pair_med_overall = float(pair_med_mean.mean())
    self_first_overall = (float(self_first_mean.mean())
                          if not np.all(np.isnan(self_first_mean))
                          else None)
    rand_med_overall = float(rand_med_mean.mean())

    print(f"Overall (averaged across layers):")
    print(f"  trained pair median angle: {pair_med_overall:.2f}°")
    if self_first_overall is not None:
        print(f"  self-consistency null:     {self_first_overall:.2f}°  "
              f"(sample noise when cov is identical)")
    print(f"  random-subset null:        {rand_med_overall:.2f}°  "
          f"(sample noise for marginal cov)")
    if self_first_overall is not None:
        print()
        print("Interpretation:")
        if pair_med_overall <= self_first_overall * 1.5:
            print("  Trained-pair angle is near the self-consistency null:")
            print("  the per-token covariances are statistically indistinguishable")
            print("  from 'identical covariance + sample noise'. The linear-Gaussian")
            print("  token-independence prediction holds.")
        elif pair_med_overall <= self_first_overall * 3:
            print("  Trained-pair angle moderately exceeds the self-consistency null:")
            print("  there is some token-dependent structure in the covariance, but")
            print("  it's not large compared to sample-noise effects. The")
            print("  linear-Gaussian prediction approximately holds.")
        else:
            print("  Trained-pair angle substantially exceeds the self-consistency")
            print("  null: per-token covariances are genuinely different beyond")
            print("  sample noise. The linear-Gaussian token-independence")
            print("  prediction fails, in this sense.")


def run_single_analysis(run_dir: str, seed: int, step: int, k: int,
                         verbose: bool = True
                         ) -> Dict[str, np.ndarray]:
    """Run the per-token covariance similarity analysis for one
    (seed, step, k) combination.

    Returns a dict with keys 'layers', 'pair_angles', 'self_null',
    'random_null', 'pair_keys' for downstream aggregation.
    """
    if verbose:
        print(f"  Loading seed {seed} step {step} activations ...")
    payload = load_augmented_activations(augmented_path(run_dir, seed, step))
    states = payload["states"]
    input_ids = payload["input_ids"]

    fset = select_token_set(input_ids, view="forward", top_k=20, min_count=10)
    if verbose:
        print(f"    {states.shape[1]} pilots, "
              f"forward token set: {fset.token_ids.size} tokens, "
              f"counts {fset.counts.min()}-{fset.counts.max()}")

    subspaces = compute_per_token_subspaces(states, input_ids,
                                            fset.token_ids, k)
    pairs, pair_angles = compute_pairwise_angles(subspaces)

    rng = np.random.default_rng(42)
    self_null = compute_self_consistency_null(states, input_ids,
                                              fset.token_ids, k, rng)
    subset_size = int(np.median(fset.counts))
    random_null = compute_random_subset_null(states, k, rng,
                                              subset_size=subset_size,
                                              n_reps=10)

    L = states.shape[0]
    layers = np.arange(L)
    return {
        "layers": layers,
        "pair_keys": np.array(pairs),
        "pair_angles": pair_angles,
        "self_null": self_null,
        "random_null": random_null,
        "k": np.int32(k),
        "n_tokens_used": np.int32(len(subspaces)),
    }


def checkpoints_in_seed(run_dir: str, seed: int) -> List[int]:
    """List checkpoint steps available for a seed under multiview/."""
    seed_dir = os.path.join(run_dir, "multiview", f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return []
    steps = []
    for fname in os.listdir(seed_dir):
        if fname.startswith("augmented_step_") and fname.endswith(".npz"):
            try:
                step = int(fname.removeprefix("augmented_step_")
                           .removesuffix(".npz"))
                steps.append(step)
            except ValueError:
                continue
    return sorted(steps)


def seeds_available(run_dir: str) -> List[int]:
    mv_root = os.path.join(run_dir, "multiview")
    if not os.path.isdir(mv_root):
        return []
    seeds = []
    for d in os.listdir(mv_root):
        if d.startswith("seed_") and d[5:].isdigit():
            seeds.append(int(d[5:]))
    return sorted(seeds)


# ----------------------------------------------------------------------
# Sweep 1: k-robustness check.
# ----------------------------------------------------------------------
def run_k_sweep(args) -> None:
    """Run the analysis at multiple k values to verify the finding is
    robust to the choice of subspace dimensionality.

    If the per-token covariance structure is genuinely token-specific
    (not a sample-noise artifact in the tail), the pair-vs-self gap
    should persist across reasonable k values. If it's a tail effect,
    smaller k would show a much smaller gap.
    """
    k_values = [5, 10, 20, 50]
    print(f"\n== k-sweep: {k_values} ==")
    results = {}
    for k in k_values:
        # k=50 might exceed available samples for low-count tokens; in
        # that case top_k_subspace returns fewer than k columns and
        # principal_angles handles it gracefully.
        if k > 50:
            print(f"  k={k} is large; some tokens may yield smaller subspaces")
        print(f"  k = {k} ...")
        r = run_single_analysis(args.run_dir, args.seed, args.step, k,
                                 verbose=False)
        results[k] = r
        # Per-layer mean of the median angle: the single number that
        # most cleanly summarizes how token-specific the covariance is.
        pair_med_per_layer = np.degrees(
            np.median(r["pair_angles"], axis=-1).mean(axis=0))
        self_first_per_layer = (
            np.degrees(r["self_null"][:, :, :, 0].mean(axis=(0, 1)))
            if r["self_null"].size else np.full(r["layers"].size, np.nan)
        )
        print(f"    pair median (avg across layers): "
              f"{pair_med_per_layer.mean():.1f}° | "
              f"self first: {self_first_per_layer.mean():.1f}°")

    # Plot: one panel per k, showing the median curve vs the
    # self-consistency band for each.
    fig, axes = plt.subplots(1, len(k_values), figsize=(4.2 * len(k_values), 5.2),
                              sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    for ax, k in zip(axes, k_values):
        r = results[k]
        layers = r["layers"]
        deg = np.degrees
        pair_med = deg(np.median(r["pair_angles"], axis=-1))  # (n_pairs, L)

        for i in range(pair_med.shape[0]):
            ax.plot(layers, pair_med[i], "-", color="C0", alpha=0.12, lw=0.6)
        ax.plot(layers, pair_med.mean(axis=0), "-o", color="C0",
                lw=2.0, markersize=4,
                label=f"trained pair median (k={k})",
                zorder=5)

        if r["self_null"].size:
            self_med = deg(np.median(r["self_null"], axis=-1))
            self_flat = self_med.reshape(-1, self_med.shape[-1])
            ax.fill_between(layers,
                            np.percentile(self_flat, 5, axis=0),
                            np.percentile(self_flat, 95, axis=0),
                            color="C2", alpha=0.20,
                            label="self-consistency null")
            ax.plot(layers, self_flat.mean(axis=0), "--", color="C2", lw=1.3)

        rand_med = deg(np.median(r["random_null"], axis=-1))
        ax.plot(layers, rand_med.mean(axis=0), "--", color="C3", lw=1.3,
                label="random-subset null")

        ax.set_xlabel("layer state index t")
        ax.set_title(f"k = {k}")
        ax.set_xticks(layers)
        ax.set_xlim(-0.3, layers[-1] + 0.3)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("median principal angle (degrees)")
            ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    fig.suptitle(f"k-robustness of per-token covariance subspace alignment "
                 f"(seed {args.seed}, step {args.step})")
    fig.tight_layout()
    out_path = os.path.join(args.figures_dir, "fig11b_k_sweep.png")
    os.makedirs(args.figures_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Sweep 2: cross-seed reproducibility check.
# ----------------------------------------------------------------------
def run_all_seeds(args) -> None:
    """Run the analysis for every available seed at the final
    checkpoint. The per-token-covariance finding should reproduce
    across seeds if it's a property of trained transformers, not of
    one particular initialization."""
    seeds = seeds_available(args.run_dir)
    print(f"\n== cross-seed sweep: seeds {seeds} ==")
    results = {}
    for seed in seeds:
        print(f"  seed = {seed} ...")
        r = run_single_analysis(args.run_dir, seed, args.step, args.k,
                                 verbose=False)
        results[seed] = r
        pair_med = np.degrees(np.median(r["pair_angles"], axis=-1).mean(axis=0))
        self_first = (np.degrees(r["self_null"][:, :, :, 0].mean(axis=(0, 1)))
                      if r["self_null"].size else None)
        print(f"    pair median (avg): {pair_med.mean():.1f}°"
              + (f" | self first: {self_first.mean():.1f}°"
                 if self_first is not None else ""))

    # Plot: overlay all seeds' median curves on a single panel.
    fig, (ax_first, ax_med) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    cmap = plt.get_cmap("tab10")
    deg = np.degrees

    for ax, angle_idx_or_med, title in [
        (ax_first, "first",
         "First principal angle (best alignment direction)"),
        (ax_med, "median",
         f"Median across top-{args.k} principal angles"),
    ]:
        for i, seed in enumerate(seeds):
            r = results[seed]
            layers = r["layers"]
            if angle_idx_or_med == "first":
                pair_data = deg(r["pair_angles"][:, :, 0])
            else:
                pair_data = deg(np.median(r["pair_angles"], axis=-1))
            ax.plot(layers, pair_data.mean(axis=0), "-o",
                    color=cmap(i), lw=1.8, markersize=4,
                    label=f"seed {seed} (n={pair_data.shape[0]} pairs)")
            ax.fill_between(layers,
                            np.percentile(pair_data, 25, axis=0),
                            np.percentile(pair_data, 75, axis=0),
                            color=cmap(i), alpha=0.10)

        # Shared null bands: use seed 0's nulls as representative
        # (the nulls are intrinsic to the model and dataset).
        r0 = results[seeds[0]]
        layers = r0["layers"]
        if angle_idx_or_med == "first":
            self_data = (deg(r0["self_null"][:, :, :, 0])
                         if r0["self_null"].size else None)
            rand_data = deg(r0["random_null"][:, :, 0])
        else:
            self_data = (deg(np.median(r0["self_null"], axis=-1))
                         if r0["self_null"].size else None)
            rand_data = deg(np.median(r0["random_null"], axis=-1))
        if self_data is not None:
            self_flat = self_data.reshape(-1, self_data.shape[-1])
            ax.fill_between(layers,
                            np.percentile(self_flat, 5, axis=0),
                            np.percentile(self_flat, 95, axis=0),
                            color="gray", alpha=0.18,
                            label="self-consistency null (seed 0)")
            ax.plot(layers, self_flat.mean(axis=0), "--", color="gray",
                    lw=1.2)
        ax.plot(layers, rand_data.mean(axis=0), "--", color="black",
                lw=1.2, label="random-subset null (seed 0)")

        ax.set_xlabel("layer state index t")
        ax.set_xticks(layers)
        ax.set_xlim(-0.3, layers[-1] + 0.3)
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        if ax is ax_first:
            ax.set_ylabel("principal angle (degrees)")
            ax.legend(loc="best", fontsize=8, framealpha=0.92)

    fig.suptitle(f"Cross-seed reproducibility of per-token covariance "
                 f"non-alignment (step {args.step}, k={args.k})")
    fig.tight_layout()
    out_path = os.path.join(args.figures_dir, "fig11c_all_seeds.png")
    os.makedirs(args.figures_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ----------------------------------------------------------------------
# Sweep 3: training-evolution.
# ----------------------------------------------------------------------
def run_training_evolution(args) -> None:
    """Run the analysis across all checkpoints for seed 0, producing a
    heatmap of (training step, layer) showing how token-specific
    covariance emerges through training.

    If the finding is created by training, this heatmap will show the
    angles starting low (near self-consistency) at early checkpoints
    and rising toward the converged ~73° plateau by the end of training.
    If the finding is intrinsic to the architecture, it will be present
    at all checkpoints with comparable magnitude.
    """
    steps = checkpoints_in_seed(args.run_dir, args.seed)
    print(f"\n== training-evolution sweep: {len(steps)} checkpoints for "
          f"seed {args.seed} ==")
    if not steps:
        print(f"  no augmented checkpoints found for seed {args.seed}")
        return

    L = None
    pair_med_per_step = []  # list of (n_pairs, L) arrays
    self_first_per_step = []  # list of (L,) arrays
    pair_summary = []  # list of (L,) arrays — median of pair angles
    self_summary = []  # list of (L,) arrays

    for step in steps:
        print(f"  step {step} ...", end="", flush=True)
        try:
            r = run_single_analysis(args.run_dir, args.seed, step, args.k,
                                     verbose=False)
        except (FileNotFoundError, OSError) as e:
            print(f" SKIP ({type(e).__name__}: {e})")
            continue
        layers = r["layers"]
        if L is None:
            L = layers.size

        pair_med = np.degrees(np.median(r["pair_angles"], axis=-1))  # (n_pairs, L)
        self_first = (np.degrees(r["self_null"][:, :, :, 0].mean(axis=(0, 1)))
                      if r["self_null"].size else np.full(L, np.nan))
        pair_summary.append(pair_med.mean(axis=0))  # (L,)
        self_summary.append(self_first)
        print(f"  pair median (avg): {pair_med.mean():.1f}°  "
              f"self first: {self_first.mean():.1f}°")

    if not pair_summary:
        print("  no valid checkpoints; aborting")
        return

    steps_arr = np.array(steps[:len(pair_summary)])
    pair_arr = np.stack(pair_summary, axis=0)  # (n_steps, L)
    self_arr = np.stack(self_summary, axis=0)  # (n_steps, L)
    excess_arr = pair_arr - self_arr  # (n_steps, L), the "anomaly"

    # Heatmap layout: two panels.
    #   Left: pair median angle (the absolute angle).
    #   Right: excess over self-consistency null (the "anomaly").
    from matplotlib.colors import Normalize
    fig, (ax_abs, ax_excess) = plt.subplots(1, 2, figsize=(15, 6),
                                             sharey=True)

    log_steps = np.log10(steps_arr.astype(np.float64))
    edges = np.empty(steps_arr.size + 1)
    edges[1:-1] = (log_steps[:-1] + log_steps[1:]) / 2
    edges[0] = log_steps[0] - (log_steps[1] - log_steps[0]) / 2
    edges[-1] = log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2
    step_edges = 10 ** edges
    layer_edges = np.arange(L + 1) - 0.5

    # Left: absolute angle.
    im0 = ax_abs.pcolormesh(layer_edges, step_edges, pair_arr,
                             cmap="viridis",
                             norm=Normalize(vmin=0, vmax=90),
                             shading="flat")
    ax_abs.set_yscale("log")
    ax_abs.set_xticks(np.arange(L))
    ax_abs.set_xlabel("layer state index t")
    ax_abs.set_ylabel("training step")
    ax_abs.set_title("pair median angle  (absolute)")
    cbar0 = fig.colorbar(im0, ax=ax_abs, shrink=0.85)
    cbar0.set_label("median principal angle (degrees)")

    # Right: excess over null. Use a diverging colormap centered at 0
    # so "above null" is red and "near null" is white.
    vmax = max(abs(np.nanmin(excess_arr)), abs(np.nanmax(excess_arr)))
    im1 = ax_excess.pcolormesh(layer_edges, step_edges, excess_arr,
                                cmap="RdBu_r",
                                norm=Normalize(vmin=-vmax, vmax=vmax),
                                shading="flat")
    ax_excess.set_yscale("log")
    ax_excess.set_xticks(np.arange(L))
    ax_excess.set_xlabel("layer state index t")
    ax_excess.set_title(f"excess over self-consistency null  (k={args.k})")
    cbar1 = fig.colorbar(im1, ax=ax_excess, shrink=0.85)
    cbar1.set_label("pair median − self-consistency (degrees)")

    fig.suptitle(f"Training evolution of per-token covariance "
                 f"non-alignment  (seed {args.seed})")
    fig.tight_layout()
    out_path = os.path.join(args.figures_dir, "fig11d_training_evolution.png")
    os.makedirs(args.figures_dir, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Save raw data.
    out_dir = os.path.join(args.run_dir, "multiview", "covariance")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir,
                              f"training_evolution_seed{args.seed}_k{args.k}.npz")
    np.savez(data_path,
             steps=steps_arr, layers=np.arange(L),
             pair_median=pair_arr, self_first=self_arr,
             excess=excess_arr, k=np.int32(args.k))
    print(f"Saved raw data to {data_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--step", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20,
                    help="Number of principal directions to compare.")
    ap.add_argument("--figures-dir", default="../phase1_runs_gelu/figures")
    ap.add_argument("--k-sweep", action="store_true",
                    help="Run the analysis at multiple k values "
                         "(5,10,20,50) and produce fig11b.")
    ap.add_argument("--all-seeds", action="store_true",
                    help="Run the analysis on every available seed at "
                         "the final checkpoint and produce fig11c.")
    ap.add_argument("--training-evolution", action="store_true",
                    help="Run the analysis across all checkpoints for "
                         "seed 0 and produce fig11d.")
    args = ap.parse_args()

    # Dispatch.
    if args.k_sweep:
        run_k_sweep(args)
        return
    if args.all_seeds:
        run_all_seeds(args)
        return
    if args.training_evolution:
        run_training_evolution(args)
        return

    # Default: single analysis (the fig11 path).
    print(f"Loading seed {args.seed} step {args.step} activations ...")
    result = run_single_analysis(args.run_dir, args.seed, args.step, args.k)
    print(f"  {result['n_tokens_used']} tokens had enough pilots for "
          f"a top-{args.k} subspace")
    print(f"  {result['pair_keys'].shape[0]} token pairs, "
          f"angles shape {result['pair_angles'].shape}")

    print_summary(result["layers"], result["pair_angles"],
                  result["self_null"], result["random_null"])

    # Save raw data.
    out_dir = os.path.join(args.run_dir, "multiview", "covariance")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(
        out_dir, f"principal_angles_seed{args.seed}_step{args.step:08d}.npz")
    np.savez(data_path,
             layers=result["layers"],
             token_pairs=result["pair_keys"],
             pair_angles=result["pair_angles"],
             self_null=result["self_null"],
             random_null=result["random_null"],
             k=result["k"])
    print(f"\nRaw data saved to {data_path}")

    os.makedirs(args.figures_dir, exist_ok=True)
    fig_path = os.path.join(args.figures_dir,
                            "fig11_per_token_covariance_angles.png")
    plot_angles(result["layers"], result["pair_angles"],
                result["self_null"], result["random_null"],
                args.seed, args.step, args.k, fig_path)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
    