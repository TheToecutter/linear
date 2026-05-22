"""
Test 1: Is the layer-update drift well-defined as a function of position?

Background
----------
The overdamped Langevin reading of the lines-of-thought framework writes

    dx = b(x, t) dt + sigma(t) dw,

i.e., the layer increment Delta_x_t = x_{t+1} - x_t is, in expectation,
a deterministic function b(x, t) of the current position x_t and the
depth t. The "noise" is everything else.

This is a strong claim. It says that two pilots that happen to sit at
the same place at layer t should, on average, receive the same update
from block t -- regardless of which left-context they came from, which
chunk-position they live at, or which next-token they're heading toward.
If two pilots at the same x_t receive systematically different updates
because of their left-context, then the SDE drift is not a function of
position alone; it is a function of (position, hidden state we cannot
see), and the framework is a marginal fit rather than a generative
description.

Method
------
At each layer t:

  1. Reduce x_t to a low-dimensional summary by projecting onto the
     top-d principal components of the all-to-all ensemble at layer t.
     Use d = 5 by default (tunable). This is the "position bin
     coordinate" -- coarse, basis-invariant up to the seed's chosen
     rotation of the residual stream, but reproducible within a seed.

  2. Partition the pilots into K equal-population clusters in this
     d-dimensional reduced space using k-means. K = 24 by default.

  3. For each cluster c, compute the mean update mu_c = E[Delta_x_t | c]
     and the within-cluster covariance C_c = Cov[Delta_x_t | c]. The
     average within-cluster variance is V_within(t) = E_c[trace(C_c)] / H.
     The between-cluster variance is V_between(t) = Var_c[mu_c] / H,
     where Var_c is the per-coordinate variance across cluster means.

  4. The position-explained ratio is

         R^2_pos(t) = V_between(t) / (V_within(t) + V_between(t)).

     This is the fraction of layer-t update variance that the position
     binning at layer t explains. R^2_pos = 1 means the update is
     perfectly determined by binned position; R^2_pos = 0 means the
     binning explains nothing.

  5. Null baseline: randomly permute the cluster assignments and
     recompute R^2_pos. The shuffled value is the floor from finite-K
     binning artifact -- it should be near zero. The gap between the
     real R^2_pos and the shuffled R^2_pos is the test statistic.

  6. Report this profile across layers, plus a single-number summary
     (mean over layers, mean gap above shuffle null).

Interpretation
--------------
The headline quantity is the *resolved fraction*

    resolved(t) = R^2_pos(t) / x_var_captured(t)

i.e., R^2_pos normalized by the ceiling that the top-d PCs of x_t set
at layer t. resolved(t) answers the right question: of the variance the
top-d binning actually resolves at layer t, what fraction predicts the
layer update?

This normalization matters because the residual stream has effective
rank ~256 (Sarfati et al.; Phase 1 reproduces this). Binning on the
top-5 PCs captures only a small fraction of x_t's variance, so R^2_pos
has a hard ceiling well below 1 even when the drift IS perfectly a
function of position. The resolved fraction removes this ceiling
effect.

  - resolved >= 0.6 (PASS): The drift is well-defined as a function of
    position in the observed subspace. The overdamped Langevin framing
    has predictive content per-pilot.

  - 0.3 <= resolved < 0.6 (PARTIAL): The drift has a position-
    dependent component but a substantial fraction of the update is
    not explained by binned position. Run a d-sweep to disambiguate.

  - resolved < 0.3 (WEAK): Position-binning predicts only a small
    fraction of resolvable variance. Drift is not strongly a function
    of binned position.

  - Statistically non-significant gap above shuffle null
    (INCONCLUSIVE): the binning explains nothing the binning itself
    can't explain by chance.

resolved > 1 at some layer is mechanistically interesting: it means
the layer update concentrates in directions where the position
distribution is narrower than average. This is a signature of
late-layer prediction commitment, where the network's computation
contracts onto a low-dimensional output subspace.

Caveats
-------
  - d = 5 and K = 24 are choices. Increasing either should monotonically
    increase R^2_pos (finer bins explain more variance), until sample
    sizes per bin get small. Sensitivity is reported by a sweep.
  - The R^2_pos ceiling is set by how much of x_t's variance the top-d
    PCs capture. The plot overlays this fraction ("x-var captured") so
    you can see the ceiling. If the residual stream has effective rank
    ~256 (Sarfati et al.) and you bin on only 5 PCs, the ceiling can
    be substantially below 1 even when the drift IS a function of x_t.
    The diagnostic to look at is the GAP between R^2_pos and the
    shuffle null, not R^2_pos in absolute terms. A small R^2_pos that
    sits far above shuffle null is still a positive result.
  - The "position" we condition on is the post-block-t state. Block t
    operates on x_t and produces Delta_x_t = block_t(x_t, attention(x_t,
    other_tokens)). Attention's input depends on all previous tokens,
    which is the "hidden state" the position can't expose. This test
    measures how much of that hidden state's effect projects out into
    the residual stream by layer t -- i.e., how much of the relevant
    context is already encoded in x_t itself.
  - Boundary layer t = 0 (post-embedding) and t = L-1 (post-final-norm)
    behave differently: at t = 0 the embedding determines x_t entirely
    from the token id, so position binning collapses to token-identity
    binning at the embedding layer. We report all layers but flag the
    boundary cases.

Usage
-----
    python3 drift_welldef_test.py [--run-dir PATH] [--seed S] [--step STEP]
                                  [--d D] [--k K] [--n-shuffles N]

Outputs
-------
    drift_welldef/seed_S_step_STEP_d_D_k_K.npz   (raw arrays)
    drift_welldef/seed_S_step_STEP_d_D_k_K.png   (profile plot)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lazy import: only needed in the driver, not in the pure-numerics
# functions. Keeps the math testable without the project on the path.
def _load_augmented_activations(path: str) -> Dict[str, np.ndarray]:
    from multiview import load_augmented_activations
    return load_augmented_activations(path)


# ----------------------------------------------------------------------
# Core computation.
# ----------------------------------------------------------------------
def pca_projection(X: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Project (N, H) data onto its top-d principal components.

    Returns:
        projections: (N, d) projected coordinates
        components: (d, H) component matrix
        explained_fraction: scalar in [0, 1], the fraction of total
            variance captured by the top-d PCs. This is the structural
            ceiling on R^2_pos: if the top-d PCs miss most of x_t's
            variance, the position binning misses most of x_t's
            structure too.
    """
    mu = X.mean(axis=0, keepdims=True)        # (1, H)
    Xc = X - mu                                # (N, H)
    # Economy SVD; for N >= H this is fine and exact.
    if Xc.shape[0] >= Xc.shape[1]:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    else:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:d, :]                     # (d, H)
    projections = Xc @ components.T            # (N, d)
    # Fraction of variance in the top d singular values.
    s2 = s ** 2
    total = float(s2.sum())
    if total > 0:
        explained_fraction = float(s2[:d].sum()) / total
    else:
        explained_fraction = 0.0
    return projections, components, explained_fraction


def kmeans_assignments(X: np.ndarray, k: int, n_iter: int = 50,
                       seed: int = 0) -> np.ndarray:
    """Simple Lloyd's algorithm k-means, returns cluster assignments.

    We roll our own to avoid an sklearn dependency. Initialization is
    k-means++ for stability. Stops when assignments are stable or
    n_iter is reached.
    """
    rng = np.random.default_rng(seed)
    N, d = X.shape
    # k-means++ init.
    centers = np.empty((k, d), dtype=X.dtype)
    idx0 = rng.integers(0, N)
    centers[0] = X[idx0]
    # Squared distance to nearest center, updated as we go.
    dist2 = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = dist2 / dist2.sum() if dist2.sum() > 0 else np.full(N, 1.0 / N)
        idx = rng.choice(N, p=probs)
        centers[c] = X[idx]
        new_dist2 = np.sum((X - centers[c]) ** 2, axis=1)
        dist2 = np.minimum(dist2, new_dist2)

    # Lloyd iterations.
    assignments = np.zeros(N, dtype=np.int32)
    for _ in range(n_iter):
        # Assign each point to nearest center. (N, k) matrix of dist^2.
        # Use broadcasting; N * k * d is OK at our sizes (N ~ 9500, k ~ 24).
        d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # (N, k)
        new_assignments = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        # Recompute centers; handle empty clusters by leaving them put.
        for c in range(k):
            mask = (assignments == c)
            if mask.any():
                centers[c] = X[mask].mean(axis=0)
    return assignments


def within_between_var(deltas: np.ndarray,
                       assignments: np.ndarray,
                       k: int) -> Tuple[float, float, np.ndarray]:
    """Variance partition of `deltas` by cluster `assignments`.

    Args:
        deltas: (N, H) array of layer increments.
        assignments: (N,) int array of cluster ids in [0, k).
        k: total number of clusters.

    Returns:
        v_within: scalar, mean-over-coords of mean-over-clusters of
                  within-cluster variance.
        v_between: scalar, mean-over-coords of variance-across-clusters
                   of the per-cluster mean.
        cluster_counts: (k,) array of cluster sizes.
    """
    N, H = deltas.shape
    cluster_counts = np.zeros(k, dtype=np.int64)
    cluster_means = np.zeros((k, H), dtype=np.float64)
    cluster_var_sums = np.zeros((k, H), dtype=np.float64)  # sum of sq deviations

    # Per-cluster mean.
    for c in range(k):
        mask = (assignments == c)
        n_c = int(mask.sum())
        cluster_counts[c] = n_c
        if n_c == 0:
            continue
        sub = deltas[mask].astype(np.float64)
        cluster_means[c] = sub.mean(axis=0)
        cluster_var_sums[c] = np.sum((sub - cluster_means[c]) ** 2, axis=0)

    # V_within: average of per-cluster, per-coord variances, weighted by
    # cluster size (Bessel-uncorrected -- we're computing exact moments,
    # not estimators of a population variance).
    # Per-coord within variance = sum_c var_sum[c] / (sum_c n_c).
    total_n = cluster_counts.sum()
    if total_n == 0:
        return 0.0, 0.0, cluster_counts
    per_coord_within = cluster_var_sums.sum(axis=0) / total_n     # (H,)
    v_within = float(per_coord_within.mean())                      # scalar

    # V_between: per-coord variance of the cluster means, weighted by
    # cluster size (this is the standard between-group sum of squares
    # divided by N, mean over coords).
    grand_mean = (cluster_means * cluster_counts[:, None]).sum(axis=0) / total_n  # (H,)
    per_coord_between = (
        (cluster_means - grand_mean[None, :]) ** 2 * cluster_counts[:, None]
    ).sum(axis=0) / total_n                                        # (H,)
    v_between = float(per_coord_between.mean())                    # scalar

    return v_within, v_between, cluster_counts


def shuffle_null(deltas: np.ndarray, k: int, n_shuffles: int,
                 base_seed: int) -> Tuple[float, float]:
    """Null distribution of v_between under random cluster assignment.

    Returns the mean and std of v_between across `n_shuffles` random
    assignments. v_within is implicitly v_total - v_between, so we
    don't return it separately.
    """
    N, _ = deltas.shape
    rng = np.random.default_rng(base_seed)
    vals = np.empty(n_shuffles, dtype=np.float64)
    for s in range(n_shuffles):
        # Random equal-population assignment.
        assignments = rng.integers(0, k, size=N).astype(np.int32)
        _, vb, _ = within_between_var(deltas, assignments, k)
        vals[s] = vb
    return float(vals.mean()), float(vals.std())


# ----------------------------------------------------------------------
# Per-layer pipeline.
# ----------------------------------------------------------------------
def analyze_layer(states_t: np.ndarray,
                  states_tp1: np.ndarray,
                  d: int, k: int,
                  n_shuffles: int,
                  rng_seed: int) -> Dict[str, float]:
    """Run the position-binning analysis at a single layer transition.

    Args:
        states_t: (N, H) post-block-t-or-embedding state, used as the
                  position to bin on.
        states_tp1: (N, H) state after the next block; the increment is
                    Delta = states_tp1 - states_t.
        d: PCA dimension for the bin coordinate.
        k: number of k-means clusters.
        n_shuffles: number of shuffle-null replicates.
        rng_seed: seed for k-means init and shuffle null.

    Returns:
        dict with v_within, v_between, v_total, r2_pos, r2_pos_shuffle_mean,
        r2_pos_shuffle_std, mean_cluster_size, min_cluster_size.
    """
    deltas = (states_tp1 - states_t).astype(np.float64)
    # PCA on the position state, not the delta.
    proj, _, x_var_captured = pca_projection(states_t.astype(np.float64), d)
    assignments = kmeans_assignments(proj, k, seed=rng_seed)

    v_within, v_between, cluster_counts = within_between_var(
        deltas, assignments, k
    )
    v_total = v_within + v_between
    r2_pos = v_between / v_total if v_total > 0 else 0.0

    null_mean, null_std = shuffle_null(deltas, k, n_shuffles, rng_seed + 1)
    r2_null_mean = null_mean / v_total if v_total > 0 else 0.0
    r2_null_std = null_std / v_total if v_total > 0 else 0.0

    return {
        "v_within": v_within,
        "v_between": v_between,
        "v_total": v_total,
        "r2_pos": r2_pos,
        "r2_pos_shuffle_mean": r2_null_mean,
        "r2_pos_shuffle_std": r2_null_std,
        "x_var_captured": x_var_captured,
        "mean_cluster_size": float(cluster_counts.mean()),
        "min_cluster_size": int(cluster_counts.min()),
    }


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(run_dir, "multiview", f"seed_{seed}",
                        f"augmented_step_{step:08d}.npz")


def run_test1(run_dir: str, seed: int, step: int,
              d: int, k: int, n_shuffles: int,
              out_dir: str) -> Dict[str, np.ndarray]:
    """Run Test 1 across all layer transitions and save results."""
    path = augmented_path(run_dir, seed, step)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No augmented activations at {path}. "
            f"Run the multi-view activation collection first."
        )

    print(f"Loading {path} ...")
    payload = _load_augmented_activations(path)
    states = payload["states"]                     # (L, N, H)
    L, N, H = states.shape
    print(f"  states: L={L}, N={N}, H={H}")

    # Layer transitions: t -> t+1 for t in 0 .. L-2.
    n_transitions = L - 1
    results = {
        "layer_from": np.arange(n_transitions, dtype=np.int32),
        "v_within": np.zeros(n_transitions),
        "v_between": np.zeros(n_transitions),
        "v_total": np.zeros(n_transitions),
        "r2_pos": np.zeros(n_transitions),
        "r2_pos_shuffle_mean": np.zeros(n_transitions),
        "r2_pos_shuffle_std": np.zeros(n_transitions),
        "x_var_captured": np.zeros(n_transitions),
        "mean_cluster_size": np.zeros(n_transitions),
        "min_cluster_size": np.zeros(n_transitions, dtype=np.int32),
    }

    for t in range(n_transitions):
        print(f"  layer {t} -> {t+1} ...", end=" ", flush=True)
        layer_result = analyze_layer(
            states[t], states[t + 1],
            d=d, k=k, n_shuffles=n_shuffles,
            rng_seed=10_000 * seed + t,
        )
        for key in ("v_within", "v_between", "v_total", "r2_pos",
                    "r2_pos_shuffle_mean", "r2_pos_shuffle_std",
                    "x_var_captured", "mean_cluster_size"):
            results[key][t] = layer_result[key]
        results["min_cluster_size"][t] = layer_result["min_cluster_size"]
        print(f"R^2_pos = {layer_result['r2_pos']:.3f} "
              f"(shuffle {layer_result['r2_pos_shuffle_mean']:.3f}), "
              f"x-var cap {layer_result['x_var_captured']:.2f}, "
              f"min clust {layer_result['min_cluster_size']}")

    # Save.
    os.makedirs(out_dir, exist_ok=True)
    stem = f"seed_{seed}_step_{step:08d}_d_{d}_k_{k}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez(npz_path, **results,
             config=np.array([seed, step, d, k, n_shuffles, L, N, H],
                             dtype=np.int64))
    print(f"Saved {npz_path}")

    # Plot. Two panels:
    #   top    -- R^2_pos absolute, with shuffle null and x-var ceiling.
    #             This is the raw read.
    #   bottom -- resolved = R^2_pos / x-var-captured. This is the
    #             basis-invariant, ceiling-normalized read. Values
    #             above 1 are mechanistically interesting (layer
    #             updates concentrate in directions where the position
    #             distribution is narrower than average).
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]}
    )
    layers = results["layer_from"] + 0.5  # midpoint of the transition

    # --- top panel: absolute R^2 ---
    ax_top.plot(layers, results["r2_pos"], "o-", lw=2, ms=6,
                color="C0", label=r"$R^2_{\mathrm{pos}}$ (real)")
    ax_top.fill_between(
        layers,
        results["r2_pos_shuffle_mean"] - 2 * results["r2_pos_shuffle_std"],
        results["r2_pos_shuffle_mean"] + 2 * results["r2_pos_shuffle_std"],
        alpha=0.3, color="gray", label=r"shuffle null $\pm 2\sigma$"
    )
    ax_top.plot(layers, results["r2_pos_shuffle_mean"], "--",
                color="gray", lw=1)
    ax_top.plot(layers, results["x_var_captured"], "s-", lw=1.5, ms=4,
                color="C1", alpha=0.7,
                label=f"top-{d} PCs of $x_t$: variance captured (ceiling)")
    ax_top.set_ylabel(r"$R^2_{\mathrm{pos}}$  =  $V_{\mathrm{between}}$ / $V_{\mathrm{total}}$")
    ax_top.set_title(
        f"Drift well-definedness: fraction of layer-update variance\n"
        f"explained by top-{d} PCA bin (k={k} clusters)\n"
        f"seed {seed}, step {step}"
    )
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc="upper left", fontsize=9)

    # --- bottom panel: resolved fraction = R^2 / ceiling ---
    # Handle layers where ceiling is zero (shouldn't happen for real
    # data, but guard anyway).
    ceiling = results["x_var_captured"]
    resolved_per_layer = np.where(
        ceiling > 0, results["r2_pos"] / np.maximum(ceiling, 1e-12), 0.0
    )
    ax_bot.plot(layers, resolved_per_layer, "D-", lw=2, ms=6,
                color="C2",
                label=r"resolved fraction = $R^2_{\mathrm{pos}}$ / ceiling")
    ax_bot.axhline(1.0, color="black", lw=1, ls=":", alpha=0.6,
                   label="ceiling (resolved variance fully explained)")
    ax_bot.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
    ax_bot.set_xlabel("layer transition  (t -> t+1)")
    ax_bot.set_ylabel(r"$R^2_{\mathrm{pos}}$ / ceiling")
    # Set y limits to a sensible range that shows both <1 and >1 regions.
    y_top = max(1.2, float(resolved_per_layer.max()) * 1.1)
    ax_bot.set_ylim(-0.05, y_top)
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"Saved {png_path}")

    # Summary print.
    mean_r2 = float(results["r2_pos"].mean())
    mean_null = float(results["r2_pos_shuffle_mean"].mean())
    gap = mean_r2 - mean_null
    mean_xvar = float(results["x_var_captured"].mean())
    mean_null_std = float(results["r2_pos_shuffle_std"].mean())
    # Resolved fraction: of the variance the top-d PCs resolve, how much
    # of it predicts the update. This is the basis-invariant, ceiling-
    # normalized version of R^2_pos and is what should be compared
    # across choices of d and across layers.
    resolved = mean_r2 / mean_xvar if mean_xvar > 0 else 0.0
    # Significance: gap above null in units of null std. The null std is
    # tiny (~1e-3) compared to typical R^2 values, so this is essentially
    # always large in practice; we report it for completeness.
    z_gap = gap / mean_null_std if mean_null_std > 0 else float("inf")
    print()
    print("=" * 60)
    print(f"Summary (mean across {n_transitions} layer transitions):")
    print(f"  R^2_pos:                {mean_r2:.4f}")
    print(f"  shuffle null:           {mean_null:.4f}  (std {mean_null_std:.4f})")
    print(f"  gap above null:         {gap:.4f}  ({z_gap:.0f} sigma)")
    print(f"  top-{d} x-var ceiling:   {mean_xvar:.4f}")
    print(f"  resolved fraction:      {resolved:.4f}    "
          f"(= R^2 / ceiling)")
    print("=" * 60)
    # Verdict logic.
    #
    # The right basis-invariant quantity here is `resolved = R^2 /
    # ceiling`. It says: of the variance the top-d PCs actually resolve
    # at layer t, what fraction does that resolution predict about the
    # layer update? This is independent of d to first order -- d
    # changes the ceiling and R^2 in parallel.
    #
    # Gate on the null-significance first (must be statistically
    # different from random binning). Then read off the resolved
    # fraction.
    #
    # Thresholds calibrated against the post-Phase-1 observation that
    # the residual stream has effective rank ~256 and the top-5 PCs
    # capture ~14% of x_t's variance; the framework predicts that
    # whatever fraction is resolved should be heavily explained by
    # binned position if the SDE drift is well-defined.
    if z_gap < 3:
        verdict = (
            "INCONCLUSIVE: gap above shuffle null is not statistically "
            "significant.\n"
            "              Drift may not be well-defined as a function "
            "of position;\n"
            "              or sample size / binning resolution is too "
            "low to detect."
        )
    elif resolved >= 0.6:
        verdict = (
            "PASS: of the variance the top-d PCs resolve, the position\n"
            "      binning predicts the layer update strongly\n"
            f"      ({resolved:.0%} resolved fraction). The drift is\n"
            "      well-defined as a function of position in the\n"
            "      observed subspace. The overdamped Langevin framing\n"
            "      has predictive content per-pilot; Tests 2-5 are\n"
            "      worth running."
        )
    elif resolved >= 0.3:
        verdict = (
            "PARTIAL: position binning predicts a moderate fraction of\n"
            f"         the resolved update variance ({resolved:.0%}).\n"
            "         The drift has a position-dependent component but\n"
            "         a substantial fraction of the update is not\n"
            "         explained by binned position. Run a d-sweep to\n"
            "         see whether the resolved fraction saturates with\n"
            "         d (-> ceiling-limited, framework still applies)\n"
            "         or plateaus below 1 (-> genuine context\n"
            "         dependence beyond x_t)."
        )
    else:
        verdict = (
            "WEAK: position binning predicts only a small fraction of\n"
            f"      the resolved update variance ({resolved:.0%}).\n"
            "      The drift is not strongly a function of binned\n"
            "      position; layer updates depend on context that\n"
            "      does not project visibly into x_t."
        )
    print(verdict)
    print()

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="../phase1_runs_gelu",
                    help="Directory containing multiview/seed_*/augmented_*.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=24000,
                    help="Training step of the checkpoint to analyze")
    ap.add_argument("--d", type=int, default=5,
                    help="PCA dimension for the position bin coordinate")
    ap.add_argument("--k", type=int, default=24,
                    help="Number of k-means clusters")
    ap.add_argument("--n-shuffles", type=int, default=20,
                    help="Number of shuffle-null replicates")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: run-dir/drift_welldef/)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "drift_welldef")
    run_test1(args.run_dir, args.seed, args.step,
              args.d, args.k, args.n_shuffles, out_dir)


if __name__ == "__main__":
    main()
    