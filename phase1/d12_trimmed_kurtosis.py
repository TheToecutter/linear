"""
10.1 Trimmed kurtosis test.

Tests Possibility 2: rare-extreme-context mixture hypothesis. If the
heavy tails in p(x_t | v_i) come from a small fraction of pilots in
unusual contexts, then trimming the most extreme pilots (by Mahalanobis
distance from the per-token mean) should collapse the kurtosis. If the
heavy tails are intrinsic (Possibility 1) or come from high-order
context dependency that the extreme tail doesn't capture (Possibility 3),
trimming should leave kurtosis substantial.

For each input token in the forward set at each of the four
representative training-phase checkpoints:

  1. Compute per-pilot Mahalanobis distance from the per-token mean in
     a 32-dim PCA subspace at each layer (we use one shared subspace
     across layers, derived from the all-pilot bundle, for stability).
  2. Define a per-pilot "extremity" as the max Mahalanobis distance
     across interior layers 2-10 (avoid the boundary layers).
  3. Sort pilots by extremity. For each trim fraction f in {0, 1, 5,
     10, 20, 33}%, drop the top-f% most extreme pilots and recompute
     the per-layer kurtosis on the trimmed bundle.

Reading: if removing 5% of pilots drops layer-7 kurtosis from ~6 to ~1
or below at the final checkpoint, Possibility 2 is strongly supported.
If kurtosis is still ~4-5 even at 20% trimming, Possibility 2 is
refuted.

Output:
    run_dir/multiview/model_abc/d12_trimmed_kurtosis.npz
    run_dir/multiview/model_abc/figures/d12_trimmed_kurtosis.png

This script uses the *original* fixed-position augmented files
(augmented_step_NNNNNNNN.npz, not the _random.npz files), because we
want maximum sample-size per token to make the trim percentages
meaningful. The trimming question doesn't depend on position
decorrelation, so the original scheme is fine.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiview import load_augmented_activations
from multiview_campaign import (
    augmented_path,
    load_token_sets,
)


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


PHASE_LABELS = {
    479:   "Phase I",
    2563:  "Phase II",
    9809:  "Phase III mid",
    24000: "Phase III final",
}


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------
def _per_layer_kurtosis_in_basis(
    states_sub: np.ndarray,
    basis_per_layer: np.ndarray,
    mean_per_layer: np.ndarray,
) -> np.ndarray:
    """Excess kurtosis per layer in a shared (per-layer) basis."""
    L, n, H = states_sub.shape
    out = np.full(L, np.nan, dtype=np.float64)
    if n < 5:
        return out
    for t in range(L):
        X = states_sub[t].astype(np.float64) - mean_per_layer[t]
        Z = X @ basis_per_layer[t].T
        var = Z.var(axis=0)
        if not np.all(var > 0):
            continue
        m4 = ((Z - Z.mean(axis=0)) ** 4).mean(axis=0)
        out[t] = float(np.mean(m4 / (var ** 2) - 3.0))
    return out


def _global_basis(
    states: np.ndarray, d: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-d PC basis at each layer from the full all-pilot bundle."""
    L, N, H = states.shape
    d_eff = min(d, H, N - 1)
    basis = np.zeros((L, d_eff, H), dtype=np.float64)
    means = np.zeros((L, H), dtype=np.float64)
    for t in range(L):
        X = states[t].astype(np.float64)
        means[t] = X.mean(0)
        Xc = X - means[t]
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        basis[t] = Vt[:d_eff]
    return basis, means


def _per_token_mahalanobis(
    states_token: np.ndarray,
    basis_per_layer: np.ndarray,
    layer_range: Tuple[int, int],
) -> np.ndarray:
    """Per-pilot extremity = max(Mahalanobis distance) over layers in
    layer_range. Uses diagonal covariance in the PC basis (since PCs
    are orthogonal eigenvectors of the all-bundle covariance, but per-
    token covariance in this basis is not diagonal in general; we
    approximate with diagonal for stability with limited samples)."""
    L, n, H = states_token.shape
    lo, hi = layer_range
    extremity = np.zeros(n, dtype=np.float64)
    for t in range(lo, hi + 1):
        if t < 0 or t >= L:
            continue
        # Project token bundle into the all-bundle PC basis.
        X = states_token[t].astype(np.float64)
        mu = X.mean(0)
        Z = (X - mu) @ basis_per_layer[t].T   # (n, d)
        # Diagonal Mahalanobis: distance under per-coord variance of the
        # token bundle itself.
        var = Z.var(axis=0)
        var = np.maximum(var, 1e-12)
        d2 = ((Z * Z) / var).sum(axis=1)       # (n,)
        # Track elementwise max across layers.
        extremity = np.maximum(extremity, d2)
    return extremity


def analyze_token(
    states: np.ndarray, input_ids: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    token_id: int, trim_fractions: List[float],
    extremity_layer_range: Tuple[int, int],
) -> Dict:
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < 40:                # need enough to trim meaningfully
        return {"insufficient": True, "n_total": n_total}
    states_v = states[:, mask, :]
    L = states_v.shape[0]

    # Untrimmed baseline.
    baseline = _per_layer_kurtosis_in_basis(states_v, basis, means)

    # Extremity per pilot.
    extremity = _per_token_mahalanobis(
        states_v, basis, extremity_layer_range)
    # Sort ascending; top f% are the rightmost entries.
    order = np.argsort(extremity)

    trimmed_kurts: Dict[float, np.ndarray] = {}
    trimmed_sizes: Dict[float, int] = {}
    for f in trim_fractions:
        n_keep = int(np.floor(n_total * (1.0 - f)))
        if n_keep < 20:
            trimmed_kurts[f] = np.full(L, np.nan)
            trimmed_sizes[f] = n_keep
            continue
        keep_idx = order[:n_keep]
        sub = states_v[:, keep_idx, :]
        trimmed_kurts[f] = _per_layer_kurtosis_in_basis(sub, basis, means)
        trimmed_sizes[f] = n_keep

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline": baseline,
        "trimmed_kurts": trimmed_kurts,
        "trimmed_sizes": trimmed_sizes,
    }


def analyze_checkpoint(
    run_dir: str, seed: int, step: int, tids: np.ndarray,
    trim_fractions: List[float], max_pc_dim: int,
    extremity_layer_range: Tuple[int, int],
) -> Optional[Dict]:
    aug_path = augmented_path(run_dir, seed, step)
    if not os.path.exists(aug_path):
        print(f"  [skip] missing {aug_path}")
        return None
    aug = load_augmented_activations(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")

    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    for tok in tids:
        results[int(tok)] = analyze_token(
            states, input_ids, basis, means,
            int(tok),
            trim_fractions=trim_fractions,
            extremity_layer_range=extremity_layer_range,
        )
    valid = [r for r in results.values()
             if not r.get("insufficient", True)]
    print(f"    {len(valid)}/{tids.size} tokens with sufficient samples")
    return {
        "seed": seed, "step": step,
        "results_per_token": results,
    }


def aggregate(checkpoint_results: Dict,
              trim_fractions: List[float]) -> Dict:
    valid = [r for r in checkpoint_results["results_per_token"].values()
             if not r.get("insufficient", True)]
    if not valid:
        return {}
    L = valid[0]["baseline"].size
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights /= weights.sum()

    def _wmean(extract) -> np.ndarray:
        arr = np.stack([extract(r) for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]
            v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=weights[v]))
        return out

    profiles = {
        "baseline": _wmean(lambda r: r["baseline"]),
        "trimmed": {f: _wmean(lambda r, ff=f: r["trimmed_kurts"][ff])
                    for f in trim_fractions},
        "n_valid": len(valid),
    }
    return profiles


def plot_results(
    run_dir: str, aggregates: Dict[int, Dict],
    trim_fractions: List[float], seed: int, layer: int,
) -> None:
    steps = sorted(aggregates.keys())
    if not steps:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))

    # Panel 1: trimmed kurtosis at layer of interest vs trim fraction.
    ax = axes[0]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        baseline_at_t = prof["baseline"][layer]
        vals = [prof["trimmed"][f][layer] for f in trim_fractions]
        ax.plot([f * 100 for f in trim_fractions], vals,
                "o-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, str(step))} "
                      f"(base={baseline_at_t:.2f})")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("trim fraction (%)")
    ax.set_ylabel(f"trimmed kurtosis at layer {layer}")
    ax.set_title("Trimmed kurtosis vs trim fraction")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: ratio trimmed / baseline vs trim fraction.
    ax = axes[1]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        b = prof["baseline"][layer]
        if abs(b) < 1e-3:
            continue
        vals = [prof["trimmed"][f][layer] / b for f in trim_fractions]
        ax.plot([f * 100 for f in trim_fractions], vals,
                "o-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, str(step))}")
    ax.axhline(1.0, color="k", lw=0.7, ls=":", label="baseline")
    ax.axhline(0.2, color="C3", lw=1, ls="--",
               label="80% reduction threshold")
    ax.set_xlabel("trim fraction (%)")
    ax.set_ylabel(f"trimmed / baseline kurtosis (layer {layer})")
    ax.set_title("Reduction ratio vs trim fraction")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: full per-layer profile at largest trim, vs baseline,
    # at the final checkpoint.
    ax = axes[2]
    final_step = steps[-1]
    prof = aggregates[final_step]
    if prof:
        L = prof["baseline"].size
        layers_axis = np.arange(L)
        ax.plot(layers_axis, prof["baseline"], "k-", lw=2.5,
                label="baseline (no trim)")
        for ci, f in enumerate(trim_fractions):
            if f == 0:
                continue
            ax.plot(layers_axis, prof["trimmed"][f], "-",
                    color=plt.cm.plasma(0.15 + 0.7 * (ci / len(trim_fractions))),
                    lw=1.5, label=f"trim {int(f*100)}%")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title(f"Per-layer profile at {PHASE_LABELS.get(final_step, str(final_step))}")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"D12: trimmed kurtosis test (Possibility 2), seed {seed}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir),
                       "d12_trimmed_kurtosis.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


DEFAULT_STEPS = [479, 2563, 9809, 24000]
DEFAULT_TRIM = [0.0, 0.01, 0.05, 0.10, 0.20, 0.33]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--trim-fractions", type=float, nargs="+",
                    default=DEFAULT_TRIM)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--extremity-layer-lo", type=int, default=2)
    ap.add_argument("--extremity-layer-hi", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")
    print(f"Steps: {args.steps}")
    print(f"Trim fractions: {args.trim_fractions}")

    aggregates = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, str(step))}:")
        result = analyze_checkpoint(
            args.run_dir, args.seed, step, tids,
            trim_fractions=args.trim_fractions,
            max_pc_dim=args.max_pc_dim,
            extremity_layer_range=(args.extremity_layer_lo,
                                   args.extremity_layer_hi),
        )
        if result is None:
            continue
        aggregates[step] = aggregate(result, args.trim_fractions)
        prof = aggregates[step]
        if prof:
            t = args.layer
            base = prof["baseline"][t]
            print(f"    layer {t} baseline = {base:.3f}")
            print(f"    layer {t} trimmed kurtosis:")
            print(f"    {'trim %':>8s}  {'kurtosis':>10s}  "
                  f"{'ratio to base':>14s}")
            for f in args.trim_fractions:
                k = prof["trimmed"][f][t]
                ratio = k / base if abs(base) > 1e-3 else np.nan
                print(f"    {f*100:>7.1f}%  {k:>10.3f}  {ratio:>14.3f}")
            print(f"    n_valid_tokens = {prof['n_valid']}")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed:.1f}s")

    # Save.
    out_arrays = {
        "steps": np.array(sorted(aggregates.keys()), dtype=np.int64),
        "trim_fractions": np.array(args.trim_fractions, dtype=np.float64),
        "layer": np.int32(args.layer),
        "seed": np.int32(args.seed),
    }
    for step, prof in aggregates.items():
        if not prof:
            continue
        out_arrays[f"step_{step}_baseline"] = prof["baseline"]
        for f in args.trim_fractions:
            out_arrays[f"step_{step}_trim_{int(f*1000):04d}"] = \
                prof["trimmed"][f]
    out_path = os.path.join(output_root(args.run_dir),
                            "d12_trimmed_kurtosis.npz")
    np.savez(out_path, **out_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates,
                 trim_fractions=args.trim_fractions,
                 seed=args.seed, layer=args.layer)


if __name__ == "__main__":
    main()
