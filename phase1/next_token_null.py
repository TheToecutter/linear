"""
Next-token null control.

The D10 position-granularity experiment revealed that most of the
"position effect" on conditional kurtosis is a sample-partitioning
artifact: random label assignment produces nearly identical kurtosis
reduction to position binning. We did not control for this artifact
when reporting the next-token reduction.

This script applies the same null control to next-token. For each
input token v_i in the forward set, the (input, next) partition
produces some number of valid sub-bundles K_i (next-tokens with at
least min_subbundle pilots). The matched random-labels null is to
shuffle K_i random labels onto v_i's pilots and compute the same
weighted-mean kurtosis. We average over multiple random labelings to
stabilize the null.

The diagnostic statistic is:

    real_signal = (baseline - next_kurt) - (baseline - random_null)
                = random_null_kurt - next_kurt

A real_signal of zero means next-token reduces kurtosis only because
partitioning into ~K_i sub-bundles reduces sample kurtosis estimates
(no information specific to which token is next). A real_signal
substantially above zero means next-token carries genuine context
information beyond the partitioning artifact.

We run this analysis across all four representative checkpoints from
the three training phases identified in §8.2.

Output:
    run_dir/multiview/model_abc/d11_next_token_null.npz
    run_dir/multiview/model_abc/figures/d11_next_token_null.png

Usage:
    python next_token_null.py --run-dir ../phase1_runs_gelu
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


def randomized_augmented_path(run_dir: str, seed: int, step: int) -> str:
    return augmented_path(run_dir, seed, step).replace(
        ".npz", "_random.npz")


PHASE_LABELS = {
    479:   "Phase I",
    2563:  "Phase II",
    9809:  "Phase III mid",
    24000: "Phase III final",
}


# ----------------------------------------------------------------------
# Kurtosis helpers (same convention as d9/d10).
# ----------------------------------------------------------------------
def _per_layer_kurtosis(
    states_sub: np.ndarray,
    basis_per_layer: np.ndarray,
    global_mean_per_layer: np.ndarray,
) -> np.ndarray:
    L, n, H = states_sub.shape
    out = np.full(L, np.nan, dtype=np.float64)
    if n < 5:
        return out
    for t in range(L):
        X = states_sub[t].astype(np.float64) - global_mean_per_layer[t]
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


def _aggregate_kurt(
    states_v: np.ndarray,
    group_ids: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    min_subbundle: int,
) -> Tuple[np.ndarray, int]:
    L = states_v.shape[0]
    unique, counts = np.unique(group_ids, return_counts=True)
    keep = unique[counts >= min_subbundle]
    if keep.size == 0:
        return np.full(L, np.nan), 0
    weights, kurts = [], []
    for u in keep:
        mask = (group_ids == u)
        sub = states_v[:, mask, :]
        k = _per_layer_kurtosis(sub, basis, means)
        kurts.append(k)
        weights.append(int(mask.sum()))
    K = np.stack(kurts)
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    out = np.full(L, np.nan)
    for t in range(L):
        col = K[:, t]
        v = np.isfinite(col)
        if v.sum():
            out[t] = float(np.average(col[v], weights=w[v]))
    return out, int(keep.size)


# ----------------------------------------------------------------------
# Per-token analysis: next-token reduction with matched random null.
# ----------------------------------------------------------------------
def analyze_token(
    states: np.ndarray,
    input_ids: np.ndarray, next_ids: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    token_id: int, min_subbundle: int,
    n_random_reps: int, rng: np.random.Generator,
) -> Dict:
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_subbundle * 2:
        return {"insufficient": True, "n_total": n_total}

    states_v = states[:, mask, :]
    next_v = next_ids[mask]
    L = states_v.shape[0]

    baseline = _per_layer_kurtosis(states_v, basis, means)
    next_kurt, n_next_sub = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)

    # Matched random null: assign random labels in [0, n_next_sub).
    # The number of bins (k) matches the effective bin count produced
    # by the next-token partition (sub-bundles with at least
    # min_subbundle pilots). The mean-bucket-size after the random
    # assignment will be close to the mean-bucket-size in the next-
    # token partition (modulo the small fraction of next-token-mass
    # excluded by the min_subbundle filter).
    if n_next_sub < 2:
        # Random null is undefined; skip.
        return {
            "insufficient": False,
            "token_id": int(token_id),
            "n_total": n_total,
            "baseline": baseline,
            "next_kurt": next_kurt,
            "n_next_subbundles": n_next_sub,
            "random_null_kurt_mean": np.full(L, np.nan),
            "random_null_kurt_std":  np.full(L, np.nan),
            "skipped_random_null": True,
        }
    reps = []
    for _ in range(n_random_reps):
        rl = rng.integers(0, n_next_sub, size=n_total, dtype=np.int32)
        kurt_r, _ = _aggregate_kurt(
            states_v, rl, basis, means, min_subbundle)
        reps.append(kurt_r)
    reps_arr = np.stack(reps)
    random_mean = np.nanmean(reps_arr, axis=0)
    random_std = np.nanstd(reps_arr, axis=0, ddof=1)

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline": baseline,
        "next_kurt": next_kurt,
        "n_next_subbundles": n_next_sub,
        "random_null_kurt_mean": random_mean,
        "random_null_kurt_std": random_std,
        "skipped_random_null": False,
    }


def analyze_checkpoint(
    run_dir: str, seed: int, step: int,
    tids: np.ndarray, min_subbundle: int,
    n_random_reps: int, max_pc_dim: int,
    rng: np.random.Generator,
) -> Optional[Dict]:
    aug_path = randomized_augmented_path(run_dir, seed, step)
    if not os.path.exists(aug_path):
        print(f"  [skip] missing {aug_path}")
        return None
    aug = load_augmented_activations(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")
    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    for tok in tids:
        results[int(tok)] = analyze_token(
            states, input_ids, next_ids, basis, means,
            int(tok), min_subbundle=min_subbundle,
            n_random_reps=n_random_reps, rng=rng,
        )

    valid = [r for r in results.values() if not r.get("insufficient", True)]
    print(f"    {len(valid)}/{tids.size} tokens with sufficient samples")
    return {
        "seed": seed, "step": step,
        "results_per_token": results,
    }


# ----------------------------------------------------------------------
# Aggregate.
# ----------------------------------------------------------------------
def aggregate(checkpoint_results: Dict) -> Dict:
    valid = [r for r in checkpoint_results["results_per_token"].values()
             if (not r.get("insufficient", True)
                 and not r.get("skipped_random_null", False))]
    if not valid:
        return {}
    L = valid[0]["baseline"].size
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights /= weights.sum()

    def _wmean(key: str) -> np.ndarray:
        arr = np.stack([r[key] for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]
            v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=weights[v]))
        return out

    baseline = _wmean("baseline")
    next_kurt = _wmean("next_kurt")
    null_kurt = _wmean("random_null_kurt_mean")
    n_subbundles_arr = np.array([r["n_next_subbundles"] for r in valid])
    return {
        "baseline": baseline,
        "next_kurt": next_kurt,
        "null_kurt": null_kurt,
        "reduction_next": baseline - next_kurt,
        "reduction_null": baseline - null_kurt,
        "real_signal": null_kurt - next_kurt,
        "n_subbundles_mean": float(n_subbundles_arr.mean()),
        "n_subbundles_min":  int(n_subbundles_arr.min()),
        "n_subbundles_max":  int(n_subbundles_arr.max()),
        "n_valid_tokens": len(valid),
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(
    run_dir: str, aggregates: Dict[int, Dict],
    seed: int, layer_of_interest: int,
) -> None:
    steps = sorted(aggregates.keys())
    if not steps:
        return

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))

    # Panel 1: baseline, next, null kurtosis vs layer at each phase.
    ax = axes[0]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        L = prof["baseline"].size
        layers = np.arange(L)
        c = cmap[ci]
        ax.plot(layers, prof["baseline"], "-", color=c, lw=2,
                label=f"{PHASE_LABELS.get(step, str(step))}: baseline")
        ax.plot(layers, prof["next_kurt"], "--", color=c, lw=1.5,
                label=f"{PHASE_LABELS.get(step, str(step))}: | next")
        ax.plot(layers, prof["null_kurt"], ":", color=c, lw=1.5,
                label=f"{PHASE_LABELS.get(step, str(step))}: random null")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Baseline vs next-token vs matched random null")
    ax.legend(fontsize=7, loc="best", ncol=2)

    # Panel 2: bar plot of three reduction components at the
    # representative layer.
    ax = axes[1]
    width = 0.27
    x = np.arange(len(steps))
    reduction_next = [aggregates[s]["reduction_next"][layer_of_interest]
                      for s in steps]
    reduction_null = [aggregates[s]["reduction_null"][layer_of_interest]
                      for s in steps]
    real_signal = [aggregates[s]["real_signal"][layer_of_interest]
                   for s in steps]
    ax.bar(x - width, reduction_next, width,
           label="next-token reduction (uncorrected)", color="C0")
    ax.bar(x, reduction_null, width,
           label="random-null reduction (artifact)", color="gray")
    ax.bar(x + width, real_signal, width,
           label="real signal = null - next", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(s, str(s))
                        for s in steps], rotation=20, fontsize=8)
    ax.set_ylabel(f"kurtosis reduction at layer {layer_of_interest}")
    ax.set_title(f"Next-token: total vs artifact vs real signal "
                 f"(layer {layer_of_interest})")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="k", lw=0.7, ls=":")

    # Panel 3: real_signal per layer, one line per phase.
    ax = axes[2]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        L = prof["real_signal"].size
        layers = np.arange(L)
        ax.plot(layers, prof["real_signal"], "-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, str(step))}: "
                      f"n_sub mean={prof['n_subbundles_mean']:.1f}")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("real signal: null - next  (kurtosis units)")
    ax.set_title("Real next-token signal per layer, by phase")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"D11: next-token null control, seed {seed}, randomized stage A",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d11_next_token_null.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
DEFAULT_STEPS = [479, 2563, 9809, 24000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--min-subbundle", type=int, default=20)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--n-random-reps", type=int, default=20,
                    help="More replicates than D10 because the null is "
                         "the headline quantity here.")
    ap.add_argument("--layer", type=int, default=7,
                    help="Layer index for the summary statistics.")
    ap.add_argument("--rng-seed", type=int, default=20260521)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")
    print(f"Checkpoints: {args.steps}")
    print(f"Random null reps per token: {args.n_random_reps}")

    rng = np.random.default_rng(args.rng_seed)
    aggregates: Dict[int, Dict] = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, str(step))}:")
        result = analyze_checkpoint(
            args.run_dir, args.seed, step, tids,
            min_subbundle=args.min_subbundle,
            n_random_reps=args.n_random_reps,
            max_pc_dim=args.max_pc_dim,
            rng=rng,
        )
        if result is None:
            continue
        aggregates[step] = aggregate(result)
        prof = aggregates[step]
        if prof:
            t = args.layer
            print(f"    layer {t} summary:")
            print(f"      baseline                = {prof['baseline'][t]:.3f}")
            print(f"      next_kurt               = {prof['next_kurt'][t]:.3f}")
            print(f"      null_kurt               = {prof['null_kurt'][t]:.3f}")
            print(f"      reduction (next)        = "
                  f"{prof['reduction_next'][t]:.3f}")
            print(f"      reduction (null)        = "
                  f"{prof['reduction_null'][t]:.3f}")
            print(f"      real signal             = "
                  f"{prof['real_signal'][t]:.3f}")
            print(f"      n_subbundles per token  = "
                  f"mean {prof['n_subbundles_mean']:.1f}, "
                  f"min {prof['n_subbundles_min']}, "
                  f"max {prof['n_subbundles_max']}")
            print(f"      n_valid_tokens          = {prof['n_valid_tokens']}")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed:.1f}s")

    out_arrays = {
        "steps": np.array(sorted(aggregates.keys()), dtype=np.int64),
        "layer_of_interest": np.int32(args.layer),
        "seed": np.int32(args.seed),
    }
    for step, prof in aggregates.items():
        if not prof:
            continue
        for key in ("baseline", "next_kurt", "null_kurt",
                    "reduction_next", "reduction_null", "real_signal"):
            out_arrays[f"step_{step}_{key}"] = prof[key]
        out_arrays[f"step_{step}_n_subbundles_mean"] = np.float64(
            prof["n_subbundles_mean"])
    out_path = os.path.join(output_root(args.run_dir),
                            "d11_next_token_null.npz")
    np.savez(out_path, **out_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates,
                 seed=args.seed, layer_of_interest=args.layer)


if __name__ == "__main__":
    main()
