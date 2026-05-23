"""
10.3 n-gram partition test (fixed version).

The original D14 had a sample-size bug: at the joint (prev, next)
partition resolution, most input tokens produced zero sub-bundles
passing the min_subbundle filter. The null computation then operated
on the same sparse partition and gave nonsensical NaN-laced numbers.

Fixes in this version:

  1. Filter to input tokens with at least min_pilots_per_token total
     pilots (default 200), so that any partition produces meaningful
     sub-bundles.

  2. Use min_subbundle = 15 by default (down from 20 in the original
     subcondition.py, since the joint partition is naturally finer).

  3. Track per-token validity separately for each partition variable.
     A token contributes to the aggregate next/prev/joint result ONLY
     if its partition produced >= 2 sub-bundles passing min_subbundle.

  4. The matched random null uses the SAME effective bin count that
     the real partition produced for that token (after filtering),
     ensuring an apples-to-apples comparison.

  5. Explicit reporting of how many tokens contributed to each
     aggregate, so we can see immediately if a partition is starved.

Reads existing *_ngram.npz files from the previous run (no inference
needed). If those files do not exist, the script tells the user to
run the original d14_ngram_partition.py first to generate them.
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

from multiview_campaign import (
    augmented_path,
    load_token_sets,
)


PHASE_LABELS = {
    479:   "Phase I",
    2563:  "Phase II",
    9809:  "Phase III mid",
    24000: "Phase III final",
}


def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


def ngram_augmented_path(run_dir: str, seed: int, step: int) -> str:
    return augmented_path(run_dir, seed, step).replace(
        ".npz", "_ngram.npz")


def load_ngram_payload(path: str) -> dict:
    with np.load(path) as f:
        return {
            "states":    f["states"],
            "input_ids": f["input_ids"],
            "next_ids":  f["next_ids"],
            "prev_ids":  f["prev_ids"],
            "pred_ids":  f["pred_ids"],
            "positions": f["positions"],
        }


# ----------------------------------------------------------------------
# Kurtosis helpers.
# ----------------------------------------------------------------------
def _per_layer_kurtosis(states_sub, basis, means):
    L, n, H = states_sub.shape
    out = np.full(L, np.nan, dtype=np.float64)
    if n < 5:
        return out
    for t in range(L):
        X = states_sub[t].astype(np.float64) - means[t]
        Z = X @ basis[t].T
        var = Z.var(axis=0)
        if not np.all(var > 0):
            continue
        m4 = ((Z - Z.mean(axis=0)) ** 4).mean(axis=0)
        out[t] = float(np.mean(m4 / (var ** 2) - 3.0))
    return out


def _global_basis(states, d):
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


def _aggregate_kurt(states_v, group_ids, basis, means, min_subbundle):
    """Returns (per-layer kurtosis, number of valid sub-bundles)."""
    L = states_v.shape[0]
    unique, counts = np.unique(group_ids, return_counts=True)
    keep = unique[counts >= min_subbundle]
    if keep.size < 2:           # NEW: require at least 2 sub-bundles
        return np.full(L, np.nan), int(keep.size)
    weights, kurts = [], []
    for u in keep:
        mask = (group_ids == u)
        sub = states_v[:, mask, :]
        k = _per_layer_kurtosis(sub, basis, means)
        kurts.append(k)
        weights.append(int(mask.sum()))
    K = np.stack(kurts)
    w = np.array(weights, dtype=np.float64); w /= w.sum()
    out = np.full(L, np.nan)
    for t in range(L):
        col = K[:, t]
        v = np.isfinite(col)
        if v.sum():
            out[t] = float(np.average(col[v], weights=w[v]))
    return out, int(keep.size)


# ----------------------------------------------------------------------
# Per-token analysis with explicit validity tracking.
# ----------------------------------------------------------------------
def analyze_token(
    states, input_ids, next_ids, prev_ids,
    basis, means, token_id,
    min_pilots_per_token: int, min_subbundle: int,
    n_random_reps: int, rng,
):
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_pilots_per_token:
        return {"insufficient": True, "n_total": n_total,
                "reason": f"n_total={n_total} < {min_pilots_per_token}"}
    states_v = states[:, mask, :]
    next_v = next_ids[mask]
    prev_v = prev_ids[mask]
    L = states_v.shape[0]

    baseline = _per_layer_kurtosis(states_v, basis, means)
    next_kurt,  n_next  = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)
    prev_kurt,  n_prev  = _aggregate_kurt(
        states_v, prev_v, basis, means, min_subbundle)

    # Joint (prev, next): encode as pair_id.
    M = int(max(prev_v.max(), next_v.max())) + 1
    joint_ids = prev_v.astype(np.int64) * M + next_v.astype(np.int64)
    joint_kurt, n_joint = _aggregate_kurt(
        states_v, joint_ids, basis, means, min_subbundle)

    # Matched random nulls: only compute if the corresponding partition
    # produced >= 2 valid sub-bundles. Use the *actual* number of valid
    # sub-bundles as k for the random null. This matches the
    # post-filter cell count and lets us interpret the null as
    # "partitioning at this same effective bin count, but at random".
    def _null(k_bins: int, n_reps: int) -> np.ndarray:
        if k_bins < 2:
            return np.full(L, np.nan)
        # Random null can also fail the min_subbundle filter at high k.
        # Use rejection sampling: if a particular replicate produces
        # fewer than 2 valid cells, drop it and re-roll. Cap retries.
        valid_reps = []
        max_attempts = n_reps * 5
        attempts = 0
        while len(valid_reps) < n_reps and attempts < max_attempts:
            rl = rng.integers(0, k_bins, size=n_total, dtype=np.int32)
            kr, n_valid_cells = _aggregate_kurt(
                states_v, rl, basis, means, min_subbundle)
            attempts += 1
            if n_valid_cells >= 2:
                valid_reps.append(kr)
        if not valid_reps:
            return np.full(L, np.nan)
        return np.nanmean(np.stack(valid_reps), axis=0)

    null_next  = _null(n_next,  n_random_reps)
    null_prev  = _null(n_prev,  n_random_reps)
    null_joint = _null(n_joint, n_random_reps)

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline": baseline,
        "next_kurt": next_kurt,  "n_next":  n_next,
        "prev_kurt": prev_kurt,  "n_prev":  n_prev,
        "joint_kurt": joint_kurt, "n_joint": n_joint,
        "null_next":  null_next,
        "null_prev":  null_prev,
        "null_joint": null_joint,
        # Explicit per-partition validity flags.
        "next_valid":  bool(n_next  >= 2 and np.isfinite(next_kurt).any()
                            and np.isfinite(null_next).any()),
        "prev_valid":  bool(n_prev  >= 2 and np.isfinite(prev_kurt).any()
                            and np.isfinite(null_prev).any()),
        "joint_valid": bool(n_joint >= 2 and np.isfinite(joint_kurt).any()
                            and np.isfinite(null_joint).any()),
    }


def analyze_checkpoint_data(
    aug: dict, tids: np.ndarray,
    min_pilots_per_token: int, min_subbundle: int,
    max_pc_dim: int, n_random_reps: int, rng,
):
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    prev_ids = aug["prev_ids"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")
    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    for tok in tids:
        r = analyze_token(
            states, input_ids, next_ids, prev_ids, basis, means,
            int(tok),
            min_pilots_per_token=min_pilots_per_token,
            min_subbundle=min_subbundle,
            n_random_reps=n_random_reps,
            rng=rng,
        )
        results[int(tok)] = r
    return results


def aggregate(results: Dict, partition: str) -> Dict:
    """Aggregate results across tokens for a specific partition.

    partition is one of 'next', 'prev', 'joint'. Tokens whose validity
    flag for that partition is False are excluded."""
    valid_key = f"{partition}_valid"
    kurt_key = f"{partition}_kurt"
    null_key = f"null_{partition}"
    n_key = f"n_{partition}"

    valid = [r for r in results.values()
             if (not r.get("insufficient", True)) and r.get(valid_key, False)]
    if not valid:
        return {"n_valid_tokens": 0}

    L = valid[0]["baseline"].size
    w = np.array([r["n_total"] for r in valid], dtype=np.float64)
    w /= w.sum()

    def _wm(extract):
        arr = np.stack([extract(r) for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]; v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=w[v]))
        return out

    baseline = _wm(lambda r: r["baseline"])
    partition_kurt = _wm(lambda r: r[kurt_key])
    null_kurt = _wm(lambda r: r[null_key])
    n_sub_arr = np.array([r[n_key] for r in valid])
    return {
        "baseline":   baseline,
        "partition_kurt": partition_kurt,
        "null_kurt": null_kurt,
        "reduction": baseline - partition_kurt,
        "null_reduction": baseline - null_kurt,
        "real_signal": null_kurt - partition_kurt,
        "n_sub_mean": float(n_sub_arr.mean()),
        "n_sub_min":  int(n_sub_arr.min()),
        "n_sub_max":  int(n_sub_arr.max()),
        "n_valid_tokens": len(valid),
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(run_dir: str, aggregates_by_step_partition: Dict,
                 seed: int, layer: int) -> None:
    steps = sorted({step for (step, _) in aggregates_by_step_partition.keys()})
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: bar chart of real signals per partition per phase.
    ax = axes[0]
    width = 0.27
    x = np.arange(len(steps))
    for pi, partition in enumerate(["next", "prev", "joint"]):
        vals = []
        for s in steps:
            agg = aggregates_by_step_partition.get((s, partition), {})
            v = agg.get("real_signal", np.full(0, np.nan))
            vals.append(v[layer] if len(v) > layer else np.nan)
        color = ["C0", "C1", "C2"][pi]
        label = ("real signal: next" if partition == "next"
                 else "real signal: prev" if partition == "prev"
                 else "real signal: (prev, next)")
        ax.bar(x + (pi - 1) * width, vals, width, label=label, color=color)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(s, str(s)) for s in steps],
                       rotation=15, fontsize=8)
    ax.set_ylabel(f"real kurtosis signal at layer {layer}")
    ax.set_title("D14 (fixed): real signal (null-corrected) by partition")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: per-layer profile at the final checkpoint (next partition
    # only as representative; all three are similar in shape and would
    # clutter the plot).
    ax = axes[1]
    final_step = steps[-1]
    next_agg  = aggregates_by_step_partition.get((final_step, "next"), {})
    prev_agg  = aggregates_by_step_partition.get((final_step, "prev"), {})
    joint_agg = aggregates_by_step_partition.get((final_step, "joint"), {})
    if next_agg:
        L = next_agg["baseline"].size
        layers = np.arange(L)
        ax.plot(layers, next_agg["baseline"], "k-", lw=2.5, label="baseline")
        ax.plot(layers, next_agg["partition_kurt"], "--", color="C0", lw=1.5,
                label=f"| next  (n_sub mean {next_agg['n_sub_mean']:.1f})")
        ax.plot(layers, next_agg["null_kurt"], ":", color="C0", lw=1.5,
                label="null at next k")
        if prev_agg:
            ax.plot(layers, prev_agg["partition_kurt"], "--", color="C1",
                    lw=1.5,
                    label=f"| prev  (n_sub mean {prev_agg['n_sub_mean']:.1f})")
            ax.plot(layers, prev_agg["null_kurt"], ":", color="C1", lw=1.5,
                    label="null at prev k")
        if joint_agg:
            ax.plot(layers, joint_agg["partition_kurt"], "--", color="C2",
                    lw=1.5,
                    label=f"| (prev, next)  (n_sub mean "
                          f"{joint_agg['n_sub_mean']:.1f})")
            ax.plot(layers, joint_agg["null_kurt"], ":", color="C2", lw=1.5,
                    label="null at joint k")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title(
        f"Per-layer profiles at "
        f"{PHASE_LABELS.get(final_step, str(final_step))}")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"D14 FIXED: n-gram partition (Possibility 3), seed {seed}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir),
                       "d14_ngram_partition_fixed.png")
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
    ap.add_argument("--min-pilots-per-token", type=int, default=200,
                    help="Filter to input tokens with this many pilots. "
                         "Default 200, gives joint partitions a fighting "
                         "chance.")
    ap.add_argument("--min-subbundle", type=int, default=15)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--n-random-reps", type=int, default=20)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--rng-seed", type=int, default=20260521)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)

    # Verify all ngram files exist.
    for step in args.steps:
        path = ngram_augmented_path(args.run_dir, args.seed, step)
        if not os.path.exists(path):
            print(f"ERROR: missing {path}")
            print(f"Run the original d14_ngram_partition.py first to "
                  f"generate these files.")
            sys.exit(1)

    print(f"Forward set: {tids.size} tokens")
    print(f"Steps: {args.steps}")
    print(f"min_pilots_per_token = {args.min_pilots_per_token}")
    print(f"min_subbundle = {args.min_subbundle}")

    rng = np.random.default_rng(args.rng_seed)
    aggregates_by_step_partition = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, str(step))}:")
        aug_path = ngram_augmented_path(args.run_dir, args.seed, step)
        aug = load_ngram_payload(aug_path)
        results = analyze_checkpoint_data(
            aug, tids,
            min_pilots_per_token=args.min_pilots_per_token,
            min_subbundle=args.min_subbundle,
            max_pc_dim=args.max_pc_dim,
            n_random_reps=args.n_random_reps,
            rng=rng,
        )
        # Report which tokens passed the per-token filter.
        passing = sum(1 for r in results.values()
                      if not r.get("insufficient", True))
        print(f"    tokens passing min_pilots_per_token: "
              f"{passing}/{tids.size}")
        if passing == 0:
            print(f"    [skip] no tokens have enough pilots for this analysis")
            continue
        # Per-partition aggregates.
        for partition in ["next", "prev", "joint"]:
            agg = aggregate(results, partition)
            aggregates_by_step_partition[(step, partition)] = agg
            t = args.layer
            if agg.get("n_valid_tokens", 0) > 0:
                print(f"    {partition:>6s}:  "
                      f"reduction={agg['reduction'][t]:.3f}  "
                      f"null={agg['null_reduction'][t]:.3f}  "
                      f"real={agg['real_signal'][t]:.3f}  "
                      f"n_sub mean={agg['n_sub_mean']:.1f}  "
                      f"({agg['n_sub_min']}-{agg['n_sub_max']})  "
                      f"n_tok={agg['n_valid_tokens']}")
            else:
                print(f"    {partition:>6s}:  NO VALID TOKENS")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed:.1f}s")

    # Save.
    out_arrays = {
        "steps": np.array(sorted({s for (s, _) in aggregates_by_step_partition.keys()}),
                          dtype=np.int64),
        "layer": np.int32(args.layer),
        "seed": np.int32(args.seed),
        "min_pilots_per_token": np.int32(args.min_pilots_per_token),
        "min_subbundle": np.int32(args.min_subbundle),
    }
    for (step, partition), agg in aggregates_by_step_partition.items():
        if not agg.get("n_valid_tokens"):
            continue
        for key in ("baseline", "partition_kurt", "null_kurt",
                    "reduction", "null_reduction", "real_signal"):
            out_arrays[f"step_{step}_{partition}_{key}"] = agg[key]
        out_arrays[f"step_{step}_{partition}_n_valid_tokens"] = np.int32(
            agg["n_valid_tokens"])
        out_arrays[f"step_{step}_{partition}_n_sub_mean"] = np.float64(
            agg["n_sub_mean"])
    out_path = os.path.join(output_root(args.run_dir),
                            "d14_ngram_partition_fixed.npz")
    np.savez(out_path, **out_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates_by_step_partition,
                 seed=args.seed, layer=args.layer)


if __name__ == "__main__":
    main()
