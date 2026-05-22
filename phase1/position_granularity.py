"""
Position granularity experiment.

The Path 2 result showed position is a more effective mixture index than
next-token for the C-verdict bundle structure. This experiment asks at
what *resolution* the relevant positional structure lives: does the
effect saturate at coarse (~2-5) bins, indicating a simple early-vs-
late dichotomy, or does it require fine resolution (~20+ bins) to
recover?

Three null/treatment families, swept over bin counts k:

  Quantile bins:   each token's pilots split into k equal-count buckets
                   by their position values.

  Uniform bins:    each token's pilots split into k equal-width buckets
                   on the raw position axis (0..T-2).

  Random shuffle:  each pilot assigned a random integer in [0, k). This
                   is the *null* control: it shows how much kurtosis
                   reduction comes purely from sample partitioning,
                   regardless of any covariate structure. Repeat with
                   multiple random seeds and average.

Plus reference signal:

  Next-token:      conditioning on the actual next token in text.

For each checkpoint (one per training phase), bin count k, and
condition, compute the per-token aggregate kurtosis at layer 7
(representative interior layer where Phase III heavy tails are strong)
and the full per-layer profile.

The diagnostic statistic is the *fraction of saturation* achieved at
each k, relative to the maximum reduction across all tested k values.
If 80% of reduction is achieved by k=3, structure is coarse. If 80%
requires k>=15, structure is fine.

We use the randomized stage A files (suffix _random.npz) so that
position is decorrelated from chunk identity.

Output:
    run_dir/multiview/model_abc/d10_position_granularity.npz
    run_dir/multiview/model_abc/figures/d10_position_granularity.png

Usage:
    python position_granularity.py --run-dir ../phase1_runs_gelu
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


# ----------------------------------------------------------------------
# Phase labels.
# ----------------------------------------------------------------------
PHASE_LABELS = {
    479:   "Phase I",
    2563:  "Phase II",
    9809:  "Phase III mid",
    24000: "Phase III final",
}


# ----------------------------------------------------------------------
# Kurtosis helpers (same convention as d9 / subcondition).
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
    """Sample-size-weighted mean kurtosis across sub-bundles defined
    by group_ids."""
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
# Binning schemes.
# ----------------------------------------------------------------------
def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin into n_bins equal-count quantile buckets."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros_like(values, dtype=np.int32)
    binned = np.searchsorted(edges, values, side="right") - 1
    return np.clip(binned, 0, len(edges) - 2).astype(np.int32)


def _uniform_bin(values: np.ndarray, n_bins: int,
                 v_min: int, v_max: int) -> np.ndarray:
    """Bin into n_bins equal-width buckets on the raw position axis."""
    if v_max <= v_min:
        return np.zeros_like(values, dtype=np.int32)
    edges = np.linspace(v_min, v_max + 1, n_bins + 1)
    binned = np.searchsorted(edges, values, side="right") - 1
    return np.clip(binned, 0, n_bins - 1).astype(np.int32)


def _random_labels(n: int, n_bins: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Assign each of n pilots a uniformly random bucket label in
    [0, n_bins)."""
    return rng.integers(0, n_bins, size=n, dtype=np.int32)


# ----------------------------------------------------------------------
# Per-token analysis.
# ----------------------------------------------------------------------
def analyze_token(
    states: np.ndarray,
    input_ids: np.ndarray,
    next_ids: np.ndarray,
    positions: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    token_id: int, min_subbundle: int,
    bin_ladder: List[int], n_random_reps: int,
    rng: np.random.Generator,
    seq_len: int,
) -> Dict:
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_subbundle * 2:
        return {"insufficient": True, "n_total": n_total}

    states_v = states[:, mask, :]
    next_v = next_ids[mask]
    pos_v = positions[mask]
    L = states_v.shape[0]

    baseline = _per_layer_kurtosis(states_v, basis, means)
    next_kurt, n_next_sub = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)

    # Position binnings: quantile + uniform across the full ladder.
    quantile_kurts: Dict[int, np.ndarray] = {}
    quantile_nsub: Dict[int, int] = {}
    uniform_kurts: Dict[int, np.ndarray] = {}
    uniform_nsub: Dict[int, int] = {}
    for k in bin_ladder:
        qb = _quantile_bin(pos_v, k)
        kurt_q, n_sub_q = _aggregate_kurt(
            states_v, qb, basis, means, min_subbundle)
        quantile_kurts[k] = kurt_q
        quantile_nsub[k] = n_sub_q
        ub = _uniform_bin(pos_v, k, v_min=0, v_max=seq_len - 2)
        kurt_u, n_sub_u = _aggregate_kurt(
            states_v, ub, basis, means, min_subbundle)
        uniform_kurts[k] = kurt_u
        uniform_nsub[k] = n_sub_u

    # Random shuffle null at each k. Average n_random_reps independent
    # random labelings to get a stable estimate of the bucket-count
    # artifact baseline.
    random_kurts: Dict[int, np.ndarray] = {}
    random_kurts_std: Dict[int, np.ndarray] = {}
    for k in bin_ladder:
        reps = []
        for _ in range(n_random_reps):
            rl = _random_labels(n_total, k, rng)
            kurt_r, _ = _aggregate_kurt(
                states_v, rl, basis, means, min_subbundle)
            reps.append(kurt_r)
        reps_arr = np.stack(reps)
        random_kurts[k] = np.nanmean(reps_arr, axis=0)
        random_kurts_std[k] = np.nanstd(reps_arr, axis=0, ddof=1)

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline": baseline,
        "next": next_kurt,
        "quantile_kurts": quantile_kurts,
        "quantile_nsub": quantile_nsub,
        "uniform_kurts": uniform_kurts,
        "uniform_nsub": uniform_nsub,
        "random_kurts": random_kurts,
        "random_kurts_std": random_kurts_std,
    }


def analyze_checkpoint(
    run_dir: str, seed: int, step: int,
    tids: np.ndarray, min_subbundle: int,
    bin_ladder: List[int], n_random_reps: int,
    max_pc_dim: int, seq_len: int,
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
    positions = aug["positions"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")

    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    for tok in tids:
        results[int(tok)] = analyze_token(
            states, input_ids, next_ids, positions,
            basis, means, int(tok),
            min_subbundle=min_subbundle,
            bin_ladder=bin_ladder,
            n_random_reps=n_random_reps,
            rng=rng,
            seq_len=seq_len,
        )

    valid = [r for r in results.values() if not r.get("insufficient", True)]
    print(f"    {len(valid)}/{tids.size} tokens with sufficient samples")
    return {
        "seed": seed, "step": step,
        "results_per_token": results,
    }


# ----------------------------------------------------------------------
# Aggregate per-token results to mean profiles.
# ----------------------------------------------------------------------
def aggregate_profiles(
    checkpoint_results: Dict, bin_ladder: List[int],
) -> Dict:
    valid = [r for r in checkpoint_results["results_per_token"].values()
             if not r.get("insufficient", True)]
    if not valid:
        return {}
    L = valid[0]["baseline"].size
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights /= weights.sum()

    def _wmean(extractor) -> np.ndarray:
        arr = np.stack([extractor(r) for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]
            v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=weights[v]))
        return out

    profiles = {
        "baseline": _wmean(lambda r: r["baseline"]),
        "next": _wmean(lambda r: r["next"]),
        "quantile": {k: _wmean(lambda r, kk=k: r["quantile_kurts"][kk])
                      for k in bin_ladder},
        "uniform":  {k: _wmean(lambda r, kk=k: r["uniform_kurts"][kk])
                      for k in bin_ladder},
        "random":   {k: _wmean(lambda r, kk=k: r["random_kurts"][kk])
                      for k in bin_ladder},
    }
    return profiles


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(
    run_dir: str,
    checkpoint_aggregates: Dict[int, Dict],
    bin_ladder: List[int],
    layer_of_interest: int,
    seed: int,
) -> None:
    steps_present = sorted(checkpoint_aggregates.keys())
    if not steps_present:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps_present)))

    # Panel 1: kurtosis reduction at layer_of_interest vs bin count,
    # one line per checkpoint, for quantile binning.
    ax = axes[0, 0]
    for ci, step in enumerate(steps_present):
        prof = checkpoint_aggregates[step]
        if not prof:
            continue
        baseline_at_t = prof["baseline"][layer_of_interest]
        reductions = [
            baseline_at_t - prof["quantile"][k][layer_of_interest]
            for k in bin_ladder
        ]
        ax.plot(bin_ladder, reductions, "o-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, f'step {step}')} "
                      f"(base={baseline_at_t:.2f})")
    ax.set_xscale("log")
    ax.set_xlabel("number of position bins (quantile)")
    ax.set_ylabel(f"kurtosis reduction at layer {layer_of_interest}")
    ax.set_title("Quantile binning: kurtosis reduction vs bin count")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(bin_ladder)
    ax.set_xticklabels([str(k) for k in bin_ladder])

    # Panel 2: same but for uniform binning.
    ax = axes[0, 1]
    for ci, step in enumerate(steps_present):
        prof = checkpoint_aggregates[step]
        if not prof:
            continue
        baseline_at_t = prof["baseline"][layer_of_interest]
        reductions = [
            baseline_at_t - prof["uniform"][k][layer_of_interest]
            for k in bin_ladder
        ]
        ax.plot(bin_ladder, reductions, "s-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, f'step {step}')}")
    ax.set_xscale("log")
    ax.set_xlabel("number of position bins (uniform on raw axis)")
    ax.set_ylabel(f"kurtosis reduction at layer {layer_of_interest}")
    ax.set_title("Uniform binning: kurtosis reduction vs bin count")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(bin_ladder)
    ax.set_xticklabels([str(k) for k in bin_ladder])

    # Panel 3: random null vs both binning schemes vs next-token, at
    # Phase III final.
    ax = axes[1, 0]
    final_step = steps_present[-1]
    prof = checkpoint_aggregates[final_step]
    if prof:
        base = prof["baseline"][layer_of_interest]
        q_red = [base - prof["quantile"][k][layer_of_interest]
                 for k in bin_ladder]
        u_red = [base - prof["uniform"][k][layer_of_interest]
                 for k in bin_ladder]
        r_red = [base - prof["random"][k][layer_of_interest]
                 for k in bin_ladder]
        next_red = base - prof["next"][layer_of_interest]
        ax.plot(bin_ladder, q_red, "o-", color="C0", lw=2,
                label="position (quantile)")
        ax.plot(bin_ladder, u_red, "s-", color="C2", lw=2,
                label="position (uniform)")
        ax.plot(bin_ladder, r_red, "d-", color="gray", lw=2,
                label="random labels (null)")
        ax.axhline(next_red, color="C3", ls="--", lw=2,
                   label=f"next-token reduction = {next_red:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("number of bins")
        ax.set_ylabel(f"kurtosis reduction at layer {layer_of_interest}")
        ax.set_title(
            f"All schemes at {PHASE_LABELS.get(final_step, 'final')}, "
            f"layer {layer_of_interest}")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(bin_ladder)
        ax.set_xticklabels([str(k) for k in bin_ladder])

    # Panel 4: saturation curve. Fraction of maximum reduction achieved
    # at each k, for quantile binning, per checkpoint. Saturating near
    # 1.0 quickly = coarse structure. Slow rise = fine structure.
    ax = axes[1, 1]
    for ci, step in enumerate(steps_present):
        prof = checkpoint_aggregates[step]
        if not prof:
            continue
        base = prof["baseline"][layer_of_interest]
        reds = np.array([
            base - prof["quantile"][k][layer_of_interest]
            for k in bin_ladder
        ])
        nulls = np.array([
            base - prof["random"][k][layer_of_interest]
            for k in bin_ladder
        ])
        # Signal = reduction minus null. Saturation against max signal.
        signal = reds - nulls
        max_sig = signal.max() if signal.max() > 0 else 1.0
        frac = signal / max_sig
        ax.plot(bin_ladder, frac, "o-", color=cmap[ci], lw=2,
                label=f"{PHASE_LABELS.get(step, f'step {step}')}")
    ax.axhline(0.8, color="k", lw=0.7, ls=":",
               label="80% saturation (granularity floor)")
    ax.set_xscale("log")
    ax.set_xlabel("number of position bins (quantile)")
    ax.set_ylabel("fraction of max signal reduction")
    ax.set_title("Saturation curve: how coarse is the position effect?")
    ax.legend(fontsize=8, loc="best")
    ax.set_xticks(bin_ladder)
    ax.set_xticklabels([str(k) for k in bin_ladder])

    fig.suptitle(
        f"D10: position-effect granularity, seed {seed}, randomized stage A",
        fontsize=13, y=1.005,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d10_position_granularity.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
DEFAULT_STEPS = [479, 2563, 9809, 24000]
DEFAULT_BIN_LADDER = [2, 3, 4, 5, 7, 10, 15, 20, 30]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--bin-ladder", type=int, nargs="+",
                    default=DEFAULT_BIN_LADDER)
    ap.add_argument("--n-random-reps", type=int, default=10,
                    help="Number of independent random labelings to "
                         "average per k for the null control.")
    ap.add_argument("--min-subbundle", type=int, default=20)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--layer", type=int, default=7,
                    help="Layer index for the summary statistics.")
    ap.add_argument("--seq-len", type=int, default=1024,
                    help="Sequence length T (for uniform binning bounds).")
    ap.add_argument("--rng-seed", type=int, default=20260521)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")
    print(f"Checkpoints: {args.steps}")
    print(f"Bin ladder: {args.bin_ladder}")
    print(f"Random null reps per k: {args.n_random_reps}")

    rng = np.random.default_rng(args.rng_seed)

    aggregates: Dict[int, Dict] = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, f'step {step}')}:")
        result = analyze_checkpoint(
            args.run_dir, args.seed, step, tids,
            min_subbundle=args.min_subbundle,
            bin_ladder=args.bin_ladder,
            n_random_reps=args.n_random_reps,
            max_pc_dim=args.max_pc_dim,
            seq_len=args.seq_len,
            rng=rng,
        )
        if result is None:
            continue
        aggregates[step] = aggregate_profiles(result, args.bin_ladder)
        prof = aggregates[step]
        if prof:
            base = prof["baseline"][args.layer]
            print(f"    layer {args.layer} baseline = {base:.3f}")
            print(f"    layer {args.layer} reductions:")
            print(f"    {'k':>4s}  {'quantile':>10s}  {'uniform':>10s}  "
                  f"{'random null':>12s}  {'q - null':>10s}")
            for k in args.bin_ladder:
                q = base - prof["quantile"][k][args.layer]
                u = base - prof["uniform"][k][args.layer]
                r = base - prof["random"][k][args.layer]
                print(f"    {k:>4d}  {q:>10.3f}  {u:>10.3f}  "
                      f"{r:>12.3f}  {q - r:>10.3f}")
            next_red = base - prof["next"][args.layer]
            print(f"    next-token reduction = {next_red:.3f} (reference)")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed:.1f}s")

    # Save raw arrays.
    out_arrays = {
        "steps": np.array(sorted(aggregates.keys()), dtype=np.int64),
        "bin_ladder": np.array(args.bin_ladder, dtype=np.int32),
        "layer_of_interest": np.int32(args.layer),
        "seed": np.int32(args.seed),
    }
    for step, prof in aggregates.items():
        out_arrays[f"step_{step}_baseline"] = prof["baseline"]
        out_arrays[f"step_{step}_next"] = prof["next"]
        for k in args.bin_ladder:
            out_arrays[f"step_{step}_quantile_{k}"] = prof["quantile"][k]
            out_arrays[f"step_{step}_uniform_{k}"]  = prof["uniform"][k]
            out_arrays[f"step_{step}_random_{k}"]   = prof["random"][k]
    out_path = os.path.join(output_root(args.run_dir),
                            "d10_position_granularity.npz")
    np.savez(out_path, **out_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates, args.bin_ladder,
                 layer_of_interest=args.layer, seed=args.seed)


if __name__ == "__main__":
    main()
