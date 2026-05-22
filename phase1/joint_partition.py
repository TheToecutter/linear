"""
Joint-partition disambiguation for the sub-conditioning result.

Question: in the sub-conditioning analysis (d6), partitioning by
position reduced within-input kurtosis nearly as much as partitioning
by next-token. This could mean two things:

  (a) Position acts as a *proxy* for next-token in this held-out set
      (each pilot position is in a similar textual region, so the
      next-token distribution is locally homogeneous). Next-token is
      the real explanatory variable; position helps because they're
      correlated.

  (b) Position is itself the explanatory variable (something about
      where in the context window a token sits drives the heavy-tail
      structure). Next-token works mostly because it's correlated with
      context, which is correlated with position.

This script disambiguates by running the *joint* partition
(next_token, position) and comparing:

      kurtosis( bundle | input = v_i )                     baseline
      kurtosis( bundle | input = v_i, position = p )       pos only
      kurtosis( bundle | input = v_i, next = w )           next only
      kurtosis( bundle | input = v_i, next = w, pos = p )  joint

If joint gives a substantial reduction beyond position alone, next-token
adds real information -> interpretation (a) holds and the underlying
driver is local context (which next-token reflects more directly than
position).

If joint and position-only give the same kurtosis, position carries
all the information that next-token does -> interpretation (b), or at
least, in this dataset they are perfectly entangled.

Also computes a structural diagnostic: for each token, how
concentrated is the next-token distribution within each position-bucket?
A token where position p1 is dominated by "the followed by space" and
position p2 by "the followed by comma" would mean position and next are
heavily correlated by construction.

Output:
    run_dir/multiview/model_abc/d6b_joint_partition.npz
    run_dir/multiview/model_abc/figures/d6b_joint_partition.png

Usage:
    python joint_partition.py --run-dir ../phase1_runs_gelu
    python joint_partition.py --run-dir ../phase1_runs_gelu \\
        --min-subbundle 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiview import load_augmented_activations
from multiview_campaign import (
    checkpoints_in_seed,
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


# ----------------------------------------------------------------------
# Kurtosis on per-layer shared PC basis (same convention as d6).
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


def _global_basis(states: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, int, int]:
    """Sample-weighted mean kurtosis over sub-bundles defined by group_ids.

    Returns (per-layer kurtosis, n_valid_subbundles, n_pilots_used)
    """
    L = states_v.shape[0]
    unique, counts = np.unique(group_ids, return_counts=True)
    keep = unique[counts >= min_subbundle]
    if keep.size == 0:
        return np.full(L, np.nan), 0, 0
    weights, kurts = [], []
    n_used = 0
    for u in keep:
        mask = (group_ids == u)
        n_sub = int(mask.sum())
        sub = states_v[:, mask, :]
        k = _per_layer_kurtosis(sub, basis, means)
        kurts.append(k)
        weights.append(n_sub)
        n_used += n_sub
    K = np.stack(kurts)
    w = np.array(weights, dtype=np.float64); w /= w.sum()
    out = np.full(L, np.nan)
    for t in range(L):
        col = K[:, t]
        v = np.isfinite(col)
        if v.sum():
            out[t] = float(np.average(col[v], weights=w[v]))
    return out, int(keep.size), n_used


def _entanglement_index(next_ids: np.ndarray, positions: np.ndarray) -> float:
    """How concentrated is the next-token distribution within each
    position-bucket? Reports normalized mutual information
    I(next; pos) / H(next).

    Value near 0: next-token distribution is the same across positions
                  (next and position are independent in this dataset).
    Value near 1: each position has a unique next-token (perfect
                  entanglement; position alone determines next).
    """
    n = next_ids.size
    if n < 2:
        return float("nan")
    # Joint and marginal counts.
    # Use np.unique with return_inverse for compact indexing.
    nu, ni = np.unique(next_ids, return_inverse=True)
    pu, pi = np.unique(positions, return_inverse=True)
    joint = np.zeros((nu.size, pu.size), dtype=np.float64)
    np.add.at(joint, (ni, pi), 1.0)
    joint /= n
    p_next = joint.sum(1)
    p_pos = joint.sum(0)
    # MI = sum joint * log(joint / (p_next * p_pos))
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(p_next, p_pos)
        ratio = np.where((joint > 0) & (denom > 0), joint / denom, 1.0)
        mi = float(np.sum(joint * np.log(np.where(ratio > 0, ratio, 1.0))))
    h_next = -float(np.sum(p_next[p_next > 0] * np.log(p_next[p_next > 0])))
    if h_next <= 0:
        return float("nan")
    return mi / h_next


def analyze_token(
    states: np.ndarray,
    input_ids: np.ndarray, next_ids: np.ndarray, positions: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    token_id: int, min_subbundle: int,
    verbose: bool = False,
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
    pos_kurt, n_pos_groups, n_pos_used = _aggregate_kurt(
        states_v, pos_v, basis, means, min_subbundle)
    next_kurt, n_next_groups, n_next_used = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)
    # Joint partition: encode (next, pos) into a single integer label.
    joint_key = next_v.astype(np.int64) * (positions.max() + 10) + pos_v
    joint_kurt, n_joint_groups, n_joint_used = _aggregate_kurt(
        states_v, joint_key, basis, means, min_subbundle)

    nmi = _entanglement_index(next_v, pos_v)

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline_kurt": baseline,
        "pos_kurt": pos_kurt,
        "next_kurt": next_kurt,
        "joint_kurt": joint_kurt,
        "n_pos_groups": n_pos_groups,
        "n_next_groups": n_next_groups,
        "n_joint_groups": n_joint_groups,
        "n_pos_used": n_pos_used,
        "n_next_used": n_next_used,
        "n_joint_used": n_joint_used,
        "nmi_next_pos": nmi,
    }


def plot(run_dir: str, results: List[Dict],
         seed: int, step: int, min_subbundle: int) -> None:
    valid = [r for r in results if not r.get("insufficient", True)]
    if not valid:
        return
    L = valid[0]["baseline_kurt"].size
    layers = np.arange(L)

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

    baseline = _wmean("baseline_kurt")
    pos = _wmean("pos_kurt")
    nxt = _wmean("next_kurt")
    joint = _wmean("joint_kurt")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: aggregate kurtosis comparison.
    ax = axes[0]
    ax.plot(layers, baseline, "k-", lw=2.5, label="baseline | v_i")
    ax.plot(layers, pos, "C2--", lw=1.7, label="| v_i, position")
    ax.plot(layers, nxt, "C0-", lw=1.7, label="| v_i, next")
    ax.plot(layers, joint, "C3-", lw=2.0, label="| v_i, (next, position)")
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title(f"Aggregate (seed {seed} step {step}, "
                 f"min_subbundle={min_subbundle})")
    ax.legend(fontsize=9, loc="best")

    # Panel 2: kurtosis reduction relative to baseline.
    ax = axes[1]
    reduction_pos = baseline - pos
    reduction_next = baseline - nxt
    reduction_joint = baseline - joint
    ax.plot(layers, reduction_pos, "C2--", lw=1.7,
            label="reduction from position")
    ax.plot(layers, reduction_next, "C0-", lw=1.7,
            label="reduction from next-token")
    ax.plot(layers, reduction_joint, "C3-", lw=2.0,
            label="reduction from joint")
    # Gap between joint and pos = "what next-token adds beyond position"
    marginal_next_over_pos = reduction_joint - reduction_pos
    ax.plot(layers, marginal_next_over_pos, "k:", lw=1.5,
            label="next-token marginal over position")
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("kurtosis reduction from baseline")
    ax.set_title("Marginal explanatory power")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: NMI per token (entanglement of next and position).
    ax = axes[2]
    nmis = [r["nmi_next_pos"] for r in valid]
    tids = [r["token_id"] for r in valid]
    ax.bar(range(len(valid)), nmis, color="C4")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([str(t) for t in tids], rotation=45, fontsize=7)
    ax.set_ylabel("NMI( next ; position ) / H(next)")
    ax.set_ylim([0, max(0.5, max(nmis) * 1.1) if nmis else 1])
    ax.axhline(0.1, color="k", ls=":", lw=1,
               label="0.1 (effectively independent)")
    ax.set_xlabel("token id")
    ax.set_title("Are next-token and position entangled in this data?")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d6b_joint_partition.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--min-subbundle", type=int, default=20)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    if args.step is None:
        steps = [s for s, _ in checkpoints_in_seed(args.run_dir, args.seed)]
        args.step = steps[-1]

    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    aug_path = augmented_path(args.run_dir, args.seed, args.step)

    print(f"Loading {aug_path} ...")
    aug = load_augmented_activations(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    positions = aug["positions"]
    L, N, H = states.shape
    print(f"  states {states.shape}, pilots {N}")

    print(f"Global PC basis (top-{args.max_pc_dim}) per layer ...")
    basis, means = _global_basis(states, args.max_pc_dim)

    print(f"Joint-partition analysis on {tids.size} tokens "
          f"(min_subbundle={args.min_subbundle}):")
    results = []
    for ii, tok in enumerate(tids):
        r = analyze_token(states, input_ids, next_ids, positions,
                          basis, means, int(tok),
                          min_subbundle=args.min_subbundle)
        results.append(r)
        if r.get("insufficient"):
            print(f"  [{ii+1}/{tids.size}] tok {int(tok)}: insufficient "
                  f"(N={r['n_total']})")
            continue
        mid = L // 2
        print(f"  [{ii+1}/{tids.size}] tok {int(tok)} (N={r['n_total']}): "
              f"baseline={r['baseline_kurt'][mid]:.2f}  "
              f"pos={r['pos_kurt'][mid]:.2f}  "
              f"next={r['next_kurt'][mid]:.2f}  "
              f"joint={r['joint_kurt'][mid]:.2f}  "
              f"NMI={r['nmi_next_pos']:.3f}  "
              f"groups(pos/next/joint)={r['n_pos_groups']}/"
              f"{r['n_next_groups']}/{r['n_joint_groups']}")

    # Save.
    valid = [r for r in results if not r.get("insufficient", True)]
    if valid:
        out_path = os.path.join(output_root(args.run_dir),
                                "d6b_joint_partition.npz")
        baseline_arr = np.full((tids.size, L), np.nan)
        pos_arr = np.full((tids.size, L), np.nan)
        next_arr = np.full((tids.size, L), np.nan)
        joint_arr = np.full((tids.size, L), np.nan)
        nmi_arr = np.full(tids.size, np.nan)
        n_total = np.zeros(tids.size, dtype=np.int64)
        for k, tok in enumerate(tids):
            r = results[k]
            if r.get("insufficient"):
                n_total[k] = r.get("n_total", 0)
                continue
            baseline_arr[k] = r["baseline_kurt"]
            pos_arr[k] = r["pos_kurt"]
            next_arr[k] = r["next_kurt"]
            joint_arr[k] = r["joint_kurt"]
            nmi_arr[k] = r["nmi_next_pos"]
            n_total[k] = r["n_total"]
        np.savez(
            out_path,
            seed=np.int32(args.seed), step=np.int64(args.step),
            tids=tids, n_total=n_total,
            baseline_kurt=baseline_arr,
            pos_kurt=pos_arr,
            next_kurt=next_arr,
            joint_kurt=joint_arr,
            nmi_next_pos=nmi_arr,
            min_subbundle=np.int32(args.min_subbundle),
        )
        print(f"[npz] -> {out_path}")

    plot(args.run_dir, results, args.seed, args.step, args.min_subbundle)


if __name__ == "__main__":
    main()
