"""
Characterize the extreme-tail pilots driving the C-verdict heavy tails.

D12 (trimmed kurtosis) showed that trimming the top 5% most extreme
pilots (by Mahalanobis distance from the per-token mean) collapses the
conditional kurtosis to ~30% of baseline at Phase III final, and to
near zero at 10% trim. The heavy tails are driven by a small fraction
of extreme-context pilots.

This script characterizes those extreme pilots: for each input token in
the forward set, who are the top 5% Mahalanobis-distance pilots? What
are their immediate contexts (prev_token, next_token, position)? Are
they concentrated in particular grammatical contexts, or are they
uniformly distributed across the same context space as the bulk of the
bundle?

For each input token v_i, we:

  1. Compute per-pilot Mahalanobis distance using the per-token covariance
     at an interior layer (default layer 7), in the top-32 PC subspace.
  2. Identify the top 5% most extreme pilots.
  3. Compare the distributions of next_token, prev_token, position
     between the extreme set and the bulk set (the bottom 95%):
       - For categorical variables (next, prev): Top-k counts in each
         set, and the over-representation ratio
         (extreme_freq / bulk_freq) for the top contexts.
       - For position: KS-test between extreme and bulk position
         distributions, and quantile summary.
  4. Decode the top contexts with the Mistral tokenizer.

Uses the _ngram.npz augmented files (which have prev_ids) at Phase III
final. If you want to extend to all four phases, pass --all-steps.

Output:
    run_dir/multiview/model_abc/d16_extreme_tail_characterization.json
    run_dir/multiview/model_abc/figures/d16_extreme_tail_characterization.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scistats
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
# Mahalanobis distance in per-token covariance.
# ----------------------------------------------------------------------
def _mahalanobis_in_pc(
    states_t: np.ndarray, basis_t: np.ndarray, mean_t: np.ndarray,
) -> np.ndarray:
    """Mahalanobis distance in the global PC basis at one layer, using
    per-bundle covariance estimated from the same data."""
    n, _ = states_t.shape
    Z = (states_t.astype(np.float64) - mean_t) @ basis_t.T  # (n, d)
    mu = Z.mean(0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / max(n - 1, 1)
    # Regularize.
    cov = cov + 1e-6 * np.trace(cov) / Zc.shape[1] * np.eye(Zc.shape[1])
    try:
        L = np.linalg.cholesky(cov)
        sol = np.linalg.solve(L, Zc.T)
        d2 = (sol * sol).sum(axis=0)
    except np.linalg.LinAlgError:
        # Fall back to diagonal Mahalanobis.
        var = np.maximum(Zc.var(axis=0), 1e-12)
        d2 = ((Zc * Zc) / var).sum(axis=1)
    return d2


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


# ----------------------------------------------------------------------
# Tokenizer (lazy).
# ----------------------------------------------------------------------
_TOK = None
def _decode(tids):
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        except Exception as e:
            print(f"  [warn] could not load Mistral tokenizer: {e}")
            _TOK = "FAILED"
    if _TOK == "FAILED":
        return [f"<id {int(t)}>" for t in tids]
    out = []
    for t in tids:
        try:
            s = _TOK.decode([int(t)], skip_special_tokens=False)
            out.append(repr(s))
        except Exception:
            out.append(f"<id {int(t)}>")
    return out


# ----------------------------------------------------------------------
# Per-input-token characterization.
# ----------------------------------------------------------------------
def characterize_token(
    states_v: np.ndarray,
    next_v: np.ndarray, prev_v: np.ndarray, pos_v: np.ndarray,
    basis_t: np.ndarray, mean_t: np.ndarray,
    extreme_fraction: float, top_k_contexts: int,
    layer: int,
) -> Dict:
    n_total = states_v.shape[1]
    # Mahalanobis distance at the chosen layer.
    d2 = _mahalanobis_in_pc(states_v[layer], basis_t, mean_t)
    # Top-fraction by extremity.
    n_extreme = max(1, int(round(extreme_fraction * n_total)))
    order = np.argsort(d2)
    extreme_idx = order[-n_extreme:]
    bulk_idx = order[:n_total - n_extreme]
    # Context frequencies.
    def _top_contexts(values: np.ndarray, idx: np.ndarray) -> List[Tuple[int, int]]:
        u, c = np.unique(values[idx], return_counts=True)
        order_c = np.argsort(c)[::-1]
        return [(int(u[i]), int(c[i])) for i in order_c[:top_k_contexts]]

    def _frequency_table(values: np.ndarray, idx_a: np.ndarray,
                          idx_b: np.ndarray) -> Dict:
        """For each unique value, report extreme_freq, bulk_freq, and
        the ratio. Returns the top-K by extreme_freq."""
        u_all = np.unique(values)
        out = []
        for u in u_all:
            n_a = int((values[idx_a] == u).sum())
            n_b = int((values[idx_b] == u).sum())
            f_a = n_a / max(idx_a.size, 1)
            f_b = n_b / max(idx_b.size, 1)
            # Smoothed ratio to avoid divide-by-zero for novel-in-extreme.
            ratio = (f_a + 1e-6) / (f_b + 1e-6)
            out.append({
                "value": int(u),
                "n_extreme": n_a, "n_bulk": n_b,
                "f_extreme": float(f_a), "f_bulk": float(f_b),
                "ratio": float(ratio),
            })
        # Sort by n_extreme descending, then ratio descending.
        out.sort(key=lambda r: (-r["n_extreme"], -r["ratio"]))
        return out[:top_k_contexts]

    # KS test on position distribution.
    if pos_v[extreme_idx].size >= 5 and pos_v[bulk_idx].size >= 5:
        try:
            ks_stat, ks_p = scistats.ks_2samp(
                pos_v[extreme_idx], pos_v[bulk_idx])
        except Exception:
            ks_stat, ks_p = float("nan"), float("nan")
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    return {
        "n_total": int(n_total),
        "n_extreme": int(n_extreme),
        "mean_d2_extreme": float(d2[extreme_idx].mean()),
        "mean_d2_bulk": float(d2[bulk_idx].mean()),
        "extreme_over_bulk_d2_ratio":
            float(d2[extreme_idx].mean() / max(d2[bulk_idx].mean(), 1e-9)),
        "next_table": _frequency_table(next_v, extreme_idx, bulk_idx),
        "prev_table": _frequency_table(prev_v, extreme_idx, bulk_idx),
        "position_extreme_quantiles": {
            "p25": float(np.percentile(pos_v[extreme_idx], 25)),
            "p50": float(np.percentile(pos_v[extreme_idx], 50)),
            "p75": float(np.percentile(pos_v[extreme_idx], 75)),
        },
        "position_bulk_quantiles": {
            "p25": float(np.percentile(pos_v[bulk_idx], 25)),
            "p50": float(np.percentile(pos_v[bulk_idx], 50)),
            "p75": float(np.percentile(pos_v[bulk_idx], 75)),
        },
        "position_ks_stat": float(ks_stat),
        "position_ks_pvalue": float(ks_p),
    }


# ----------------------------------------------------------------------
# Cross-token aggregation.
# ----------------------------------------------------------------------
def aggregate_across_tokens(per_token: Dict, top_k_contexts: int) -> Dict:
    """Build a global picture of what kinds of contexts dominate the
    extreme tails across all input tokens."""
    next_pool, prev_pool = {}, {}
    for tid, info in per_token.items():
        for entry in info["next_table"]:
            v = entry["value"]
            next_pool.setdefault(v, {"n_extreme_total": 0,
                                      "n_bulk_total": 0,
                                      "appears_in_tokens": 0})
            next_pool[v]["n_extreme_total"] += entry["n_extreme"]
            next_pool[v]["n_bulk_total"] += entry["n_bulk"]
            if entry["n_extreme"] > 0:
                next_pool[v]["appears_in_tokens"] += 1
        for entry in info["prev_table"]:
            v = entry["value"]
            prev_pool.setdefault(v, {"n_extreme_total": 0,
                                      "n_bulk_total": 0,
                                      "appears_in_tokens": 0})
            prev_pool[v]["n_extreme_total"] += entry["n_extreme"]
            prev_pool[v]["n_bulk_total"] += entry["n_bulk"]
            if entry["n_extreme"] > 0:
                prev_pool[v]["appears_in_tokens"] += 1

    def _rank(pool, by="n_extreme_total"):
        items = list(pool.items())
        items.sort(key=lambda kv: -kv[1][by])
        return items[:top_k_contexts]

    return {
        "top_next_tokens_in_extreme_tails": _rank(next_pool),
        "top_prev_tokens_in_extreme_tails": _rank(prev_pool),
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(
    run_dir: str, per_token: Dict, aggregate: Dict, layer: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: top next-tokens that appear in extreme tails (global view).
    ax = axes[0]
    items = aggregate["top_next_tokens_in_extreme_tails"][:15]
    tids = [item[0] for item in items]
    counts = [item[1]["n_extreme_total"] for item in items]
    appears = [item[1]["appears_in_tokens"] for item in items]
    decoded = _decode(tids)
    y = np.arange(len(items))
    ax.barh(y, counts, color="C0")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} ({a}/n_tok)"
                        for d, a in zip(decoded, appears)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("total extreme-pilot count across forward-set tokens")
    ax.set_title("Top next-tokens in extreme tails (annotation: token-input-count)")

    # Panel 2: top prev-tokens.
    ax = axes[1]
    items = aggregate["top_prev_tokens_in_extreme_tails"][:15]
    tids = [item[0] for item in items]
    counts = [item[1]["n_extreme_total"] for item in items]
    appears = [item[1]["appears_in_tokens"] for item in items]
    decoded = _decode(tids)
    y = np.arange(len(items))
    ax.barh(y, counts, color="C1")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d} ({a}/n_tok)"
                        for d, a in zip(decoded, appears)],
                       fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("total extreme-pilot count across forward-set tokens")
    ax.set_title("Top prev-tokens in extreme tails (annotation: token-input-count)")

    # Panel 3: position quantile comparison (extreme vs bulk).
    ax = axes[2]
    tids = sorted(per_token.keys())
    decoded = _decode(tids)
    n = len(tids)
    width = 0.4
    x = np.arange(n)
    extreme_p50 = [per_token[t]["position_extreme_quantiles"]["p50"]
                   for t in tids]
    bulk_p50 = [per_token[t]["position_bulk_quantiles"]["p50"]
                for t in tids]
    ax.barh(x - width/2, extreme_p50, width, color="C3", label="extreme p50")
    ax.barh(x + width/2, bulk_p50, width, color="C2", label="bulk p50")
    ax.set_yticks(x)
    ax.set_yticklabels(decoded, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("median pilot position")
    ax.set_title("Median position: extreme vs bulk per input token")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"D16: extreme-tail pilot characterization at layer {layer}, "
        f"seed 0, Phase III final",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir),
                       "d16_extreme_tail_characterization.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=24000,
                    help="Checkpoint to characterize (default Phase III final).")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--extreme-fraction", type=float, default=0.05)
    ap.add_argument("--top-k-contexts", type=int, default=10)
    ap.add_argument("--min-pilots", type=int, default=40,
                    help="Skip input tokens with fewer pilots.")
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)

    aug_path = ngram_augmented_path(args.run_dir, args.seed, args.step)
    if not os.path.exists(aug_path):
        print(f"ERROR: missing {aug_path}")
        sys.exit(1)
    print(f"Loading {aug_path} ...")
    aug = load_ngram_payload(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    prev_ids = aug["prev_ids"]
    positions = aug["positions"]
    L, N, H = states.shape
    print(f"  ({L=}, {N=}, {H=})")

    print(f"Computing top-{args.max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, args.max_pc_dim)

    print(f"\nCharacterizing extreme tails at layer {args.layer}, "
          f"extreme fraction {args.extreme_fraction:.0%}")
    print(f"{'='*72}")
    per_token = {}
    for tok in tids:
        mask = (input_ids == int(tok))
        n_total = int(mask.sum())
        if n_total < args.min_pilots:
            continue
        states_v = states[:, mask, :]
        next_v = next_ids[mask]
        prev_v = prev_ids[mask]
        pos_v = positions[mask]
        info = characterize_token(
            states_v, next_v, prev_v, pos_v,
            basis[args.layer], means[args.layer],
            extreme_fraction=args.extreme_fraction,
            top_k_contexts=args.top_k_contexts,
            layer=args.layer,
        )
        per_token[int(tok)] = info

    # Print per-token summaries.
    print(f"\n{'Tok':>6s}  {'decoded':<14s}  {'N':>5s}  "
          f"{'d2_x/b':>8s}  {'pos_p50_x':>10s}  {'pos_p50_b':>10s}  "
          f"{'ks_p':>8s}  top-3 next in extreme")
    for tid, info in sorted(per_token.items()):
        decoded = _decode([tid])[0]
        next_top = info["next_table"][:3]
        next_decoded = _decode([e["value"] for e in next_top])
        next_summary = ", ".join(
            f"{d}({e['n_extreme']})"
            for d, e in zip(next_decoded, next_top))
        print(f"{tid:>6d}  {decoded:<14s}  {info['n_total']:>5d}  "
              f"{info['extreme_over_bulk_d2_ratio']:>8.2f}  "
              f"{info['position_extreme_quantiles']['p50']:>10.1f}  "
              f"{info['position_bulk_quantiles']['p50']:>10.1f}  "
              f"{info['position_ks_pvalue']:>8.4f}  "
              f"{next_summary}")

    print(f"\n{'='*72}")
    print(f"Aggregated view across input tokens:")
    agg = aggregate_across_tokens(per_token,
                                   top_k_contexts=args.top_k_contexts * 2)
    print(f"\nTop next-tokens appearing in extreme tails (pooled across "
          f"all input tokens):")
    print(f"  {'next_id':>8s}  {'decoded':<14s}  {'n_ext':>6s}  "
          f"{'n_bulk':>7s}  {'n_input_toks':>13s}  {'ratio':>7s}")
    for next_tid, stats in agg["top_next_tokens_in_extreme_tails"][:15]:
        decoded = _decode([next_tid])[0]
        ne = stats["n_extreme_total"]
        nb = stats["n_bulk_total"]
        ratio = ((ne / max(ne + nb, 1)) /
                 (0.05))     # vs 5% expected if uniform across extreme/bulk
        print(f"  {next_tid:>8d}  {decoded:<14s}  {ne:>6d}  {nb:>7d}  "
              f"{stats['appears_in_tokens']:>13d}  {ratio:>7.2f}")

    print(f"\nTop prev-tokens appearing in extreme tails (pooled):")
    print(f"  {'prev_id':>8s}  {'decoded':<14s}  {'n_ext':>6s}  "
          f"{'n_bulk':>7s}  {'n_input_toks':>13s}  {'ratio':>7s}")
    for prev_tid, stats in agg["top_prev_tokens_in_extreme_tails"][:15]:
        decoded = _decode([prev_tid])[0]
        ne = stats["n_extreme_total"]
        nb = stats["n_bulk_total"]
        ratio = ((ne / max(ne + nb, 1)) / 0.05)
        print(f"  {prev_tid:>8d}  {decoded:<14s}  {ne:>6d}  {nb:>7d}  "
              f"{stats['appears_in_tokens']:>13d}  {ratio:>7.2f}")

    # Save.
    payload = {
        "step": int(args.step),
        "seed": int(args.seed),
        "layer": int(args.layer),
        "extreme_fraction": float(args.extreme_fraction),
        "per_token": {str(k): v for k, v in per_token.items()},
        "aggregate": {
            "top_next_tokens_in_extreme_tails": [
                {
                    "tid": int(tid),
                    "decoded": _decode([tid])[0],
                    "n_extreme_total": int(s["n_extreme_total"]),
                    "n_bulk_total": int(s["n_bulk_total"]),
                    "appears_in_tokens": int(s["appears_in_tokens"]),
                }
                for tid, s in agg["top_next_tokens_in_extreme_tails"]
            ],
            "top_prev_tokens_in_extreme_tails": [
                {
                    "tid": int(tid),
                    "decoded": _decode([tid])[0],
                    "n_extreme_total": int(s["n_extreme_total"]),
                    "n_bulk_total": int(s["n_bulk_total"]),
                    "appears_in_tokens": int(s["appears_in_tokens"]),
                }
                for tid, s in agg["top_prev_tokens_in_extreme_tails"]
            ],
        },
    }
    out_path = os.path.join(output_root(args.run_dir),
                            "d16_extreme_tail_characterization.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[json] -> {out_path}")

    plot_results(args.run_dir, per_token, agg, args.layer)


if __name__ == "__main__":
    main()
