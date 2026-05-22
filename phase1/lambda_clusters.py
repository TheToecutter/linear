"""
Lambda-cluster analysis: characterize the bimodal lambda distribution
found in D3.

The D3 final-checkpoint scatter showed a bimodal per-token lambda
distribution:

  bulk cluster:  lambda ~ 0.55-0.60  (~15 tokens)
  low cluster:   lambda ~ 0.40       (~5 tokens, suspected
                                       punctuation/whitespace-adjacent)

This script:

  1. Loads d3_per_token_fits.npz to get per-token lambda values.
  2. Uses k-means (k=2) to cluster the per-token lambdas at the final
     checkpoint, averaged across seeds.
  3. Decodes the token ids using the Mistral tokenizer and prints the
     two clusters with their lexical content.
  4. For each diagnostic available on disk -- trace(Sigma_i), effective
     rank profile, kurtosis profile, log_alpha -- computes cluster-mean
     profiles and a Welch t-test of cluster-mean equality at each layer.
  5. Tests cross-seed stability: does the cluster membership at the
     final checkpoint hold across all four seeds?
  6. Tests cross-step stability: when does the bimodality emerge during
     training? Plots cluster separation through training.
  7. Produces a multi-panel figure summarizing the findings.

Output:
    run_dir/multiview/model_abc/d7_lambda_clusters.json
        (cluster membership, decoded tokens, cluster statistics)
    run_dir/multiview/model_abc/figures/d7_lambda_clusters.png

Usage:
    python lambda_clusters.py --run-dir ../phase1_runs_gelu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scistats


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


# ----------------------------------------------------------------------
# Cluster on per-token lambda values.
# ----------------------------------------------------------------------
def kmeans_1d(values: np.ndarray, k: int = 2, n_iter: int = 100,
              rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """1-D k-means. Returns (centers sorted ascending, labels in [0..k-1]
    aligned to the sorted centers)."""
    if rng is None:
        rng = np.random.default_rng(20260521)
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    v = values[finite]
    if v.size < k:
        return np.full(k, np.nan), np.full(values.shape, -1, dtype=np.int32)
    # k-means++ init: pick min and max as initial seeds for k=2.
    centers = np.array([v.min(), v.max()]) if k == 2 else \
              rng.choice(v, size=k, replace=False)
    for _ in range(n_iter):
        # Assign.
        dist = (v[:, None] - centers[None, :]) ** 2
        lbl = dist.argmin(axis=1)
        new_centers = np.array([
            v[lbl == j].mean() if (lbl == j).any() else centers[j]
            for j in range(k)
        ])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    # Sort by center.
    order = np.argsort(centers)
    centers = centers[order]
    # Remap labels to sorted order.
    inverse = np.argsort(order)
    new_lbl = inverse[lbl]
    full_lbl = np.full(values.shape, -1, dtype=np.int32)
    full_lbl[finite] = new_lbl
    return centers, full_lbl


# ----------------------------------------------------------------------
# Decode tokens.
# ----------------------------------------------------------------------
def decode_tokens(tids: np.ndarray) -> List[str]:
    """Decode token ids using the Mistral tokenizer. Returns repr-like
    strings showing whitespace explicitly."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return [f"<id {int(t)}>" for t in tids]
    try:
        tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    except Exception as e:
        print(f"  [warn] could not load Mistral tokenizer: {e}")
        return [f"<id {int(t)}>" for t in tids]
    out = []
    for t in tids:
        try:
            s = tok.decode([int(t)], skip_special_tokens=False)
            # Make whitespace visible.
            s_repr = repr(s)
            out.append(s_repr)
        except Exception:
            out.append(f"<id {int(t)}>")
    return out


# ----------------------------------------------------------------------
# Welch t-test cluster comparison, per layer.
# ----------------------------------------------------------------------
def per_layer_cluster_test(
    arr: np.ndarray,            # (n_tokens, L) or (n_seeds, n_tokens, L)
    labels: np.ndarray,         # (n_tokens,) in {0, 1, -1=invalid}
) -> Dict[str, np.ndarray]:
    """Welch's t-test between cluster 0 and cluster 1 at each layer.

    If arr is 3-D (seeds x tokens x layers), pools across seeds by
    averaging the same token across seeds first (n_tokens samples per
    cluster per layer).
    """
    if arr.ndim == 3:
        # Average across seeds: per-token mean over seeds.
        arr_avg = np.nanmean(arr, axis=0)        # (n_tokens, L)
    else:
        arr_avg = arr
    n_tok, L = arr_avg.shape
    mask0 = (labels == 0)
    mask1 = (labels == 1)
    mean0 = np.full(L, np.nan)
    mean1 = np.full(L, np.nan)
    std0 = np.full(L, np.nan)
    std1 = np.full(L, np.nan)
    tstat = np.full(L, np.nan)
    pvalue = np.full(L, np.nan)
    for t in range(L):
        a = arr_avg[mask0, t]
        b = arr_avg[mask1, t]
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size >= 2 and b.size >= 2:
            mean0[t] = a.mean(); std0[t] = a.std(ddof=1)
            mean1[t] = b.mean(); std1[t] = b.std(ddof=1)
            try:
                t_, p_ = scistats.ttest_ind(a, b, equal_var=False)
                tstat[t] = float(t_); pvalue[t] = float(p_)
            except Exception:
                pass
    return {
        "mean_cluster0": mean0, "mean_cluster1": mean1,
        "std_cluster0": std0, "std_cluster1": std1,
        "tstat": tstat, "pvalue": pvalue,
    }


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------
def plot_clusters(
    run_dir: str,
    tids: np.ndarray, lambdas_final: np.ndarray,
    labels: np.ndarray, centers: np.ndarray,
    trace_test: Dict, erank_test: Dict, kurt_test: Dict,
    steps: np.ndarray, lambdas_through_training: np.ndarray,
) -> None:
    """Six-panel figure summarizing the cluster analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    color_low = "#d62728"      # red
    color_high = "#1f77b4"     # blue
    cmap = {0: color_low, 1: color_high, -1: "gray"}

    # Panel 1: per-token lambda histogram with cluster colors.
    ax = axes[0, 0]
    lo_mask = (labels == 0); hi_mask = (labels == 1)
    ax.hist(lambdas_final[hi_mask], bins=12, color=color_high, alpha=0.7,
            label=f"high cluster (n={hi_mask.sum()}, mean={centers[1]:.3f})")
    ax.hist(lambdas_final[lo_mask], bins=8, color=color_low, alpha=0.7,
            label=f"low cluster  (n={lo_mask.sum()}, mean={centers[0]:.3f})")
    ax.axvline(centers[0], color=color_low, ls="--", lw=1.2)
    ax.axvline(centers[1], color=color_high, ls="--", lw=1.2)
    ax.set_xlabel("per-token lambda (avg over seeds, final ckpt)")
    ax.set_ylabel("count")
    ax.set_title("D7.1: lambda bimodality")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: cluster trace(Sigma_i) profile across layers, with t-test
    # significance shading.
    ax = axes[0, 1]
    L = trace_test["mean_cluster0"].size
    layers = np.arange(L)
    ax.plot(layers, trace_test["mean_cluster0"], color=color_low, lw=2,
            label="low cluster mean")
    ax.fill_between(layers,
                    trace_test["mean_cluster0"] - trace_test["std_cluster0"],
                    trace_test["mean_cluster0"] + trace_test["std_cluster0"],
                    color=color_low, alpha=0.18)
    ax.plot(layers, trace_test["mean_cluster1"], color=color_high, lw=2,
            label="high cluster mean")
    ax.fill_between(layers,
                    trace_test["mean_cluster1"] - trace_test["std_cluster1"],
                    trace_test["mean_cluster1"] + trace_test["std_cluster1"],
                    color=color_high, alpha=0.18)
    # Mark layers where the t-test is significant.
    sig = trace_test["pvalue"] < 0.01
    for t in range(L):
        if sig[t]:
            ax.axvspan(t - 0.3, t + 0.3, color="yellow", alpha=0.15)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("trace(Sigma_i)")
    ax.set_title("D7.2: within-input variance by cluster (yellow=p<0.01)")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: cluster effective rank profile.
    ax = axes[0, 2]
    ax.plot(layers, erank_test["mean_cluster0"], color=color_low, lw=2,
            label="low cluster")
    ax.fill_between(layers,
                    erank_test["mean_cluster0"] - erank_test["std_cluster0"],
                    erank_test["mean_cluster0"] + erank_test["std_cluster0"],
                    color=color_low, alpha=0.18)
    ax.plot(layers, erank_test["mean_cluster1"], color=color_high, lw=2,
            label="high cluster")
    ax.fill_between(layers,
                    erank_test["mean_cluster1"] - erank_test["std_cluster1"],
                    erank_test["mean_cluster1"] + erank_test["std_cluster1"],
                    color=color_high, alpha=0.18)
    sig = erank_test["pvalue"] < 0.01
    for t in range(L):
        if sig[t]:
            ax.axvspan(t - 0.3, t + 0.3, color="yellow", alpha=0.15)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("effective rank")
    ax.set_title("D7.3: effective rank by cluster")
    ax.legend(fontsize=8, loc="best")

    # Panel 4: cluster kurtosis profile.
    ax = axes[1, 0]
    ax.plot(layers, kurt_test["mean_cluster0"], color=color_low, lw=2,
            label="low cluster")
    ax.fill_between(layers,
                    kurt_test["mean_cluster0"] - kurt_test["std_cluster0"],
                    kurt_test["mean_cluster0"] + kurt_test["std_cluster0"],
                    color=color_low, alpha=0.18)
    ax.plot(layers, kurt_test["mean_cluster1"], color=color_high, lw=2,
            label="high cluster")
    ax.fill_between(layers,
                    kurt_test["mean_cluster1"] - kurt_test["std_cluster1"],
                    kurt_test["mean_cluster1"] + kurt_test["std_cluster1"],
                    color=color_high, alpha=0.18)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    sig = kurt_test["pvalue"] < 0.01
    for t in range(L):
        if sig[t]:
            ax.axvspan(t - 0.3, t + 0.3, color="yellow", alpha=0.15)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("D7.4: kurtosis profile by cluster")
    ax.legend(fontsize=8, loc="best")

    # Panel 5: lambda through training, colored by final cluster.
    ax = axes[1, 1]
    # lambdas_through_training has shape (n_seeds, n_steps, n_tokens).
    # Average across seeds, plot per-token trajectories.
    lam_avg = np.nanmean(lambdas_through_training, axis=0)   # (n_steps, n_tok)
    n_tok = lam_avg.shape[1]
    for k in range(n_tok):
        c = cmap[int(labels[k])]
        ax.plot(steps, lam_avg[:, k], "-", color=c, alpha=0.4, lw=0.9)
    # Cluster centroid trajectories.
    lo_mean = np.nanmean(lam_avg[:, labels == 0], axis=1)
    hi_mean = np.nanmean(lam_avg[:, labels == 1], axis=1)
    ax.plot(steps, lo_mean, color=color_low, lw=3, label="low cluster mean")
    ax.plot(steps, hi_mean, color=color_high, lw=3, label="high cluster mean")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("per-token lambda")
    ax.set_title("D7.5: lambda trajectories through training (by final cluster)")
    ax.legend(fontsize=8, loc="best")

    # Panel 6: cluster separation through training (gap between centroids
    # divided by within-cluster std).
    ax = axes[1, 2]
    sep = np.full(steps.size, np.nan)
    pool_std = np.full(steps.size, np.nan)
    for s in range(steps.size):
        lo = lam_avg[s, labels == 0]
        hi = lam_avg[s, labels == 1]
        lo = lo[np.isfinite(lo)]
        hi = hi[np.isfinite(hi)]
        if lo.size >= 2 and hi.size >= 2:
            sep[s] = hi.mean() - lo.mean()
            pool_std[s] = np.sqrt(
                ((lo.size - 1) * lo.var(ddof=1) +
                 (hi.size - 1) * hi.var(ddof=1)) /
                max(lo.size + hi.size - 2, 1)
            )
    cohen_d = sep / pool_std
    ax.plot(steps, sep, "k-", lw=2, label="mean gap (high - low)")
    ax.plot(steps, cohen_d, "C2--", lw=2, label="Cohen's d (gap / pooled std)")
    ax.axhline(0.8, color="C2", ls=":", lw=1, alpha=0.5,
               label="d=0.8 (large effect)")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("cluster separation")
    ax.set_title("D7.6: when does the bimodality emerge?")
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d7_lambda_clusters.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    # Load D3 output for lambda.
    d3_path = os.path.join(output_root(args.run_dir), "d3_per_token_fits.npz")
    if not os.path.exists(d3_path):
        print(f"Missing {d3_path}; run model_abc_discriminator.py first.")
        sys.exit(1)
    d3 = np.load(d3_path)
    tids = d3["tids"]                                # (n_tok,)
    lam = d3["lambda_per_token"]                     # (n_seeds, n_steps, n_tok)
    log_alpha = d3["log_alpha_per_token"]
    steps = d3["steps"]
    seeds = d3["seeds"]
    n_seeds, n_steps, n_tok = lam.shape

    # Cluster on average per-token lambda at final checkpoint.
    lam_final_per_seed = lam[:, -1, :]               # (n_seeds, n_tok)
    lam_final_avg = np.nanmean(lam_final_per_seed, axis=0)  # (n_tok,)
    print(f"\nClustering {n_tok} tokens by final-ckpt lambda (avg over seeds):")
    centers, labels = kmeans_1d(lam_final_avg, k=2)
    print(f"  cluster centers: low={centers[0]:.4f}  high={centers[1]:.4f}")
    print(f"  cluster sizes:   low={int((labels == 0).sum())}  "
          f"high={int((labels == 1).sum())}")

    # Cross-seed consistency: would each seed give the same labels?
    consistent_per_token = np.zeros(n_tok, dtype=bool)
    seed_labels = np.full((n_seeds, n_tok), -1, dtype=np.int32)
    for s in range(n_seeds):
        _, lbl_s = kmeans_1d(lam_final_per_seed[s], k=2)
        seed_labels[s] = lbl_s
    for k in range(n_tok):
        col = seed_labels[:, k]
        col = col[col >= 0]
        consistent_per_token[k] = (col == labels[k]).all() and col.size > 0
    print(f"  cross-seed label consistency: "
          f"{int(consistent_per_token.sum())}/{n_tok}")

    # Decode tokens.
    decoded = decode_tokens(tids)
    print(f"\nLow cluster (lambda ~ {centers[0]:.3f}):")
    for k in range(n_tok):
        if labels[k] == 0:
            stable = "stable" if consistent_per_token[k] else "UNSTABLE"
            print(f"  tid {int(tids[k]):>6}  lambda={lam_final_avg[k]:.3f}  "
                  f"{decoded[k]:<20s}  ({stable})")
    print(f"\nHigh cluster (lambda ~ {centers[1]:.3f}):")
    for k in range(n_tok):
        if labels[k] == 1:
            stable = "stable" if consistent_per_token[k] else "UNSTABLE"
            print(f"  tid {int(tids[k]):>6}  lambda={lam_final_avg[k]:.3f}  "
                  f"{decoded[k]:<20s}  ({stable})")

    # Cluster comparison on other diagnostics. We need trace, effective
    # rank, and kurtosis profiles per token. These come from d4a_kurtosis
    # (kurtosis) and require a fresh aggregation pass for trace and erank
    # since d1_token_cv only stores the CV summary, not per-token.
    # We'll reload them from the multiview stage-C results.
    print(f"\nLoading per-token trace / erank / kurtosis profiles from stage C ...")
    sys.path.insert(0, ".")
    from multiview import load_multi_view_result
    from multiview_campaign import mvr_dir
    final_step = int(steps[-1])
    L = None
    trace_per_token = None    # (n_seeds, n_tok, L)
    erank_per_token = None    # (n_seeds, n_tok, L)
    kurt_per_token = None     # (n_seeds, n_tok, L)
    for si, seed in enumerate(seeds):
        d = mvr_dir(args.run_dir, int(seed), final_step)
        r = load_multi_view_result(d, skip_arrays={"R", "pairwise_residual_variance"})
        flows = r.forward_flows
        if L is None:
            sample_flow = next(iter(flows.values()))
            L = sample_flow["singular_values"].shape[0]
            trace_per_token = np.full((n_seeds, n_tok, L), np.nan)
            erank_per_token = np.full((n_seeds, n_tok, L), np.nan)
            kurt_per_token = np.full((n_seeds, n_tok, L), np.nan)
        for k, tok in enumerate(tids):
            f = flows.get(int(tok))
            if f is None or f.get("failed", False):
                continue
            sv = f["singular_values"].astype(np.float64)
            n = float(f["n_pilots"])
            if n > 1:
                trace_per_token[si, k] = (sv ** 2).sum(axis=1) / n
            erank_per_token[si, k] = f["effective_rank"].astype(np.float64)
            kurt_per_token[si, k] = f["kurtosis_per_layer"].astype(np.float64)

    # Run Welch tests per layer.
    print(f"\nWelch t-test per layer (low vs high cluster, n_seeds-averaged):")
    trace_test = per_layer_cluster_test(trace_per_token, labels)
    erank_test = per_layer_cluster_test(erank_per_token, labels)
    kurt_test = per_layer_cluster_test(kurt_per_token, labels)

    print(f"  layer  trace_p   erank_p   kurt_p   trace_lo/hi   erank_lo/hi")
    for t in range(L):
        print(f"  {t:5d}  "
              f"{trace_test['pvalue'][t]:7.4f}  "
              f"{erank_test['pvalue'][t]:7.4f}  "
              f"{kurt_test['pvalue'][t]:7.4f}  "
              f"{trace_test['mean_cluster0'][t]:>5.1f}/"
              f"{trace_test['mean_cluster1'][t]:<5.1f}  "
              f"{erank_test['mean_cluster0'][t]:>5.1f}/"
              f"{erank_test['mean_cluster1'][t]:<5.1f}")

    # Save JSON.
    out_json = os.path.join(output_root(args.run_dir), "d7_lambda_clusters.json")
    payload = {
        "centers": {"low": float(centers[0]), "high": float(centers[1])},
        "cluster_sizes": {
            "low": int((labels == 0).sum()),
            "high": int((labels == 1).sum()),
        },
        "low_cluster": [
            {
                "tid": int(tids[k]),
                "decoded": decoded[k],
                "lambda_avg_over_seeds": float(lam_final_avg[k]),
                "cross_seed_stable": bool(consistent_per_token[k]),
            }
            for k in range(n_tok) if labels[k] == 0
        ],
        "high_cluster": [
            {
                "tid": int(tids[k]),
                "decoded": decoded[k],
                "lambda_avg_over_seeds": float(lam_final_avg[k]),
                "cross_seed_stable": bool(consistent_per_token[k]),
            }
            for k in range(n_tok) if labels[k] == 1
        ],
        "per_layer_tests": {
            "trace_pvalue": trace_test["pvalue"].tolist(),
            "erank_pvalue": erank_test["pvalue"].tolist(),
            "kurt_pvalue": kurt_test["pvalue"].tolist(),
            "trace_mean_low": trace_test["mean_cluster0"].tolist(),
            "trace_mean_high": trace_test["mean_cluster1"].tolist(),
            "erank_mean_low": erank_test["mean_cluster0"].tolist(),
            "erank_mean_high": erank_test["mean_cluster1"].tolist(),
            "kurt_mean_low": kurt_test["mean_cluster0"].tolist(),
            "kurt_mean_high": kurt_test["mean_cluster1"].tolist(),
        },
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[json] -> {out_json}")

    plot_clusters(
        args.run_dir, tids, lam_final_avg, labels, centers,
        trace_test, erank_test, kurt_test,
        steps, lam,
    )


if __name__ == "__main__":
    main()
