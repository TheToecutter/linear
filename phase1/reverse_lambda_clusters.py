"""
Reverse lambda^contract cluster analysis: hypothesis N2.

The forward investigation found a bimodal lambda distribution across
input tokens, with cluster membership predictable from grammatical
role (connectives in the high-lambda cluster, terminators / sub-word
fragments in the low cluster).

Hypothesis N2 asks whether the same bimodal structure exists in the
reverse direction. The reverse analog of lambda_v is lambda_w^contract:
the log-linear contraction rate of within-successor variance on the
descending layers [t_w^*, L-1]. If the model's contraction rate toward
its prediction is structured by the grammatical role of the predicted
token, the same connective/terminator split should appear.

This script:

  1. Loads d3_per_token_fits_reverse_actual.npz to get per-cell
     lambda_contract values.
  2. Uses k-means (k=2) to cluster the per-cell values at the final
     checkpoint, averaged across seeds.
  3. Decodes the token ids and prints the two clusters with their
     lexical content.
  4. For each available per-cell diagnostic -- trace(Sigma_w),
     effective rank, kurtosis, log_alpha_contract -- computes
     cluster-mean profiles and a Welch t-test per layer.
  5. Tests cross-seed stability of cluster membership.
  6. Produces a figure summary parallel to lambda_clusters.py.

Output:
    run_dir/multiview/model_abc/d_n2_reverse_lambda_clusters.json
    run_dir/multiview/model_abc/figures/d_n2_reverse_lambda_clusters.png

Usage:
    python reverse_lambda_clusters.py --run-dir ../phase1_runs_gelu

Reuses kmeans_1d, decode_tokens, per_layer_cluster_test from the
existing lambda_clusters module to keep numerics identical with the
forward cluster output.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lambda_clusters import (
    kmeans_1d,
    decode_tokens,
    per_layer_cluster_test,
)
from reverse_buildup import output_root, figures_dir


# ----------------------------------------------------------------------
# Cross-seed stability.
# ----------------------------------------------------------------------
def cross_seed_stability(
    per_seed_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Test how many tokens have consistent cluster assignment across seeds.

    Args:
        per_seed_values: (n_seeds, n_tokens) per-cell lambda_contract
                         values at the final checkpoint, one row per seed.

    Returns:
        (per_seed_labels, agreement, stable_fraction)
            per_seed_labels: (n_seeds, n_tokens) cluster labels in
                             {0, 1, -1} per seed.
            agreement:       (n_tokens,) 1 if all four seeds agree, 0
                             otherwise. Tokens with any invalid (-1)
                             label count as disagreement.
            stable_fraction: fraction of tokens with cross-seed-stable
                             assignment.
    """
    n_seeds, n_tok = per_seed_values.shape
    per_seed_labels = np.full((n_seeds, n_tok), -1, dtype=np.int32)
    for si in range(n_seeds):
        _, lbl = kmeans_1d(per_seed_values[si], k=2)
        per_seed_labels[si] = lbl
    agreement = np.zeros(n_tok, dtype=np.int32)
    for ti in range(n_tok):
        labels = per_seed_labels[:, ti]
        if (labels == -1).any():
            continue
        # All-same check.
        if (labels == labels[0]).all():
            agreement[ti] = 1
    stable_fraction = float(agreement.sum()) / max(n_tok, 1)
    return per_seed_labels, agreement, stable_fraction


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_reverse_clusters(
    run_dir: str,
    tids: np.ndarray,
    lambda_contract_final: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    decoded: List[str],
    trace_test: Dict,
    kurt_test: Dict,
    lambda_through_training: np.ndarray,
    steps: np.ndarray,
    stable_fraction: float,
) -> None:
    """Multi-panel figure summarizing the reverse cluster analysis.

    Layout parallels lambda_clusters.plot_clusters: histogram, scatter
    vs forward, cluster trajectory through training, per-layer Welch
    t-test for trace and kurtosis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    color_low = "#d62728"      # red
    color_high = "#1f77b4"     # blue
    cmap = {0: color_low, 1: color_high, -1: "gray"}

    # Panel 1: lambda_contract histogram by cluster.
    ax = axes[0, 0]
    lo_mask = (labels == 0); hi_mask = (labels == 1)
    if lo_mask.any():
        ax.hist(lambda_contract_final[lo_mask], bins=8, color=color_low,
                alpha=0.7,
                label=f"low cluster (n={int(lo_mask.sum())}, "
                      f"mean={centers[0]:.3f})")
    if hi_mask.any():
        ax.hist(lambda_contract_final[hi_mask], bins=12, color=color_high,
                alpha=0.7,
                label=f"high cluster (n={int(hi_mask.sum())}, "
                      f"mean={centers[1]:.3f})")
    ax.axvline(centers[0], color=color_low, ls="--", lw=1.2)
    ax.axvline(centers[1], color=color_high, ls="--", lw=1.2)
    ax.set_xlabel(r"$\lambda_w^{\mathrm{contract}}$ (per-successor)")
    ax.set_ylabel("count")
    ax.set_title(r"Reverse $\lambda^{\mathrm{contract}}$ distribution")
    ax.legend(fontsize=8)

    # Panel 2: scatter of token id index vs lambda_contract, colored.
    ax = axes[0, 1]
    for i, lbl in enumerate(labels):
        ax.scatter(i, lambda_contract_final[i],
                   color=cmap[int(lbl)], s=40)
        ax.annotate(decoded[i] if i < len(decoded) else str(int(tids[i])),
                    (i, lambda_contract_final[i]),
                    fontsize=6, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("successor index (sorted by frequency)")
    ax.set_ylabel(r"$\lambda_w^{\mathrm{contract}}$")
    ax.set_title("Per-successor contraction rates")
    ax.axhline(0.0, color="black", ls=":", lw=0.5)

    # Panel 3: lambda through training, color-coded by final-checkpoint
    # cluster assignment.
    ax = axes[0, 2]
    # lambda_through_training has shape (n_steps, n_tokens).
    for ti in range(lambda_through_training.shape[1]):
        ax.plot(steps, lambda_through_training[:, ti],
                color=cmap[int(labels[ti])], alpha=0.4, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\lambda_w^{\mathrm{contract}}$")
    ax.set_title("Contraction rate through training")

    # Panel 4: trace cluster test.
    ax = axes[1, 0]
    if trace_test is not None:
        L = trace_test["mean_cluster0"].size
        layers = np.arange(L)
        ax.plot(layers, trace_test["mean_cluster0"],
                color=color_low, label="low cluster", lw=1.5)
        ax.plot(layers, trace_test["mean_cluster1"],
                color=color_high, label="high cluster", lw=1.5)
        ax.fill_between(
            layers,
            trace_test["mean_cluster0"] - trace_test["std_cluster0"],
            trace_test["mean_cluster0"] + trace_test["std_cluster0"],
            color=color_low, alpha=0.2)
        ax.fill_between(
            layers,
            trace_test["mean_cluster1"] - trace_test["std_cluster1"],
            trace_test["mean_cluster1"] + trace_test["std_cluster1"],
            color=color_high, alpha=0.2)
        ax.set_xlabel("layer")
        ax.set_ylabel(r"trace($\Sigma_w$)")
        ax.set_title("Cluster trace profile")
        ax.legend(fontsize=8)

    # Panel 5: kurtosis cluster test.
    ax = axes[1, 1]
    if kurt_test is not None:
        L = kurt_test["mean_cluster0"].size
        layers = np.arange(L)
        ax.plot(layers, kurt_test["mean_cluster0"],
                color=color_low, lw=1.5)
        ax.plot(layers, kurt_test["mean_cluster1"],
                color=color_high, lw=1.5)
        ax.fill_between(
            layers,
            kurt_test["mean_cluster0"] - kurt_test["std_cluster0"],
            kurt_test["mean_cluster0"] + kurt_test["std_cluster0"],
            color=color_low, alpha=0.2)
        ax.fill_between(
            layers,
            kurt_test["mean_cluster1"] - kurt_test["std_cluster1"],
            kurt_test["mean_cluster1"] + kurt_test["std_cluster1"],
            color=color_high, alpha=0.2)
        ax.set_xlabel("layer")
        ax.set_ylabel("excess kurtosis")
        ax.set_title("Cluster kurtosis profile")

    # Panel 6: stability and decoded contents.
    ax = axes[1, 2]
    ax.axis("off")
    txt = [
        f"Cross-seed stability: {stable_fraction:.1%} of tokens "
        f"have consistent assignment",
        "",
        "Low cluster:",
    ]
    for i, lbl in enumerate(labels):
        if lbl == 0:
            txt.append(f"  {decoded[i] if i < len(decoded) else int(tids[i])}: "
                       f"{lambda_contract_final[i]:+.3f}")
    txt.append("")
    txt.append("High cluster:")
    for i, lbl in enumerate(labels):
        if lbl == 1:
            txt.append(f"  {decoded[i] if i < len(decoded) else int(tids[i])}: "
                       f"{lambda_contract_final[i]:+.3f}")
    ax.text(0.0, 1.0, "\n".join(txt[:30]),
            va="top", ha="left", fontsize=8, family="monospace")

    plt.tight_layout()
    out_path = os.path.join(figures_dir(run_dir),
                            "d_n2_reverse_lambda_clusters.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[N2] figure -> {out_path}")


# ----------------------------------------------------------------------
# Main entry: run on the saved D3 reverse output.
# ----------------------------------------------------------------------
def run_reverse_lambda_clusters(
    run_dir: str,
    view: str = "reverse_actual",
) -> Dict:
    """Run the reverse lambda^contract cluster analysis.

    Args:
        run_dir: project run directory.
        view: 'reverse_actual' (default) or 'reverse_pred'.

    Returns:
        Result dict; also writes JSON + figure to disk.

    Robustness note. The proposal's expectation is that reverse cells'
    within-cell variance has an interior peak followed by a contraction
    phase, in which case lambda_contract is the natural statistic. On
    real data, trace(Sigma_w) may grow monotonically through depth even
    for reverse cells (the within-cell ensemble can spread in new
    directions even as it converges in older ones). When that happens,
    lambda_contract is undefined for most cells. We detect this case
    and fall back to clustering on lambda_per_token (the standard
    log-linear fit on all 14 layers), reporting which statistic was
    used so the result is unambiguous.
    """
    d3_path = os.path.join(output_root(run_dir),
                           f"d3_per_token_fits_{view}.npz")
    if not os.path.exists(d3_path):
        raise FileNotFoundError(
            f"D3 output not found at {d3_path}. Run "
            f"reverse_buildup_campaign.py phase A first."
        )
    d3 = dict(np.load(d3_path, allow_pickle=False))
    tids = d3["tids"]
    lam_contract = d3["lambda_contract"]                    # (n_seeds, n_steps, n_tok)
    lam_perTok = d3["lambda_per_token"]                     # same shape, full-depth fit
    # Readout-step statistic was added after first results showed the
    # multi-layer contraction phase doesn't exist for reverse-actual.
    lam_readout = d3.get("lambda_readout_step", None)
    t_peak = d3.get("t_peak", None)
    n_seeds, n_steps, n_tok = lam_contract.shape

    # Pick the best-defined statistic for this view's data.
    # Priority: readout_step (if dense and non-degenerate) >
    #           lambda_contract (the proposal's original choice) >
    #           lambda_per_token (always defined, weakest separator).
    lam_c_final = lam_contract[:, -1, :]
    contract_density = float(np.isfinite(lam_c_final).sum()) / max(lam_c_final.size, 1)

    if lam_readout is not None:
        readout_final = lam_readout[:, -1, :]
        readout_density = float(np.isfinite(readout_final).sum()) / max(readout_final.size, 1)
    else:
        readout_density = 0.0

    if readout_density >= 0.5:
        statistic_used = "lambda_readout_step"
        lam_array = lam_readout
        print(f"[N2/{view}] Using lambda_readout_step (density "
              f"{readout_density:.1%}). This statistic measures the "
              f"one-step variance compression at the readout, which is "
              f"the actual reverse-view contraction mechanism in this "
              f"data (cf. the original lambda_contract: density "
              f"{contract_density:.1%}).")
    elif contract_density >= 0.5:
        statistic_used = "lambda_contract"
        lam_array = lam_contract
        print(f"[N2/{view}] Using lambda_contract (density "
              f"{contract_density:.1%} >= 0.5).")
    else:
        statistic_used = "lambda_per_token"
        lam_array = lam_perTok
        print(f"[N2/{view}] Both lambda_readout_step ({readout_density:.1%}) "
              f"and lambda_contract ({contract_density:.1%}) are too "
              f"sparse; falling back to lambda_per_token. This is the "
              f"weakest of the three statistics but is always defined.")
        if t_peak is not None:
            tp_final = t_peak[:, -1, :]
            L_total = 14  # standard for this project
            late_frac = float(np.mean(tp_final >= L_total - 2))
            print(f"[N2/{view}] Of final-checkpoint cells, "
                  f"{late_frac:.1%} have t_peak in the last two layers "
                  f"(no multi-layer contraction phase available).")

    # Final checkpoint, averaged across seeds.
    lam_final_per_seed = lam_array[:, -1, :]
    lam_final = np.nanmean(lam_final_per_seed, axis=0)       # (n_tok,)

    # Cluster.
    centers, labels = kmeans_1d(lam_final, k=2)

    # Decode tokens.
    decoded = decode_tokens(tids)

    # Cross-seed stability.
    per_seed_labels, agreement, stable_fraction = cross_seed_stability(
        lam_final_per_seed)

    # Per-layer cluster tests on saved diagnostics.
    d4a_path = os.path.join(output_root(run_dir),
                            f"d4a_kurtosis_{view}.npz")
    kurt_test = None
    if os.path.exists(d4a_path):
        d4a = dict(np.load(d4a_path, allow_pickle=False))
        # kurtosis_per_token shape: (n_seeds, n_steps, n_tok, L).
        # Take final step, average across seeds.
        kurt_final = np.nanmean(d4a["kurtosis_per_token"][:, -1, :, :],
                                axis=0)  # (n_tok, L)
        kurt_test = per_layer_cluster_test(kurt_final, labels)

    # Lambda trajectory through training, averaged across seeds.
    lam_through = np.nanmean(lam_array, axis=0)               # (n_steps, n_tok)
    steps = d3["steps"]

    # Plot.
    plot_reverse_clusters(
        run_dir, tids, lam_final, labels, centers, decoded,
        trace_test=None, kurt_test=kurt_test,
        lambda_through_training=lam_through, steps=steps,
        stable_fraction=stable_fraction,
    )

    # Persist results.
    result = {
        "view": view,
        "statistic_used": statistic_used,
        "lambda_contract_density_at_final": contract_density,
        "tids": tids.tolist(),
        "decoded": decoded,
        "lambda_final": lam_final.tolist(),
        "lambda_final_per_seed": lam_final_per_seed.tolist(),
        "cluster_centers": centers.tolist(),
        "cluster_labels": labels.tolist(),
        "per_seed_labels": per_seed_labels.tolist(),
        "agreement": agreement.tolist(),
        "stable_fraction": stable_fraction,
    }
    out_json = os.path.join(output_root(run_dir),
                            f"d_n2_reverse_lambda_clusters_{view}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[N2/{view}] statistic={statistic_used}, "
          f"cluster centers: {centers}")
    print(f"[N2/{view}] stable fraction: {stable_fraction:.1%}")
    print(f"[N2/{view}] -> {out_json}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--view", default="reverse_actual",
                   choices=["reverse_actual", "reverse_pred"])
    args = p.parse_args()
    run_reverse_lambda_clusters(args.run_dir, view=args.view)


if __name__ == "__main__":
    main()
    