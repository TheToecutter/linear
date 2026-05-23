"""
10.4 Cross-checkpoint Model A/B/C verdict.

Re-runs the Model A/B/C discriminator logic at the four representative
checkpoints from §6 (Phase I, II, III mid, III final), across all
seeds, reading the existing multiview stage-C results.

The original §4 verdict was computed only at the final checkpoint
(step 24000). §6's training-trajectory analysis showed that the C
verdict is a Phase III phenomenon — conditional non-Gaussianity is
weak at Phase II and emerges progressively through Phase III. This
script provides checkpoint-level resolution on the discriminator
outputs themselves, complementing the indirect evidence already in §6.

The discriminators reproduced here (subset of D1, D3, D4):

  D1: cross-token CV of trace(Sigma_i(t)) at each layer.
  D3: per-token lambda fit and cross-token CV of lambda.
  D4: per-coordinate excess kurtosis of conditional bundles (D4a only,
      since D4b requires the augmented activations which we may not
      have at every checkpoint).

D2 (principal angles) is omitted: previously demoted to confirmatory
diagnostic and sample-size-degenerate at small subspaces.
D5 (GMM reconstruction) is omitted: requires augmented files, run only
at the final checkpoint in the original §4.5.

The verdict logic:
  - Cross-token CV(trace) > threshold     => B/C signal on D1
  - Cross-token CV(lambda) > threshold    => B/C signal on D3
  - Conditional kurtosis > threshold      => C signal on D4
  - All three weak                        => A
  - Only D1 or D3, not D4                 => B
  - D4 present                            => C

This script reads from run_dir/multiview/seed_S/multi_view_step_NNNNNNNN.npz
(the multiview stage-C output files) and does NOT re-run any
expensive computation. It is fast: ~30s per (seed, step) pair.

Output:
    run_dir/multiview/model_abc/d15_crossphase_verdict.npz
    run_dir/multiview/model_abc/figures/d15_crossphase_verdict.png
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

from multiview import load_multi_view_result
from multiview_campaign import mvr_dir


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


# ----------------------------------------------------------------------
# Extract per-token statistics from forward flows.
# ----------------------------------------------------------------------
def extract_stats(forward_flows: Dict[int, Dict]) -> Dict[str, np.ndarray]:
    """From the per-token forward flow dicts, build arrays of
    (n_tokens, L) for trace, effective_rank, kurtosis; plus per-token
    scalars lambda and log_alpha."""
    valid_items = [(tok, f) for tok, f in forward_flows.items()
                   if not f.get("failed", False)]
    if not valid_items:
        return {}
    L = next(iter(forward_flows.values()))["singular_values"].shape[0]
    n_tok = len(valid_items)
    trace = np.full((n_tok, L), np.nan)
    erank = np.full((n_tok, L), np.nan)
    kurt = np.full((n_tok, L), np.nan)
    lam = np.full(n_tok, np.nan)
    log_alpha = np.full(n_tok, np.nan)
    tids = np.zeros(n_tok, dtype=np.int64)
    n_pilots = np.zeros(n_tok, dtype=np.int64)
    for i, (tok, f) in enumerate(valid_items):
        tids[i] = int(tok)
        n = float(f["n_pilots"])
        sv = f["singular_values"].astype(np.float64)
        if n > 1:
            trace[i] = (sv ** 2).sum(axis=1) / n
        erank[i] = f["effective_rank"].astype(np.float64)
        kurt[i] = f["kurtosis_per_layer"].astype(np.float64)
        # Use paper convention if available.
        lam[i] = float(f.get("lambda_paper", f.get("lambda", np.nan)))
        log_alpha[i] = float(
            f.get("log_alpha_paper", f.get("log_alpha", np.nan)))
        n_pilots[i] = int(n)
    return {
        "tids": tids,
        "trace": trace, "erank": erank, "kurt": kurt,
        "lambda": lam, "log_alpha": log_alpha,
        "n_pilots": n_pilots,
    }


def cv_across_tokens(arr_per_token_per_layer: np.ndarray) -> np.ndarray:
    """CV across tokens at each layer."""
    L = arr_per_token_per_layer.shape[1]
    out = np.full(L, np.nan)
    for t in range(L):
        col = arr_per_token_per_layer[:, t]
        col = col[np.isfinite(col)]
        if col.size >= 2 and abs(col.mean()) > 1e-12:
            out[t] = float(col.std(ddof=1) / abs(col.mean()))
    return out


def trim_outlier_indices(values: np.ndarray, k: int = 1) -> np.ndarray:
    """Return indices of values with the k largest absolute deviations
    from the median removed. We use this to trim the newline-token
    outlier from cross-token statistics."""
    if values.size <= k:
        return np.arange(values.size)
    med = np.nanmedian(values)
    dev = np.abs(values - med)
    keep = np.argsort(dev)[: values.size - k]
    return np.sort(keep)


def verdict_per_layer(
    stats: Dict[str, np.ndarray],
    threshold_cv_trace: float, threshold_cv_lambda: float,
    threshold_kurt: float, trim_d1_outlier: bool = True,
) -> Tuple[List[str], Dict]:
    """Compute the A/B/C verdict per layer.

    Returns (per-layer verdict strings, diagnostic dict)."""
    if not stats:
        return [], {}
    trace = stats["trace"]
    kurt = stats["kurt"]
    lam = stats["lambda"]
    L = trace.shape[1]

    if trim_d1_outlier and trace.shape[0] >= 5:
        keep = trim_outlier_indices(trace.mean(axis=1), k=1)
        cv_trace = cv_across_tokens(trace[keep])
    else:
        cv_trace = cv_across_tokens(trace)

    finite_lam = lam[np.isfinite(lam)]
    if finite_lam.size >= 2 and abs(finite_lam.mean()) > 1e-12:
        cv_lambda = float(finite_lam.std(ddof=1) / abs(finite_lam.mean()))
    else:
        cv_lambda = float("nan")

    mean_kurt = np.nanmean(kurt, axis=0)

    verdicts = []
    for t in range(L):
        d1 = cv_trace[t] > threshold_cv_trace if np.isfinite(cv_trace[t]) \
             else False
        d3 = cv_lambda > threshold_cv_lambda if np.isfinite(cv_lambda) \
             else False
        d4 = mean_kurt[t] > threshold_kurt if np.isfinite(mean_kurt[t]) \
             else False
        if d4:
            verdicts.append("C")
        elif d1 or d3:
            verdicts.append("B")
        else:
            verdicts.append("A")
    return verdicts, {
        "cv_trace": cv_trace,
        "cv_lambda": float(cv_lambda),
        "mean_kurt": mean_kurt,
    }


# ----------------------------------------------------------------------
# Analyze one (seed, step).
# ----------------------------------------------------------------------
def analyze(run_dir: str, seed: int, step: int,
            threshold_cv_trace: float, threshold_cv_lambda: float,
            threshold_kurt: float) -> Optional[Dict]:
    d = mvr_dir(run_dir, seed, step)
    if not os.path.isdir(d):
        return None
    try:
        r = load_multi_view_result(
            d, skip_arrays={"R", "pairwise_residual_variance"})
    except Exception as e:
        print(f"  [skip] load failed: {e}")
        return None

    stats = extract_stats(r.forward_flows)
    if not stats:
        return None
    verdicts, diag = verdict_per_layer(
        stats,
        threshold_cv_trace=threshold_cv_trace,
        threshold_cv_lambda=threshold_cv_lambda,
        threshold_kurt=threshold_kurt,
    )
    return {
        "seed": seed, "step": step,
        "verdicts": verdicts,
        "cv_trace": diag["cv_trace"],
        "cv_lambda": diag["cv_lambda"],
        "mean_kurt": diag["mean_kurt"],
        "n_tokens": stats["trace"].shape[0],
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(
    run_dir: str, by_seed_step: Dict[Tuple[int, int], Dict],
    seeds: List[int], steps: List[int],
) -> None:
    if not by_seed_step:
        return
    L = next(iter(by_seed_step.values()))["mean_kurt"].size

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: per-layer mean conditional kurtosis (D4a) for each step,
    # averaged across seeds.
    ax = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))
    for ci, step in enumerate(steps):
        per_seed = [by_seed_step[(s, step)]["mean_kurt"]
                    for s in seeds if (s, step) in by_seed_step]
        if not per_seed:
            continue
        m = np.nanmean(np.stack(per_seed), axis=0)
        s_ = np.nanstd(np.stack(per_seed), axis=0)
        ax.plot(np.arange(L), m, "-", color=cmap[ci], lw=2,
                label=PHASE_LABELS.get(step, str(step)))
        ax.fill_between(np.arange(L), m - s_, m + s_,
                        color=cmap[ci], alpha=0.15)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("mean conditional excess kurtosis (D4a)")
    ax.set_title("D4a kurtosis profile across phases")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: CV(trace) per layer (D1).
    ax = axes[0, 1]
    for ci, step in enumerate(steps):
        per_seed = [by_seed_step[(s, step)]["cv_trace"]
                    for s in seeds if (s, step) in by_seed_step]
        if not per_seed:
            continue
        m = np.nanmean(np.stack(per_seed), axis=0)
        ax.plot(np.arange(L), m, "-", color=cmap[ci], lw=2,
                label=PHASE_LABELS.get(step, str(step)))
    ax.axhline(0.115, color="k", ls="--", lw=1,
               label="threshold (trimmed)")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("CV(trace) across tokens")
    ax.set_title("D1: cross-token CV of trace, by phase")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: CV(lambda) per step (one scalar per (seed, step)).
    ax = axes[1, 0]
    for ci, step in enumerate(steps):
        vals = [by_seed_step[(s, step)]["cv_lambda"]
                for s in seeds if (s, step) in by_seed_step]
        if not vals:
            continue
        ax.bar(ci, np.nanmean(vals), color=cmap[ci],
               yerr=np.nanstd(vals), capsize=4)
    ax.axhline(0.062, color="k", ls="--", lw=1, label="threshold")
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([PHASE_LABELS.get(s, str(s)) for s in steps],
                       rotation=15, fontsize=8)
    ax.set_ylabel("CV(lambda) across tokens")
    ax.set_title("D3: cross-token CV of lambda, by phase")
    ax.legend(fontsize=8, loc="best")

    # Panel 4: verdict heatmap (rows = steps, cols = layers).
    ax = axes[1, 1]
    verdict_to_int = {"A": 0, "B": 1, "C": 2}
    grid = np.full((len(steps), L), -1, dtype=int)
    for si, step in enumerate(steps):
        per_seed_verdicts = []
        for s in seeds:
            if (s, step) in by_seed_step:
                per_seed_verdicts.append(by_seed_step[(s, step)]["verdicts"])
        if not per_seed_verdicts:
            continue
        # Majority vote per layer.
        for t in range(L):
            counts = {"A": 0, "B": 0, "C": 0}
            for v in per_seed_verdicts:
                if t < len(v):
                    counts[v[t]] += 1
            best = max(counts, key=lambda k: counts[k])
            grid[si, t] = verdict_to_int[best]

    im = ax.imshow(grid, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=2, interpolation="nearest")
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels([PHASE_LABELS.get(s, str(s)) for s in steps],
                       fontsize=9)
    ax.set_xticks(range(L))
    ax.set_xticklabels(range(L), fontsize=8)
    ax.set_xlabel("layer state index t")
    ax.set_title("Per-layer majority verdict (cross-seed): A=blue, B=yellow, C=red")
    # Annotate cells.
    int_to_verdict = {0: "A", 1: "B", 2: "C", -1: "?"}
    for si in range(len(steps)):
        for t in range(L):
            txt = int_to_verdict[grid[si, t]]
            ax.text(t, si, txt, ha="center", va="center",
                    color="black", fontsize=8)

    fig.suptitle(
        f"D15: cross-phase A/B/C verdict (seeds averaged)",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir),
                       "d15_crossphase_verdict.png")
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
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--threshold-cv-trace", type=float, default=0.115)
    ap.add_argument("--threshold-cv-lambda", type=float, default=0.062)
    ap.add_argument("--threshold-kurt", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    print(f"Seeds: {args.seeds}")
    print(f"Steps: {args.steps}")
    print(f"Thresholds: CV(trace) > {args.threshold_cv_trace}, "
          f"CV(lambda) > {args.threshold_cv_lambda}, "
          f"mean kurtosis > {args.threshold_kurt}")

    by_seed_step = {}
    t0 = time.time()
    for seed in args.seeds:
        for step in args.steps:
            print(f"\n[{seed=}, {step=}] {PHASE_LABELS.get(step, str(step))}")
            r = analyze(args.run_dir, seed, step,
                        threshold_cv_trace=args.threshold_cv_trace,
                        threshold_cv_lambda=args.threshold_cv_lambda,
                        threshold_kurt=args.threshold_kurt)
            if r is None:
                print(f"  [skip] no data")
                continue
            by_seed_step[(seed, step)] = r
            print(f"  per-layer verdict: {''.join(r['verdicts'])}")
            print(f"  CV(lambda) = {r['cv_lambda']:.3f}  "
                  f"n_tokens = {r['n_tokens']}")
            print(f"  CV(trace) layer 7 = {r['cv_trace'][7]:.3f}, "
                  f"layer 11 = {r['cv_trace'][11]:.3f}")
            print(f"  mean kurt layer 7 = {r['mean_kurt'][7]:.3f}, "
                  f"layer 11 = {r['mean_kurt'][11]:.3f}")
    elapsed = time.time() - t0
    print(f"\nAll (seed, step) pairs done in {elapsed:.1f}s")

    # Save.
    if by_seed_step:
        out_arrays = {
            "seeds": np.array(args.seeds, dtype=np.int32),
            "steps": np.array(args.steps, dtype=np.int64),
            "threshold_cv_trace": np.float64(args.threshold_cv_trace),
            "threshold_cv_lambda": np.float64(args.threshold_cv_lambda),
            "threshold_kurt": np.float64(args.threshold_kurt),
        }
        for (seed, step), r in by_seed_step.items():
            out_arrays[f"s{seed}_st{step}_cv_trace"] = r["cv_trace"]
            out_arrays[f"s{seed}_st{step}_cv_lambda"] = np.float64(
                r["cv_lambda"])
            out_arrays[f"s{seed}_st{step}_mean_kurt"] = r["mean_kurt"]
            out_arrays[f"s{seed}_st{step}_verdicts"] = np.array(
                r["verdicts"], dtype="<U2")
        out_path = os.path.join(output_root(args.run_dir),
                                "d15_crossphase_verdict.npz")
        np.savez(out_path, **out_arrays)
        print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, by_seed_step, args.seeds, args.steps)


if __name__ == "__main__":
    main()
