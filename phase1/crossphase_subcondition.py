"""
Path 2: cross-phase sub-conditioning on randomized stage A files.

Goals (from the investigation writeup's §8.3 and §5.2):

  1. Run sub-conditioning on (input, next), (input, position_binned),
     (input, pred) at four representative training checkpoints, one
     per phase from §8.2:
        step  ~479 (Phase I,   rapid SVD consolidation)
        step ~2563 (Phase II,  consolidated plateau)
        step ~9809 (Phase III mid, late-training restructuring)
        step 24000 (Phase III final)

  2. Use the randomized augmented files (suffix _random.npz) where
     position is decoupled from chunk identity. Coarse-bin position
     into a small number of quantile buckets (default 5) so each
     bucket has enough pilots for kurtosis estimation, and so the
     overall NMI(next; position) matches the original scheme (~0.25
     at 5 buckets, per the diagnose_nmi.py output).

  3. Compare the kurtosis-reduction trajectories across phases for
     each partition variable. This addresses two questions:

     (a) §8.3 trajectory question: does the context-mixture structure
         (C verdict) exist at the Phase II consolidation plateau, or
         is it a Phase III phenomenon? If baseline kurtosis is low at
         Phase II and high at Phase III, the structure emerges during
         late training.

     (b) §5.2 identifiability: with coarse-binned position now at
         comparable NMI to next-token, does position still collapse
         the C-verdict kurtosis, or does next-token collapse it more
         strongly? Differential effectiveness at matched entanglement
         distinguishes the variables.

Output:
    run_dir/multiview/model_abc/d9_crossphase_subconditioning.npz
    run_dir/multiview/model_abc/figures/d9_crossphase_subconditioning.png

Usage:
    python crossphase_subcondition.py --run-dir ../phase1_runs_gelu
    python crossphase_subcondition.py --run-dir ../phase1_runs_gelu \\
        --position-bins 5 10 --min-subbundle 20
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
    checkpoints_in_seed,
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
# Phase labels for the default checkpoints.
# ----------------------------------------------------------------------
PHASE_LABELS = {
    479:   "Phase I (~step 500: rapid SVD consolidation)",
    2563:  "Phase II (~step 2500: consolidation plateau)",
    9809:  "Phase III mid (~step 10000: late-training restructuring)",
    24000: "Phase III final (step 24000)",
}


# ----------------------------------------------------------------------
# Kurtosis: shared PC basis at each layer, per-coord excess kurtosis,
# averaged across coords. Same convention as subcondition.py.
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


def _quantile_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Coarse-bin a 1-D array by quantiles."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros_like(values, dtype=np.int32)
    binned = np.searchsorted(edges, values, side="right") - 1
    return np.clip(binned, 0, len(edges) - 2).astype(np.int32)


def _aggregate_kurt(
    states_v: np.ndarray,
    group_ids: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    min_subbundle: int,
) -> Tuple[np.ndarray, int]:
    """Sample-size-weighted mean per-layer kurtosis across sub-bundles
    defined by group_ids."""
    L = states_v.shape[0]
    unique, counts = np.unique(group_ids, return_counts=True)
    keep = unique[counts >= min_subbundle]
    if keep.size == 0:
        return np.full(L, np.nan), 0
    weights, kurts = [], []
    for u in keep:
        mask = (group_ids == u)
        n_sub = int(mask.sum())
        sub = states_v[:, mask, :]
        k = _per_layer_kurtosis(sub, basis, means)
        kurts.append(k)
        weights.append(n_sub)
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
# NMI helper for reporting the entanglement at each binning.
# ----------------------------------------------------------------------
def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    n = a.size
    if n < 2:
        return float("nan")
    au, ai = np.unique(a, return_inverse=True)
    bu, bi = np.unique(b, return_inverse=True)
    joint = np.zeros((au.size, bu.size), dtype=np.float64)
    np.add.at(joint, (ai, bi), 1.0)
    joint /= n
    p_a = joint.sum(1)
    p_b = joint.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(p_a, p_b)
        ratio = np.where((joint > 0) & (denom > 0), joint / denom, 1.0)
        mi = float(np.sum(joint * np.log(np.where(ratio > 0, ratio, 1.0))))
    h_a = -float(np.sum(p_a[p_a > 0] * np.log(p_a[p_a > 0])))
    if h_a <= 0:
        return float("nan")
    return mi / h_a


# ----------------------------------------------------------------------
# Per-token analysis for one (seed, step) augmented file.
# ----------------------------------------------------------------------
def analyze_token(
    states: np.ndarray,
    input_ids: np.ndarray,
    next_ids: np.ndarray,
    pred_ids: np.ndarray,
    positions: np.ndarray,
    basis: np.ndarray,
    means: np.ndarray,
    token_id: int,
    min_subbundle: int,
    position_bins: List[int],
) -> Dict:
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_subbundle * 2:
        return {"insufficient": True, "n_total": n_total}

    states_v = states[:, mask, :]
    next_v = next_ids[mask]
    pred_v = pred_ids[mask]
    pos_v = positions[mask]
    L = states_v.shape[0]

    baseline = _per_layer_kurtosis(states_v, basis, means)
    next_kurt, n_next = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)
    pred_kurt, n_pred = _aggregate_kurt(
        states_v, pred_v, basis, means, min_subbundle)

    # Position kurtosis at multiple binning resolutions.
    pos_results = {}
    for n_bins in position_bins:
        pos_binned = _quantile_bin(pos_v, n_bins)
        pos_kurt, n_pos = _aggregate_kurt(
            states_v, pos_binned, basis, means, min_subbundle)
        nmi_pos_next = _nmi(next_v, pos_binned)
        pos_results[n_bins] = {
            "kurt": pos_kurt,
            "n_subbundles": n_pos,
            "nmi_with_next": nmi_pos_next,
        }

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline_kurt": baseline,
        "next_kurt": next_kurt,
        "pred_kurt": pred_kurt,
        "n_next_subbundles": n_next,
        "n_pred_subbundles": n_pred,
        "pos_results": pos_results,
    }


def analyze_checkpoint(
    run_dir: str, seed: int, step: int,
    tids: np.ndarray, min_subbundle: int,
    position_bins: List[int], max_pc_dim: int,
) -> Optional[Dict]:
    aug_path = randomized_augmented_path(run_dir, seed, step)
    if not os.path.exists(aug_path):
        print(f"  [skip] missing {aug_path}")
        return None

    aug = load_augmented_activations(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    pred_ids = aug["pred_ids"]
    positions = aug["positions"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")

    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results_per_token = {}
    for tok in tids:
        r = analyze_token(
            states, input_ids, next_ids, pred_ids, positions,
            basis, means, int(tok),
            min_subbundle=min_subbundle,
            position_bins=position_bins,
        )
        results_per_token[int(tok)] = r

    n_valid = sum(1 for r in results_per_token.values()
                  if not r.get("insufficient", True))
    print(f"    {n_valid}/{tids.size} tokens with sufficient samples")
    return {
        "seed": seed, "step": step,
        "n_pilots_total": int(input_ids.size),
        "results_per_token": results_per_token,
    }


# ----------------------------------------------------------------------
# Aggregate per-checkpoint results to mean profiles across tokens.
# ----------------------------------------------------------------------
def aggregate_profiles(
    checkpoint_results: Dict,
    position_bins: List[int],
) -> Dict[str, np.ndarray]:
    """Sample-weighted mean across tokens of each partition's kurtosis
    profile. Returns dict with keys: 'baseline', 'next', 'pred',
    'pos_<bins>'."""
    valid = [r for r in checkpoint_results["results_per_token"].values()
             if not r.get("insufficient", True)]
    if not valid:
        return {}
    L = valid[0]["baseline_kurt"].size
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights = weights / weights.sum()

    def _wmean(key_or_extractor):
        if isinstance(key_or_extractor, str):
            arr = np.stack([r[key_or_extractor] for r in valid])
        else:
            arr = np.stack([key_or_extractor(r) for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]
            v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=weights[v]))
        return out

    profiles = {
        "baseline": _wmean("baseline_kurt"),
        "next": _wmean("next_kurt"),
        "pred": _wmean("pred_kurt"),
    }
    for n_bins in position_bins:
        profiles[f"pos_{n_bins}"] = _wmean(
            lambda r, b=n_bins: r["pos_results"][b]["kurt"])
    return profiles


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(
    run_dir: str,
    checkpoint_aggregates: Dict[int, Dict],
    position_bins: List[int],
    seed: int,
) -> None:
    steps_present = sorted(checkpoint_aggregates.keys())
    n_panels = len(steps_present)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(2, n_panels, figsize=(4.2 * n_panels, 9))
    if n_panels == 1:
        axes = axes[:, None]

    colors = {
        "baseline": "k",
        "next": "C0",
        "pred": "C1",
    }
    pos_cmap = plt.cm.viridis(np.linspace(0.3, 0.9, len(position_bins)))

    # Top row: per-checkpoint partition comparison.
    for col_idx, step in enumerate(steps_present):
        ax = axes[0, col_idx]
        profiles = checkpoint_aggregates[step]
        if not profiles:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"step {step}")
            continue
        L = profiles["baseline"].size
        layers = np.arange(L)

        ax.plot(layers, profiles["baseline"], "-", color=colors["baseline"],
                lw=2.5, label="baseline | v_i")
        ax.plot(layers, profiles["next"], "-", color=colors["next"],
                lw=1.7, label="| v_i, next")
        ax.plot(layers, profiles["pred"], "-", color=colors["pred"],
                lw=1.7, label="| v_i, pred")
        for ci, n_bins in enumerate(position_bins):
            key = f"pos_{n_bins}"
            ax.plot(layers, profiles[key], "--", color=pos_cmap[ci], lw=1.4,
                    label=f"| v_i, pos_bin({n_bins})")
        ax.axhline(0.0, color="k", lw=0.7, ls=":")
        ax.set_xlabel("layer state index t")
        if col_idx == 0:
            ax.set_ylabel("excess kurtosis (PC space)")
        phase_label = PHASE_LABELS.get(step, f"step {step}")
        # Truncate phase label for title display.
        title_lines = phase_label.replace(":", ":\n", 1)
        ax.set_title(title_lines, fontsize=9)
        ax.legend(fontsize=7, loc="best")

    # Bottom row: kurtosis reduction (baseline - partitioned) for each
    # partition variable across checkpoints. This makes the cross-phase
    # comparison direct: when does each partition variable start to
    # reduce kurtosis?
    for col_idx, step in enumerate(steps_present):
        ax = axes[1, col_idx]
        profiles = checkpoint_aggregates[step]
        if not profiles:
            continue
        baseline = profiles["baseline"]
        L = baseline.size
        layers = np.arange(L)

        ax.plot(layers, baseline - profiles["next"], "-",
                color=colors["next"], lw=1.7, label="next reduction")
        ax.plot(layers, baseline - profiles["pred"], "-",
                color=colors["pred"], lw=1.7, label="pred reduction")
        for ci, n_bins in enumerate(position_bins):
            key = f"pos_{n_bins}"
            ax.plot(layers, baseline - profiles[key], "--",
                    color=pos_cmap[ci], lw=1.4,
                    label=f"pos_bin({n_bins}) reduction")
        ax.axhline(0.0, color="k", lw=0.7, ls=":")
        ax.set_xlabel("layer state index t")
        if col_idx == 0:
            ax.set_ylabel("kurtosis reduction from baseline")
        ax.set_title(f"reductions at step {step}", fontsize=9)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(
        f"D9: cross-phase sub-conditioning, seed {seed}, randomized stage A",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir),
                       "d9_crossphase_subconditioning.png")
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
    ap.add_argument("--seed", type=int, default=0,
                    help="Which training seed to analyze. Default 0.")
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS,
                    help=f"Checkpoint steps to process. Default {DEFAULT_STEPS}.")
    ap.add_argument("--position-bins", type=int, nargs="+",
                    default=[5, 10, 20],
                    help="Position quantile-binning counts to test. "
                         "Default [5, 10, 20]. The diagnose_nmi.py output "
                         "suggested 5 is matched-NMI with the original "
                         "scheme.")
    ap.add_argument("--min-subbundle", type=int, default=20)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")
    print(f"Checkpoints: {args.steps}")
    print(f"Position binning: {args.position_bins}")
    print(f"min_subbundle = {args.min_subbundle}, "
          f"max_pc_dim = {args.max_pc_dim}")

    # Process each checkpoint.
    aggregates: Dict[int, Dict] = {}
    raw_per_checkpoint: Dict[int, Dict] = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, f'step {step}')}:")
        result = analyze_checkpoint(
            args.run_dir, args.seed, step, tids,
            min_subbundle=args.min_subbundle,
            position_bins=args.position_bins,
            max_pc_dim=args.max_pc_dim,
        )
        if result is None:
            continue
        raw_per_checkpoint[step] = result
        aggregates[step] = aggregate_profiles(result, args.position_bins)
        # Print a representative-layer summary.
        prof = aggregates[step]
        if prof:
            L = prof["baseline"].size
            mid = L // 2
            print(f"    layer {mid} kurtoses:")
            print(f"      baseline    = {prof['baseline'][mid]:.3f}")
            print(f"      | next      = {prof['next'][mid]:.3f}  "
                  f"(reduction {prof['baseline'][mid] - prof['next'][mid]:.3f})")
            print(f"      | pred      = {prof['pred'][mid]:.3f}  "
                  f"(reduction {prof['baseline'][mid] - prof['pred'][mid]:.3f})")
            for n_bins in args.position_bins:
                k = prof[f"pos_{n_bins}"][mid]
                print(f"      | pos_{n_bins:>2}    = {k:.3f}  "
                      f"(reduction {prof['baseline'][mid] - k:.3f})")
            # Show mean NMI(next; pos_binned) at each binning.
            for n_bins in args.position_bins:
                valid_tokens = [
                    r for r in result["results_per_token"].values()
                    if not r.get("insufficient", True)
                ]
                nmis = [r["pos_results"][n_bins]["nmi_with_next"]
                        for r in valid_tokens]
                if nmis:
                    print(f"      NMI(next; pos_{n_bins}) mean = "
                          f"{np.mean(nmis):.3f}")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed:.1f}s")

    # Save raw arrays.
    out_npz_arrays = {}
    for step, prof in aggregates.items():
        for key, arr in prof.items():
            out_npz_arrays[f"step_{step}_{key}"] = arr
    out_npz_arrays["steps"] = np.array(sorted(aggregates.keys()),
                                       dtype=np.int64)
    out_npz_arrays["position_bins"] = np.array(args.position_bins,
                                               dtype=np.int32)
    out_npz_arrays["seed"] = np.int32(args.seed)
    out_path = os.path.join(output_root(args.run_dir),
                            "d9_crossphase_subconditioning.npz")
    np.savez(out_path, **out_npz_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates, args.position_bins,
                 seed=args.seed)


if __name__ == "__main__":
    main()
