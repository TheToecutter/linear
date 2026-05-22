"""
Sub-conditioning diagnostic for the Model C verdict.

If the C verdict from model_abc_discriminator.py reflects the
"conditional bundle is itself a mixture over context" interpretation,
then sub-conditioning on context (next token / predicted token /
position) should collapse the within-input kurtosis toward Gaussian
levels.

Procedure for each top-k forward token v_i:

    1. Compute kurtosis profile of the full input-conditioned bundle
       p(x_t | v_i). This is the baseline (the D4a number).

    2. Partition the v_i-conditioned pilots into sub-bundles by:
          (a) next_token       p(x_t | v_i, next_token = w)
          (b) predicted_token  p(x_t | v_i, pred_token  = w)
          (c) position         p(x_t | v_i, position    = p)

    3. Compute kurtosis profile of each sub-bundle (with enough samples).
       Aggregate as sample-weighted mean of sub-bundle kurtosis at each
       layer.

    4. Compare baseline kurtosis vs sub-conditioned kurtosis. The
       interpretation:

       baseline_kurtosis HIGH, subcond_kurtosis LOW  -> context mixture
         confirmed. The C verdict is "conditional is a sub-mixture";
         further conditioning yields Gaussian bundles.

       baseline_kurtosis HIGH, subcond_kurtosis HIGH -> not a context
         mixture (at least not on this axis). Could be heavy tails
         intrinsic to the token, mode-switching by some other latent
         variable, or genuinely non-Gaussian structure.

       The contrast is meaningful only where sub-bundles have enough
       samples for a stable kurtosis estimate. Default minimum: 30.

Outputs:
    run_dir/multiview/model_abc/d6_subconditioning.npz
    run_dir/multiview/model_abc/figures/d6_subconditioning.png

Usage:
    python subcondition.py --run-dir ../phase1_runs_gelu
    python subcondition.py --run-dir ../phase1_runs_gelu --seed 0 \\
        --step 24000 --min-subbundle 30
"""

from __future__ import annotations

import argparse
import json
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
    seeds_in_run,
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
# Kurtosis computation. We use per-coordinate excess kurtosis averaged
# over coordinates, computed in the global PCA basis at each layer.
# This matches the D5 reconstruction kurtosis so the comparison is
# apples-to-apples with the D4a/D5 numbers.
# ----------------------------------------------------------------------
def _per_layer_kurtosis(
    states_sub: np.ndarray,           # (L, n, H)
    basis_per_layer: np.ndarray,      # (L, d, H)  global PCA basis
    global_mean_per_layer: np.ndarray,  # (L, H)
) -> np.ndarray:
    """Compute per-layer excess kurtosis of a sub-bundle, projected onto
    a *layer-specific* shared PC basis.

    Returns (L,) array, NaN where n is too small.
    """
    L, n, H = states_sub.shape
    out = np.full(L, np.nan, dtype=np.float64)
    if n < 5:
        return out
    for t in range(L):
        X = states_sub[t].astype(np.float64) - global_mean_per_layer[t]
        Z = X @ basis_per_layer[t].T                          # (n, d)
        var = Z.var(axis=0)
        if not np.all(var > 0):
            continue
        m4 = ((Z - Z.mean(axis=0)) ** 4).mean(axis=0)
        out[t] = float(np.mean(m4 / (var ** 2) - 3.0))
    return out


def _global_basis(states: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the top-d PC basis at each layer from the full bundle.

    Returns (basis, means):
        basis: (L, d, H)
        means: (L, H)
    """
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
# Main analysis.
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
    min_subbundle: int = 30,
    verbose: bool = False,
) -> Dict:
    """Compute baseline + sub-conditioned kurtosis profiles for one token.

    Returns dict with:
        baseline_kurt: (L,)            -- kurtosis of full v_i bundle
        subcond_next_kurt: (L,)        -- sample-weighted mean over next-
                                          token sub-bundles
        subcond_pred_kurt: (L,)        -- same, conditioning on pred token
        subcond_pos_kurt:  (L,)        -- same, conditioning on position
        n_subbundles_next: int         -- count of sub-bundles with n>=min
        n_subbundles_pred: int
        n_subbundles_pos:  int
        n_total: int                   -- total pilots for this token
    """
    L = states.shape[0]
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_subbundle * 2:
        return {"insufficient": True, "n_total": n_total}

    states_v = states[:, mask, :]                       # (L, n_total, H)
    next_v = next_ids[mask]
    pred_v = pred_ids[mask]
    pos_v = positions[mask]

    baseline = _per_layer_kurtosis(states_v, basis, means)

    def _subcond(group_ids: np.ndarray) -> Tuple[np.ndarray, int]:
        """Group by the values in group_ids, compute per-subbundle
        kurtosis, return sample-weighted mean across sub-bundles."""
        unique, counts = np.unique(group_ids, return_counts=True)
        keep = unique[counts >= min_subbundle]
        if keep.size == 0:
            return np.full(L, np.nan), 0
        # Stack per-subbundle kurtosis weighted by sample size.
        kurt_stack = []
        weights = []
        for u in keep:
            sub_mask = (group_ids == u)
            n_sub = int(sub_mask.sum())
            sub_states = states_v[:, sub_mask, :]
            k = _per_layer_kurtosis(sub_states, basis, means)
            kurt_stack.append(k)
            weights.append(n_sub)
        kurt_arr = np.stack(kurt_stack)                    # (n_groups, L)
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()
        # NaN-safe weighted mean: ignore NaN rows per column.
        weighted = np.full(L, np.nan)
        for t in range(L):
            col = kurt_arr[:, t]
            valid = np.isfinite(col)
            if valid.sum() == 0:
                continue
            weighted[t] = float(np.average(col[valid], weights=w[valid]))
        return weighted, int(keep.size)

    next_k, n_next = _subcond(next_v)
    pred_k, n_pred = _subcond(pred_v)
    pos_k, n_pos = _subcond(pos_v)

    if verbose:
        print(f"    token {token_id}: n_total={n_total}, "
              f"sub-bundles next={n_next} pred={n_pred} pos={n_pos}")

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline_kurt": baseline,
        "subcond_next_kurt": next_k,
        "subcond_pred_kurt": pred_k,
        "subcond_pos_kurt": pos_k,
        "n_subbundles_next": n_next,
        "n_subbundles_pred": n_pred,
        "n_subbundles_pos": n_pos,
    }


def plot_subconditioning(
    run_dir: str,
    results: List[Dict],
    seed: int, step: int,
) -> None:
    valid = [r for r in results if not r.get("insufficient", True)]
    if not valid:
        print("[plot] no valid tokens; skipping figure")
        return
    L = valid[0]["baseline_kurt"].size
    layers = np.arange(L)

    # Aggregate: weight by n_total when averaging across tokens.
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights /= weights.sum()

    def _wmean(key: str) -> np.ndarray:
        arr = np.stack([r[key] for r in valid])           # (n_tok, L)
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]
            v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=weights[v]))
        return out

    baseline = _wmean("baseline_kurt")
    next_k = _wmean("subcond_next_kurt")
    pred_k = _wmean("subcond_pred_kurt")
    pos_k = _wmean("subcond_pos_kurt")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: aggregate comparison.
    ax = axes[0]
    ax.plot(layers, baseline, "k-", lw=2.5,
            label="baseline: p(x_t | v_i)")
    ax.plot(layers, next_k, "C0-", lw=1.7,
            label="sub-cond on next token: p(x_t | v_i, next)")
    ax.plot(layers, pred_k, "C1-", lw=1.7,
            label="sub-cond on pred token: p(x_t | v_i, pred)")
    ax.plot(layers, pos_k, "C2--", lw=1.5,
            label="sub-cond on position (control)")
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis (mean over coords in shared PC basis)")
    ax.set_title(f"Sub-conditioning: aggregate over forward tokens "
                 f"(seed {seed} step {step})")
    ax.legend(fontsize=9, loc="best")

    # Right: per-token baseline vs (input, next).
    ax = axes[1]
    for r in valid:
        ax.plot(layers, r["baseline_kurt"], "k-", alpha=0.25, lw=1)
        ax.plot(layers, r["subcond_next_kurt"], "C0-", alpha=0.25, lw=1)
    # Heavy lines for the aggregate.
    ax.plot(layers, baseline, "k-", lw=2.5, label="baseline (mean)")
    ax.plot(layers, next_k, "C0-", lw=2.5, label="sub-cond on next (mean)")
    ax.axhline(0.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Per-token (light) and aggregate (heavy)")
    ax.legend(fontsize=9, loc="best")

    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d6_subconditioning.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Sub-conditioning diagnostic for Model C interpretation.")
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=None,
                    help="Default: last common step.")
    ap.add_argument("--min-subbundle", type=int, default=30,
                    help="Minimum samples to include a sub-bundle.")
    ap.add_argument("--max-pc-dim", type=int, default=32,
                    help="PC subspace for kurtosis comparison.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(output_root(args.run_dir), exist_ok=True)
    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    if args.step is None:
        steps_avail = [s for s, _ in checkpoints_in_seed(args.run_dir, args.seed)]
        if not steps_avail:
            print(f"No checkpoints for seed {args.seed}")
            sys.exit(1)
        args.step = steps_avail[-1]

    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    aug_path = augmented_path(args.run_dir, args.seed, args.step)
    if not os.path.exists(aug_path):
        print(f"Missing augmented file: {aug_path}")
        sys.exit(1)

    if args.dry_run:
        sz_mb = os.path.getsize(aug_path) / (1024 ** 2)
        est_min = tids.size * 0.3 + 1.0   # rough
        print(f"Would analyze {tids.size} tokens at seed {args.seed} "
              f"step {args.step}")
        print(f"Augmented file: {sz_mb:.1f} MB at {aug_path}")
        print(f"min-subbundle = {args.min_subbundle}, "
              f"max-pc-dim = {args.max_pc_dim}")
        print(f"Estimated runtime ~{est_min:.0f} min")
        return

    print(f"Loading {aug_path} ...")
    aug = load_augmented_activations(aug_path)
    states = aug["states"]              # (L, N, H)
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    pred_ids = aug["pred_ids"]
    positions = aug["positions"]
    L, N, H = states.shape
    print(f"  states {states.shape}, pilots {N}")

    print(f"Computing global PC basis (top-{args.max_pc_dim}) per layer ...")
    basis, means = _global_basis(states, args.max_pc_dim)

    print(f"Sub-conditioning analysis on {tids.size} tokens ...")
    results = []
    t0 = time.time()
    for ii, tok in enumerate(tids):
        t_tok = time.time()
        r = analyze_token(
            states, input_ids, next_ids, pred_ids, positions,
            basis, means, int(tok),
            min_subbundle=args.min_subbundle, verbose=True,
        )
        results.append(r)
        if r.get("insufficient"):
            print(f"  [{ii+1}/{tids.size}] tok {int(tok)}: insufficient "
                  f"(N={r['n_total']})")
            continue
        # Spot-check at a few layers.
        bl = r["baseline_kurt"]
        nx = r["subcond_next_kurt"]
        # Pick a middle interior layer for a printable summary.
        mid = L // 2
        print(f"  [{ii+1}/{tids.size}] tok {int(tok)} (N={r['n_total']}): "
              f"layer {mid} baseline={bl[mid]:.2f} "
              f"subcond_next={nx[mid]:.2f} "
              f"({time.time() - t_tok:.1f}s)")
    print(f"Done in {(time.time() - t0)/60:.1f} min")

    # Save raw arrays.
    out_path = os.path.join(output_root(args.run_dir), "d6_subconditioning.npz")
    valid = [r for r in results if not r.get("insufficient", True)]
    if valid:
        # Stack into arrays aligned with the forward token order.
        baseline_arr = np.full((tids.size, L), np.nan)
        next_arr = np.full((tids.size, L), np.nan)
        pred_arr = np.full((tids.size, L), np.nan)
        pos_arr = np.full((tids.size, L), np.nan)
        n_total = np.zeros(tids.size, dtype=np.int64)
        n_sub_next = np.zeros(tids.size, dtype=np.int64)
        n_sub_pred = np.zeros(tids.size, dtype=np.int64)
        n_sub_pos = np.zeros(tids.size, dtype=np.int64)
        for k, tok in enumerate(tids):
            r = results[k]
            if r.get("insufficient"):
                n_total[k] = r.get("n_total", 0)
                continue
            baseline_arr[k] = r["baseline_kurt"]
            next_arr[k] = r["subcond_next_kurt"]
            pred_arr[k] = r["subcond_pred_kurt"]
            pos_arr[k] = r["subcond_pos_kurt"]
            n_total[k] = r["n_total"]
            n_sub_next[k] = r["n_subbundles_next"]
            n_sub_pred[k] = r["n_subbundles_pred"]
            n_sub_pos[k] = r["n_subbundles_pos"]
        np.savez(
            out_path,
            seed=np.int32(args.seed), step=np.int64(args.step),
            tids=tids, n_total=n_total,
            n_subbundles_next=n_sub_next,
            n_subbundles_pred=n_sub_pred,
            n_subbundles_pos=n_sub_pos,
            baseline_kurt=baseline_arr,
            subcond_next_kurt=next_arr,
            subcond_pred_kurt=pred_arr,
            subcond_pos_kurt=pos_arr,
            min_subbundle=np.int32(args.min_subbundle),
            max_pc_dim=np.int32(args.max_pc_dim),
        )
        print(f"[npz] -> {out_path}")

    plot_subconditioning(args.run_dir, results, args.seed, args.step)


if __name__ == "__main__":
    main()
