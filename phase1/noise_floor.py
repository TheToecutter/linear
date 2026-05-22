"""
Noise-floor calibrator for the Model A/B/C discriminator suite.

Measures the within-token bootstrap distribution of each discriminator's
core statistic, on the actual pilot counts available in your campaign.
The result is a JSON of per-statistic noise floors that can be loaded by
model_abc_discriminator.py to set Model-A-vs-B thresholds adaptively
instead of using my conservative first-guess defaults.

Logic:
  Under Model A, two tokens i and j have the same true within-input
  covariance Sigma_0(t). Any observed difference is finite-sample noise
  from N_i pilots for token i and N_j pilots for token j.

  We estimate the finite-sample noise floor by resampling within a
  single token's pilots (with replacement), recomputing the statistic
  on each resample, and reporting the standard deviation across
  resamples. This is the "if the truth were fixed, how much would the
  measurement bounce around" floor.

  For a cross-token comparison (CV across tokens), the relevant noise
  floor is roughly sqrt(2) * mean(within-token bootstrap std / mean),
  because the cross-token CV sees noise from both tokens. We report
  both pieces so the calling script can compose them however it wants.

  We also report a *cross-seed* noise floor — same statistic, same
  token, across the four seeds at the final checkpoint. This is a
  sanity check: under Model A with seed-stable training (Phase 1's
  result), this should be comparable to the bootstrap floor. If it's
  much larger, seed-to-seed initialization noise is contributing more
  than finite-sample noise, which itself is worth knowing.

Statistics calibrated:
  D1 — trace(Sigma_i(t))            [per layer]
  D1 — effective_rank_i(t)          [per layer]
  D3 — lambda_i                     [scalar per token]
  D3 — log_alpha_i                  [scalar per token]
  D4a — kurtosis_i(t)               [per layer]

D2 (principal angles) is a *pair* statistic and noise floor for it is
qualitatively different — we compute it separately as a self-pair angle
(angle between two bootstrap-resampled bases of the *same* token's
bundle). That's the most direct floor for "could two different-looking
bases be the same token's bundle observed twice."

Usage:
    python noise_floor.py --run-dir ../phase1_runs_gelu

    # Use seed 0 step 24000 (default) with B=100 bootstraps:
    python noise_floor.py --run-dir ../phase1_runs_gelu --seed 0 --step 24000

    # Tighter calibration, more bootstraps:
    python noise_floor.py --run-dir ../phase1_runs_gelu --n-bootstrap 200

Output:
    run_dir/multiview/model_abc/noise_floor.json
    run_dir/multiview/model_abc/figures/noise_floor.png

The JSON has top-level keys 'bootstrap' and 'cross_seed', each with
per-statistic bootstrap stds (and CVs where applicable).
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

from analyze import recover_linear_flow
from multiview import (
    TokenSet,
    load_augmented_activations,
)
from multiview_campaign import (
    seeds_in_run,
    checkpoints_in_seed,
    mvr_dir,
    augmented_path,
    load_token_sets,
)


# ----------------------------------------------------------------------
# Path helpers (mirror model_abc_discriminator.py).
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


def ensure_dirs(run_dir: str) -> None:
    os.makedirs(output_root(run_dir), exist_ok=True)
    os.makedirs(figures_dir(run_dir), exist_ok=True)


# ----------------------------------------------------------------------
# Statistic extractors. Each takes a (L, N, H) per-token activation
# tensor and returns a dict of statistics.
# ----------------------------------------------------------------------
def _statistic_pack(sub_states: np.ndarray) -> Dict:
    """Compute the panel of statistics we calibrate from a per-token
    activation tensor of shape (L, N, H).

    Returns:
        trace_per_layer: (L,)         -- sum of squared singular values / N
        effective_rank:  (L,)         -- entropy-based dim from recover_linear_flow
        kurtosis_per_layer: (L,)      -- as computed by recover_linear_flow
        log_alpha:       float        -- as fit by recover_linear_flow
        lambda_:         float        -- as fit by recover_linear_flow
        R_top:           (L, k_top, H) -- top-k_top right-singular vectors per
                                        layer, kept for D2 self-pair angles.
    """
    L, N, H = sub_states.shape
    flow = recover_linear_flow(sub_states, center=True)
    s = flow["singular_values"].astype(np.float64)        # (L, H)
    trace = (s ** 2).sum(axis=1) / float(N)
    return {
        "trace_per_layer": trace,
        "effective_rank": flow["effective_rank"].astype(np.float64),
        "kurtosis_per_layer": flow["kurtosis_per_layer"].astype(np.float64),
        "log_alpha": float(flow["log_alpha"]),
        "lambda_": float(flow["lambda"]),
        "R_top": flow["R"][:, : min(10, H), :].astype(np.float64),
    }


def _principal_angles_topk(R1: np.ndarray, R2: np.ndarray, k: int) -> np.ndarray:
    """Principal angles between two row-orthonormal bases (top-k rows)."""
    A = R1[:k]
    B = R2[:k]
    M = A @ B.T
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.sort(np.arccos(sv))


# ----------------------------------------------------------------------
# Bootstrap noise floor for one token.
# ----------------------------------------------------------------------
def bootstrap_one_token(
    states: np.ndarray,                # (L, N_total, H)
    mask: np.ndarray,                  # (N_total,) bool
    n_bootstrap: int,
    k_angle: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Resample the pilots of one token with replacement; recompute the
    statistic pack on each resample.

    Returns per-statistic arrays of shape (B, ...) and the bootstrap
    standard deviation summaries.
    """
    if rng is None:
        rng = np.random.default_rng(20260521)
    L = states.shape[0]
    sub_idx = np.nonzero(mask)[0]
    n = sub_idx.size
    if n < 20:
        return {"insufficient": True, "n_pilots": int(n)}

    boot = {
        "trace_per_layer": np.full((n_bootstrap, L), np.nan),
        "effective_rank": np.full((n_bootstrap, L), np.nan),
        "kurtosis_per_layer": np.full((n_bootstrap, L), np.nan),
        "log_alpha": np.full(n_bootstrap, np.nan),
        "lambda_": np.full(n_bootstrap, np.nan),
    }
    R_keep: List[np.ndarray] = []   # for self-pair angle floor; keep first few

    for b in range(n_bootstrap):
        sel = rng.choice(sub_idx, size=n, replace=True)
        sub = states[:, sel, :]
        try:
            pack = _statistic_pack(sub)
        except np.linalg.LinAlgError:
            continue
        boot["trace_per_layer"][b] = pack["trace_per_layer"]
        boot["effective_rank"][b] = pack["effective_rank"]
        boot["kurtosis_per_layer"][b] = pack["kurtosis_per_layer"]
        boot["log_alpha"][b] = pack["log_alpha"]
        boot["lambda_"][b] = pack["lambda_"]
        # Keep two random R basis snapshots (per layer) for self-pair angle.
        # Using two replicates is enough; sampling all of them would be huge.
        if b < 2:
            R_keep.append(pack["R_top"])

    # Self-pair principal angle: between two independent bootstrap bases.
    if len(R_keep) >= 2:
        R_a, R_b = R_keep[0], R_keep[1]
        self_pair_angles = np.array([
            _principal_angles_topk(R_a[t], R_b[t], k_angle) for t in range(L)
        ])  # (L, k_angle)
    else:
        self_pair_angles = np.full((L, k_angle), np.nan)

    summary = {
        "insufficient": False,
        "n_pilots": int(n),
        "n_bootstrap": int(n_bootstrap),
        "trace_per_layer": {
            "mean": np.nanmean(boot["trace_per_layer"], axis=0).tolist(),
            "std":  np.nanstd(boot["trace_per_layer"], axis=0, ddof=1).tolist(),
        },
        "effective_rank": {
            "mean": np.nanmean(boot["effective_rank"], axis=0).tolist(),
            "std":  np.nanstd(boot["effective_rank"], axis=0, ddof=1).tolist(),
        },
        "kurtosis_per_layer": {
            "mean": np.nanmean(boot["kurtosis_per_layer"], axis=0).tolist(),
            "std":  np.nanstd(boot["kurtosis_per_layer"], axis=0, ddof=1).tolist(),
        },
        "log_alpha": {
            "mean": float(np.nanmean(boot["log_alpha"])),
            "std":  float(np.nanstd(boot["log_alpha"], ddof=1)),
        },
        "lambda_": {
            "mean": float(np.nanmean(boot["lambda_"])),
            "std":  float(np.nanstd(boot["lambda_"], ddof=1)),
        },
        "self_pair_angles_rad": {
            "mean_top_k": np.nanmean(self_pair_angles).tolist(),
            "per_layer_mean": np.nanmean(self_pair_angles, axis=1).tolist(),
        },
    }
    return summary


# ----------------------------------------------------------------------
# Cross-seed noise floor (across the 4 seeds at the same step, same token).
# ----------------------------------------------------------------------
def cross_seed_noise_floor(
    run_dir: str, token_ids: np.ndarray, step: int,
    verbose: bool = True,
) -> Dict:
    """For each token, compute the cross-seed std of the statistic pack
    at the given step.

    Uses the saved per-token flow files from stage C (no recomputation).
    """
    from multiview import load_multi_view_result
    seeds = seeds_in_run(run_dir)
    pack_per_seed = {s: None for s in seeds}
    for s in seeds:
        d = mvr_dir(run_dir, s, step)
        if not os.path.exists(os.path.join(d, "meta.json")):
            continue
        # We don't need R for the cross-seed scalar/profile noise floor.
        r = load_multi_view_result(d, skip_arrays={"R", "pairwise_residual_variance"})
        pack_per_seed[s] = r.forward_flows

    if all(v is None for v in pack_per_seed.values()):
        return {"insufficient": True}

    # Probe L from one available token in one available seed.
    probe = next((flows for flows in pack_per_seed.values()
                  if flows), None)
    if probe is None:
        return {"insufficient": True}
    one = next(iter(probe.values()))
    L = one["singular_values"].shape[0]

    per_token = {}
    for tok in token_ids:
        tok = int(tok)
        trace_seeds = []
        erank_seeds = []
        kurt_seeds = []
        loga_seeds = []
        lam_seeds = []
        for s in seeds:
            flows = pack_per_seed.get(s)
            if not flows:
                continue
            f = flows.get(tok)
            if f is None or f.get("failed", False):
                continue
            n = float(f["n_pilots"])
            if n <= 1:
                continue
            tr = (f["singular_values"].astype(np.float64) ** 2).sum(axis=1) / n
            trace_seeds.append(tr)
            erank_seeds.append(f["effective_rank"].astype(np.float64))
            kurt_seeds.append(f["kurtosis_per_layer"].astype(np.float64))
            loga_seeds.append(float(f["log_alpha"]))
            lam_seeds.append(float(f["lambda"]))
        if len(trace_seeds) < 2:
            continue
        trace_arr = np.stack(trace_seeds)        # (S, L)
        erank_arr = np.stack(erank_seeds)
        kurt_arr = np.stack(kurt_seeds)
        loga_arr = np.array(loga_seeds)
        lam_arr = np.array(lam_seeds)
        per_token[tok] = {
            "n_seeds": int(trace_arr.shape[0]),
            "trace_per_layer": {
                "mean": trace_arr.mean(0).tolist(),
                "std":  trace_arr.std(0, ddof=1).tolist(),
            },
            "effective_rank": {
                "mean": erank_arr.mean(0).tolist(),
                "std":  erank_arr.std(0, ddof=1).tolist(),
            },
            "kurtosis_per_layer": {
                "mean": kurt_arr.mean(0).tolist(),
                "std":  kurt_arr.std(0, ddof=1).tolist(),
            },
            "log_alpha": {
                "mean": float(loga_arr.mean()),
                "std":  float(loga_arr.std(ddof=1)),
            },
            "lambda_": {
                "mean": float(lam_arr.mean()),
                "std":  float(lam_arr.std(ddof=1)),
            },
        }
    return {"insufficient": False, "per_token": per_token}


# ----------------------------------------------------------------------
# Aggregate: take the per-token bootstrap dicts and condense to floor
# thresholds the discriminator script can use.
# ----------------------------------------------------------------------
def derive_thresholds(
    bootstrap_per_token: Dict[int, Dict],
    cross_seed: Dict,
) -> Dict:
    """Convert raw bootstrap output into the threshold values the
    discriminator's verdict logic uses.

    The conversion is:
        cv_trace_threshold        = 3 * mean across tokens of
                                    (within-token bootstrap CV of trace)
                                    * sqrt(2)           # 2-token comparison
        cv_lambda_threshold       = 3 * mean across tokens of
                                    (within-token bootstrap std of lambda)
                                    / mean lambda
                                    * sqrt(2)
        pair_angle_threshold_deg  = 3 * mean across tokens of
                                    self-pair angle (radians, mean over
                                    layers and over top-k) converted to deg
        kurt_threshold            = 3 * mean across tokens of
                                    bootstrap std of kurtosis_per_layer
                                    (mean over layers)

    The "3x" multiplier is a ~3-sigma cutoff under a Gaussian null —
    cross-token spreads larger than this are unlikely to arise from
    finite-sample noise alone, and indicate genuine token-dependent
    structure (i.e., evidence against Model A).

    Reports both the bootstrap-derived thresholds and the
    cross-seed-derived ones for comparison; defaults to bootstrap.
    """
    valid_tokens = [t for t, r in bootstrap_per_token.items()
                    if not r.get("insufficient", True)]
    if not valid_tokens:
        return {"insufficient": True}

    # Within-token bootstrap CV of trace at each layer, mean across tokens.
    # Use median-over-layers as a stable summary.
    trace_cvs = []
    erank_cvs = []
    kurt_stds = []
    lambda_stds_rel = []
    self_pair_means = []
    log_alpha_stds = []
    for t in valid_tokens:
        r = bootstrap_per_token[t]
        # Trace CV per layer = std / mean.
        m_tr = np.array(r["trace_per_layer"]["mean"])
        s_tr = np.array(r["trace_per_layer"]["std"])
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_layers = np.where(m_tr > 0, s_tr / m_tr, np.nan)
        trace_cvs.append(float(np.nanmedian(cv_layers)))

        m_er = np.array(r["effective_rank"]["mean"])
        s_er = np.array(r["effective_rank"]["std"])
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_er = np.where(m_er > 0, s_er / m_er, np.nan)
        erank_cvs.append(float(np.nanmedian(cv_er)))

        s_k = np.array(r["kurtosis_per_layer"]["std"])
        kurt_stds.append(float(np.nanmedian(s_k)))

        m_l = r["lambda_"]["mean"]
        s_l = r["lambda_"]["std"]
        if m_l and abs(m_l) > 1e-9:
            lambda_stds_rel.append(abs(s_l / m_l))

        log_alpha_stds.append(abs(r["log_alpha"]["std"]))

        self_pair_means.append(r["self_pair_angles_rad"]["mean_top_k"])

    # Compose.
    sqrt2 = np.sqrt(2.0)
    cv_trace_floor = float(np.nanmean(trace_cvs)) * sqrt2 if trace_cvs else float("nan")
    cv_lambda_floor = (float(np.nanmean(lambda_stds_rel)) * sqrt2
                       if lambda_stds_rel else float("nan"))
    angle_floor_rad = float(np.nanmean(self_pair_means)) if self_pair_means else float("nan")
    kurt_floor = float(np.nanmean(kurt_stds)) if kurt_stds else float("nan")

    bootstrap_thresholds = {
        "cv_trace_threshold": 3.0 * cv_trace_floor,
        "cv_lambda_threshold": 3.0 * cv_lambda_floor,
        "pair_angle_threshold_deg": 3.0 * float(np.degrees(angle_floor_rad)),
        "kurt_threshold": 3.0 * kurt_floor,
        # Per-statistic raw floors (1-sigma).
        "_raw": {
            "cv_trace_floor_1sigma": cv_trace_floor / sqrt2,    # within-token
            "cv_lambda_floor_1sigma": cv_lambda_floor / sqrt2,
            "self_pair_angle_floor_rad": angle_floor_rad,
            "kurt_floor_1sigma": kurt_floor,
            "log_alpha_floor_1sigma": (
                float(np.nanmean(log_alpha_stds)) if log_alpha_stds else None
            ),
        },
        "n_tokens_used": len(valid_tokens),
        "_explanation": (
            "Thresholds are 3-sigma cutoffs under the Model A null. A "
            "cross-token CV (or angle) above the threshold is unlikely "
            "to arise from finite-sample noise at the observed pilot "
            "counts; treat as evidence against Model A in favor of B."
        ),
    }

    # Cross-seed comparison (same statistics, across the four seeds).
    cross_seed_thresholds = {}
    if not cross_seed.get("insufficient", True):
        cs_tokens = list(cross_seed["per_token"].keys())
        if cs_tokens:
            trace_cs_cvs = []
            lam_cs_rel = []
            for tok in cs_tokens:
                r = cross_seed["per_token"][tok]
                m_tr = np.array(r["trace_per_layer"]["mean"])
                s_tr = np.array(r["trace_per_layer"]["std"])
                with np.errstate(divide="ignore", invalid="ignore"):
                    cv = np.where(m_tr > 0, s_tr / m_tr, np.nan)
                trace_cs_cvs.append(float(np.nanmedian(cv)))
                m_l = r["lambda_"]["mean"]
                s_l = r["lambda_"]["std"]
                if abs(m_l) > 1e-9:
                    lam_cs_rel.append(abs(s_l / m_l))
            cross_seed_thresholds = {
                "cv_trace_floor_cross_seed": float(np.nanmean(trace_cs_cvs)),
                "cv_lambda_floor_cross_seed": (
                    float(np.nanmean(lam_cs_rel)) if lam_cs_rel else None
                ),
                "n_tokens_used": len(cs_tokens),
                "_note": (
                    "If cross-seed floors are much larger than bootstrap "
                    "floors, seed-init noise dominates finite-sample "
                    "noise. Use the larger of the two as the floor."
                ),
            }

    return {
        "insufficient": False,
        "bootstrap_thresholds": bootstrap_thresholds,
        "cross_seed_thresholds": cross_seed_thresholds,
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_noise_floor(
    run_dir: str,
    bootstrap_per_token: Dict[int, Dict],
    cross_seed: Dict,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    valid = [(t, r) for t, r in bootstrap_per_token.items()
             if not r.get("insufficient", True)]

    # Panel 1: per-token bootstrap CV of trace, vs layer.
    ax = axes[0]
    for tok, r in valid:
        m = np.array(r["trace_per_layer"]["mean"])
        s = np.array(r["trace_per_layer"]["std"])
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(m > 0, s / m, np.nan)
        ax.plot(cv, "-", alpha=0.55, lw=1, label=f"tok {tok} (N={r['n_pilots']})")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("bootstrap CV of trace(Sigma_i)")
    ax.set_title("Within-token bootstrap noise floor: trace")
    ax.legend(fontsize=7, ncol=2, loc="best")

    # Panel 2: bootstrap distribution of lambda per token.
    ax = axes[1]
    means = [r["lambda_"]["mean"] for _, r in valid]
    stds = [r["lambda_"]["std"] for _, r in valid]
    toks = [t for t, _ in valid]
    ax.errorbar(range(len(valid)), means, yerr=stds, fmt="o", capsize=3)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([str(t) for t in toks], rotation=45, fontsize=7)
    ax.set_xlabel("token id")
    ax.set_ylabel("lambda_i (mean +/- bootstrap std)")
    ax.set_title("Bootstrap noise floor on lambda_i")

    # Panel 3: self-pair principal angle vs layer.
    ax = axes[2]
    for tok, r in valid:
        per_layer = np.array(r["self_pair_angles_rad"]["per_layer_mean"])
        ax.plot(np.degrees(per_layer), "-", alpha=0.55, lw=1,
                label=f"tok {tok}")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("self-pair angle (deg, mean over top-k)")
    ax.set_title("Within-token bootstrap angle floor (for D2)")
    ax.legend(fontsize=7, ncol=2, loc="best")

    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "noise_floor.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap noise-floor calibrator for model_abc thresholds.")
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed to draw bootstrap samples from.")
    ap.add_argument("--step", type=int, default=None,
                    help="Checkpoint step. Defaults to the final checkpoint.")
    ap.add_argument("--n-bootstrap", type=int, default=100,
                    help="Number of bootstrap resamples per token.")
    ap.add_argument("--k-angle", type=int, default=10,
                    help="Top-k for self-pair principal angle.")
    ap.add_argument("--skip-cross-seed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ensure_dirs(args.run_dir)

    seeds = seeds_in_run(args.run_dir)
    if args.seed not in seeds:
        print(f"Seed {args.seed} not found in {args.run_dir}")
        sys.exit(1)
    if args.step is None:
        steps_avail = [step for step, _ in checkpoints_in_seed(
            args.run_dir, args.seed)]
        if not steps_avail:
            print(f"No checkpoints for seed {args.seed}")
            sys.exit(1)
        args.step = steps_avail[-1]
    print(f"Calibrating noise floor at seed {args.seed} step {args.step}")

    # Load token sets and augmented activations.
    try:
        forward_set, _, _ = load_token_sets(args.run_dir)
    except Exception as e:
        print(f"Could not load token sets: {e}")
        sys.exit(1)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")

    aug_path = augmented_path(args.run_dir, args.seed, args.step)
    if not os.path.exists(aug_path):
        print(f"Missing augmented file: {aug_path}")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry run: would bootstrap {args.n_bootstrap} resamples on "
              f"each of {tids.size} tokens, plus cross-seed comparison "
              f"across {len(seeds)} seeds.")
        print(f"Output: {os.path.join(output_root(args.run_dir), 'noise_floor.json')}")
        # Estimate time roughly: each bootstrap is one SVD on ~N x H,
        # ~1 second per bootstrap on a single CPU at H=768.
        est_min = args.n_bootstrap * tids.size * 1.0 / 60.0
        print(f"Estimated runtime: ~{est_min:.1f} min (single CPU)")
        return

    print(f"Loading augmented activations from {aug_path} ...")
    aug = load_augmented_activations(aug_path)
    states = aug["states"]                # (L, N, H)
    input_ids = aug["input_ids"]
    print(f"  states shape: {states.shape}, total pilots: {input_ids.size}")

    # Bootstrap each token.
    print(f"\nBootstrap noise floor across {tids.size} tokens, "
          f"B={args.n_bootstrap} each:")
    rng = np.random.default_rng(20260521)
    bootstrap_per_token = {}
    t0 = time.time()
    for ii, tok in enumerate(tids):
        mask = (input_ids == int(tok))
        n = int(mask.sum())
        if n < 20:
            print(f"  [{ii+1}/{tids.size}] tok {int(tok)}: only {n} "
                  f"pilots, skipping")
            bootstrap_per_token[int(tok)] = {"insufficient": True,
                                              "n_pilots": n}
            continue
        t_tok = time.time()
        result = bootstrap_one_token(states, mask, args.n_bootstrap,
                                     k_angle=args.k_angle, rng=rng)
        bootstrap_per_token[int(tok)] = result
        dt = time.time() - t_tok
        if not result.get("insufficient", True):
            tr_cv_layers = (
                np.array(result["trace_per_layer"]["std"]) /
                np.maximum(np.array(result["trace_per_layer"]["mean"]), 1e-12)
            )
            tr_cv_med = float(np.nanmedian(tr_cv_layers))
            lam_rel = (abs(result["lambda_"]["std"]) /
                       max(abs(result["lambda_"]["mean"]), 1e-12))
            print(f"  [{ii+1}/{tids.size}] tok {int(tok)} (N={n}): "
                  f"trace CV median {tr_cv_med:.3f}, lambda rel std "
                  f"{lam_rel:.3f}  ({dt:.1f}s)")

    # Cross-seed noise floor.
    cross_seed = {"insufficient": True}
    if not args.skip_cross_seed:
        print(f"\nCross-seed noise floor (same statistics, "
              f"step {args.step}, across {len(seeds)} seeds) ...")
        cross_seed = cross_seed_noise_floor(args.run_dir, tids, args.step)

    # Derive thresholds.
    print(f"\nDeriving threshold candidates ...")
    thresholds = derive_thresholds(bootstrap_per_token, cross_seed)

    # Save.
    out_json = os.path.join(output_root(args.run_dir), "noise_floor.json")
    with open(out_json, "w") as f:
        json.dump({
            "seed": args.seed,
            "step": args.step,
            "n_bootstrap": args.n_bootstrap,
            "n_tokens_attempted": int(tids.size),
            "bootstrap_per_token": bootstrap_per_token,
            "cross_seed": cross_seed,
            "thresholds": thresholds,
        }, f, indent=2)
    print(f"[json] -> {out_json}")

    plot_noise_floor(args.run_dir, bootstrap_per_token, cross_seed)

    # Friendly summary.
    if thresholds.get("insufficient", True):
        print("\nInsufficient data for threshold derivation.")
        return
    bt = thresholds["bootstrap_thresholds"]
    print(f"\n{'=' * 64}")
    print(f"Calibrated threshold candidates (3-sigma against Model A null):")
    print(f"  --cv-trace-threshold        {bt['cv_trace_threshold']:.4f}")
    print(f"  --cv-lambda-threshold       {bt['cv_lambda_threshold']:.4f}")
    print(f"  --pair-angle-threshold-deg  {bt['pair_angle_threshold_deg']:.2f}")
    print(f"  --kurt-threshold            {bt['kurt_threshold']:.4f}")
    print(f"\nRaw 1-sigma floors:")
    for k, v in bt["_raw"].items():
        if v is not None:
            print(f"  {k}: {v:.4g}")
    if thresholds["cross_seed_thresholds"]:
        ct = thresholds["cross_seed_thresholds"]
        print(f"\nCross-seed floors (for comparison; should be similar to bootstrap):")
        print(f"  cv_trace_floor_cross_seed:  {ct['cv_trace_floor_cross_seed']:.4f}")
        if ct.get("cv_lambda_floor_cross_seed") is not None:
            print(f"  cv_lambda_floor_cross_seed: {ct['cv_lambda_floor_cross_seed']:.4f}")
    print(f"\nTo use, pass these as CLI flags to model_abc_discriminator.py.")


if __name__ == "__main__":
    main()
