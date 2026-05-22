"""
Model A/B/C discriminator suite for the multiview campaign.

Resolves whether the input-conditioned forward bundles look like:

  Model A — Gaussian conditionals with a shared, token-independent
            covariance Sigma_0(t). The macro Gaussian arises by a GMM
            in which only the drift mu_i(t) differs across input tokens,
            and Sigma_i(t) = Sigma_0(t) for all i. Universal (alpha, lambda)
            live at the micro scale.

  Model B — Gaussian conditionals, but Sigma_i(t) depends on i. The macro
            Gaussian, if it exists, is an aggregate average that hides
            token-specific covariance structure.

  Model C — Conditionals are not Gaussian; the marginal can still appear
            Gaussian by mixing.

Operates on the disk output of multiview_campaign.py's stages A and C.

Five discriminators (all produce a row in the verdict CSV):

  D1   Token-CV of total within-variance trace and effective rank.
       Source: per-token flow files (singular_values, effective_rank).
       Cost:   seconds per (seed, step). Run across all checkpoints.

  D2   Principal-angle matrix of per-token SVD bases (and vs all-to-all).
       Source: per-token flow files (R matrix). Loads R, which is large.
       Cost:   tens of seconds per checkpoint. Run on final checkpoint
               of each seed by default; --d2-all-steps to expand.

  D3   Per-token (alpha_i, lambda_i) fits. Source: per-token flow files
       (log_alpha, lambda). Cost: negligible. Run across all checkpoints.

  D4   Gaussianity diagnostics on within-input bundles. Two parts:
        D4a — kurtosis_per_layer scalar, from per-token flow files.
              Cost: negligible. Run across all checkpoints.
        D4b — multivariate Mardia kurtosis + per-PC marginal Gaussianity
              tests. Requires the raw augmented activation file. Cost:
              ~minutes per checkpoint. Run on final checkpoint of seed 0
              by default; --d4b-all-seeds expands to all seeds at final.

  D5   GMM reconstruction test. Sample from the per-token Gaussians,
       compare moments (mean, covariance, kurtosis) against the directly
       observed all-to-all bundle. Source: stage C + augmented file.
       Cost: minutes per checkpoint. Run on final checkpoint of seed 0
       by default.

The script is idempotent: outputs are written per-checkpoint to disk,
re-running skips work that's already done.

Usage:
    python model_abc_discriminator.py --run-dir ../phase1_runs_gelu

    # Just D1+D3 across all checkpoints (cheap pass):
    python model_abc_discriminator.py --run-dir ../phase1_runs_gelu \
        --discriminators 1 3

    # Final-checkpoint deep dive on every seed:
    python model_abc_discriminator.py --run-dir ../phase1_runs_gelu \
        --d4b-all-seeds --d5-all-seeds

Output:
    run_dir/multiview/model_abc/
      d1_token_cv.npz                # (n_seeds, n_steps, L) CV arrays
      d2_principal_angles/           # one file per analyzed (seed, step)
        seed{S}_step{T:08d}.npz
      d3_per_token_fits.npz          # per-token (alpha, lambda) arrays
      d4a_kurtosis.npz               # per-token kurtosis arrays
      d4b_gaussianity/               # one file per analyzed (seed, step)
        seed{S}_step{T:08d}.npz
      d5_reconstruction/             # one file per analyzed (seed, step)
        seed{S}_step{T:08d}.npz
      verdict.csv                    # per (seed, step) summary verdict
      verdict.txt                    # human-readable summary
      figures/
        d1_within_cv.png
        d2_angle_matrix_final.png
        d3_alpha_lambda_scatter.png
        d4_qq_panels.png
        d5_reconstruction_diagnostic.png

The verdict logic (per-checkpoint) follows the priority from the
discussion: D1 decides A-vs-rest first, D3 confirms A-vs-B at the
universality level, D2 confirms A-vs-B at the basis level, D4 separates
C from {A,B}, D5 is the synthesis check. Per-layer verdicts are reported
because the right answer can be layer-dependent (e.g. A early, B late).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Stage C output loader and helpers from the existing codebase.
from multiview import (
    TokenSet,
    load_multi_view_result,
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
# Path conventions.
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


def ensure_dirs(run_dir: str) -> None:
    os.makedirs(output_root(run_dir), exist_ok=True)
    os.makedirs(figures_dir(run_dir), exist_ok=True)
    for sub in ("d2_principal_angles", "d4b_gaussianity", "d5_reconstruction"):
        os.makedirs(os.path.join(output_root(run_dir), sub), exist_ok=True)


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------
def _trace_within_per_token(flow: Dict) -> np.ndarray:
    """For a per-token flow, return per-layer trace(Sigma_i(t)) computed
    from the squared singular values normalized by sample count.

    The 'singular_values' field is per-layer s_d(t) from SVD of the
    centered subset; trace of the per-token covariance is then
    sum_d s_d(t)^2 / n_pilots.

    Returns NaN-array if the flow failed.
    """
    if flow.get("failed", False):
        L = flow["singular_values"].shape[0]
        return np.full(L, np.nan, dtype=np.float64)
    s = flow["singular_values"].astype(np.float64)         # (L, H)
    n = float(flow["n_pilots"])
    if n <= 1:
        return np.full(s.shape[0], np.nan)
    return (s ** 2).sum(axis=1) / n                         # (L,)


def _principal_angles_topk(R1: np.ndarray, R2: np.ndarray, k: int) -> np.ndarray:
    """Principal angles between two row-orthonormal bases.

    R1, R2 are (H, H) rotation matrices whose rows are orthonormal
    principal directions sorted by descending singular value. The top-k
    subspace is span of the first k rows.

    Returns angles in radians, sorted ascending, length k.
    """
    A = R1[:k]                          # (k, H)
    B = R2[:k]                          # (k, H)
    # Cross-correlation; SVD gives cosines of principal angles.
    M = A @ B.T                         # (k, k)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)
    return np.sort(np.arccos(sv))


def _safe_logspace_fit(layers: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Fit log v ~ log_alpha + lambda * log(layer index).

    Skips layer 0 (where v can be ~0 for conditional bundles by
    construction). Returns (log_alpha, lambda); both NaN if fewer than 3
    finite points are usable.
    """
    mask = np.isfinite(v) & (v > 0) & (layers > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(layers[mask].astype(np.float64))
    y = np.log(v[mask])
    A = np.vstack([np.ones_like(x), x]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _coefficient_of_variation(x: np.ndarray) -> float:
    """CV = std/mean of a 1-D array, ignoring NaN. Returns NaN if too few."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    m = x.mean()
    if abs(m) < 1e-30:
        return float("nan")
    return float(x.std(ddof=1) / abs(m))


# ----------------------------------------------------------------------
# D1: token-CV of within-input variance and effective rank.
# ----------------------------------------------------------------------
def run_d1(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    verbose: bool = True,
) -> Dict:
    """For every (seed, step), compute coefficient of variation across
    tokens of trace(Sigma_i(t)) and of effective_rank_i(t), per layer.

    Output array shape: (n_seeds, n_steps, L). Saves to d1_token_cv.npz.
    """
    if verbose:
        print(f"[D1] Token-CV across {len(seeds)} seeds x {len(common_steps)} "
              f"steps ...")

    # Probe shape from one available checkpoint.
    probe_seed, probe_step = seeds[0], common_steps[-1]
    probe = load_multi_view_result(mvr_dir(run_dir, probe_seed, probe_step),
                                   skip_arrays={"R", "pairwise_residual_variance"})
    L = probe.all_to_all["singular_values"].shape[0]

    cv_trace = np.full((len(seeds), len(common_steps), L), np.nan, dtype=np.float64)
    cv_erank = np.full_like(cv_trace, np.nan)
    mean_trace = np.full_like(cv_trace, np.nan)
    mean_erank = np.full_like(cv_trace, np.nan)

    for si, seed in enumerate(seeds):
        for ti, step in enumerate(common_steps):
            d = mvr_dir(run_dir, seed, step)
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            try:
                r = load_multi_view_result(
                    d, skip_arrays={"R", "pairwise_residual_variance"})
            except Exception as e:
                if verbose:
                    print(f"[D1] seed {seed} step {step}: load failed: {e}")
                continue

            flows = r.forward_flows
            if not flows:
                continue

            tids = sorted(flows.keys())
            # Per-token trace and effective rank profiles.
            tr_mat = np.stack([_trace_within_per_token(flows[t]) for t in tids],
                              axis=0)        # (n_tokens, L)
            er_mat = np.stack([
                flows[t]["effective_rank"].astype(np.float64)
                if not flows[t].get("failed", False)
                else np.full(L, np.nan) for t in tids
            ], axis=0)

            with np.errstate(invalid="ignore"):
                for t in range(L):
                    cv_trace[si, ti, t] = _coefficient_of_variation(tr_mat[:, t])
                    cv_erank[si, ti, t] = _coefficient_of_variation(er_mat[:, t])
                    mean_trace[si, ti, t] = np.nanmean(tr_mat[:, t])
                    mean_erank[si, ti, t] = np.nanmean(er_mat[:, t])

    out_path = os.path.join(output_root(run_dir), "d1_token_cv.npz")
    np.savez(
        out_path,
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(common_steps, dtype=np.int64),
        cv_trace=cv_trace,
        cv_erank=cv_erank,
        mean_trace=mean_trace,
        mean_erank=mean_erank,
    )
    if verbose:
        print(f"[D1] -> {out_path}")
    return {
        "cv_trace": cv_trace,
        "cv_erank": cv_erank,
        "mean_trace": mean_trace,
        "mean_erank": mean_erank,
    }


# ----------------------------------------------------------------------
# D2: principal-angle matrix of per-token bases.
# ----------------------------------------------------------------------
def run_d2(
    run_dir: str,
    seeds: List[int],
    steps_to_analyze: List[Tuple[int, int]],
    k: int = 10,
    verbose: bool = True,
) -> List[Dict]:
    """For each (seed, step) in steps_to_analyze, compute the top-k
    principal angle matrix between every pair of token SVD bases and
    between each token basis and the all-to-all basis.

    Saves one file per checkpoint to d2_principal_angles/.

    Returns the list of result dicts (also kept in memory for plotting).
    """
    if verbose:
        print(f"[D2] Principal-angle analysis on {len(steps_to_analyze)} "
              f"checkpoints, top-{k} subspace ...")
    results = []
    for seed, step in steps_to_analyze:
        d = mvr_dir(run_dir, seed, step)
        if not os.path.exists(os.path.join(d, "meta.json")):
            if verbose:
                print(f"[D2] seed {seed} step {step}: missing, skip")
            continue
        out_path = os.path.join(output_root(run_dir), "d2_principal_angles",
                                f"seed{seed}_step{step:08d}.npz")
        if os.path.exists(out_path):
            with np.load(out_path) as f:
                results.append({
                    "seed": seed, "step": step,
                    "tids": f["tids"], "mean_pair_angles": f["mean_pair_angles"],
                    "mean_vs_all_angles": f["mean_vs_all_angles"],
                    "pair_angle_matrix": f["pair_angle_matrix"],
                    "k": int(f["k"]),
                })
            continue

        if verbose:
            print(f"[D2] seed {seed} step {step}: loading R matrices ...")
        # We need R for this; cannot skip it.
        r = load_multi_view_result(d, skip_arrays={"pairwise_residual_variance"})
        flows = r.forward_flows
        R_all = r.all_to_all["R"]            # (L, H, H)
        L = R_all.shape[0]

        tids = sorted(t for t in flows.keys() if not flows[t].get("failed", False))
        n_tok = len(tids)
        if n_tok < 2:
            if verbose:
                print(f"[D2] seed {seed} step {step}: <2 valid tokens, skip")
            continue

        # Pair angle matrix per layer: (L, n_tok, n_tok, k).
        pair_mat = np.full((L, n_tok, n_tok, k), np.nan, dtype=np.float64)
        vs_all = np.full((L, n_tok, k), np.nan, dtype=np.float64)

        for t in range(L):
            R_a = R_all[t]
            for ii, ti in enumerate(tids):
                R_i = flows[ti]["R"][t]
                vs_all[t, ii] = _principal_angles_topk(R_i, R_a, k)
                for jj in range(ii + 1, n_tok):
                    R_j = flows[tids[jj]]["R"][t]
                    ang = _principal_angles_topk(R_i, R_j, k)
                    pair_mat[t, ii, jj] = ang
                    pair_mat[t, jj, ii] = ang
                pair_mat[t, ii, ii] = 0.0

        # Summary: mean over pairs (excluding diagonal) and mean over tokens
        # for vs-all, both as a (L, k) profile.
        with np.errstate(invalid="ignore"):
            tril = np.tril_indices(n_tok, k=-1)
            mean_pair = np.array([
                np.nanmean(pair_mat[t][tril], axis=0) for t in range(L)
            ])     # (L, k)
            mean_vs_all = np.nanmean(vs_all, axis=1)  # (L, k)

        np.savez(
            out_path,
            seed=np.int32(seed), step=np.int64(step), k=np.int32(k),
            tids=np.array(tids, dtype=np.int32),
            pair_angle_matrix=pair_mat,
            angles_vs_all=vs_all,
            mean_pair_angles=mean_pair,
            mean_vs_all_angles=mean_vs_all,
        )
        results.append({
            "seed": seed, "step": step, "tids": np.array(tids),
            "mean_pair_angles": mean_pair, "mean_vs_all_angles": mean_vs_all,
            "pair_angle_matrix": pair_mat, "k": k,
        })
        if verbose:
            print(f"[D2] seed {seed} step {step}: -> {out_path}")
    return results


# ----------------------------------------------------------------------
# D3: per-token (alpha, lambda) fits + universality check.
# ----------------------------------------------------------------------
def run_d3(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    verbose: bool = True,
) -> Dict:
    """For every (seed, step), collect per-token log_alpha_i and lambda_i
    from the saved flow dicts, plus the all-to-all values.

    Output array shape per quantity: (n_seeds, n_steps, max_n_tokens).
    Tokens are aligned to the frozen forward set; missing tokens are
    NaN-padded. Also computes refit per-token alpha/lambda from the
    saved singular values for cross-checking the on-file values.
    """
    if verbose:
        print(f"[D3] Per-token alpha/lambda fits ...")

    # Discover the maximal token-set size and a canonical token order
    # from the frozen forward set.
    forward_set, _, _ = load_token_sets(run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    n_tok = tids.size
    if n_tok == 0:
        if verbose:
            print(f"[D3] No frozen forward set; skipping.")
        return {}

    log_alpha = np.full((len(seeds), len(common_steps), n_tok), np.nan)
    lam = np.full_like(log_alpha, np.nan)
    log_alpha_all = np.full((len(seeds), len(common_steps)), np.nan)
    lam_all = np.full_like(log_alpha_all, np.nan)
    refit_log_alpha = np.full_like(log_alpha, np.nan)
    refit_lam = np.full_like(log_alpha, np.nan)

    for si, seed in enumerate(seeds):
        for ti, step in enumerate(common_steps):
            d = mvr_dir(run_dir, seed, step)
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            try:
                r = load_multi_view_result(
                    d, skip_arrays={"R", "pairwise_residual_variance"})
            except Exception:
                continue
            log_alpha_all[si, ti] = r.all_to_all.get("log_alpha", float("nan"))
            lam_all[si, ti] = r.all_to_all.get("lambda", float("nan"))

            flows = r.forward_flows
            for k, tok in enumerate(tids):
                f = flows.get(int(tok))
                if f is None or f.get("failed", False):
                    continue
                log_alpha[si, ti, k] = float(f.get("log_alpha", float("nan")))
                lam[si, ti, k] = float(f.get("lambda", float("nan")))
                # Refit trace-based alpha/lambda as a cross-check: this
                # uses the per-token total-variance growth law, which is
                # what Model A's universality predicts most directly.
                v = _trace_within_per_token(f)
                layers = np.arange(v.size, dtype=np.float64)
                la, lm = _safe_logspace_fit(layers, v)
                refit_log_alpha[si, ti, k] = la
                refit_lam[si, ti, k] = lm

    out_path = os.path.join(output_root(run_dir), "d3_per_token_fits.npz")
    np.savez(
        out_path,
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(common_steps, dtype=np.int64),
        tids=tids,
        log_alpha_per_token=log_alpha,
        lambda_per_token=lam,
        log_alpha_all=log_alpha_all,
        lambda_all=lam_all,
        refit_log_alpha_per_token=refit_log_alpha,
        refit_lambda_per_token=refit_lam,
    )
    if verbose:
        print(f"[D3] -> {out_path}")
    return {
        "tids": tids,
        "log_alpha_per_token": log_alpha,
        "lambda_per_token": lam,
        "log_alpha_all": log_alpha_all,
        "lambda_all": lam_all,
        "refit_log_alpha_per_token": refit_log_alpha,
        "refit_lambda_per_token": refit_lam,
    }


# ----------------------------------------------------------------------
# D4a: kurtosis from saved flows.
# ----------------------------------------------------------------------
def run_d4a(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    verbose: bool = True,
) -> Dict:
    """Per-token kurtosis_per_layer averaged across tokens, with cross-
    token std as the spread. Saves a single npz."""
    if verbose:
        print(f"[D4a] Kurtosis profiles per token ...")
    forward_set, _, _ = load_token_sets(run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    n_tok = tids.size

    # Probe L.
    probe = load_multi_view_result(
        mvr_dir(run_dir, seeds[0], common_steps[-1]),
        skip_arrays={"R", "pairwise_residual_variance"})
    L = probe.all_to_all["kurtosis_per_layer"].shape[0]

    kurt = np.full((len(seeds), len(common_steps), n_tok, L), np.nan)
    kurt_all = np.full((len(seeds), len(common_steps), L), np.nan)
    for si, seed in enumerate(seeds):
        for ti, step in enumerate(common_steps):
            d = mvr_dir(run_dir, seed, step)
            if not os.path.exists(os.path.join(d, "meta.json")):
                continue
            try:
                r = load_multi_view_result(
                    d, skip_arrays={"R", "pairwise_residual_variance"})
            except Exception:
                continue
            kurt_all[si, ti] = r.all_to_all.get(
                "kurtosis_per_layer", np.full(L, np.nan))
            for k, tok in enumerate(tids):
                f = r.forward_flows.get(int(tok))
                if f is None or f.get("failed", False):
                    continue
                kurt[si, ti, k] = f["kurtosis_per_layer"].astype(np.float64)

    out_path = os.path.join(output_root(run_dir), "d4a_kurtosis.npz")
    np.savez(out_path,
             seeds=np.array(seeds, dtype=np.int32),
             steps=np.array(common_steps, dtype=np.int64),
             tids=tids,
             kurtosis_per_token=kurt,
             kurtosis_all=kurt_all)
    if verbose:
        print(f"[D4a] -> {out_path}")
    return {"tids": tids, "kurtosis_per_token": kurt, "kurtosis_all": kurt_all}


# ----------------------------------------------------------------------
# D4b: multivariate Gaussianity on within-input bundles.
# Requires the augmented activation file.
# ----------------------------------------------------------------------
def _mardia_kurtosis(X: np.ndarray, max_dim: int = 64) -> float:
    """Mardia's multivariate kurtosis on (n, d) data. For Gaussian data
    in d dimensions, expected value is d*(d+2). Returns the standardized
    excess: (b - d*(d+2)) / sqrt(8*d*(d+2)/n), which is asymptotically
    N(0, 1) under the Gaussian null.

    To keep this stable at high d / low n, project onto the top max_dim
    PCs first if d > max_dim. The diagnostic value is on Gaussianity in
    the principal subspace, which is the relevant subspace here anyway.
    """
    n, d = X.shape
    if n <= d + 2:
        return float("nan")
    if d > max_dim:
        Xc = X - X.mean(0, keepdims=True)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        X = Xc @ Vt[:max_dim].T
        d = max_dim
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / n
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    M = Xc @ cov_inv @ Xc.T
    diag = np.diag(M)                 # squared Mahalanobis distances
    b = (diag ** 2).mean()            # Mardia b_{2,d}
    expected = d * (d + 2)
    se = np.sqrt(8.0 * d * (d + 2) / n)
    return float((b - expected) / se)


def run_d4b(
    run_dir: str,
    steps_to_analyze: List[Tuple[int, int]],
    max_pc_dim: int = 32,
    verbose: bool = True,
) -> List[Dict]:
    """Multivariate Gaussianity per (seed, step) and per token, on the
    within-input bundle projected to its top-max_pc_dim PCs.

    Reports: Mardia kurtosis Z (asymptotic N(0,1) under Gaussian);
    per-PC kurtosis (should be ~0 if Gaussian); per-PC Anderson-Darling
    statistic against normal (lighter weight than full multivariate
    test).
    """
    if verbose:
        print(f"[D4b] Multivariate Gaussianity on {len(steps_to_analyze)} "
              f"checkpoints ...")
    from scipy import stats as scistats

    forward_set, _, _ = load_token_sets(run_dir)
    tids = forward_set.token_ids.astype(np.int32)

    results = []
    for seed, step in steps_to_analyze:
        out_path = os.path.join(output_root(run_dir), "d4b_gaussianity",
                                f"seed{seed}_step{step:08d}.npz")
        if os.path.exists(out_path):
            with np.load(out_path) as f:
                results.append({k: f[k] for k in f.files})
            continue

        aug_path = augmented_path(run_dir, seed, step)
        if not os.path.exists(aug_path):
            if verbose:
                print(f"[D4b] seed {seed} step {step}: missing augmented file, "
                      f"skip")
            continue
        if verbose:
            print(f"[D4b] seed {seed} step {step}: loading activations ...")
        aug = load_augmented_activations(aug_path)
        states = aug["states"]              # (L, N, H)
        input_ids = aug["input_ids"]        # (N,)
        L = states.shape[0]

        mardia_z = np.full((tids.size, L), np.nan)
        # Per-PC marginal kurtosis (excess) on top max_pc_dim PCs.
        per_pc_kurt = np.full((tids.size, L, max_pc_dim), np.nan)
        per_pc_ad = np.full((tids.size, L, max_pc_dim), np.nan)
        n_pilots = np.zeros(tids.size, dtype=np.int64)

        for k, tok in enumerate(tids):
            mask = (input_ids == int(tok))
            n = int(mask.sum())
            n_pilots[k] = n
            if n < max_pc_dim + 2:
                continue
            for t in range(L):
                X = states[t, mask].astype(np.float64)          # (n, H)
                Xc = X - X.mean(0, keepdims=True)
                # Top-max_pc_dim PCA projection.
                # H is large (~768); do PCA via thin SVD on Xc.
                try:
                    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                Z = Xc @ Vt[:max_pc_dim].T                       # (n, m)
                mardia_z[k, t] = _mardia_kurtosis(Z, max_dim=max_pc_dim)
                # Per-PC kurtosis (excess) and AD.
                for j in range(min(max_pc_dim, Z.shape[1])):
                    zj = Z[:, j]
                    # Excess kurtosis: E[(z-mu)^4]/var^2 - 3
                    m4 = ((zj - zj.mean()) ** 4).mean()
                    v2 = zj.var() ** 2
                    per_pc_kurt[k, t, j] = (
                        m4 / v2 - 3.0 if v2 > 0 else float("nan")
                    )
                    try:
                        ad = scistats.anderson(zj, dist="norm")
                        per_pc_ad[k, t, j] = float(ad.statistic)
                    except Exception:
                        per_pc_ad[k, t, j] = float("nan")

        np.savez(
            out_path,
            seed=np.int32(seed), step=np.int64(step),
            tids=tids, n_pilots=n_pilots,
            mardia_z=mardia_z,
            per_pc_kurtosis=per_pc_kurt,
            per_pc_anderson_darling=per_pc_ad,
        )
        results.append({
            "seed": seed, "step": step, "tids": tids,
            "n_pilots": n_pilots,
            "mardia_z": mardia_z,
            "per_pc_kurtosis": per_pc_kurt,
            "per_pc_anderson_darling": per_pc_ad,
        })
        if verbose:
            print(f"[D4b] seed {seed} step {step}: -> {out_path}")
    return results


# ----------------------------------------------------------------------
# D5: GMM reconstruction.
# ----------------------------------------------------------------------
def run_d5(
    run_dir: str,
    steps_to_analyze: List[Tuple[int, int]],
    max_pc_dim: int = 32,
    n_samples: int = 5000,
    verbose: bool = True,
) -> List[Dict]:
    """Synthesize samples from the per-token Gaussian fits, mix at
    empirical frequencies, and compare moments + marginal Gaussianity
    against the directly observed all-to-all bundle.

    Three reconstructions:
      Model A reconstruction:  shared Sigma_0(t) = mean of Sigma_i(t)
                               under uniform token weighting, with the
                               observed mu_i(t).
      Model B reconstruction:  per-token Sigma_i(t), frequency-weighted.
      Empirical baseline:      the actual all-to-all bundle.

    Reports per layer:
      mean(empirical) - mean(model-B reconstruction) -- should be ~0
      cov spectrum overlap (cosine of top-k singular value vectors)
      kurtosis of the model-B mixture sample, model-A mixture sample,
        and the empirical sample.

    The kurtosis comparison is the discriminator: if A and empirical
    agree, Model A holds; if B agrees but A doesn't, Model B holds; if
    neither agrees, Model C is plausible.
    """
    if verbose:
        print(f"[D5] GMM reconstruction on {len(steps_to_analyze)} checkpoints ...")
    rng = np.random.default_rng(20260521)

    forward_set, _, _ = load_token_sets(run_dir)
    tids = forward_set.token_ids.astype(np.int32)

    results = []
    for seed, step in steps_to_analyze:
        out_path = os.path.join(output_root(run_dir), "d5_reconstruction",
                                f"seed{seed}_step{step:08d}.npz")
        if os.path.exists(out_path):
            with np.load(out_path) as f:
                results.append({k: f[k] for k in f.files})
            continue

        aug_path = augmented_path(run_dir, seed, step)
        if not os.path.exists(aug_path):
            if verbose:
                print(f"[D5] seed {seed} step {step}: missing augmented file, skip")
            continue

        aug = load_augmented_activations(aug_path)
        states = aug["states"]
        input_ids = aug["input_ids"]
        L, N, H = states.shape

        # Per-token empirical mean and covariance in top-max_pc_dim PC
        # space at each layer. We use a *layer-wise common* PCA basis
        # (the all-to-all basis at that layer), so the per-token Gaussians
        # live in the same coordinate system as the empirical bundle.
        d = max_pc_dim
        global_means = np.zeros((L, H), dtype=np.float64)
        global_basis = np.zeros((L, d, H), dtype=np.float64)
        # Empirical kurtosis (mean of per-coordinate excess) per layer.
        emp_kurt = np.full(L, np.nan)
        # Per-layer per-token contributions (Model B).
        empirical_freqs = np.zeros(tids.size, dtype=np.float64)
        for k, tok in enumerate(tids):
            empirical_freqs[k] = float((input_ids == int(tok)).sum())
        empirical_freqs /= max(empirical_freqs.sum(), 1.0)

        # Allocate mixture sample buffers.
        recon_A_kurt = np.full(L, np.nan)
        recon_B_kurt = np.full(L, np.nan)
        mean_err_A = np.full(L, np.nan)
        mean_err_B = np.full(L, np.nan)
        cov_trace_err_A = np.full(L, np.nan)
        cov_trace_err_B = np.full(L, np.nan)

        for t in range(L):
            X = states[t].astype(np.float64)                 # (N, H)
            mu_global = X.mean(0)
            Xc = X - mu_global
            try:
                U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            V = Vt[:d]                                       # (d, H)
            global_means[t] = mu_global
            global_basis[t] = V
            Z_emp = Xc @ V.T                                 # (N, d)
            emp_kurt[t] = float(np.mean(
                ((Z_emp - Z_emp.mean(0)) ** 4).mean(0)
                / (Z_emp.var(0) ** 2 + 1e-30) - 3.0
            ))

            # Per-token mu, cov in the *global* layer basis.
            mu_i = []
            cov_i = []
            mu_count = []
            for k, tok in enumerate(tids):
                mask = (input_ids == int(tok))
                n = int(mask.sum())
                if n < d + 2:
                    mu_i.append(None); cov_i.append(None); mu_count.append(0)
                    continue
                Zi = (states[t, mask].astype(np.float64) - mu_global) @ V.T
                mu_i.append(Zi.mean(0))
                cov_i.append(np.cov(Zi.T, ddof=1))
                mu_count.append(n)

            # Frequency-weighted mean (Model B reconstruction).
            valid = [k for k in range(tids.size) if mu_i[k] is not None]
            if not valid:
                continue
            w = np.array([mu_count[k] for k in valid], dtype=np.float64)
            w = w / w.sum()
            mu_B = sum(w[i] * mu_i[valid[i]] for i in range(len(valid)))
            mean_err_B[t] = float(np.linalg.norm(mu_B))  # should be ~0
            # Shared Sigma for Model A: average of per-token covs.
            Sigma_0 = sum(w[i] * cov_i[valid[i]] for i in range(len(valid)))

            # Empirical covariance in PC basis (diagonal of singular values^2/n).
            Sigma_emp = np.cov(Z_emp.T, ddof=1)
            cov_trace_err_A[t] = abs(np.trace(Sigma_0) - np.trace(Sigma_emp)) / \
                                  max(abs(np.trace(Sigma_emp)), 1e-30)
            # Model B trace: within + between
            Sigma_B_within = sum(w[i] * cov_i[valid[i]] for i in range(len(valid)))
            mus = np.stack([mu_i[valid[i]] - mu_B for i in range(len(valid))])
            Sigma_B_between = (w[:, None] * mus).T @ mus
            Sigma_B = Sigma_B_within + Sigma_B_between
            cov_trace_err_B[t] = abs(np.trace(Sigma_B) - np.trace(Sigma_emp)) / \
                                  max(abs(np.trace(Sigma_emp)), 1e-30)

            # Sample from Model A and Model B mixtures.
            n_per = n_samples
            choices = rng.choice(len(valid), size=n_per, p=w)
            samples_A = np.zeros((n_per, d))
            samples_B = np.zeros((n_per, d))
            # Cholesky once per token for efficiency.
            try:
                L_A = np.linalg.cholesky(Sigma_0 + 1e-9 * np.eye(d))
            except np.linalg.LinAlgError:
                L_A = None
            chol_B = {}
            for vi, ki in enumerate(valid):
                try:
                    chol_B[ki] = np.linalg.cholesky(
                        cov_i[ki] + 1e-9 * np.eye(d))
                except np.linalg.LinAlgError:
                    chol_B[ki] = None

            for r_idx in range(n_per):
                ki = valid[choices[r_idx]]
                z = rng.standard_normal(d)
                if L_A is not None:
                    samples_A[r_idx] = mu_i[ki] + L_A @ z
                if chol_B[ki] is not None:
                    samples_B[r_idx] = mu_i[ki] + chol_B[ki] @ z

            def _per_coord_excess_kurt(arr):
                with np.errstate(invalid="ignore"):
                    return float(np.mean(
                        ((arr - arr.mean(0)) ** 4).mean(0) /
                        (arr.var(0) ** 2 + 1e-30) - 3.0
                    ))

            recon_A_kurt[t] = _per_coord_excess_kurt(samples_A)
            recon_B_kurt[t] = _per_coord_excess_kurt(samples_B)

        np.savez(
            out_path,
            seed=np.int32(seed), step=np.int64(step),
            empirical_kurt=emp_kurt,
            recon_A_kurt=recon_A_kurt,
            recon_B_kurt=recon_B_kurt,
            mean_err_A=mean_err_A,
            mean_err_B=mean_err_B,
            cov_trace_err_A=cov_trace_err_A,
            cov_trace_err_B=cov_trace_err_B,
        )
        results.append({
            "seed": seed, "step": step,
            "empirical_kurt": emp_kurt,
            "recon_A_kurt": recon_A_kurt,
            "recon_B_kurt": recon_B_kurt,
            "mean_err_A": mean_err_A,
            "mean_err_B": mean_err_B,
            "cov_trace_err_A": cov_trace_err_A,
            "cov_trace_err_B": cov_trace_err_B,
        })
        if verbose:
            print(f"[D5] seed {seed} step {step}: -> {out_path}")
    return results


# ----------------------------------------------------------------------
# Verdict logic.
# ----------------------------------------------------------------------
@dataclass
class LayerVerdict:
    layer: int
    d1_cv_trace: float           # cross-token CV of trace(Sigma_i)
    d3_cv_lambda: float          # cross-token CV of lambda_i
    d2_mean_pair_angle_deg: float
    d4a_max_kurt: float          # max per-token kurtosis (Gaussianity proxy)
    verdict: str                 # "A", "A_partial", "B", "C", "unclear"


def derive_layer_verdicts(
    seed: int, step: int,
    d1: Dict, d3: Dict, d2: Optional[Dict], d4a: Dict,
    seeds: List[int], steps: List[int],
    cv_trace_threshold: float = 0.15,
    cv_lambda_threshold: float = 0.20,
    pair_angle_threshold_deg: float = 25.0,
    kurt_threshold: float = 1.0,
) -> List[LayerVerdict]:
    """Combine discriminators into a per-layer verdict.

    Decision tree:
      if D4a |kurt| large    -> "C"
      elif D1 low AND D3 low -> "A"   (and D2 low if available is consistency)
      elif D1 low AND D3 high -> "A_partial"  (shape uniform, scale varies)
      elif D1 high           -> "B"
      else                   -> "unclear"

    Thresholds are conservative defaults; expose them to the CLI.
    """
    si = seeds.index(seed)
    ti = steps.index(step)
    L = d1["cv_trace"].shape[2]
    verdicts = []
    for t in range(L):
        cv_t = d1["cv_trace"][si, ti, t]
        # CV of lambda_i across tokens at this (seed, step).
        lam_per_tok = d3["lambda_per_token"][si, ti]    # (n_tok,)
        cv_lam = _coefficient_of_variation(lam_per_tok)
        # Kurtosis: max over tokens at this layer.
        kurt_per_tok = d4a["kurtosis_per_token"][si, ti, :, t]
        max_kurt = float(np.nanmax(np.abs(kurt_per_tok))) if (
            np.isfinite(kurt_per_tok).any()) else float("nan")
        # Mean pair angle at this layer if D2 was run.
        if d2 is not None:
            angle_rad = float(np.nanmean(d2["mean_pair_angles"][t]))
            angle_deg = np.degrees(angle_rad)
        else:
            angle_deg = float("nan")

        # Decision.
        if np.isfinite(max_kurt) and max_kurt > kurt_threshold * 3:
            v = "C"
        elif np.isfinite(cv_t) and cv_t < cv_trace_threshold:
            if np.isfinite(cv_lam) and cv_lam < cv_lambda_threshold:
                v = "A"
            else:
                v = "A_partial"
        elif np.isfinite(cv_t) and cv_t > cv_trace_threshold * 2:
            v = "B"
        else:
            v = "unclear"

        verdicts.append(LayerVerdict(
            layer=t, d1_cv_trace=float(cv_t),
            d3_cv_lambda=float(cv_lam),
            d2_mean_pair_angle_deg=float(angle_deg),
            d4a_max_kurt=float(max_kurt),
            verdict=v,
        ))
    return verdicts


def write_verdict_csv(
    run_dir: str,
    seeds: List[int], steps: List[int],
    d1: Dict, d3: Dict, d2_by_key: Dict[Tuple[int, int], Dict],
    d4a: Dict,
    thresholds: Dict[str, float],
) -> None:
    out_path = os.path.join(output_root(run_dir), "verdict.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed", "step", "layer", "verdict",
            "d1_cv_trace", "d3_cv_lambda",
            "d2_mean_pair_angle_deg", "d4a_max_abs_kurt",
        ])
        for seed in seeds:
            for step in steps:
                d2_here = d2_by_key.get((seed, step))
                verdicts = derive_layer_verdicts(
                    seed=seed, step=step,
                    d1=d1, d3=d3, d2=d2_here, d4a=d4a,
                    seeds=seeds, steps=steps,
                    **thresholds,
                )
                for v in verdicts:
                    w.writerow([
                        seed, step, v.layer, v.verdict,
                        f"{v.d1_cv_trace:.6f}",
                        f"{v.d3_cv_lambda:.6f}",
                        f"{v.d2_mean_pair_angle_deg:.4f}",
                        f"{v.d4a_max_kurt:.6f}",
                    ])
    print(f"[verdict] -> {out_path}")


def write_verdict_summary(
    run_dir: str,
    seeds: List[int], steps: List[int],
    d1: Dict, d3: Dict, d4a: Dict,
    d2_by_key: Dict[Tuple[int, int], Dict],
    d5_results: List[Dict],
    thresholds: Dict[str, float],
) -> None:
    """Write a human-readable summary that picks the final checkpoint
    and summarizes the per-layer verdict pattern."""
    out_path = os.path.join(output_root(run_dir), "verdict.txt")
    final_step = steps[-1]
    with open(out_path, "w") as f:
        f.write("Model A/B/C discriminator verdict\n")
        f.write("==================================\n\n")
        f.write(f"run_dir: {run_dir}\n")
        f.write(f"seeds:   {seeds}\n")
        f.write(f"steps:   {len(steps)} checkpoints, final = {final_step}\n\n")
        f.write("Per-seed per-layer verdict at the final checkpoint:\n\n")
        for seed in seeds:
            d2_here = d2_by_key.get((seed, final_step))
            verdicts = derive_layer_verdicts(
                seed=seed, step=final_step,
                d1=d1, d3=d3, d2=d2_here, d4a=d4a,
                seeds=seeds, steps=steps, **thresholds,
            )
            counts: Dict[str, int] = {}
            for v in verdicts:
                counts[v.verdict] = counts.get(v.verdict, 0) + 1
            f.write(f"  seed {seed}: ")
            f.write(" / ".join(
                f"{k}={c}" for k, c in sorted(counts.items(),
                                              key=lambda kv: -kv[1])))
            f.write("\n    layers: ")
            f.write(" ".join(v.verdict for v in verdicts))
            f.write("\n")

        f.write("\nD5 reconstruction (final checkpoint, seed 0 by default):\n\n")
        for r in d5_results:
            if r["step"] != final_step:
                continue
            f.write(f"  seed {int(r['seed'])}: per-layer kurtosis comparison\n")
            f.write("    layer  empirical  recon_A  recon_B\n")
            for t in range(len(r["empirical_kurt"])):
                f.write(f"    {t:5d}  {r['empirical_kurt'][t]:>8.3f}  "
                        f"{r['recon_A_kurt'][t]:>7.3f}  "
                        f"{r['recon_B_kurt'][t]:>7.3f}\n")
            f.write("\n")

        f.write("\nReading:\n")
        f.write("  A         shared-covariance GMM picture is consistent\n")
        f.write("  A_partial shape uniform across tokens, scale (lambda) varies\n")
        f.write("  B         conditional covariances vary across tokens\n")
        f.write("  C         conditional distributions are not Gaussian\n")
        f.write("  unclear   sample-size-limited or marginal; expand pilots\n")
    print(f"[verdict] -> {out_path}")


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_d1(run_dir: str, d1: Dict, steps: List[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cv_trace = d1["cv_trace"]            # (S, T, L)
    cv_erank = d1["cv_erank"]
    S, T, L = cv_trace.shape
    layers = np.arange(L)

    # Final-checkpoint per-seed CV profiles.
    ax = axes[0]
    for s in range(S):
        ax.plot(layers, cv_trace[s, -1], "-", alpha=0.85,
                label=f"seed {s}")
    ax.axhline(0.15, color="k", ls=":", lw=1,
               label="A/B boundary (CV=0.15)")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("CV(trace Sigma_i)")
    ax.set_title("D1: cross-token CV of within-input variance "
                 "(final ckpt)")
    ax.legend(fontsize=8, loc="best")

    # CV vs training step at a representative interior layer.
    ax = axes[1]
    mid = L // 2
    for s in range(S):
        ax.plot(steps, cv_trace[s, :, mid], "-", alpha=0.85,
                label=f"seed {s}")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"CV(trace Sigma_i) at layer {mid}")
    ax.set_title(f"D1: training trajectory at layer {mid}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d1_within_cv.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def plot_d2(run_dir: str, d2_results: List[Dict]) -> None:
    if not d2_results:
        return
    # Use the latest step from each (seed, step) result, but we usually
    # have one per seed if running default.
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in d2_results:
        mean_pair = r["mean_pair_angles"]    # (L, k)
        # Mean over k = top-k principal angles.
        mean_angle_deg = np.degrees(np.nanmean(mean_pair, axis=1))
        ax.plot(mean_angle_deg, "-", alpha=0.85,
                label=f"seed {r['seed']} step {r['step']} (pairwise)")
        mean_vs_all = r["mean_vs_all_angles"]
        mean_angle_va = np.degrees(np.nanmean(mean_vs_all, axis=1))
        ax.plot(mean_angle_va, "--", alpha=0.6,
                label=f"seed {r['seed']} (vs all-to-all)")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("mean top-k principal angle (deg)")
    ax.set_title("D2: principal angles between conditional bases")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d2_angle_matrix_final.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def plot_d3(run_dir: str, d3: Dict, steps: List[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    log_alpha = d3["log_alpha_per_token"]       # (S, T, K)
    lam = d3["lambda_per_token"]
    la_all = d3["log_alpha_all"]                # (S, T)
    lam_all = d3["lambda_all"]
    # Final checkpoint scatter across tokens.
    ax = axes[0]
    S, T, K = log_alpha.shape
    for s in range(S):
        ax.scatter(log_alpha[s, -1], lam[s, -1], alpha=0.6,
                   label=f"seed {s} per-token", s=18)
        ax.scatter([la_all[s, -1]], [lam_all[s, -1]],
                   marker="x", s=100, color="k",
                   label=f"seed {s} all-to-all" if s == 0 else None)
    ax.set_xlabel("log alpha (per token)")
    ax.set_ylabel("lambda (per token)")
    ax.set_title("D3: alpha/lambda scatter, final checkpoint")
    ax.legend(fontsize=7, loc="best")

    # CV(lambda) and CV(log_alpha) across training.
    ax = axes[1]
    cv_lam = np.array([
        [_coefficient_of_variation(lam[s, t]) for t in range(T)]
        for s in range(S)
    ])
    for s in range(S):
        ax.plot(steps, cv_lam[s], "-", label=f"seed {s} CV(lambda)")
    ax.axhline(0.20, color="k", ls=":", lw=1, label="A/B boundary (CV=0.20)")
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("CV across tokens")
    ax.set_title("D3: universality of lambda through training")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d3_alpha_lambda_scatter.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def plot_d4(run_dir: str, d4a: Dict, d4b_results: List[Dict],
            steps: List[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    kurt = d4a["kurtosis_per_token"]            # (S, T, K, L)
    kurt_all = d4a["kurtosis_all"]              # (S, T, L)

    ax = axes[0]
    S, T, K, L = kurt.shape
    layers = np.arange(L)
    # Per-seed mean and 5/95% band over tokens, final checkpoint.
    for s in range(S):
        per_t = kurt[s, -1]                     # (K, L)
        m = np.nanmean(per_t, axis=0)
        lo = np.nanpercentile(per_t, 5, axis=0)
        hi = np.nanpercentile(per_t, 95, axis=0)
        ax.plot(layers, m, "-", alpha=0.9, label=f"seed {s} mean")
        ax.fill_between(layers, lo, hi, alpha=0.15)
        ax.plot(layers, kurt_all[s, -1], "--", alpha=0.7,
                label=f"seed {s} all-to-all" if s == 0 else None)
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("D4a: kurtosis profile (final ckpt)")
    ax.legend(fontsize=8, loc="best")

    # D4b Mardia Z values at final checkpoint.
    ax = axes[1]
    if d4b_results:
        for r in d4b_results:
            mardia = np.array(r["mardia_z"])    # (K, L)
            mean_mardia = np.nanmean(mardia, axis=0)
            ax.plot(layers, mean_mardia, "-", alpha=0.9,
                    label=f"seed {int(r['seed'])} step {int(r['step'])}")
        ax.axhline(2.0, color="k", ls=":", lw=1,
                   label="|Z|=2 (Gaussian rejection at ~5%)")
        ax.axhline(-2.0, color="k", ls=":", lw=1)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("Mardia kurtosis Z (per-token mean)")
    ax.set_title("D4b: multivariate Gaussianity of conditional bundles")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d4_qq_panels.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


def plot_d5(run_dir: str, d5_results: List[Dict]) -> None:
    if not d5_results:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for r in d5_results:
        L = len(r["empirical_kurt"])
        layers = np.arange(L)
        ax.plot(layers, r["empirical_kurt"], "k-", lw=2,
                label=f"empirical (seed {int(r['seed'])} step {int(r['step'])})")
        ax.plot(layers, r["recon_A_kurt"], "C0--", lw=1.5,
                label="Model A reconstruction")
        ax.plot(layers, r["recon_B_kurt"], "C3-.", lw=1.5,
                label="Model B reconstruction")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis (top-d PCs)")
    ax.set_title("D5: GMM reconstruction — empirical vs Model A vs Model B")
    ax.legend(fontsize=8, loc="best")

    ax = axes[1]
    for r in d5_results:
        L = len(r["empirical_kurt"])
        layers = np.arange(L)
        ax.plot(layers, r["cov_trace_err_A"], "C0--", lw=1.5,
                label=f"|trace err| Model A (seed {int(r['seed'])})")
        ax.plot(layers, r["cov_trace_err_B"], "C3-.", lw=1.5,
                label=f"|trace err| Model B (seed {int(r['seed'])})")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("relative trace error")
    ax.set_title("D5: covariance-trace reconstruction error")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d5_reconstruction_diagnostic.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Dry-run reporter.
# ----------------------------------------------------------------------
def dry_run_report(
    run_dir: str,
    seeds: List[int],
    common: List[int],
    d2_steps: List[Tuple[int, int]],
    d4b_steps: List[Tuple[int, int]],
    d5_steps: List[Tuple[int, int]],
    discs: set,
) -> None:
    """Report what each discriminator would do without doing it.

    Checks that the input files exist on disk and that the expected
    output paths are writable. Reports per-stage what would run, what
    would be loaded, and what would be skipped (already done).
    """
    out_root = output_root(run_dir)
    print("\n" + "=" * 72)
    print("DRY RUN — nothing will be executed")
    print("=" * 72)

    # Discover token sets.
    print(f"\nToken sets file: ", end="")
    ts_path = os.path.join(out_root.rsplit("/model_abc", 1)[0],
                            "token_sets.json") \
        if "/model_abc" in out_root else \
        os.path.join(run_dir, "multiview", "token_sets.json")
    if os.path.exists(ts_path):
        try:
            forward_set, _, _ = load_token_sets(run_dir)
            print(f"OK ({forward_set.token_ids.size} forward tokens)")
        except Exception as e:
            print(f"present but unreadable: {e}")
    else:
        print(f"MISSING — D3, D4a, D4b, D5 will skip")

    # Sample availability of stage C results.
    print(f"\nStage C coverage (per-checkpoint multi-view results):")
    n_total = len(seeds) * len(common)
    n_present = 0
    n_with_R = 0
    for seed in seeds:
        for step in common:
            d = mvr_dir(run_dir, seed, step)
            if os.path.exists(os.path.join(d, "meta.json")):
                n_present += 1
                # Cheap probe of whether per-token R matrices are saved.
                flows_dir = os.path.join(d, "flows_forward")
                if os.path.isdir(flows_dir):
                    one = next((f for f in os.listdir(flows_dir)
                                if f.endswith(".npz")), None)
                    if one is not None:
                        with np.load(os.path.join(flows_dir, one)) as zf:
                            if "R" in zf.files:
                                n_with_R += 1
    print(f"  meta.json present: {n_present}/{n_total}")
    print(f"  per-token R matrices saved: {n_with_R}/{n_total}")
    if n_present < n_total:
        print(f"  -> {n_total - n_present} (seed, step) pairs would be "
              f"skipped by D1/D3/D4a")
    if n_with_R == 0:
        print(f"  -> WARNING: no R matrices on disk; D2 cannot run")

    # Sample availability of stage A augmented files.
    if "4b" in discs or "5" in discs:
        print(f"\nStage A coverage (augmented activation files):")
        needed = set()
        if "4b" in discs:
            needed |= set(d4b_steps)
        if "5" in discs:
            needed |= set(d5_steps)
        n_needed = len(needed)
        n_aug = sum(1 for (s, st) in needed
                    if os.path.exists(augmented_path(run_dir, s, st)))
        print(f"  augmented_step_*.npz present for required (seed, step): "
              f"{n_aug}/{n_needed}")
        # Probe file size for one of them.
        present = [(s, st) for (s, st) in needed
                   if os.path.exists(augmented_path(run_dir, s, st))]
        if present:
            p = augmented_path(run_dir, *present[0])
            sz_mb = os.path.getsize(p) / (1024 ** 2)
            print(f"  example file size: {sz_mb:.1f} MB ({p})")

    # Per-discriminator plan.
    print(f"\nDiscriminator plan:")
    if "1" in discs:
        out = os.path.join(out_root, "d1_token_cv.npz")
        status = "EXISTS, would overwrite" if os.path.exists(out) else "new"
        print(f"  [D1]  token-CV across {len(seeds)} seeds x "
              f"{len(common)} steps  -> {os.path.basename(out)} ({status})")
    if "2" in discs:
        d2_dir = os.path.join(out_root, "d2_principal_angles")
        n_done = sum(1 for (s, st) in d2_steps if os.path.exists(
            os.path.join(d2_dir, f"seed{s}_step{st:08d}.npz")))
        print(f"  [D2]  principal angles on {len(d2_steps)} checkpoints "
              f"(skip {n_done} already-done)")
        if d2_steps:
            print(f"        first: seed {d2_steps[0][0]} step "
                  f"{d2_steps[0][1]}")
    if "3" in discs:
        out = os.path.join(out_root, "d3_per_token_fits.npz")
        status = "EXISTS, would overwrite" if os.path.exists(out) else "new"
        print(f"  [D3]  per-token alpha/lambda  -> {os.path.basename(out)} "
              f"({status})")
    if "4a" in discs:
        out = os.path.join(out_root, "d4a_kurtosis.npz")
        status = "EXISTS, would overwrite" if os.path.exists(out) else "new"
        print(f"  [D4a] kurtosis profiles  -> {os.path.basename(out)} "
              f"({status})")
    if "4b" in discs:
        d4b_dir = os.path.join(out_root, "d4b_gaussianity")
        n_done = sum(1 for (s, st) in d4b_steps if os.path.exists(
            os.path.join(d4b_dir, f"seed{s}_step{st:08d}.npz")))
        print(f"  [D4b] multivariate Gaussianity on {len(d4b_steps)} "
              f"checkpoints (skip {n_done})")
        if d4b_steps:
            print(f"        first: seed {d4b_steps[0][0]} step "
                  f"{d4b_steps[0][1]}")
    if "5" in discs:
        d5_dir = os.path.join(out_root, "d5_reconstruction")
        n_done = sum(1 for (s, st) in d5_steps if os.path.exists(
            os.path.join(d5_dir, f"seed{s}_step{st:08d}.npz")))
        print(f"  [D5]  GMM reconstruction on {len(d5_steps)} checkpoints "
              f"(skip {n_done})")

    # Rough wall-time estimate.
    n_d1 = n_present if "1" in discs else 0
    n_d2 = len(d2_steps) if "2" in discs else 0
    n_d3 = n_present if "3" in discs else 0
    n_d4a = n_present if "4a" in discs else 0
    n_d4b = len(d4b_steps) if "4b" in discs else 0
    n_d5 = len(d5_steps) if "5" in discs else 0
    # Rough per-checkpoint estimates from the operations involved:
    #   D1, D3, D4a: ~3 s each (scalar pass over stage-C scalars/profiles)
    #   D2:          ~60 s   (loads R matrices, computes SVDs)
    #   D4b:         ~120 s  (loads augmented file, PCA + Mardia per token)
    #   D5:          ~180 s  (loads augmented file, fits + samples)
    est_min = (
        n_d1 * 3 + n_d2 * 60 + n_d3 * 3 + n_d4a * 3
        + n_d4b * 120 + n_d5 * 180
    ) / 60.0
    print(f"\nRough wall-time estimate: ~{est_min:.0f} min "
          f"(single CPU; D4b and D5 dominate)")

    print(f"\nOutput root: {out_root}")
    print(f"Figures:     {figures_dir(run_dir)}")
    print(f"\nTo execute, re-run without --dry-run.")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Model A/B/C discriminator suite for multiview campaign.")
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--discriminators", nargs="+",
                    default=["1", "2", "3", "4a", "4b", "5"],
                    help="Which discriminators to run.")
    ap.add_argument("--d2-all-steps", action="store_true",
                    help="Run D2 on every common (seed, step) instead of "
                         "just the final checkpoint per seed.")
    ap.add_argument("--d4b-all-seeds", action="store_true",
                    help="Run D4b on every seed at the final checkpoint.")
    ap.add_argument("--d4b-all-steps", action="store_true",
                    help="Run D4b at every checkpoint of seed 0.")
    ap.add_argument("--d5-all-seeds", action="store_true",
                    help="Run D5 on every seed at the final checkpoint.")
    ap.add_argument("--d5-all-steps", action="store_true",
                    help="Run D5 at every checkpoint of seed 0.")
    ap.add_argument("--top-k-subspace", type=int, default=10,
                    help="k for D2 principal-angle analysis.")
    ap.add_argument("--max-pc-dim", type=int, default=32,
                    help="PC subspace dim for D4b and D5 multivariate tests.")
    ap.add_argument("--cv-trace-threshold", type=float, default=0.15)
    ap.add_argument("--cv-lambda-threshold", type=float, default=0.20)
    ap.add_argument("--pair-angle-threshold-deg", type=float, default=25.0)
    ap.add_argument("--kurt-threshold", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't run anything; report what would be done, "
                         "what's present on disk, and what would be skipped.")
    args = ap.parse_args()

    ensure_dirs(args.run_dir)

    seeds = seeds_in_run(args.run_dir)
    if not seeds:
        print(f"No seeds found in {args.run_dir}")
        sys.exit(1)
    step_lists = {s: [step for step, _ in checkpoints_in_seed(args.run_dir, s)]
                  for s in seeds}
    common = sorted(set.intersection(*[set(v) for v in step_lists.values()]))
    if not common:
        print("No common steps across seeds")
        sys.exit(1)
    print(f"Run dir: {args.run_dir}")
    print(f"Seeds:   {seeds}")
    print(f"Common steps: {len(common)} (range {common[0]}..{common[-1]})")

    # Decide which (seed, step) pairs get the expensive treatments.
    final_step = common[-1]
    if args.d2_all_steps:
        d2_steps = [(s, st) for s in seeds for st in common]
    else:
        d2_steps = [(s, final_step) for s in seeds]

    if args.d4b_all_steps:
        d4b_steps = [(0, st) for st in common]
    elif args.d4b_all_seeds:
        d4b_steps = [(s, final_step) for s in seeds]
    else:
        d4b_steps = [(seeds[0], final_step)]

    if args.d5_all_steps:
        d5_steps = [(0, st) for st in common]
    elif args.d5_all_seeds:
        d5_steps = [(s, final_step) for s in seeds]
    else:
        d5_steps = [(seeds[0], final_step)]

    discs = set(args.discriminators)

    # Dry run: report and exit without executing anything.
    if args.dry_run:
        dry_run_report(args.run_dir, seeds, common,
                       d2_steps, d4b_steps, d5_steps, discs)
        return

    # Run discriminators.
    t0 = time.time()
    d1 = d3 = d4a = None
    d2_results: List[Dict] = []
    d4b_results: List[Dict] = []
    d5_results: List[Dict] = []
    if "1" in discs:
        d1 = run_d1(args.run_dir, seeds, common)
    if "2" in discs:
        d2_results = run_d2(args.run_dir, seeds, d2_steps,
                            k=args.top_k_subspace)
    if "3" in discs:
        d3 = run_d3(args.run_dir, seeds, common)
    if "4a" in discs:
        d4a = run_d4a(args.run_dir, seeds, common)
    if "4b" in discs:
        d4b_results = run_d4b(args.run_dir, d4b_steps,
                              max_pc_dim=args.max_pc_dim)
    if "5" in discs:
        d5_results = run_d5(args.run_dir, d5_steps,
                            max_pc_dim=args.max_pc_dim)
    elapsed = time.time() - t0
    print(f"\nAll discriminators done in {elapsed/60:.1f} min")

    # Plots (only where we have data).
    if d1 is not None:
        plot_d1(args.run_dir, d1, common)
    if d2_results:
        plot_d2(args.run_dir, d2_results)
    if d3 is not None:
        plot_d3(args.run_dir, d3, common)
    if d4a is not None:
        plot_d4(args.run_dir, d4a, d4b_results, common)
    if d5_results:
        plot_d5(args.run_dir, d5_results)

    # Verdict CSV + text summary.
    d2_by_key = {(int(r["seed"]), int(r["step"])): r for r in d2_results}
    if d1 is not None and d3 is not None and d4a is not None:
        thresholds = {
            "cv_trace_threshold": args.cv_trace_threshold,
            "cv_lambda_threshold": args.cv_lambda_threshold,
            "pair_angle_threshold_deg": args.pair_angle_threshold_deg,
            "kurt_threshold": args.kurt_threshold,
        }
        write_verdict_csv(args.run_dir, seeds, common, d1, d3, d2_by_key, d4a,
                          thresholds)
        write_verdict_summary(args.run_dir, seeds, common, d1, d3, d4a,
                              d2_by_key, d5_results, thresholds)
    else:
        print("[verdict] Skipped (need D1, D3, D4a all enabled).")


if __name__ == "__main__":
    main()
