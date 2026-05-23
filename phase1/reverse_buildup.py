"""
Parameterized Model A/B/C discriminator for the reverse build-up project.

This module is a strict generalization of `model_abc_discriminator.py`.
The original suite hardcodes the forward (input-conditioned) view at
six points:

  - load_token_sets(run_dir)[0]   # forward set
  - r.forward_flows               # per-cell flows
  - aug["input_ids"]              # partition label

This module replaces those three hardcodes with a `view` argument that
selects between:

  view="forward"         -> forward_set,        forward_flows,        input_ids
  view="reverse_actual"  -> reverse_actual_set, reverse_actual_flows, next_ids
  view="reverse_pred"    -> reverse_pred_set,   reverse_pred_flows,   pred_ids

All five discriminators (D1, D3, D4a, D4b, D5) are computationally
view-agnostic; the math is the same. Only the input loaders and the
output paths differ. Output paths gain a view suffix:

  d1_token_cv_{view}.npz
  d3_per_token_fits_{view}.npz
  d4a_kurtosis_{view}.npz
  d4b_gaussianity_{view}/seed{S}_step{T:08d}.npz
  d5_reconstruction_{view}/seed{S}_step{T:08d}.npz

Bit-identical behavior on the forward view is a hard requirement
(verified by test_reverse_buildup.py::test_forward_parameterized_identity).

Two reverse-view-specific helpers live here that have no forward analog:

  contraction_fit:   per-cell lambda fit on the layers AFTER the
                     per-cell variance peak. Replaces the forward
                     log-linear fit, which is monotonic in depth.
  peak_variance_layer: t_w^* = argmax_t V_within-w(t). The cell-specific
                       reverse crossover layer.

These produce reverse-only summaries used downstream by
reverse_lambda_clusters.py (hypothesis N2).

The shuffle-null protocol (for hypothesis F4 risk mitigation R2) lives
in reverse_null.py; this module only provides the discriminators.

The unembedding-subspace decomposition (hypothesis N1) lives in
unembedding_subspace.py; this module has no awareness of it.

Usage from the campaign driver (reverse_buildup_campaign.py):

    from reverse_buildup import (
        run_d1_view, run_d3_view, run_d4a_view, run_d4b_view, run_d5_view,
        contraction_fit, peak_variance_layer,
    )
    run_d1_view(run_dir, seeds, steps, view="reverse_actual")
    ...
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Phase 1 / multiview infrastructure.
from multiview import (
    TokenSet,
    load_multi_view_result,
    load_augmented_activations,
)
from multiview_campaign import (
    mvr_dir,
    augmented_path,
    load_token_sets,
)


# ----------------------------------------------------------------------
# View dispatch.
# ----------------------------------------------------------------------
VIEWS = ("forward", "reverse_actual", "reverse_pred")


def select_token_set_for_view(run_dir: str, view: str) -> TokenSet:
    """Return the frozen token set for the requested view."""
    fwd, rev_act, rev_pred = load_token_sets(run_dir)
    if view == "forward":
        return fwd
    if view == "reverse_actual":
        return rev_act
    if view == "reverse_pred":
        return rev_pred
    raise ValueError(f"Unknown view: {view!r}; expected one of {VIEWS}")


def select_flows_for_view(mvr, view: str) -> Dict[int, Dict]:
    """Return the per-cell flow dict matching the requested view."""
    if view == "forward":
        return mvr.forward_flows
    if view == "reverse_actual":
        return mvr.reverse_actual_flows
    if view == "reverse_pred":
        return mvr.reverse_pred_flows
    raise ValueError(f"Unknown view: {view!r}; expected one of {VIEWS}")


def select_id_array_for_view(aug: Dict[str, np.ndarray], view: str) -> np.ndarray:
    """Return the (N,) partition-label id array matching the requested view."""
    if view == "forward":
        return aug["input_ids"]
    if view == "reverse_actual":
        return aug["next_ids"]
    if view == "reverse_pred":
        return aug["pred_ids"]
    raise ValueError(f"Unknown view: {view!r}; expected one of {VIEWS}")


# ----------------------------------------------------------------------
# Path conventions.
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    """Same output root the forward discriminator uses; we add view-suffixed
    files alongside the existing forward outputs to avoid a separate tree."""
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


def ensure_dirs(run_dir: str, view: str) -> None:
    os.makedirs(output_root(run_dir), exist_ok=True)
    os.makedirs(figures_dir(run_dir), exist_ok=True)
    for sub in (f"d4b_gaussianity_{view}", f"d5_reconstruction_{view}"):
        os.makedirs(os.path.join(output_root(run_dir), sub), exist_ok=True)


# ----------------------------------------------------------------------
# Helpers (copied verbatim from model_abc_discriminator so this module is
# import-side-effect-free w.r.t. the original).
# ----------------------------------------------------------------------
def _trace_within_per_cell(flow: Dict) -> np.ndarray:
    """Per-layer trace(Sigma_w(t)) from a per-cell flow's saved singular
    values: trace = sum_d s_d(t)^2 / n_pilots.

    Cell-agnostic name; for forward the 'cell' is an input token, for
    reverse it's a successor token. NaN-array if the flow failed.
    """
    if flow.get("failed", False):
        L = flow["singular_values"].shape[0]
        return np.full(L, np.nan, dtype=np.float64)
    s = flow["singular_values"].astype(np.float64)
    n = float(flow["n_pilots"])
    if n <= 1:
        return np.full(s.shape[0], np.nan)
    return (s ** 2).sum(axis=1) / n


def _safe_logspace_fit(layers: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Fit log v ~ log_alpha + lambda * log(layer index)."""
    mask = np.isfinite(v) & (v > 0) & (layers > 0)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    x = np.log(layers[mask].astype(np.float64))
    y = np.log(v[mask])
    A = np.vstack([np.ones_like(x), x]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(sol[0]), float(sol[1])


def _coefficient_of_variation(x: np.ndarray) -> float:
    """CV = std/mean of a 1-D array, ignoring NaN."""
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    m = x.mean()
    if abs(m) < 1e-30:
        return float("nan")
    return float(x.std(ddof=1) / abs(m))


# ----------------------------------------------------------------------
# Reverse-only fit helpers.
# ----------------------------------------------------------------------
def peak_variance_layer(v_within: np.ndarray) -> int:
    """Return t_w^* = argmax_t v_within(t).

    For forward conditionals v_within is monotonically increasing (or
    nearly so), so t* = L-1. For reverse conditionals v_within has a
    peak in the interior and contracts at late layers; t* is the layer
    where the bundle is most spread before commitment begins.

    Args:
        v_within: (L,) per-layer within-cell variance.

    Returns:
        Integer layer index in [0, L-1]. Layer 0 is excluded only when
        v_within[0] is exactly zero (the forward-view degenerate case);
        otherwise it can win the argmax.
    """
    v = np.asarray(v_within, dtype=np.float64)
    if not np.any(np.isfinite(v)):
        return -1
    # If layer 0 is exactly zero (forward boundary condition), exclude it
    # to avoid the argmax landing at a structurally trivial location for
    # reverse data that nominally has v[0] >= 0.
    vv = v.copy()
    if vv.size > 1 and vv[0] == 0.0:
        vv[0] = -np.inf
    return int(np.nanargmax(vv))


def contraction_fit(v_within: np.ndarray) -> Tuple[float, float, int]:
    """Fit log-linear lambda^contract on the contraction phase.

    The contraction phase is layers t in [t_w^*, L-1] inclusive, where
    t_w^* is the peak-variance layer. The fit shape is

        log v(t) = log_alpha^contract + lambda^contract * log(t)

    matching the forward convention, but applied only to the descending
    half. For monotonically growing v_within (forward case), t_w^* = L-1
    and the contraction fit is undefined; we return (nan, nan, t_w^*).

    Args:
        v_within: (L,) per-layer within-cell variance.

    Returns:
        (log_alpha_contract, lambda_contract, t_peak).
        lambda_contract is expected to be negative for reverse cells
        with a clean contraction phase.
    """
    v = np.asarray(v_within, dtype=np.float64)
    L = v.size
    t_peak = peak_variance_layer(v)
    if t_peak < 0 or t_peak >= L - 2:
        return float("nan"), float("nan"), int(t_peak)
    layers = np.arange(L, dtype=np.float64)
    la, lm = _safe_logspace_fit(layers[t_peak:], v[t_peak:])
    return la, lm, int(t_peak)


# ----------------------------------------------------------------------
# D1: cross-cell CV of trace and effective rank.
# ----------------------------------------------------------------------
def run_d1_view(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    view: str,
    verbose: bool = True,
) -> Dict:
    """Parameterized D1.

    For every (seed, step) and every layer, compute the cross-cell
    coefficient of variation of trace(Sigma_w(t)) and of effective rank.
    For reverse views this measures whether successor cells differ
    substantially in spread at each depth.

    Output: d1_token_cv_{view}.npz with arrays of shape
    (n_seeds, n_steps, L) for cv_trace, cv_erank, mean_trace, mean_erank.
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")

    if verbose:
        print(f"[D1/{view}] Token-CV across {len(seeds)} seeds x "
              f"{len(common_steps)} steps ...")

    # Probe shape from one available checkpoint.
    probe_seed, probe_step = seeds[0], common_steps[-1]
    probe = load_multi_view_result(
        mvr_dir(run_dir, probe_seed, probe_step),
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
                    print(f"[D1/{view}] seed {seed} step {step}: "
                          f"load failed: {e}")
                continue

            flows = select_flows_for_view(r, view)
            if not flows:
                continue
            tids = sorted(flows.keys())
            tr_mat = np.stack([_trace_within_per_cell(flows[t]) for t in tids],
                              axis=0)
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

    out_path = os.path.join(output_root(run_dir), f"d1_token_cv_{view}.npz")
    np.savez(
        out_path,
        view=np.array(view),
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(common_steps, dtype=np.int64),
        cv_trace=cv_trace,
        cv_erank=cv_erank,
        mean_trace=mean_trace,
        mean_erank=mean_erank,
    )
    if verbose:
        print(f"[D1/{view}] -> {out_path}")
    return {
        "view": view,
        "cv_trace": cv_trace,
        "cv_erank": cv_erank,
        "mean_trace": mean_trace,
        "mean_erank": mean_erank,
    }


# ----------------------------------------------------------------------
# D3: per-cell (alpha, lambda) fits with optional contraction fit.
# ----------------------------------------------------------------------
def run_d3_view(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    view: str,
    compute_contraction: bool = True,
    verbose: bool = True,
) -> Dict:
    """Parameterized D3.

    Collects per-cell log_alpha and lambda from the saved flow dicts.

    For reverse views, additionally fits the contraction-phase lambda
    (lambda_contract) on layers [t_w^*, L-1] and reports the per-cell
    peak-variance layer t_w^*.

    Output: d3_per_token_fits_{view}.npz with arrays:
      log_alpha_per_token     (n_seeds, n_steps, n_tok)   -- on-file fit
      lambda_per_token        (n_seeds, n_steps, n_tok)   -- on-file fit
      refit_log_alpha_per_token (n_seeds, n_steps, n_tok) -- refit from trace
      refit_lambda_per_token  (n_seeds, n_steps, n_tok)
      log_alpha_contract      (n_seeds, n_steps, n_tok)   -- contraction-only
      lambda_contract         (n_seeds, n_steps, n_tok)
      t_peak                  (n_seeds, n_steps, n_tok)   int
      log_alpha_all           (n_seeds, n_steps)
      lambda_all              (n_seeds, n_steps)
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")

    if verbose:
        print(f"[D3/{view}] Per-cell alpha/lambda fits ...")

    token_set = select_token_set_for_view(run_dir, view)
    tids = token_set.token_ids.astype(np.int32)
    n_tok = tids.size
    if n_tok == 0:
        if verbose:
            print(f"[D3/{view}] No frozen token set; skipping.")
        return {}

    log_alpha = np.full((len(seeds), len(common_steps), n_tok), np.nan)
    lam = np.full_like(log_alpha, np.nan)
    log_alpha_all = np.full((len(seeds), len(common_steps)), np.nan)
    lam_all = np.full_like(log_alpha_all, np.nan)
    refit_log_alpha = np.full_like(log_alpha, np.nan)
    refit_lam = np.full_like(log_alpha, np.nan)
    log_alpha_contract = np.full_like(log_alpha, np.nan)
    lam_contract = np.full_like(log_alpha, np.nan)
    t_peak = np.full((len(seeds), len(common_steps), n_tok), -1, dtype=np.int32)
    # Single-layer readout-step compression statistic. Captures the
    # one-step variance compression at the final residual block, which
    # the empirical data shows is the actual reverse-view contraction
    # mechanism (cf. proposal N2 reformulation after first results).
    #     lambda_readout_step[w] = log V_w(L-1) - log V_w(L-2)
    # Negative means the readout compresses the cell's spread; zero
    # means no compression; positive means continued growth.
    lam_readout_step = np.full_like(log_alpha, np.nan)

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

            flows = select_flows_for_view(r, view)
            for k, tok in enumerate(tids):
                f = flows.get(int(tok))
                if f is None or f.get("failed", False):
                    continue
                log_alpha[si, ti, k] = float(f.get("log_alpha", float("nan")))
                lam[si, ti, k] = float(f.get("lambda", float("nan")))
                v = _trace_within_per_cell(f)
                layers = np.arange(v.size, dtype=np.float64)
                la, lm = _safe_logspace_fit(layers, v)
                refit_log_alpha[si, ti, k] = la
                refit_lam[si, ti, k] = lm
                if compute_contraction:
                    la_c, lm_c, tp = contraction_fit(v)
                    log_alpha_contract[si, ti, k] = la_c
                    lam_contract[si, ti, k] = lm_c
                    t_peak[si, ti, k] = tp
                    # Readout-step compression: defined whenever the
                    # last two layers both have positive finite variance.
                    if v.size >= 2:
                        v_last = float(v[-1])
                        v_pen = float(v[-2])
                        if (np.isfinite(v_last) and np.isfinite(v_pen)
                                and v_last > 0 and v_pen > 0):
                            lam_readout_step[si, ti, k] = (
                                np.log(v_last) - np.log(v_pen)
                            )

    out_path = os.path.join(output_root(run_dir), f"d3_per_token_fits_{view}.npz")
    np.savez(
        out_path,
        view=np.array(view),
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(common_steps, dtype=np.int64),
        tids=tids,
        log_alpha_per_token=log_alpha,
        lambda_per_token=lam,
        refit_log_alpha_per_token=refit_log_alpha,
        refit_lambda_per_token=refit_lam,
        log_alpha_contract=log_alpha_contract,
        lambda_contract=lam_contract,
        lambda_readout_step=lam_readout_step,
        t_peak=t_peak,
        log_alpha_all=log_alpha_all,
        lambda_all=lam_all,
    )
    if verbose:
        print(f"[D3/{view}] -> {out_path}")
    return {
        "view": view,
        "tids": tids,
        "log_alpha_per_token": log_alpha,
        "lambda_per_token": lam,
        "refit_log_alpha_per_token": refit_log_alpha,
        "refit_lambda_per_token": refit_lam,
        "log_alpha_contract": log_alpha_contract,
        "lambda_contract": lam_contract,
        "lambda_readout_step": lam_readout_step,
        "t_peak": t_peak,
        "log_alpha_all": log_alpha_all,
        "lambda_all": lam_all,
    }


# ----------------------------------------------------------------------
# D4a: per-cell kurtosis from saved flows.
# ----------------------------------------------------------------------
def run_d4a_view(
    run_dir: str,
    seeds: List[int],
    common_steps: List[int],
    view: str,
    verbose: bool = True,
) -> Dict:
    """Parameterized D4a. Per-cell kurtosis_per_layer averaged across cells."""
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")

    if verbose:
        print(f"[D4a/{view}] Kurtosis profiles per cell ...")
    token_set = select_token_set_for_view(run_dir, view)
    tids = token_set.token_ids.astype(np.int32)
    n_tok = tids.size

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
            flows = select_flows_for_view(r, view)
            for k, tok in enumerate(tids):
                f = flows.get(int(tok))
                if f is None or f.get("failed", False):
                    continue
                kurt[si, ti, k] = f["kurtosis_per_layer"].astype(np.float64)

    out_path = os.path.join(output_root(run_dir), f"d4a_kurtosis_{view}.npz")
    np.savez(
        out_path,
        view=np.array(view),
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(common_steps, dtype=np.int64),
        tids=tids,
        kurtosis_per_token=kurt,
        kurtosis_all=kurt_all,
    )
    if verbose:
        print(f"[D4a/{view}] -> {out_path}")
    return {"view": view, "tids": tids,
            "kurtosis_per_token": kurt, "kurtosis_all": kurt_all}


# ----------------------------------------------------------------------
# D4b: Mardia multivariate kurtosis. Requires augmented activations.
# ----------------------------------------------------------------------
def _mardia_kurtosis_standardized(X: np.ndarray, max_dim: int = 64) -> float:
    """Mardia's b_{2,d} on (n, d) data, standardized to N(0,1) under
    the Gaussian null. Projects onto top-max_dim PCs if d > max_dim.

    Identical numerics to model_abc_discriminator._mardia_kurtosis; we
    duplicate to keep this module standalone.
    """
    n, d = X.shape
    if n <= d + 2:
        return float("nan")
    if d > max_dim:
        Xc = X - X.mean(0, keepdims=True)
        try:
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            return float("nan")
        X = Xc @ Vt[:max_dim].T
        d = max_dim
    Xc = X - X.mean(0, keepdims=True)
    cov = (Xc.T @ Xc) / n
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return float("nan")
    M = Xc @ cov_inv @ Xc.T
    diag = np.diag(M)
    b = (diag ** 2).mean()
    expected = d * (d + 2)
    se = np.sqrt(8.0 * d * (d + 2) / n)
    return float((b - expected) / se)


def run_d4b_view(
    run_dir: str,
    steps_to_analyze: List[Tuple[int, int]],
    view: str,
    max_pc_dim: int = 32,
    verbose: bool = True,
) -> List[Dict]:
    """Parameterized D4b.

    For each (seed, step) in steps_to_analyze, compute Mardia Z and
    per-PC kurtosis for every cell in the view's token set, in a
    PCA basis estimated from the cell's own subset.

    Saves one file per (seed, step) under d4b_gaussianity_{view}/.
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")
    if verbose:
        print(f"[D4b/{view}] Multivariate Gaussianity on "
              f"{len(steps_to_analyze)} checkpoints ...")
    try:
        from scipy import stats as scistats
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    ensure_dirs(run_dir, view)
    token_set = select_token_set_for_view(run_dir, view)
    tids = token_set.token_ids.astype(np.int32)

    results = []
    for seed, step in steps_to_analyze:
        out_path = os.path.join(output_root(run_dir),
                                f"d4b_gaussianity_{view}",
                                f"seed{seed}_step{step:08d}.npz")
        if os.path.exists(out_path):
            with np.load(out_path) as f:
                results.append({k: f[k] for k in f.files})
            continue

        aug_path = augmented_path(run_dir, seed, step)
        if not os.path.exists(aug_path):
            if verbose:
                print(f"[D4b/{view}] seed {seed} step {step}: "
                      f"missing augmented file, skip")
            continue
        if verbose:
            print(f"[D4b/{view}] seed {seed} step {step}: "
                  f"loading activations ...")
        aug = load_augmented_activations(aug_path)
        states = aug["states"]
        id_array = select_id_array_for_view(aug, view)
        L = states.shape[0]

        mardia_z = np.full((tids.size, L), np.nan)
        per_pc_kurt = np.full((tids.size, L, max_pc_dim), np.nan)
        per_pc_ad = np.full((tids.size, L, max_pc_dim), np.nan)
        n_pilots = np.zeros(tids.size, dtype=np.int64)

        for k, tok in enumerate(tids):
            mask = (id_array == int(tok))
            n = int(mask.sum())
            n_pilots[k] = n
            if n < max_pc_dim + 2:
                continue
            for t in range(L):
                X = states[t, mask].astype(np.float64)
                Xc = X - X.mean(0, keepdims=True)
                try:
                    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
                except np.linalg.LinAlgError:
                    continue
                Z = Xc @ Vt[:max_pc_dim].T
                mardia_z[k, t] = _mardia_kurtosis_standardized(
                    Z, max_dim=max_pc_dim)
                for j in range(min(max_pc_dim, Z.shape[1])):
                    zj = Z[:, j]
                    m4 = ((zj - zj.mean()) ** 4).mean()
                    v2 = zj.var() ** 2
                    per_pc_kurt[k, t, j] = (
                        m4 / v2 - 3.0 if v2 > 0 else float("nan")
                    )
                    if _have_scipy:
                        try:
                            ad = scistats.anderson(zj, dist="norm")
                            per_pc_ad[k, t, j] = float(ad.statistic)
                        except Exception:
                            per_pc_ad[k, t, j] = float("nan")

        np.savez(
            out_path,
            view=np.array(view),
            seed=np.int32(seed), step=np.int64(step),
            tids=tids, n_pilots=n_pilots,
            mardia_z=mardia_z,
            per_pc_kurtosis=per_pc_kurt,
            per_pc_anderson_darling=per_pc_ad,
        )
        results.append({
            "view": view,
            "seed": seed, "step": step, "tids": tids,
            "n_pilots": n_pilots,
            "mardia_z": mardia_z,
            "per_pc_kurtosis": per_pc_kurt,
            "per_pc_anderson_darling": per_pc_ad,
        })
        if verbose:
            print(f"[D4b/{view}] seed {seed} step {step}: -> {out_path}")
    return results


# ----------------------------------------------------------------------
# D5: GMM reconstruction. Requires augmented activations.
# ----------------------------------------------------------------------
def _per_coord_excess_kurt(arr: np.ndarray) -> float:
    """Mean across coordinates of per-coordinate excess kurtosis."""
    with np.errstate(invalid="ignore"):
        return float(np.mean(
            ((arr - arr.mean(0)) ** 4).mean(0) /
            (arr.var(0) ** 2 + 1e-30) - 3.0
        ))


def run_d5_view(
    run_dir: str,
    steps_to_analyze: List[Tuple[int, int]],
    view: str,
    max_pc_dim: int = 32,
    n_samples: int = 5000,
    id_array_override: Optional[np.ndarray] = None,
    output_suffix: str = "",
    verbose: bool = True,
) -> List[Dict]:
    """Parameterized D5.

    GMM reconstruction: per-cell Gaussian fit, mix at empirical
    frequencies, sample, compare to empirical marginal.

    The reverse-view interpretation is identical to forward — Model A
    (shared Sigma), Model B (per-cell Sigma) — but the cells are
    successor or predicted-successor tokens instead of input tokens.

    Args:
        run_dir, steps_to_analyze, view, max_pc_dim, n_samples: standard.

        id_array_override: if provided, use this (N,) array instead of
            the view's natural id array. Used by reverse_null.py to feed
            shuffled labels through the same D5 machinery. Length must
            equal the augmented file's pilot count.

        output_suffix: extra suffix appended to the output filename.
            Used by reverse_null.py to write the shuffled-null result
            alongside the real result without clobbering. Empty by
            default.

    Saves one file per (seed, step) under
    d5_reconstruction_{view}{output_suffix}/.
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")

    if verbose:
        suffix_msg = f" (suffix={output_suffix!r})" if output_suffix else ""
        print(f"[D5/{view}] GMM reconstruction on "
              f"{len(steps_to_analyze)} checkpoints{suffix_msg} ...")
    rng = np.random.default_rng(20260521)

    token_set = select_token_set_for_view(run_dir, view)
    tids = token_set.token_ids.astype(np.int32)
    if tids.size == 0:
        if verbose:
            print(f"[D5/{view}] No frozen token set; skipping.")
        return []

    out_subdir = f"d5_reconstruction_{view}{output_suffix}"
    os.makedirs(os.path.join(output_root(run_dir), out_subdir), exist_ok=True)

    results = []
    for seed, step in steps_to_analyze:
        out_path = os.path.join(output_root(run_dir), out_subdir,
                                f"seed{seed}_step{step:08d}.npz")
        if os.path.exists(out_path) and id_array_override is None:
            # We only reuse cached results for the real id array; null
            # runs always recompute because the override is non-stable.
            with np.load(out_path) as f:
                results.append({k: f[k] for k in f.files})
            continue

        aug_path = augmented_path(run_dir, seed, step)
        if not os.path.exists(aug_path):
            if verbose:
                print(f"[D5/{view}] seed {seed} step {step}: "
                      f"missing augmented file, skip")
            continue

        aug = load_augmented_activations(aug_path)
        states = aug["states"]
        if id_array_override is None:
            id_array = select_id_array_for_view(aug, view)
        else:
            if id_array_override.shape[0] != states.shape[1]:
                raise ValueError(
                    f"id_array_override has length {id_array_override.shape[0]} "
                    f"but augmented file has {states.shape[1]} pilots."
                )
            id_array = id_array_override
        L, N, H = states.shape

        d = max_pc_dim
        emp_kurt = np.full(L, np.nan)
        recon_A_kurt = np.full(L, np.nan)
        recon_B_kurt = np.full(L, np.nan)
        mean_err_A = np.full(L, np.nan)
        mean_err_B = np.full(L, np.nan)
        cov_trace_err_A = np.full(L, np.nan)
        cov_trace_err_B = np.full(L, np.nan)

        for t in range(L):
            X = states[t].astype(np.float64)
            mu_global = X.mean(0)
            Xc = X - mu_global
            try:
                U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            V = Vt[:d]
            Z_emp = Xc @ V.T
            emp_kurt[t] = _per_coord_excess_kurt(Z_emp)

            mu_i = []
            cov_i = []
            mu_count = []
            for k, tok in enumerate(tids):
                mask = (id_array == int(tok))
                n = int(mask.sum())
                if n < d + 2:
                    mu_i.append(None); cov_i.append(None); mu_count.append(0)
                    continue
                Zi = (states[t, mask].astype(np.float64) - mu_global) @ V.T
                mu_i.append(Zi.mean(0))
                cov_i.append(np.cov(Zi.T, ddof=1))
                mu_count.append(n)

            valid = [k for k in range(tids.size) if mu_i[k] is not None]
            if not valid:
                continue
            w = np.array([mu_count[k] for k in valid], dtype=np.float64)
            w = w / w.sum()
            mu_B = sum(w[i] * mu_i[valid[i]] for i in range(len(valid)))
            mean_err_B[t] = float(np.linalg.norm(mu_B))
            Sigma_0 = sum(w[i] * cov_i[valid[i]] for i in range(len(valid)))

            Sigma_emp = np.cov(Z_emp.T, ddof=1)
            cov_trace_err_A[t] = abs(np.trace(Sigma_0) - np.trace(Sigma_emp)) / \
                                  max(abs(np.trace(Sigma_emp)), 1e-30)
            Sigma_B_within = sum(w[i] * cov_i[valid[i]] for i in range(len(valid)))
            mus = np.stack([mu_i[valid[i]] - mu_B for i in range(len(valid))])
            Sigma_B_between = (w[:, None] * mus).T @ mus
            Sigma_B = Sigma_B_within + Sigma_B_between
            cov_trace_err_B[t] = abs(np.trace(Sigma_B) - np.trace(Sigma_emp)) / \
                                  max(abs(np.trace(Sigma_emp)), 1e-30)

            n_per = n_samples
            choices = rng.choice(len(valid), size=n_per, p=w)
            samples_A = np.zeros((n_per, d))
            samples_B = np.zeros((n_per, d))
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

            recon_A_kurt[t] = _per_coord_excess_kurt(samples_A)
            recon_B_kurt[t] = _per_coord_excess_kurt(samples_B)

        np.savez(
            out_path,
            view=np.array(view),
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
            "view": view,
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
            print(f"[D5/{view}] seed {seed} step {step}: -> {out_path}")
    return results
    