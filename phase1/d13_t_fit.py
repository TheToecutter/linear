"""
10.2 Multivariate t-distribution fit.

Tests Possibility 1: intrinsic heavy-tailed conditionals. If the
conditional bundle p(x_t | v_i) is genuinely a single heavy-tailed
distribution rather than a Gaussian mixture, fitting a multivariate
t-distribution with free degrees-of-freedom nu should give:

  - Substantially higher log-likelihood than a Gaussian fit.
  - Best-fit nu small enough to indicate heavy tails (say nu < 10).

If the conditionals are Gaussian-with-outliers (Possibility 2)
disguised as t-distributions, the fit will still prefer small nu
because t with low nu is the natural shape for "Gaussian core plus
heavy tails". The trimmed-kurtosis test (10.1) is the cleaner
disambiguator; this fit characterizes the bundle shape regardless of
mechanism, and is informative whether 10.1 found rare-extreme structure
or not.

For each input token in the forward set at each of the four
representative checkpoints, we:

  1. Project the conditional bundle into a 32-dim PCA subspace at each
     interior layer (layers 2-10), one bundle at a time.
  2. Fit a multivariate t-distribution by EM (Liu and Rubin, 1995
     EM-style; nu updated by 1-D root-finding on the digamma-based
     score equation each iteration).
  3. Fit a multivariate Gaussian as baseline.
  4. Compute the log-likelihood difference (per-pilot, for
     comparability across token bundle sizes).

The headline statistic is best-fit nu and Delta log-likelihood per
pilot. Small nu and large Delta = intrinsic heavy-tailed.

This is computationally expensive: per-token EM at 32 dims with
hundreds of pilots, repeated across tokens, layers, and checkpoints.
We limit to interior layers 2-10 and use only one seed by default.

Output:
    run_dir/multiview/model_abc/d13_t_fit.npz
    run_dir/multiview/model_abc/figures/d13_t_fit.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import digamma, gammaln
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiview import load_augmented_activations
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


# ----------------------------------------------------------------------
# Multivariate t-distribution log-likelihood.
# ----------------------------------------------------------------------
def _mvt_log_likelihood(
    X: np.ndarray, mu: np.ndarray,
    Sigma: np.ndarray, nu: float,
) -> float:
    """Sum of log-densities under multivariate t with location mu,
    scatter Sigma, degrees of freedom nu."""
    n, d = X.shape
    diff = X - mu
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-6 * np.trace(Sigma) / d * np.eye(d))
    z = solve_triangular(L, diff.T, lower=True)
    quad = (z * z).sum(axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    const = (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0)
             - 0.5 * d * np.log(nu * np.pi)
             - 0.5 * log_det)
    log_lik = const - 0.5 * (nu + d) * np.log1p(quad / nu)
    return float(log_lik.sum())


def _mvn_log_likelihood(
    X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
) -> float:
    n, d = X.shape
    diff = X - mu
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(Sigma + 1e-6 * np.trace(Sigma) / d * np.eye(d))
    z = solve_triangular(L, diff.T, lower=True)
    quad = (z * z).sum(axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    log_lik = -0.5 * d * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad
    return float(log_lik.sum())


# ----------------------------------------------------------------------
# EM for multivariate t (Liu & Rubin 1995 ECME variant).
# ----------------------------------------------------------------------
def fit_mvt_em(
    X: np.ndarray,
    nu_init: float = 10.0,
    max_iter: int = 100,
    tol: float = 1e-4,
    nu_lo: float = 2.5, nu_hi: float = 200.0,
) -> Dict:
    """ECME for multivariate t. Returns dict with mu, Sigma, nu,
    n_iter, converged, log_lik."""
    n, d = X.shape
    # Initialization: sample mean, sample covariance.
    mu = X.mean(0)
    Sigma = np.cov(X.T) + 1e-6 * np.eye(d)
    nu = float(nu_init)
    log_lik_prev = -np.inf

    for it in range(max_iter):
        # E-step: posterior expected u_i = (nu + d) / (nu + quad_i).
        diff = X - mu
        try:
            L = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma = Sigma + 1e-4 * np.trace(Sigma) / d * np.eye(d)
            L = np.linalg.cholesky(Sigma)
        z = solve_triangular(L, diff.T, lower=True)
        quad = (z * z).sum(axis=0)             # (n,)
        u = (nu + d) / (nu + quad)             # (n,)

        # M-step: weighted mean and scatter.
        w_sum = u.sum()
        mu = (u[:, None] * X).sum(0) / w_sum
        diff = X - mu
        Sigma = (diff.T * u) @ diff / n
        Sigma = 0.5 * (Sigma + Sigma.T)        # symmetrize

        # CM-step: nu by 1D root-find on score equation.
        # The ECME nu update solves:
        #   -digamma(nu/2) + log(nu/2) + 1 + (1/n) sum(log u_i - u_i)
        #     + digamma((nu+d)/2) - log((nu+d)/2) = 0
        # over nu in (nu_lo, nu_hi).
        const_term = (1.0 + (np.log(u) - u).mean())
        def _score(nu_val: float) -> float:
            return (-digamma(nu_val / 2.0) + np.log(nu_val / 2.0)
                    + const_term
                    + digamma((nu_val + d) / 2.0)
                    - np.log((nu_val + d) / 2.0))
        # Search for sign change in [nu_lo, nu_hi].
        s_lo = _score(nu_lo)
        s_hi = _score(nu_hi)
        if s_lo * s_hi < 0:
            try:
                nu = float(brentq(_score, nu_lo, nu_hi, xtol=1e-3))
            except Exception:
                pass
        else:
            # No sign change; pick endpoint closer to zero.
            nu = nu_lo if abs(s_lo) < abs(s_hi) else nu_hi

        # Check convergence by log-likelihood.
        log_lik = _mvt_log_likelihood(X, mu, Sigma, nu)
        if np.isfinite(log_lik_prev) and abs(log_lik - log_lik_prev) < tol:
            return {
                "mu": mu, "Sigma": Sigma, "nu": float(nu),
                "n_iter": it + 1, "converged": True,
                "log_lik": float(log_lik),
            }
        log_lik_prev = log_lik

    return {
        "mu": mu, "Sigma": Sigma, "nu": float(nu),
        "n_iter": max_iter, "converged": False,
        "log_lik": float(log_lik_prev),
    }


def fit_mvn(X: np.ndarray) -> Dict:
    n, d = X.shape
    mu = X.mean(0)
    Sigma = np.cov(X.T) + 1e-6 * np.eye(d)
    log_lik = _mvn_log_likelihood(X, mu, Sigma)
    return {"mu": mu, "Sigma": Sigma, "log_lik": float(log_lik)}


# ----------------------------------------------------------------------
# Per-token, per-layer analysis.
# ----------------------------------------------------------------------
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


def analyze_token(
    states: np.ndarray, input_ids: np.ndarray,
    basis: np.ndarray, means: np.ndarray,
    token_id: int, layers: List[int],
    min_pilots: int,
) -> Dict:
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_pilots:
        return {"insufficient": True, "n_total": n_total}
    nu_per_layer = {}
    delta_loglik_per_layer = {}      # (t_loglik - n_loglik) / n_pilots
    n_iter_per_layer = {}
    converged_per_layer = {}
    for t in layers:
        X = states[t, mask, :].astype(np.float64) - means[t]
        Z = X @ basis[t].T                # (n_total, d)
        try:
            t_fit = fit_mvt_em(Z)
            n_fit = fit_mvn(Z)
            nu_per_layer[t] = t_fit["nu"]
            delta = (t_fit["log_lik"] - n_fit["log_lik"]) / n_total
            delta_loglik_per_layer[t] = float(delta)
            n_iter_per_layer[t] = int(t_fit["n_iter"])
            converged_per_layer[t] = bool(t_fit["converged"])
        except Exception as e:
            nu_per_layer[t] = float("nan")
            delta_loglik_per_layer[t] = float("nan")
            n_iter_per_layer[t] = -1
            converged_per_layer[t] = False
    return {
        "insufficient": False,
        "token_id": int(token_id), "n_total": n_total,
        "nu_per_layer": nu_per_layer,
        "delta_loglik_per_layer": delta_loglik_per_layer,
        "n_iter_per_layer": n_iter_per_layer,
        "converged_per_layer": converged_per_layer,
    }


def analyze_checkpoint(
    run_dir: str, seed: int, step: int, tids: np.ndarray,
    layers: List[int], max_pc_dim: int, min_pilots: int,
) -> Optional[Dict]:
    aug_path = augmented_path(run_dir, seed, step)
    if not os.path.exists(aug_path):
        print(f"  [skip] missing {aug_path}")
        return None
    aug = load_augmented_activations(aug_path)
    states = aug["states"]
    input_ids = aug["input_ids"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")
    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    n_tok = tids.size
    t_start = time.time()
    for i, tok in enumerate(tids):
        r = analyze_token(
            states, input_ids, basis, means,
            int(tok), layers=layers, min_pilots=min_pilots,
        )
        results[int(tok)] = r
        if (i + 1) % 5 == 0 or i + 1 == n_tok:
            print(f"      [{i+1}/{n_tok}] elapsed {time.time()-t_start:.1f}s")
    valid = [r for r in results.values()
             if not r.get("insufficient", True)]
    print(f"    {len(valid)}/{n_tok} tokens with sufficient samples")
    return {"seed": seed, "step": step, "results_per_token": results}


def aggregate(checkpoint_results: Dict, layers: List[int]) -> Dict:
    valid = [r for r in checkpoint_results["results_per_token"].values()
             if not r.get("insufficient", True)]
    if not valid:
        return {}
    weights = np.array([r["n_total"] for r in valid], dtype=np.float64)
    weights /= weights.sum()
    nu_mean = {}
    nu_median = {}
    delta_mean = {}
    delta_median = {}
    for t in layers:
        nus = np.array([r["nu_per_layer"].get(t, np.nan) for r in valid])
        deltas = np.array(
            [r["delta_loglik_per_layer"].get(t, np.nan) for r in valid])
        v = np.isfinite(nus) & np.isfinite(deltas)
        if v.sum():
            nu_mean[t] = float(np.average(nus[v], weights=weights[v]))
            nu_median[t] = float(np.median(nus[v]))
            delta_mean[t] = float(np.average(deltas[v], weights=weights[v]))
            delta_median[t] = float(np.median(deltas[v]))
        else:
            nu_mean[t] = nu_median[t] = float("nan")
            delta_mean[t] = delta_median[t] = float("nan")
    return {
        "nu_mean": nu_mean, "nu_median": nu_median,
        "delta_mean": delta_mean, "delta_median": delta_median,
        "n_valid": len(valid),
        "per_token_nu": {int(r["token_id"]): r["nu_per_layer"]
                         for r in valid},
    }


def plot_results(run_dir, aggregates, layers, seed):
    steps = sorted(aggregates.keys())
    if not steps:
        return
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))

    # Panel 1: best-fit nu (median across tokens) per layer per phase.
    ax = axes[0]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        ys = [prof["nu_median"].get(t, np.nan) for t in layers]
        ax.plot(layers, ys, "o-", color=cmap[ci], lw=2,
                label=PHASE_LABELS.get(step, str(step)))
    ax.axhline(10, color="C3", ls="--", lw=1,
               label="nu=10 (heavy-tail threshold)")
    ax.axhline(30, color="gray", ls=":", lw=1,
               label="nu=30 (effectively Gaussian)")
    ax.set_xlabel("layer index t")
    ax.set_ylabel("median best-fit nu across tokens")
    ax.set_yscale("log")
    ax.set_title("Best-fit degrees of freedom by layer/phase")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: Delta log-likelihood per pilot.
    ax = axes[1]
    for ci, step in enumerate(steps):
        prof = aggregates[step]
        if not prof:
            continue
        ys = [prof["delta_median"].get(t, np.nan) for t in layers]
        ax.plot(layers, ys, "o-", color=cmap[ci], lw=2,
                label=PHASE_LABELS.get(step, str(step)))
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer index t")
    ax.set_ylabel("median (LL_t - LL_gaussian) / n_pilots")
    ax.set_title("Per-pilot log-likelihood gap (t over Gaussian)")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: per-token nu scatter at the layer of interest at final
    # checkpoint, ordered by token frequency.
    ax = axes[2]
    final_step = steps[-1]
    prof = aggregates[final_step]
    if prof and "per_token_nu" in prof:
        layer_mid = layers[len(layers) // 2]
        items = [(tid, nu_per_layer.get(layer_mid, np.nan))
                 for tid, nu_per_layer in prof["per_token_nu"].items()]
        items_sorted = sorted(items, key=lambda x: x[1] if np.isfinite(x[1])
                              else 1e9)
        xs = list(range(len(items_sorted)))
        ys = [it[1] for it in items_sorted]
        ax.bar(xs, ys, color="C0")
        ax.axhline(10, color="C3", ls="--", lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(it[0])) for it in items_sorted],
                           rotation=70, fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel("token id (sorted by nu)")
        ax.set_ylabel(f"best-fit nu at layer {layer_mid}")
        ax.set_title(f"Per-token nu at {PHASE_LABELS.get(final_step)}")

    fig.suptitle(
        f"D13: multivariate t-distribution fit (Possibility 1), seed {seed}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d13_t_fit.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


DEFAULT_STEPS = [479, 2563, 9809, 24000]
DEFAULT_LAYERS = [2, 4, 6, 7, 8, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--min-pilots", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")
    print(f"Steps: {args.steps}")
    print(f"Layers: {args.layers}")

    aggregates = {}
    t0 = time.time()
    for step in args.steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, str(step))}:")
        result = analyze_checkpoint(
            args.run_dir, args.seed, step, tids,
            layers=args.layers, max_pc_dim=args.max_pc_dim,
            min_pilots=args.min_pilots,
        )
        if result is None:
            continue
        aggregates[step] = aggregate(result, args.layers)
        prof = aggregates[step]
        if prof:
            print(f"    n_valid_tokens = {prof['n_valid']}")
            print(f"    median nu by layer:")
            for t in args.layers:
                print(f"      layer {t:>2d}: nu = {prof['nu_median'][t]:.2f}  "
                      f"delta_LL/n = {prof['delta_median'][t]:.4f}")

    elapsed = time.time() - t0
    print(f"\nAll checkpoints done in {elapsed/60:.1f} min")

    out_arrays = {
        "steps": np.array(sorted(aggregates.keys()), dtype=np.int64),
        "layers": np.array(args.layers, dtype=np.int32),
        "seed": np.int32(args.seed),
    }
    for step, prof in aggregates.items():
        if not prof:
            continue
        for t in args.layers:
            out_arrays[f"step_{step}_nu_median_layer_{t}"] = (
                np.float64(prof["nu_median"][t]))
            out_arrays[f"step_{step}_delta_median_layer_{t}"] = (
                np.float64(prof["delta_median"][t]))
            out_arrays[f"step_{step}_nu_mean_layer_{t}"] = (
                np.float64(prof["nu_mean"][t]))
            out_arrays[f"step_{step}_delta_mean_layer_{t}"] = (
                np.float64(prof["delta_mean"][t]))
    out_path = os.path.join(output_root(args.run_dir), "d13_t_fit.npz")
    np.savez(out_path, **out_arrays)
    print(f"[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates, args.layers, args.seed)


if __name__ == "__main__":
    main()
