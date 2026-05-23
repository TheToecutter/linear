"""
Unembedding-subspace decomposition: hypothesis N1.

This diagnostic has no forward analog. It tests whether the marginal's
non-Gaussianity is localized to the orthogonal complement of the
unembedding's rowspan.

Setup. Let W_U in R^(|V| x H) be the unembedding matrix, with SVD
W_U = U_W S_W V_W^T. The matrix V_W in R^(H x r) has columns spanning
the rowspan of W_U (up to numerical rank r), which is the subspace of
the residual stream that the logits actually see. For tied-embedding
architectures, W_U = W_E^T and r is the rank of the embedding matrix
(typically full, r ~ H).

For a truncation rank d_parallel <= r, define:

    P_parallel = V_W[:, :d_parallel] @ V_W[:, :d_parallel].T   (H x H)
    P_perp     = I - P_parallel                                (H x H)

x_t_parallel = P_parallel @ x_t lives in the readout-visible subspace
of dimension d_parallel; x_t_perp lives in its orthogonal complement
of dimension H - d_parallel.

Hypothesis N1. At the late layers (especially t = L_total - 1), the
per-coordinate excess kurtosis of x_t_parallel is substantially smaller
than that of x_t_perp. Read: the unembedding's geometric constraint
regularizes the readout-visible component into approximate Gaussianity,
and the marginal's heavy tails live in the off-readout complement.

Implementation. We load the model checkpoint, extract W_U via
model.get_lm_head_weight() (which handles the tied-embedding case),
compute the SVD once per checkpoint, and reuse V_W across the
truncation-rank sweep d_parallel in {32, 64, 128, 256}.

For each (seed, checkpoint, d_parallel) we report:

  - empirical marginal kurtosis in parallel subspace (per-layer profile)
  - empirical marginal kurtosis in perpendicular subspace (per-layer)
  - empirical marginal kurtosis on the full residual stream
    (cross-check; per-layer)
  - per-cell kurtosis in parallel and perpendicular subspaces, for the
    reverse_actual token set (used for hypothesis F2 refinement: do
    cells become Gaussian in the parallel subspace at late layers?)

Output files:
    d_n1_unembedding_subspace_{view}/seed{S}_step{T:08d}_d{d_parallel}.npz

R4 mitigation. Tied embeddings make the rowspan of W_U identical to the
columnspan of the embedding. This could render N1 trivially true if
forward conditionals were non-Gaussian *inside* this subspace at t = 0
(they are point masses there, so technically degenerate). The
diagnostic against the R4 artifact: report the parallel-vs-perp
kurtosis gap at *all* 14 layers, not just the last. A real late-layer
regularization shows the gap growing toward the last layer; a
construction artifact shows it constant or shrinking.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None

from multiview import load_augmented_activations
from multiview_campaign import (
    augmented_path,
    checkpoints_in_seed,
    load_token_sets,
)
from reverse_buildup import (
    VIEWS,
    output_root,
    select_id_array_for_view,
)


# Default truncation rank sweep, matching proposal §3.3.
DEFAULT_D_PARALLEL_SWEEP = (32, 64, 128, 256)


# ----------------------------------------------------------------------
# Unembedding extraction and SVD cache.
# ----------------------------------------------------------------------
def extract_unembedding_basis(
    ckpt_path: str,
    model_cfg,
    max_rank: int = 768,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a checkpoint, instantiate the model, extract W_U, compute SVD.

    Args:
        ckpt_path: filesystem path to a .pt checkpoint file.
        model_cfg: ModelConfig instance matching the checkpoint's
                   architecture (passed in, not loaded, because config
                   is run-level not per-checkpoint).
        max_rank: cap on the number of singular vectors to retain. The
                  default 768 matches the 150M model's hidden size; for
                  larger models, set higher.

    Returns:
        (V_W, sigma)
            V_W:   (H, r) array. Columns are right singular vectors of
                   W_U, sorted by descending singular value. The first
                   d columns span the rank-d truncation of W_U's
                   rowspan.
            sigma: (r,) singular values of W_U, sorted descending.

    Implementation: for tied-embedding architectures, W_U = W_E^T where
    W_E is the embedding matrix. The model's get_lm_head_weight()
    method handles both tied and untied cases, returning a (|V|, H)
    tensor in both. We compute the SVD on CPU to keep the GPU free for
    other work; the matrix is at most (~32k, 768), so the SVD is
    sub-second.
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "extract_unembedding_basis requires torch; install the project "
            "environment or run on a machine with torch available."
        )
    from model import LlamaStyleTransformer

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LlamaStyleTransformer(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        W_U = model.get_lm_head_weight().detach().cpu().float().numpy()
    # W_U has shape (|V|, H). The rowspan of W_U is the column span of
    # W_U^T, which equals the column span of W_U^T. The right singular
    # vectors of W_U (== rows of V^T from SVD) span this rowspan.
    # We do SVD on W_U directly to get V.
    U, s, Vt = np.linalg.svd(W_U, full_matrices=False)
    # Vt has shape (min(|V|, H), H). Each row is a right singular vec.
    # Transpose to (H, r) for column-major access.
    V_W = Vt.T.astype(np.float64)            # (H, r)
    sigma = s.astype(np.float64)              # (r,)
    if V_W.shape[1] > max_rank:
        V_W = V_W[:, :max_rank]
        sigma = sigma[:max_rank]
    return V_W, sigma


# ----------------------------------------------------------------------
# Per-coordinate excess kurtosis on a (N, d) array.
# ----------------------------------------------------------------------
def _per_coord_excess_kurt(arr: np.ndarray) -> float:
    """Mean across coordinates of per-coordinate excess kurtosis."""
    with np.errstate(invalid="ignore"):
        m = arr.mean(0)
        v = arr.var(0)
        m4 = ((arr - m) ** 4).mean(0)
        return float(np.mean(m4 / (v ** 2 + 1e-30) - 3.0))


# ----------------------------------------------------------------------
# Main projection-and-kurtosis routine.
# ----------------------------------------------------------------------
def run_unembedding_decomposition(
    run_dir: str,
    seed: int,
    step: int,
    ckpt_path: str,
    model_cfg,
    view: str = "reverse_actual",
    d_parallel_sweep: Tuple[int, ...] = DEFAULT_D_PARALLEL_SWEEP,
    verbose: bool = True,
) -> List[Dict]:
    """Project pilots into parallel/perp subspaces of W_U; compute
    kurtosis profiles.

    Args:
        run_dir, seed, step: identify which augmented file to load.
        ckpt_path: path to the model checkpoint (needed to extract W_U).
        model_cfg: ModelConfig instance.
        view: which token set to use for per-cell breakdown. Default
              reverse_actual; reverse_pred is also valid.
        d_parallel_sweep: list of truncation ranks to try.

    Returns:
        list of result dicts, one per d_parallel. Also writes one .npz
        per d_parallel to
        d_n1_unembedding_subspace_{view}/seed{S}_step{T:08d}_d{d}.npz.

    Output dict per file:
        d_parallel        (int)
        layers            (L,)
        marg_kurt_full    (L,)  empirical marginal per-coord excess kurt
        marg_kurt_par     (L,)  marginal in parallel subspace
        marg_kurt_perp    (L,)  marginal in orthogonal complement
        cell_kurt_par     (n_cells, L) per-cell, parallel subspace
        cell_kurt_perp    (n_cells, L) per-cell, orthogonal complement
        cell_kurt_full    (n_cells, L) per-cell, full residual stream
        tids              (n_cells,) the token ids of the chosen view's set
        gap_par_minus_perp (L,) = marg_kurt_par - marg_kurt_perp
                                 (the headline N1 statistic)
    """
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view!r}")

    aug_path = augmented_path(run_dir, seed, step)
    if not os.path.exists(aug_path):
        if verbose:
            print(f"[N1/{view}] seed {seed} step {step}: "
                  f"missing augmented file, skip")
        return []

    # Early skip: if every requested d_parallel file already exists on
    # disk, return the cached results without loading the checkpoint or
    # the augmented activations. This makes --expand-d cheap for seeds
    # that were already processed in a previous run.
    out_subdir = os.path.join(output_root(run_dir),
                              f"d_n1_unembedding_subspace_{view}")
    expected_paths = [
        os.path.join(out_subdir,
                     f"seed{seed}_step{step:08d}_d{d_par:04d}.npz")
        for d_par in d_parallel_sweep
    ]
    if all(os.path.exists(p) for p in expected_paths):
        if verbose:
            print(f"[N1/{view}] seed {seed} step {step}: all d_par files "
                  f"cached, skipping setup")
        cached = []
        for p in expected_paths:
            with np.load(p, allow_pickle=False) as f:
                cached.append({k: f[k] for k in f.files})
        return cached

    if verbose:
        print(f"[N1/{view}] seed {seed} step {step}: "
              f"extracting unembedding ...")
    V_W, sigma = extract_unembedding_basis(ckpt_path, model_cfg)
    H = V_W.shape[0]
    if verbose:
        print(f"[N1/{view}] V_W shape: ({H}, {V_W.shape[1]}), "
              f"sigma range: [{sigma.min():.3f}, {sigma.max():.3f}]")

    if verbose:
        print(f"[N1/{view}] loading activations ...")
    aug = load_augmented_activations(aug_path)
    states = aug["states"]                      # (L, N, H)
    L, N, H_aug = states.shape
    if H_aug != H:
        raise RuntimeError(
            f"Augmented hidden size {H_aug} != unembedding rows {H}. "
            f"Mismatched checkpoint and augmented file?"
        )
    id_array = select_id_array_for_view(aug, view)

    # Frozen token set for the chosen view (for per-cell breakdown).
    fwd, rev_act, rev_pred = load_token_sets(run_dir)
    if view == "forward":
        token_set = fwd
    elif view == "reverse_actual":
        token_set = rev_act
    else:
        token_set = rev_pred
    tids = token_set.token_ids.astype(np.int32)

    os.makedirs(out_subdir, exist_ok=True)

    results = []
    for d_par in d_parallel_sweep:
        if d_par > V_W.shape[1]:
            if verbose:
                print(f"[N1/{view}] d_parallel={d_par} exceeds available "
                      f"singular vectors ({V_W.shape[1]}); skipping")
            continue
        out_path = os.path.join(
            out_subdir, f"seed{seed}_step{step:08d}_d{d_par:04d}.npz")
        if os.path.exists(out_path):
            with np.load(out_path) as f:
                results.append({k: f[k] for k in f.files})
            continue

        # Projector matrices.
        Vp = V_W[:, :d_par]                  # (H, d_par)
        # P_par = Vp @ Vp.T;  P_perp = I - P_par. We don't materialize
        # P_par or P_perp explicitly because that's an H x H matrix;
        # instead, we project on the fly using Vp.

        marg_kurt_full = np.full(L, np.nan)
        marg_kurt_par = np.full(L, np.nan)
        marg_kurt_perp = np.full(L, np.nan)
        cell_kurt_full = np.full((tids.size, L), np.nan)
        cell_kurt_par = np.full((tids.size, L), np.nan)
        cell_kurt_perp = np.full((tids.size, L), np.nan)

        for t in range(L):
            X = states[t].astype(np.float64)        # (N, H)
            Z_par = X @ Vp                         # (N, d_par)
            # Perpendicular component: X - P_par @ X = X - (X @ Vp) @ Vp.T.
            # Don't form the full (N, H) array repeatedly; compute kurt
            # on the reconstruction explicitly.
            X_par_full = Z_par @ Vp.T              # (N, H)
            X_perp = X - X_par_full                # (N, H)

            marg_kurt_full[t] = _per_coord_excess_kurt(X)
            # For the parallel component, the meaningful coordinates
            # are the d_par PCs of W_U's rowspan, not the H-dim
            # embedding. Compute kurt on Z_par (the d_par-dimensional
            # rep), not on X_par_full (the H-dim padding).
            marg_kurt_par[t] = _per_coord_excess_kurt(Z_par)
            # For the perpendicular component, the meaningful
            # coordinates are the H - d_par directions orthogonal to
            # W_U's rowspan; we report the kurt averaged over those
            # directions by computing per-coord kurt on X_perp (which
            # has zero variance along the d_par parallel directions,
            # so they wash out in the average -- correct behavior).
            # To be precise, we project X_perp onto its non-degenerate
            # subspace by simply zeroing the coordinates whose
            # variance is below a threshold.
            v_perp = X_perp.var(0)
            valid_perp = v_perp > 1e-12 * v_perp.max()
            if valid_perp.any():
                marg_kurt_perp[t] = _per_coord_excess_kurt(
                    X_perp[:, valid_perp])

            for k, tok in enumerate(tids):
                mask = (id_array == int(tok))
                n = int(mask.sum())
                if n < d_par + 2:
                    continue
                Xc = X[mask]
                Z_c = Z_par[mask]
                Xperp_c = X_perp[mask]
                cell_kurt_full[k, t] = _per_coord_excess_kurt(Xc)
                cell_kurt_par[k, t] = _per_coord_excess_kurt(Z_c)
                if valid_perp.any():
                    cell_kurt_perp[k, t] = _per_coord_excess_kurt(
                        Xperp_c[:, valid_perp])

        gap = marg_kurt_par - marg_kurt_perp

        np.savez(
            out_path,
            view=np.array(view),
            seed=np.int32(seed), step=np.int64(step),
            d_parallel=np.int32(d_par),
            layers=np.arange(L, dtype=np.int32),
            sigma=sigma,
            marg_kurt_full=marg_kurt_full,
            marg_kurt_par=marg_kurt_par,
            marg_kurt_perp=marg_kurt_perp,
            cell_kurt_full=cell_kurt_full,
            cell_kurt_par=cell_kurt_par,
            cell_kurt_perp=cell_kurt_perp,
            gap_par_minus_perp=gap,
            tids=tids,
        )
        results.append({
            "view": view,
            "seed": seed, "step": step,
            "d_parallel": d_par,
            "layers": np.arange(L),
            "marg_kurt_full": marg_kurt_full,
            "marg_kurt_par": marg_kurt_par,
            "marg_kurt_perp": marg_kurt_perp,
            "cell_kurt_full": cell_kurt_full,
            "cell_kurt_par": cell_kurt_par,
            "cell_kurt_perp": cell_kurt_perp,
            "gap_par_minus_perp": gap,
            "tids": tids,
        })
        if verbose:
            print(f"[N1/{view}] d_parallel={d_par}: -> {out_path}")
    return results


# ----------------------------------------------------------------------
# Convenience: find the checkpoint path for a given (seed, step).
# ----------------------------------------------------------------------
def find_checkpoint(run_dir: str, seed: int, step: int) -> Optional[str]:
    """Return the .pt path for (seed, step) in run_dir, or None."""
    for s, path in checkpoints_in_seed(run_dir, seed):
        if s == step:
            return path
    return None
    