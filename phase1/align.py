"""
Vocabulary-anchored Procrustes alignment for cross-model linear-flow comparison.

The problem this solves: two trained transformer models — even with identical
architecture and tokenizer — have different learned embedding matrices, so
their residual streams live in formally different vector spaces. Direct
comparison of recovered SVD bases R(t) is ill-defined.

The solution: use the shared vocabulary or shared inputs as a "Rosetta stone"
to construct an orthogonal map between the two models' hidden spaces. After
transport via this map, the linear flows can be quantitatively compared.

Two flavors of alignment, both implemented here:

  1. Embedding-space alignment (`align_embeddings`):
        Given two embedding matrices E^(A) and E^(B) over the same vocabulary,
        find the orthogonal matrix Q minimizing ||E^(A) Q - E^(B)||_F.

  2. Per-layer activation alignment (`align_activations_per_layer`):
        Given the same input corpus passed through models A and B, collect
        per-layer activations X_t^(A) and X_t^(B). For each layer t, find
        the orthogonal Q_t minimizing ||X_t^(A) Q_t - X_t^(B)||_F.

Both routines return the orthogonal map Q AND the residual ratio ρ that
measures alignment quality. A small ρ means the spaces correspond well;
a large ρ means they don't.

Once alignments are computed, `transport_R(R_A, Q_source, Q_target)` transports
model A's linear-flow basis into model B's coordinate system, after which
Frobenius distance comparisons are meaningful.

Mathematical background — orthogonal Procrustes:
    Given two N×H matrices Y and X, the problem
        min_Q  ||X @ Q - Y||_F   subject to Q^T Q = I
    has closed-form solution: compute the SVD of Y^T X = U S V^T, then
    Q = V U^T. See Schönemann (1966).

Reference implementation note: scipy.linalg.orthogonal_procrustes does
exactly this, but we implement it directly to avoid the scipy dependency
and to control fp precision precisely.
"""

from typing import Optional, Tuple, Dict, List

import numpy as np


# ----------------------------------------------------------------------
# Core: orthogonal Procrustes.
# ----------------------------------------------------------------------
def orthogonal_procrustes(
    X: np.ndarray, Y: np.ndarray,
    allow_reflection: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Find the orthogonal matrix Q minimizing ||X @ Q - Y||_F.

    Args:
        X, Y: (N, H) matrices. Each row of X corresponds to the same anchor
              point (e.g., a vocabulary token, or a specific input) as the
              corresponding row of Y. N >= H typical but not required.
        allow_reflection: if True, Q is allowed to be an arbitrary orthogonal
              matrix (det ±1). If False, Q is restricted to proper rotations
              (det +1). For comparing learned representations across models,
              True is usually right — there's no a priori reason to disallow
              reflections in the embedding space.

    Returns:
        Q: (H, H) orthogonal matrix.
        residual_ratio: ||X @ Q - Y||_F / ||Y||_F, the normalized residual.
            Small ratio = good alignment, large ratio = bad.

    Note on numerics: we cast to fp64 internally because the SVD step can
    lose precision in fp32 for moderately ill-conditioned matrices.
    """
    assert X.shape == Y.shape, (
        f"X.shape {X.shape} != Y.shape {Y.shape}"
    )
    X64 = X.astype(np.float64)
    Y64 = Y.astype(np.float64)

    # Procrustes solution: SVD of Y^T X = U S V^T, then Q = V U^T.
    # (Some references use Q = U V^T; sign depends on which side X is on.
    #  We want to find Q such that X Q ≈ Y, which gives Q = V U^T where
    #  Y^T X = U S V^T.)
    M = Y64.T @ X64  # (H, H)
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = Vt.T @ U.T  # (H, H)

    if not allow_reflection:
        # Force det Q = +1. If currently -1, flip the sign of the last column
        # of Vt^T (equivalently, the last row of Vt).
        if np.linalg.det(Q) < 0:
            Vt_fixed = Vt.copy()
            Vt_fixed[-1, :] *= -1
            Q = Vt_fixed.T @ U.T

    # Compute the residual.
    X_aligned = X64 @ Q  # (N, H)
    err = X_aligned - Y64
    err_norm = np.linalg.norm(err)
    y_norm = np.linalg.norm(Y64)
    residual_ratio = float(err_norm / max(y_norm, 1e-30))

    return Q.astype(np.float32), residual_ratio


# ----------------------------------------------------------------------
# Embedding-space alignment.
# ----------------------------------------------------------------------
def align_embeddings(
    E_A: np.ndarray, E_B: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Align embedding matrices of two models with the same vocabulary.

    Args:
        E_A, E_B: (V, H) embedding matrices. Row i is the embedding of
            vocabulary token i in each model's hidden space.

    Returns:
        Q: (H, H) orthogonal map from A's space to B's space.
        residual_ratio: how well A's embeddings align to B's after rotation.

    The residual_ratio is a useful first measurement of cross-model
    correspondence — if two models with the same architecture and corpus
    have residual_ratio ~ 0.05, they roughly agree on vocabulary geometry;
    if it's ~ 0.5, they don't.
    """
    return orthogonal_procrustes(E_A, E_B)


# ----------------------------------------------------------------------
# Per-layer activation alignment.
# ----------------------------------------------------------------------
def align_activations_per_layer(
    activations_A: np.ndarray, activations_B: np.ndarray,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Align per-layer activations of two models that have processed the
    same input corpus.

    Args:
        activations_A, activations_B: (num_layers, N, H) arrays. The i-th
            row at layer t in both A and B should be the model's activation
            at the same (input, position) pair. This is the "shared inputs"
            assumption that makes the alignment well-posed.

    Returns:
        Qs: list of (H, H) orthogonal matrices, one per layer.
        residual_ratios: list of floats, one per layer.

    Note: the alignment is independent per layer, so it CAN drift across
    layers — that's a feature, not a bug. The drift itself is informative
    (it tells us how the two models' bases evolve differently with depth).
    """
    L = activations_A.shape[0]
    assert activations_A.shape == activations_B.shape, (
        f"Activation shape mismatch: {activations_A.shape} vs {activations_B.shape}"
    )
    Qs = []
    residuals = []
    for t in range(L):
        Q, r = orthogonal_procrustes(activations_A[t], activations_B[t])
        Qs.append(Q)
        residuals.append(r)
    return Qs, residuals


# ----------------------------------------------------------------------
# Transport an R matrix through the alignment.
# ----------------------------------------------------------------------
def transport_R(
    R_A: np.ndarray, Q_source: np.ndarray, Q_target: np.ndarray,
) -> np.ndarray:
    """
    Transport an R matrix from model A's coordinate system to model B's.

    The R matrix encodes a linear transformation in A's hidden space. To
    express the same transformation in B's hidden space, we conjugate:

        R_in_B_coords = Q_source.T @ R_A @ Q_target

    where Q_source aligns the *input* side (where the R operates on input
    vectors from A's space) and Q_target aligns the *output* side.

    For per-layer-state R(t), both sides typically use the same alignment
    Q_t at the layer t in question. For pairwise predictions involving
    R(t) and R(t+τ), the source uses Q_t and the target uses Q_{t+τ}.

    Args:
        R_A: (H, H) — model A's R matrix, in A's coordinate system.
        Q_source: (H, H) — alignment from A's space (input side) to B's.
        Q_target: (H, H) — alignment from A's space (output side) to B's.

    Returns:
        R_in_B: (H, H) — the same R, expressed in B's coordinate system.
    """
    return Q_source.T @ R_A @ Q_target


def transport_R_per_layer(
    R_A: np.ndarray, Qs: List[np.ndarray],
) -> np.ndarray:
    """
    Transport per-layer-state R matrices into the aligned coordinate system.

    Each R_A[t] gets transported as Q_t.T @ R_A[t] @ Q_t (input and output
    side use the same Q_t since R(t) operates within layer t's coordinate
    system).

    Args:
        R_A: (L, H, H) — per-layer-state R matrices.
        Qs: list of L (H, H) orthogonal alignments.

    Returns:
        R_transported: (L, H, H) — R matrices in the aligned coordinate
            system, ready for direct comparison.
    """
    L = R_A.shape[0]
    assert len(Qs) == L, (
        f"len(Qs)={len(Qs)} != R_A.shape[0]={L}"
    )
    out = np.zeros_like(R_A)
    for t in range(L):
        out[t] = transport_R(R_A[t], Qs[t], Qs[t])
    return out


# ----------------------------------------------------------------------
# Convenience: align two flow dicts end to end.
# ----------------------------------------------------------------------
def align_two_flows(
    flow_A: Dict, flow_B: Dict,
    shared_activations_A: Optional[np.ndarray] = None,
    shared_activations_B: Optional[np.ndarray] = None,
) -> Dict:
    """
    End-to-end alignment of two flow dicts.

    Args:
        flow_A, flow_B: dicts produced by analyze.recover_linear_flow() or
            analyze.load_flow(). Must have keys 'R', 'singular_values',
            'means' at minimum.
        shared_activations_A, shared_activations_B: (L, N, H) arrays of
            activations from the same inputs through models A and B,
            for the per-layer activation alignment. If None, alignment
            is skipped (returns the flows unchanged).

    Returns:
        A dict with:
          - 'Qs': list of L per-layer alignments (or None if activations
                  weren't provided)
          - 'residual_ratios': list of per-layer alignment residual ratios
          - 'R_A_transported': flow_A['R'] transported into B's coords
          - 'flow_A': original flow_A
          - 'flow_B': original flow_B
    """
    out = {"flow_A": flow_A, "flow_B": flow_B}
    if shared_activations_A is None or shared_activations_B is None:
        out["Qs"] = None
        out["residual_ratios"] = None
        out["R_A_transported"] = None
        return out

    Qs, residuals = align_activations_per_layer(
        shared_activations_A, shared_activations_B,
    )
    out["Qs"] = Qs
    out["residual_ratios"] = residuals
    out["R_A_transported"] = transport_R_per_layer(flow_A["R"], Qs)
    return out
