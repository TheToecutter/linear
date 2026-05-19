"""
Distance metrics for comparing recovered linear flows across checkpoints
and across models.

Two frames, both supported:

  Frame 1 (per-layer):
    Compare quantities layer by layer. R(t), Σ(t), kurtosis(t), effective
    rank(t), etc. Most direct, easiest to interpret. Used for convergence
    analysis (within a single model across checkpoints) and for
    within-architecture cross-seed comparison.

  Frame 2 (continuous trajectory):
    Treat {R(t)}_t as a sample of a continuous curve R: [0, L] →
    Stiefel(H), interpolate to common τ ∈ [0, 1], compare curves as
    geometric objects. Required for cross-architecture comparison when
    architectures have different layer counts, but useful even at fixed
    layer count because curve-level statistics (total angular distance,
    smoothness, curvature) are intrinsically basis-invariant.

Public functions:

  Basis-invariant distances (Frame 1):
    - singular_value_distance(flow_A, flow_B): per-layer log-Σ distance
    - lambda_distance(flow_A, flow_B): variance scaling rate difference
    - effective_rank_distance(flow_A, flow_B): manifold dim difference
    - gaussianity_distance(flow_A, flow_B): kurtosis + isotropy distance

  Aligned distances (Frame 1, requires alignment):
    - aligned_R_distance(R_A_transported, R_B): per-layer R Frobenius
    - principal_angle_profile(R_A_transported, R_B): per-layer angle stats

  Curve-level statistics (Frame 2):
    - total_angular_distance(R): cumulative angle traveled through depth
    - mean_smoothness(R): typical per-layer rotation angle
    - reparameterized_R(R, num_samples): resample R curve to common τ grid
    - curve_distance(R_A_resampled, R_B_resampled): integrated distance
        along the resampled curves

  Composite report:
    - compare_flows(flow_A, flow_B, aligned=False): one-stop comparison
        returning a dict of all relevant metrics.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Basis-invariant distances (Frame 1).
# ----------------------------------------------------------------------
def singular_value_distance(
    flow_A: Dict, flow_B: Dict, per_layer: bool = False,
) -> np.ndarray:
    """
    Per-layer distance between singular value spectra, in log space.

    For each layer t, compute ||log Σ_A(t) - log Σ_B(t)||_2. This is
    basis-invariant (singular values are gauge-invariant scalars).

    Args:
        flow_A, flow_B: flow dicts with 'singular_values' field.
        per_layer: if True, return per-layer distances; else return their sum.

    Returns:
        (L,) array if per_layer=True, scalar otherwise.
    """
    sv_A = flow_A["singular_values"]  # (L, H)
    sv_B = flow_B["singular_values"]
    if sv_A.shape != sv_B.shape:
        raise ValueError(
            f"singular_values shape mismatch: {sv_A.shape} vs {sv_B.shape}"
        )
    log_A = np.log(np.maximum(sv_A, 1e-12))
    log_B = np.log(np.maximum(sv_B, 1e-12))
    per_layer_dist = np.linalg.norm(log_A - log_B, axis=-1)  # (L,)
    return per_layer_dist if per_layer else float(per_layer_dist.sum())


def lambda_distance(flow_A: Dict, flow_B: Dict) -> float:
    """Absolute difference in variance scaling rate λ."""
    return abs(float(flow_A["lambda"]) - float(flow_B["lambda"]))


def effective_rank_distance(
    flow_A: Dict, flow_B: Dict, per_layer: bool = False,
):
    """Per-layer distance in effective rank (manifold dimensionality)."""
    er_A = flow_A["effective_rank"]
    er_B = flow_B["effective_rank"]
    diff = np.abs(er_A - er_B)
    return diff if per_layer else float(diff.sum())


def gaussianity_distance(flow_A: Dict, flow_B: Dict) -> Dict[str, float]:
    """
    Compare Gaussianity diagnostics between two flows.

    Returns:
        dict with 'kurtosis_diff', 'isotropy_diff' (per-layer L1 sums).
    """
    kurt_A = np.array(flow_A["kurtosis_per_layer"])
    kurt_B = np.array(flow_B["kurtosis_per_layer"])
    iso_A = np.array(flow_A["isotropy_per_layer"])
    iso_B = np.array(flow_B["isotropy_per_layer"])
    # Mask NaNs (layer 0 has no residuals).
    kurt_mask = ~(np.isnan(kurt_A) | np.isnan(kurt_B))
    iso_mask = ~(np.isnan(iso_A) | np.isnan(iso_B))
    return {
        "kurtosis_diff": float(
            np.abs(kurt_A[kurt_mask] - kurt_B[kurt_mask]).sum()
        ),
        "isotropy_diff": float(
            np.abs(iso_A[iso_mask] - iso_B[iso_mask]).sum()
        ),
    }


# ----------------------------------------------------------------------
# Aligned distances (Frame 1, requires alignment).
# ----------------------------------------------------------------------
def aligned_R_distance(
    R_A_transported: np.ndarray, R_B: np.ndarray, per_layer: bool = False,
):
    """
    Per-layer Frobenius distance between R matrices after alignment.

    Args:
        R_A_transported: (L, H, H) — model A's R matrices already transported
            into model B's coordinate system (via transport_R_per_layer).
        R_B: (L, H, H) — model B's R matrices in their native coordinates.

    Returns:
        (L,) array or scalar sum.
    """
    if R_A_transported.shape != R_B.shape:
        raise ValueError(
            f"R shape mismatch: {R_A_transported.shape} vs {R_B.shape}"
        )
    L = R_A_transported.shape[0]
    per_layer_dist = np.zeros(L, dtype=np.float32)
    for t in range(L):
        per_layer_dist[t] = float(np.linalg.norm(R_A_transported[t] - R_B[t]))
    return per_layer_dist if per_layer else float(per_layer_dist.sum())


def principal_angle_profile(
    R_A_transported: np.ndarray, R_B: np.ndarray, top_k: int = 10,
) -> np.ndarray:
    """
    Per-layer principal-angle statistics between two R matrices' top-k
    principal directions, after alignment.

    For each layer t, compute the top-k principal angles between the rows
    of R_A_transported[t] and R_B[t]. Returns the *mean* angle in degrees
    across the top-k directions.

    Args:
        R_A_transported, R_B: (L, H, H) — R matrices, A's already transported.
        top_k: how many leading directions to include.

    Returns:
        (L,) array of mean principal angles in degrees.
    """
    L = R_A_transported.shape[0]
    out = np.zeros(L, dtype=np.float32)
    for t in range(L):
        # Top-k rows of each R.
        A_top = R_A_transported[t, :top_k]
        B_top = R_B[t, :top_k]
        # Singular values of A_top @ B_top.T are cosines of principal angles.
        M = A_top @ B_top.T  # (top_k, top_k)
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        angles = np.degrees(np.arccos(s))
        out[t] = float(angles.mean())
    return out


# ----------------------------------------------------------------------
# Curve-level statistics (Frame 2).
# ----------------------------------------------------------------------
def successive_layer_angles(
    R: np.ndarray, top_k: Optional[int] = None,
) -> np.ndarray:
    """
    Mean principal angle (in degrees) between R(t) and R(t+1) for each t.

    This is a basis-invariant property of the trajectory — it tells you
    how much the principal direction basis rotates per layer. Smooth
    trajectories have small angles; chaotic ones have large angles.

    Args:
        R: (L, H, H) — per-layer-state R matrices.
        top_k: if not None, only consider the top_k rows (which carry
            the most variance). Useful because trailing rows are noisy.

    Returns:
        (L-1,) array of mean principal angles in degrees.
    """
    L, H, _ = R.shape
    k = top_k if top_k is not None else H
    out = np.zeros(L - 1, dtype=np.float32)
    for t in range(L - 1):
        A = R[t, :k]
        B = R[t + 1, :k]
        M = A @ B.T  # (k, k)
        s = np.linalg.svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        angles = np.degrees(np.arccos(s))
        out[t] = float(angles.mean())
    return out


def total_angular_distance(R: np.ndarray, top_k: Optional[int] = None) -> float:
    """
    Cumulative angular distance traveled by R from layer 0 to layer L.

    Approximated as the sum of successive_layer_angles. Basis-invariant
    summary of the trajectory's total geometric extent.
    """
    return float(successive_layer_angles(R, top_k=top_k).sum())


def reparameterize_singular_values(
    Sigma: np.ndarray, num_samples: int,
) -> np.ndarray:
    """
    Resample singular value trajectories to a common τ ∈ [0, 1] grid.

    Uses simple linear interpolation per-direction.

    Args:
        Sigma: (L, H) — per-layer singular value spectra.
        num_samples: how many samples to resample to.

    Returns:
        (num_samples, H) — singular values evaluated at τ = i/(num_samples-1)
        for i = 0, 1, ..., num_samples-1.
    """
    L, H = Sigma.shape
    out = np.zeros((num_samples, H), dtype=np.float32)
    tau_grid = np.linspace(0, L - 1, num_samples)
    for h in range(H):
        out[:, h] = np.interp(tau_grid, np.arange(L), Sigma[:, h])
    return out


def reparameterized_singular_value_distance(
    flow_A: Dict, flow_B: Dict, num_samples: int = 100,
) -> float:
    """
    Frame 2 version: reparameterize both flows' singular value trajectories
    to a common τ grid, then compute distance in log-space.

    This is the analog of singular_value_distance for cross-architecture
    comparison where the two flows might have different layer counts L.

    Returns:
        Scalar: integrated log-Σ distance over the common τ grid.
    """
    sigma_A = reparameterize_singular_values(
        flow_A["singular_values"], num_samples
    )
    sigma_B = reparameterize_singular_values(
        flow_B["singular_values"], num_samples
    )
    log_A = np.log(np.maximum(sigma_A, 1e-12))
    log_B = np.log(np.maximum(sigma_B, 1e-12))
    per_sample_dist = np.linalg.norm(log_A - log_B, axis=-1)
    # Trapezoidal integration over τ ∈ [0, 1].
    integral = np.trapz(per_sample_dist, dx=1.0 / max(num_samples - 1, 1))
    return float(integral)


# ----------------------------------------------------------------------
# Composite comparison report.
# ----------------------------------------------------------------------
def compare_flows(
    flow_A: Dict, flow_B: Dict,
    R_A_transported: Optional[np.ndarray] = None,
    alignment_residuals: Optional[List[float]] = None,
    name_A: str = "A", name_B: str = "B",
) -> Dict:
    """
    Comprehensive comparison of two flows. Returns a dict of metrics
    plus a printable summary string.

    Args:
        flow_A, flow_B: flow dicts from analyze.recover_linear_flow().
        R_A_transported: optional, A's R matrices already transported into
            B's coordinate system (via align.transport_R_per_layer).
            If provided, aligned-distance metrics are also computed.
        alignment_residuals: optional, per-layer alignment residual ratios
            (from align.align_activations_per_layer). Included in the
            report so the reader can judge alignment quality alongside
            post-alignment distances.
        name_A, name_B: human-readable names for the printable summary.

    Returns:
        dict with metric keys.
    """
    report = {}

    # Basis-invariant (Frame 1).
    report["sv_distance_per_layer"] = singular_value_distance(
        flow_A, flow_B, per_layer=True
    )
    report["sv_distance_total"] = float(report["sv_distance_per_layer"].sum())
    report["lambda_distance"] = lambda_distance(flow_A, flow_B)
    report["eff_rank_distance_per_layer"] = effective_rank_distance(
        flow_A, flow_B, per_layer=True
    )
    report["eff_rank_distance_total"] = float(
        report["eff_rank_distance_per_layer"].sum()
    )
    gauss = gaussianity_distance(flow_A, flow_B)
    report.update(gauss)

    # Curve-level (Frame 2).
    report["A_total_angular_distance"] = total_angular_distance(flow_A["R"])
    report["B_total_angular_distance"] = total_angular_distance(flow_B["R"])
    report["angular_distance_diff"] = abs(
        report["A_total_angular_distance"] - report["B_total_angular_distance"]
    )
    report["reparam_sv_distance"] = reparameterized_singular_value_distance(
        flow_A, flow_B
    )

    # Aligned (if available).
    if R_A_transported is not None:
        report["aligned_R_distance_per_layer"] = aligned_R_distance(
            R_A_transported, flow_B["R"], per_layer=True
        )
        report["aligned_R_distance_total"] = float(
            report["aligned_R_distance_per_layer"].sum()
        )
        report["principal_angle_profile"] = principal_angle_profile(
            R_A_transported, flow_B["R"], top_k=10
        )
        report["principal_angle_mean"] = float(
            report["principal_angle_profile"].mean()
        )

    if alignment_residuals is not None:
        report["alignment_residuals"] = list(alignment_residuals)
        report["alignment_residual_mean"] = float(
            np.mean(alignment_residuals)
        )

    # Printable summary.
    lines = [
        f"Comparison: {name_A} vs {name_B}",
        "=" * 60,
        "BASIS-INVARIANT (Frame 1):",
        f"  Σ-spectrum L2 distance (log space, summed over layers): "
        f"{report['sv_distance_total']:.4f}",
        f"  λ (variance scaling rate) difference: "
        f"{report['lambda_distance']:.4f}",
        f"  Effective rank total difference: "
        f"{report['eff_rank_distance_total']:.3f}",
        f"  Kurtosis diff (L1): {report['kurtosis_diff']:.3f}",
        f"  Isotropy diff (L1): {report['isotropy_diff']:.3f}",
        "",
        "CURVE-LEVEL (Frame 2):",
        f"  Total angular distance — {name_A}: "
        f"{report['A_total_angular_distance']:.2f}°  "
        f"{name_B}: {report['B_total_angular_distance']:.2f}°  "
        f"(diff: {report['angular_distance_diff']:.2f}°)",
        f"  Reparameterized Σ distance (τ ∈ [0,1]): "
        f"{report['reparam_sv_distance']:.4f}",
    ]
    if "aligned_R_distance_total" in report:
        lines += [
            "",
            "ALIGNED (Frame 1 + alignment):",
            f"  R Frobenius distance (post-alignment): "
            f"{report['aligned_R_distance_total']:.3f}",
            f"  Mean principal angle (top-10 directions): "
            f"{report['principal_angle_mean']:.2f}°",
        ]
    if "alignment_residual_mean" in report:
        lines.append(
            f"  Alignment residual (mean across layers): "
            f"{report['alignment_residual_mean']:.4f}"
        )
    report["summary"] = "\n".join(lines)

    return report
