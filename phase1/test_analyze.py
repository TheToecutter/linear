"""
Tests for analyze.py.

These tests use synthetic data with known properties to verify that
recover_linear_flow() produces the right answers.

Run:  python3 test_analyze.py
"""

import sys
import numpy as np
import tempfile
import os

from analyze import (
    recover_linear_flow, save_flow, load_flow, default_pilot_positions,
)


# ----------------------------------------------------------------------
# Synthetic data generators.
# ----------------------------------------------------------------------
def make_pure_linear_trajectories(L=8, N=2000, H=32, lambda_true=0.2,
                                    log_alpha_true=0.5, rng_seed=0,
                                    signal_noise_ratio=10.0,
                                    rotation_angle_per_layer=0.05):
    """
    Generate L layers of activations consistent with the lines-of-thought
    model:

      x(t+1) = R(t+1) Λ(t,1) R(t)^T x(t) + ε,   ε ~ N(0, exp(log_α + λ(t+1)) I)

    Returns (activations, R_true, sigma_true_per_layer).

    Important: the lines-of-thought analysis assumes that consecutive
    layers' principal direction bases R(t) and R(t+1) are CLOSE to each
    other (the paper's Fig 2(a) shows this empirically — angles between
    axis-i at successive layers are small). The analyzer's element-wise
    scaling Λ_ii = σ_i(t+1)/σ_i(t) only makes sense when axis-i at layer
    t corresponds to axis-i at layer t+1.

    We enforce this property here by generating R(t+1) as a small
    perturbation of R(t): apply a random rotation of angle
    `rotation_angle_per_layer` (in radians, on a random 2-plane) to
    each row of R(t) to get R(t+1). This produces smoothly-evolving
    bases like trained transformers have.

    With signal_noise_ratio=10 and rotation_angle_per_layer=0.05 (small
    rotations), the analyzer should recover λ, log α, and the singular
    value spectra accurately.
    """
    rng = np.random.default_rng(rng_seed)
    R_true = np.zeros((L, H, H), dtype=np.float32)
    # R(0): random orthogonal.
    M0 = rng.standard_normal((H, H)).astype(np.float32)
    Q0, _ = np.linalg.qr(M0)
    R_true[0] = Q0.T  # rows are principal directions

    # R(t+1) = R(t) @ G  where G is a small rotation (close to identity).
    # We construct G as exp(skew) with skew a random skew-symmetric matrix
    # of small norm.
    for t in range(L - 1):
        # Random skew-symmetric matrix of controlled magnitude.
        S = rng.standard_normal((H, H)).astype(np.float32)
        S = (S - S.T) / 2.0
        S *= rotation_angle_per_layer / np.linalg.norm(S, ord="fro") * H
        # Cayley transform gives a small rotation matrix: G = (I - S/2)(I + S/2)^-1.
        # Equivalent to exp(S) for small S, but cheaper.
        I = np.eye(H, dtype=np.float32)
        G = np.linalg.solve(I + S / 2.0, I - S / 2.0)
        R_true[t + 1] = R_true[t] @ G  # rotated rows
        # QR re-orthogonalize for numerical safety.
        Q, _ = np.linalg.qr(R_true[t + 1].T)
        R_true[t + 1] = Q.T

    # Per-layer scale factors (singular values), growing as exp(0.05*t).
    max_noise_var = np.exp(log_alpha_true + lambda_true * (L - 1))
    min_signal_var = signal_noise_ratio * max_noise_var
    sigma_base = np.geomspace(
        np.sqrt(min_signal_var) * 10.0,
        np.sqrt(min_signal_var),
        H, dtype=np.float32,
    )
    sigma_per_layer = np.zeros((L, H), dtype=np.float32)
    for t in range(L):
        sigma_per_layer[t] = sigma_base * np.exp(0.05 * t)

    # Generate x_0 as Gaussian in the PC basis of R(0), then transform.
    # PC coordinates of x_0 should have variance ~ sigma_per_layer[0]^2.
    pc_0 = rng.standard_normal((N, H)).astype(np.float32) * sigma_per_layer[0]
    x_0 = pc_0 @ R_true[0]  # transform to original basis

    activations = np.zeros((L, N, H), dtype=np.float32)
    activations[0] = x_0

    for t in range(L - 1):
        # Linear prediction.
        pc_t = activations[t] @ R_true[t].T  # PC coords in R(t) basis
        scale = sigma_per_layer[t + 1] / np.maximum(sigma_per_layer[t], 1e-12)
        pc_pred = pc_t * scale[None, :]
        x_pred = pc_pred @ R_true[t + 1]
        # Add noise. The variance per-coordinate should be α exp(λ (t+1)).
        noise_var = np.exp(log_alpha_true + lambda_true * (t + 1))
        noise = rng.standard_normal((N, H)).astype(np.float32) * np.sqrt(noise_var)
        activations[t + 1] = x_pred + noise

    return activations, R_true, sigma_per_layer


# ----------------------------------------------------------------------
# Tests.
# ----------------------------------------------------------------------
def test_singular_values_recover_known_spectrum():
    """When activations have a known singular value spectrum, the recovered
    singular values should match."""
    print("test_singular_values_recover_known_spectrum ... ", end="")
    L, N, H = 4, 5000, 16
    activations, _, sigma_true = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=0)
    flow = recover_linear_flow(activations, center=True)
    sigma_rec = flow["singular_values"]  # (L, H)
    # The recovered singular values should match the true singular values
    # to within sampling noise (~ 1/sqrt(N) of the values themselves).
    for t in range(L):
        # Recovered should be sorted descending; the true ones are too.
        rel_err = np.abs(sigma_rec[t] - sigma_true[t]) / np.maximum(sigma_true[t], 1e-9)
        max_err = rel_err.max()
        assert max_err < 0.1, (
            f"Singular values at layer {t} differ by max {max_err:.3f} "
            f"(relative); too much given N={N}"
        )
    print("OK")


def test_lambda_recovery():
    """When we generate trajectories with variance ~ exp(λ (t+τ)) injected at
    each step, the recovered λ measures the *envelope* of accumulated noise
    across (t, t+τ) pairs, not the per-step λ. With smoothly-evolving Λ ≈
    exp(0.05 I) per layer, the recovered λ should be of the same sign as
    the true λ but may differ in magnitude.

    For this test we just check that λ is in the same ballpark."""
    print("test_lambda_recovery ... ", end="")
    L, N, H = 10, 5000, 16
    lambda_true = 0.25
    log_alpha_true = -1.0
    activations, _, _ = make_pure_linear_trajectories(
        L=L, N=N, H=H, lambda_true=lambda_true,
        log_alpha_true=log_alpha_true, rng_seed=42,
        signal_noise_ratio=100.0,
    )
    flow = recover_linear_flow(activations, center=True)
    lam_rec = flow["lambda"]
    log_alpha_rec = flow["log_alpha"]
    # λ should be positive (variance grows with depth).
    assert lam_rec > 0.0, (
        f"λ should be positive (variance grows with depth); recovered {lam_rec}"
    )
    # Sanity check: recovered λ within 4× of true (loose; envelope effects).
    assert 0.25 * lambda_true < lam_rec < 4.0 * lambda_true, (
        f"λ recovery too far from truth: "
        f"true={lambda_true}, recovered={lam_rec:.4f}"
    )
    print(f"OK (λ={lam_rec:.3f}; envelope of true λ={lambda_true})")


def test_residuals_are_finite_and_positive():
    """Residual variances should be positive and finite, with the diagonal
    (τ=0) NaN and entries below the diagonal (τ<0) NaN."""
    print("test_residuals_are_finite_and_positive ... ", end="")
    L, N, H = 6, 2000, 16
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=1)
    flow = recover_linear_flow(activations, center=True)
    pwv = flow["pairwise_residual_variance"]
    # Upper triangle (t < target) should be finite positive.
    for t in range(L):
        for target in range(t + 1, L):
            assert np.isfinite(pwv[t, target]), (
                f"pairwise_residual_variance[{t}, {target}] is not finite"
            )
            assert pwv[t, target] > 0, (
                f"pairwise_residual_variance[{t}, {target}] = {pwv[t, target]} "
                f"is non-positive"
            )
    # Lower triangle and diagonal should be NaN.
    for t in range(L):
        for target in range(0, t + 1):
            assert np.isnan(pwv[t, target]), (
                f"pairwise_residual_variance[{t}, {target}] = {pwv[t, target]} "
                f"should be NaN"
            )
    print("OK")


def test_effective_rank_at_most_H():
    """Effective rank can't exceed H. And for high-rank synthetic data,
    it should be close to H (since we generate non-degenerate spectra)."""
    print("test_effective_rank_at_most_H ... ", end="")
    L, N, H = 4, 3000, 16
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=2)
    flow = recover_linear_flow(activations, center=True)
    eff_rank = flow["effective_rank"]
    for t in range(L):
        assert eff_rank[t] <= H + 1e-3, (
            f"Effective rank {eff_rank[t]} > H={H} at layer {t}"
        )
        # For our power-law-decaying spectrum (1.0 down to 0.1), effective
        # rank should be substantial but not equal to H. Looser bound: > 2.
        assert eff_rank[t] > 2.0, (
            f"Effective rank {eff_rank[t]} suspiciously low at layer {t}"
        )
    print(f"OK (eff_rank range: [{eff_rank.min():.1f}, {eff_rank.max():.1f}], H={H})")


def test_kurtosis_finite_for_gaussian_noise():
    """When residuals are pure Gaussian, excess kurtosis should be near 0."""
    print("test_kurtosis_finite_for_gaussian_noise ... ", end="")
    L, N, H = 6, 10000, 16
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=3)
    flow = recover_linear_flow(activations, center=True)
    kurt = flow["kurtosis_per_layer"]
    # Layer 0 has no residuals (it's the source); should be NaN.
    assert np.isnan(kurt[0])
    # Other layers should have small excess kurtosis (within 0.2 of zero
    # for N=10000 pilots).
    for t in range(1, L):
        assert abs(kurt[t]) < 0.3, (
            f"Excess kurtosis {kurt[t]:.3f} at layer {t} suggests "
            f"non-Gaussianity, but we generated Gaussian noise"
        )
    print(f"OK (kurt range: [{np.nanmin(kurt):.3f}, {np.nanmax(kurt):.3f}])")


def test_isotropy_small_for_isotropic_noise():
    """When residuals are isotropic (same variance per dim), the isotropy
    metric (std of log per-dim variance) should be small."""
    print("test_isotropy_small_for_isotropic_noise ... ", end="")
    L, N, H = 6, 10000, 16
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=4)
    flow = recover_linear_flow(activations, center=True)
    iso = flow["isotropy_per_layer"]
    for t in range(1, L):
        # log-variance std should be small (< 0.3) for isotropic Gaussian noise.
        assert iso[t] < 0.3, (
            f"Isotropy metric {iso[t]:.3f} at layer {t} is large despite "
            f"isotropic noise"
        )
    print(f"OK (isotropy range: [{np.nanmin(iso[1:]):.3f}, {np.nanmax(iso[1:]):.3f}])")


def test_save_and_load_flow():
    """Round-trip save/load should preserve all fields, including the
    paper-convention fields added alongside 'ours' variants."""
    print("test_save_and_load_flow ... ", end="")
    L, N, H = 4, 1000, 8
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=5)
    flow = recover_linear_flow(activations, center=True)
    # Augment with metadata that save_flow expects.
    flow["checkpoint_step"] = 12345
    flow["checkpoint_path"] = "/fake/path.pt"
    flow["checkpoint_loss"] = 3.14
    flow["checkpoint_eval_loss"] = 2.71
    flow["checkpoint_seed"] = 7
    flow["pilot_positions"] = [50, 100, 150]
    flow["analysis_time_sec"] = 42.0

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_flow(flow, path)
        loaded = load_flow(path)
        # Spot-check fields.
        assert loaded["checkpoint_step"] == 12345
        assert loaded["checkpoint_seed"] == 7
        assert abs(loaded["lambda"] - flow["lambda"]) < 1e-7
        assert abs(loaded["lambda_paper"] - flow["lambda_paper"]) < 1e-7
        assert abs(loaded["log_alpha_paper"] - flow["log_alpha_paper"]) < 1e-7
        assert np.allclose(loaded["R"], flow["R"])
        assert np.allclose(loaded["singular_values"], flow["singular_values"])
        assert np.allclose(
            loaded["pairwise_mean_log_var"], flow["pairwise_mean_log_var"],
            equal_nan=True,
        )
        assert np.allclose(
            loaded["kurtosis_abs_per_layer"], flow["kurtosis_abs_per_layer"],
            equal_nan=True,
        )
        # NaNs should round-trip.
        assert np.array_equal(
            np.isnan(loaded["pairwise_residual_variance"]),
            np.isnan(flow["pairwise_residual_variance"]),
        )
    finally:
        os.unlink(path)
    print("OK")


def test_paper_and_ours_conventions_both_present():
    """Both 'ours' (log-of-mean) and 'paper' (mean-of-log) conventions
    should be computed in a single call, with the Jensen-inequality
    relation mean_d log(var_d) ≤ log(mean_d var_d) holding per (source,
    target) pair where both are finite."""
    print("test_paper_and_ours_conventions_both_present ... ", end="")
    L, N, H = 6, 3000, 16
    activations, _, _ = make_pure_linear_trajectories(
        L=L, N=N, H=H, rng_seed=11, signal_noise_ratio=50.0,
    )
    flow = recover_linear_flow(activations, center=True)

    # Both scalar fits exist and are finite.
    for key in ("lambda", "lambda_paper", "log_alpha", "log_alpha_paper"):
        assert key in flow, f"Missing field {key!r}"
        assert np.isfinite(flow[key]), f"{key} = {flow[key]} is non-finite"

    # Pairwise: per-pair Jensen inequality.
    pwv = flow["pairwise_residual_variance"]
    pmlv = flow["pairwise_mean_log_var"]
    assert pwv.shape == pmlv.shape, (
        f"shape mismatch: pwv={pwv.shape} vs pmlv={pmlv.shape}"
    )
    for t in range(L):
        for target in range(t + 1, L):
            v = pwv[t, target]
            mlv = pmlv[t, target]
            assert np.isfinite(v) and np.isfinite(mlv)
            # mean_d log(var_d) ≤ log(mean_d var_d), with equality iff
            # all var_d equal. For synthetic non-isotropic residuals
            # there should be a strict (small) gap.
            log_v = np.log(v)
            assert mlv <= log_v + 1e-4, (
                f"Jensen violated at (t={t}, target={target}): "
                f"mean_d log(var_d) = {mlv:.4f} > log(mean_d var_d) = {log_v:.4f}"
            )

    # Endpoint-aggregated fits: same direction.
    elv = flow["endpoint_log_var"]
    emlv = flow["endpoint_mean_log_var"]
    assert elv.shape == emlv.shape
    assert np.all(emlv <= elv + 1e-4), (
        f"endpoint mean_log_var should be ≤ endpoint log_var elementwise"
    )
    # The two fit slopes can differ but should both be positive on this
    # variance-growing synthetic data.
    assert flow["lambda_paper"] > 0, (
        f"paper λ should be positive on variance-growing data; "
        f"got {flow['lambda_paper']}"
    )
    print(f"OK (log_α: ours={flow['log_alpha']:+.3f} paper={flow['log_alpha_paper']:+.3f}, "
          f"λ: ours={flow['lambda']:.3f} paper={flow['lambda_paper']:.3f})")


def test_kurtosis_abs_is_nonnegative_and_bounds_signed_mean():
    """The paper convention mean_d |κᵢ| must be non-negative and must
    bound the absolute value of the signed mean mean_d κᵢ — by the
    triangle inequality |E[X]| ≤ E[|X|]."""
    print("test_kurtosis_abs_is_nonnegative_and_bounds_signed_mean ... ", end="")
    L, N, H = 6, 5000, 16
    activations, _, _ = make_pure_linear_trajectories(L=L, N=N, H=H, rng_seed=13)
    flow = recover_linear_flow(activations, center=True)
    kurt = flow["kurtosis_per_layer"]            # signed mean
    kurt_abs = flow["kurtosis_abs_per_layer"]    # mean of absolute values
    # Source-only layer 0 is NaN in both.
    assert np.isnan(kurt[0]) and np.isnan(kurt_abs[0])
    # For t ≥ 1: mean of absolutes is ≥ |signed mean| and ≥ 0.
    for t in range(1, L):
        assert kurt_abs[t] >= 0.0, (
            f"<|κ|> at layer {t} = {kurt_abs[t]:.4f} is negative"
        )
        assert kurt_abs[t] >= abs(kurt[t]) - 1e-5, (
            f"<|κ|> at layer {t} = {kurt_abs[t]:.4f} should be ≥ "
            f"|<κ>| = {abs(kurt[t]):.4f} (triangle inequality)"
        )
    print(f"OK (<κ> range: [{np.nanmin(kurt):+.3f}, {np.nanmax(kurt):+.3f}], "
          f"<|κ|> range: [{np.nanmin(kurt_abs):.3f}, {np.nanmax(kurt_abs):.3f}])")


def test_default_pilot_positions():
    """Sanity check the pilot position picker."""
    print("test_default_pilot_positions ... ", end="")
    pos = default_pilot_positions(seq_len=1024, stride=50, start=50)
    assert len(pos) > 0
    assert pos[0] == 50
    assert all(pos[i+1] - pos[i] == 50 for i in range(len(pos) - 1))
    assert all(p < 1024 - 50 for p in pos), (
        f"Last position {pos[-1]} too close to seq end 1024"
    )
    # Should be reasonable count.
    assert 15 <= len(pos) <= 25, f"Got {len(pos)} positions, expected 15-25"
    print(f"OK ({len(pos)} positions: {pos[:3]}...{pos[-3:]})")


def test_recovery_with_random_noise_only():
    """If all activations are independent random noise (no trajectory
    structure), the residuals should be roughly the same as the activation
    variance (predicting from one random pile of noise to another gives
    huge residuals)."""
    print("test_recovery_with_random_noise_only ... ", end="")
    L, N, H = 4, 2000, 16
    rng = np.random.default_rng(123)
    activations = rng.standard_normal((L, N, H)).astype(np.float32) * 2.0
    flow = recover_linear_flow(activations, center=True)
    # Linear flow should give a poor prediction here — residual variance
    # should be on the order of the activation variance (~4 = 2²) or larger.
    pwv = flow["pairwise_residual_variance"]
    for t in range(L):
        for target in range(t + 1, L):
            # Residual variance per dim should be at least ~1 (close to true 4).
            # We just check it's not tiny.
            assert pwv[t, target] > 0.5, (
                f"Linear flow on pure noise predicted too well: "
                f"residual variance {pwv[t, target]:.3f} too small"
            )
    print("OK")


def main():
    tests = [
        test_default_pilot_positions,
        test_singular_values_recover_known_spectrum,
        test_lambda_recovery,
        test_residuals_are_finite_and_positive,
        test_effective_rank_at_most_H,
        test_kurtosis_finite_for_gaussian_noise,
        test_isotropy_small_for_isotropic_noise,
        test_recovery_with_random_noise_only,
        test_save_and_load_flow,
        test_paper_and_ours_conventions_both_present,
        test_kurtosis_abs_is_nonnegative_and_bounds_signed_mean,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL\n   {e}")
            failures.append((fn.__name__, str(e)))
        except Exception as e:
            print(f"ERROR\n   {type(e).__name__}: {e}")
            failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
    print()
    if failures:
        print(f"❌ {len(failures)}/{len(tests)} tests failed:")
        for name, msg in failures:
            print(f"   - {name}: {msg}")
        sys.exit(1)
    print(f"✅ All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
    