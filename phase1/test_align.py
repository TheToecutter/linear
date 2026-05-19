"""
Tests for align.py.

Run:  python3 test_align.py
"""

import sys
import numpy as np

from align import (
    orthogonal_procrustes, align_embeddings,
    align_activations_per_layer, transport_R, transport_R_per_layer,
)


def random_orthogonal(H, rng):
    """Generate a uniformly-random orthogonal H×H matrix via QR of Gaussian."""
    M = rng.standard_normal((H, H)).astype(np.float64)
    Q, _ = np.linalg.qr(M)
    # QR can give det -1; for proper rotation we'd flip but we allow both.
    return Q.astype(np.float32)


def test_procrustes_recovers_known_rotation():
    """If Y = X @ Q_true, Procrustes should recover Q_true exactly."""
    print("test_procrustes_recovers_known_rotation ... ", end="")
    rng = np.random.default_rng(0)
    N, H = 200, 16
    X = rng.standard_normal((N, H)).astype(np.float32)
    Q_true = random_orthogonal(H, rng)
    Y = X @ Q_true
    Q_recovered, residual_ratio = orthogonal_procrustes(X, Y)
    # Q should equal Q_true within fp32 precision.
    err = np.linalg.norm(Q_recovered - Q_true)
    assert err < 1e-4, f"||Q_recovered - Q_true|| = {err}, too large"
    # Residual should be near-zero (Y is exactly X @ Q_true).
    assert residual_ratio < 1e-5, f"Residual ratio = {residual_ratio}"
    print(f"OK (err = {err:.2e}, residual = {residual_ratio:.2e})")


def test_procrustes_handles_noise():
    """With Y = X @ Q_true + small noise, Procrustes still recovers Q_true approximately."""
    print("test_procrustes_handles_noise ... ", end="")
    rng = np.random.default_rng(1)
    N, H = 500, 16
    X = rng.standard_normal((N, H)).astype(np.float32)
    Q_true = random_orthogonal(H, rng)
    noise_level = 0.01
    Y = X @ Q_true + rng.standard_normal((N, H)).astype(np.float32) * noise_level
    Q_recovered, residual_ratio = orthogonal_procrustes(X, Y)
    err = np.linalg.norm(Q_recovered - Q_true)
    # With small noise, error should be small (but not zero).
    assert err < 0.1, f"||Q_recovered - Q_true|| = {err}, too large for noise level {noise_level}"
    # Residual should be roughly noise_level × sqrt(N×H) / ||Y||.
    # For X ~ N(0,1) and Y ≈ X @ Q + noise, ||Y||² ≈ N×H so residual_ratio ≈ noise_level.
    assert residual_ratio < 0.05, f"Residual ratio {residual_ratio} too large"
    print(f"OK (err = {err:.3f}, residual = {residual_ratio:.3f})")


def test_procrustes_residual_for_unrelated_data():
    """Unrelated X and Y should give a large residual (Q can do something but not much)."""
    print("test_procrustes_residual_for_unrelated_data ... ", end="")
    rng = np.random.default_rng(2)
    N, H = 500, 16
    X = rng.standard_normal((N, H)).astype(np.float32)
    Y = rng.standard_normal((N, H)).astype(np.float32)  # totally unrelated
    Q, residual_ratio = orthogonal_procrustes(X, Y)
    # Q is still orthogonal.
    QQ = Q @ Q.T
    assert np.allclose(QQ, np.eye(H), atol=1e-4), (
        f"Q not orthogonal: ||QQ.T - I||_F = {np.linalg.norm(QQ - np.eye(H))}"
    )
    # Residual should be near 1.0 (independent random data).
    # Specifically, ||X @ Q - Y||² / ||Y||² ≈ (||X||² + ||Y||²) / ||Y||² ≈ 2 (without alignment),
    # and Procrustes finds Q that reduces this to roughly ||Y - mean_aligned||² / ||Y||² ≈ 1
    # for independent fresh data.
    assert 0.8 < residual_ratio < 1.5, (
        f"Residual {residual_ratio} unexpected for unrelated data (should be ~1)"
    )
    print(f"OK (residual = {residual_ratio:.3f}, expected ~1.0)")


def test_align_embeddings():
    """Embedding-space alignment between two embedding matrices that differ
    only by a known rotation."""
    print("test_align_embeddings ... ", end="")
    rng = np.random.default_rng(3)
    V, H = 1000, 32
    E_A = rng.standard_normal((V, H)).astype(np.float32)
    Q_true = random_orthogonal(H, rng)
    E_B = E_A @ Q_true
    Q_recovered, residual = align_embeddings(E_A, E_B)
    err = np.linalg.norm(Q_recovered - Q_true)
    assert err < 1e-4, f"Q recovery error {err}"
    assert residual < 1e-5, f"Residual {residual} should be near zero"
    print(f"OK (err = {err:.2e}, residual = {residual:.2e})")


def test_transport_R_inverts_rotation():
    """Transporting R through Q twice (forward and back) should give R back."""
    print("test_transport_R_inverts_rotation ... ", end="")
    rng = np.random.default_rng(4)
    H = 16
    Q = random_orthogonal(H, rng)
    R = random_orthogonal(H, rng)
    # Forward transport.
    R_transported = transport_R(R, Q, Q)  # Q.T @ R @ Q
    # Reverse transport with Q.T (which is the inverse of Q).
    R_back = transport_R(R_transported, Q.T, Q.T)  # Q @ (Q.T R Q) @ Q.T = R
    err = np.linalg.norm(R_back - R)
    assert err < 1e-4, f"Round-trip error {err} (should be near zero)"
    print(f"OK (round-trip err = {err:.2e})")


def test_align_per_layer_recovers_layer_rotations():
    """Per-layer alignment should recover the per-layer rotations applied
    to a known activation tensor."""
    print("test_align_per_layer_recovers_layer_rotations ... ", end="")
    rng = np.random.default_rng(5)
    L, N, H = 5, 200, 16
    activations_A = rng.standard_normal((L, N, H)).astype(np.float32)
    # Apply a different known rotation per layer.
    Q_trues = [random_orthogonal(H, rng) for _ in range(L)]
    activations_B = np.stack(
        [activations_A[t] @ Q_trues[t] for t in range(L)], axis=0
    )
    Qs_recovered, residuals = align_activations_per_layer(
        activations_A, activations_B,
    )
    for t in range(L):
        err = np.linalg.norm(Qs_recovered[t] - Q_trues[t])
        assert err < 1e-3, f"Layer {t}: Q recovery error {err}"
        assert residuals[t] < 1e-4, f"Layer {t}: residual {residuals[t]}"
    print("OK (all layers recovered)")


def test_alignment_residual_grows_with_added_noise():
    """As we add more noise to the alignment, the residual should grow."""
    print("test_alignment_residual_grows_with_added_noise ... ", end="")
    rng = np.random.default_rng(6)
    N, H = 500, 16
    X = rng.standard_normal((N, H)).astype(np.float32)
    Q_true = random_orthogonal(H, rng)
    residuals_seen = []
    for noise_level in [0.0, 0.01, 0.1, 0.5, 1.0]:
        Y = X @ Q_true + rng.standard_normal((N, H)).astype(np.float32) * noise_level
        _, r = orthogonal_procrustes(X, Y)
        residuals_seen.append(r)
    # Residuals should be monotonically non-decreasing.
    for i in range(len(residuals_seen) - 1):
        assert residuals_seen[i] <= residuals_seen[i + 1] + 0.05, (
            f"Residual decreased with more noise: {residuals_seen}"
        )
    print(f"OK (residuals: {[f'{r:.3f}' for r in residuals_seen]})")


def test_procrustes_with_dim_mismatch_raises():
    """Mismatched dimensions should raise AssertionError."""
    print("test_procrustes_with_dim_mismatch_raises ... ", end="")
    X = np.random.randn(100, 16).astype(np.float32)
    Y = np.random.randn(100, 8).astype(np.float32)
    raised = False
    try:
        orthogonal_procrustes(X, Y)
    except AssertionError:
        raised = True
    assert raised
    print("OK")


def main():
    tests = [
        test_procrustes_recovers_known_rotation,
        test_procrustes_handles_noise,
        test_procrustes_residual_for_unrelated_data,
        test_align_embeddings,
        test_transport_R_inverts_rotation,
        test_align_per_layer_recovers_layer_rotations,
        test_alignment_residual_grows_with_added_noise,
        test_procrustes_with_dim_mismatch_raises,
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
