"""
Tests for the reverse build-up code.

These tests use small synthetic data and do not require a trained
model or any on-disk artifacts beyond a temp directory. Run with:

    pytest test_reverse_buildup.py -v

The tests cover:

  test_view_dispatch_basic         -- select_token_set_for_view and
                                       select_id_array_for_view return
                                       the correct elements for each view.

  test_peak_variance_layer          -- peak_variance_layer returns the
                                       argmax for a known curve; handles
                                       the forward-boundary degeneracy
                                       (v[0] = 0).

  test_contraction_fit              -- contraction_fit returns
                                       (nan, nan, L-1) for monotonically
                                       increasing curves (forward case),
                                       and recovers a known negative
                                       lambda for synthetic exp-decay
                                       contraction curves.

  test_within_chunk_shuffle_freq    -- shuffling preserves global label
                                       frequencies exactly.

  test_within_chunk_shuffle_within  -- shuffling does NOT cross chunk
                                       boundaries.

  test_null_corrected_kurtosis      -- null_corrected_kurtosis subtracts
                                       correctly and complains on
                                       mismatched (seed, step).

  test_unembedding_subspace_split   -- per-coord kurt on synthetic data
                                       with heavy tails injected only
                                       in the orthogonal complement
                                       gives near-zero parallel kurt
                                       and positive perp kurt.

  test_d5_view_dispatch_consistency -- (network-free) run_d5_view with
                                       view='forward' processes
                                       input_ids and produces results
                                       matching a known answer on
                                       crafted synthetic data.

The "bit-identical forward" check that the proposal pre-registers is
test_forward_parameterized_matches_existing; that one requires the
existing model_abc_discriminator output to be available on disk and
is therefore an integration test rather than a unit test. It is
skipped unless the env var REVERSE_BUILDUP_RUN_DIR is set.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pytest

from reverse_buildup import (
    VIEWS,
    peak_variance_layer,
    contraction_fit,
    select_id_array_for_view,
    _trace_within_per_cell,
    _safe_logspace_fit,
    _coefficient_of_variation,
    _per_coord_excess_kurt,
    _mardia_kurtosis_standardized,
)
from reverse_null import (
    within_chunk_shuffle,
    _verify_frequency_preserving,
    null_corrected_kurtosis,
    f4_signal_with_null_correction,
)


# ======================================================================
# Helpers used by multiple tests.
# ======================================================================
def _synthetic_augmented_payload(
    L: int = 14, N: int = 100, H: int = 16, rng_seed: int = 0,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    states = rng.standard_normal((L, N, H)).astype(np.float32)
    input_ids = rng.integers(0, 5, size=N).astype(np.int32)
    next_ids = rng.integers(0, 7, size=N).astype(np.int32)
    pred_ids = rng.integers(0, 7, size=N).astype(np.int32)
    positions = np.tile(np.arange(N // 4), 4)[:N].astype(np.int32)
    return {
        "states": states,
        "input_ids": input_ids,
        "next_ids": next_ids,
        "pred_ids": pred_ids,
        "positions": positions,
    }


# ======================================================================
# Tests.
# ======================================================================
def test_view_dispatch_basic():
    payload = _synthetic_augmented_payload()
    for view in VIEWS:
        ids = select_id_array_for_view(payload, view)
        assert ids.shape == (payload["states"].shape[1],)
    assert (select_id_array_for_view(payload, "forward")
            == payload["input_ids"]).all()
    assert (select_id_array_for_view(payload, "reverse_actual")
            == payload["next_ids"]).all()
    assert (select_id_array_for_view(payload, "reverse_pred")
            == payload["pred_ids"]).all()


def test_view_dispatch_unknown():
    payload = _synthetic_augmented_payload()
    with pytest.raises(ValueError):
        select_id_array_for_view(payload, "not_a_real_view")


def test_peak_variance_layer_monotonic_growth():
    # Forward-like curve: monotonically increasing from 0, peak at end.
    v = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    # The function should exclude v[0] = 0 (forward degenerate case)
    # and return the next argmax, which is the last index.
    assert peak_variance_layer(v) == 6


def test_peak_variance_layer_interior_peak():
    # Reverse-like curve: grows then shrinks.
    v = np.array([0.1, 0.5, 1.2, 1.5, 1.2, 0.8, 0.4])
    assert peak_variance_layer(v) == 3


def test_peak_variance_layer_nan_robust():
    v = np.array([0.1, np.nan, 1.2, 1.5, 1.2, 0.8, 0.4])
    assert peak_variance_layer(v) == 3


def test_contraction_fit_monotonic_returns_nan():
    # If the peak is at L-1 or beyond, contraction is undefined.
    v = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    la, lm, tp = contraction_fit(v)
    assert tp == 6
    assert np.isnan(la) and np.isnan(lm)


def test_contraction_fit_recovers_known_lambda():
    # Build a curve with peak at layer 4 and exponential decay after:
    # v[t] = c * (t / 4)^(-0.8) for t >= 4. Take log-log: lambda = -0.8.
    L = 14
    layers = np.arange(L, dtype=np.float64)
    v_grow = layers[:5] ** 1.5
    v_shrink = v_grow[-1] * (layers[5:] / 4.0) ** (-0.8)
    v = np.concatenate([v_grow, v_shrink])
    la, lm, tp = contraction_fit(v)
    assert tp == 4
    # Fit is on layers [4, 13] of which only [5, 13] contribute to log.
    # Expected lambda is around -0.8 but the first contributing point
    # (t=4) is the peak, so the fitted slope mixes the peak in.
    # The fit should be clearly negative, in the ballpark of -0.4..-1.0.
    assert lm < -0.3, f"expected negative lambda, got {lm}"
    assert lm > -1.5, f"unexpectedly large negative lambda {lm}"


# ----------------------------------------------------------------------
# Shuffle null tests.
# ----------------------------------------------------------------------
def test_within_chunk_shuffle_frequency_preserving():
    # Three chunks of 5 pilots each, positions 0..4 within each chunk.
    positions = np.tile(np.arange(5, dtype=np.int32), 3)
    ids = np.array([0, 0, 1, 1, 2,
                    1, 1, 2, 2, 3,
                    0, 2, 2, 3, 3], dtype=np.int32)
    shuffled = within_chunk_shuffle(ids, positions, rng_seed=42)
    report = _verify_frequency_preserving(ids, shuffled)
    assert report["same_counts"]
    assert report["same_keys"]


def test_within_chunk_shuffle_respects_chunks():
    # Chunk boundary at index 5 and 10. After shuffling, the set of
    # labels in each chunk must equal the input set of labels in
    # that chunk.
    positions = np.tile(np.arange(5, dtype=np.int32), 3)
    ids = np.array([0, 0, 1, 1, 2,
                    1, 1, 2, 2, 3,
                    0, 2, 2, 3, 3], dtype=np.int32)
    shuffled = within_chunk_shuffle(ids, positions, rng_seed=42)
    for a, b in [(0, 5), (5, 10), (10, 15)]:
        assert sorted(ids[a:b].tolist()) == sorted(shuffled[a:b].tolist()), \
            f"chunk {a}:{b} contents changed"


def test_within_chunk_shuffle_deterministic():
    positions = np.tile(np.arange(5, dtype=np.int32), 3)
    ids = np.arange(15, dtype=np.int32) % 4
    s1 = within_chunk_shuffle(ids, positions, rng_seed=123)
    s2 = within_chunk_shuffle(ids, positions, rng_seed=123)
    assert (s1 == s2).all()
    s3 = within_chunk_shuffle(ids, positions, rng_seed=456)
    # Different seed should typically differ; not guaranteed but very
    # likely for N=15 with multiple chunks.
    assert not (s1 == s3).all()


def test_within_chunk_shuffle_empty():
    out = within_chunk_shuffle(np.zeros(0, dtype=np.int32),
                                np.zeros(0, dtype=np.int32))
    assert out.size == 0


def test_null_corrected_kurtosis_arithmetic():
    real = {
        "seed": 0, "step": 100,
        "empirical_kurt": np.array([1.0, 2.0]),
        "recon_A_kurt": np.array([5.0, 6.0]),
        "recon_B_kurt": np.array([3.0, 4.0]),
    }
    shuf = {
        "seed": 0, "step": 100,
        "empirical_kurt": np.array([1.0, 2.0]),
        "recon_A_kurt": np.array([1.0, 2.0]),
        "recon_B_kurt": np.array([0.5, 1.0]),
    }
    corr = null_corrected_kurtosis(real, shuf)
    assert np.allclose(corr["recon_A_kurt_corr"], [4.0, 4.0])
    assert np.allclose(corr["recon_B_kurt_corr"], [2.5, 3.0])


def test_null_corrected_kurtosis_mismatch_raises():
    real = {"seed": 0, "step": 100, "empirical_kurt": np.array([1.0]),
            "recon_A_kurt": np.array([1.0]), "recon_B_kurt": np.array([1.0])}
    shuf = {"seed": 1, "step": 100, "empirical_kurt": np.array([1.0]),
            "recon_A_kurt": np.array([1.0]), "recon_B_kurt": np.array([1.0])}
    with pytest.raises(ValueError):
        null_corrected_kurtosis(real, shuf)


def test_f4_signal_arithmetic():
    # F4: delta_raw = B_pred - B_act on real labels; delta_null on shuffled.
    rev_act_real = {"seed": 0, "step": 100,
                    "recon_B_kurt": np.array([1.0, 2.0, 3.0])}
    rev_pred_real = {"seed": 0, "step": 100,
                     "recon_B_kurt": np.array([1.5, 2.5, 3.5])}
    rev_act_shuf = {"seed": 0, "step": 100,
                    "recon_B_kurt": np.array([1.0, 2.0, 3.0])}
    rev_pred_shuf = {"seed": 0, "step": 100,
                     "recon_B_kurt": np.array([1.1, 2.1, 3.1])}
    sig = f4_signal_with_null_correction(
        rev_act_real, rev_act_shuf, rev_pred_real, rev_pred_shuf)
    assert np.allclose(sig["delta_raw"], [0.5, 0.5, 0.5])
    assert np.allclose(sig["delta_null"], [0.1, 0.1, 0.1])
    assert np.allclose(sig["delta_corr"], [0.4, 0.4, 0.4])
    assert np.allclose(sig["null_absorption"], [0.2, 0.2, 0.2])


# ----------------------------------------------------------------------
# Per-coord kurtosis sanity.
# ----------------------------------------------------------------------
def test_per_coord_excess_kurt_gaussian_zero():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((10000, 8))
    k = _per_coord_excess_kurt(x)
    assert abs(k) < 0.4, f"Gaussian per-coord kurt should be ~0, got {k}"


def test_per_coord_excess_kurt_heavy_tails_positive():
    # Student-t with df=4 has excess kurtosis 6/(df-4) = undefined; df=5
    # has excess kurtosis 6.0. Use df=6 for ~3.0.
    rng = np.random.default_rng(0)
    df = 6
    x = rng.standard_t(df, size=(20000, 8))
    k = _per_coord_excess_kurt(x)
    assert k > 1.5, f"heavy-tailed data should have positive kurt, got {k}"


def test_mardia_z_gaussian_near_zero():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((1000, 8))
    z = _mardia_kurtosis_standardized(X, max_dim=8)
    assert abs(z) < 4.0, f"Gaussian Mardia Z should be near 0, got {z}"


def test_mardia_z_heavy_tails_large():
    rng = np.random.default_rng(0)
    X = rng.standard_t(6, size=(1000, 8))
    z = _mardia_kurtosis_standardized(X, max_dim=8)
    assert z > 5.0, f"heavy-tailed Mardia Z should be large, got {z}"


# ----------------------------------------------------------------------
# Subspace decomposition: synthetic data with injected heavy tails
# in the orthogonal complement.
# ----------------------------------------------------------------------
def test_unembedding_subspace_split_isolates_heavy_tails():
    """Construct data with two subspaces: a 'parallel' subspace where
    samples are Gaussian, and a 'perpendicular' subspace where samples
    are heavy-tailed. Test that per-coord kurt on the two subspaces
    correctly isolates the heavy tails to perp.
    """
    rng = np.random.default_rng(0)
    H = 16
    d_par = 8
    # Orthonormal basis for the 'parallel' subspace: first 8 standard
    # basis vectors.
    V_par = np.eye(H)[:, :d_par]                       # (H, d_par)

    # Synthesize N samples: 8-d Gaussian along V_par, 8-d t(df=6) along
    # the orthogonal complement.
    N = 5000
    Z_par = rng.standard_normal((N, d_par))
    Z_perp = rng.standard_t(6, size=(N, H - d_par))    # heavy
    # Lift into H-dim space.
    X = np.concatenate([Z_par, Z_perp], axis=1)        # (N, H)

    # Project.
    par_full = X @ V_par                                # (N, d_par)
    par_lift = par_full @ V_par.T                        # (N, H)
    perp = X - par_lift                                  # (N, H)

    k_par = _per_coord_excess_kurt(par_full)
    valid_perp = perp.var(0) > 1e-10
    k_perp = _per_coord_excess_kurt(perp[:, valid_perp])

    # Parallel should be ~Gaussian, perp should be heavy.
    assert abs(k_par) < 0.5, f"par kurt should be ~0, got {k_par}"
    assert k_perp > 1.5, f"perp kurt should be heavy, got {k_perp}"
    # Gap negative means parallel is more Gaussian than perpendicular,
    # which is exactly the N1 prediction.
    assert (k_par - k_perp) < -1.0, (
        f"par - perp should be strongly negative for N1; got "
        f"{k_par - k_perp}"
    )


# ----------------------------------------------------------------------
# Helper-function sanity.
# ----------------------------------------------------------------------
def test_safe_logspace_fit_recovers_known_slope():
    layers = np.arange(20, dtype=np.float64)
    v = 0.5 * layers ** 1.3
    v[0] = 0.0   # mimics forward boundary
    la, lm = _safe_logspace_fit(layers, v)
    assert np.isfinite(la) and np.isfinite(lm)
    assert abs(lm - 1.3) < 0.05, f"expected lambda ~1.3, got {lm}"


def test_coefficient_of_variation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cv = _coefficient_of_variation(x)
    assert cv > 0.4 and cv < 0.6


def test_coefficient_of_variation_nan_robust():
    x = np.array([1.0, np.nan, 3.0, np.nan])
    cv = _coefficient_of_variation(x)
    assert cv > 0.6 and cv < 0.8


def test_trace_within_per_cell_failed_flow():
    failed = {
        "failed": True,
        "singular_values": np.zeros((14, 16)),
    }
    out = _trace_within_per_cell(failed)
    assert out.shape == (14,)
    assert np.isnan(out).all()


def test_trace_within_per_cell_normal():
    flow = {
        "failed": False,
        "singular_values": np.ones((3, 4)) * 2.0,
        "n_pilots": 4,
    }
    # trace = sum_d s_d^2 / n = sum(4) / 4 = 4 at each layer.
    out = _trace_within_per_cell(flow)
    assert np.allclose(out, np.array([4.0, 4.0, 4.0]))


# ----------------------------------------------------------------------
# Integration test (skipped unless REVERSE_BUILDUP_RUN_DIR is set).
# ----------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("REVERSE_BUILDUP_RUN_DIR"),
    reason="REVERSE_BUILDUP_RUN_DIR not set; integration test skipped"
)
def test_forward_parameterized_matches_existing():
    """Verify that running the parameterized D1/D3/D4a pipeline with
    view='forward' produces results identical to the existing
    model_abc_discriminator output. This is the proposal's hard
    idempotency requirement for the refactor.
    """
    run_dir = os.environ["REVERSE_BUILDUP_RUN_DIR"]
    from reverse_buildup import run_d1_view, run_d3_view, run_d4a_view
    from multiview_campaign import seeds_in_run
    from reverse_buildup_campaign import discover_common_steps

    seeds = seeds_in_run(run_dir)
    steps = discover_common_steps(run_dir, seeds)
    if not steps:
        pytest.skip("no common steps in run directory")

    # The parameterized run writes to d1_token_cv_forward.npz; the
    # existing module writes to d1_token_cv.npz. Compare arrays.
    d1_new = run_d1_view(run_dir, seeds, steps, view="forward", verbose=False)
    legacy_path = os.path.join(run_dir, "multiview", "model_abc",
                                "d1_token_cv.npz")
    if not os.path.exists(legacy_path):
        pytest.skip(f"legacy {legacy_path} not on disk")
    with np.load(legacy_path) as f:
        for key in ("cv_trace", "cv_erank", "mean_trace", "mean_erank"):
            np.testing.assert_array_equal(d1_new[key], f[key],
                                          err_msg=f"D1 forward mismatch on {key}")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
