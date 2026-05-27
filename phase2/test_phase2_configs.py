"""
Tests for phase2_configs.py.

These tests don't require a GPU or any data — they just check that
the variant catalog resolves correctly, that the factories produce
configs with the expected dimensions, and that the parameter-count
calculations match what the proposal says.

Run with:
    python3 -m pytest test_phase2_configs.py -v
"""

import pytest
import math

from config import ModelConfig
from phase2_configs import (
    make_baseline_gelu_config,
    ALL_TIER1_VARIANTS, ALL_VARIANTS,
    DEPTH_VARIANTS, WIDTH_VARIANTS, FFN_RATIO_VARIANTS,
    find_variant, variants_by_axis, summarize,
    BASELINE_LABEL,
)


# ----------------------------------------------------------------------
# Baseline.
# ----------------------------------------------------------------------
class TestBaseline:
    def test_baseline_dims(self):
        cfg = make_baseline_gelu_config()
        assert cfg.num_hidden_layers == 12
        assert cfg.hidden_size == 896
        assert cfg.intermediate_size == 2432
        assert cfg.num_attention_heads == 14
        assert cfg.ffn_type == "gelu"
        assert cfg.head_dim == 64

    def test_baseline_param_count(self):
        """The proposal says ~146M params. Tolerate small deviation."""
        cfg = make_baseline_gelu_config()
        p = cfg.estimate_param_count()
        # From the proposal: ~146.4M.
        assert 140e6 < p < 150e6, f"param count {p/1e6:.1f}M out of expected range"

    def test_baseline_gelu_matches_swiglu_params(self):
        """The proposal asserts GELU at I_gelu = 1.5 × I_swiglu is
        parameter-matched to SwiGLU. Verify."""
        gelu = make_baseline_gelu_config()
        swiglu = make_baseline_gelu_config()
        swiglu.ffn_type = "swiglu"
        assert gelu.estimate_param_count() == swiglu.estimate_param_count()


# ----------------------------------------------------------------------
# Depth sweep.
# ----------------------------------------------------------------------
class TestDepthSweep:
    def test_depth_variants_exist(self):
        labels = {v.label for v in DEPTH_VARIANTS}
        assert labels == {"L06", "L18", "L24"}, (
            "Depth sweep should be {L=6, L=18, L=24}; L=12 is the baseline. "
            "L=18 was added as a fourth depth-axis point to test λL "
            "conservation against the L=12 baseline + L=6 + L=24 measurements."
        )

    def test_depth_only_changes_layers(self):
        baseline = make_baseline_gelu_config()
        for variant in DEPTH_VARIANTS:
            cfg = variant.config_factory()
            # Only num_hidden_layers should differ from the baseline.
            assert cfg.hidden_size == baseline.hidden_size
            assert cfg.intermediate_size == baseline.intermediate_size
            assert cfg.num_attention_heads == baseline.num_attention_heads
            assert cfg.ffn_type == baseline.ffn_type
            assert cfg.head_dim == baseline.head_dim

    def test_depth_axis_values(self):
        for v in DEPTH_VARIANTS:
            cfg = v.config_factory()
            assert int(v.axis_value) == cfg.num_hidden_layers, (
                f"variant {v.label} has axis_value={v.axis_value} but "
                f"num_hidden_layers={cfg.num_hidden_layers}"
            )


# ----------------------------------------------------------------------
# Width sweep.
# ----------------------------------------------------------------------
class TestWidthSweep:
    def test_width_variants_exist(self):
        labels = {v.label for v in WIDTH_VARIANTS}
        assert labels == {"H0448", "H1792"}, (
            "Width sweep should be {H=448, H=1792}; H=896 is the baseline."
        )

    def test_width_keeps_head_dim_64(self):
        """The proposal specifies head_dim=64 fixed across the width sweep."""
        for v in WIDTH_VARIANTS:
            cfg = v.config_factory()
            assert cfg.head_dim == 64, (
                f"variant {v.label}: head_dim={cfg.head_dim}, expected 64"
            )
            assert cfg.hidden_size % cfg.num_attention_heads == 0

    def test_width_intermediate_scales_proportionally(self):
        """At H=448 the intermediate should be ~half of baseline 2432;
        at H=1792 it should be ~double. Allow rounding to nearest 64.
        """
        baseline_ratio = 2432 / 896  # ≈ 2.71428
        for v in WIDTH_VARIANTS:
            cfg = v.config_factory()
            actual_ratio = cfg.intermediate_size / cfg.hidden_size
            assert abs(actual_ratio - baseline_ratio) < 0.05, (
                f"variant {v.label}: I/H ratio = {actual_ratio:.4f}, "
                f"expected ≈ {baseline_ratio:.4f}"
            )
            assert cfg.intermediate_size % 64 == 0, (
                f"variant {v.label}: I = {cfg.intermediate_size} should be "
                f"a multiple of 64 for vectorization."
            )

    def test_width_only_changes_dimensions(self):
        baseline = make_baseline_gelu_config()
        for v in WIDTH_VARIANTS:
            cfg = v.config_factory()
            assert cfg.num_hidden_layers == baseline.num_hidden_layers
            assert cfg.ffn_type == baseline.ffn_type


# ----------------------------------------------------------------------
# FFN ratio sweep.
# ----------------------------------------------------------------------
class TestFFNRatioSweep:
    def test_ratio_variants_exist(self):
        labels = {v.label for v in FFN_RATIO_VARIANTS}
        assert labels == {"ffn_1p5x", "ffn_3p0x"}, (
            "FFN ratio sweep should be {1.5x, 3.0x}; 4.0x is approximated "
            "by the GELU baseline (I_gelu/H ≈ 4.07)."
        )

    def test_ratio_actual_gelu_intermediate(self):
        """Verify that the actual GELU intermediate matches the target ratio.

        LlamaBlock applies I_gelu = 3 × I_stored / 2. So stored I should
        equal target_I_gelu × 2/3.
        """
        H = 896
        for v in FFN_RATIO_VARIANTS:
            cfg = v.config_factory()
            stored_I = cfg.intermediate_size
            # LlamaBlock: gelu_intermediate = (3 * intermediate_size) // 2
            actual_I_gelu = (3 * stored_I) // 2
            target = v.axis_value * H
            # Allow ±64 (one rounding bucket).
            assert abs(actual_I_gelu - target) <= 64, (
                f"variant {v.label}: actual GELU intermediate {actual_I_gelu} "
                f"differs from target {target} by more than 64."
            )

    def test_ratio_only_changes_intermediate(self):
        baseline = make_baseline_gelu_config()
        for v in FFN_RATIO_VARIANTS:
            cfg = v.config_factory()
            assert cfg.num_hidden_layers == baseline.num_hidden_layers
            assert cfg.hidden_size == baseline.hidden_size
            assert cfg.num_attention_heads == baseline.num_attention_heads
            assert cfg.ffn_type == baseline.ffn_type
            assert cfg.intermediate_size != baseline.intermediate_size


# ----------------------------------------------------------------------
# Variant catalog API.
# ----------------------------------------------------------------------
class TestCatalog:
    def test_all_tier1_combines_three_sweeps(self):
        expected_labels = (
            {"L06", "L18", "L24"}
            | {"H0448", "H1792"}
            | {"ffn_1p5x", "ffn_3p0x"}
        )
        actual_labels = {v.label for v in ALL_TIER1_VARIANTS}
        assert actual_labels == expected_labels

    def test_find_variant_round_trip(self):
        for v in ALL_VARIANTS:
            found = find_variant(v.label)
            assert found is v, f"find_variant({v.label!r}) returned wrong object"

    def test_find_variant_raises_on_unknown(self):
        with pytest.raises(ValueError):
            find_variant("nonexistent_variant_xyz")

    def test_variants_by_axis(self):
        expected_counts = {"depth": 3, "width": 2, "ffn_ratio": 2}
        for axis, expected_n in expected_counts.items():
            vs = variants_by_axis(axis)
            assert len(vs) == expected_n, (
                f"axis {axis} should have {expected_n} variants; "
                f"got {len(vs)}"
            )
            assert all(v.axis == axis for v in vs)

    def test_summarize_runs(self):
        text = summarize()
        # Sanity: every variant label should appear somewhere in the summary.
        for v in ALL_VARIANTS:
            assert v.label in text, f"summary missing {v.label}"
        assert BASELINE_LABEL in text


# ----------------------------------------------------------------------
# Parameter counts make sense.
# ----------------------------------------------------------------------
class TestParamCounts:
    def test_depth_param_counts_scale_with_L(self):
        baseline = make_baseline_gelu_config()
        baseline_p = baseline.estimate_param_count()
        l6 = next(v for v in DEPTH_VARIANTS if v.label == "L06").config_factory()
        l24 = next(v for v in DEPTH_VARIANTS if v.label == "L24").config_factory()
        # L=6 should have fewer params than baseline; L=24 should have more.
        assert l6.estimate_param_count() < baseline_p
        assert l24.estimate_param_count() > baseline_p

    def test_width_param_counts_scale_with_H(self):
        baseline = make_baseline_gelu_config()
        baseline_p = baseline.estimate_param_count()
        h_narrow = next(
            v for v in WIDTH_VARIANTS if v.label == "H0448"
        ).config_factory()
        h_wide = next(
            v for v in WIDTH_VARIANTS if v.label == "H1792"
        ).config_factory()
        assert h_narrow.estimate_param_count() < baseline_p
        assert h_wide.estimate_param_count() > baseline_p


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
    