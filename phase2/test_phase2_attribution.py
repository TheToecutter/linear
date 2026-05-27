"""
Tests for phase2_attribution.py.

These tests are pure logic — no GPU, no data, no model loading.
They exercise the classify_axis function with carefully constructed
inputs designed to cover each branch:

  - ROBUST: all variants within threshold of baseline.
  - CONTROLS↑: variants increase monotonically with axis_value, beyond threshold.
  - CONTROLS↓: variants decrease monotonically with axis_value, beyond threshold.
  - NON_MONOTONIC: variants cross threshold but not monotonically.
  - INSUFFICIENT: no variants, or NaN threshold.

Run with:
    python3 -m pytest test_phase2_attribution.py -v
"""

import math
import numpy as np
import pytest

from phase2_attribution import (
    classify_axis,
    ROBUST, CONTROLS_UP, CONTROLS_DOWN, NON_MONOTONIC, INSUFFICIENT,
)


class TestClassifyAxis:
    # threshold is generous (0.1); baseline_mean = 1.0.
    BM = 1.0
    TH = 0.1

    def test_robust_when_all_within_threshold(self):
        # Variants land at 0.95, 1.05 -- both within 0.1 of 1.0.
        variants = [(6, 0.95), (24, 1.05)]
        c, deltas = classify_axis(self.BM, self.TH, variants)
        assert c == ROBUST
        assert deltas == pytest.approx([-0.05, 0.05])

    def test_controls_up_monotonic_increase(self):
        # Variant means rise with axis_value past threshold.
        variants = [(6, 1.05), (24, 1.5)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        assert c == CONTROLS_UP

    def test_controls_down_monotonic_decrease(self):
        # Variant means fall with axis_value past threshold.
        variants = [(6, 1.5), (24, 0.5)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        assert c == CONTROLS_DOWN

    def test_non_monotonic_when_violates_order(self):
        # Variant means rise then fall -- non-monotonic at axis_value ordering.
        variants = [(6, 1.5), (12, 2.0), (24, 1.0)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        assert c == NON_MONOTONIC

    def test_insufficient_when_no_variants(self):
        c, _ = classify_axis(self.BM, self.TH, [])
        assert c == INSUFFICIENT

    def test_insufficient_when_threshold_nan(self):
        c, _ = classify_axis(self.BM, float("nan"), [(6, 1.5)])
        assert c == INSUFFICIENT

    def test_insufficient_when_threshold_zero(self):
        c, _ = classify_axis(self.BM, 0.0, [(6, 1.5)])
        assert c == INSUFFICIENT

    def test_single_variant_threshold_crossed_non_monotonic(self):
        # Only one variant, exceeds threshold -- can't establish monotonicity
        # in the traditional sense; we label NON_MONOTONIC.
        c, _ = classify_axis(self.BM, self.TH, [(6, 1.5)])
        assert c == NON_MONOTONIC

    def test_boundary_case_just_under_threshold(self):
        # |delta| = 0.099 just under threshold 0.1.
        variants = [(6, 0.901), (24, 1.099)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        assert c == ROBUST

    def test_boundary_case_just_over_threshold(self):
        # Both variants over threshold, both increasing.
        variants = [(6, 0.85), (24, 1.15)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        # Smaller axis_value 6: delta = -0.15 (below baseline by 0.15)
        # Larger axis_value 24: delta = +0.15 (above baseline by 0.15)
        # Monotonic increasing in axis_value. So CONTROLS_UP.
        assert c == CONTROLS_UP

    def test_one_variant_within_one_outside_below(self):
        # One variant within threshold (robust look), one outside.
        # We should NOT call it ROBUST because at least one crosses.
        variants = [(6, 1.02), (24, 1.5)]
        c, _ = classify_axis(self.BM, self.TH, variants)
        # axis 6: delta = +0.02 within threshold
        # axis 24: delta = +0.5 well above
        # Both deltas positive, axis_value increasing → CONTROLS_UP.
        assert c == CONTROLS_UP


class TestDeltaComputation:
    def test_deltas_are_signed(self):
        BM, TH = 1.0, 100.0  # threshold huge, so always ROBUST
        variants = [(6, 0.5), (24, 1.5)]
        _, deltas = classify_axis(BM, TH, variants)
        assert deltas == pytest.approx([-0.5, 0.5])

    def test_deltas_preserve_order(self):
        BM, TH = 1.0, 100.0
        variants = [(24, 1.5), (6, 0.5)]
        _, deltas = classify_axis(BM, TH, variants)
        # Deltas should be in input order, not axis-sorted.
        assert deltas == pytest.approx([0.5, -0.5])


class TestEdgeCases:
    def test_zero_delta(self):
        # All variants exactly at baseline.
        variants = [(6, 1.0), (24, 1.0)]
        c, deltas = classify_axis(1.0, 0.1, variants)
        assert c == ROBUST
        assert deltas == pytest.approx([0.0, 0.0])

    def test_negative_baseline(self):
        # Real example: log α is negative around -3.7.
        BM, TH = -3.7, 0.1
        variants = [(6, -3.5), (24, -4.0)]
        # axis 6: delta = +0.2 (above)
        # axis 24: delta = -0.3 (below)
        # both cross threshold, axis_value increasing → delta decreasing
        # → CONTROLS_DOWN.
        c, _ = classify_axis(BM, TH, variants)
        assert c == CONTROLS_DOWN


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
