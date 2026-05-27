"""
Tests for phase2_plots.py.

Plotting tests are smoke tests: we feed in synthetic AttributionCell
lists and Tier 1b row dicts, and verify the rendering produces a valid
PNG file. We don't visually inspect the result, but we catch crashes
in the rendering path (which is where the bugs tend to be).

Run with:
    python3 -m pytest test_phase2_plots.py -v
"""

import os
import tempfile
import pytest
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless tests

from phase2_attribution import (
    AttributionCell,
    ROBUST, CONTROLS_UP, CONTROLS_DOWN, NON_MONOTONIC, INSUFFICIENT,
)
from phase2_plots import plot_attribution_heatmap, plot_tier1b_bars


def make_cells():
    """Build a small synthetic attribution-matrix dataset spanning all
    classification categories."""
    return [
        AttributionCell(
            statistic="λ", axis="depth", classification=CONTROLS_DOWN,
            baseline_mean=0.43, baseline_threshold=0.01,
            variant_labels=["L06", "L24"],
            variant_means=[0.85, 0.22],
            deltas=[+0.42, -0.21],
        ),
        AttributionCell(
            statistic="λ", axis="width", classification=ROBUST,
            baseline_mean=0.43, baseline_threshold=0.01,
            variant_labels=["H0448", "H1792"],
            variant_means=[0.435, 0.42],
            deltas=[+0.005, -0.01],
        ),
        AttributionCell(
            statistic="λ", axis="ffn_ratio", classification=NON_MONOTONIC,
            baseline_mean=0.43, baseline_threshold=0.01,
            variant_labels=["ffn_1p5x", "ffn_3p0x"],
            variant_means=[0.50, 0.40],
            deltas=[+0.07, -0.03],
        ),
        AttributionCell(
            statistic="log α", axis="depth", classification=CONTROLS_UP,
            baseline_mean=-3.7, baseline_threshold=0.1,
            variant_labels=["L06", "L24"],
            variant_means=[-3.9, -3.5],
            deltas=[-0.2, +0.2],
        ),
        AttributionCell(
            statistic="log α", axis="width", classification=ROBUST,
            baseline_mean=-3.7, baseline_threshold=0.1,
            variant_labels=["H0448", "H1792"],
            variant_means=[-3.71, -3.69],
            deltas=[-0.01, +0.01],
        ),
        AttributionCell(
            statistic="log α", axis="ffn_ratio", classification=INSUFFICIENT,
            baseline_mean=-3.7, baseline_threshold=0.1,
            variant_labels=[],
            variant_means=[], deltas=[],
        ),
    ]


def make_tier1b_rows():
    """Synthetic Tier 1b table covering 3 statistics × 3 distributions."""
    return [
        {"input_distribution": "real",     "statistic": "λ",
         "mean": 0.43, "std": 0.005, "n_seeds": 4},
        {"input_distribution": "shuffled", "statistic": "λ",
         "mean": 0.41, "std": 0.006, "n_seeds": 4},
        {"input_distribution": "random",   "statistic": "λ",
         "mean": 0.40, "std": 0.008, "n_seeds": 4},

        {"input_distribution": "real",     "statistic": "log α",
         "mean": -3.7, "std": 0.07, "n_seeds": 4},
        {"input_distribution": "shuffled", "statistic": "log α",
         "mean": -4.1, "std": 0.08, "n_seeds": 4},
        {"input_distribution": "random",   "statistic": "log α",
         "mean": -4.3, "std": 0.09, "n_seeds": 4},

        {"input_distribution": "real",     "statistic": "eff rank mid",
         "mean": 490, "std": 15, "n_seeds": 4},
        {"input_distribution": "shuffled", "statistic": "eff rank mid",
         "mean": 480, "std": 18, "n_seeds": 4},
        {"input_distribution": "random",   "statistic": "eff rank mid",
         "mean": 470, "std": 20, "n_seeds": 4},
    ]


class TestAttributionHeatmap:
    def test_smoke_render(self, tmp_path):
        cells = make_cells()
        out = tmp_path / "heatmap.png"
        plot_attribution_heatmap(cells, str(out))
        assert out.exists()
        assert out.stat().st_size > 5000  # PNGs are at least a few KB

    def test_empty_cells_no_crash(self, tmp_path):
        out = tmp_path / "heatmap_empty.png"
        plot_attribution_heatmap([], str(out))
        # No file written; we just check no exception was raised.


class TestTier1bBars:
    def test_smoke_render(self, tmp_path):
        rows = make_tier1b_rows()
        out = tmp_path / "tier1b.png"
        plot_tier1b_bars(rows, str(out))
        assert out.exists()
        assert out.stat().st_size > 5000

    def test_empty_rows_no_crash(self, tmp_path):
        out = tmp_path / "tier1b_empty.png"
        plot_tier1b_bars([], str(out))

    def test_no_real_baseline_no_crash(self, tmp_path):
        rows = [
            {"input_distribution": "shuffled", "statistic": "λ",
             "mean": 0.41, "std": 0.006, "n_seeds": 4},
        ]
        out = tmp_path / "tier1b_no_real.png"
        plot_tier1b_bars(rows, str(out))
        # No real reference → skipped without crash.


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
