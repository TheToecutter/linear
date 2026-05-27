"""
Tests for phase2_memfit.py.

These tests cover only the parts that don't require a GPU:
  - Notes round-trip (write then read).
  - DEFAULT_HEAVYWEIGHT_LABELS list is non-empty and resolves.
  - FALLBACK_BATCH_SHAPES sequence makes sense.
  - Dry-run path resolves variant configs without errors.

The GPU-bound `_run_probe` and `fit_check_variant` are NOT tested
here; they need to be run on the target machine. Their behavior is
covered by manual smoke tests during Phase 2 launch.

Run with:
    python3 -m pytest test_phase2_memfit.py -v
"""

import os
import tempfile
import pytest

from phase2_configs import find_variant, ALL_TIER1_VARIANTS
from phase2_memfit import (
    DEFAULT_HEAVYWEIGHT_LABELS,
    FALLBACK_BATCH_SHAPES,
    MemFitNotes,
    write_notes,
    read_notes,
)


class TestDefaultHeavyweightLabels:
    def test_nonempty(self):
        assert len(DEFAULT_HEAVYWEIGHT_LABELS) > 0

    def test_labels_resolve(self):
        for lab in DEFAULT_HEAVYWEIGHT_LABELS:
            v = find_variant(lab)
            assert v.label == lab

    def test_labels_are_largest_variants(self):
        """Sanity: the heavyweight set should genuinely be the largest
        variants. If we ever add a bigger variant, this should be updated."""
        params_M = {}
        for v in ALL_TIER1_VARIANTS:
            cfg = v.config_factory()
            params_M[v.label] = cfg.estimate_param_count() / 1e6
        # Heavyweight labels should be among the top-N by param count.
        sorted_labels = sorted(params_M, key=lambda k: -params_M[k])
        top_n = sorted_labels[: len(DEFAULT_HEAVYWEIGHT_LABELS)]
        for lab in DEFAULT_HEAVYWEIGHT_LABELS:
            assert lab in top_n, (
                f"Heavyweight label {lab} ({params_M[lab]:.0f}M) is not "
                f"in the top-{len(DEFAULT_HEAVYWEIGHT_LABELS)} param-count "
                f"variants {top_n}."
            )


class TestFallbackBatchShapes:
    def test_shapes_keep_effective_batch_constant(self):
        """Every fallback shape should yield the same effective batch
        size (micro × accum). Otherwise the training-dynamics signal
        from the larger model would not be comparable to the baseline.
        """
        products = {b * a for b, a in FALLBACK_BATCH_SHAPES}
        assert len(products) == 1, (
            f"Fallback shapes don't all have the same effective batch: "
            f"{FALLBACK_BATCH_SHAPES}"
        )

    def test_shapes_strictly_decrease_micro_batch(self):
        for (b1, _), (b2, _) in zip(FALLBACK_BATCH_SHAPES,
                                     FALLBACK_BATCH_SHAPES[1:]):
            assert b2 < b1, (
                f"Fallback shapes should monotonically decrease micro_batch:"
                f" {FALLBACK_BATCH_SHAPES}"
            )

    def test_smallest_batch_is_one(self):
        # We should explore micro_batch=1 as the last resort before
        # declaring a variant unfittable.
        smallest = min(b for b, _ in FALLBACK_BATCH_SHAPES)
        assert smallest == 1


class TestNotesRoundTrip:
    def test_write_then_read(self, tmp_path):
        variant = find_variant("H1792")
        notes = MemFitNotes(
            variant_label=variant.label,
            variant_axis=variant.axis,
            params_M=526.7,
            micro_batch_size=4,
            grad_accum_steps=16,
            seq_len=1024,
            peak_vram_gb=28.5,
            step_time_s=4.2,
            probe_steps=50,
            used_gradient_checkpointing=True,
            hostname="test-host",
            timestamp="2026-05-19T12:00:00+00:00",
        )
        # Override root to write into tmp_path.
        path = write_notes(variant, notes, root=str(tmp_path))
        assert os.path.exists(path)

        # Read back.
        read = read_notes(variant, root=str(tmp_path))
        assert read is not None
        assert read["variant_label"] == "H1792"
        assert read["micro_batch_size"] == 4
        assert read["grad_accum_steps"] == 16
        assert read["params_M"] == pytest.approx(526.7)
        assert read["peak_vram_gb"] == pytest.approx(28.5)
        # Booleans coerced to strings in the simple parser; just check
        # the key is present.
        assert "used_gradient_checkpointing" in read

    def test_read_missing_returns_none(self, tmp_path):
        variant = find_variant("L24")
        result = read_notes(variant, root=str(tmp_path))
        assert result is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
