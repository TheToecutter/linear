"""
Integration test for the launch ↔ memfit handshake.

Verifies that when a MEMFIT_NOTES.txt is present in the variant's
parent directory, the launcher reads it and applies the
micro_batch_size / grad_accum_steps overrides to the TrainingConfig
it constructs.

We can't run actual training here (no GPU), so this test:
  1. Writes a MEMFIT_NOTES file with non-default batch shape.
  2. Patches train.train_one_run to capture the args passed to it.
  3. Calls phase2_launch.launch_one() with dry_run=False but with
     train.train_one_run patched out.
  4. Asserts the captured TrainingConfig has the overridden values.

Run with:
    python3 -m pytest test_phase2_launch_memfit.py -v
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from phase2_configs import find_variant
from phase2_launch import launch_one
from phase2_memfit import MemFitNotes, write_notes


def test_memfit_overrides_applied():
    """When MEMFIT_NOTES.txt is present, launch_one should override
    the TrainingConfig's batch shape and pass it through to train_one_run."""
    variant = find_variant("H1792")
    with tempfile.TemporaryDirectory() as tmp:
        # Write a notes file with non-default batch shape.
        notes = MemFitNotes(
            variant_label=variant.label, variant_axis=variant.axis,
            params_M=526.7,
            micro_batch_size=4, grad_accum_steps=16,
            seq_len=1024, peak_vram_gb=28.0,
            step_time_s=4.5, probe_steps=50,
            used_gradient_checkpointing=True,
            hostname="test", timestamp="now",
        )
        write_notes(variant, notes, root=tmp)

        # Patch train_one_run so it captures, not trains.
        captured = {}

        def fake_train_one_run(model_cfg, train_cfg, seed, run_dir, **kw):
            captured["model_cfg"] = model_cfg
            captured["train_cfg"] = train_cfg
            captured["seed"] = seed
            captured["run_dir"] = run_dir

        # The launcher imports train_one_run lazily inside the function;
        # we patch the module-level binding at the source.
        with patch("train.train_one_run", side_effect=fake_train_one_run):
            launch_one(
                variant=variant, seed=0,
                total_steps=10,  # quick exit if anything slipped through
                dry_run=False, force=True,
                device="cpu", root=tmp, apply_memfit=True,
            )

        assert "train_cfg" in captured
        assert captured["train_cfg"].micro_batch_size == 4
        assert captured["train_cfg"].grad_accum_steps == 16
        # Effective batch should still be 64.
        assert (captured["train_cfg"].micro_batch_size
                * captured["train_cfg"].grad_accum_steps) == 64


def test_memfit_skipped_when_disabled():
    """When apply_memfit=False, the launcher should NOT apply overrides
    even if a notes file is present."""
    variant = find_variant("H1792")
    with tempfile.TemporaryDirectory() as tmp:
        notes = MemFitNotes(
            variant_label=variant.label, variant_axis=variant.axis,
            params_M=526.7,
            micro_batch_size=2, grad_accum_steps=32,
            seq_len=1024, peak_vram_gb=30.0,
            step_time_s=8.0, probe_steps=50,
            used_gradient_checkpointing=True,
            hostname="test", timestamp="now",
        )
        write_notes(variant, notes, root=tmp)

        captured = {}

        def fake_train_one_run(model_cfg, train_cfg, seed, run_dir, **kw):
            captured["train_cfg"] = train_cfg

        with patch("train.train_one_run", side_effect=fake_train_one_run):
            launch_one(
                variant=variant, seed=0,
                total_steps=10, dry_run=False, force=True,
                device="cpu", root=tmp, apply_memfit=False,
            )

        # Default batch shape preserved.
        assert captured["train_cfg"].micro_batch_size == 8
        assert captured["train_cfg"].grad_accum_steps == 8


def test_memfit_missing_keeps_defaults():
    """When no notes file is present, launch_one should use the
    TrainingConfig defaults."""
    variant = find_variant("L06")
    with tempfile.TemporaryDirectory() as tmp:
        captured = {}

        def fake_train_one_run(model_cfg, train_cfg, seed, run_dir, **kw):
            captured["train_cfg"] = train_cfg

        with patch("train.train_one_run", side_effect=fake_train_one_run):
            launch_one(
                variant=variant, seed=0,
                total_steps=10, dry_run=False, force=True,
                device="cpu", root=tmp, apply_memfit=True,
            )

        # Defaults preserved.
        assert captured["train_cfg"].micro_batch_size == 8
        assert captured["train_cfg"].grad_accum_steps == 8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
