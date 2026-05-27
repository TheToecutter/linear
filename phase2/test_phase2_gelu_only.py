"""
Tests for the Phase 2 GELU-only invariant.

Phase 2 is GELU-only by design: every variant must use ffn_type="gelu"
so its statistics are comparable to the Phase 1 GELU baseline. This is
enforced at three levels:

  1. Catalog level: every variant in ALL_TIER1_VARIANTS resolves to a
     ModelConfig with ffn_type="gelu" at definition time.
  2. Tier 2: TIER2_VARIANTS is empty (no SwiGLU one-shot left in the
     catalog).
  3. Runtime: phase2_launch.launch_one() raises ValueError if any
     non-GELU config is about to launch, as a backstop against a
     future config edit that bypasses the catalog.

Run with:
    python3 -m pytest test_phase2_gelu_only.py -v
"""

import os
import tempfile
import pytest
from unittest.mock import patch

from phase2_configs import (
    ALL_TIER1_VARIANTS, ALL_VARIANTS, TIER2_VARIANTS, VariantSpec,
    make_baseline_gelu_config, find_variant,
)
from phase2_launch import launch_one


class TestCatalogLevel:
    """Every variant the catalog hands out must be GELU."""

    def test_baseline_is_gelu(self):
        assert make_baseline_gelu_config().ffn_type == "gelu"

    def test_every_tier1_variant_is_gelu(self):
        for variant in ALL_TIER1_VARIANTS:
            cfg = variant.config_factory()
            assert cfg.ffn_type == "gelu", (
                f"Tier 1 variant {variant.label!r} has ffn_type="
                f"{cfg.ffn_type!r}; expected 'gelu'."
            )

    def test_every_variant_in_all_variants_is_gelu(self):
        """ALL_VARIANTS includes Tier 2 too; everything must still be GELU."""
        for variant in ALL_VARIANTS:
            cfg = variant.config_factory()
            assert cfg.ffn_type == "gelu", (
                f"Variant {variant.label!r} (tier={variant.tier}) has "
                f"ffn_type={cfg.ffn_type!r}; Phase 2 is GELU-only."
            )

    def test_tier2_is_empty(self):
        """Tier 2 catalog must not contain a SwiGLU one-shot."""
        # If a Tier 2 variant is ever added, it MUST be GELU (covered by
        # test_every_variant_in_all_variants_is_gelu). The "empty"
        # invariant here is a stricter form: we don't yet have a use
        # case for any Tier 2 variant, and adding one prematurely risks
        # un-validated paths.
        assert TIER2_VARIANTS == [], (
            f"TIER2_VARIANTS is no longer empty: "
            f"{[v.label for v in TIER2_VARIANTS]}. If you're adding a "
            f"Tier 2 variant intentionally, update this test."
        )


class TestRuntimeBackstop:
    """The launcher refuses to run any non-GELU variant."""

    def test_launch_one_refuses_swiglu(self, tmp_path):
        """Hand-crafted non-GELU variant should be rejected by launch_one."""
        def make_bad_config():
            cfg = make_baseline_gelu_config()
            cfg.ffn_type = "swiglu"   # deliberate violation
            return cfg

        bad_variant = VariantSpec(
            label="evil_swiglu",
            axis="gating",
            tier="1a",
            config_factory=make_bad_config,
            seeds=[0],
            description="Test fixture: should never actually launch.",
        )

        # Patch train_one_run so even if the assertion fails to fire,
        # we don't accidentally launch a real training job.
        with patch("train.train_one_run") as mock_train:
            with pytest.raises(ValueError, match="GELU-only"):
                launch_one(
                    variant=bad_variant, seed=0,
                    total_steps=10, dry_run=False, force=True,
                    device="cpu", root=str(tmp_path), apply_memfit=False,
                )
            # train_one_run must not have been called.
            mock_train.assert_not_called()

    def test_launch_one_accepts_legitimate_gelu_variant(self, tmp_path):
        """Sanity: real GELU variants should pass the check."""
        variant = find_variant("L06")
        with patch("train.train_one_run") as mock_train:
            launch_one(
                variant=variant, seed=0,
                total_steps=10, dry_run=False, force=True,
                device="cpu", root=str(tmp_path), apply_memfit=False,
            )
            mock_train.assert_called_once()

    def test_assertion_mentions_variant_label(self, tmp_path):
        """The error message should name the offending variant so future-
        you can find it without grep."""
        def make_bad_config():
            cfg = make_baseline_gelu_config()
            cfg.ffn_type = "swiglu"
            return cfg

        bad_variant = VariantSpec(
            label="my_uniquely_named_variant",
            axis="gating", tier="1a",
            config_factory=make_bad_config,
            seeds=[0],
        )
        with patch("train.train_one_run"):
            with pytest.raises(ValueError) as exc_info:
                launch_one(
                    variant=bad_variant, seed=0,
                    total_steps=10, dry_run=False, force=True,
                    device="cpu", root=str(tmp_path), apply_memfit=False,
                )
            # Message should name the variant AND the actual ffn_type.
            msg = str(exc_info.value)
            assert "my_uniquely_named_variant" in msg
            assert "swiglu" in msg


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
