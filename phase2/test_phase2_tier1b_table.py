"""
Test for the rewritten Tier 1b table builder in phase2_attribution.py.

The earlier implementation routed through SeedBundle/load_flow_series,
which hardcodes 'flow_analysis/' regardless of the requested
distribution -- producing bit-identical 'real'/'shuffled'/'random'
columns. This test verifies the rewrite reads each distribution's own
subdirectory and produces distinct values for distinct inputs.

Run with:
    python3 -m pytest test_phase2_tier1b_table.py -v
"""

import os
import numpy as np
import pytest

# We test using synthetic flow files in a temp directory. The function
# reads PHASE1_GELU_ROOT, which is a module-level constant; we
# monkeypatch it for the test.
import phase2_attribution


@pytest.fixture
def synthetic_baseline(tmp_path, monkeypatch):
    """Build a fake phase1_runs_gelu/ tree with three distributions per
    seed, each containing a flow .npz with DIFFERENT λ values so we can
    distinguish them in the output."""
    root = tmp_path / "phase1_runs_gelu"
    root.mkdir()
    # Per-distribution offsets so we can verify the right files are read.
    offsets = {"flow_analysis": 0.0,
               "flow_analysis_shuffled": 0.1,
               "flow_analysis_random": 0.2}
    for seed in range(4):
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir()
        for subdir, offset in offsets.items():
            sd = seed_dir / subdir
            sd.mkdir()
            flow = {
                "lambda": np.float32(0.35 + offset + 0.001 * seed),
                "lambda_paper": np.float32(0.34 + offset + 0.001 * seed),
                "log_alpha": np.float32(-3.3 + offset),
                "log_alpha_paper": np.float32(-3.3 + offset),
                "kurtosis_per_layer": np.array(
                    [0.9 + offset, 0.95 + offset, 0.85 + offset],
                    dtype=np.float32,
                ),
                "kurtosis_abs_per_layer": np.array(
                    [0.9 + offset, 0.95 + offset, 0.85 + offset],
                    dtype=np.float32,
                ),
                "effective_rank": np.array(
                    [180 + 100 * offset, 480 + 100 * offset, 400 + 100 * offset],
                    dtype=np.float32,
                ),
                "checkpoint_step": 24000,
                "checkpoint_eval_loss": np.float32(2.92),
                "checkpoint_loss": np.float32(2.85),
                "checkpoint_seed": seed,
                "num_layers_total": 3,
                "hidden_dim": 896,
            }
            np.savez(sd / "flow_step_24000.npz", **flow)
    monkeypatch.setattr(phase2_attribution, "PHASE1_GELU_ROOT", str(root))
    return root


class TestBuildTier1bTable:
    def test_distributions_produce_distinct_values(self, synthetic_baseline):
        rows = phase2_attribution.build_tier1b_table(
            input_distributions=("real", "shuffled", "random"),
        )
        # Group by statistic + distribution.
        by_stat = {}
        for r in rows:
            by_stat.setdefault(r["statistic"], {})[r["input_distribution"]] = r
        # λ should differ across distributions by ~0.1 (the offset).
        lam = by_stat["λ (ours)"]
        assert abs(lam["real"]["mean"] - 0.35) < 0.01
        assert abs(lam["shuffled"]["mean"] - 0.45) < 0.01
        assert abs(lam["random"]["mean"] - 0.55) < 0.01
        # Crucially, the three means are not bit-identical.
        assert lam["real"]["mean"] != lam["shuffled"]["mean"]
        assert lam["shuffled"]["mean"] != lam["random"]["mean"]

    def test_eff_rank_reflects_distribution(self, synthetic_baseline):
        rows = phase2_attribution.build_tier1b_table()
        by = {(r["statistic"], r["input_distribution"]): r for r in rows}
        # Mean effective rank: synthetic values mean(180+480+400)/3 = 353.33
        # plus offset * 100.
        assert abs(by[("mean effective rank", "real")]["mean"] - 353.33) < 1.0
        assert by[("mean effective rank", "shuffled")]["mean"] > \
               by[("mean effective rank", "real")]["mean"]
        assert by[("mean effective rank", "random")]["mean"] > \
               by[("mean effective rank", "shuffled")]["mean"]

    def test_n_seeds_recorded(self, synthetic_baseline):
        rows = phase2_attribution.build_tier1b_table()
        for r in rows:
            assert r["n_seeds"] == 4

    def test_excludes_eval_loss(self, synthetic_baseline):
        """Eval loss is a stored checkpoint metadata field, not recomputed
        on the Tier 1b inputs. The rewrite excludes it to avoid the
        false impression of input-dependent loss."""
        rows = phase2_attribution.build_tier1b_table()
        names = {r["statistic"] for r in rows}
        assert "eval loss (final)" not in names
        # H1 statistics are trajectory-derived and also excluded.
        assert "H1: last-quarter std" not in names
        assert "H1: total reduction" not in names

    def test_includes_expected_statistics(self, synthetic_baseline):
        rows = phase2_attribution.build_tier1b_table()
        names = {r["statistic"] for r in rows}
        for expected in ("λ (ours)", "λ (paper)",
                          "log α (ours)", "log α (paper)",
                          "<κ> (signed mean kurt)", "<|κ|> (paper kurt)",
                          "mean effective rank",
                          "eff rank L=0", "eff rank mid"):
            assert expected in names, f"missing statistic: {expected}"

    def test_unknown_distribution_raises(self, synthetic_baseline):
        with pytest.raises(ValueError, match="Unknown input distribution"):
            phase2_attribution.build_tier1b_table(
                input_distributions=("real", "made_up_dist"),
            )

    def test_missing_distribution_handled(self, tmp_path, monkeypatch):
        """If a distribution's subdir doesn't exist at all, that
        distribution should be silently skipped (with a warning), not
        crash. We build a root with ONLY the real distribution."""
        root = tmp_path / "phase1_runs_gelu"
        root.mkdir()
        for seed in range(2):
            sd = root / f"seed_{seed}" / "flow_analysis"
            sd.mkdir(parents=True)
            np.savez(sd / "flow_step_24000.npz",
                     **{
                         "lambda": np.float32(0.35),
                         "lambda_paper": np.float32(0.34),
                         "log_alpha": np.float32(-3.3),
                         "log_alpha_paper": np.float32(-3.3),
                         "kurtosis_per_layer": np.array([0.9], dtype=np.float32),
                         "kurtosis_abs_per_layer": np.array([0.9], dtype=np.float32),
                         "effective_rank": np.array([180], dtype=np.float32),
                         "checkpoint_step": 24000,
                         "checkpoint_eval_loss": np.float32(2.92),
                         "checkpoint_loss": np.float32(2.85),
                         "checkpoint_seed": seed,
                         "num_layers_total": 1,
                         "hidden_dim": 896,
                     })
        monkeypatch.setattr(phase2_attribution, "PHASE1_GELU_ROOT", str(root))
        rows = phase2_attribution.build_tier1b_table()
        dists = {r["input_distribution"] for r in rows}
        # Real should be present; shuffled / random should be absent
        # (and have produced a warning, but not crashed).
        assert "real" in dists
        assert "shuffled" not in dists
        assert "random" not in dists


class TestBuildPerVariantTier1bTable:
    """Tests for the per-variant Tier 1b extension."""

    @pytest.fixture
    def synthetic_full(self, tmp_path, monkeypatch):
        """Build both phase1_runs_gelu (baseline, 4 seeds) AND a
        phase2_runs/depth/L06/ variant (2 seeds), each with 3 input
        distributions. Different λ per (target, distribution) so we can
        verify the right files get read."""
        import numpy as np

        # Phase 1 GELU baseline.
        baseline_root = tmp_path / "phase1_runs_gelu"
        baseline_root.mkdir()
        # Phase 2 variant.
        phase2_root = tmp_path / "phase2_runs"
        phase2_root.mkdir()
        (phase2_root / "depth").mkdir()
        (phase2_root / "depth" / "L06").mkdir()

        subdirs = ("flow_analysis", "flow_analysis_shuffled",
                   "flow_analysis_random")
        # baseline: λ = 0.36 + offset
        # L06 variant: λ = 0.73 + offset (heavier per-layer flow due to fewer layers)
        for seed in range(4):
            for k, sub in enumerate(subdirs):
                sd = baseline_root / f"seed_{seed}" / sub
                sd.mkdir(parents=True)
                np.savez(sd / "flow_step_24000.npz", **{
                    "lambda": np.float32(0.36 + 0.1 * k + 0.001 * seed),
                    "lambda_paper": np.float32(0.34 + 0.1 * k),
                    "log_alpha": np.float32(-3.3 + 0.05 * k),
                    "log_alpha_paper": np.float32(-3.3 + 0.05 * k),
                    "kurtosis_per_layer": np.array([0.9 + 0.05 * k], dtype=np.float32),
                    "kurtosis_abs_per_layer": np.array([0.9 + 0.05 * k], dtype=np.float32),
                    "effective_rank": np.array([180 + 50 * k, 480], dtype=np.float32),
                    "checkpoint_step": 24000,
                    "checkpoint_eval_loss": np.float32(2.92),
                    "checkpoint_loss": np.float32(2.85),
                    "checkpoint_seed": seed,
                    "num_layers_total": 2,
                    "hidden_dim": 896,
                })
        for seed in range(2):
            for k, sub in enumerate(subdirs):
                sd = phase2_root / "depth" / "L06" / f"seed_{seed}" / sub
                sd.mkdir(parents=True)
                np.savez(sd / "flow_step_24000.npz", **{
                    "lambda": np.float32(0.73 + 0.1 * k + 0.001 * seed),
                    "lambda_paper": np.float32(0.67 + 0.1 * k),
                    "log_alpha": np.float32(-4.0 + 0.05 * k),
                    "log_alpha_paper": np.float32(-4.0 + 0.05 * k),
                    "kurtosis_per_layer": np.array([1.0 + 0.05 * k], dtype=np.float32),
                    "kurtosis_abs_per_layer": np.array([1.0 + 0.05 * k], dtype=np.float32),
                    "effective_rank": np.array([186 + 50 * k, 427], dtype=np.float32),
                    "checkpoint_step": 24000,
                    "checkpoint_eval_loss": np.float32(3.04),
                    "checkpoint_loss": np.float32(2.97),
                    "checkpoint_seed": seed,
                    "num_layers_total": 2,
                    "hidden_dim": 896,
                })

        monkeypatch.setattr(phase2_attribution, "PHASE1_GELU_ROOT", str(baseline_root))
        monkeypatch.setattr(phase2_attribution, "PHASE2_ROOT", str(phase2_root))
        # phase2_launch's run_dir_for reads PHASE2_ROOT from phase2_launch,
        # but the per-variant table uses the imported PHASE2_ROOT in
        # phase2_attribution. Sync them.
        import phase2_launch
        monkeypatch.setattr(phase2_launch, "PHASE2_ROOT", str(phase2_root))
        return baseline_root, phase2_root

    def test_includes_baseline_and_variants(self, synthetic_full):
        rows = phase2_attribution.build_per_variant_tier1b_table()
        targets = {r["target"] for r in rows}
        assert "baseline" in targets
        assert "L06" in targets

    def test_per_target_per_distribution_means(self, synthetic_full):
        rows = phase2_attribution.build_per_variant_tier1b_table()
        # Index by (target, dist, stat).
        get = {(r["target"], r["input_distribution"], r["statistic"]): r
               for r in rows}
        # baseline real should be ~0.36; L06 real should be ~0.73.
        assert abs(get[("baseline", "real", "λ (ours)")]["mean"] - 0.36) < 0.01
        assert abs(get[("L06", "real", "λ (ours)")]["mean"] - 0.73) < 0.01
        # baseline shuffled should be > baseline real (synthetic offset).
        assert get[("baseline", "shuffled", "λ (ours)")]["mean"] > \
               get[("baseline", "real", "λ (ours)")]["mean"]
        # L06 shuffled > L06 real, same pattern.
        assert get[("L06", "shuffled", "λ (ours)")]["mean"] > \
               get[("L06", "real", "λ (ours)")]["mean"]

    def test_baseline_excluded_when_requested(self, synthetic_full):
        rows = phase2_attribution.build_per_variant_tier1b_table(
            include_baseline=False,
        )
        targets = {r["target"] for r in rows}
        assert "baseline" not in targets
        assert "L06" in targets

    def test_n_seeds_recorded_correctly(self, synthetic_full):
        rows = phase2_attribution.build_per_variant_tier1b_table()
        # baseline has 4 seeds, L06 has 2.
        by_target_seeds = {r["target"]: r["n_seeds"] for r in rows
                            if r["statistic"] == "λ (ours)"
                            and r["input_distribution"] == "real"}
        assert by_target_seeds["baseline"] == 4
        assert by_target_seeds["L06"] == 2

    def test_render_runs(self, synthetic_full):
        rows = phase2_attribution.build_per_variant_tier1b_table()
        text = phase2_attribution.render_per_variant_tier1b_text(rows)
        # Sanity: text mentions both targets and computes Δ percentages.
        assert "baseline" in text
        assert "L06" in text
        assert "%" in text  # the Δshuf/|real| column

    def test_missing_variant_subdir_skipped(self, tmp_path, monkeypatch):
        """If a variant lacks shuffled flows, those rows should be absent
        but real-input rows should still appear."""
        import numpy as np
        baseline_root = tmp_path / "phase1_runs_gelu"
        baseline_root.mkdir()
        phase2_root = tmp_path / "phase2_runs"
        (phase2_root / "depth" / "L06").mkdir(parents=True)
        # Baseline: all 3 distributions, 2 seeds.
        for seed in range(2):
            for sub in ("flow_analysis", "flow_analysis_shuffled",
                        "flow_analysis_random"):
                sd = baseline_root / f"seed_{seed}" / sub
                sd.mkdir(parents=True)
                np.savez(sd / "flow_step_24000.npz", **{
                    "lambda": np.float32(0.36),
                    "lambda_paper": np.float32(0.34),
                    "log_alpha": np.float32(-3.3),
                    "log_alpha_paper": np.float32(-3.3),
                    "kurtosis_per_layer": np.array([0.9], dtype=np.float32),
                    "kurtosis_abs_per_layer": np.array([0.9], dtype=np.float32),
                    "effective_rank": np.array([180], dtype=np.float32),
                    "checkpoint_step": 24000,
                    "checkpoint_eval_loss": np.float32(2.92),
                    "checkpoint_loss": np.float32(2.85),
                    "checkpoint_seed": seed,
                    "num_layers_total": 1,
                    "hidden_dim": 896,
                })
        # L06 variant: ONLY real, no shuffled or random.
        for seed in range(2):
            sd = phase2_root / "depth" / "L06" / f"seed_{seed}" / "flow_analysis"
            sd.mkdir(parents=True)
            np.savez(sd / "flow_step_24000.npz", **{
                "lambda": np.float32(0.73),
                "lambda_paper": np.float32(0.67),
                "log_alpha": np.float32(-4.0),
                "log_alpha_paper": np.float32(-4.0),
                "kurtosis_per_layer": np.array([1.0], dtype=np.float32),
                "kurtosis_abs_per_layer": np.array([1.0], dtype=np.float32),
                "effective_rank": np.array([186], dtype=np.float32),
                "checkpoint_step": 24000,
                "checkpoint_eval_loss": np.float32(3.04),
                "checkpoint_loss": np.float32(2.97),
                "checkpoint_seed": seed,
                "num_layers_total": 1,
                "hidden_dim": 896,
            })

        monkeypatch.setattr(phase2_attribution, "PHASE1_GELU_ROOT", str(baseline_root))
        monkeypatch.setattr(phase2_attribution, "PHASE2_ROOT", str(phase2_root))
        import phase2_launch
        monkeypatch.setattr(phase2_launch, "PHASE2_ROOT", str(phase2_root))

        rows = phase2_attribution.build_per_variant_tier1b_table()
        # baseline should have all 3 distributions.
        baseline_dists = {r["input_distribution"] for r in rows
                          if r["target"] == "baseline"}
        assert baseline_dists == {"real", "shuffled", "random"}
        # L06 should have only real.
        l06_dists = {r["input_distribution"] for r in rows
                     if r["target"] == "L06"}
        assert l06_dists == {"real"}


class TestPerSeedSignConsistency:
    """Tests for the _per_seed_sign_consistency helper and its surfacing
    in the renderers."""

    def test_sign_consistent_when_all_seeds_agree(self):
        """Synthetic rows: all 4 seeds give Δ < 0 → sign-consistent."""
        rows = [
            {"target": "baseline", "input_distribution": "real",
             "statistic": "λ",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             "per_seed_values": [0.36, 0.36, 0.36, 0.36],
             "mean": 0.36, "std": 0.0, "n_seeds": 4},
            {"target": "baseline", "input_distribution": "shuffled",
             "statistic": "λ",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             "per_seed_values": [0.30, 0.29, 0.30, 0.29],
             "mean": 0.295, "std": 0.005, "n_seeds": 4},
        ]
        info = phase2_attribution._per_seed_sign_consistency(
            rows, target_filter="baseline",
        )
        cell = info[("baseline", "shuffled", "λ")]
        assert cell["sign_consistent"] is True
        assert cell["n_seeds_paired"] == 4
        # All deltas should be negative.
        assert all(d < 0 for d in cell["per_seed_delta"])

    def test_sign_inconsistent_when_seeds_disagree(self):
        """The exact case from the user's data: log α (paper) shows
        2 negative and 2 positive Δs."""
        rows = [
            {"target": "baseline", "input_distribution": "real",
             "statistic": "log α (paper)",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             "per_seed_values": [-3.3331, -3.3325, -3.3305, -3.2972],
             "mean": -3.3233, "std": 0.017, "n_seeds": 4},
            {"target": "baseline", "input_distribution": "shuffled",
             "statistic": "log α (paper)",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             "per_seed_values": [-3.3519, -3.3213, -3.3067, -3.3085],
             "mean": -3.3221, "std": 0.020, "n_seeds": 4},
        ]
        info = phase2_attribution._per_seed_sign_consistency(
            rows, target_filter="baseline",
        )
        cell = info[("baseline", "shuffled", "log α (paper)")]
        assert cell["sign_consistent"] is False
        assert cell["n_seeds_paired"] == 4
        # Two negative, two positive.
        positives = sum(1 for d in cell["per_seed_delta"] if d > 0)
        negatives = sum(1 for d in cell["per_seed_delta"] if d < 0)
        assert positives == 2 and negatives == 2

    def test_none_when_fewer_than_two_seeds(self):
        rows = [
            {"target": "baseline", "input_distribution": "real",
             "statistic": "λ",
             "seed_labels": ["seed_0"],
             "per_seed_values": [0.36],
             "mean": 0.36, "std": float("nan"), "n_seeds": 1},
            {"target": "baseline", "input_distribution": "shuffled",
             "statistic": "λ",
             "seed_labels": ["seed_0"],
             "per_seed_values": [0.30],
             "mean": 0.30, "std": float("nan"), "n_seeds": 1},
        ]
        info = phase2_attribution._per_seed_sign_consistency(
            rows, target_filter="baseline",
        )
        cell = info[("baseline", "shuffled", "λ")]
        assert cell["sign_consistent"] is None
        assert cell["n_seeds_paired"] == 1

    def test_handles_nan_seeds(self):
        """If some seeds are NaN (missing for one distribution),
        sign-consistency uses only the non-NaN per-seed Δs."""
        rows = [
            {"target": "baseline", "input_distribution": "real",
             "statistic": "λ",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             "per_seed_values": [0.36, 0.36, 0.36, 0.36],
             "mean": 0.36, "std": 0.0, "n_seeds": 4},
            {"target": "baseline", "input_distribution": "shuffled",
             "statistic": "λ",
             "seed_labels": ["seed_0", "seed_1", "seed_2", "seed_3"],
             # seeds 0 and 1 missing this distribution.
             "per_seed_values": [float("nan"), float("nan"), 0.30, 0.29],
             "mean": 0.295, "std": 0.005, "n_seeds": 2},
        ]
        info = phase2_attribution._per_seed_sign_consistency(
            rows, target_filter="baseline",
        )
        cell = info[("baseline", "shuffled", "λ")]
        assert cell["sign_consistent"] is True
        assert cell["n_seeds_paired"] == 2

    def test_baseline_renderer_includes_sign_marker(self, tmp_path, monkeypatch):
        """Real end-to-end: write a synthetic baseline where seeds
        disagree on sign for one statistic, and check the rendered text
        contains a ✗ marker."""
        import numpy as np
        baseline_root = tmp_path / "phase1_runs_gelu"
        baseline_root.mkdir()
        # Construct per-seed log α (paper) values matching the user's data.
        seed_real = {0: -3.3331, 1: -3.3325, 2: -3.3305, 3: -3.2972}
        seed_shuf = {0: -3.3519, 1: -3.3213, 2: -3.3067, 3: -3.3085}
        for seed in range(4):
            for sub, val_table in (("flow_analysis", seed_real),
                                    ("flow_analysis_shuffled", seed_shuf)):
                sd = baseline_root / f"seed_{seed}" / sub
                sd.mkdir(parents=True)
                np.savez(sd / "flow_step_24000.npz", **{
                    "lambda": np.float32(0.36),
                    "lambda_paper": np.float32(0.34),
                    "log_alpha": np.float32(-3.3),
                    "log_alpha_paper": np.float32(val_table[seed]),
                    "kurtosis_per_layer": np.array([0.9], dtype=np.float32),
                    "kurtosis_abs_per_layer": np.array([0.9], dtype=np.float32),
                    "effective_rank": np.array([180], dtype=np.float32),
                    "checkpoint_step": 24000,
                    "checkpoint_eval_loss": np.float32(2.92),
                    "checkpoint_loss": np.float32(2.85),
                    "checkpoint_seed": seed,
                    "num_layers_total": 1,
                    "hidden_dim": 896,
                })
        monkeypatch.setattr(phase2_attribution, "PHASE1_GELU_ROOT",
                             str(baseline_root))
        rows = phase2_attribution.build_tier1b_table(
            input_distributions=("real", "shuffled"),
        )
        text = phase2_attribution.render_tier1b_text(rows)
        # log α (paper) shuffled column should have ✗ (sign-inconsistent).
        # Find the log α (paper) row.
        for line in text.splitlines():
            if line.startswith("log α (paper)"):
                assert "✗" in line, (
                    f"Expected ✗ marker on sign-inconsistent log α (paper) "
                    f"shuffled cell. Got:\n{line}"
                )
                break
        else:
            assert False, "log α (paper) row not found in rendered text"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
    