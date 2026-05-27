"""
Tests for phase2_queue.py.

The queue itself is just orchestration — no GPU, no training. We can
test:
  - Queue construction (pass ordering, cheapest-first).
  - Idempotency: tasks with is_done_check returning True are skipped.
  - Failure tolerance: a task that raises doesn't stop the queue.
  - Ledger persistence: status JSON is written and reflects task state.
  - Tee logging: stdout reaches both the master log and the per-task log.

Run with:
    python3 -m pytest test_phase2_queue.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from phase2_queue import (
    QueueTask, QueueLedger, run_queue, build_queue,
    SEED_PASS_ORDER, tee_to_file,
)
from phase2_configs import ALL_TIER1_VARIANTS


# ----------------------------------------------------------------------
# Queue construction.
# ----------------------------------------------------------------------
class TestQueueConstruction:
    def test_seed_pass_order_covers_all_tier1(self):
        labels_in_order = set(SEED_PASS_ORDER)
        labels_in_catalog = {v.label for v in ALL_TIER1_VARIANTS}
        assert labels_in_order == labels_in_catalog

    def test_default_queue_has_memfit_first(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path)
        # First two tasks should be memfit.
        assert tasks[0].kind == "memfit"
        assert tasks[1].kind == "memfit"
        # Then train_and_analyze.
        assert tasks[2].kind == "train_and_analyze"

    def test_default_queue_variant_order_matches_seed_pass_order(self, tmp_path):
        """Variants appear in the queue in SEED_PASS_ORDER (heaviest-first
        by default), with each variant's seeds back-to-back."""
        tasks = build_queue(out_dir=tmp_path)
        train_tasks = [t for t in tasks if t.kind == "train_and_analyze"]
        # Each variant's seeds should appear back-to-back in cheapest-
        # first variant order. So the first 12 train_tasks should be:
        # (ffn_1p5x, seed 0), (ffn_1p5x, seed 1),
        # (ffn_3p0x, seed 0), (ffn_3p0x, seed 1), ... etc.
        expected_variant_order = SEED_PASS_ORDER
        actual_variants_in_order = []
        for t in train_tasks:
            # Extract the variant label from the task id:
            # "train_and_analyze_<label>_seed<n>"
            stripped = t.id.replace("train_and_analyze_", "")
            label = stripped.rsplit("_seed", 1)[0]
            actual_variants_in_order.append(label)
        # Each variant should appear twice in a row.
        for i in range(0, len(actual_variants_in_order), 2):
            assert actual_variants_in_order[i] == actual_variants_in_order[i+1], (
                f"Variant at position {i}/{i+1} not back-to-back: "
                f"{actual_variants_in_order[i:i+2]}"
            )
        # And the order of variants should match SEED_PASS_ORDER.
        unique_in_order = list(dict.fromkeys(actual_variants_in_order))
        assert unique_in_order == expected_variant_order

    def test_default_queue_has_tier1b_last(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path)
        assert tasks[-1].kind == "tier1b"

    def test_skip_memfit_drops_pass_1(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path, skip_memfit=True)
        assert all(t.kind != "memfit" for t in tasks)

    def test_skip_tier1b_drops_last_pass(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path, skip_tier1b=True)
        assert all(t.kind != "tier1b" for t in tasks)

    def test_max_pass_1_stops_after_memfit(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path, max_pass=1)
        # Only memfit tasks.
        assert all(t.kind == "memfit" for t in tasks)
        assert len(tasks) == 2  # L24 and H1792

    def test_max_pass_2_includes_all_train_tasks(self, tmp_path):
        tasks = build_queue(out_dir=tmp_path, max_pass=2)
        train_tasks = [t for t in tasks if t.kind == "train_and_analyze"]
        # All Tier-1 variants × all seeds. Derive expected count from the
        # catalog so this test doesn't break when variants are added.
        from phase2_configs import ALL_TIER1_VARIANTS
        expected = sum(len(v.seeds) for v in ALL_TIER1_VARIANTS)
        assert len(train_tasks) == expected, (
            f"Expected {expected} train tasks (sum of len(v.seeds) for all "
            f"Tier 1 variants), got {len(train_tasks)}."
        )
        # But no Tier 1b yet.
        assert not any(t.kind == "tier1b" for t in tasks)


# ----------------------------------------------------------------------
# Queue execution.
# ----------------------------------------------------------------------
class TestQueueExecution:
    def _make_synthetic_tasks(self, behaviors):
        """behaviors: list of (id, should_skip, should_fail). Returns
        list of QueueTask whose runner raises iff should_fail and whose
        is_done_check returns should_skip."""
        tasks = []
        for tid, should_skip, should_fail in behaviors:
            def make_runner(fail):
                def runner(log):
                    if fail:
                        raise RuntimeError(f"synthetic failure in {tid}")
                    print(f"task {tid} executed")
                return runner
            def make_done_check(skip):
                return lambda: skip
            tasks.append(QueueTask(
                id=tid, kind="synthetic",
                runner=make_runner(should_fail),
                is_done_check=make_done_check(should_skip),
            ))
        return tasks

    def test_done_tasks_get_skipped(self, tmp_path):
        tasks = self._make_synthetic_tasks([
            ("t1", True, False),
            ("t2", False, False),
        ])
        failed = run_queue(tasks, out_dir=tmp_path)
        assert failed == 0
        statuses = {t.id: t.status for t in tasks}
        assert statuses == {"t1": "skipped", "t2": "done"}

    def test_failure_does_not_stop_queue(self, tmp_path):
        tasks = self._make_synthetic_tasks([
            ("t1", False, True),    # fails
            ("t2", False, False),   # but t2 still runs
            ("t3", False, True),    # fails
            ("t4", False, False),   # and t4 still runs
        ])
        failed = run_queue(tasks, out_dir=tmp_path)
        assert failed == 2
        statuses = {t.id: t.status for t in tasks}
        assert statuses["t1"] == "failed"
        assert statuses["t2"] == "done"
        assert statuses["t3"] == "failed"
        assert statuses["t4"] == "done"

    def test_failed_task_records_error_message(self, tmp_path):
        tasks = self._make_synthetic_tasks([("t1", False, True)])
        run_queue(tasks, out_dir=tmp_path)
        assert tasks[0].error is not None
        assert "RuntimeError" in tasks[0].error
        assert "synthetic failure" in tasks[0].error

    def test_ledger_persisted_after_each_task(self, tmp_path):
        tasks = self._make_synthetic_tasks([
            ("t1", False, False),
            ("t2", False, True),
            ("t3", True, False),
        ])
        run_queue(tasks, out_dir=tmp_path)
        ledger_path = tmp_path / "queue_status.json"
        assert ledger_path.exists()
        with open(ledger_path) as f:
            data = json.load(f)
        assert "tasks" in data
        assert "summary" in data
        assert data["summary"]["done"] == 1
        assert data["summary"]["failed"] == 1
        assert data["summary"]["skipped"] == 1


# ----------------------------------------------------------------------
# Tee logging.
# ----------------------------------------------------------------------
class TestTeeLogging:
    def test_tee_writes_to_file_and_stdout(self, tmp_path, capsys):
        log_path = tmp_path / "test.log"
        with tee_to_file(log_path):
            print("hello tee")
        # File should contain the message.
        contents = log_path.read_text()
        assert "hello tee" in contents
        # Stdout should also contain it (via capsys).
        captured = capsys.readouterr()
        assert "hello tee" in captured.out

    def test_tee_restores_stdout_after_context(self, tmp_path):
        original = sys.stdout
        log_path = tmp_path / "test.log"
        with tee_to_file(log_path):
            pass
        assert sys.stdout is original

    def test_tee_restores_stdout_after_exception(self, tmp_path):
        original = sys.stdout
        log_path = tmp_path / "test.log"
        with pytest.raises(ValueError):
            with tee_to_file(log_path):
                raise ValueError("oops")
        assert sys.stdout is original


# ----------------------------------------------------------------------
# Per-task log isolation.
# ----------------------------------------------------------------------
class TestPerTaskLogs:
    def test_each_task_writes_to_its_own_log(self, tmp_path):
        tasks = []
        for tid in ("alpha", "beta", "gamma"):
            tasks.append(QueueTask(
                id=tid, kind="synthetic",
                runner=(lambda x: lambda log: print(f"hello from {x}"))(tid),
                is_done_check=lambda: False,
                log_path=str(tmp_path / f"{tid}.log"),
            ))
        run_queue(tasks, out_dir=tmp_path)
        for tid in ("alpha", "beta", "gamma"):
            log = (tmp_path / f"{tid}.log").read_text()
            assert f"hello from {tid}" in log
            # Other tasks' messages should NOT appear in this log.
            for other in ("alpha", "beta", "gamma"):
                if other != tid:
                    # NOTE: the inner per-task tee runs INSIDE the outer
                    # master tee, so the per-task log only sees output
                    # from when that task is active. The master log
                    # sees everything.
                    assert f"hello from {other}" not in log

    def test_master_log_sees_everything(self, tmp_path):
        tasks = []
        for tid in ("alpha", "beta"):
            tasks.append(QueueTask(
                id=tid, kind="synthetic",
                runner=(lambda x: lambda log: print(f"hello from {x}"))(tid),
                is_done_check=lambda: False,
                log_path=str(tmp_path / f"{tid}.log"),
            ))
        run_queue(tasks, out_dir=tmp_path)
        master = (tmp_path / "queue.log").read_text()
        assert "hello from alpha" in master
        assert "hello from beta" in master


# ----------------------------------------------------------------------
# QueueLedger summary.
# ----------------------------------------------------------------------
class TestLedger:
    def test_summary_counts_match_statuses(self, tmp_path):
        ledger = QueueLedger(tmp_path / "ledger.json")
        ledger.set_tasks([
            QueueTask(id="a", kind="x", runner=None, is_done_check=None,
                      status="done"),
            QueueTask(id="b", kind="x", runner=None, is_done_check=None,
                      status="done"),
            QueueTask(id="c", kind="x", runner=None, is_done_check=None,
                      status="failed"),
            QueueTask(id="d", kind="x", runner=None, is_done_check=None,
                      status="queued"),
        ])
        with open(tmp_path / "ledger.json") as f:
            data = json.load(f)
        assert data["summary"]["done"] == 2
        assert data["summary"]["failed"] == 1
        assert data["summary"]["queued"] == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
    