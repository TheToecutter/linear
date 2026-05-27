"""
Phase 2 unattended queue runner.

Orchestrates the full Phase 2 Tier 1a + 1b pipeline as a single,
failure-tolerant, priority-ordered sequence designed to run unattended
for multiple days.

What this script does
---------------------
For every (variant, seed) pair in a priority-ordered list:
  1. Train (skipping if checkpoints/run_metadata.json already present).
  2. Immediately analyze the resulting run (flow recovery, 50 ckpts).
  3. On failure (OOM, NaN, anything raising), record the failure to
     the queue ledger and move to the next item without stopping.

After all training+analysis is done (or as much as fits in the time
budget), runs the Tier 1b shuffled/random analysis against every
variant that finished — including the Phase 1 GELU baseline.

Priority ordering
-----------------
Variants are ordered heaviest-first, with BOTH SEEDS of each variant
running back-to-back before moving to the next variant. The
heaviest-first order assumes memfit pre-checks have validated that
the big variants fit; under that assumption the dominant failure
mode is gone and front-loading the heavyweights means they're done
early rather than perched at the tail where any stall would block
Tier 1b.

The seed-pairing means an interrupted queue leaves you with COMPLETE
2-seed measurements for some variants — exactly what the attribution
matrix consumes — rather than 1-seed-each scattered across all
variants.

  Pass 1 (always first): memfit pre-checks for H1792 and L24.
  Pass 2 (variant pairs, heaviest-first):
    H1792    (527M) seed 0 → seed 1   ← biggest, memfit-gated, front
    L24      (263M) seed 0 → seed 1
    ffn_3p0x (126M) seed 0 → seed 1
    ffn_1p5x ( 97M) seed 0 → seed 1
    L06      ( 88M) seed 0 → seed 1
    H0448    ( 44M) seed 0 → seed 1   ← smallest, sails through at tail
  Pass 3: Tier 1b — shuffled + random input distributions, baseline
    + every Tier-1a variant that finished.

Logging
-------
All output is teed:
  - stdout/stderr (so a live SSH session sees progress).
  - <out_dir>/queue.log              (master orchestrator log).
  - <out_dir>/<task_id>.log          (per-task isolated log).
  - <out_dir>/queue_status.json      (machine-readable status table,
                                       updated after every task).

Status JSON
-----------
Lets you (or a downstream script) check on the queue's progress
without parsing logs. Schema:
  {
    "started_at": <iso>,
    "updated_at": <iso>,
    "tasks": [
      {"id": "...", "kind": "memfit|train_and_analyze|tier1b",
       "status": "queued|running|done|failed|skipped",
       "started_at": ..., "ended_at": ..., "elapsed_s": ...,
       "log_path": "...", "error": "..." (if failed) },
      ...
    ]
  }

Usage
-----
    # Dry-run to preview the full queue:
    python3 phase2_queue.py --dry_run

    # Launch unattended (recommended: under nohup or tmux):
    nohup python3 phase2_queue.py > phase2_queue_main.out 2>&1 &

    # Restart after an interruption (idempotent — skips done items):
    python3 phase2_queue.py

    # Restrict to one pass for testing:
    python3 phase2_queue.py --max_pass 1

    # Skip memfit (e.g. if you've already validated H1792 fits):
    python3 phase2_queue.py --skip_memfit
"""

import argparse
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from phase2_configs import (
    ALL_TIER1_VARIANTS, find_variant, VariantSpec,
)
from phase2_launch import run_dir_for, PHASE2_ROOT, check_baseline_present


DEFAULT_OUT_DIR = "phase2_queue_logs"
STATUS_FILENAME = "queue_status.json"
MASTER_LOG_FILENAME = "queue.log"


# ----------------------------------------------------------------------
# Task descriptor.
# ----------------------------------------------------------------------
@dataclass
class QueueTask:
    """One task in the queue. Tasks are typed because their idempotency
    check, status update, and logging differ slightly across kinds."""
    id: str
    kind: str                                  # memfit | train_and_analyze | tier1b
    runner: Callable[[Path], None]             # called with a log Path; raises on failure
    is_done_check: Callable[[], bool]          # idempotency: returns True if work already done
    status: str = "queued"
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    elapsed_s: float = 0.0
    log_path: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        # Exclude non-serializable runner/is_done_check callables.
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "elapsed_s": self.elapsed_s, "log_path": self.log_path,
            "error": self.error,
        }


# ----------------------------------------------------------------------
# Logging: tee stdout/stderr to a per-task file while preserving the
# console stream. Importing modules use print(), which goes to whatever
# sys.stdout is currently bound to.
# ----------------------------------------------------------------------
class _Tee:
    """Write to multiple file-like sinks. Used to tee stdout to disk."""
    def __init__(self, *sinks):
        self.sinks = sinks
    def write(self, data):
        for s in self.sinks:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.sinks:
            try:
                s.flush()
            except Exception:
                pass
    def isatty(self):
        # Some libraries (transformers progress bars) check this.
        return False


@contextmanager
def tee_to_file(log_path: Path):
    """Redirect stdout+stderr to both the original streams and a file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "a", buffering=1)  # line-buffered
    fh.write(f"\n{'='*72}\n>> log opened at {datetime.now(timezone.utc).isoformat()}\n{'='*72}\n")
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)
    try:
        yield fh
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()


# ----------------------------------------------------------------------
# Status persistence.
# ----------------------------------------------------------------------
class QueueLedger:
    """Persists the queue's state to JSON after every task transition.

    Threadsafe is not a concern -- the queue is single-threaded.
    """

    def __init__(self, path: Path):
        self.path = path
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.tasks: List[QueueTask] = []

    def set_tasks(self, tasks: List[QueueTask]):
        self.tasks = tasks
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "started_at": self.started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "tasks": [t.to_dict() for t in self.tasks],
                "summary": self._summary(),
            }, f, indent=2)

    def _summary(self) -> dict:
        out = {"queued": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
        for t in self.tasks:
            out[t.status] = out.get(t.status, 0) + 1
        return out


# ----------------------------------------------------------------------
# Idempotency helpers.
# ----------------------------------------------------------------------
def _has_trained(variant: VariantSpec, seed: int) -> bool:
    """True iff training has produced run_metadata.json + at least one
    checkpoint for the given (variant, seed)."""
    rd = Path(run_dir_for(variant, seed))
    if not (rd / "run_metadata.json").exists():
        return False
    ckpt_dir = rd / "checkpoints"
    if not ckpt_dir.is_dir():
        return False
    return any(ckpt_dir.glob("step_*.pt"))


def _has_analyzed(variant: VariantSpec, seed: int) -> bool:
    """True iff flow_analysis/ has at least one flow .npz file."""
    rd = Path(run_dir_for(variant, seed))
    flow_dir = rd / "flow_analysis"
    if not flow_dir.is_dir():
        return False
    return any(flow_dir.glob("flow_step_*.npz"))


def _has_memfit_notes(variant: VariantSpec) -> bool:
    """True iff a MEMFIT_NOTES.txt has been written for this variant."""
    var_dir = Path(run_dir_for(variant, seed=0)).parent
    return (var_dir / "MEMFIT_NOTES.txt").exists()


def _has_tier1b(variant_or_baseline_dir: str) -> bool:
    """True iff both shuffled and random Tier-1b flows have been
    computed for the given run dir."""
    d = Path(variant_or_baseline_dir)
    shuf = d / "flow_analysis_shuffled"
    rand = d / "flow_analysis_random"
    has_shuf = shuf.is_dir() and any(shuf.glob("flow_step_*.npz"))
    has_rand = rand.is_dir() and any(rand.glob("flow_step_*.npz"))
    return has_shuf and has_rand


# ----------------------------------------------------------------------
# Task builders.
# ----------------------------------------------------------------------
def build_memfit_task(variant: VariantSpec, out_dir: Path) -> QueueTask:
    """Memory-fit pre-check for one heavyweight variant.

    Writes MEMFIT_NOTES.txt in the variant's parent dir on success;
    raises on failure (all fallback batch shapes OOMed).
    """
    task_id = f"memfit_{variant.label}"
    log_path = out_dir / f"{task_id}.log"

    def runner(log: Path):
        # Lazy-import: phase2_memfit pulls torch.
        from phase2_memfit import fit_check_variant
        notes = fit_check_variant(
            variant=variant, probe_steps=50, device="cuda", write=True,
        )
        if notes is None:
            raise RuntimeError(
                f"memfit for {variant.label}: all fallback batch shapes OOMed."
            )

    return QueueTask(
        id=task_id, kind="memfit",
        runner=runner,
        is_done_check=lambda: _has_memfit_notes(variant),
        log_path=str(log_path),
    )


def build_train_and_analyze_task(
    variant: VariantSpec, seed: int, out_dir: Path,
) -> QueueTask:
    """One full (train → analyze) cycle for a (variant, seed) pair.

    Training picks up MEMFIT_NOTES.txt overrides automatically via
    phase2_launch.launch_one(apply_memfit=True).

    Analysis is run immediately after training so the attribution
    matrix is partially valid even if a later variant fails.
    """
    task_id = f"train_and_analyze_{variant.label}_seed{seed}"
    log_path = out_dir / f"{task_id}.log"

    def runner(log: Path):
        # Step 1: train. Idempotency is already in launch_one (it refuses
        # to clobber existing run dirs without --force), but the queue's
        # is_done_check catches the case before we ever call launch_one.
        if not _has_trained(variant, seed):
            from phase2_launch import launch_one
            launch_one(
                variant=variant, seed=seed,
                total_steps=None,            # default 24000
                dry_run=False, force=False,
                device="cuda", num_proc=None,
                apply_memfit=True,
            )
        else:
            print(f">> {variant.label}/seed_{seed}: training already done; "
                  f"skipping to analysis.")

        # Step 2: analyze. Uses the same code path as phase2_analyze.py
        # but with filters narrowed to the one variant we just trained.
        if not _has_analyzed(variant, seed):
            from phase2_analyze import analyze_variants
            analyze_variants(
                only_variant=variant.label,
                device="cuda", num_proc=None,
                skip_existing=True,
            )
        else:
            print(f">> {variant.label}/seed_{seed}: analysis already done.")

    return QueueTask(
        id=task_id, kind="train_and_analyze",
        runner=runner,
        is_done_check=lambda: (_has_trained(variant, seed)
                               and _has_analyzed(variant, seed)),
        log_path=str(log_path),
    )


def build_tier1b_task(out_dir: Path) -> QueueTask:
    """Tier 1b: shuffled + random analysis on the baseline + every
    Tier-1a variant that has trained. The phase2_analyze.run_tier1b
    function handles iteration over what's available.
    """
    task_id = "tier1b_all"
    log_path = out_dir / f"{task_id}.log"

    def runner(log: Path):
        from phase2_analyze import run_tier1b
        run_tier1b(
            include_baseline=True,
            device="cuda", num_proc=None,
            skip_existing=True,
            input_distributions=("shuffled", "random"),
        )

    def is_done():
        # Done when every variant + baseline that has flow_analysis/
        # also has flow_analysis_shuffled/ and flow_analysis_random/.
        # If nothing has been trained yet, return False so the task
        # at least runs once to confirm there's nothing to do.
        from phase2_analyze import find_baseline_run_dirs, find_variant_run_dirs
        targets = find_baseline_run_dirs() + find_variant_run_dirs(
            ALL_TIER1_VARIANTS
        )
        if not targets:
            return False
        return all(_has_tier1b(d) for d in targets)

    return QueueTask(
        id=task_id, kind="tier1b",
        runner=runner,
        is_done_check=is_done,
        log_path=str(log_path),
    )


# ----------------------------------------------------------------------
# Queue construction.
# ----------------------------------------------------------------------
# Variant ordering within the queue. Heaviest-first: if memfit passes,
# H1792 is the only real OOM risk, and the dominant failure mode is
# already gated. Front-loading the heaviest means it's done early in
# the run rather than perched at the end where a late stall would
# block tier1b. Cheap variants then sail through at the tail.
#
# If the memfit-gated assumption is wrong (i.e. H1792 fails for a
# non-memory reason like loss NaN or a kernel issue), this ordering
# loses more queue time before falling back to cheap variants. The
# alternative is cheapest-first; revisit if the failure profile
# changes.
SEED_PASS_ORDER: List[str] = [
    "H1792",      # 527M — heaviest, highest risk; memfit-gated
    "L24",        # 263M — heavyweight #2
    "L18",        # 200M — added for the depth-axis 4th data point
    "ffn_3p0x",   # 126M
    "ffn_1p5x",   #  97M
    "L06",        #  88M
    "H0448",      #  44M — smallest, cheapest
]

# Sanity check: SEED_PASS_ORDER must cover every Tier-1a variant exactly once.
_known_labels = {v.label for v in ALL_TIER1_VARIANTS}
assert set(SEED_PASS_ORDER) == _known_labels, (
    f"SEED_PASS_ORDER {SEED_PASS_ORDER} doesn't match Tier-1a variants "
    f"{sorted(_known_labels)}. Was a variant added or renamed?"
)


def build_queue(
    out_dir: Path,
    skip_memfit: bool = False,
    skip_tier1b: bool = False,
    max_pass: Optional[int] = None,
) -> List[QueueTask]:
    """Construct the ordered task list.

    Ordering: heaviest-first variants, with both seeds of each variant
    back-to-back. The heaviest-first order assumes memfit has gated
    the dominant failure mode; the seed-pairing means an interrupted
    queue leaves you with COMPLETE 2-seed measurements for some
    variants rather than 1-seed-each across all variants.

    max_pass uses 1-based counting matching the CLI help text:
      1 = memfit only
      2 = memfit + variant pairs
      3 = memfit + variant pairs + Tier 1b
    """
    tasks: List[QueueTask] = []
    passes_done = 0

    # Pass 1: memfit on heavyweight variants.
    if not skip_memfit:
        for label in ("L24", "H1792"):  # L24 first (smaller) so an
                                          # unexpected OOM on H1792 doesn't
                                          # block L24.
            tasks.append(build_memfit_task(find_variant(label), out_dir))
    passes_done += 1
    if max_pass is not None and passes_done >= max_pass:
        return tasks

    # Pass 2: variants in heaviest-first order; both seeds back-to-back
    # per variant. An interrupted run leaves completed variants as
    # complete 2-seed measurements, which is exactly what the
    # attribution matrix consumes. The heaviest variants front-load so
    # they finish early; smaller variants sail through at the tail.
    for label in SEED_PASS_ORDER:
        v = find_variant(label)
        for seed in v.seeds:
            tasks.append(
                build_train_and_analyze_task(v, seed=seed, out_dir=out_dir)
            )
    passes_done += 1
    if max_pass is not None and passes_done >= max_pass:
        return tasks

    # Pass 3: Tier 1b at the end. Cheap; runs against whatever finished.
    if not skip_tier1b:
        tasks.append(build_tier1b_task(out_dir))

    return tasks


# ----------------------------------------------------------------------
# Queue runner.
# ----------------------------------------------------------------------
def run_queue(tasks: List[QueueTask], out_dir: Path,
              dry_run: bool = False) -> int:
    """Execute the queue. Returns the number of failed tasks.

    Continues past failures (logging them) so an unattended run gets
    as much done as possible. Re-running is safe: each task's
    is_done_check is consulted before execution, and done tasks
    transition to status="skipped" rather than re-running.
    """
    ledger = QueueLedger(out_dir / STATUS_FILENAME)
    ledger.set_tasks(tasks)
    master_log = out_dir / MASTER_LOG_FILENAME

    failed = 0
    queue_start = time.time()
    with tee_to_file(master_log):
        print(f">> Phase 2 queue: {len(tasks)} tasks, out_dir={out_dir}")
        for i, task in enumerate(tasks):
            elapsed_total_h = (time.time() - queue_start) / 3600
            print(f"\n{'='*72}\n"
                  f"[{i+1}/{len(tasks)}] {task.id}  (kind={task.kind})  "
                  f"  [queue elapsed: {elapsed_total_h:.2f}h]\n{'='*72}")

            # Idempotency.
            try:
                already_done = task.is_done_check()
            except Exception as e:
                print(f"   ⚠️  is_done_check raised: {e}; will attempt task.")
                already_done = False

            if already_done:
                print(f"   ↳ already done; skipping.")
                task.status = "skipped"
                ledger.save()
                continue

            if dry_run:
                print(f"   ↳ dry run; not executed.")
                task.status = "skipped"
                ledger.save()
                continue

            task.status = "running"
            task.started_at = datetime.now(timezone.utc).isoformat()
            ledger.save()
            t0 = time.time()

            try:
                # Each task gets its own log file *in addition to* the
                # master tee. Nested tee_to_file is fine because the
                # outer context restores stdout/stderr after the inner
                # context exits.
                if task.log_path:
                    with tee_to_file(Path(task.log_path)):
                        task.runner(Path(task.log_path))
                else:
                    task.runner(out_dir / f"{task.id}.log")
                task.status = "done"
                print(f"   ↳ ✅ done in {(time.time()-t0)/60:.1f} min")
            except KeyboardInterrupt:
                # Allow ctrl-C to bail out cleanly.
                task.status = "failed"
                task.error = "KeyboardInterrupt"
                task.ended_at = datetime.now(timezone.utc).isoformat()
                task.elapsed_s = time.time() - t0
                ledger.save()
                print("\n>> KeyboardInterrupt — stopping queue.")
                raise
            except Exception as e:
                task.status = "failed"
                task.error = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(f"   ↳ ❌ FAILED after {(time.time()-t0)/60:.1f} min")
                print(f"      {type(e).__name__}: {e}")
                print(tb)
                # Also write the traceback to the task log if we have one.
                if task.log_path:
                    try:
                        with open(task.log_path, "a") as f:
                            f.write(f"\n>> TRACEBACK\n{tb}\n")
                    except Exception:
                        pass
                failed += 1
            finally:
                task.ended_at = datetime.now(timezone.utc).isoformat()
                task.elapsed_s = time.time() - t0
                ledger.save()

        # Final summary.
        n_done = sum(1 for t in tasks if t.status == "done")
        n_skipped = sum(1 for t in tasks if t.status == "skipped")
        n_failed = sum(1 for t in tasks if t.status == "failed")
        print(f"\n{'='*72}\n>> Queue complete: {n_done} done, "
              f"{n_skipped} skipped, {n_failed} failed.")
        print(f">> Status: {out_dir / STATUS_FILENAME}")
        print(f">> Master log: {master_log}")

    return failed


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 unattended queue runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument(
        "--out_dir", type=str, default=DEFAULT_OUT_DIR,
        help=f"Directory for logs + status JSON. Default {DEFAULT_OUT_DIR}/.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the planned queue and exit without running anything.",
    )
    parser.add_argument(
        "--skip_memfit", action="store_true",
        help="Skip the memfit pre-checks (Pass 0). Use this only if you "
             "have already validated that H1792 and L24 fit on the "
             "target GPU.",
    )
    parser.add_argument(
        "--skip_tier1b", action="store_true",
        help="Skip the Tier 1b shuffled/random analysis at the end "
             "(Pass 3). Useful if you only care about the cross-axis "
             "attribution matrix.",
    )
    parser.add_argument(
        "--max_pass", type=int, default=None,
        help="Stop after this many passes (1 = memfit only, 2 = +seed 0, "
             "3 = +seed 1, 4 = +Tier 1b). Default: run all.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity check the baseline before queueing anything depending on it.
    if not args.skip_tier1b:
        print(">> Pre-flight: checking Phase 1 GELU baseline ...")
        check_baseline_present(verbose=True)

    tasks = build_queue(
        out_dir=out_dir,
        skip_memfit=args.skip_memfit,
        skip_tier1b=args.skip_tier1b,
        max_pass=args.max_pass,
    )

    print(f">> Built {len(tasks)} tasks. Order:")
    for i, t in enumerate(tasks):
        print(f"   [{i+1:>3}] {t.id}   (kind={t.kind})")
    if args.dry_run:
        print("\n>> Dry run; not executing.")
        return

    failed = run_queue(tasks, out_dir=out_dir, dry_run=False)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
    