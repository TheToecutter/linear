"""
Phase 1 multi-seed orchestration.

Drives a multi-seed training campaign with automatic post-training analysis
and plot generation. For each seed:

  1. Run train.py to produce ~50 checkpoints (~7 hours on RTX 5090).
  2. Run analyze_run() to produce flow_analysis/ from the checkpoints.
  3. Run report.py to produce plots/ from the flow analysis.

The script is resumable: if interrupted between seeds, re-running it picks
up where it left off (skipping completed seeds, retrying incomplete ones).
Per-seed state is tracked via files on disk — no separate state database.

Usage:
    # Run seeds 0 through 3 (4 seeds total) at the production config.
    python3 run_phase1.py --seeds 0 1 2 3 --output_root phase1_runs

    # Run a single seed (useful for the first seed, to validate timing).
    python3 run_phase1.py --seeds 0 --output_root phase1_runs

    # Skip analysis/plotting (only training):
    python3 run_phase1.py --seeds 0 1 2 3 --output_root phase1_runs --no_analyze

    # Resume an interrupted campaign — same command, it'll skip completed runs.
    python3 run_phase1.py --seeds 0 1 2 3 --output_root phase1_runs

Output structure:
    phase1_runs/
      orchestration.log       # high-level orchestration log
      orchestration_state.json # current state across seeds
      seed_0/
        run_metadata.json
        train.log
        train.csv
        checkpoints/
          step_*.pt
        flow_analysis/
          flow_step_*.npz
        plots/
          01_loss_curves.png  ... 08_pairwise_residual_heatmap.png
        analyze.log
        report.log
      seed_1/  (same structure)
      seed_2/
      ...

Design choices:

  - Each training run is a fresh subprocess. Crashes in one seed do not
    affect other seeds.
  - Analysis runs inline (in-process), not as a subprocess. This is
    because analysis is much shorter than training, and the process
    isolation isn't worth the overhead.
  - The "is this seed complete?" check looks at three files: the final
    expected checkpoint, the corresponding flow_analysis file, and the
    final expected plot. Each phase is checked independently so the
    script can pick up partway through a seed.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional


# ----------------------------------------------------------------------
# State per-seed.
# ----------------------------------------------------------------------
@dataclass
class SeedState:
    """Status of one seed's run."""
    seed: int
    run_dir: str
    train_status: str = "pending"      # pending | running | done | failed
    analyze_status: str = "pending"    # pending | done | failed | skipped
    plot_status: str = "pending"       # pending | done | failed | skipped
    train_start_time: Optional[float] = None
    train_end_time: Optional[float] = None
    train_elapsed_sec: Optional[float] = None
    analyze_elapsed_sec: Optional[float] = None
    plot_elapsed_sec: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SeedState":
        return cls(**d)


# ----------------------------------------------------------------------
# Completion detection.
# ----------------------------------------------------------------------
def is_training_done(run_dir: str, expected_final_step: Optional[int] = None) -> bool:
    """
    A training run is considered done if:
      - run_metadata.json exists
      - At least one checkpoint exists in checkpoints/
      - If expected_final_step provided, the final-step checkpoint exists
    """
    if not os.path.exists(os.path.join(run_dir, "run_metadata.json")):
        return False
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return False
    ckpts = [f for f in os.listdir(ckpt_dir) if f.startswith("step_") and f.endswith(".pt")]
    if not ckpts:
        return False
    if expected_final_step is not None:
        final_filename = f"step_{expected_final_step:08d}.pt"
        if final_filename not in ckpts:
            return False
    return True


def is_analysis_done(run_dir: str) -> bool:
    """Analysis is done if flow_analysis/ has at least one .npz per checkpoint."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    flow_dir = os.path.join(run_dir, "flow_analysis")
    if not os.path.isdir(flow_dir):
        return False
    ckpts = sorted([f for f in os.listdir(ckpt_dir)
                    if f.startswith("step_") and f.endswith(".pt")])
    flows = sorted([f for f in os.listdir(flow_dir)
                    if f.startswith("flow_step_") and f.endswith(".npz")])
    return len(flows) >= len(ckpts)


def is_plotting_done(run_dir: str, expected_plots: int = 8) -> bool:
    """Plotting is done if plots/ has the expected number of PNGs."""
    plot_dir = os.path.join(run_dir, "plots")
    if not os.path.isdir(plot_dir):
        return False
    pngs = [f for f in os.listdir(plot_dir) if f.endswith(".png")]
    return len(pngs) >= expected_plots


# ----------------------------------------------------------------------
# State persistence.
# ----------------------------------------------------------------------
def load_state(state_path: str) -> Dict[int, SeedState]:
    """Load orchestration state. Returns dict mapping seed -> SeedState."""
    if not os.path.exists(state_path):
        return {}
    with open(state_path) as f:
        raw = json.load(f)
    return {int(s): SeedState.from_dict(d) for s, d in raw.items()}


def save_state(state: Dict[int, SeedState], state_path: str):
    """Persist orchestration state."""
    serialized = {str(s): st.to_dict() for s, st in state.items()}
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(serialized, f, indent=2)
    os.replace(tmp_path, state_path)  # atomic rename


# ----------------------------------------------------------------------
# Orchestration phases.
# ----------------------------------------------------------------------
def run_training(seed: int, run_dir: str, train_extra_args: List[str],
                 log_path: str) -> bool:
    """
    Launch train.py as a subprocess.

    Returns True if training completed successfully.

    Streams output both to the log file and to stdout (prefixed) so the
    user can monitor progress in real time.
    """
    cmd = [
        sys.executable, "train.py",
        "--seed", str(seed),
        "--run_dir", run_dir,
    ] + train_extra_args

    print(f"   ▶ Training command: {' '.join(cmd)}")
    print(f"   ▶ Log: {log_path}")

    with open(log_path, "w", buffering=1) as log_file:
        try:
            # Pipe stdout/stderr; write to both log file and our stdout (prefixed).
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                log_file.write(line)
                print(f"   [seed {seed}] {line}", end="")
            proc.wait()
            return proc.returncode == 0
        except KeyboardInterrupt:
            print(f"\n   ▶ Interrupted, terminating training subprocess ...")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise


def run_analysis(run_dir: str, log_path: str) -> bool:
    """
    Run analyze_run() on the completed checkpoints in run_dir.

    Returns True on success. Logs to log_path.

    Runs in-process (not as a subprocess) because analysis is short and
    inheriting our import state means we don't re-tokenize the corpus.
    """
    print(f"   ▶ Running analysis ...")
    print(f"   ▶ Log: {log_path}")

    # Redirect stdout/stderr to log file for the analysis duration.
    import io
    import contextlib

    log_buf = io.StringIO()

    try:
        # Late imports so this script doesn't pay the cost unless needed.
        from config import load_config_pair
        from data import prepare_dataset, make_dataloaders
        from analyze import analyze_run

        with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
            metadata_path = os.path.join(run_dir, "run_metadata.json")
            model_cfg, train_cfg = load_config_pair(metadata_path)

            # Build eval loader from cached tokenized dataset.
            _, held_out_dataset = prepare_dataset(
                model_cfg=model_cfg, train_cfg=train_cfg,
            )
            _, eval_loader = make_dataloaders(
                train_dataset=held_out_dataset,
                held_out_dataset=held_out_dataset,
                train_cfg=train_cfg, seed=0, num_workers=2,
            )

            # Use CUDA if available — analysis is much faster on GPU than CPU.
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            analyze_run(
                run_dir=run_dir,
                eval_loader=eval_loader,
                device=device,
                skip_existing=True,
                max_pilots=10_000,
            )

        with open(log_path, "w") as f:
            f.write(log_buf.getvalue())
        return True

    except Exception as e:
        with open(log_path, "w") as f:
            f.write(log_buf.getvalue())
            f.write(f"\n\nEXCEPTION: {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
        print(f"   ✗ Analysis failed: {type(e).__name__}: {e}")
        return False


def run_plotting(run_dir: str, log_path: str) -> bool:
    """
    Run report.py to produce plots from the analysis output.
    """
    cmd = [sys.executable, "report.py", "--run_dir", run_dir]
    print(f"   ▶ Plotting command: {' '.join(cmd)}")

    try:
        with open(log_path, "w") as log_file:
            result = subprocess.run(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True,
            )
        return result.returncode == 0
    except KeyboardInterrupt:
        raise
    except Exception as e:
        with open(log_path, "a") as f:
            f.write(f"\n\nEXCEPTION: {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
        return False


# ----------------------------------------------------------------------
# Top-level orchestration.
# ----------------------------------------------------------------------
def orchestrate(
    seeds: List[int],
    output_root: str,
    do_analyze: bool = True,
    do_plot: bool = True,
    train_extra_args: Optional[List[str]] = None,
    force_retrain: bool = False,
):
    """
    Run the full Phase 1 campaign across the given seeds.
    """
    train_extra_args = train_extra_args or []
    os.makedirs(output_root, exist_ok=True)
    state_path = os.path.join(output_root, "orchestration_state.json")
    summary_log_path = os.path.join(output_root, "orchestration.log")

    # Load or initialize state.
    state = load_state(state_path)
    for seed in seeds:
        if seed not in state:
            run_dir = os.path.join(output_root, f"seed_{seed}")
            state[seed] = SeedState(seed=seed, run_dir=run_dir)

    def log_summary(msg: str):
        """Append to summary log and print."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(summary_log_path, "a") as f:
            f.write(line + "\n")

    log_summary(f"================ ORCHESTRATION START ================")
    log_summary(f"Seeds: {seeds}")
    log_summary(f"Output root: {output_root}")
    log_summary(f"Train extra args: {train_extra_args}")
    log_summary(f"do_analyze={do_analyze}, do_plot={do_plot}, "
                f"force_retrain={force_retrain}")
    log_summary("")

    # Estimate timing. Per-seed budget is approximately:
    #   - Training: ~5 hours at ~86k tok/s on RTX 5090, 1.57B tokens total
    #   - Analysis: ~50 minutes at ~60s per checkpoint × 50 checkpoints
    #   - Plotting: <1 minute
    # Together: ~6 hours per seed end-to-end.
    incomplete_seeds = [
        seed for seed in seeds
        if state[seed].train_status != "done"
        or (do_analyze and state[seed].analyze_status not in ("done", "skipped"))
        or (do_plot and state[seed].plot_status not in ("done", "skipped"))
    ]
    log_summary(f"Incomplete seeds: {incomplete_seeds}")
    log_summary(f"Estimated time: {len(incomplete_seeds) * 6:.0f} hours "
                f"(assuming ~6 hours per seed end-to-end on RTX 5090: "
                f"~5h training + ~50min analysis + ~1min plotting)")
    log_summary("")

    overall_start = time.time()

    for seed in seeds:
        seed_state = state[seed]
        log_summary(f"--- SEED {seed} ---")
        os.makedirs(seed_state.run_dir, exist_ok=True)

        # --- Training phase ---
        already_trained = is_training_done(seed_state.run_dir)
        if already_trained and not force_retrain:
            log_summary(f"  Training already complete in {seed_state.run_dir}, "
                        f"skipping.")
            seed_state.train_status = "done"
        else:
            if force_retrain and already_trained:
                log_summary(f"  --force_retrain set; removing existing run_dir.")
                shutil.rmtree(seed_state.run_dir)
                os.makedirs(seed_state.run_dir, exist_ok=True)

            log_summary(f"  Starting training ...")
            seed_state.train_status = "running"
            seed_state.train_start_time = time.time()
            save_state(state, state_path)

            train_log = os.path.join(seed_state.run_dir, "train.log")
            try:
                ok = run_training(
                    seed=seed, run_dir=seed_state.run_dir,
                    train_extra_args=train_extra_args, log_path=train_log,
                )
            except KeyboardInterrupt:
                log_summary(f"  Training interrupted by user.")
                seed_state.train_status = "failed"
                seed_state.error_message = "Interrupted by user"
                save_state(state, state_path)
                raise

            seed_state.train_end_time = time.time()
            seed_state.train_elapsed_sec = (
                seed_state.train_end_time - seed_state.train_start_time
            )
            if ok:
                seed_state.train_status = "done"
                log_summary(f"  ✅ Training done "
                            f"[{seed_state.train_elapsed_sec/60:.1f} min].")
            else:
                seed_state.train_status = "failed"
                seed_state.error_message = "train.py returned non-zero"
                log_summary(f"  ✗ Training failed. Continuing to next seed.")
                save_state(state, state_path)
                continue
            save_state(state, state_path)

        # --- Analysis phase ---
        if not do_analyze:
            seed_state.analyze_status = "skipped"
            save_state(state, state_path)
            continue

        already_analyzed = is_analysis_done(seed_state.run_dir)
        if already_analyzed:
            log_summary(f"  Analysis already complete, skipping.")
            seed_state.analyze_status = "done"
        else:
            log_summary(f"  Running analysis ...")
            t0 = time.time()
            analyze_log = os.path.join(seed_state.run_dir, "analyze.log")
            ok = run_analysis(seed_state.run_dir, analyze_log)
            seed_state.analyze_elapsed_sec = time.time() - t0
            if ok:
                seed_state.analyze_status = "done"
                log_summary(f"  ✅ Analysis done "
                            f"[{seed_state.analyze_elapsed_sec/60:.1f} min].")
            else:
                seed_state.analyze_status = "failed"
                log_summary(f"  ✗ Analysis failed (see {analyze_log}). "
                            f"Skipping plotting for this seed.")
                save_state(state, state_path)
                continue
        save_state(state, state_path)

        # --- Plotting phase ---
        if not do_plot:
            seed_state.plot_status = "skipped"
            save_state(state, state_path)
            continue

        already_plotted = is_plotting_done(seed_state.run_dir)
        if already_plotted:
            log_summary(f"  Plotting already complete, skipping.")
            seed_state.plot_status = "done"
        else:
            log_summary(f"  Running plotting ...")
            t0 = time.time()
            plot_log = os.path.join(seed_state.run_dir, "report.log")
            ok = run_plotting(seed_state.run_dir, plot_log)
            seed_state.plot_elapsed_sec = time.time() - t0
            if ok:
                seed_state.plot_status = "done"
                log_summary(f"  ✅ Plotting done "
                            f"[{seed_state.plot_elapsed_sec:.1f} s].")
            else:
                seed_state.plot_status = "failed"
                log_summary(f"  ✗ Plotting failed (see {plot_log}).")
        save_state(state, state_path)

    # --- Final summary ---
    overall_elapsed = time.time() - overall_start
    log_summary("")
    log_summary(f"================ ORCHESTRATION COMPLETE ================")
    log_summary(f"Total elapsed: {overall_elapsed/3600:.2f} hours")
    log_summary("")

    train_done = sum(1 for s in state.values() if s.train_status == "done")
    analyze_done = sum(1 for s in state.values()
                       if s.analyze_status in ("done", "skipped"))
    plot_done = sum(1 for s in state.values()
                    if s.plot_status in ("done", "skipped"))
    failed = [s.seed for s in state.values()
              if s.train_status == "failed" or
              s.analyze_status == "failed" or
              s.plot_status == "failed"]

    log_summary(f"  Trained: {train_done}/{len(seeds)}")
    log_summary(f"  Analyzed: {analyze_done}/{len(seeds)}")
    log_summary(f"  Plotted: {plot_done}/{len(seeds)}")
    if failed:
        log_summary(f"  Failed seeds: {failed}")
    log_summary("")

    if failed:
        return 1
    return 0


# ----------------------------------------------------------------------
# Entry point.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="Seeds to run (e.g., --seeds 0 1 2 3).")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root directory under which per-seed run "
                             "directories will be created.")
    parser.add_argument("--no_analyze", action="store_true",
                        help="Skip the analysis step after training.")
    parser.add_argument("--no_plot", action="store_true",
                        help="Skip the plotting step after analysis.")
    parser.add_argument("--force_retrain", action="store_true",
                        help="Delete and re-run already-trained seeds. "
                             "Use with care.")
    parser.add_argument("--train_extra_args", type=str, default="",
                        help="Extra args to pass to train.py (quoted), e.g. "
                             "'--num_steps 1000 --eval_every 100'.")
    args = parser.parse_args()

    train_extra_args = args.train_extra_args.split() if args.train_extra_args else []

    rc = orchestrate(
        seeds=args.seeds,
        output_root=args.output_root,
        do_analyze=not args.no_analyze,
        do_plot=not args.no_plot,
        train_extra_args=train_extra_args,
        force_retrain=args.force_retrain,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
    