"""
Phase 2 launcher: train one Phase 2 variant × seed.

Usage
-----
    # List all known variants:
    python3 phase2_launch.py --list

    # Dry-run a single variant (prints config + run dir, does not train):
    python3 phase2_launch.py --variant L06 --seed 0 --dry_run

    # Launch one training run:
    python3 phase2_launch.py --variant L06 --seed 0

    # Launch all tier-1 variants × all their seeds sequentially:
    python3 phase2_launch.py --launch_tier 1a

    # Override total_steps (smoke test):
    python3 phase2_launch.py --variant L06 --seed 0 --total_steps 200 --dry_run

Run-directory layout
--------------------
    phase2_runs/
      <axis>/
        <variant_label>/
          seed_<n>/
            checkpoints/
            flow_analysis/        (populated later by phase2_analyze.py)
            run_metadata.json
            training_log.csv

Resume / skip behavior
----------------------
If `run_dir/run_metadata.json` already exists and `--force` is not set,
the launcher refuses to overwrite. To resume an interrupted run, delete
the run_dir manually (or pass --force, which is destructive).

Phase 1 GELU prerequisite
-------------------------
The launcher emits a warning when it can't find the Phase 1 GELU runs
at the expected location (phase1_runs_gelu/seed_*) -- those provide the
L=12 / H=896 / GELU baseline data points and the within-variant
dispersion thresholds. The training itself doesn't require them, but
the attribution analysis at the end of Phase 2 does.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from config import ModelConfig, TrainingConfig
from phase2_configs import (
    ALL_VARIANTS,
    ALL_TIER1_VARIANTS,
    TIER2_VARIANTS,
    BASELINE_LABEL,
    VariantSpec,
    find_variant,
    summarize,
    make_baseline_gelu_config,
)


# Output root for Phase 2 training runs. Overridable via the
# PHASE2_RUNS_ROOT environment variable so the queue can be launched
# from one directory but write outputs to another (e.g., launch from
# linear/phase2/ but write to linear/phase2_runs/).
PHASE2_ROOT = os.environ.get("PHASE2_RUNS_ROOT", "phase2_runs")
# Same for the Phase 1 GELU baseline location, read by the baseline-
# presence check and by Tier 1b's analyze pass.
PHASE1_GELU_ROOT = os.environ.get("PHASE1_GELU_ROOT", "phase1_runs_gelu")


# ----------------------------------------------------------------------
# Run-directory layout.
# ----------------------------------------------------------------------
def run_dir_for(variant: VariantSpec, seed: int, root: str = PHASE2_ROOT) -> str:
    """Compose the canonical run directory for a (variant, seed)."""
    return os.path.join(root, variant.axis, variant.label, f"seed_{seed}")


# ----------------------------------------------------------------------
# Sanity check: does the Phase 1 GELU baseline exist?
# ----------------------------------------------------------------------
def check_baseline_present(verbose: bool = True) -> bool:
    """Return True if at least one Phase 1 GELU seed appears trained.

    A "trained seed" means a run_metadata.json exists AND at least one
    checkpoint file is present. We don't require all 4 seeds to be
    finished -- Phase 1 GELU is allowed to be in progress when Phase 2
    variants are being launched in parallel (e.g., on multiple GPUs or
    sequentially on the same workstation).
    """
    root = Path(PHASE1_GELU_ROOT)
    if not root.exists():
        if verbose:
            print(
                f"⚠️  Phase 1 GELU baseline directory not found at "
                f"{PHASE1_GELU_ROOT}/. Phase 2 training can still proceed, "
                f"but the attribution analysis at the end needs at least "
                f"one GELU seed."
            )
        return False
    seed_dirs = sorted(root.glob("seed_*"))
    trained = [
        d for d in seed_dirs
        if (d / "run_metadata.json").exists()
        and any((d / "checkpoints").glob("step_*.pt") if (d / "checkpoints").exists() else [])
    ]
    if verbose:
        if trained:
            print(
                f"✓ Phase 1 GELU baseline: found {len(trained)} trained seed(s) "
                f"in {PHASE1_GELU_ROOT}/."
            )
        else:
            print(
                f"⚠️  Phase 1 GELU directory {PHASE1_GELU_ROOT}/ exists but "
                f"no trained seeds found. Phase 2 training can proceed; "
                f"the attribution analysis needs the baseline to complete."
            )
    return bool(trained)


# ----------------------------------------------------------------------
# Banner printing.
# ----------------------------------------------------------------------
def print_variant_banner(variant: VariantSpec, seed: int,
                          run_dir: str, total_steps: int):
    cfg = variant.config_factory()
    params_M = cfg.estimate_param_count() / 1e6
    print("=" * 72)
    print(f"Phase 2 variant: {variant.label}  (axis={variant.axis}, tier={variant.tier})")
    print("=" * 72)
    print(f"  Description : {variant.description}")
    print(f"  Architecture: L={cfg.num_hidden_layers}, "
          f"H={cfg.hidden_size}, "
          f"I_stored={cfg.intermediate_size}, "
          f"heads={cfg.num_attention_heads}, "
          f"ffn={cfg.ffn_type}")
    print(f"  Params      : {params_M:.1f}M")
    print(f"  Seed        : {seed}")
    print(f"  Total steps : {total_steps}")
    print(f"  Run dir     : {run_dir}")
    print("=" * 72)


# ----------------------------------------------------------------------
# Single-run launcher.
# ----------------------------------------------------------------------
def launch_one(
    variant: VariantSpec,
    seed: int,
    total_steps: Optional[int] = None,
    dry_run: bool = False,
    force: bool = False,
    device: Optional[str] = None,
    num_proc: Optional[int] = None,
    root: str = PHASE2_ROOT,
    apply_memfit: bool = True,
):
    """Launch one Phase 2 training run.

    If dry_run is True, prints the config and resolved run dir without
    actually training. If force is False (default) and the run dir
    already has a run_metadata.json, raises FileExistsError.

    If apply_memfit is True (default) and a MEMFIT_NOTES.txt exists for
    this variant, the micro_batch_size / grad_accum_steps from the
    notes file override the TrainingConfig defaults. This is how a
    fit-check on H1792 carries forward into the full training run.
    """
    run_dir = run_dir_for(variant, seed, root=root)
    cfg = variant.config_factory()
    train_cfg = TrainingConfig(seed=seed)

    # Phase 2 invariant: every variant must use GELU so statistics are
    # comparable to the Phase 1 GELU baseline. The catalog enforces
    # this at definition time, but a hand-edited config or a future
    # variant could bypass that check; this assertion is the runtime
    # backstop. Violating it would silently produce a SwiGLU run
    # mislabeled under a Phase 2 directory, breaking the attribution
    # matrix.
    if cfg.ffn_type != "gelu":
        raise ValueError(
            f"Phase 2 variant {variant.label!r} has ffn_type="
            f"{cfg.ffn_type!r}, but Phase 2 is GELU-only by design "
            f"(comparable to the Phase 1 GELU baseline). Refusing to "
            f"launch. If you genuinely need a SwiGLU run, use the "
            f"Phase 1 SwiGLU directory, not the Phase 2 launcher."
        )

    # Apply memfit notes if present.
    if apply_memfit:
        # Lazy import to avoid a circular dependency at module import time
        # (phase2_memfit may not exist yet during early-stage development).
        try:
            from phase2_memfit import read_notes
            notes = read_notes(variant, root=root)
        except ImportError:
            notes = None
        if notes is not None:
            old_micro = train_cfg.micro_batch_size
            old_accum = train_cfg.grad_accum_steps
            new_micro = int(notes["micro_batch_size"])
            new_accum = int(notes["grad_accum_steps"])
            if (new_micro, new_accum) != (old_micro, old_accum):
                print(
                    f">> Applying memfit notes: "
                    f"micro_batch {old_micro} → {new_micro}, "
                    f"grad_accum {old_accum} → {new_accum} "
                    f"(effective batch unchanged at "
                    f"{new_micro * new_accum})"
                )
                train_cfg.micro_batch_size = new_micro
                train_cfg.grad_accum_steps = new_accum

    if total_steps is not None:
        train_cfg.total_steps = total_steps
        # Mirror the smoke-test cadence adjustments from train.main().
        train_cfg.warmup_steps = min(train_cfg.warmup_steps, total_steps // 4)
        train_cfg.eval_every = min(
            train_cfg.eval_every, max(50, total_steps // 4),
        )
        train_cfg.first_checkpoint_step = min(
            train_cfg.first_checkpoint_step, max(10, total_steps // 10),
        )

    print_variant_banner(variant, seed, run_dir, train_cfg.total_steps)

    if not dry_run:
        # Refuse to clobber existing runs unless --force.
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if os.path.exists(meta_path) and not force:
            raise FileExistsError(
                f"Run dir {run_dir} already has run_metadata.json. "
                f"Refusing to overwrite. Pass --force to overwrite, or "
                f"delete the directory manually to resume from scratch."
            )
        # Lazy import: this pulls in torch/datasets/transformers, which
        # we don't want to load for --dry_run or --list.
        from train import train_one_run
        os.makedirs(run_dir, exist_ok=True)
        t0 = time.time()
        train_one_run(
            model_cfg=cfg,
            train_cfg=train_cfg,
            seed=seed,
            run_dir=run_dir,
            num_proc=num_proc,
            device=device,
        )
        print(f">> Training completed in {(time.time() - t0) / 3600:.2f} h")
    else:
        print(">> Dry run: no training launched.")


# ----------------------------------------------------------------------
# Batch launchers.
# ----------------------------------------------------------------------
def launch_tier(
    tier: str,
    total_steps: Optional[int] = None,
    dry_run: bool = False,
    force: bool = False,
    device: Optional[str] = None,
    num_proc: Optional[int] = None,
    only_axis: Optional[str] = None,
):
    """Launch all variants in the named tier, sequentially.

    On a single workstation this serializes the GPU; on multi-GPU
    setups, use the per-variant launcher externally.
    """
    if tier == "1a":
        variants = ALL_TIER1_VARIANTS
    elif tier == "2":
        variants = TIER2_VARIANTS
    else:
        raise ValueError(f"Unknown tier: {tier!r}. Expected '1a' or '2'.")
    if only_axis is not None:
        variants = [v for v in variants if v.axis == only_axis]
    if not variants:
        print(f"No variants to launch for tier={tier} only_axis={only_axis}.")
        return
    total = sum(len(v.seeds) for v in variants)
    print(f">> Tier {tier}: {len(variants)} variants × seeds = {total} runs")
    for v in variants:
        if not v.seeds:
            print(f"   (skipping {v.label}: no seeds configured)")
            continue
        for seed in v.seeds:
            launch_one(
                variant=v, seed=seed,
                total_steps=total_steps,
                dry_run=dry_run, force=force,
                device=device, num_proc=num_proc,
            )


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Launch Phase 2 training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the full Phase 2 variant catalog and exit.",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Variant label to launch (see --list).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed to use for the named variant. If omitted with "
             "--variant, all configured seeds for the variant launch.",
    )
    parser.add_argument(
        "--launch_tier", type=str, default=None, choices=["1a", "2"],
        help="Launch a whole tier sequentially.",
    )
    parser.add_argument(
        "--only_axis", type=str, default=None,
        help="When launching a tier, restrict to one axis "
             "(depth|width|ffn_ratio|gating|...).",
    )
    parser.add_argument(
        "--total_steps", type=int, default=None,
        help="Override TrainingConfig.total_steps (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Resolve and print configs, but don't train.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing run dirs (DESTRUCTIVE).",
    )
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"])
    parser.add_argument("--num_proc", type=int, default=None)
    args = parser.parse_args()

    if args.list:
        print(summarize())
        return

    # Sanity check the baseline directory (warning only).
    if not args.dry_run:
        check_baseline_present()

    if args.launch_tier is not None:
        if args.variant is not None or args.seed is not None:
            print("Error: --launch_tier is mutually exclusive with "
                  "--variant / --seed.", file=sys.stderr)
            sys.exit(2)
        launch_tier(
            tier=args.launch_tier,
            total_steps=args.total_steps,
            dry_run=args.dry_run,
            force=args.force,
            device=args.device,
            num_proc=args.num_proc,
            only_axis=args.only_axis,
        )
        return

    if args.variant is None:
        print("Error: must specify one of --list / --variant / --launch_tier.",
              file=sys.stderr)
        sys.exit(2)

    variant = find_variant(args.variant)
    if args.seed is None:
        # Launch all configured seeds for the variant.
        if not variant.seeds:
            print(f"Variant {variant.label} has no configured seeds.",
                  file=sys.stderr)
            sys.exit(1)
        for seed in variant.seeds:
            launch_one(
                variant=variant, seed=seed,
                total_steps=args.total_steps,
                dry_run=args.dry_run, force=args.force,
                device=args.device, num_proc=args.num_proc,
            )
    else:
        launch_one(
            variant=variant, seed=args.seed,
            total_steps=args.total_steps,
            dry_run=args.dry_run, force=args.force,
            device=args.device, num_proc=args.num_proc,
        )


if __name__ == "__main__":
    main()
    