"""
Phase 2 analyzer driver.

Two jobs:

1. Standard analysis (Tier 1a): run analyze.analyze_run() over every
   completed Phase 2 training-run directory. Same flow-recovery
   pipeline as Phase 1, just driven across the phase2_runs/ tree.
   Each run gets its own flow_analysis/ subdirectory.

2. Input-distribution decomposition (Tier 1b): for each *final*
   checkpoint (Phase 1 GELU baseline + every Phase 2 variant), run
   the analyzer three times against three different input loaders
   (real, shuffled-within-chunk, random vocabulary). Outputs go to
   flow_analysis_shuffled/ and flow_analysis_random/ alongside the
   existing flow_analysis/.

Usage
-----
    # Standard analysis over all Tier-1 Phase 2 variants:
    python3 phase2_analyze.py --analyze_variants

    # Standard analysis over the Phase 1 GELU baseline (just delegates
    # to analyze.analyze_run; included here for completeness):
    python3 phase2_analyze.py --analyze_baseline

    # Tier 1b: input-distribution decomposition on final checkpoints
    # of the baseline + all variants:
    python3 phase2_analyze.py --tier1b

    # Restrict to one axis:
    python3 phase2_analyze.py --analyze_variants --only_axis depth

    # Restrict to one variant:
    python3 phase2_analyze.py --analyze_variants --only_variant L06

Output layout
-------------
For each run directory `D = phase2_runs/<axis>/<label>/seed_<n>/`:
    D/flow_analysis/          <- 50 flow .npz files (Tier 1a, all ckpts)
    D/flow_analysis_shuffled/ <- 1 flow .npz file (final ckpt, shuffled input)
    D/flow_analysis_random/   <- 1 flow .npz file (final ckpt, random input)

The single-file Tier 1b output directories use the same filename
convention as flow_analysis (flow_step_<step>.npz) so the existing
loaders work unchanged.
"""

import argparse
import os
import glob
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch

from config import ModelConfig, TrainingConfig, load_config_pair
from data import prepare_dataset
from analyze import analyze_run, analyze_checkpoint, save_flow, default_pilot_positions
from phase2_configs import (
    ALL_TIER1_VARIANTS, ALL_VARIANTS, find_variant, VariantSpec,
)
from phase2_launch import run_dir_for, PHASE2_ROOT, PHASE1_GELU_ROOT
from phase2_input_distributions import (
    make_input_distribution_loaders, FLOW_SUBDIR_FOR_INPUT, INPUT_DIST_NAMES,
)


# ----------------------------------------------------------------------
# Discovering completed runs.
# ----------------------------------------------------------------------
def find_baseline_run_dirs(root: str = PHASE1_GELU_ROOT) -> List[str]:
    """Return paths to Phase 1 GELU seed_* directories that look trained."""
    root_p = Path(root)
    if not root_p.exists():
        return []
    out = []
    for d in sorted(root_p.glob("seed_*")):
        meta = d / "run_metadata.json"
        ckpts = list((d / "checkpoints").glob("step_*.pt")) if (d / "checkpoints").exists() else []
        if meta.exists() and ckpts:
            out.append(str(d))
    return out


def find_variant_run_dirs(
    variants: List[VariantSpec],
    root: str = PHASE2_ROOT,
) -> List[str]:
    """Return paths to Phase 2 variant run dirs that look trained."""
    out = []
    for v in variants:
        for seed in v.seeds:
            rd = Path(run_dir_for(v, seed, root=root))
            meta = rd / "run_metadata.json"
            ckpts = list((rd / "checkpoints").glob("step_*.pt")) if (rd / "checkpoints").exists() else []
            if meta.exists() and ckpts:
                out.append(str(rd))
    return out


def find_final_checkpoint(run_dir: str) -> Optional[str]:
    """Return the path to the final checkpoint in run_dir, or None."""
    files = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.pt")))
    if not files:
        return None
    # checkpoint_step suffix sorts lexicographically up to differing lengths;
    # extract the integer step to sort numerically.
    def step_of(p):
        base = os.path.basename(p)
        return int(base.replace("step_", "").replace(".pt", ""))
    return max(files, key=step_of)


# ----------------------------------------------------------------------
# Tier 1a: standard analysis over the full run tree.
# ----------------------------------------------------------------------
def analyze_variants(
    variants: Optional[List[VariantSpec]] = None,
    only_axis: Optional[str] = None,
    only_variant: Optional[str] = None,
    device: str = "cuda",
    num_proc: Optional[int] = None,
    skip_existing: bool = True,
):
    """Run analyze.analyze_run() over every completed Phase 2 variant.

    The eval loader is built fresh from the *first* variant's model
    config -- since all variants share the same training corpus and
    seq_len (per the proposal), the loader is interchangeable.
    """
    if variants is None:
        variants = ALL_TIER1_VARIANTS
    if only_axis is not None:
        variants = [v for v in variants if v.axis == only_axis]
    if only_variant is not None:
        variants = [v for v in variants if v.label == only_variant]
    if not variants:
        print("No variants matched the filter; nothing to do.")
        return

    run_dirs = find_variant_run_dirs(variants)
    if not run_dirs:
        print("No completed variant run dirs found yet.")
        return

    print(f">> Analyzing {len(run_dirs)} variant run(s):")
    for d in run_dirs:
        print(f"   - {d}")

    # Build a single eval loader from one of the run-dirs' configs.
    # All Phase 2 variants share the same training corpus and seq_len,
    # so the loader is reusable across them.
    print()
    print(">> Preparing held-out evaluation dataset ...")
    sample_meta = os.path.join(run_dirs[0], "run_metadata.json")
    model_cfg, train_cfg = load_config_pair(sample_meta)
    _, held_out = prepare_dataset(
        model_cfg=model_cfg, train_cfg=train_cfg, num_proc=num_proc,
    )
    from torch.utils.data import DataLoader

    def collate(batch):
        ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)
        return {"input_ids": ids}

    eval_loader = DataLoader(
        held_out, batch_size=train_cfg.eval_batch_size,
        shuffle=False, collate_fn=collate,
        num_workers=2, pin_memory=True, drop_last=False,
    )

    for run_dir in run_dirs:
        print()
        print(f">> Analyzing {run_dir}")
        t0 = time.time()
        analyze_run(
            run_dir=run_dir, eval_loader=eval_loader, device=device,
            output_subdir="flow_analysis", skip_existing=skip_existing,
        )
        print(f"   ↳ Done in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------------
# Tier 1b: input-distribution decomposition on final checkpoints.
# ----------------------------------------------------------------------
def run_tier1b(
    include_baseline: bool = True,
    variants: Optional[List[VariantSpec]] = None,
    only_axis: Optional[str] = None,
    only_variant: Optional[str] = None,
    device: str = "cuda",
    num_proc: Optional[int] = None,
    random_top_k: int = 4096,
    skip_existing: bool = True,
    input_distributions: List[str] = ("shuffled", "random"),
):
    """Run shuffled and random analyses against the FINAL checkpoint of
    every selected run.

    The "real" input distribution is already covered by the standard
    analysis (flow_analysis/ contains the real-language flow). Tier 1b
    adds flow_analysis_shuffled/ and flow_analysis_random/.

    The flow_analysis_<dist>/ directories contain exactly one .npz
    file (named flow_step_<final_step>.npz). They reuse the standard
    save_flow / load_flow format so all downstream analysis code
    works unchanged.

    Why only final checkpoint: §5.4 of the proposal frames Tier 1b as a
    final-state characterization, not a trajectory. The flow-recovery
    is non-trivial to compute and storage / compute would explode by 50×
    for trajectory measurements; the marginal value is unclear.
    """
    if variants is None:
        variants = ALL_TIER1_VARIANTS
    if only_axis is not None:
        variants = [v for v in variants if v.axis == only_axis]
    if only_variant is not None:
        variants = [v for v in variants if v.label == only_variant]

    targets: List[str] = []
    if include_baseline:
        targets.extend(find_baseline_run_dirs())
    targets.extend(find_variant_run_dirs(variants))
    if not targets:
        print("No run dirs found for Tier 1b.")
        return

    for dist in input_distributions:
        if dist not in INPUT_DIST_NAMES:
            raise ValueError(
                f"Unknown input distribution {dist!r}. "
                f"Expected one of {INPUT_DIST_NAMES}."
            )
    print(f">> Tier 1b: {len(targets)} run dirs × "
          f"{len(input_distributions)} input distributions "
          f"({', '.join(input_distributions)})")

    # Build the loaders once from a sample run's training config -- all
    # variants share train_cfg.train_seq_len and eval_batch_size, so the
    # held-out dataset and the derived loaders are reusable.
    print(">> Preparing the held-out dataset and input loaders ...")
    sample_meta = os.path.join(targets[0], "run_metadata.json")
    sample_model_cfg, sample_train_cfg = load_config_pair(sample_meta)
    _, held_out = prepare_dataset(
        model_cfg=sample_model_cfg, train_cfg=sample_train_cfg,
        num_proc=num_proc,
    )
    loaders = make_input_distribution_loaders(
        held_out_dataset=held_out,
        train_cfg=sample_train_cfg,
        model_cfg=sample_model_cfg,
        seed=0,
        num_workers=2,
        pin_memory=True,
        random_top_k=random_top_k,
    )

    # Process each run dir's final checkpoint.
    for run_dir in targets:
        meta_path = os.path.join(run_dir, "run_metadata.json")
        model_cfg, _ = load_config_pair(meta_path)
        final_ckpt = find_final_checkpoint(run_dir)
        if final_ckpt is None:
            print(f"   ⚠️  {run_dir}: no final checkpoint, skipping")
            continue
        final_step = int(
            os.path.basename(final_ckpt).replace("step_", "").replace(".pt", "")
        )

        for dist in input_distributions:
            subdir = FLOW_SUBDIR_FOR_INPUT[dist]
            out_dir = os.path.join(run_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"flow_step_{final_step}.npz")
            if skip_existing and os.path.exists(out_path):
                print(f"   - {run_dir} [{dist}]: cached, skipping")
                continue
            print(f"   - {run_dir} [{dist}]: analyzing final ckpt "
                  f"step {final_step}")
            t0 = time.time()
            pilot_positions = default_pilot_positions(
                seq_len=sample_train_cfg.train_seq_len,
            )
            flow = analyze_checkpoint(
                checkpoint_path=final_ckpt,
                eval_loader=loaders[dist],
                model_cfg=model_cfg,
                device=device,
                pilot_positions=pilot_positions,
                verbose=True,
            )
            # Tag the flow with the input distribution so downstream
            # consumers don't accidentally mix real / shuffled / random
            # flows together.
            flow["input_distribution"] = dist
            save_flow(flow, out_path)
            print(f"     ↳ Saved {out_path}  [{time.time() - t0:.1f}s]")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 analyzer driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    parser.add_argument(
        "--analyze_variants", action="store_true",
        help="Run analyze_run over all completed Phase 2 variants "
             "(Tier 1a standard analysis).",
    )
    parser.add_argument(
        "--analyze_baseline", action="store_true",
        help="Run analyze_run over the Phase 1 GELU baseline seeds.",
    )
    parser.add_argument(
        "--tier1b", action="store_true",
        help="Run the input-distribution decomposition on final checkpoints "
             "(shuffled + random inputs).",
    )
    parser.add_argument(
        "--only_axis", type=str, default=None,
        help="Restrict to one axis (depth|width|ffn_ratio|...).",
    )
    parser.add_argument(
        "--only_variant", type=str, default=None,
        help="Restrict to one variant label.",
    )
    parser.add_argument(
        "--tier1b_no_baseline", action="store_true",
        help="Skip Tier 1b on the Phase 1 GELU baseline runs.",
    )
    parser.add_argument(
        "--tier1b_dists", type=str, default="shuffled,random",
        help="Comma-separated list of Tier 1b input distributions to run "
             "(default: shuffled,random; the 'real' distribution is already "
             "covered by the standard analysis).",
    )
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if output files already exist.",
    )
    args = parser.parse_args()

    if not (args.analyze_variants or args.analyze_baseline or args.tier1b):
        print("Error: specify at least one of --analyze_variants / "
              "--analyze_baseline / --tier1b.", file=sys.stderr)
        sys.exit(2)

    skip_existing = not args.force

    if args.analyze_baseline:
        # Just delegate to analyze.analyze_run over the baseline dirs.
        # We re-implement only because analyze_run takes a single
        # run_dir, not a list.
        baseline_dirs = find_baseline_run_dirs()
        if not baseline_dirs:
            print(f"⚠️  No baseline seeds found in {PHASE1_GELU_ROOT}/.")
        else:
            print(f">> Analyzing {len(baseline_dirs)} baseline seed(s)")
            # Build eval loader from one baseline run's config.
            from torch.utils.data import DataLoader
            sample_meta = os.path.join(baseline_dirs[0], "run_metadata.json")
            model_cfg, train_cfg = load_config_pair(sample_meta)
            _, held_out = prepare_dataset(
                model_cfg=model_cfg, train_cfg=train_cfg, num_proc=args.num_proc,
            )

            def _coll(b):
                return {"input_ids": torch.stack([e["input_ids"] for e in b])}

            eval_loader = DataLoader(
                held_out, batch_size=train_cfg.eval_batch_size,
                shuffle=False, collate_fn=_coll,
                num_workers=2, pin_memory=True, drop_last=False,
            )
            for d in baseline_dirs:
                print(f">> Analyzing baseline: {d}")
                analyze_run(
                    run_dir=d, eval_loader=eval_loader, device=args.device,
                    output_subdir="flow_analysis",
                    skip_existing=skip_existing,
                )

    if args.analyze_variants:
        analyze_variants(
            only_axis=args.only_axis,
            only_variant=args.only_variant,
            device=args.device,
            num_proc=args.num_proc,
            skip_existing=skip_existing,
        )

    if args.tier1b:
        dists = [d.strip() for d in args.tier1b_dists.split(",") if d.strip()]
        run_tier1b(
            include_baseline=(not args.tier1b_no_baseline),
            only_axis=args.only_axis,
            only_variant=args.only_variant,
            device=args.device,
            num_proc=args.num_proc,
            skip_existing=skip_existing,
            input_distributions=dists,
        )


if __name__ == "__main__":
    main()
