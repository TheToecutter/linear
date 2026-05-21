"""
Multi-view analysis campaign driver.

Orchestrates the full multi-view study described in the project proposal:

  Stage A:  Augmented activation collection (200 inference passes, ~10h GPU).
            For each (seed, checkpoint), load the model, run inference on the
            held-out set, save the (states, input_ids, next_ids, pred_ids,
            positions) bundle to disk.

  Stage B:  Token set selection (CPU-only, seconds).
            From the final-checkpoint augmented activations of seed 0, pick
            top-k input tokens, top-k actual-successor tokens, and top-k
            predicted-successor tokens. Save the chosen token IDs. These are
            then frozen and reused for every (seed, checkpoint) downstream.

  Stage C:  Per-checkpoint multi-view analysis (CPU-bound, ~70h CPU).
            For each (seed, checkpoint), load the augmented activations,
            run run_multi_view with the frozen token sets, save the
            MultiViewResult.

  Stage D:  Cross-checkpoint and cross-seed aggregation (CPU, seconds).
            Build trajectories of crossover layers, basis-invariant
            statistics, etc., as functions of training step. Save as
            consolidated trajectory files for plotting.

Each stage is idempotent: rerunning skips work that's already on disk.

Usage:
    python multiview_campaign.py --run-dir /path/to/run --stage A
    python multiview_campaign.py --run-dir /path/to/run --stage B
    python multiview_campaign.py --run-dir /path/to/run --stage C
    python multiview_campaign.py --run-dir /path/to/run --stage D
    python multiview_campaign.py --run-dir /path/to/run --stage all

The --run-dir is the project's top-level directory containing per-seed
subdirectories. Expected layout:

    run_dir/
      seed_0/
        checkpoints/
          ckpt_step_00000100.pt
          ckpt_step_00000300.pt
          ...
          ckpt_step_00024000.pt
      seed_1/
        checkpoints/
          ...
      ...

After the campaign, the run_dir will additionally contain:

    run_dir/
      multiview/
        token_sets.json                    -- frozen token sets (Stage B)
        seed_0/
          augmented_step_NNNNNNNN.npz      -- Stage A outputs
          mvr_step_NNNNNNNN/               -- Stage C output dirs
            meta.json
            all_to_all.npz
            decomp_*.npz
            flows_*/
        ...
        trajectories/                      -- Stage D outputs
          crossover.npz
          variance_fit.npz
          ...
"""

from __future__ import annotations

import os
import re
import sys
import glob
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ModelConfig, load_config_pair
from model import LlamaStyleTransformer
from data import prepare_dataset, make_dataloaders
from analyze import default_pilot_positions

from multiview import (
    collect_activations_with_metadata,
    save_augmented_activations,
    load_augmented_activations,
    select_token_set,
    run_multi_view,
    save_multi_view_result,
    load_multi_view_result,
    TokenSet,
    DecompositionResult,
    crossover_layer,
)


# ----------------------------------------------------------------------
# Path conventions.
# ----------------------------------------------------------------------
def seeds_in_run(run_dir: str) -> List[int]:
    """Return sorted list of seed indices found in run_dir."""
    seeds = []
    for entry in os.listdir(run_dir):
        m = re.match(r"^seed_(\d+)$", entry)
        if m and os.path.isdir(os.path.join(run_dir, entry)):
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def checkpoints_in_seed(run_dir: str, seed: int) -> List[Tuple[int, str]]:
    """
    Return [(step, path), ...] for every checkpoint of the given seed,
    sorted by step.
    """
    ckpt_dir = os.path.join(run_dir, f"seed_{seed}", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    out = []
    for fn in os.listdir(ckpt_dir):
        m = re.match(r"^step_(\d+)\.pt$", fn)
        if m:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def multiview_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview")


def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(
        multiview_dir(run_dir), f"seed_{seed}",
        f"augmented_step_{step:08d}.npz",
    )


def mvr_dir(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(
        multiview_dir(run_dir), f"seed_{seed}",
        f"mvr_step_{step:08d}",
    )


def token_sets_path(run_dir: str) -> str:
    return os.path.join(multiview_dir(run_dir), "token_sets.json")


def trajectories_dir(run_dir: str) -> str:
    return os.path.join(multiview_dir(run_dir), "trajectories")


# ----------------------------------------------------------------------
# Stage A: augmented activation collection.
# ----------------------------------------------------------------------
def stage_a_collect(
    run_dir: str,
    model_cfg: ModelConfig,
    eval_loader,
    device: str = "cuda",
    max_pilots: int = 100_000,
    compute_predictions: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run augmented activation collection on every checkpoint of every seed.

    Skips checkpoints whose augmented file already exists (idempotent).
    """
    seeds = seeds_in_run(run_dir)
    if not seeds:
        raise RuntimeError(f"No seed_N directories found in {run_dir}")

    if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    pilot_positions = default_pilot_positions(seq_len=model_cfg.max_position_embeddings)
    # Trim final positions so p+1 is always defined (the multiview
    # collector also re-validates, but we trim conservatively here too).
    pilot_positions = [p for p in pilot_positions if p + 1 < model_cfg.max_position_embeddings - 1]

    for seed in seeds:
        out_dir = os.path.join(multiview_dir(run_dir), f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        seed_ckpts = checkpoints_in_seed(run_dir, seed)
        if not seed_ckpts:
            raise RuntimeError(
                f"No checkpoints found for seed {seed} in "
                f"{os.path.join(run_dir, f'seed_{seed}', 'checkpoints')}. "
                f"Expected files matching the regex in checkpoints_in_seed(). "
                f"If your filenames look different, edit that regex."
            )
        if verbose:
            print(f"[A] seed {seed}: {len(seed_ckpts)} checkpoints to process")

        for step, ckpt_path in seed_ckpts:
            out_path = augmented_path(run_dir, seed, step)
            if skip_existing and os.path.exists(out_path):
                if verbose:
                    print(f"[A] seed {seed} step {step}: skip (exists)")
                continue

            t0 = time.time()
            if verbose:
                print(f"[A] seed {seed} step {step}: collecting ...", flush=True)

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = LlamaStyleTransformer(model_cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            payload = collect_activations_with_metadata(
                model=model,
                eval_loader=eval_loader,
                pilot_positions=pilot_positions,
                device=device,
                autocast_dtype=autocast_dtype,
                max_pilots=max_pilots,
                compute_predictions=compute_predictions,
            )
            save_augmented_activations(payload, out_path)

            del model
            if device == "cuda":
                torch.cuda.empty_cache()

            if verbose:
                N = payload["states"].shape[1]
                print(f"[A] seed {seed} step {step}: saved {N:,} pilots "
                      f"in {time.time() - t0:.1f}s -> {out_path}")


# ----------------------------------------------------------------------
# Stage B: token set selection.
# ----------------------------------------------------------------------
def stage_b_select_tokens(
    run_dir: str,
    reference_seed: int = 0,
    reference_step: Optional[int] = None,  # None means "last checkpoint of reference_seed"
    top_k: int = 20,
    min_count_single: int = 50,
    verbose: bool = True,
) -> Dict:
    """
    Select the three token sets (forward, reverse_actual, reverse_pred)
    from the reference checkpoint, save to token_sets.json.

    Convention: pick the reference checkpoint to be the FINAL step of
    seed 0 unless otherwise specified. This is the most stable
    distribution of pilot-position tokens (model is fully trained, and
    seed 0 is the project's canonical reference).
    """
    seeds = seeds_in_run(run_dir)
    if reference_seed not in seeds:
        raise RuntimeError(
            f"reference_seed={reference_seed} not found in run "
            f"(seeds present: {seeds})"
        )

    ckpts = checkpoints_in_seed(run_dir, reference_seed)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found for seed {reference_seed}")
    if reference_step is None:
        reference_step = ckpts[-1][0]
    if not any(s == reference_step for s, _ in ckpts):
        raise RuntimeError(
            f"reference_step={reference_step} not found in seed "
            f"{reference_seed}'s checkpoints"
        )

    aug_path = augmented_path(run_dir, reference_seed, reference_step)
    if not os.path.exists(aug_path):
        raise RuntimeError(
            f"Augmented activations missing for the reference checkpoint: "
            f"{aug_path}. Run stage A first."
        )

    aug = load_augmented_activations(aug_path)

    forward_set = select_token_set(
        tags=aug["input_ids"],
        view="forward",
        top_k=top_k,
        min_count=min_count_single,
    )
    reverse_actual_set = select_token_set(
        tags=aug["next_ids"],
        view="reverse_actual",
        top_k=top_k,
        min_count=min_count_single,
    )
    # Predicted tokens require lm_head argmax (compute_predictions=True in stage A).
    if (aug["pred_ids"] >= 0).any():
        reverse_pred_set = select_token_set(
            tags=aug["pred_ids"],
            view="reverse_pred",
            top_k=top_k,
            min_count=min_count_single,
        )
    else:
        reverse_pred_set = TokenSet(
            view="reverse_pred",
            token_ids=np.zeros(0, dtype=np.int32),
            counts=np.zeros(0, dtype=np.int64),
            min_count=min_count_single,
            total_pilots=int(aug["input_ids"].size),
        )

    out = {
        "reference_seed": int(reference_seed),
        "reference_step": int(reference_step),
        "top_k": int(top_k),
        "min_count_single": int(min_count_single),
        "forward": forward_set.to_dict(),
        "reverse_actual": reverse_actual_set.to_dict(),
        "reverse_pred": reverse_pred_set.to_dict(),
    }
    os.makedirs(multiview_dir(run_dir), exist_ok=True)
    with open(token_sets_path(run_dir), "w") as f:
        json.dump(out, f, indent=2)

    if verbose:
        print(f"[B] Selected token sets from seed {reference_seed} step {reference_step}:")
        print(f"    forward:        {forward_set.token_ids.size} tokens, "
              f"coverage {forward_set.coverage_fraction():.3f}")
        print(f"    reverse_actual: {reverse_actual_set.token_ids.size} tokens, "
              f"coverage {reverse_actual_set.coverage_fraction():.3f}")
        print(f"    reverse_pred:   {reverse_pred_set.token_ids.size} tokens, "
              f"coverage {reverse_pred_set.coverage_fraction():.3f}")

    return out


def load_token_sets(run_dir: str) -> Tuple[TokenSet, TokenSet, TokenSet]:
    """Load the three frozen token sets from disk."""
    with open(token_sets_path(run_dir), "r") as f:
        d = json.load(f)

    def _from_dict(dd):
        return TokenSet(
            view=dd["view"],
            token_ids=np.array(dd["token_ids"], dtype=np.int32),
            counts=np.array(dd["counts"], dtype=np.int64),
            min_count=int(dd["min_count"]),
            total_pilots=int(dd["total_pilots"]),
        )

    return (
        _from_dict(d["forward"]),
        _from_dict(d["reverse_actual"]),
        _from_dict(d["reverse_pred"]),
    )


# ----------------------------------------------------------------------
# Stage C: per-checkpoint multi-view analysis.
# ----------------------------------------------------------------------
def stage_c_analyze(
    run_dir: str,
    skip_existing: bool = True,
    verbose: bool = True,
) -> None:
    """
    For every (seed, checkpoint) with a stage-A augmented file on disk,
    compute the multi-view result and save it.

    Uses the frozen token sets from token_sets.json.
    """
    forward_set, reverse_actual_set, reverse_pred_set = load_token_sets(run_dir)

    seeds = seeds_in_run(run_dir)
    for seed in seeds:
        for step, _ in checkpoints_in_seed(run_dir, seed):
            aug_path = augmented_path(run_dir, seed, step)
            if not os.path.exists(aug_path):
                if verbose:
                    print(f"[C] seed {seed} step {step}: skip (no augmented file)")
                continue

            out_dir = mvr_dir(run_dir, seed, step)
            if skip_existing and os.path.exists(os.path.join(out_dir, "meta.json")):
                if verbose:
                    print(f"[C] seed {seed} step {step}: skip (exists)")
                continue

            t0 = time.time()
            if verbose:
                print(f"[C] seed {seed} step {step}: analyzing ...", flush=True)

            aug = load_augmented_activations(aug_path)
            result = run_multi_view(
                augmented=aug,
                forward_set=forward_set,
                reverse_actual_set=reverse_actual_set,
                reverse_pred_set=reverse_pred_set,
                step=step,
                seed=seed,
            )
            save_multi_view_result(result, out_dir)

            if verbose:
                print(f"[C] seed {seed} step {step}: done in {time.time() - t0:.1f}s")


# ----------------------------------------------------------------------
# Stage D: cross-checkpoint aggregation.
# ----------------------------------------------------------------------
def stage_d_trajectories(
    run_dir: str,
    verbose: bool = True,
) -> None:
    """
    Sweep across all (seed, checkpoint) MultiViewResults; assemble
    cross-checkpoint trajectory arrays for plotting.

    Outputs go to multiview/trajectories/:
      crossover.npz: (n_seeds, n_steps) arrays for forward & reverse_actual
                     crossover layers + status flags.
      variance_fit.npz: (n_seeds, n_steps) arrays for log_alpha, lambda
                        per view (all-to-all, per-token averaged forward,
                        per-token averaged reverse_actual).
      decomposition.npz: (n_seeds, n_steps, L_total) v_within, v_between,
                         v_subset_total, v_all_to_all per view.
      effective_rank.npz: (n_seeds, n_steps, L_total) effective rank per view.
      post_final_norm.npz: (n_seeds, n_steps) post-final-norm anomaly per view.
    """
    seeds = seeds_in_run(run_dir)
    # Determine the step grid from seed 0 (assume identical across seeds).
    step_list_by_seed = {s: [step for step, _ in checkpoints_in_seed(run_dir, s)]
                        for s in seeds}
    common_steps = sorted(set.intersection(*[set(v) for v in step_list_by_seed.values()]))
    if not common_steps:
        raise RuntimeError("No common steps across seeds.")

    # Probe one result to determine L_total. We need singular_values shape,
    # so we skip only the R matrix (the largest per-flow array, unused here).
    probe = load_multi_view_result(mvr_dir(run_dir, seeds[0], common_steps[0]),
                                   skip_arrays={"R"})
    L = probe.all_to_all["singular_values"].shape[0]
    del probe  # release memory

    n_s = len(seeds)
    n_t = len(common_steps)

    # Allocate.
    crossover_fwd = np.full((n_s, n_t), np.nan, dtype=np.float64)
    crossover_rev_actual = np.full((n_s, n_t), np.nan, dtype=np.float64)
    crossover_rev_pred = np.full((n_s, n_t), np.nan, dtype=np.float64)

    log_alpha = {  # view -> (n_s, n_t)
        "all_to_all": np.full((n_s, n_t), np.nan),
        "forward": np.full((n_s, n_t), np.nan),
        "reverse_actual": np.full((n_s, n_t), np.nan),
    }
    lam = {k: np.full((n_s, n_t), np.nan) for k in log_alpha}

    v_within_traj = {
        "forward": np.full((n_s, n_t, L), np.nan),
        "reverse_actual": np.full((n_s, n_t, L), np.nan),
        "reverse_pred": np.full((n_s, n_t, L), np.nan),
    }
    v_between_traj = {k: np.full((n_s, n_t, L), np.nan) for k in v_within_traj}
    v_subset_traj = {k: np.full((n_s, n_t, L), np.nan) for k in v_within_traj}
    v_all_traj = np.full((n_s, n_t, L), np.nan)

    eff_rank_traj = {
        "all_to_all": np.full((n_s, n_t, L), np.nan),
        "forward": np.full((n_s, n_t, L), np.nan),
        "reverse_actual": np.full((n_s, n_t, L), np.nan),
    }

    total = n_s * n_t
    done = 0
    if verbose:
        print(f"[D] Aggregating across {n_s} seeds x {n_t} steps = {total} checkpoints")
    for si, seed in enumerate(seeds):
        for ti, step in enumerate(common_steps):
            d = mvr_dir(run_dir, seed, step)
            if not os.path.exists(os.path.join(d, "meta.json")):
                done += 1
                continue
            # Skip R matrices (unused, ~44 MB each, 60+ per checkpoint).
            r = load_multi_view_result(d, skip_arrays={"R"})

            # Crossover layers.
            c_fwd, _ = crossover_layer(r.forward_decomp.v_within,
                                       r.forward_decomp.v_between,
                                       direction="forward")
            crossover_fwd[si, ti] = c_fwd

            c_ra, _ = crossover_layer(r.reverse_actual_decomp.v_within,
                                      r.reverse_actual_decomp.v_between,
                                      direction="reverse")
            crossover_rev_actual[si, ti] = c_ra

            if r.reverse_pred_set.token_ids.size > 0:
                c_rp, _ = crossover_layer(r.reverse_pred_decomp.v_within,
                                          r.reverse_pred_decomp.v_between,
                                          direction="reverse")
                crossover_rev_pred[si, ti] = c_rp

            # Variance-fit scalars (all-to-all is from the flow dict).
            la = float(r.all_to_all.get("log_alpha", np.nan))
            ll = float(r.all_to_all.get("lambda", np.nan))
            log_alpha["all_to_all"][si, ti] = la
            lam["all_to_all"][si, ti] = ll

            # Per-token averaged variance-fit for forward & reverse views:
            # we take the frequency-weighted mean of per-token log_alpha
            # and lambda.
            for vname, flows, ts in [
                ("forward", r.forward_flows, r.forward_set),
                ("reverse_actual", r.reverse_actual_flows, r.reverse_actual_set),
            ]:
                if ts.token_ids.size == 0:
                    continue
                w = ts.counts.astype(np.float64)
                w = w / w.sum() if w.sum() > 0 else w
                las, lls = [], []
                for tid, ww in zip(ts.token_ids, w):
                    f = flows.get(int(tid))
                    if f is None:
                        las.append(np.nan)
                        lls.append(np.nan)
                    else:
                        las.append(float(f.get("log_alpha", np.nan)))
                        lls.append(float(f.get("lambda", np.nan)))
                la_w = np.nansum(np.array(las) * w)
                ll_w = np.nansum(np.array(lls) * w)
                log_alpha[vname][si, ti] = la_w
                lam[vname][si, ti] = ll_w

            # Decomposition arrays.
            v_within_traj["forward"][si, ti] = r.forward_decomp.v_within
            v_between_traj["forward"][si, ti] = r.forward_decomp.v_between
            v_subset_traj["forward"][si, ti] = r.forward_decomp.v_subset_total
            v_all_traj[si, ti] = r.forward_decomp.v_all_to_all

            v_within_traj["reverse_actual"][si, ti] = r.reverse_actual_decomp.v_within
            v_between_traj["reverse_actual"][si, ti] = r.reverse_actual_decomp.v_between
            v_subset_traj["reverse_actual"][si, ti] = r.reverse_actual_decomp.v_subset_total

            if r.reverse_pred_set.token_ids.size > 0:
                v_within_traj["reverse_pred"][si, ti] = r.reverse_pred_decomp.v_within
                v_between_traj["reverse_pred"][si, ti] = r.reverse_pred_decomp.v_between
                v_subset_traj["reverse_pred"][si, ti] = r.reverse_pred_decomp.v_subset_total

            # Effective rank.
            eff_rank_traj["all_to_all"][si, ti] = r.all_to_all["effective_rank"]
            # Per-token averaged effective rank.
            for vname, flows, ts in [
                ("forward", r.forward_flows, r.forward_set),
                ("reverse_actual", r.reverse_actual_flows, r.reverse_actual_set),
            ]:
                if ts.token_ids.size == 0:
                    continue
                w = ts.counts.astype(np.float64)
                w = w / w.sum() if w.sum() > 0 else w
                stack = np.full((len(ts.token_ids), L), np.nan)
                for k, tid in enumerate(ts.token_ids):
                    f = flows.get(int(tid))
                    if f is not None and "effective_rank" in f:
                        stack[k] = f["effective_rank"]
                eff_rank_traj[vname][si, ti] = np.nansum(stack * w[:, None], axis=0)

            # Drop the loaded result so memory doesn't accumulate across
            # the inner loop. The per-flow dicts hold ~100 MB of arrays
            # each at full precision; without explicit deallocation Python
            # may keep multiple alive between iterations.
            del r
            done += 1
            if verbose and (done % 10 == 0 or done == total):
                print(f"[D] aggregated {done}/{total} checkpoints", flush=True)

    out_dir = trajectories_dir(run_dir)
    os.makedirs(out_dir, exist_ok=True)
    seeds_arr = np.array(seeds, dtype=np.int32)
    steps_arr = np.array(common_steps, dtype=np.int64)

    np.savez_compressed(
        os.path.join(out_dir, "crossover.npz"),
        seeds=seeds_arr, steps=steps_arr,
        crossover_forward=crossover_fwd,
        crossover_reverse_actual=crossover_rev_actual,
        crossover_reverse_pred=crossover_rev_pred,
    )
    np.savez_compressed(
        os.path.join(out_dir, "variance_fit.npz"),
        seeds=seeds_arr, steps=steps_arr,
        log_alpha_all_to_all=log_alpha["all_to_all"],
        log_alpha_forward=log_alpha["forward"],
        log_alpha_reverse_actual=log_alpha["reverse_actual"],
        lambda_all_to_all=lam["all_to_all"],
        lambda_forward=lam["forward"],
        lambda_reverse_actual=lam["reverse_actual"],
    )
    np.savez_compressed(
        os.path.join(out_dir, "decomposition.npz"),
        seeds=seeds_arr, steps=steps_arr,
        v_within_forward=v_within_traj["forward"],
        v_between_forward=v_between_traj["forward"],
        v_subset_forward=v_subset_traj["forward"],
        v_within_reverse_actual=v_within_traj["reverse_actual"],
        v_between_reverse_actual=v_between_traj["reverse_actual"],
        v_subset_reverse_actual=v_subset_traj["reverse_actual"],
        v_within_reverse_pred=v_within_traj["reverse_pred"],
        v_between_reverse_pred=v_between_traj["reverse_pred"],
        v_subset_reverse_pred=v_subset_traj["reverse_pred"],
        v_all_to_all=v_all_traj,
    )
    np.savez_compressed(
        os.path.join(out_dir, "effective_rank.npz"),
        seeds=seeds_arr, steps=steps_arr,
        eff_rank_all_to_all=eff_rank_traj["all_to_all"],
        eff_rank_forward=eff_rank_traj["forward"],
        eff_rank_reverse_actual=eff_rank_traj["reverse_actual"],
    )

    if verbose:
        print(f"[D] Trajectories saved to {out_dir}")


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Top-level run directory")
    parser.add_argument("--stage", required=True, choices=["A", "B", "C", "D", "all"])
    parser.add_argument("--config", default=None,
                        help="Path to ModelConfig YAML/JSON (required for stage A)")
    parser.add_argument("--top-k", type=int, default=20, help="Stage B: tokens per view")
    parser.add_argument("--min-count", type=int, default=50,
                        help="Stage B: minimum pilot count per token")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-run even when output files exist")
    args = parser.parse_args()

    if args.stage in ("A", "all"):
        if args.config is None:
            raise SystemExit("Stage A requires --config")
        model_cfg, train_cfg = load_config_pair(args.config)
        # Build the held-out dataloader: prepare_dataset returns
        # (train_dataset, held_out_dataset); we discard the train one,
        # then make_dataloaders gives us (train_loader, eval_loader).
        _, held_out_dataset = prepare_dataset(model_cfg, train_cfg)
        # Use a dummy train_dataset for make_dataloaders since we only
        # need the eval loader; we can pass held_out_dataset twice and
        # discard the train loader.
        _, eval_loader = make_dataloaders(held_out_dataset, held_out_dataset,
                                          train_cfg)
        stage_a_collect(
            run_dir=args.run_dir,
            model_cfg=model_cfg,
            eval_loader=eval_loader,
            skip_existing=not args.no_skip,
        )

    if args.stage in ("B", "all"):
        stage_b_select_tokens(
            run_dir=args.run_dir,
            top_k=args.top_k,
            min_count_single=args.min_count,
        )

    if args.stage in ("C", "all"):
        stage_c_analyze(
            run_dir=args.run_dir,
            skip_existing=not args.no_skip,
        )

    if args.stage in ("D", "all"):
        stage_d_trajectories(run_dir=args.run_dir)


if __name__ == "__main__":
    main()
    