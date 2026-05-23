"""
Campaign driver for the reverse build-up project.

Orchestrates the five phases laid out in REVERSE_BUILDUP_PROPOSAL.md §6:

  Phase A: Parameterization and pass 1
    - Run parameterized D1, D3, D4a on reverse_actual and reverse_pred
      across all seeds and all checkpoints.
    - Verify forward parameterized output matches the existing forward
      output bit-identically (idempotency check) -- run by the test
      suite, not here.

  Phase B: D4b on final checkpoint, both reverse views, all seeds.
    - Compute Mardia Z depth profile.
    - Gate: if F2 monotonicity is not supported, downstream passes run
      with reduced scope.

  Phase C: D5 with shuffle null on final checkpoint, seed 0.
    - Real and shuffled-null D5 on reverse_actual and reverse_pred.
    - Compute F4 signal with null correction; report null absorption.
    - Expand to all seeds if signal survives null.

  Phase D: Unembedding-subspace decomposition on final checkpoint, seed 0.
    - Sweep d_parallel in {32, 64, 128, 256}.
    - Test hypothesis N1 via the par-vs-perp kurtosis gap profile.
    - Expand to all seeds if N1 supported.

  Phase E: Reverse lambda^contract clusters and writeup.
    - Run reverse_lambda_clusters.run_reverse_lambda_clusters on the
      D3 output from Phase A.

Skip-existing semantics throughout: re-running this driver picks up
where it left off based on the presence of output files on disk.

Usage:
    python reverse_buildup_campaign.py --run-dir ../phase1_runs_gelu \\
        --phases A,B,C,D,E

    # Run only the cheap passes:
    python reverse_buildup_campaign.py --run-dir ../phase1_runs_gelu \\
        --phases A,E

    # Phase D requires a Python environment with torch + the project's
    # model module importable. Skip it if running on a machine that
    # only has the on-disk artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from multiview import load_multi_view_result
from multiview_campaign import (
    seeds_in_run,
    checkpoints_in_seed,
    mvr_dir,
    augmented_path,
)

from reverse_buildup import (
    VIEWS,
    output_root,
    run_d1_view,
    run_d3_view,
    run_d4a_view,
    run_d4b_view,
    run_d5_view,
)
from reverse_null import run_d5_shuffle_null, f4_signal_with_null_correction
from reverse_lambda_clusters import run_reverse_lambda_clusters


# ----------------------------------------------------------------------
# Step discovery shared with the existing campaign.
# ----------------------------------------------------------------------
def discover_common_steps(run_dir: str, seeds: List[int]) -> List[int]:
    """Return the sorted list of steps that have a multi-view result
    for *every* seed. Mirrors the convention used by
    model_abc_discriminator.py."""
    if not seeds:
        return []
    step_sets = []
    for seed in seeds:
        steps = []
        for s, _ in checkpoints_in_seed(run_dir, seed):
            d = mvr_dir(run_dir, seed, s)
            if os.path.exists(os.path.join(d, "meta.json")):
                steps.append(s)
        step_sets.append(set(steps))
    common = sorted(set.intersection(*step_sets)) if step_sets else []
    return common


# ----------------------------------------------------------------------
# Phase A: D1, D3, D4a on reverse views.
# ----------------------------------------------------------------------
def phase_a(run_dir: str, seeds: List[int], common_steps: List[int],
            verbose: bool = True) -> Dict:
    """Run D1, D3, D4a on the two reverse views and (for cross-check
    against existing forward output) on the forward view too.

    The forward run is a strict idempotency check: bit-for-bit
    identity with the existing model_abc_discriminator output is
    verified by test_reverse_buildup; this driver merely re-emits the
    files under the new naming convention (d{1,3,4a}_{statistic}_{view}.npz).
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE A: D1, D3, D4a parameterized over view")
        print("=" * 70)

    results = {}
    for view in VIEWS:
        if verbose:
            print(f"\n--- view={view} ---")
        t0 = time.time()
        d1 = run_d1_view(run_dir, seeds, common_steps, view=view,
                         verbose=verbose)
        d3 = run_d3_view(run_dir, seeds, common_steps, view=view,
                         compute_contraction=(view != "forward"),
                         verbose=verbose)
        d4a = run_d4a_view(run_dir, seeds, common_steps, view=view,
                           verbose=verbose)
        results[view] = {"d1": d1, "d3": d3, "d4a": d4a}
        if verbose:
            print(f"--- view={view} done in {time.time() - t0:.1f}s ---")

    if verbose:
        print("\n" + "-" * 70)
        print("Phase A summary:")
        # Headline D1 gate: cross-cell CV of trace at interior layers.
        # The forward verdict expected ~0.6-0.8. Reverse is unknown.
        for view in VIEWS:
            cv = results[view]["d1"]["cv_trace"]
            if cv.size > 0:
                # Median across (seeds, steps) of mid-layer CV.
                L = cv.shape[-1]
                mid = cv[:, :, L // 4:3 * L // 4]
                med = float(np.nanmedian(mid))
                print(f"  D1 CV(trace) median over interior layers, "
                      f"view={view:14s}: {med:.3f}")
        print("-" * 70 + "\n")
    return results


# ----------------------------------------------------------------------
# Phase B: D4b on final checkpoint, both reverse views, all seeds.
# ----------------------------------------------------------------------
def phase_b(run_dir: str, seeds: List[int], common_steps: List[int],
            max_pc_dim: int = 32,
            verbose: bool = True) -> Dict:
    """Run D4b (multivariate Mardia Z) on the final checkpoint for all
    seeds on the two reverse views. Returns a verdict on whether F2
    monotonicity is supported.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE B: D4b on reverse views, final checkpoint, all seeds")
        print("=" * 70)
    if not common_steps:
        if verbose:
            print("[B] No common steps; nothing to do.")
        return {}
    final_step = common_steps[-1]
    steps_to_analyze = [(s, final_step) for s in seeds]

    results = {}
    for view in ("reverse_actual", "reverse_pred"):
        if verbose:
            print(f"\n--- view={view} ---")
        res_list = run_d4b_view(run_dir, steps_to_analyze, view=view,
                                max_pc_dim=max_pc_dim, verbose=verbose)
        results[view] = res_list

    # Gating decision: is F2 (monotonic Mardia Z decrease toward final
    # layer) supported on reverse_actual?
    f2_verdict = _evaluate_f2(results.get("reverse_actual", []),
                              verbose=verbose)
    results["f2_verdict"] = f2_verdict

    # Persist verdict.
    verdict_path = os.path.join(output_root(run_dir),
                                "reverse_buildup_phase_b_verdict.json")
    with open(verdict_path, "w") as fh:
        json.dump({"f2_verdict": f2_verdict}, fh, indent=2)
    if verbose:
        print(f"\n[B] Verdict written to {verdict_path}")
    return results


def _evaluate_f2(d4b_results: List[Dict], verbose: bool = True) -> Dict:
    """Test hypothesis F2: does Mardia Z decrease monotonically toward
    the final layer for reverse_actual conditionals?

    Criterion: at the cross-seed mean level, the slope of Mardia Z
    averaged across cells over the last 4 layers is negative AND the
    final-layer Z is less than 70% of the interior maximum Z.
    """
    if not d4b_results:
        return {"supported": False, "reason": "no_data"}
    # Stack per-seed Mardia Z across cells.
    # Each result has shape (n_cells, L).
    Z_stack = np.stack([r["mardia_z"] for r in d4b_results], axis=0)
    # Average across (seeds, cells) -> (L,).
    Z_profile = np.nanmean(Z_stack, axis=(0, 1))
    L = Z_profile.size

    if L < 4:
        return {"supported": False, "reason": "too_few_layers"}

    # Slope over last 4 layers.
    last = Z_profile[-4:]
    layer_idx = np.arange(L - 4, L, dtype=np.float64)
    if np.isnan(last).any():
        return {"supported": False, "reason": "nan_in_late_layers"}
    slope = float(np.polyfit(layer_idx, last, 1)[0])
    interior_max = float(np.nanmax(Z_profile[1:L - 1]))
    final_z = float(Z_profile[-1])
    ratio = final_z / interior_max if interior_max > 0 else float("nan")

    supported = (slope < 0.0) and (ratio < 0.7)
    verdict = {
        "supported": bool(supported),
        "slope_last4": slope,
        "interior_max_z": interior_max,
        "final_z": final_z,
        "ratio_final_to_interior_max": ratio,
        "criterion": "slope<0 AND ratio<0.7",
        "Z_profile": Z_profile.tolist(),
    }
    if verbose:
        print(f"\n[B] F2 evaluation: slope(last4)={slope:+.3f}, "
              f"interior_max={interior_max:.2f}, final={final_z:.2f}, "
              f"ratio={ratio:.3f}")
        print(f"[B] F2 verdict: {'SUPPORTED' if supported else 'NOT supported'}")
    return verdict


# ----------------------------------------------------------------------
# Phase C: D5 with shuffle null.
# ----------------------------------------------------------------------
def phase_c(run_dir: str, seeds: List[int], common_steps: List[int],
            expand_to_all_seeds: bool = False,
            verbose: bool = True) -> Dict:
    """Run D5 with shuffle null on reverse views at the final checkpoint.

    Always runs seed 0 first; expands to all seeds only if
    expand_to_all_seeds is True. The caller (typically the campaign
    driver) sets that flag based on Phase B's gating result.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE C: D5 with shuffle null on reverse views")
        print("=" * 70)
    if not common_steps:
        if verbose:
            print("[C] No common steps; nothing to do.")
        return {}
    final_step = common_steps[-1]
    seed_subset = seeds if expand_to_all_seeds else seeds[:1]
    steps_to_analyze = [(s, final_step) for s in seed_subset]

    results = {}
    for view in ("reverse_actual", "reverse_pred"):
        if verbose:
            print(f"\n--- view={view} (real labels) ---")
        real = run_d5_view(run_dir, steps_to_analyze, view=view,
                           verbose=verbose)
        if verbose:
            print(f"\n--- view={view} (shuffled null) ---")
        shuf = run_d5_shuffle_null(run_dir, steps_to_analyze, view=view,
                                   verbose=verbose)
        results[view] = {"real": real, "shuffled": shuf}

    # Compute F4 signal with null correction. We do it per-seed and
    # report the cross-seed sign agreement.
    f4 = _evaluate_f4(results, verbose=verbose)
    results["f4_verdict"] = f4
    verdict_path = os.path.join(output_root(run_dir),
                                "reverse_buildup_phase_c_verdict.json")
    with open(verdict_path, "w") as fh:
        json.dump({"f4_verdict": f4}, fh, indent=2)
    if verbose:
        print(f"\n[C] Verdict written to {verdict_path}")
    return results


def _evaluate_f4(d5_results: Dict, verbose: bool = True) -> Dict:
    """Compute the F4 null-corrected signal and the R2 gate verdict."""
    rev_act = d5_results.get("reverse_actual", {})
    rev_prd = d5_results.get("reverse_pred", {})
    if not rev_act or not rev_prd:
        return {"supported": False, "reason": "missing_view"}
    real_act = rev_act.get("real", [])
    real_prd = rev_prd.get("real", [])
    shuf_act = rev_act.get("shuffled", [])
    shuf_prd = rev_prd.get("shuffled", [])
    if not (real_act and real_prd and shuf_act and shuf_prd):
        return {"supported": False, "reason": "empty_results"}

    # Group by (seed, step).
    def _index(lst):
        return {(int(r["seed"]), int(r["step"])): r for r in lst}
    iA, iB, iAs, iBs = (_index(real_act), _index(real_prd),
                        _index(shuf_act), _index(shuf_prd))
    keys = sorted(set(iA) & set(iB) & set(iAs) & set(iBs))
    if not keys:
        return {"supported": False, "reason": "no_paired_results"}

    deltas_corr = []
    null_absorptions = []
    for k in keys:
        sig = f4_signal_with_null_correction(iA[k], iAs[k], iB[k], iBs[k])
        deltas_corr.append(sig["delta_corr"])
        null_absorptions.append(sig["null_absorption"])
    deltas_corr = np.stack(deltas_corr, axis=0)            # (n_keys, L)
    null_absorptions = np.stack(null_absorptions, axis=0)

    # Gate per R2: if null absorbs >70% of the raw signal at most
    # late layers (last 4), F4 is unresolved.
    L = deltas_corr.shape[1]
    late = null_absorptions[:, max(0, L - 4):L]
    high_absorption = np.nanmean(late) > 0.7

    # Cross-seed sign agreement at each late layer.
    sign_agreement = np.full(L, np.nan)
    for t in range(L):
        signs = np.sign(deltas_corr[:, t])
        signs = signs[np.isfinite(signs) & (signs != 0)]
        if signs.size == 0:
            continue
        sign_agreement[t] = float(np.mean(signs == signs[0]))

    verdict = {
        "n_seeds": len(keys),
        "delta_corr_mean": np.nanmean(deltas_corr, axis=0).tolist(),
        "delta_corr_per_seed": deltas_corr.tolist(),
        "null_absorption_mean": np.nanmean(null_absorptions, axis=0).tolist(),
        "sign_agreement_per_layer": sign_agreement.tolist(),
        "late_null_absorption_mean": float(np.nanmean(late)),
        "supported": bool(
            (not bool(high_absorption))
            and (np.nanmean(sign_agreement[max(0, L - 4):L]) > 0.75)
        ),
        "criterion": (
            "late_null_absorption<0.7 AND late_sign_agreement>0.75"
        ),
    }
    if verbose:
        print(f"\n[C] F4 evaluation: late null absorption = "
              f"{verdict['late_null_absorption_mean']:.3f}")
        late_sa = np.nanmean(sign_agreement[max(0, L - 4):L])
        print(f"[C] F4 late sign agreement = {late_sa:.3f}")
        print(f"[C] F4 verdict: "
              f"{'SUPPORTED' if verdict['supported'] else 'NOT supported'}")
    return verdict


# ----------------------------------------------------------------------
# Phase D: unembedding-subspace decomposition.
# ----------------------------------------------------------------------
def phase_d(run_dir: str, seeds: List[int], common_steps: List[int],
            model_cfg=None,
            expand_to_all_seeds: bool = False,
            verbose: bool = True) -> Dict:
    """Run the unembedding-subspace decomposition (hypothesis N1).

    Requires the model_cfg argument because the unembedding extractor
    needs to instantiate the model. If model_cfg is None, skips with a
    helpful message.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE D: Unembedding-subspace decomposition (N1)")
        print("=" * 70)
    if model_cfg is None:
        if verbose:
            print("[D] model_cfg=None; skipping. Pass --model-config-yaml to "
                  "enable Phase D, or run it separately via "
                  "unembedding_subspace.run_unembedding_decomposition.")
        return {"skipped": True, "reason": "no_model_cfg"}
    try:
        from unembedding_subspace import (
            run_unembedding_decomposition,
            find_checkpoint,
            DEFAULT_D_PARALLEL_SWEEP,
        )
    except ImportError as e:
        if verbose:
            print(f"[D] Could not import unembedding_subspace: {e}")
        return {"skipped": True, "reason": f"import_failed:{e}"}

    if not common_steps:
        return {}
    final_step = common_steps[-1]
    seed_subset = seeds if expand_to_all_seeds else seeds[:1]

    results = {}
    for seed in seed_subset:
        ckpt = find_checkpoint(run_dir, seed, final_step)
        if ckpt is None:
            if verbose:
                print(f"[D] seed {seed} step {final_step}: no checkpoint found")
            continue
        res = run_unembedding_decomposition(
            run_dir, seed, final_step, ckpt, model_cfg,
            view="reverse_actual",
            d_parallel_sweep=DEFAULT_D_PARALLEL_SWEEP,
            verbose=verbose,
        )
        results[seed] = res

    # Evaluate N1: at the final layer, is the par-vs-perp gap negative
    # (parallel kurtosis < perpendicular kurtosis)?
    n1_verdict = _evaluate_n1(results, verbose=verbose)
    verdict_path = os.path.join(output_root(run_dir),
                                "reverse_buildup_phase_d_verdict.json")
    with open(verdict_path, "w") as fh:
        json.dump({"n1_verdict": n1_verdict}, fh, indent=2)
    if verbose:
        print(f"\n[D] Verdict written to {verdict_path}")
    return {"per_seed": results, "n1_verdict": n1_verdict}


def _evaluate_n1(per_seed_results: Dict, verbose: bool = True) -> Dict:
    """Hypothesis N1 verdict: at the final layer, parallel kurtosis is
    substantially smaller than perpendicular kurtosis, in a way that
    is robust across the d_parallel sweep.

    Criterion: gap_par_minus_perp at the final layer is negative for
    every d_parallel value at the median seed.
    """
    if not per_seed_results:
        return {"supported": False, "reason": "no_data"}
    all_gaps = []
    for seed, res_list in per_seed_results.items():
        for r in res_list:
            gap = np.asarray(r["gap_par_minus_perp"], dtype=np.float64)
            all_gaps.append((int(r["d_parallel"]), gap))
    if not all_gaps:
        return {"supported": False, "reason": "no_d_par_results"}
    # Per d_par, final-layer gap.
    by_d = {}
    for d_par, gap in all_gaps:
        by_d.setdefault(d_par, []).append(gap[-1])
    summary = {int(d): float(np.nanmedian(v)) for d, v in by_d.items()}
    supported = all(v < -0.5 for v in summary.values())
    verdict = {
        "final_layer_gap_per_d_parallel": summary,
        "supported": bool(supported),
        "criterion": "median final-layer gap < -0.5 for every d_parallel",
    }
    if verbose:
        print(f"\n[D] N1 evaluation: final-layer gap_par-perp per d_par:")
        for d, g in summary.items():
            print(f"     d_par={d:4d}: gap = {g:+.3f}")
        print(f"[D] N1 verdict: "
              f"{'SUPPORTED' if supported else 'NOT supported'}")
    return verdict


# ----------------------------------------------------------------------
# Phase E: reverse lambda^contract clusters.
# ----------------------------------------------------------------------
def phase_e(run_dir: str, verbose: bool = True) -> Dict:
    """Run the reverse lambda^contract cluster analysis (hypothesis N2)."""
    if verbose:
        print("\n" + "=" * 70)
        print("PHASE E: Reverse lambda^contract cluster analysis (N2)")
        print("=" * 70)
    results = {}
    for view in ("reverse_actual", "reverse_pred"):
        try:
            r = run_reverse_lambda_clusters(run_dir, view=view)
            results[view] = r
        except FileNotFoundError as e:
            if verbose:
                print(f"[E/{view}] {e}")
            results[view] = {"error": str(e)}
    return results


# ----------------------------------------------------------------------
# Main entry.
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Project run directory (contains seed_*/checkpoints/).")
    p.add_argument("--phases", default="A,B,C,E",
                   help="Comma-separated phase list. Phase D requires "
                        "--model-config-yaml.")
    p.add_argument("--model-config-yaml", default=None,
                   help="Optional path to a YAML or JSON ModelConfig. "
                        "Default behavior: auto-discover from "
                        "<run-dir>/seed_*/run_metadata.json (saved at "
                        "training time). Use this flag only if you need "
                        "to override the saved metadata.")
    p.add_argument("--expand-c", action="store_true",
                   help="Run Phase C on all seeds (default: seed 0 only).")
    p.add_argument("--expand-d", action="store_true",
                   help="Run Phase D on all seeds (default: seed 0 only).")
    args = p.parse_args()

    phases = set(s.strip().upper() for s in args.phases.split(","))
    seeds = seeds_in_run(args.run_dir)
    if not seeds:
        print(f"No seeds found in {args.run_dir}", file=sys.stderr)
        sys.exit(2)
    common_steps = discover_common_steps(args.run_dir, seeds)
    print(f"Seeds: {seeds}")
    print(f"Common steps: {len(common_steps)} "
          f"({common_steps[0]}..{common_steps[-1]} if any)")
    print(f"Phases requested: {sorted(phases)}")

    # Model config (Phase D only).
    # Resolution order:
    #   1. --model-config-yaml argument (manual override; YAML or JSON).
    #   2. Auto-discovery: load run_metadata.json from seed 0's directory
    #      (saved at training time by config.save_config_pair). This is
    #      the default path and requires no hand-written YAML.
    model_cfg = None
    if "D" in phases:
        if args.model_config_yaml is not None:
            model_cfg = _load_model_cfg(args.model_config_yaml)
            print(f"[main] Loaded ModelConfig from --model-config-yaml: "
                  f"{args.model_config_yaml}")
        else:
            model_cfg = _autodiscover_model_cfg(args.run_dir, seeds)

    if "A" in phases:
        phase_a(args.run_dir, seeds, common_steps)
    if "B" in phases:
        phase_b(args.run_dir, seeds, common_steps)
    if "C" in phases:
        phase_c(args.run_dir, seeds, common_steps,
                expand_to_all_seeds=args.expand_c)
    if "D" in phases:
        phase_d(args.run_dir, seeds, common_steps, model_cfg=model_cfg,
                expand_to_all_seeds=args.expand_d)
    if "E" in phases:
        phase_e(args.run_dir)


def _autodiscover_model_cfg(run_dir: str, seeds: List[int]):
    """Auto-discover ModelConfig from the saved run_metadata.json that
    train.py writes per seed. Tries each seed in order; returns the first
    one that loads successfully, or None if none do."""
    try:
        from config import load_config_pair
    except ImportError as e:
        print(f"[main] Could not import config.load_config_pair: {e}")
        return None
    for seed in seeds:
        # Standard convention: <run_dir>/seed_<S>/run_metadata.json.
        candidate = os.path.join(run_dir, f"seed_{seed}", "run_metadata.json")
        if os.path.exists(candidate):
            try:
                model_cfg, _ = load_config_pair(candidate)
                print(f"[main] Auto-loaded ModelConfig from {candidate}")
                return model_cfg
            except Exception as e:
                print(f"[main] Failed to load {candidate}: {e}")
                continue
    print(f"[main] Phase D requested but no run_metadata.json found in any "
          f"seed_*/ subdirectory of {run_dir}, and --model-config-yaml not "
          f"given; Phase D will be skipped.")
    return None


def _load_model_cfg(path: str):
    """Load a ModelConfig from a YAML or JSON file. The JSON case handles
    both the dataclass-flat form ({hidden_size: ...}) and the
    run_metadata.json wrapped form ({model: {hidden_size: ...}})."""
    from config import ModelConfig
    import dataclasses
    with open(path, "r") as f:
        text = f.read()
    if path.endswith(".json"):
        d = json.loads(text)
    else:
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            raise RuntimeError(
                "PyYAML is required to load a YAML model config; "
                "either install pyyaml or pass a .json config.")
    # Handle the run_metadata.json schema with a 'model' key.
    if isinstance(d, dict) and "model" in d and isinstance(d["model"], dict):
        d = d["model"]
    # Drop unknown keys (forward-compat with newer ModelConfig fields).
    model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    return ModelConfig(**{k: v for k, v in d.items() if k in model_fields})


if __name__ == "__main__":
    main()
    