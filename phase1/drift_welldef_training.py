"""
Test 1 over training: drift well-definedness across checkpoints.

Background
----------
drift_welldef_test.py establishes that at the final checkpoint, position
binning explains ~12% of the layer-update variance, regardless of binning
dimensionality d (the d-sweep saturates by d=3-5). This script asks the
follow-up: how does that 12% emerge across training?

Three hypotheses worth distinguishing:

  H_monotone:   R^2_pos rises monotonically through training. Drift
                well-definedness is something the network develops as
                it learns to use position information.

  H_hump:       R^2_pos has a hump that co-locates with the Phase 1
                log_alpha hump (step ~5000) and Sigma-distance bump.
                In that case, the same training-dynamic event would
                be visible in three different geometric statistics --
                a macro-to-micro bridge.

  H_flat:       R^2_pos is approximately constant across training.
                Position-dependence of the layer update is a fixed
                structural property of the transformer architecture,
                not a learned property.

Empirically distinguishing these is what this script does.

Method
------
1. Discover all augmented_step_*.npz checkpoints for the chosen seed.
2. Sub-sample to a log-spaced set of ~12 checkpoints (configurable).
3. At each checkpoint, run analyze_layer (from drift_welldef_test) at
   every layer transition.
4. Collect the per-layer-per-checkpoint R^2_pos matrix.
5. Plot three derived curves on one figure (with shared x-axis = log
   training step):
     - R^2_pos averaged across all layer transitions
     - R^2_pos at the prediction-commitment layer (default 11->12)
     - R^2_pos at a deep interior layer (default 5->6)
   Each curve shows the cross-seed mean if multiple seeds requested.

Co-location annotations
-----------------------
The Phase 1 log_alpha hump peaks at step ~5000 (range 4483-5014 across
seeds). The post-final-norm anomaly emerges in the same window.
Reference vertical lines at step 5000 are drawn so the user can read
off any co-location at a glance.

Usage
-----
    python3 drift_welldef_training.py
        [--run-dir PATH]
        [--seed S]                  # default 0
        [--seeds S1,S2,...]         # alternative: cross-seed mean
        [--d D]                     # default 5 (saturation regime)
        [--k K]                     # default 24
        [--n-checkpoints N]         # default 12 (log-spaced subsample)
        [--interior-layer T]        # default 6 (the t->t+1 transition,
                                    # so 6 means layer 6 -> layer 7)
        [--commit-layer T]          # default 11 (default 11 -> 12, the
                                    # observed prediction-commitment
                                    # transition in the GELU model)

Output
------
    ../phase1_runs_gelu/drift_welldef/
        training_trajectory_seed_<S>_d_<D>_k_<K>.npz   (raw matrix)
        training_trajectory_seed_<S>_d_<D>_k_<K>.png   (3-curve plot)

When run with --seeds, files are suffixed with seeds_<S1>_<S2>_... and
the plot shows cross-seed mean +/- std bands.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the per-checkpoint kernel.
from drift_welldef_test import analyze_layer


# ----------------------------------------------------------------------
# Checkpoint discovery and sub-sampling.
# ----------------------------------------------------------------------
def checkpoints_in_seed(run_dir: str, seed: int) -> List[int]:
    """List augmented_step_*.npz checkpoints available for a seed.

    Convention matches covariance_similarity.checkpoints_in_seed.
    """
    seed_dir = os.path.join(run_dir, "multiview", f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return []
    steps = []
    for fname in os.listdir(seed_dir):
        if fname.startswith("augmented_step_") and fname.endswith(".npz"):
            try:
                step = int(fname.removeprefix("augmented_step_")
                           .removesuffix(".npz"))
                steps.append(step)
            except ValueError:
                continue
    return sorted(steps)


def log_spaced_subsample(steps: List[int], n_target: int) -> List[int]:
    """Pick a log-spaced subset of `n_target` steps from `steps`.

    Always includes the first and last checkpoints. Intermediate picks
    minimize log-distance to a uniformly log-spaced ideal.
    """
    if len(steps) <= n_target:
        return list(steps)
    steps_arr = np.array(steps, dtype=np.float64)
    # Step 0 (if present) would break log spacing -- replace with 1.
    log_steps = np.log10(np.maximum(steps_arr, 1.0))
    target_log = np.linspace(log_steps[0], log_steps[-1], n_target)
    picked = set()
    for t in target_log:
        i = int(np.argmin(np.abs(log_steps - t)))
        picked.add(steps[i])
    return sorted(picked)


def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(run_dir, "multiview", f"seed_{seed}",
                        f"augmented_step_{step:08d}.npz")


# ----------------------------------------------------------------------
# Per-seed sweep over checkpoints.
# ----------------------------------------------------------------------
def run_one_seed(run_dir: str, seed: int, steps: List[int],
                 d: int, k: int, n_shuffles: int) -> Dict[str, np.ndarray]:
    """Compute R^2_pos at every (checkpoint, layer transition) for a seed.

    Returns a dict with arrays of shape (n_checkpoints, n_transitions)
    where applicable.
    """
    # Lazy multiview import (kept consistent with drift_welldef_test).
    from multiview import load_augmented_activations

    n_steps = len(steps)
    L_total = None
    n_transitions = None
    r2_mat = None
    null_mat = None
    ceiling_mat = None

    for si, step in enumerate(steps):
        path = augmented_path(run_dir, seed, step)
        if not os.path.exists(path):
            print(f"  [seed {seed}, step {step}] missing; skipping")
            continue
        print(f"  [seed {seed}, step {step}] loading ...", end=" ", flush=True)
        payload = load_augmented_activations(path)
        states = payload["states"]
        L = states.shape[0]
        if L_total is None:
            L_total = L
            n_transitions = L - 1
            r2_mat = np.full((n_steps, n_transitions), np.nan)
            null_mat = np.full((n_steps, n_transitions), np.nan)
            ceiling_mat = np.full((n_steps, n_transitions), np.nan)
        elif L != L_total:
            print(f"layer count mismatch ({L} vs {L_total}); skipping")
            continue

        for t in range(n_transitions):
            lr = analyze_layer(
                states[t], states[t + 1],
                d=d, k=k, n_shuffles=n_shuffles,
                rng_seed=10_000 * seed + 100 * (step % 1000) + t,
            )
            r2_mat[si, t] = lr["r2_pos"]
            null_mat[si, t] = lr["r2_pos_shuffle_mean"]
            ceiling_mat[si, t] = lr["x_var_captured"]
        print(f"done (R^2 mean across layers = "
              f"{np.nanmean(r2_mat[si]):.3f})")

    return {
        "steps": np.array(steps, dtype=np.int64),
        "r2": r2_mat,
        "null": null_mat,
        "ceiling": ceiling_mat,
    }


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------
# Phase 1 reference events (from PHASE_1_WRITEUP.md): the log_alpha hump
# peaks at step ~5000 across seeds. The post-final-norm anomaly reaches
# near-final magnitude by step ~5000. The mid-training Sigma-distance
# bump is centered in steps 5000-10000.
PHASE1_HUMP_STEP = 5000


def plot_training_trajectory(
    per_seed_results: List[Dict[str, np.ndarray]],
    steps: List[int],
    d: int, k: int,
    interior_layer: int, commit_layer: int,
    seeds_used: List[int],
    out_png: str,
) -> None:
    """Plot the three training-trajectory curves.

    Curves:
      - mean R^2 across all layer transitions
      - R^2 at the prediction-commitment layer (default 11 -> 12)
      - R^2 at the deep-interior layer (default 6 -> 7)

    If multiple seeds are present, draws cross-seed mean as a solid line
    and individual seeds as faint traces.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.array(steps, dtype=np.float64)

    # Stack the per-seed R^2 matrices: (n_seeds, n_steps, n_transitions).
    r2_stack = np.stack([r["r2"] for r in per_seed_results], axis=0)
    n_seeds = r2_stack.shape[0]
    n_transitions = r2_stack.shape[2]

    # Per-seed curves we'll derive.
    def curve(layer_or_mean: str | int) -> np.ndarray:
        """Return (n_seeds, n_steps) array for the chosen aggregation."""
        if layer_or_mean == "mean":
            return np.nanmean(r2_stack, axis=2)
        else:
            t = int(layer_or_mean)
            if t < 0 or t >= n_transitions:
                raise ValueError(
                    f"layer transition {t} out of range "
                    f"[0, {n_transitions - 1}]"
                )
            return r2_stack[:, :, t]

    series = [
        ("mean", "C0", "o", "mean across all layer transitions"),
        (commit_layer, "C3", "s",
         f"prediction-commitment layer ({commit_layer} -> {commit_layer + 1})"),
        (interior_layer, "C2", "^",
         f"deep-interior layer ({interior_layer} -> {interior_layer + 1})"),
    ]

    for key, color, marker, label in series:
        arr = curve(key)                                # (n_seeds, n_steps)
        mean_ = np.nanmean(arr, axis=0)
        if n_seeds > 1:
            std_ = np.nanstd(arr, axis=0)
            ax.fill_between(x, mean_ - std_, mean_ + std_,
                            color=color, alpha=0.18)
            # Light per-seed traces.
            for s in range(n_seeds):
                ax.plot(x, arr[s], color=color, lw=0.6, alpha=0.4)
        ax.plot(x, mean_, marker=marker, ms=6, lw=2,
                color=color, label=label)

    # Phase 1 reference: the log_alpha hump.
    ax.axvline(PHASE1_HUMP_STEP, color="black", ls="--", lw=1, alpha=0.5)
    ax.text(PHASE1_HUMP_STEP * 1.05, ax.get_ylim()[1] * 0.95,
            r"Phase 1 $\log\alpha$ hump (step ${\approx}5000$)",
            rotation=0, fontsize=8, color="black", alpha=0.7,
            ha="left", va="top")

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$R^2_{\mathrm{pos}}$")
    seed_str = (f"seed {seeds_used[0]}" if n_seeds == 1
                else f"seeds {','.join(str(s) for s in seeds_used)}")
    ax.set_title(
        f"Drift well-definedness across training\n"
        f"d={d}, k={k}, {seed_str}"
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


def plot_layer_heatmap(
    per_seed_results: List[Dict[str, np.ndarray]],
    steps: List[int],
    d: int, k: int,
    seeds_used: List[int],
    out_png: str,
) -> None:
    """Auxiliary plot: layer x training-step heatmap of R^2_pos.

    Reveals features the curve plot may collapse (e.g. layer 2->3 spike
    in seed 2 at the final checkpoint that you spotted earlier).
    """
    # Cross-seed mean of R^2 at each (step, layer).
    r2_stack = np.stack([r["r2"] for r in per_seed_results], axis=0)
    r2_mean = np.nanmean(r2_stack, axis=0)                # (n_steps, n_transitions)
    n_steps, n_transitions = r2_mean.shape

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # Show with steps on x-axis (log spacing) and layers on y-axis.
    # Use pcolormesh with explicit log-spaced edges in x for a clean
    # display.
    x = np.array(steps, dtype=np.float64)
    log_x = np.log10(np.maximum(x, 1.0))
    edges_x = np.empty(n_steps + 1)
    edges_x[1:-1] = (log_x[:-1] + log_x[1:]) / 2
    edges_x[0] = log_x[0] - (log_x[1] - log_x[0]) / 2
    edges_x[-1] = log_x[-1] + (log_x[-1] - log_x[-2]) / 2
    edges_x = 10 ** edges_x
    edges_y = np.arange(n_transitions + 1) - 0.5

    pcm = ax.pcolormesh(edges_x, edges_y, r2_mean.T,
                        cmap="viridis", vmin=0,
                        vmax=max(0.35, float(r2_mean.max())))
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"$R^2_{\mathrm{pos}}$")
    ax.set_xscale("log")
    ax.axvline(PHASE1_HUMP_STEP, color="white", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("training step")
    ax.set_ylabel("layer transition  (t -> t+1)")
    seed_str = (f"seed {seeds_used[0]}" if len(seeds_used) == 1
                else f"seeds {','.join(str(s) for s in seeds_used)}")
    ax.set_title(
        f"R^2_pos heatmap across training\n"
        f"d={d}, k={k}, {seed_str}"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"Saved {out_png}")


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--seed", type=int, default=0,
                    help="Single seed (used when --seeds not given)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated list of seeds (overrides --seed)")
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--n-shuffles", type=int, default=5,
                    help="Per-checkpoint shuffle replicates (gap-to-null "
                         "is huge at our sample size; 5 is plenty)")
    ap.add_argument("--n-checkpoints", type=int, default=12,
                    help="Sub-sample to this many log-spaced checkpoints")
    ap.add_argument("--interior-layer", type=int, default=6,
                    help="Layer transition for the deep-interior curve "
                         "(value t means t -> t+1)")
    ap.add_argument("--commit-layer", type=int, default=11,
                    help="Layer transition for the prediction-commitment "
                         "curve (value t means t -> t+1)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # Seed list.
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        seeds = [args.seed]

    out_dir = args.out_dir or os.path.join(args.run_dir, "drift_welldef")
    os.makedirs(out_dir, exist_ok=True)

    # Pick checkpoint set: log-spaced subsample of what seed 0 has.
    # (We require all seeds to have the same checkpoints; this is the
    # Phase 1 convention.)
    reference_seed = seeds[0]
    all_steps = checkpoints_in_seed(args.run_dir, reference_seed)
    if not all_steps:
        print(f"No augmented_step_*.npz found for seed {reference_seed} "
              f"in {args.run_dir}/multiview/seed_{reference_seed}/")
        sys.exit(1)
    steps = log_spaced_subsample(all_steps, args.n_checkpoints)
    print(f"Using {len(steps)} of {len(all_steps)} available checkpoints "
          f"for seed {reference_seed}:")
    print(f"  {steps}")

    # Run each seed.
    per_seed_results = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        result = run_one_seed(args.run_dir, seed, steps,
                              d=args.d, k=args.k,
                              n_shuffles=args.n_shuffles)
        per_seed_results.append(result)

    # Save raw matrices for later replotting.
    seed_tag = ("seeds_" + "_".join(str(s) for s in seeds)
                if len(seeds) > 1 else f"seed_{seeds[0]}")
    stem = f"training_trajectory_{seed_tag}_d_{args.d}_k_{args.k}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez(
        npz_path,
        steps=np.array(steps, dtype=np.int64),
        seeds=np.array(seeds, dtype=np.int32),
        # Stack as (n_seeds, n_steps, n_transitions).
        r2=np.stack([r["r2"] for r in per_seed_results], axis=0),
        null=np.stack([r["null"] for r in per_seed_results], axis=0),
        ceiling=np.stack([r["ceiling"] for r in per_seed_results], axis=0),
        config=np.array([args.d, args.k, args.n_shuffles,
                         args.interior_layer, args.commit_layer],
                        dtype=np.int64),
    )
    print(f"Saved {npz_path}")

    # Plot the headline trajectory.
    png_path = os.path.join(out_dir, f"{stem}.png")
    plot_training_trajectory(
        per_seed_results, steps,
        d=args.d, k=args.k,
        interior_layer=args.interior_layer,
        commit_layer=args.commit_layer,
        seeds_used=seeds,
        out_png=png_path,
    )

    # Auxiliary heatmap.
    heatmap_path = os.path.join(out_dir, f"{stem}_heatmap.png")
    plot_layer_heatmap(
        per_seed_results, steps,
        d=args.d, k=args.k,
        seeds_used=seeds,
        out_png=heatmap_path,
    )

    # Print a tidy summary table.
    print()
    print("=" * 72)
    print("Summary (mean R^2 across layer transitions, per checkpoint, "
          "per seed):")
    print()
    header = " step       " + "".join(f"seed{s:>2} " for s in seeds) + " mean"
    print(header)
    print("-" * len(header))
    for si, step in enumerate(steps):
        line = f" {step:>9}  "
        seed_means = []
        for r in per_seed_results:
            v = np.nanmean(r["r2"][si])
            seed_means.append(v)
            line += f"{v:>6.3f} "
        line += f" {np.mean(seed_means):>5.3f}"
        print(line)
    print("=" * 72)


if __name__ == "__main__":
    main()
