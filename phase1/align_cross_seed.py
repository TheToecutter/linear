"""
Cross-seed Procrustes alignment for the residual-stream ensemble.

Question: at the final checkpoint, do different training seeds (same
architecture, same data, same recipe) produce residual streams that
occupy the same subspace of R^H up to rotation, or do they merely
exhibit similar statistical properties?

This script answers that by computing per-layer orthogonal Procrustes
alignments between every pair of seeds, then comparing the resulting
residual ratios to two null baselines:

  - Random-rotation floor: a known-recoverable alignment (apply a
    random rotation to seed 0's activations, then align back). The
    residual here is essentially zero modulo numerical precision; it's
    the floor of what perfect cross-seed alignment would look like.

  - Random-scramble ceiling: scramble the pilot ordering in one seed
    so the per-pilot correspondence is broken. The residual here is
    the worst-case "no alignment possible" reference; the marginal
    distributions still match but there's nothing to align row-by-row.

If the trained-vs-trained residuals across seed pairs sit close to
the floor, the residual-stream subspaces are aligned across seeds.
If they sit close to the ceiling, the subspaces are genuinely
different and the cross-seed agreement of basis-invariant statistics
(seen in fig2, fig6, etc.) is a statistical-similarity result rather
than a geometric-identity result.

The script also supports restricting the alignment to specific
conditional subsets — e.g., aligning only pilots whose input token is
in seed 0's top-20 forward set — to test whether the conditional
bundles align even if the full ensemble does not.

Usage:
    python3 align_cross_seed.py [--run-dir PATH]
                                [--step STEP]
                                [--n-pilots N]

Defaults: ../phase1_runs_gelu, step 24000 (final), all 10,000 pilots.

Output:
    ../phase1_runs_gelu/figures/fig10_procrustes_residuals.png
    ../phase1_runs_gelu/multiview/procrustes/
        residuals_step_<STEP>.npz   (raw residual data for downstream analysis)
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from align import align_activations_per_layer, orthogonal_procrustes
from multiview import load_augmented_activations


def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(run_dir, "multiview", f"seed_{seed}",
                        f"augmented_step_{step:08d}.npz")


def load_states_for_pair(run_dir: str, seed_a: int, seed_b: int, step: int,
                         n_pilots: int) -> Tuple[np.ndarray, np.ndarray,
                                                  Dict[str, np.ndarray]]:
    """Load and validate two seeds' activations at a checkpoint.

    Returns:
        states_a, states_b: (L, n_pilots, H) each.
        metadata: dict with 'input_ids', 'next_ids', 'pred_ids', 'positions'
                  from seed_a (used for conditional-view restriction).

    Raises AssertionError if the per-pilot correspondence isn't preserved
    across seeds. Since the held-out dataloader is deterministic and the
    activation collection iterates it the same way for every seed, the
    i-th pilot's input token, next token, and position should match
    between seeds (only the activations differ, since the *model* changed).
    """
    payload_a = load_augmented_activations(augmented_path(run_dir, seed_a, step))
    payload_b = load_augmented_activations(augmented_path(run_dir, seed_b, step))

    # Sanity check the per-pilot correspondence. Input token ids and
    # actual-next token ids must match across seeds for the same pilot
    # index; positions too.
    assert np.array_equal(payload_a["input_ids"], payload_b["input_ids"]), (
        f"input_ids differ between seed {seed_a} and seed {seed_b} at "
        f"step {step} — the held-out dataloader is not deterministic, "
        f"so per-pilot Procrustes correspondence is broken."
    )
    assert np.array_equal(payload_a["next_ids"], payload_b["next_ids"]), (
        f"next_ids differ between seed {seed_a} and seed {seed_b}"
    )
    assert np.array_equal(payload_a["positions"], payload_b["positions"]), (
        f"positions differ between seed {seed_a} and seed {seed_b}"
    )

    states_a = payload_a["states"][:, :n_pilots, :]
    states_b = payload_b["states"][:, :n_pilots, :]
    metadata = {
        "input_ids": payload_a["input_ids"][:n_pilots],
        "next_ids": payload_a["next_ids"][:n_pilots],
        "pred_ids": payload_a["pred_ids"][:n_pilots],
        "positions": payload_a["positions"][:n_pilots],
    }
    return states_a, states_b, metadata


def compute_pair_residuals(states_a: np.ndarray, states_b: np.ndarray
                           ) -> np.ndarray:
    """Layer-by-layer Procrustes residual ratios between two seeds."""
    _, residuals = align_activations_per_layer(states_a, states_b)
    return np.array(residuals)


def compute_random_rotation_floor(states: np.ndarray, rng: np.random.Generator
                                   ) -> np.ndarray:
    """Apply a fresh random orthogonal rotation to states, then align
    back. Residuals here should be ~0 (modulo fp precision)."""
    L, N, H = states.shape
    floor = np.zeros(L)
    for t in range(L):
        # Generate random orthogonal matrix via QR of a random Gaussian.
        A = rng.standard_normal(size=(H, H)).astype(np.float64)
        Q_rand, _ = np.linalg.qr(A)
        X_rotated = states[t].astype(np.float64) @ Q_rand
        _, r = orthogonal_procrustes(X_rotated, states[t])
        floor[t] = r
    return floor


def compute_scramble_ceiling(states_a: np.ndarray, states_b: np.ndarray,
                             rng: np.random.Generator,
                             n_repeats: int = 5) -> np.ndarray:
    """Permute the pilot order in B (so per-pilot correspondence is
    broken), then align A to B-permuted. Returns the cross-permutation
    mean residual per layer.

    The ceiling captures: 'two random subsets with the same marginal
    distribution.' If trained-vs-trained sits below this ceiling, the
    alignment is finding genuine per-pilot correspondence; if it sits
    at the ceiling, the seeds are no more aligned than random
    re-orderings of each other."""
    L, N, _ = states_a.shape
    residuals_layers = np.zeros((n_repeats, L))
    for rep in range(n_repeats):
        perm = rng.permutation(N)
        states_b_scrambled = states_b[:, perm, :]
        for t in range(L):
            _, r = orthogonal_procrustes(
                states_a[t].astype(np.float64),
                states_b_scrambled[t].astype(np.float64),
            )
            residuals_layers[rep, t] = r
    return residuals_layers.mean(axis=0)


def restrict_to_conditional_set(states: np.ndarray, tags: np.ndarray,
                                target_tags: np.ndarray) -> np.ndarray:
    """Keep only pilots whose tag is in target_tags."""
    mask = np.isin(tags, target_tags)
    return states[:, mask, :]


def plot_residuals(layers: np.ndarray, pair_residuals: Dict[str, np.ndarray],
                   floor: np.ndarray, ceiling: np.ndarray,
                   step: int, output_path: str) -> None:
    """Plot per-layer Procrustes residuals across seed pairs vs the
    two null baselines."""
    fig, ax = plt.subplots(figsize=(11, 6.5))

    pair_arr = np.array(list(pair_residuals.values()))  # (n_pairs, L)
    pair_mean = pair_arr.mean(axis=0)
    pair_min = pair_arr.min(axis=0)
    pair_max = pair_arr.max(axis=0)

    # Faint per-pair lines.
    for label, vals in pair_residuals.items():
        ax.plot(layers, vals, "-", color="C0", alpha=0.30, lw=0.9)
    # Bold mean across pairs.
    ax.plot(layers, pair_mean, "-o", color="C0", lw=2.2, markersize=5,
            label=f"trained pairs (mean of {len(pair_residuals)})", zorder=5)
    ax.fill_between(layers, pair_min, pair_max, color="C0", alpha=0.10,
                    label="trained pairs (min/max envelope)")

    # Floor: random-rotation baseline. Often essentially zero; mark with
    # a horizontal text annotation since the line itself may not be
    # visible against the axis.
    ax.plot(layers, floor, "--o", color="C2", lw=1.5, markersize=4,
            label=f"random-rotation floor ({floor.mean():.1e})", zorder=4)

    # Ceiling: scrambled-pilot baseline. Note for the writeup: this can
    # sit slightly above 1 because the cross-seed Frobenius norm ratio
    # isn't exactly 1 even after Procrustes — it's the structurally
    # worst-case alignment, not a strict upper bound at 1.
    ax.plot(layers, ceiling, "--s", color="C3", lw=1.5, markersize=4,
            label=f"random-scramble ceiling ({ceiling.mean():.3f})", zorder=4)

    ax.set_xlabel("layer state index t")
    ax.set_ylabel(r"Procrustes residual ratio  $\|X_A Q - X_B\|_F / \|X_B\|_F$")
    ax.set_xticks(layers)
    ax.set_xlim(-0.3, layers[-1] + 0.3)

    # y-axis: include the ceiling with some headroom, and don't go below
    # zero (the floor is essentially zero).
    y_max = max(ceiling.max() * 1.10, pair_max.max() * 1.10)
    ax.set_ylim(-0.02, y_max)

    # Annotate the "alignment-quality scale" in the middle-left of the
    # chart (less likely to overlap the curves than the top-left,
    # because the trained-pair curve sits low).
    mean_ratio = pair_mean.mean() / ceiling.mean()
    ax.text(0.02, 0.50,
            f"Trained pair residual ≈ {pair_mean.mean():.3f}\n"
            f"Ceiling ≈ {ceiling.mean():.3f}\n"
            f"Trained/ceiling ≈ {mean_ratio:.2%}",
            transform=ax.transAxes, va="center", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.92,
                      pad=4.0, lw=0.5))

    ax.set_title(f"Cross-seed Procrustes alignment of residual-stream activations\n"
                 f"(final checkpoint, step {step}; lower = better aligned)")
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(layers: np.ndarray, pair_residuals: Dict[str, np.ndarray],
                  floor: np.ndarray, ceiling: np.ndarray) -> None:
    """Print a compact summary table."""
    pair_arr = np.array(list(pair_residuals.values()))
    pair_mean = pair_arr.mean(axis=0)
    pair_std = pair_arr.std(axis=0)
    print()
    print(f"Per-layer Procrustes residual ratios:")
    print(f"{'t':>3} {'floor':>9} {'pair mean':>11} {'pair std':>10} "
          f"{'ceiling':>9}  {'pair/ceiling':>14}")
    print("-" * 64)
    for i, t in enumerate(layers):
        ratio_to_ceiling = pair_mean[i] / ceiling[i] if ceiling[i] > 0 else float('nan')
        print(f"{t:>3} {floor[i]:>9.5f} {pair_mean[i]:>11.4f} "
              f"{pair_std[i]:>10.4f} {ceiling[i]:>9.4f} {ratio_to_ceiling:>14.3f}")
    print()
    print("Interpretation:")
    print("  pair/ceiling near 0 -> trained subspaces align well across seeds")
    print("  pair/ceiling near 1 -> seeds occupy genuinely different subspaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../phase1_runs_gelu",
                    help="Path to the phase 1 run directory.")
    ap.add_argument("--step", type=int, default=24000,
                    help="Checkpoint step to analyze (default: 24000, final).")
    ap.add_argument("--n-pilots", type=int, default=10_000,
                    help="Number of pilots to use. The full set is 10,000.")
    ap.add_argument("--figures-dir", default="../phase1_runs_gelu/figures",
                    help="Where to write the comparison figure.")
    args = ap.parse_args()

    # Find available seeds.
    mv_root = os.path.join(args.run_dir, "multiview")
    seeds = sorted(int(d.removeprefix("seed_"))
                   for d in os.listdir(mv_root)
                   if d.startswith("seed_") and d[5:].isdigit())
    print(f"Found seeds: {seeds}")
    if len(seeds) < 2:
        raise RuntimeError(f"Need at least 2 seeds; found {seeds}")

    # Procrustes residuals for every seed pair.
    pair_residuals: Dict[str, np.ndarray] = {}
    floor: np.ndarray = None
    ceiling: np.ndarray = None
    rng = np.random.default_rng(0)

    for seed_a, seed_b in itertools.combinations(seeds, 2):
        print(f"Aligning seed {seed_a} <-> seed {seed_b} at step {args.step} ...")
        states_a, states_b, _ = load_states_for_pair(
            args.run_dir, seed_a, seed_b, args.step, args.n_pilots)
        residuals = compute_pair_residuals(states_a, states_b)
        pair_residuals[f"{seed_a}-{seed_b}"] = residuals
        print(f"  layer-mean residual: {residuals.mean():.4f}  "
              f"(layer-max: {residuals.max():.4f})")

        # Compute floor and ceiling once, using one representative pair's
        # activations. (Both quantities are intrinsic to a single seed's
        # activations and don't depend on the pairing.)
        if floor is None:
            print(f"  computing random-rotation floor ...")
            floor = compute_random_rotation_floor(states_a, rng)
            print(f"  computing random-scramble ceiling ...")
            ceiling = compute_scramble_ceiling(states_a, states_b, rng,
                                               n_repeats=3)

    L = next(iter(pair_residuals.values())).size
    layers = np.arange(L)

    print_summary(layers, pair_residuals, floor, ceiling)

    # Save raw data for downstream use.
    out_dir = os.path.join(args.run_dir, "multiview", "procrustes")
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, f"residuals_step_{args.step:08d}.npz")
    pair_keys = list(pair_residuals.keys())
    pair_arr = np.array([pair_residuals[k] for k in pair_keys])  # (n_pairs, L)
    np.savez(data_path,
             layers=layers, pair_keys=np.array(pair_keys),
             pair_residuals=pair_arr,
             floor=floor, ceiling=ceiling,
             step=np.int64(args.step))
    print(f"\nRaw residuals saved to {data_path}")

    # Plot.
    os.makedirs(args.figures_dir, exist_ok=True)
    fig_path = os.path.join(args.figures_dir, "fig10_procrustes_residuals.png")
    plot_residuals(layers, pair_residuals, floor, ceiling, args.step, fig_path)
    print(f"Figure written to {fig_path}")


if __name__ == "__main__":
    main()
