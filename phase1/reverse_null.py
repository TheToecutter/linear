"""
Shuffle-null protocol for the reverse build-up D5 reconstruction.

Hypothesis F4 (reverse-actual vs reverse-predicted reconstruction
divergence) is structurally vulnerable to the same null-correction
risk that retroactively undermined §7 of INVESTIGATION_WRITEUP.md:
simple random labelings reduce kurtosis by amounts comparable to the
real labels, so any apparent F4 signal must be compared to a
frequency-preserving null before claiming a finding.

This module implements the within-chunk shuffled-label protocol:

    For each pilot, identify its source chunk (via `positions` and the
    chunk-ordering implicit in the augmented file). Within each chunk,
    permute the partition labels (next_ids or pred_ids) across pilots.
    Frequencies of each label per chunk are preserved exactly; the
    correspondence between specific pilots and specific labels is
    destroyed. Run D5 with the shuffled label array.

The "within-chunk" constraint matters: shuffling globally across all
pilots from all chunks destroys not just the (pilot, label) link but
also the per-chunk label frequency profile, which would conflate two
distinct sources of structure. Within-chunk shuffling kills only the
specific link we want to test for, leaving the empirical marginal
unchanged.

The null is invoked from the campaign driver as part of the D5 pass.
It is mandatory (R2 mitigation in the proposal); the F4 signal is
reported as null-corrected delta:

    delta_null_corrected(t) = (k^B-rev-act(t) - k^B-rev-act-shuffled(t))
                            - (k^B-rev-pred(t) - k^B-rev-pred-shuffled(t))

If the shuffle null absorbs more than 70% of the raw F4 signal, F4 is
reported as unresolved per the proposal's R2 mitigation.

The shuffle is deterministic given a seed argument so the null is
reproducible.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from multiview import load_augmented_activations
from multiview_campaign import augmented_path
from reverse_buildup import (
    VIEWS,
    output_root,
    run_d5_view,
    select_id_array_for_view,
)


# ----------------------------------------------------------------------
# Shuffle constructor.
# ----------------------------------------------------------------------
def within_chunk_shuffle(
    id_array: np.ndarray,
    positions: np.ndarray,
    rng_seed: int = 20260522,
) -> np.ndarray:
    """Permute `id_array` within each chunk, preserving per-chunk label
    frequencies.

    A "chunk" is identified by a contiguous block of pilots that share
    the same source sequence in the held-out set. The collector
    `collect_activations_with_metadata` lays out pilots in the order
    (chunk_0_position_0, chunk_0_position_1, ..., chunk_1_position_0, ...),
    so the chunk boundary is wherever `positions` resets to its first
    pilot-position value after having advanced.

    Args:
        id_array:  (N,) integer labels.
        positions: (N,) the per-pilot position-in-chunk array from the
                   augmented file.
        rng_seed:  determines the permutation. Must be set explicitly so
                   null runs are reproducible across campaigns.

    Returns:
        (N,) array of the same dtype as `id_array` with labels permuted
        within each detected chunk.

    Implementation detail: chunks are detected by finding indices where
    `positions[i] <= positions[i-1]`. This works because pilot positions
    within a chunk are strictly increasing (the collector enumerates
    them in order from default_pilot_positions). A reset signals the
    start of a new chunk.
    """
    N = id_array.size
    if N == 0:
        return id_array.copy()
    rng = np.random.default_rng(rng_seed)

    # Detect chunk boundaries: positions[i] <= positions[i-1] -> new chunk.
    boundaries = [0]
    for i in range(1, N):
        if positions[i] <= positions[i - 1]:
            boundaries.append(i)
    boundaries.append(N)

    out = id_array.copy()
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if b - a > 1:
            perm = rng.permutation(b - a)
            out[a:b] = id_array[a:b][perm]
    return out


def _verify_frequency_preserving(orig: np.ndarray,
                                  shuffled: np.ndarray) -> Dict:
    """Sanity-check that shuffling preserves global label frequencies.

    Returns a small report; the campaign driver may log it. We verify
    here rather than in tests so the campaign aborts early if the
    shuffle is broken.
    """
    u_o, c_o = np.unique(orig, return_counts=True)
    u_s, c_s = np.unique(shuffled, return_counts=True)
    same_keys = np.array_equal(u_o, u_s)
    same_counts = same_keys and np.array_equal(c_o, c_s)
    return {
        "same_keys": bool(same_keys),
        "same_counts": bool(same_counts),
        "n_unique": int(u_o.size),
    }


# ----------------------------------------------------------------------
# Campaign-level null D5 driver.
# ----------------------------------------------------------------------
def run_d5_shuffle_null(
    run_dir: str,
    steps_to_analyze: List[Tuple[int, int]],
    view: str,
    rng_seed: int = 20260522,
    max_pc_dim: int = 32,
    n_samples: int = 5000,
    verbose: bool = True,
) -> List[Dict]:
    """Run D5 with within-chunk shuffled labels for the requested view.

    Output files live at
    d5_reconstruction_{view}_shuffled/seed{S}_step{T:08d}.npz, parallel
    to the unshuffled run. The shuffle is regenerated per (seed, step)
    using rng_seed + (seed << 32 | step) as the per-checkpoint stream
    seed; this keeps the null reproducible while making each
    checkpoint's null independent.

    Args:
        run_dir, steps_to_analyze, view, max_pc_dim, n_samples: same as
            run_d5_view.
        rng_seed: top-level rng seed; combined with (seed, step) to
            derive a per-checkpoint stream.

    Returns:
        List of result dicts (the same shape run_d5_view returns), one
        per (seed, step) that had an augmented file on disk.
    """
    if view == "forward":
        # Forward null doesn't make sense in this project (the proposal
        # is about reverse build-up), but the code path supports it for
        # symmetry / sanity testing.
        if verbose:
            print(f"[D5 null/{view}] Note: forward null is mostly a "
                  f"sanity check; reverse_actual / reverse_pred are the "
                  f"primary targets of this protocol.")

    results = []
    for seed, step in steps_to_analyze:
        aug_path_str = augmented_path(run_dir, seed, step)
        if not os.path.exists(aug_path_str):
            if verbose:
                print(f"[D5 null/{view}] seed {seed} step {step}: "
                      f"missing augmented file, skip")
            continue
        aug = load_augmented_activations(aug_path_str)
        id_array = select_id_array_for_view(aug, view)
        positions = aug["positions"]

        # Per-checkpoint stream seed: deterministic, independent across
        # (seed, step).
        stream_seed = (rng_seed
                       ^ (int(seed) * 2654435761)
                       ^ (int(step) * 1597334677)) & 0xFFFFFFFF
        shuffled = within_chunk_shuffle(id_array, positions, rng_seed=stream_seed)
        report = _verify_frequency_preserving(id_array, shuffled)
        if not report["same_counts"]:
            raise RuntimeError(
                f"[D5 null/{view}] seed {seed} step {step}: "
                f"shuffle is not frequency-preserving — "
                f"shuffler is broken or `positions` array is malformed."
            )
        if verbose:
            print(f"[D5 null/{view}] seed {seed} step {step}: shuffle ok "
                  f"({report['n_unique']} unique labels preserved)")

        # Feed the shuffled labels through the standard D5 machinery via
        # the id_array_override hook.
        partial = run_d5_view(
            run_dir,
            steps_to_analyze=[(seed, step)],
            view=view,
            max_pc_dim=max_pc_dim,
            n_samples=n_samples,
            id_array_override=shuffled,
            output_suffix="_shuffled",
            verbose=verbose,
        )
        results.extend(partial)
    return results


# ----------------------------------------------------------------------
# Null-correction helper.
# ----------------------------------------------------------------------
def null_corrected_kurtosis(
    real_d5: Dict,
    shuffled_d5: Dict,
) -> Dict[str, np.ndarray]:
    """Compute null-corrected reconstruction kurtosis from a paired D5
    result (real labels and shuffled-null labels for the same view at
    the same checkpoint).

    Returns:
        {
            'empirical_kurt':    empirical marginal kurtosis (real),
            'recon_A_kurt_corr': real recon_A_kurt - shuffled recon_A_kurt,
            'recon_B_kurt_corr': real recon_B_kurt - shuffled recon_B_kurt,
        }
    The intuition: if shuffled labels (which carry no information)
    produce a reconstruction with kurtosis K_shuf, and the real labels
    produce K_real, then K_real - K_shuf is the kurtosis component
    attributable to the *label structure* rather than the residual-
    stream + partitioning noise. The corrected value is what should be
    compared to the empirical marginal kurtosis.
    """
    if real_d5["seed"] != shuffled_d5["seed"] \
       or real_d5["step"] != shuffled_d5["step"]:
        raise ValueError(
            "Paired D5 results must have matching seed and step; got "
            f"real=({real_d5['seed']}, {real_d5['step']}), "
            f"shuf=({shuffled_d5['seed']}, {shuffled_d5['step']})"
        )
    return {
        "empirical_kurt": np.asarray(real_d5["empirical_kurt"], dtype=np.float64),
        "recon_A_kurt_corr": (
            np.asarray(real_d5["recon_A_kurt"], dtype=np.float64)
            - np.asarray(shuffled_d5["recon_A_kurt"], dtype=np.float64)
        ),
        "recon_B_kurt_corr": (
            np.asarray(real_d5["recon_B_kurt"], dtype=np.float64)
            - np.asarray(shuffled_d5["recon_B_kurt"], dtype=np.float64)
        ),
    }


def f4_signal_with_null_correction(
    rev_actual_real: Dict,
    rev_actual_shuf: Dict,
    rev_pred_real: Dict,
    rev_pred_shuf: Dict,
) -> Dict[str, np.ndarray]:
    """Compute the F4 signal (predicted-vs-actual reverse reconstruction
    divergence) with full null correction.

    Definition:

      Delta_raw(t)  = k^B-rev-pred-real(t) - k^B-rev-act-real(t)
      Delta_null(t) = k^B-rev-pred-shuf(t) - k^B-rev-act-shuf(t)
      Delta_corr(t) = Delta_raw(t) - Delta_null(t)

    The proposal's R2 mitigation: if the null absorbs more than 70% of
    the raw signal at the layers of interest, F4 is reported as
    unresolved. We don't apply that gate here -- it's a reporting
    decision -- but we return both quantities so the gate can be
    applied downstream.

    Returns:
        {
            'delta_raw':  Delta_raw(t),    (L,)
            'delta_null': Delta_null(t),   (L,)
            'delta_corr': Delta_corr(t),   (L,)
            'null_absorption': |Delta_null| / |Delta_raw| at each layer.
                               Per the R2 gate, F4 is unresolved where
                               this is > 0.7.
        }
    """
    a = np.asarray(rev_actual_real["recon_B_kurt"], dtype=np.float64)
    b = np.asarray(rev_pred_real["recon_B_kurt"], dtype=np.float64)
    a_s = np.asarray(rev_actual_shuf["recon_B_kurt"], dtype=np.float64)
    b_s = np.asarray(rev_pred_shuf["recon_B_kurt"], dtype=np.float64)
    delta_raw = b - a
    delta_null = b_s - a_s
    delta_corr = delta_raw - delta_null
    with np.errstate(invalid="ignore", divide="ignore"):
        absorption = np.where(
            np.abs(delta_raw) > 1e-30,
            np.abs(delta_null) / np.abs(delta_raw),
            np.nan,
        )
    return {
        "delta_raw": delta_raw,
        "delta_null": delta_null,
        "delta_corr": delta_corr,
        "null_absorption": absorption,
    }
