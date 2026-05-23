"""
Modified Test 3: cross-cell structure in conditional deviations.

Background
----------
The drift-Markovianity test (drift_welldef_test.py / drift_welldef_training.py)
established that ~88% of the per-layer update Delta_x_t is not a
function of x_t and is instead driven by context that does not project
visibly into x_t. We interpret this 88% as the learned function the
network constructs by making attention selective.

This script characterizes that learned function. It supports two modes:

    --mode pair: pair-conditional view. For each (input v, successor w)
        bigram cell, compute
            d_{v,w}(t) = mean_{cell}(x_t) - mu_v(t)
        where mu_v(t) is the input-marginal mean trajectory (mean over
        pilots whose input is v, regardless of successor). The
        quantity d_{v,w}(t) is the cell-specific deviation from the
        input-marginal flow — the part that depends on w in addition
        to v. SAMPLE-INTENSIVE: requires good bigram density.

    --mode reverse: reverse view. For each successor token w, compute
            d_w(t) = mean_{pilots predicting w}(x_t) - mu(t)
        where mu(t) is the all-pilots marginal mean trajectory. The
        quantity d_w(t) is the part of the trajectory that depends on
        w in addition to nothing. SAMPLE-EFFICIENT: top-K successor
        tokens each have hundreds of pilots, even at modest N.

The reverse view is preferred when bigram sample sizes are
insufficient (e.g., < ~30 cells with ≥30 pilots each at N=10k). It
answers the closely-related question "what does conditioning on the
output add to the trajectory" rather than "what does the bigram add
to the input-conditioned trajectory," but both are direct probes of
the learned function the framework treats as noise.

Note on the prediction reference
--------------------------------
We use empirical marginal means rather than the SDE-derived linear-
flow extrapolation. The Markovianity result established that the
linear-flow drift only accounts for ~12% of Delta_x_t, so using it
as a baseline would conflate two effects: "deviation from the
marginal flow" and "deviation from a known-bad baseline." Using
empirical marginals isolates exactly what we want.

Stacking and analysis (mode-independent)
----------------------------------------
The cell-conditional deviations are stacked into a tensor
D in R^{|P| x L x H} for |P| cells, L layers, H hidden dims.
We decompose D by SVD two ways:

  - "Direction view": each (cell, layer) gives a vector in R^H;
    SVD of the ((|P|*L), H) matrix reveals shared *directions* in
    residual-stream space.

  - "Trajectory view": each cell gives a full L*H trajectory deviation;
    SVD of the (|P|, L*H) matrix reveals shared *trajectory shapes*
    across cells.

Hypotheses
----------
H_low_rank: The deviation tensor has effective rank substantially
    below the shuffle null. The learned function uses a small
    dictionary of shared axes across cells. PASS if real effective
    rank is > 5σ below null and at least 30% reduced.

H_full_rank: The deviation tensor has effective rank above the
    shuffle null. Each cell drives the residual stream along a
    private direction. PASS if real effective rank is > 3σ above null.

H_no_signal: The deviation tensor has effective rank equal to the
    null. The conditioning token contributes no detectable structure
    beyond what the marginal already determines.

The null is mode-specific:
  - pair mode: within-input shuffle (permute cell labels among
    pilots sharing the same input token). This preserves the per-cell
    "shares v" property and isolates the effect of the successor w.
  - reverse mode: global shuffle (permute cell labels across all
    pilots). The reverse view's marginal subtraction is the same for
    all pilots, so there is no input-grouping to preserve.

Cross-seed reproducibility:
H_cross_seed: Effective ranks and singular value spectra of the
    deviation tensor are reproducible across the four trained seeds.
    Note: top singular *vectors* are basis-dependent and seeds learn
    different bases, so we measure agreement via singular *value
    spectra* (basis-invariant) and via the effective rank statistic.

Usage
-----
Single-checkpoint mode (default):
    python3 pair_deviation_test.py
        [--mode {pair, reverse}]    # default 'pair'
        [--run-dir PATH]            # default ../phase1_runs_gelu
        [--seed S]                  # default 0
        [--step STEP]               # default 24000
        [--min-pilots-per-cell N]   # default 30
        [--top-k-cells N]           # default 100
        [--top-k-tokens-v N]        # default 100 (pair mode only)
        [--n-shuffles N]            # default 20

Training-trajectory mode:
    python3 pair_deviation_test.py --training
        [--mode {pair, reverse}]
        [--seeds 0 1 2 3]           # default auto-detect
        [--n-checkpoints 12]        # log-spaced subsample
        (other args as above)

Outputs
-------
Single-checkpoint mode:
    pair_deviation/<mode>_seed_S_step_STEP.npz
        cells:           (n_cells, 2) int — token ids per cell
        cell_counts:     (n_cells,) int     — pilots per cell
        deviation:       (n_cells, L, H) float — d(t) per cell
        sv_directions, sv_trajectories: SVD spectra
        energy_directions, energy_trajectories: cumulative energy
        null_*: shuffle baseline statistics
        verdict, dictionary_dimension, peak_excess, peak_z
        config: run parameters
    pair_deviation/<mode>_seed_S_step_STEP.png — 3-panel summary

Training-trajectory mode:
    pair_deviation/<mode>_trajectory_seeds_S0_S1_S2_S3.npz
        seeds, steps, mode
        dictionary_dimension (n_seeds, n_steps)
        peak_excess         (n_seeds, n_steps)
        peak_z              (n_seeds, n_steps)
        effective_rank_trajectories       (n_seeds, n_steps)
        null_effective_rank_trajectories  (n_seeds, n_steps)
        energy_trajectories_real (n_seeds, n_steps, n_cells_max)
        energy_trajectories_null (n_seeds, n_steps, n_cells_max)
    pair_deviation/<mode>_trajectory_seeds_*.png — 3-panel trajectory
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


# Lazy multiview import.
def _load_augmented_activations(path: str) -> Dict[str, np.ndarray]:
    from multiview import load_augmented_activations
    return load_augmented_activations(path)


# ----------------------------------------------------------------------
# Cell selection.
# ----------------------------------------------------------------------
def select_pair_cells(
    input_ids: np.ndarray,
    next_ids: np.ndarray,
    min_pilots_per_cell: int,
    top_k_cells: int,
    top_k_tokens_v: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select (input, next) bigram cells with enough pilots.

    Args:
        input_ids: (N,) integer array of input token ids per pilot.
        next_ids:  (N,) integer array of successor token ids per pilot.
        min_pilots_per_cell: minimum pilot count for a cell to be kept.
        top_k_cells: keep at most this many cells, sorted by count desc.
        top_k_tokens_v: if > 0, restrict to cells whose input token is
            among the top-k most-frequent input tokens. 0 means no
            input filter.

    Returns:
        cells: (n_cells, 2) int — selected (input_id, next_id) pairs.
        counts: (n_cells,) int — pilot counts per selected cell.
        pilot_assignments: (N,) int — cell index in [0, n_cells) for
            each pilot, or -1 if the pilot is not in any selected cell.
    """
    # Optionally restrict to top-K input tokens.
    if top_k_tokens_v > 0:
        v_uniq, v_counts = np.unique(input_ids, return_counts=True)
        order = np.argsort(-v_counts)
        kept_v = set(v_uniq[order][:top_k_tokens_v].tolist())
        v_mask = np.array([int(v) in kept_v for v in input_ids])
    else:
        v_mask = np.ones_like(input_ids, dtype=bool)

    # Bigram counting via packed-id trick (assumes ids fit in 32 bits).
    packed = (input_ids.astype(np.int64) << 32) | next_ids.astype(np.int64)
    packed_masked = packed[v_mask]
    uniq_packed, counts = np.unique(packed_masked, return_counts=True)
    keep = counts >= min_pilots_per_cell
    uniq_packed = uniq_packed[keep]
    counts = counts[keep]
    # Sort by count descending.
    order = np.argsort(-counts)
    uniq_packed = uniq_packed[order][:top_k_cells]
    counts = counts[order][:top_k_cells]

    # Unpack.
    cells = np.stack([
        (uniq_packed >> 32).astype(np.int32),
        (uniq_packed & 0xFFFFFFFF).astype(np.int32),
    ], axis=1)
    n_cells = cells.shape[0]

    # Build the pilot-to-cell assignment.
    cell_lookup = {(int(c[0]), int(c[1])): i for i, c in enumerate(cells)}
    pilot_assignments = np.full(input_ids.shape[0], -1, dtype=np.int32)
    for k in range(input_ids.shape[0]):
        key = (int(input_ids[k]), int(next_ids[k]))
        if key in cell_lookup:
            pilot_assignments[k] = cell_lookup[key]

    return cells, counts, pilot_assignments


def select_reverse_cells(
    next_ids: np.ndarray,
    min_pilots_per_cell: int,
    top_k_cells: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select successor-token cells with enough pilots (reverse view).

    Args:
        next_ids: (N,) integer array of successor token ids per pilot.
        min_pilots_per_cell: minimum pilot count for a cell.
        top_k_cells: keep at most this many cells.

    Returns:
        cells: (n_cells, 2) int — column 0 is sentinel -1 (no input
            conditioning), column 1 is the successor token. The
            (n_cells, 2) shape matches the pair-mode return signature
            so downstream code is mode-agnostic.
        counts: (n_cells,) int — pilot counts.
        pilot_assignments: (N,) int — cell index for each pilot, -1
            if not in any selected cell.
    """
    uniq, counts = np.unique(next_ids, return_counts=True)
    keep = counts >= min_pilots_per_cell
    uniq = uniq[keep]
    counts = counts[keep]
    order = np.argsort(-counts)
    uniq = uniq[order][:top_k_cells]
    counts = counts[order][:top_k_cells]

    cells = np.stack([
        np.full(uniq.shape[0], -1, dtype=np.int32),       # sentinel for "no input"
        uniq.astype(np.int32),
    ], axis=1)
    n_cells = cells.shape[0]

    # Pilot-to-cell.
    cell_lookup = {int(w): i for i, w in enumerate(uniq)}
    pilot_assignments = np.full(next_ids.shape[0], -1, dtype=np.int32)
    for k in range(next_ids.shape[0]):
        w = int(next_ids[k])
        if w in cell_lookup:
            pilot_assignments[k] = cell_lookup[w]

    return cells, counts, pilot_assignments


# ----------------------------------------------------------------------
# Per-cell statistics.
# ----------------------------------------------------------------------
def compute_cell_means(
    states: np.ndarray,
    pilot_assignments: np.ndarray,
    n_cells: int,
) -> np.ndarray:
    """Compute per-cell, per-layer mean trajectories.

    Args:
        states: (L, N, H) per-layer hidden states.
        pilot_assignments: (N,) cell index for each pilot, -1 if unassigned.
        n_cells: number of cells.

    Returns:
        cell_means: (n_cells, L, H) per-cell, per-layer mean state.
    """
    L, N, H = states.shape
    cell_means = np.zeros((n_cells, L, H), dtype=np.float64)
    for c in range(n_cells):
        mask = (pilot_assignments == c)
        if not mask.any():
            continue
        # Average over pilots in this cell, for each layer.
        cell_means[c] = states[:, mask, :].mean(axis=1).astype(np.float64)
    return cell_means


def compute_input_marginal_means(
    states: np.ndarray,
    input_ids: np.ndarray,
    unique_inputs: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute the input-marginal mean trajectory mu_v(t) for each v.

    Args:
        states: (L, N, H) per-layer hidden states.
        input_ids: (N,) input token id for each pilot.
        unique_inputs: (n_v,) the input token ids to compute means for.

    Returns:
        Dict mapping input_id -> (L, H) mean trajectory.
    """
    L, N, H = states.shape
    out: Dict[int, np.ndarray] = {}
    for v in unique_inputs:
        mask = (input_ids == int(v))
        if not mask.any():
            continue
        out[int(v)] = states[:, mask, :].mean(axis=1).astype(np.float64)
    return out


def compute_global_marginal(states: np.ndarray) -> np.ndarray:
    """Compute the all-pilots mean trajectory mu(t).

    Used in reverse mode: every cell subtracts the same baseline,
    namely the average over the entire pilot population.

    Args:
        states: (L, N, H) per-layer hidden states.

    Returns:
        mu: (L, H) all-pilots mean per layer.
    """
    return states.mean(axis=1).astype(np.float64)


def compute_deviation_tensor(
    cells: np.ndarray,
    cell_means: np.ndarray,
    input_marginal_means: Dict[int, np.ndarray] | None = None,
    global_marginal: np.ndarray | None = None,
) -> np.ndarray:
    """For each cell, compute deviation from the appropriate marginal.

    Dispatches on which marginal is provided:
      - If `input_marginal_means` is given (pair mode), subtract
        mu_{cells[c,0]}(t) from each cell's mean.
      - If `global_marginal` is given (reverse mode), subtract mu(t)
        from every cell's mean.

    Exactly one of `input_marginal_means` and `global_marginal` must
    be non-None.

    Args:
        cells: (n_cells, 2) cell identities. In pair mode column 0 is
            the input token id; in reverse mode column 0 is the
            sentinel -1.
        cell_means: (n_cells, L, H) per-cell mean trajectories.
        input_marginal_means: (pair mode) dict input_id -> (L, H).
        global_marginal: (reverse mode) (L, H) array.

    Returns:
        deviation: (n_cells, L, H) deviation tensor.
    """
    if (input_marginal_means is None) == (global_marginal is None):
        raise ValueError(
            "Provide exactly one of input_marginal_means (pair mode) "
            "or global_marginal (reverse mode)."
        )

    n_cells, L, H = cell_means.shape
    deviation = np.zeros_like(cell_means)
    if input_marginal_means is not None:
        for c in range(n_cells):
            v = int(cells[c, 0])
            if v in input_marginal_means:
                deviation[c] = cell_means[c] - input_marginal_means[v]
    else:
        for c in range(n_cells):
            deviation[c] = cell_means[c] - global_marginal
    return deviation


# ----------------------------------------------------------------------
# Spectra and effective rank.
# ----------------------------------------------------------------------
def spectrum(matrix: np.ndarray) -> np.ndarray:
    """Singular values of a 2D matrix, descending."""
    return np.linalg.svd(matrix, compute_uv=False)


def effective_rank(singular_values: np.ndarray) -> float:
    """Entropy-based effective rank (exp of Shannon entropy of normalized
    squared singular values). 1 means rank-1; len(sv) means full rank
    with all singular values equal.

    Caveat: effective rank measures spectrum uniformity, not energy
    concentration. A spectrum with a few dominant modes plus a long
    tail can have HIGH effective rank because the tail's entropy
    counts; see energy_concentration() for the complementary
    head-focused statistic.
    """
    s2 = singular_values ** 2
    total = s2.sum()
    if total <= 0:
        return 0.0
    p = s2 / total
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p)))
    return float(np.exp(entropy))


def energy_concentration(singular_values: np.ndarray,
                         ks: np.ndarray | None = None) -> np.ndarray:
    """Cumulative fraction of squared-singular-value energy in top-k modes.

    Returns an array of length len(singular_values) where entry k
    (0-indexed: actually k+1 modes) is the fraction of total
    sum-of-squares carried by the top-(k+1) singular values.

    Args:
        singular_values: descending-sorted singular values.
        ks: if given, return only these indices. None returns the full
            cumulative curve.

    Returns:
        Cumulative energy fraction in [0, 1].
    """
    s2 = singular_values ** 2
    total = s2.sum()
    if total <= 0:
        return np.zeros_like(singular_values)
    csum = np.cumsum(s2) / total
    if ks is None:
        return csum
    return csum[ks]


def dictionary_dimension(
    real_sv: np.ndarray,
    null_sv: np.ndarray,
    enrichment_floor: float = 0.005,
) -> int:
    """Estimate the dimension of the shared-dictionary subspace.

    Defined as the largest k for which the excess concentration
        E_k^{real} - E_k^{null}
    is still rising (the largest k where adding mode k+1 enriches the
    real signal over null by at least `enrichment_floor`). Past this
    k, the remaining modes are at the null floor.

    A pure low-rank signal will give a k near the true rank; a
    completely null signal gives k = 0 (or 1, if both end up equal at
    k=0 too).

    Args:
        real_sv: descending singular values of the real deviation.
        null_sv: descending singular values of the shuffle null
            (mean across shuffles).
        enrichment_floor: minimum marginal enrichment of real over null
            to count mode k as part of the dictionary. Default 0.5%
            of total energy.

    Returns:
        Integer dictionary dimension, in [0, len(real_sv)].
    """
    n = min(len(real_sv), len(null_sv))
    real_cum = energy_concentration(real_sv[:n])
    null_cum = energy_concentration(null_sv[:n])
    excess = real_cum - null_cum
    if excess.max() <= 0:
        return 0
    # Largest k where excess is within `enrichment_floor` of its peak.
    # Equivalently, the last k where adding one more mode gained
    # >= enrichment_floor of excess.
    # We find the peak, then walk forward until the excess has dropped
    # by more than enrichment_floor below the peak.
    peak_k = int(np.argmax(excess))
    # The dictionary extends through the peak. Modes past the peak
    # have decreasing excess and are not part of the dictionary.
    return peak_k + 1  # +1 because index 0 = "top-1 mode"


def deviation_spectra(deviation: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute both flattenings of the deviation tensor and return their
    singular value spectra, effective ranks, and cumulative-energy
    curves.

    Args:
        deviation: (n_cells, L, H) tensor.

    Returns dict with:
        sv_directions: singular values of ((n_cells * L), H), descending.
        sv_trajectories: singular values of (n_cells, (L * H)), descending.
        effective_rank_directions, effective_rank_trajectories: scalars.
        energy_directions: cumulative energy fraction in top-k modes.
        energy_trajectories: cumulative energy fraction in top-k modes.
    """
    n_cells, L, H = deviation.shape
    # Direction view: each (cell, layer) is a row in R^H.
    mat_dir = deviation.reshape(n_cells * L, H)
    sv_dir = spectrum(mat_dir)
    # Trajectory view: each cell is a row in R^{L*H}.
    mat_traj = deviation.reshape(n_cells, L * H)
    sv_traj = spectrum(mat_traj)
    return {
        "sv_directions": sv_dir,
        "sv_trajectories": sv_traj,
        "effective_rank_directions": effective_rank(sv_dir),
        "effective_rank_trajectories": effective_rank(sv_traj),
        "energy_directions": energy_concentration(sv_dir),
        "energy_trajectories": energy_concentration(sv_traj),
    }


# ----------------------------------------------------------------------
# Null baseline via within-input cell-label shuffling.
# ----------------------------------------------------------------------
def shuffle_null(
    states: np.ndarray,
    pilot_assignments: np.ndarray,
    n_cells: int,
    cells: np.ndarray,
    input_ids: np.ndarray,
    input_marginal_means: Dict[int, np.ndarray] | None,
    global_marginal: np.ndarray | None,
    n_shuffles: int,
    rng_seed: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """Run shuffle null for either mode.

    pair mode: within-input shuffle. Pilot-to-cell assignments are
        randomized among pilots with the same input token v. This
        preserves the per-cell "shares v" property and isolates the
        effect of the successor token w. A naive global shuffle would
        conflate "no w structure" with "no v structure."

    reverse mode: global shuffle. The reverse view's marginal
        subtraction is the same for all pilots (the all-pilots mean),
        so there is no input-grouping to preserve.

    Args:
        states: (L, N, H) per-layer hidden states.
        pilot_assignments: (N,) real cell index, -1 if unassigned.
        n_cells: number of cells.
        cells: (n_cells, 2) cell identities; cells[c, 0] is the input
            token (pair mode) or sentinel -1 (reverse mode).
        input_ids: (N,) input token id per pilot. Used in pair mode
            for the within-input grouping; ignored in reverse mode.
        input_marginal_means: pair-mode marginals, or None.
        global_marginal: reverse-mode marginal, or None.
        n_shuffles: number of replicates.
        rng_seed: RNG seed.
        mode: 'pair' or 'reverse'.

    Returns:
        Dict with the null spectra and effective-rank statistics.
    """
    L, N, H = states.shape

    # Precompute per-pilot deviation from the appropriate marginal.
    # In pair mode this is mu_{v_k}(t); in reverse mode this is mu(t).
    pilot_mu = np.zeros((L, N, H), dtype=np.float64)
    if mode == "pair":
        assert input_marginal_means is not None
        for k in range(N):
            v = int(input_ids[k])
            if v in input_marginal_means:
                pilot_mu[:, k, :] = input_marginal_means[v]
    elif mode == "reverse":
        assert global_marginal is not None
        # Broadcast global marginal across all pilots.
        pilot_mu[:] = global_marginal[:, None, :]
    else:
        raise ValueError(f"unknown mode: {mode}")
    pilot_dev = states.astype(np.float64) - pilot_mu

    rng = np.random.default_rng(rng_seed)

    sv_dir_list: List[np.ndarray] = []
    sv_traj_list: List[np.ndarray] = []
    er_dir_list: List[float] = []
    er_traj_list: List[float] = []
    energy_dir_list: List[np.ndarray] = []
    energy_traj_list: List[np.ndarray] = []

    if mode == "pair":
        # Bucket pilots by input token (assigned pilots only).
        pilots_by_input: Dict[int, np.ndarray] = {}
        unique_vs = set(int(cells[c, 0]) for c in range(n_cells))
        for v in unique_vs:
            mask = (input_ids == v) & (pilot_assignments >= 0)
            pilots_by_input[v] = np.where(mask)[0]

        for s in range(n_shuffles):
            # For each input token v, permute the cell labels among
            # the pilots with input v.
            shuffled_assignments = np.full(N, -1, dtype=np.int32)
            for v, pilots_v in pilots_by_input.items():
                if pilots_v.size == 0:
                    continue
                shuffled_labels = rng.permutation(pilot_assignments[pilots_v])
                shuffled_assignments[pilots_v] = shuffled_labels

            shuffled_deviation = np.zeros((n_cells, L, H), dtype=np.float64)
            for c in range(n_cells):
                mask = (shuffled_assignments == c)
                if mask.any():
                    shuffled_deviation[c] = pilot_dev[:, mask, :].mean(axis=1)

            spec = deviation_spectra(shuffled_deviation)
            sv_dir_list.append(spec["sv_directions"])
            sv_traj_list.append(spec["sv_trajectories"])
            er_dir_list.append(spec["effective_rank_directions"])
            er_traj_list.append(spec["effective_rank_trajectories"])
            energy_dir_list.append(spec["energy_directions"])
            energy_traj_list.append(spec["energy_trajectories"])

    else:  # reverse mode
        # Global shuffle: permute the cell labels across all assigned
        # pilots.
        assigned_indices = np.where(pilot_assignments >= 0)[0]
        real_labels = pilot_assignments[assigned_indices]

        for s in range(n_shuffles):
            shuffled_labels = rng.permutation(real_labels)
            shuffled_assignments = np.full(N, -1, dtype=np.int32)
            shuffled_assignments[assigned_indices] = shuffled_labels

            shuffled_deviation = np.zeros((n_cells, L, H), dtype=np.float64)
            for c in range(n_cells):
                mask = (shuffled_assignments == c)
                if mask.any():
                    shuffled_deviation[c] = pilot_dev[:, mask, :].mean(axis=1)

            spec = deviation_spectra(shuffled_deviation)
            sv_dir_list.append(spec["sv_directions"])
            sv_traj_list.append(spec["sv_trajectories"])
            er_dir_list.append(spec["effective_rank_directions"])
            er_traj_list.append(spec["effective_rank_trajectories"])
            energy_dir_list.append(spec["energy_directions"])
            energy_traj_list.append(spec["energy_trajectories"])

    sv_dir_stack = np.stack(sv_dir_list, axis=0)
    sv_traj_stack = np.stack(sv_traj_list, axis=0)
    energy_dir_stack = np.stack(energy_dir_list, axis=0)
    energy_traj_stack = np.stack(energy_traj_list, axis=0)
    return {
        "null_sv_directions_mean": sv_dir_stack.mean(axis=0),
        "null_sv_directions_std": sv_dir_stack.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(sv_dir_stack[0]),
        "null_sv_trajectories_mean": sv_traj_stack.mean(axis=0),
        "null_sv_trajectories_std": sv_traj_stack.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(sv_traj_stack[0]),
        "null_effective_rank_directions_mean": float(np.mean(er_dir_list)),
        "null_effective_rank_directions_std": float(np.std(er_dir_list, ddof=1) if n_shuffles > 1 else 0.0),
        "null_effective_rank_trajectories_mean": float(np.mean(er_traj_list)),
        "null_effective_rank_trajectories_std": float(np.std(er_traj_list, ddof=1) if n_shuffles > 1 else 0.0),
        "null_energy_directions_mean": energy_dir_stack.mean(axis=0),
        "null_energy_directions_std": energy_dir_stack.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(energy_dir_stack[0]),
        "null_energy_trajectories_mean": energy_traj_stack.mean(axis=0),
        "null_energy_trajectories_std": energy_traj_stack.std(axis=0, ddof=1) if n_shuffles > 1 else np.zeros_like(energy_traj_stack[0]),
    }


# ----------------------------------------------------------------------
# Plotting.
# ----------------------------------------------------------------------
def plot_results(result: Dict[str, np.ndarray],
                 out_path: str,
                 seed: int, step: int,
                 mode: str = "pair") -> None:
    """Three-panel plot:
      - top: direction-view spectrum (log scale).
      - middle: trajectory-view spectrum (log scale).
      - bottom: trajectory-view cumulative energy concentration
        (real vs null), with the dictionary-dimension marker.
    """
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(10, 11),
        gridspec_kw={"height_ratios": [1, 1, 1]}
    )

    # --- direction view ---
    sv = result["sv_directions"]
    null_mean = result["null_sv_directions_mean"]
    null_std = result["null_sv_directions_std"]
    er = float(result["effective_rank_directions"])
    null_er = float(result["null_effective_rank_directions_mean"])
    x = np.arange(1, len(sv) + 1)
    ax_top.semilogy(x, sv, "o-", lw=1.5, ms=3, color="C0", label="real")
    ax_top.fill_between(
        x[:len(null_mean)],
        np.maximum(null_mean - 2 * null_std, 1e-30),
        null_mean + 2 * null_std,
        alpha=0.3, color="gray", label=r"shuffle null $\pm 2\sigma$"
    )
    ax_top.plot(x[:len(null_mean)], null_mean, "--", lw=1, color="gray")
    ax_top.set_xlabel("singular value index")
    ax_top.set_ylabel("singular value (log scale)")
    ax_top.set_title(
        f"Direction view spectrum: (n_cells×L) × H matrix\n"
        f"effective rank = {er:.1f} (real) vs {null_er:.1f} (null)"
    )
    ax_top.grid(alpha=0.3, which="both")
    ax_top.legend(fontsize=9)

    # --- trajectory view spectrum ---
    sv = result["sv_trajectories"]
    null_mean = result["null_sv_trajectories_mean"]
    null_std = result["null_sv_trajectories_std"]
    er = float(result["effective_rank_trajectories"])
    null_er = float(result["null_effective_rank_trajectories_mean"])
    x = np.arange(1, len(sv) + 1)
    ax_mid.semilogy(x, sv, "o-", lw=1.5, ms=4, color="C2", label="real")
    ax_mid.fill_between(
        x[:len(null_mean)],
        np.maximum(null_mean - 2 * null_std, 1e-30),
        null_mean + 2 * null_std,
        alpha=0.3, color="gray", label=r"shuffle null $\pm 2\sigma$"
    )
    ax_mid.plot(x[:len(null_mean)], null_mean, "--", lw=1, color="gray")
    ax_mid.set_xlabel("singular value index")
    ax_mid.set_ylabel("singular value (log scale)")
    ax_mid.set_title(
        f"Trajectory view spectrum: n_cells × (L·H) matrix\n"
        f"effective rank = {er:.1f} (real) vs {null_er:.1f} (null)"
    )
    ax_mid.grid(alpha=0.3, which="both")
    ax_mid.legend(fontsize=9)

    # --- energy concentration curve (trajectory view) ---
    energy = result["energy_trajectories"]
    energy_null = result["null_energy_trajectories_mean"]
    energy_null_std = result["null_energy_trajectories_std"]
    x = np.arange(1, len(energy) + 1)
    ax_bot.plot(x, 100 * energy, "D-", lw=2, ms=5,
                color="C3", label="real")
    ax_bot.fill_between(
        x[:len(energy_null)],
        100 * (energy_null - 2 * energy_null_std),
        100 * (energy_null + 2 * energy_null_std),
        alpha=0.3, color="gray", label=r"shuffle null $\pm 2\sigma$"
    )
    ax_bot.plot(x[:len(energy_null)], 100 * energy_null, "--",
                lw=1, color="gray")
    # Mark the dictionary dimension.
    dict_dim = dictionary_dimension(result["sv_trajectories"],
                                    result["null_sv_trajectories_mean"])
    if dict_dim > 0:
        ax_bot.axvline(dict_dim, color="C3", ls=":", lw=1.5, alpha=0.7)
        ax_bot.text(
            dict_dim + 0.3,
            5,
            f"dictionary\n dim = {dict_dim}",
            fontsize=9, color="C3", alpha=0.9
        )
    ax_bot.set_xlabel("top-k modes")
    ax_bot.set_ylabel("cumulative energy (%)")
    ax_bot.set_title(
        f"Trajectory-view cumulative energy concentration\n"
        f"(peak excess over null at k = {dict_dim})"
    )
    ax_bot.set_ylim(0, 105)
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc="lower right", fontsize=9)

    n_cells = int(result["n_cells"])
    fig.suptitle(
        f"Conditional-deviation structure ({mode} mode): "
        f"seed {seed}, step {step}, {n_cells} cells",
        y=1.00, fontsize=12
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------
def augmented_path(run_dir: str, seed: int, step: int) -> str:
    return os.path.join(run_dir, "multiview", f"seed_{seed}",
                        f"augmented_step_{step:08d}.npz")


def run_test(run_dir: str, seed: int, step: int,
             mode: str,
             min_pilots_per_cell: int, top_k_cells: int,
             top_k_tokens_v: int, n_shuffles: int,
             out_dir: str,
             quiet: bool = False,
             save_plot: bool = True,
             save_npz: bool = True) -> Dict:
    """Full test driver.

    Args:
        mode: 'pair' (input-conditioned pair view) or 'reverse'
            (successor-only view).
        top_k_tokens_v: only used in pair mode; ignored in reverse mode.
        quiet: if True, suppress per-checkpoint console output (used
            by training-trajectory driver to avoid spamming).
        save_plot: if True, write the 3-panel summary PNG.
        save_npz: if True, write the per-checkpoint npz.
    """
    def _say(*args, **kwargs):
        if not quiet:
            _say(*args, **kwargs)

    if mode not in ("pair", "reverse"):
        raise ValueError(f"mode must be 'pair' or 'reverse', got {mode!r}")

    path = augmented_path(run_dir, seed, step)
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing {path}")
    _say(f"Loading {path} ...")
    payload = _load_augmented_activations(path)
    states = payload["states"]
    input_ids = payload["input_ids"]
    next_ids = payload["next_ids"]
    L, N, H = states.shape
    _say(f"  states: L={L}, N={N}, H={H}")
    _say(f"  mode: {mode}")
    _say(f"  vocab observed in input_ids: {len(np.unique(input_ids))}")
    _say(f"  vocab observed in next_ids: {len(np.unique(next_ids))}")

    # Select cells (mode-dependent).
    if mode == "pair":
        cells, counts, pilot_assignments = select_pair_cells(
            input_ids, next_ids,
            min_pilots_per_cell=min_pilots_per_cell,
            top_k_cells=top_k_cells,
            top_k_tokens_v=top_k_tokens_v,
        )
    else:  # reverse
        cells, counts, pilot_assignments = select_reverse_cells(
            next_ids,
            min_pilots_per_cell=min_pilots_per_cell,
            top_k_cells=top_k_cells,
        )

    n_cells = cells.shape[0]
    n_assigned = int((pilot_assignments >= 0).sum())
    _say(f"  selected {n_cells} cells")
    if n_cells == 0:
        _say(f"  no cells met the min_pilots_per_cell={min_pilots_per_cell} "
              f"threshold; consider lowering it.")
        return {}
    _say(f"  pilot count per cell: min={counts.min()}, "
          f"median={int(np.median(counts))}, max={counts.max()}")
    _say(f"  total pilots in selected cells: {n_assigned} "
          f"({100 * n_assigned / N:.1f}% of N)")

    # Marginal computation and deviation tensor (mode-dependent).
    input_marginal_means: Dict[int, np.ndarray] | None = None
    global_marginal: np.ndarray | None = None
    if mode == "pair":
        unique_inputs = np.unique(cells[:, 0])
        _say(f"  computing input-marginal means for {len(unique_inputs)} "
              f"distinct input tokens ...")
        input_marginal_means = compute_input_marginal_means(
            states, input_ids, unique_inputs
        )
    else:
        _say(f"  computing global marginal mean ...")
        global_marginal = compute_global_marginal(states)

    _say(f"  computing per-cell means ...")
    cell_means = compute_cell_means(states, pilot_assignments, n_cells)
    _say(f"  computing deviation tensor ...")
    deviation = compute_deviation_tensor(
        cells, cell_means,
        input_marginal_means=input_marginal_means,
        global_marginal=global_marginal,
    )

    # Deviation magnitude — sanity check.
    dev_norm_per_cell_layer = np.linalg.norm(deviation, axis=2)
    _say(f"  deviation magnitude (||d(t)||_2): "
          f"mean = {dev_norm_per_cell_layer.mean():.3f}, "
          f"max = {dev_norm_per_cell_layer.max():.3f}")

    # Spectra.
    _say(f"  computing SVD spectra ...")
    spectra = deviation_spectra(deviation)
    _say(f"  effective rank (directions): "
          f"{spectra['effective_rank_directions']:.2f} "
          f"out of {min(n_cells * L, H)} possible")
    _say(f"  effective rank (trajectories): "
          f"{spectra['effective_rank_trajectories']:.2f} "
          f"out of {min(n_cells, L * H)} possible")

    # Null baseline.
    _say(f"  computing shuffle null with {n_shuffles} replicates "
          f"(mode-dependent: {'within-input' if mode == 'pair' else 'global'}) ...")
    null = shuffle_null(
        states, pilot_assignments, n_cells, cells, input_ids,
        input_marginal_means, global_marginal,
        n_shuffles=n_shuffles, rng_seed=10000 * seed + step,
        mode=mode,
    )
    _say(f"  null effective rank (directions): "
          f"{null['null_effective_rank_directions_mean']:.2f} "
          f"+/- {null['null_effective_rank_directions_std']:.2f}")
    _say(f"  null effective rank (trajectories): "
          f"{null['null_effective_rank_trajectories_mean']:.2f} "
          f"+/- {null['null_effective_rank_trajectories_std']:.2f}")

    # Verdict.
    #
    # The primary statistic is *energy concentration* in the leading k
    # singular vectors of the trajectory-view spectrum, compared to
    # the within-input (pair) or global (reverse) shuffle null. This
    # is more informative than effective rank because the spectrum
    # can be concentrated at the head (small shared dictionary) while
    # having a long tail of cell-specific structure that inflates the
    # effective-rank statistic. Energy concentration measures the
    # leading-mode dominance directly.
    #
    # The dictionary dimension is the smallest k at which the excess
    # concentration (real - null) peaks. Past that k, additional modes
    # are no longer enriched over null; the leading k modes constitute
    # the shared dictionary, the rest is cell-specific.
    er_dir = spectra["effective_rank_directions"]
    er_dir_null = null["null_effective_rank_directions_mean"]
    er_dir_null_std = null["null_effective_rank_directions_std"]
    er_traj = spectra["effective_rank_trajectories"]
    er_traj_null = null["null_effective_rank_trajectories_mean"]
    er_traj_null_std = null["null_effective_rank_trajectories_std"]

    energy_traj = spectra["energy_trajectories"]
    energy_traj_null = null["null_energy_trajectories_mean"]
    energy_traj_null_std = null["null_energy_trajectories_std"]
    excess_traj = energy_traj - energy_traj_null

    dict_dim_traj = dictionary_dimension(
        spectra["sv_trajectories"],
        null["null_sv_trajectories_mean"],
    )

    max_dir = min(n_cells * L, H)
    max_traj = min(n_cells, L * H)
    _say()
    _say("=" * 60)
    _say(f"Summary ({mode} mode):")
    _say(f"  Direction view effective rank:   "
          f"real = {er_dir:.2f} of {max_dir}; "
          f"null = {er_dir_null:.2f} (+/- {er_dir_null_std:.2f})")
    _say(f"  Trajectory view effective rank:  "
          f"real = {er_traj:.2f} of {max_traj}; "
          f"null = {er_traj_null:.2f} (+/- {er_traj_null_std:.2f})")
    _say()
    _say(f"  Trajectory-view energy concentration:")
    _say(f"    {'k':>3}  {'real':>7}  {'null':>7}  {'excess':>7}  {'z':>5}")
    cap = min(n_cells, 20)
    for k_idx in [0, 1, 2, 4, 7, 9, 14, 19, cap - 1]:
        if k_idx >= len(energy_traj):
            continue
        e_r = energy_traj[k_idx]
        e_n = energy_traj_null[k_idx]
        e_s = energy_traj_null_std[k_idx]
        z = (e_r - e_n) / max(e_s, 1e-9)
        _say(f"    {k_idx + 1:>3}  {100*e_r:>6.1f}%  {100*e_n:>6.1f}%  "
              f"{100*(e_r-e_n):>+6.1f}%  {z:>5.1f}")
    _say()
    _say(f"  Dictionary dimension (trajectory view): {dict_dim_traj}")
    _say(f"    (peak excess concentration: "
          f"{100 * excess_traj[dict_dim_traj - 1]:.1f}% over null "
          f"at k = {dict_dim_traj})")
    _say("=" * 60)

    # Verdict logic uses excess concentration at the dictionary
    # dimension as the primary signal. We translate "peak excess
    # over null" into a structural interpretation.
    peak_excess = excess_traj[dict_dim_traj - 1] if dict_dim_traj > 0 else 0.0
    peak_z = (
        peak_excess / max(energy_traj_null_std[dict_dim_traj - 1], 1e-9)
        if dict_dim_traj > 0 else 0.0
    )

    if peak_excess > 0.10 and peak_z > 5 and dict_dim_traj <= 0.3 * n_cells:
        verdict = (
            f"H_dictionary PASS: a {dict_dim_traj}-dimensional shared "
            f"dictionary explains {100*peak_excess:.0f}% more total "
            f"deviation energy than the null ({peak_z:.0f}σ). The "
            f"leading {dict_dim_traj} modes carry "
            f"{100*energy_traj[dict_dim_traj - 1]:.0f}% of total "
            f"deviation energy in real vs "
            f"{100*energy_traj_null[dict_dim_traj - 1]:.0f}% in null. "
            "The learned function uses a small dictionary of shared "
            "trajectory modes with cell-specific tails."
        )
    elif peak_excess > 0.03 and peak_z > 3:
        verdict = (
            f"H_partial_dictionary: a {dict_dim_traj}-dimensional "
            f"shared component explains {100*peak_excess:.0f}% more "
            f"total deviation energy than the null ({peak_z:.0f}σ). "
            "Some shared structure across cells, but the bulk of the "
            "deviation is in the cell-specific tail."
        )
    elif peak_excess > 0.005:
        verdict = (
            f"H_weak_dictionary: small excess concentration "
            f"({100*peak_excess:.1f}% at k = {dict_dim_traj}, "
            f"{peak_z:.1f}σ). Trace of shared structure but the "
            "signal is marginal."
        )
    else:
        baseline = ("the input token v already determines"
                    if mode == "pair" else
                    "the overall marginal already determines")
        conditioning = "successor w"
        verdict = (
            f"H_no_signal: no excess concentration of deviation energy "
            f"in the leading modes. The {conditioning} adds no "
            f"detectable structure beyond what {baseline}."
        )
    _say(verdict)
    _say()

    # Save.
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{mode}_seed_{seed}_step_{step:08d}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    # Config encodes mode as 1 for pair, 2 for reverse for backward-
    # compatible numeric storage.
    mode_code = 1 if mode == "pair" else 2
    result_dict = {
        "cells": cells,
        "cell_counts": counts,
        "deviation": deviation.astype(np.float32),
        "n_cells": np.int32(n_cells),
        "mode": mode,
        "verdict": verdict,
        "dictionary_dimension": int(dict_dim_traj),
        "peak_excess": float(peak_excess),
        "peak_z": float(peak_z),
        **spectra,
        **null,
        "config": np.array(
            [seed, step, min_pilots_per_cell, top_k_cells,
             top_k_tokens_v, n_shuffles, L, N, H, n_cells, mode_code],
            dtype=np.int64,
        ),
    }
    if save_npz:
        np.savez(npz_path, **result_dict)
        _say(f"Saved {npz_path}")

    # Plot.
    if save_plot:
        png_path = os.path.join(out_dir, f"{stem}.png")
        plot_results(result_dict, png_path, seed, step, mode=mode)
        _say(f"Saved {png_path}")

    return result_dict


# ----------------------------------------------------------------------
# Training-trajectory driver.
# ----------------------------------------------------------------------
def checkpoints_in_seed(run_dir: str, seed: int) -> List[int]:
    """List augmented_step_*.npz checkpoints available for a seed.

    Mirrors the helper of the same name in drift_welldef_training.py so
    that this script picks the same checkpoint set when both are run
    with the same log-spaced sub-sampling.
    """
    seed_dir = os.path.join(run_dir, "multiview", f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return []
    steps: List[int] = []
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
    minimize log-distance to a uniformly log-spaced ideal. Mirrors
    drift_welldef_training.log_spaced_subsample.
    """
    if len(steps) <= n_target:
        return list(steps)
    steps_arr = np.array(steps, dtype=np.float64)
    log_steps = np.log10(np.maximum(steps_arr, 1.0))
    target_log = np.linspace(log_steps[0], log_steps[-1], n_target)
    picked = set()
    for t in target_log:
        i = int(np.argmin(np.abs(log_steps - t)))
        picked.add(steps[i])
    return sorted(picked)


def run_training_trajectory(
    run_dir: str,
    seeds: List[int],
    steps: List[int],
    mode: str,
    min_pilots_per_cell: int,
    top_k_cells: int,
    top_k_tokens_v: int,
    n_shuffles: int,
    out_dir: str,
) -> Dict:
    """Run the pair-deviation test across a (seeds x checkpoints) grid.

    For every (seed, checkpoint) pair this calls run_test and collects
    the summary statistics. Per-checkpoint npz / png outputs are
    suppressed to keep the disk footprint small; only the aggregated
    trajectory npz and plot are written.

    Returns:
        Dict with aggregated arrays. Shape conventions:
            (n_seeds, n_steps) for per-checkpoint scalars.
            (n_seeds, n_steps, n_sv_max) for trajectory-view energy
            curves; rows are right-padded with NaN if cell counts
            differ across checkpoints.
    """
    n_seeds = len(seeds)
    n_steps = len(steps)

    dict_dim = np.full((n_seeds, n_steps), np.nan)
    peak_excess = np.full((n_seeds, n_steps), np.nan)
    peak_z = np.full((n_seeds, n_steps), np.nan)
    er_traj = np.full((n_seeds, n_steps), np.nan)
    er_traj_null = np.full((n_seeds, n_steps), np.nan)
    n_cells_arr = np.zeros((n_seeds, n_steps), dtype=np.int32)
    verdicts: List[List[str]] = [["" for _ in range(n_steps)]
                                 for _ in range(n_seeds)]
    energy_real_list: List[List[np.ndarray]] = [[] for _ in range(n_seeds)]
    energy_null_list: List[List[np.ndarray]] = [[] for _ in range(n_seeds)]

    print("Running pair-deviation training trajectory:")
    print(f"  mode = {mode}")
    print(f"  seeds = {seeds}")
    print(f"  {len(steps)} checkpoints: {steps[0]} ... {steps[-1]}")
    print()

    for si, seed in enumerate(seeds):
        for ki, step in enumerate(steps):
            print(f"  [seed {seed}, step {step:>6}] ", end="", flush=True)
            try:
                result = run_test(
                    run_dir, seed, step, mode,
                    min_pilots_per_cell=min_pilots_per_cell,
                    top_k_cells=top_k_cells,
                    top_k_tokens_v=top_k_tokens_v,
                    n_shuffles=n_shuffles,
                    out_dir=out_dir,
                    quiet=True,
                    save_plot=False,
                    save_npz=False,
                )
            except FileNotFoundError as e:
                print(f"missing: {e}")
                continue
            if not result:
                print("no cells passed threshold")
                continue
            dict_dim[si, ki] = result["dictionary_dimension"]
            peak_excess[si, ki] = result["peak_excess"]
            peak_z[si, ki] = result["peak_z"]
            er_traj[si, ki] = result["effective_rank_trajectories"]
            er_traj_null[si, ki] = result["null_effective_rank_trajectories_mean"]
            n_cells_arr[si, ki] = int(result["n_cells"])
            verdicts[si][ki] = str(result["verdict"]).split(":", 1)[0]
            energy_real_list[si].append(result["energy_trajectories"])
            energy_null_list[si].append(result["null_energy_trajectories_mean"])
            print(
                f"n_cells={int(result['n_cells']):>3}, "
                f"dict_dim={int(result['dictionary_dimension']):>2}, "
                f"peak_excess={100*result['peak_excess']:>+5.1f}%, "
                f"z={result['peak_z']:>5.1f}, "
                f"verdict={verdicts[si][ki]}"
            )

    # Pad energy curves to a common width.
    max_n_cells = int(n_cells_arr.max()) if n_cells_arr.size else 0
    energy_real = np.full((n_seeds, n_steps, max_n_cells), np.nan)
    energy_null = np.full((n_seeds, n_steps, max_n_cells), np.nan)
    for si in range(n_seeds):
        for ki in range(n_steps):
            if ki < len(energy_real_list[si]):
                er = energy_real_list[si][ki]
                en = energy_null_list[si][ki]
                energy_real[si, ki, :len(er)] = er
                energy_null[si, ki, :len(en)] = en

    # Cross-seed summary table.
    dict_dim_mean = np.nanmean(dict_dim, axis=0)
    dict_dim_std = np.nanstd(dict_dim, axis=0, ddof=1)
    peak_excess_mean = np.nanmean(peak_excess, axis=0)
    peak_excess_std = np.nanstd(peak_excess, axis=0, ddof=1)
    er_traj_mean = np.nanmean(er_traj, axis=0)
    er_traj_null_mean = np.nanmean(er_traj_null, axis=0)

    print()
    print("=" * 78)
    print(f"Trajectory summary (mode = {mode})")
    print(f"{'step':>7}  {'dict_dim (mean±std)':>22}  "
          f"{'peak_excess (mean±std)':>26}  {'ER gap':>8}")
    for ki, step in enumerate(steps):
        if np.isnan(dict_dim_mean[ki]):
            continue
        er_gap = er_traj_mean[ki] - er_traj_null_mean[ki]
        print(
            f"{step:>7}  {dict_dim_mean[ki]:>10.1f} ± {dict_dim_std[ki]:<6.2f}  "
            f"{100*peak_excess_mean[ki]:>+10.1f}% ± {100*peak_excess_std[ki]:<6.2f}%  "
            f"{er_gap:>+8.3f}"
        )
    print("=" * 78)
    print()

    # Save aggregated npz.
    os.makedirs(out_dir, exist_ok=True)
    seeds_str = "_".join(str(s) for s in seeds)
    stem = f"{mode}_trajectory_seeds_{seeds_str}"
    npz_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez(
        npz_path,
        seeds=np.array(seeds, dtype=np.int32),
        steps=np.array(steps, dtype=np.int64),
        dictionary_dimension=dict_dim,
        peak_excess=peak_excess,
        peak_z=peak_z,
        effective_rank_trajectories=er_traj,
        null_effective_rank_trajectories=er_traj_null,
        n_cells=n_cells_arr,
        energy_trajectories_real=energy_real.astype(np.float32),
        energy_trajectories_null=energy_null.astype(np.float32),
        mode=mode,
        config=np.array(
            [min_pilots_per_cell, top_k_cells, top_k_tokens_v, n_shuffles],
            dtype=np.int64,
        ),
    )
    print(f"Saved {npz_path}")

    # Plot.
    png_path = os.path.join(out_dir, f"{stem}.png")
    plot_training_trajectory(
        steps, dict_dim, peak_excess, er_traj, er_traj_null,
        seeds, mode, png_path
    )
    print(f"Saved {png_path}")

    return {
        "seeds": seeds,
        "steps": steps,
        "dictionary_dimension": dict_dim,
        "peak_excess": peak_excess,
        "peak_z": peak_z,
        "effective_rank_trajectories": er_traj,
        "null_effective_rank_trajectories": er_traj_null,
        "energy_trajectories_real": energy_real,
        "energy_trajectories_null": energy_null,
        "mode": mode,
    }


def plot_training_trajectory(
    steps: List[int],
    dict_dim: np.ndarray,
    peak_excess: np.ndarray,
    er_traj: np.ndarray,
    er_traj_null: np.ndarray,
    seeds: List[int],
    mode: str,
    out_path: str,
) -> None:
    """Three-panel plot for the training-trajectory aggregation:
      - Dictionary dimension vs step (per seed + cross-seed mean).
      - Peak excess concentration vs step (per seed + cross-seed mean).
      - Trajectory-view effective rank, real and null, vs step.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    steps_arr = np.array(steps, dtype=np.float64)

    # --- panel 1: dictionary dimension ---
    for si, seed in enumerate(seeds):
        ax1.plot(steps_arr, dict_dim[si], "o-", lw=1.0, ms=4,
                 alpha=0.5, label=f"seed {seed}")
    mean_dd = np.nanmean(dict_dim, axis=0)
    ax1.plot(steps_arr, mean_dd, "D-", lw=2.5, color="black",
             ms=6, label="mean")
    ax1.set_xscale("log")
    ax1.set_ylabel("dictionary dimension")
    ax1.set_title(f"Pair-deviation training trajectory ({mode} mode)\n"
                  "Dictionary dimension")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="best", ncol=2)

    # --- panel 2: peak excess concentration ---
    for si, seed in enumerate(seeds):
        ax2.plot(steps_arr, 100 * peak_excess[si], "o-", lw=1.0, ms=4,
                 alpha=0.5, label=f"seed {seed}")
    mean_pe = np.nanmean(peak_excess, axis=0)
    ax2.plot(steps_arr, 100 * mean_pe, "D-", lw=2.5, color="black",
             ms=6, label="mean")
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.set_xscale("log")
    ax2.set_ylabel("peak excess concentration (%)")
    ax2.set_title("Peak excess of cumulative energy at dictionary dimension")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="best", ncol=2)

    # --- panel 3: effective rank (real vs null) ---
    er_mean = np.nanmean(er_traj, axis=0)
    er_null_mean = np.nanmean(er_traj_null, axis=0)
    ax3.plot(steps_arr, er_mean, "o-", lw=2, ms=5,
             color="C2", label="real (mean over seeds)")
    ax3.plot(steps_arr, er_null_mean, "s--", lw=2, ms=5,
             color="gray", label="null (mean over seeds)")
    ax3.set_xscale("log")
    ax3.set_xlabel("training step (log)")
    ax3.set_ylabel("trajectory-view effective rank")
    ax3.set_title("Trajectory-view effective rank: real vs null")
    ax3.grid(alpha=0.3, which="both")
    ax3.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", choices=["pair", "reverse"], default="pair",
                    help="'pair' for input-conditioned pair view; "
                         "'reverse' for successor-only view (sample-efficient).")
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=24000)
    ap.add_argument("--min-pilots-per-cell", type=int, default=30,
                    help="Minimum pilots required for a cell to be kept.")
    ap.add_argument("--top-k-cells", type=int, default=100,
                    help="Maximum number of cells to retain.")
    ap.add_argument("--top-k-tokens-v", type=int, default=100,
                    help="(pair mode only) restrict to cells whose input "
                         "token is in this many most-frequent input tokens; "
                         "0 disables.")
    ap.add_argument("--n-shuffles", type=int, default=20,
                    help="Number of shuffle-null replicates.")
    ap.add_argument("--out-dir", default=None)
    # Training-trajectory options.
    ap.add_argument("--training", action="store_true",
                    help="Run the training-trajectory driver: iterate "
                         "over a log-spaced subsample of checkpoints. "
                         "Uses --seeds and --n-checkpoints instead of "
                         "--seed and --step.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="(--training only) Seeds to include; default "
                         "auto-detects 0,1,2,3 from disk.")
    ap.add_argument("--n-checkpoints", type=int, default=12,
                    help="(--training only) Number of log-spaced "
                         "checkpoints to sub-sample.")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "pair_deviation")

    if args.training:
        # Discover seeds and checkpoints.
        seeds = args.seeds
        if seeds is None:
            seeds = []
            for s in range(8):  # search 0..7 just in case
                if os.path.isdir(os.path.join(args.run_dir, "multiview",
                                              f"seed_{s}")):
                    seeds.append(s)
            if not seeds:
                raise RuntimeError(
                    f"could not find any seed_* directories under "
                    f"{os.path.join(args.run_dir, 'multiview')}"
                )
        reference_seed = seeds[0]
        all_steps = checkpoints_in_seed(args.run_dir, reference_seed)
        if not all_steps:
            raise RuntimeError(
                f"no augmented_step_*.npz files under "
                f"{os.path.join(args.run_dir, 'multiview', f'seed_{reference_seed}')}"
            )
        steps = log_spaced_subsample(all_steps, args.n_checkpoints)
        print(f"Using {len(steps)} of {len(all_steps)} available checkpoints "
              f"from seed {reference_seed}.")
        run_training_trajectory(
            args.run_dir, seeds, steps, args.mode,
            args.min_pilots_per_cell, args.top_k_cells,
            args.top_k_tokens_v, args.n_shuffles, out_dir,
        )
    else:
        run_test(args.run_dir, args.seed, args.step,
                 args.mode,
                 args.min_pilots_per_cell, args.top_k_cells,
                 args.top_k_tokens_v, args.n_shuffles, out_dir)


if __name__ == "__main__":
    main()
    