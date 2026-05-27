"""
Helpers for loading a complete training run's worth of flow data.

A 'flow series' is the time-series of recovered linear flows from all
checkpoints of one training run. Most analysis and plotting operates on
this object rather than on individual flow files.

Public:
  - FlowSeries: lightweight container for the stacked time-series.
  - load_flow_series(run_dir): build a FlowSeries from a run directory.
"""

import os
import glob
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np

from analyze import load_flow


@dataclass
class FlowSeries:
    """
    A training run's complete time-series of recovered linear flows.

    Attributes are stacked across checkpoints (axis 0 = checkpoint index)
    so most plotting operations become array indexing rather than dict
    lookups.

    Fields:
        steps: (K,) int array of training step indices, sorted ascending.
        eval_losses: (K,) array of held-out eval losses at each checkpoint.
        train_losses: (K,) array of training losses at each checkpoint.
        lambda_values: (K,) variance scaling rate λ per checkpoint.
        log_alpha_values: (K,) log-prefactor per checkpoint.
        singular_values: (K, L, H) per-checkpoint per-layer spectra.
        effective_ranks: (K, L) per-checkpoint per-layer effective rank.
        kurtosis: (K, L) per-checkpoint per-layer kurtosis.
        isotropy: (K, L) per-checkpoint per-layer isotropy.
        R: (K, L, H, H) per-checkpoint per-layer R matrices.
            Big — typically several GB. Loaded lazily (set to None on construction;
            populated by load_R() when needed).
        pairwise_residual_variance: (K, L, L) per-checkpoint pairwise residuals.
        means: (K, L, H) per-checkpoint per-layer activation means.
        L: number of layer states (= num_layers + 2; includes input and final-norm).
        H: hidden dim.
        K: number of checkpoints.
        seed: which seed produced this run (from the first checkpoint's metadata).
        flow_paths: list of K paths to the underlying .npz files (for lazy loading R).
    """
    steps: np.ndarray
    eval_losses: np.ndarray
    train_losses: np.ndarray
    lambda_values: np.ndarray
    log_alpha_values: np.ndarray
    singular_values: np.ndarray
    effective_ranks: np.ndarray
    kurtosis: np.ndarray
    isotropy: np.ndarray
    pairwise_residual_variance: np.ndarray
    means: np.ndarray
    L: int
    H: int
    K: int
    seed: int
    flow_paths: List[str]
    R: Optional[np.ndarray] = None  # lazy-loaded

    def load_R(self):
        """Populate the R field by loading all R matrices. Expensive — only
        needed for alignment-based analyses."""
        if self.R is not None:
            return
        # Allocate the big array.
        R_stacked = np.zeros(
            (self.K, self.L, self.H, self.H), dtype=np.float32,
        )
        for k, path in enumerate(self.flow_paths):
            flow = load_flow(path)
            R_stacked[k] = flow["R"]
        self.R = R_stacked

    def free_R(self):
        """Drop the loaded R field to free memory."""
        self.R = None


def load_flow_series(run_dir: str) -> FlowSeries:
    """
    Load all flow .npz files from `run_dir/flow_analysis/` and stack
    them into a FlowSeries.

    Args:
        run_dir: A run directory that has had analyze_run() applied
            (i.e., contains a `flow_analysis/` subdirectory).

    Returns:
        FlowSeries with everything except R populated (use .load_R() to
        get the R matrices on demand).

    Raises:
        FileNotFoundError if no flow files are present.
    """
    flow_dir = os.path.join(run_dir, "flow_analysis")
    paths = sorted(glob.glob(os.path.join(flow_dir, "flow_step_*.npz")))
    if not paths:
        raise FileNotFoundError(
            f"No flow .npz files found in {flow_dir}. "
            f"Run analyze_run() first."
        )

    # Load each flow, extract the small scalar/vector fields, leave R alone.
    K = len(paths)
    # Inspect the first to get dimensions.
    first = load_flow(paths[0])
    L = int(first["num_layers_total"])
    H = int(first["hidden_dim"])

    steps = np.zeros(K, dtype=np.int64)
    eval_losses = np.full(K, np.nan, dtype=np.float32)
    train_losses = np.full(K, np.nan, dtype=np.float32)
    lambda_values = np.full(K, np.nan, dtype=np.float32)
    log_alpha_values = np.full(K, np.nan, dtype=np.float32)
    singular_values = np.zeros((K, L, H), dtype=np.float32)
    effective_ranks = np.zeros((K, L), dtype=np.float32)
    kurtosis = np.full((K, L), np.nan, dtype=np.float32)
    isotropy = np.full((K, L), np.nan, dtype=np.float32)
    pairwise_var = np.full((K, L, L), np.nan, dtype=np.float32)
    means = np.zeros((K, L, H), dtype=np.float32)
    seed = -1

    for k, path in enumerate(paths):
        flow = load_flow(path)
        steps[k] = int(flow["checkpoint_step"])
        eval_losses[k] = float(flow["checkpoint_eval_loss"])
        train_losses[k] = float(flow["checkpoint_loss"])
        lambda_values[k] = float(flow["lambda"])
        log_alpha_values[k] = float(flow["log_alpha"])
        singular_values[k] = flow["singular_values"]
        effective_ranks[k] = flow["effective_rank"]
        kurtosis[k] = flow["kurtosis_per_layer"]
        isotropy[k] = flow["isotropy_per_layer"]
        pairwise_var[k] = flow["pairwise_residual_variance"]
        means[k] = flow["means"]
        if k == 0:
            seed = int(flow["checkpoint_seed"])

    # Sort by step (should already be sorted, but make sure).
    order = np.argsort(steps)
    steps = steps[order]
    eval_losses = eval_losses[order]
    train_losses = train_losses[order]
    lambda_values = lambda_values[order]
    log_alpha_values = log_alpha_values[order]
    singular_values = singular_values[order]
    effective_ranks = effective_ranks[order]
    kurtosis = kurtosis[order]
    isotropy = isotropy[order]
    pairwise_var = pairwise_var[order]
    means = means[order]
    paths_sorted = [paths[i] for i in order]

    return FlowSeries(
        steps=steps,
        eval_losses=eval_losses,
        train_losses=train_losses,
        lambda_values=lambda_values,
        log_alpha_values=log_alpha_values,
        singular_values=singular_values,
        effective_ranks=effective_ranks,
        kurtosis=kurtosis,
        isotropy=isotropy,
        pairwise_residual_variance=pairwise_var,
        means=means,
        L=L,
        H=H,
        K=K,
        seed=seed,
        flow_paths=paths_sorted,
    )
