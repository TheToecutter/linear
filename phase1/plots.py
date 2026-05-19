"""
Plotting functions for Phase 1 analysis.

Each function takes a FlowSeries (or two of them) plus an output path,
and writes a PNG. Functions are independent — they don't share state and
can be called in any order. The `report.py` driver invokes them all to
produce a complete plot set for a run.

Design notes:
  - Matplotlib is the only plotting dependency. No seaborn, no plotly.
    This keeps install footprint small and behavior reproducible.
  - All functions accept `figsize`, `dpi`, and `title_suffix` for
    customization. Defaults are tuned for screen viewing.
  - X-axes are training steps unless otherwise noted. Log scales are used
    where appropriate (early-training dynamics span orders of magnitude).
  - For per-layer plots colored by training step, we use a perceptually-
    uniform colormap (`viridis` for sequential, `coolwarm` for diverging).
  - All figures are tight_layout'd and saved with a reasonable DPI.

The basis-invariant plots reflect the proposal's Section 5.3 prioritization:
the headline convergence/universality findings are at the basis-invariant
level, so those plots are the most prominent.
"""

import os
from typing import Optional, List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from flow_series import FlowSeries
from distances import (
    singular_value_distance, lambda_distance, effective_rank_distance,
    successive_layer_angles,
)


# ----------------------------------------------------------------------
# Style helpers.
# ----------------------------------------------------------------------
def _setup_style():
    """Apply consistent matplotlib styling. Called once at module import."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "legend.frameon": False,
    })


_setup_style()


# ----------------------------------------------------------------------
# Figure 1: Loss curves.
# ----------------------------------------------------------------------
def plot_loss_curves(
    flow_series: FlowSeries, output_path: str,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (8, 4),
):
    """
    Training and eval loss vs training step.

    The eval loss is what's actually measured at checkpoint times (the
    train loss field is the instantaneous batch loss at the same step,
    which is noisy). Both are shown.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Train loss (raw, noisy).
    ax.plot(
        flow_series.steps, flow_series.train_losses,
        color="tab:orange", alpha=0.5, lw=1.0,
        label="train loss (single batch)",
    )
    # Eval loss (smooth, measured on full held-out set).
    ax.plot(
        flow_series.steps, flow_series.eval_losses,
        color="tab:blue", lw=2.0, marker="o", markersize=3,
        label="held-out eval loss",
    )

    ax.set_xscale("log")
    ax.set_xlabel("training step (log scale)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(f"Loss curves{title_suffix}")
    ax.legend(loc="upper right")

    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: Basis-invariant statistic trajectories.
# ----------------------------------------------------------------------
def plot_basis_invariant_trajectories(
    flow_series: FlowSeries, output_path: str,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (10, 8),
):
    """
    Five basis-invariant scalars vs training step:
      (a) λ (variance scaling rate)
      (b) log α (variance prefactor)
      (c) mean effective rank across layers
      (d) mean kurtosis across layers
      (e) mean isotropy across layers
      (f) eval loss (for cross-reference)

    These are the substantive convergence diagnostics from the proposal's
    Section 5.3. Trajectories that flatten with training step indicate
    convergence; trajectories still drifting indicate the flow hasn't
    settled.
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True)
    fs = flow_series

    # (a) λ.
    ax = axes[0, 0]
    ax.plot(fs.steps, fs.lambda_values, marker="o", markersize=3, lw=1.5,
            color="tab:blue")
    ax.set_ylabel("λ (variance scaling rate)")
    ax.set_title("λ")

    # (b) log α.
    ax = axes[0, 1]
    ax.plot(fs.steps, fs.log_alpha_values, marker="o", markersize=3, lw=1.5,
            color="tab:orange")
    ax.set_ylabel("log α (variance prefactor)")
    ax.set_title("log α")

    # (c) mean effective rank.
    ax = axes[1, 0]
    mean_er = fs.effective_ranks.mean(axis=1)
    ax.plot(fs.steps, mean_er, marker="o", markersize=3, lw=1.5,
            color="tab:green")
    ax.set_ylabel("mean effective rank")
    ax.set_title(f"Mean effective rank (across {fs.L} layer states)")
    ax.axhline(fs.H, color="gray", ls=":", lw=1, label=f"H = {fs.H}")
    ax.legend()

    # (d) mean kurtosis.
    ax = axes[1, 1]
    mean_kurt = np.nanmean(fs.kurtosis, axis=1)
    ax.plot(fs.steps, mean_kurt, marker="o", markersize=3, lw=1.5,
            color="tab:red")
    ax.set_ylabel("mean excess kurtosis of residuals")
    ax.set_title("Residual kurtosis (0 = Gaussian)")
    ax.axhline(0, color="gray", ls=":", lw=1)

    # (e) mean isotropy.
    ax = axes[2, 0]
    mean_iso = np.nanmean(fs.isotropy, axis=1)
    ax.plot(fs.steps, mean_iso, marker="o", markersize=3, lw=1.5,
            color="tab:purple")
    ax.set_ylabel("mean isotropy (std of log Σ²_residual)")
    ax.set_title("Residual isotropy (0 = perfectly isotropic)")

    # (f) eval loss.
    ax = axes[2, 1]
    ax.plot(fs.steps, fs.eval_losses, marker="o", markersize=3, lw=1.5,
            color="black")
    ax.set_ylabel("held-out eval loss")
    ax.set_title("Eval loss (for cross-reference)")

    for ax in axes.flat:
        ax.set_xscale("log")
    for ax in axes[-1, :]:
        ax.set_xlabel("training step (log scale)")

    fig.suptitle(f"Basis-invariant convergence diagnostics{title_suffix}", y=1.00)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Per-layer effective rank depth profile, colored by step.
# ----------------------------------------------------------------------
def plot_effective_rank_depth_profile(
    flow_series: FlowSeries, output_path: str,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (8, 5),
):
    """
    Effective rank vs layer index, with one curve per checkpoint colored
    by training step.

    Tracks how the depth profile of representational dimensionality
    evolves during training. Settled trained models typically show a
    distinctive depth-dependent profile (often U-shaped or monotonically
    increasing).
    """
    fig, ax = plt.subplots(figsize=figsize)
    fs = flow_series
    cmap = plt.cm.viridis

    # Color by log step so early-training checkpoints get distinct colors.
    log_steps = np.log(np.maximum(fs.steps, 1))
    norm = (log_steps - log_steps.min()) / (
        log_steps.max() - log_steps.min() + 1e-30
    )

    for k in range(fs.K):
        ax.plot(
            np.arange(fs.L), fs.effective_ranks[k],
            color=cmap(norm[k]), alpha=0.7, lw=1.0,
        )
    ax.set_xlabel("layer state (0 = post-embedding, L = post-final-norm)")
    ax.set_ylabel("effective rank")
    ax.set_title(f"Effective rank depth profile{title_suffix}")
    ax.axhline(fs.H, color="gray", ls=":", lw=1, label=f"H = {fs.H}")

    # Colorbar.
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=fs.steps.min(), vmax=fs.steps.max()),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("training step")
    ax.legend(loc="best")

    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: Flow-distance-to-final convergence curves.
# ----------------------------------------------------------------------
def plot_flow_convergence(
    flow_series: FlowSeries, output_path: str,
    reference_index: int = -1,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Distance from each checkpoint's flow to the reference checkpoint's
    flow, vs training step.

    Two distance metrics shown:
      - singular value spectrum distance (basis-invariant)
      - effective rank distance (basis-invariant)

    Plus an overlay of normalized eval loss for cross-reference.

    Args:
        reference_index: which checkpoint serves as "the final flow".
            Default -1 = the last checkpoint. If you want to compare to
            a different reference, override.
    """
    fs = flow_series
    ref_flow = {
        "singular_values": fs.singular_values[reference_index],
        "effective_rank": fs.effective_ranks[reference_index],
        "kurtosis_per_layer": fs.kurtosis[reference_index],
        "isotropy_per_layer": fs.isotropy[reference_index],
        "lambda": float(fs.lambda_values[reference_index]),
        "log_alpha": float(fs.log_alpha_values[reference_index]),
    }
    sv_dists = np.zeros(fs.K, dtype=np.float32)
    er_dists = np.zeros(fs.K, dtype=np.float32)
    lam_dists = np.zeros(fs.K, dtype=np.float32)
    for k in range(fs.K):
        flow_k = {
            "singular_values": fs.singular_values[k],
            "effective_rank": fs.effective_ranks[k],
            "kurtosis_per_layer": fs.kurtosis[k],
            "isotropy_per_layer": fs.isotropy[k],
            "lambda": float(fs.lambda_values[k]),
            "log_alpha": float(fs.log_alpha_values[k]),
        }
        sv_dists[k] = singular_value_distance(flow_k, ref_flow)
        er_dists[k] = effective_rank_distance(flow_k, ref_flow)
        lam_dists[k] = lambda_distance(flow_k, ref_flow)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    # Panel 1: distance curves.
    ax1.plot(fs.steps, sv_dists, marker="o", markersize=3, lw=1.5,
             color="tab:blue", label="Σ-spectrum distance (log space, summed over layers)")
    ax1.plot(fs.steps, er_dists, marker="s", markersize=3, lw=1.5,
             color="tab:green", label="Effective rank distance (summed over layers)")
    ax1.set_xscale("log")
    ax1.set_xlabel("training step (log scale)")
    ax1.set_ylabel(f"distance to checkpoint at step {fs.steps[reference_index]}")
    ax1.set_title("Flow-distance convergence")
    ax1.legend(loc="upper right")

    # Panel 2: comparison with loss.
    # Normalize both to [0, 1] using each's first checkpoint as the reference.
    loss_norm = (fs.eval_losses - fs.eval_losses[reference_index]) / (
        fs.eval_losses[0] - fs.eval_losses[reference_index] + 1e-30
    )
    sv_norm = sv_dists / (sv_dists[0] + 1e-30)
    ax2.plot(fs.steps, loss_norm, marker="o", markersize=3, lw=1.5,
             color="black", label="normalized eval loss")
    ax2.plot(fs.steps, sv_norm, marker="o", markersize=3, lw=1.5,
             color="tab:blue", label="normalized Σ-spectrum distance")
    ax2.set_xscale("log")
    ax2.set_xlabel("training step (log scale)")
    ax2.set_ylabel("normalized distance to final (0 = converged)")
    ax2.set_title("Flow convergence vs loss convergence")
    ax2.legend(loc="upper right")
    ax2.axhline(0, color="gray", ls=":", lw=1)
    ax2.axhline(1, color="gray", ls=":", lw=1)

    fig.suptitle(f"Convergence of the linear flow to L(K_final){title_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5: Variance scaling fit for a representative checkpoint.
# ----------------------------------------------------------------------
def plot_variance_scaling_fit(
    flow_series: FlowSeries, output_path: str,
    checkpoint_index: int = -1,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (8, 5),
):
    """
    For a single checkpoint, plot log(residual variance) vs target layer
    index (t+τ), averaged across source layers t. Overlay the linear fit
    log σ² = log α + λ (t+τ).

    A clean linear relationship validates the framework's exponential
    scaling assumption. Departure from linearity at early or late layers
    is the "first/last layer anomaly" the paper notes.
    """
    fs = flow_series
    k = checkpoint_index if checkpoint_index >= 0 else fs.K + checkpoint_index
    pwv = fs.pairwise_residual_variance[k]  # (L, L)
    L = pwv.shape[0]

    endpoint_indices = []
    endpoint_log_vars = []
    for end in range(1, L):
        vars_at_end = pwv[:end, end]
        vars_at_end = vars_at_end[~np.isnan(vars_at_end)]
        if len(vars_at_end) == 0:
            continue
        endpoint_indices.append(end)
        endpoint_log_vars.append(np.log(vars_at_end.mean()))
    endpoint_indices = np.array(endpoint_indices, dtype=np.float32)
    endpoint_log_vars = np.array(endpoint_log_vars, dtype=np.float32)

    log_alpha = float(fs.log_alpha_values[k])
    lam = float(fs.lambda_values[k])

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(endpoint_indices, endpoint_log_vars,
               color="tab:blue", s=50, label="measured")
    # Overlay the linear fit.
    fit_x = np.array([endpoint_indices.min() - 0.5, endpoint_indices.max() + 0.5])
    fit_y = log_alpha + lam * fit_x
    ax.plot(fit_x, fit_y, color="tab:red", lw=2,
            label=f"fit: log σ² = {log_alpha:.3f} + {lam:.4f} (t+τ)")
    ax.set_xlabel("target layer index (t+τ)")
    ax.set_ylabel("log(mean residual variance per coordinate)")
    ax.set_title(
        f"Variance scaling fit at step {fs.steps[k]}, "
        f"eval loss {fs.eval_losses[k]:.3f}{title_suffix}"
    )
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 6: Singular value spectra at representative checkpoints.
# ----------------------------------------------------------------------
def plot_singular_value_spectra(
    flow_series: FlowSeries, output_path: str,
    layer_indices: Optional[List[int]] = None,
    checkpoint_indices: Optional[List[int]] = None,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Singular value spectra at selected layers, for selected checkpoints.

    Two-panel plot:
      Left: spectra at multiple layers for the FINAL checkpoint (depth profile).
      Right: spectra at the MID layer over multiple checkpoints (training profile).

    Args:
        layer_indices: which layers to show in left panel. Default: 5 evenly-spaced.
        checkpoint_indices: which checkpoints to show in right panel.
            Default: 5 evenly-spaced.
    """
    fs = flow_series
    if layer_indices is None:
        layer_indices = list(np.linspace(0, fs.L - 1, 5, dtype=int))
    if checkpoint_indices is None:
        checkpoint_indices = list(np.linspace(0, fs.K - 1, 5, dtype=int))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    cmap_layer = plt.cm.viridis
    cmap_step = plt.cm.plasma

    # Left: spectra at multiple layers for the final checkpoint.
    final_k = fs.K - 1
    for li, layer in enumerate(layer_indices):
        sv = fs.singular_values[final_k, layer]
        rank = np.arange(1, len(sv) + 1)
        color = cmap_layer(li / max(len(layer_indices) - 1, 1))
        ax1.plot(rank, sv, color=color, lw=1.2,
                 label=f"layer {layer}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("rank")
    ax1.set_ylabel("singular value")
    ax1.set_title(f"Spectra by depth, step {fs.steps[final_k]}")
    ax1.legend(loc="best", fontsize=9)

    # Right: spectra at mid layer over multiple checkpoints.
    mid_layer = fs.L // 2
    for ki, k in enumerate(checkpoint_indices):
        sv = fs.singular_values[k, mid_layer]
        rank = np.arange(1, len(sv) + 1)
        color = cmap_step(ki / max(len(checkpoint_indices) - 1, 1))
        ax2.plot(rank, sv, color=color, lw=1.2,
                 label=f"step {fs.steps[k]}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("rank")
    ax2.set_ylabel("singular value")
    ax2.set_title(f"Spectra by training, layer {mid_layer}")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"Singular value spectra{title_suffix}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 7: Successive-layer principal angles (smoothness check).
# ----------------------------------------------------------------------
def plot_successive_layer_angles(
    flow_series: FlowSeries, output_path: str,
    checkpoint_indices: Optional[List[int]] = None,
    top_k: int = 10,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (8, 5),
):
    """
    Mean principal angle (degrees) between R(t) and R(t+1), for each t,
    for selected checkpoints.

    This is the Frame 2 smoothness diagnostic: small angles mean the basis
    rotates smoothly with depth (the paper's Fig 2(a) observation for
    trained models). Large or chaotic angles would indicate the model
    hasn't developed a coherent rotational structure.

    Requires R matrices, which are loaded lazily.
    """
    fs = flow_series
    if fs.R is None:
        fs.load_R()
    if checkpoint_indices is None:
        # Sparse subset: early, middle, late.
        checkpoint_indices = [0, fs.K // 4, fs.K // 2, 3 * fs.K // 4, fs.K - 1]

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.plasma
    for ki, k in enumerate(checkpoint_indices):
        angles = successive_layer_angles(fs.R[k], top_k=top_k)
        layer_pairs = np.arange(len(angles))
        color = cmap(ki / max(len(checkpoint_indices) - 1, 1))
        ax.plot(layer_pairs, angles, marker="o", markersize=4, lw=1.5,
                color=color, label=f"step {fs.steps[k]}")
    ax.set_xlabel("layer transition index t (angle between R(t) and R(t+1))")
    ax.set_ylabel(f"mean principal angle (top-{top_k} directions), degrees")
    ax.set_title(f"Smoothness of R(t) trajectory{title_suffix}")
    ax.legend(loc="best", fontsize=9)
    # A 90° reference line: if angles approach this, the basis is essentially
    # uncorrelated between consecutive layers (bad).
    ax.axhline(90, color="gray", ls=":", lw=1, label="90° (uncorrelated)")
    ax.set_ylim(0, 100)
    fig.savefig(output_path)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 8: Pairwise residual variance heatmap.
# ----------------------------------------------------------------------
def plot_pairwise_residual_heatmap(
    flow_series: FlowSeries, output_path: str,
    checkpoint_index: int = -1,
    title_suffix: str = "",
    figsize: Tuple[float, float] = (7, 6),
):
    """
    Heatmap of pairwise residual variance, log scale, for one checkpoint.

    Rows are source layers t, columns are target layers t+τ. Diagonal and
    below are NaN (undefined). The off-diagonal pattern should show
    monotonic growth with (t+τ) — that's the lines-of-thought framework's
    main prediction. Departure from monotonicity at early or late layers
    is the first/last layer anomaly.
    """
    fs = flow_series
    k = checkpoint_index if checkpoint_index >= 0 else fs.K + checkpoint_index
    pwv = fs.pairwise_residual_variance[k]  # (L, L)
    log_pwv = np.log(np.maximum(pwv, 1e-30))
    # Mask NaN.
    masked = np.ma.masked_invalid(log_pwv)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xlabel("target layer (t+τ)")
    ax.set_ylabel("source layer (t)")
    ax.set_title(
        f"log(residual variance) heatmap, step {fs.steps[k]}, "
        f"eval loss {fs.eval_losses[k]:.3f}{title_suffix}"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log(per-coord residual variance)")
    fig.savefig(output_path)
    plt.close(fig)
