"""
Plotting for the multi-view analysis.

Produces the figures called out in proposal §4:

  Figure 1: Per-view basis-invariant statistic dashboard (§4.1)
            -- side-by-side log_alpha, lambda, effective rank, kurtosis
               across all-to-all / forward / reverse views.

  Figure 2: Within/between decomposition curve (§4.2)
            -- one panel per view, layer on x-axis, within and between
               variance fractions stacked.

  Figure 3: Crossover layer trajectory through training (§4.3)
            -- training step (log scale) vs crossover layer, one curve
               per seed per view.

  Figure 4: Co-location with Phase 1 anomalies (§4.4)
            -- log_alpha trajectory with crossover stabilization events
               marked.

  Figure 5: Effective rank profiles (§4.4)
            -- effective rank vs layer, three curves (one per view) at
               the final checkpoint.

  Figure 6: Per-token forward-view spread profiles
            -- one curve per token in the forward set, showing how much
               the per-token bundle grows with depth.

These are reference plotters; each is small, takes a trajectory file or
a single MultiViewResult, and writes a PNG. They are deliberately not
batched into a "make all figures" entry point because the project's
final figure layout will probably differ in ways we can't anticipate.
Use these as building blocks.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from multiview import (
    MultiViewResult,
    DecompositionResult,
    crossover_layer,
)


# ----------------------------------------------------------------------
# Figure 1: Per-view dashboard at one checkpoint.
# ----------------------------------------------------------------------
def plot_per_view_dashboard(
    result: MultiViewResult,
    output_path: str,
    figsize: Tuple[float, float] = (12, 8),
) -> None:
    """
    Four-panel dashboard for one MultiViewResult:
      (0,0) variance-scaling slope lambda, per view (bar)
      (0,1) per-coordinate variance at last block vs post-final-norm (bar)
      (1,0) effective rank profile vs layer, per view
      (1,1) kurtosis profile vs layer, per view

    The original layout had a log_alpha bar in (0,0). That field is
    interpretable for the all-to-all marginal but becomes a degenerate
    extreme value for the conditional views (the conditional bundle has
    zero variance at t=0, so the log-linear fit extrapolates to a very
    negative intercept). Plotting all three on the same axis hides every
    other quantity. We drop log_alpha from the cross-view comparison and
    instead show lambda (the growth slope, which is well-defined for all
    views) plus a sanity check of post-final-norm variance behavior.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    views = ["all_to_all", "forward", "reverse_actual"]
    colors = {"all_to_all": "k", "forward": "C0", "reverse_actual": "C3"}
    labels = {"all_to_all": "all-to-all",
              "forward": "forward (input-cond.)",
              "reverse_actual": "reverse (output-cond.)"}

    # Collect scalar lambda per view (frequency-weighted for restricted
    # views). log_alpha is omitted from the dashboard; see docstring.
    def _scalar(view: str, key: str) -> float:
        if view == "all_to_all":
            return float(result.all_to_all.get(key, np.nan))
        flows = (result.forward_flows if view == "forward"
                 else result.reverse_actual_flows)
        tset = (result.forward_set if view == "forward"
                else result.reverse_actual_set)
        if tset.token_ids.size == 0:
            return np.nan
        w = tset.counts.astype(np.float64)
        w = w / w.sum()
        vals = []
        for tid in tset.token_ids:
            f = flows.get(int(tid))
            vals.append(float(f.get(key, np.nan)) if f is not None else np.nan)
        return float(np.nansum(np.array(vals) * w))

    # Panel (0,0): lambda bar, per view.
    ax = axes[0, 0]
    xs = np.arange(len(views))
    ys = [_scalar(v, "lambda") for v in views]
    bars = ax.bar(xs, ys, color=[colors[v] for v in views])
    for bar, y in zip(bars, ys):
        if not np.isnan(y):
            ax.text(bar.get_x() + bar.get_width() / 2, y + 0.005,
                    f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[v] for v in views], rotation=15)
    ax.set_ylabel(r"$\lambda$")
    ax.set_title("Variance growth slope per view")

    # Panel (0,1): pre vs post final-norm per-coordinate variance, per view.
    # Pulls v_subset_total[-2] (last block output) and v_subset_total[-1]
    # (post-final-norm) from each decomposition. For the all-to-all this
    # uses v_all_to_all instead, which is the marginal version.
    ax = axes[0, 1]
    pre_vals, post_vals = [], []
    for v in views:
        if v == "all_to_all":
            arr = result.forward_decomp.v_all_to_all  # same data for all decomps
            pre_vals.append(float(arr[-2]))
            post_vals.append(float(arr[-1]))
        else:
            decomp = (result.forward_decomp if v == "forward"
                      else result.reverse_actual_decomp)
            pre_vals.append(float(decomp.v_subset_total[-2]))
            post_vals.append(float(decomp.v_subset_total[-1]))
    width = 0.35
    ax.bar(xs - width / 2, pre_vals, width, color=[colors[v] for v in views],
           alpha=0.85, label="last block output (t=L)")
    ax.bar(xs + width / 2, post_vals, width, color=[colors[v] for v in views],
           alpha=0.45, label="post-final-norm (t=L+1)")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[v] for v in views], rotation=15)
    ax.set_ylabel("per-coord variance")
    ax.set_title("Variance at boundary states, per view")
    ax.legend(loc="upper left", fontsize=8)

    # Panel (1,0): effective rank profile.
    ax = axes[1, 0]
    er_aa = result.all_to_all["effective_rank"]
    layers = np.arange(er_aa.size)
    ax.plot(layers, er_aa, "-o", color=colors["all_to_all"], label=labels["all_to_all"],
            markersize=4)
    for view, flows, tset in [
        ("forward", result.forward_flows, result.forward_set),
        ("reverse_actual", result.reverse_actual_flows, result.reverse_actual_set),
    ]:
        if tset.token_ids.size == 0:
            continue
        w = tset.counts.astype(np.float64)
        w = w / w.sum()
        stack = np.full((tset.token_ids.size, er_aa.size), np.nan)
        for k, tid in enumerate(tset.token_ids):
            f = flows.get(int(tid))
            if f is not None and "effective_rank" in f:
                stack[k] = f["effective_rank"]
        avg = np.nansum(stack * w[:, None], axis=0)
        ax.plot(layers, avg, "-o", color=colors[view], label=labels[view],
                markersize=4)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("effective rank")
    ax.set_title("Effective rank profile")
    ax.legend(loc="best", fontsize=9)

    # Panel (1,1): kurtosis profile.
    ax = axes[1, 1]
    k_aa = result.all_to_all["kurtosis_per_layer"]
    ax.plot(layers, k_aa, "-o", color=colors["all_to_all"], label=labels["all_to_all"],
            markersize=4)
    for view, flows, tset in [
        ("forward", result.forward_flows, result.forward_set),
        ("reverse_actual", result.reverse_actual_flows, result.reverse_actual_set),
    ]:
        if tset.token_ids.size == 0:
            continue
        w = tset.counts.astype(np.float64)
        w = w / w.sum()
        stack = np.full((tset.token_ids.size, k_aa.size), np.nan)
        for k_idx, tid in enumerate(tset.token_ids):
            f = flows.get(int(tid))
            if f is not None and "kurtosis_per_layer" in f:
                stack[k_idx] = f["kurtosis_per_layer"]
        avg = np.nansum(stack * w[:, None], axis=0)
        ax.plot(layers, avg, "-o", color=colors[view], label=labels[view],
                markersize=4)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title("Kurtosis profile")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(f"Per-view dashboard  (seed {result.seed}, step {result.step})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 2: Within/between decomposition curve.
# ----------------------------------------------------------------------
def plot_decomposition(
    decomp: DecompositionResult,
    output_path: str,
    crossover_dir: str = "forward",
    title_suffix: str = "",
    figsize: Tuple[float, float] = (8, 5),
) -> None:
    """
    Layer-by-layer within/between variance plot for one decomposition.

    Plots v_within and v_between as fractions of v_subset_total, stacked
    (so they sum to 1.0). Marks the crossover layer if it exists.

    crossover_dir is 'forward' or 'reverse'; determines which way the
    crossover gets detected for annotation.
    """
    L = decomp.v_within.size
    layers = np.arange(L)
    w_frac = decomp.within_fraction
    b_frac = decomp.between_fraction

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(layers, 0, b_frac, alpha=0.55, color="C0",
                    label=r"between-condition $\mathrm{Var}_z[\mu_t(z)]$")
    ax.fill_between(layers, b_frac, b_frac + w_frac, alpha=0.55, color="C3",
                    label=r"within-condition $\mathbb{E}_z[V_z(t)]$")
    ax.plot(layers, b_frac, "-", color="C0", lw=1.2)
    ax.plot(layers, b_frac + w_frac, "-", color="C3", lw=1.2)
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("variance fraction")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, L - 1)

    c, status = crossover_layer(decomp.v_within, decomp.v_between, direction=crossover_dir)
    if status == "crossover":
        ax.axvline(c, color="k", ls="--", lw=1)
        ax.annotate(f"crossover ≈ {c:.2f}",
                    xy=(c, 0.5), xytext=(c + 0.5, 0.55),
                    fontsize=10, color="k")
    elif status == "always_true":
        ax.text(0.5, 0.92, f"always {crossover_dir}-dominant", transform=ax.transAxes,
                ha="center", fontsize=10)
    elif status == "no_crossover":
        ax.text(0.5, 0.92, f"no {crossover_dir} crossover", transform=ax.transAxes,
                ha="center", fontsize=10)

    cov = decomp.subset_coverage
    cov_mean = float(np.nanmean(cov))
    ax.text(0.02, 0.96, f"subset covers {cov_mean*100:.1f}% of all-to-all variance",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")

    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Within/between decomposition: {decomp.view}{title_suffix}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_decomposition_triptych(
    result: MultiViewResult,
    output_path: str,
    figsize: Tuple[float, float] = (14, 5),
) -> None:
    """Three-panel: forward, reverse_actual, reverse_pred decompositions."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, decomp, direction, title in [
        (axes[0], result.forward_decomp, "forward", "Forward (input-conditioned)"),
        (axes[1], result.reverse_actual_decomp, "reverse",
         "Reverse, actual successor"),
        (axes[2], result.reverse_pred_decomp, "reverse",
         "Reverse, predicted successor"),
    ]:
        L = decomp.v_within.size
        layers = np.arange(L)
        if decomp.token_ids.size == 0:
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center")
            ax.set_title(title)
            continue
        w_frac = decomp.within_fraction
        b_frac = decomp.between_fraction
        ax.fill_between(layers, 0, b_frac, alpha=0.55, color="C0",
                        label="between")
        ax.fill_between(layers, b_frac, b_frac + w_frac, alpha=0.55, color="C3",
                        label="within")
        ax.plot(layers, b_frac, "-", color="C0", lw=1.2)
        ax.plot(layers, b_frac + w_frac, "-", color="C3", lw=1.2)
        c, status = crossover_layer(decomp.v_within, decomp.v_between,
                                    direction=direction)
        if status == "crossover":
            ax.axvline(c, color="k", ls="--", lw=1)
            ax.text(c + 0.2, 0.05, f"{c:.2f}", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("layer state index t")
        ax.set_xlim(0, L - 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("variance fraction")
    fig.suptitle(f"Three-view decomposition  "
                 f"(seed {result.seed}, step {result.step})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 3: Crossover trajectory through training.
# ----------------------------------------------------------------------
def plot_crossover_trajectory(
    crossover_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (10, 6),
    phase1_events: Optional[Dict[str, int]] = None,
) -> None:
    """
    Crossover layer vs training step, one curve per seed per view.

    Args:
        crossover_path: path to trajectories/crossover.npz from stage D.
        phase1_events: optional dict mapping event name to step, drawn as
                       vertical reference lines. Typical entries:
                         {'post-final-norm anomaly emerges': 1000,
                          'log alpha hump peak': 5000,
                          'sigma-distance bump': 7500}
    """
    data = np.load(crossover_path)
    seeds = data["seeds"]
    steps = data["steps"]
    c_fwd = data["crossover_forward"]            # (n_s, n_t)
    c_ra = data["crossover_reverse_actual"]      # (n_s, n_t)
    c_rp = data["crossover_reverse_pred"]        # (n_s, n_t)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, arr, title in [
        (axes[0], c_fwd, "Forward crossover"),
        (axes[1], c_ra, "Reverse (actual)"),
        (axes[2], c_rp, "Reverse (predicted)"),
    ]:
        for si, seed in enumerate(seeds):
            ax.plot(steps, arr[si], "-o", markersize=3, lw=1, label=f"seed {seed}")
        if phase1_events:
            for name, step in phase1_events.items():
                ax.axvline(step, color="gray", ls=":", lw=1)
                ax.text(step, ax.get_ylim()[1] * 0.95, name,
                        rotation=90, fontsize=7, color="gray",
                        ha="right", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("training step (log)")
        ax.set_title(title)
        ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    axes[0].set_ylabel("crossover layer")
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Crossover layer trajectory through training")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 4: training trajectories of variance-fit scalars.
# ----------------------------------------------------------------------
def plot_log_alpha_trajectory(
    variance_fit_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (12, 5),
    phase1_events: Optional[Dict[str, int]] = None,
) -> None:
    """Two-panel training-step trajectories from variance_fit.npz.

    Left panel: log_alpha for the all-to-all view only. The conditional
    views' log_alpha values are mathematically well-defined but
    extremely negative (the per-token bundle has near-zero variance at
    t=0, so the log-linear fit extrapolates to a huge negative
    intercept). Plotting all three on the same axis hides the all-to-all
    curve entirely, which is the one whose trajectory carries the
    interpretable training-dynamic signal (the 'log alpha hump').

    Right panel: lambda (the growth slope), which is well-defined across
    views. This is where the cross-view comparison through training
    actually works.

    The right panel is the one that supports proposal §4.4 (co-location
    with Phase 1 anomalies) for the cross-view comparison; the left
    panel mirrors the Phase 1 log_alpha-through-training plot.
    """
    data = np.load(variance_fit_path)
    seeds = data["seeds"]
    steps = data["steps"]
    log_alpha_aa = data["log_alpha_all_to_all"]
    lambda_aa = data["lambda_all_to_all"]
    lambda_fwd = data["lambda_forward"]
    lambda_rev = data["lambda_reverse_actual"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # Left panel: log_alpha for all-to-all only.
    for si in range(log_alpha_aa.shape[0]):
        ax_left.plot(steps, log_alpha_aa[si], "-", color="k",
                     alpha=0.25, lw=0.7)
    ax_left.plot(steps, np.nanmean(log_alpha_aa, axis=0), "-o", color="k",
                 lw=2, markersize=4, label="all-to-all")
    ax_left.set_xscale("log")
    ax_left.set_xlabel("training step")
    ax_left.set_ylabel(r"$\log\alpha$")
    ax_left.set_title(r"$\log\alpha$ through training (all-to-all)")
    ax_left.legend(loc="best", fontsize=9)
    ax_left.grid(True, ls=":", lw=0.5, alpha=0.5)
    if phase1_events:
        ymin, ymax = ax_left.get_ylim()
        for name, step in phase1_events.items():
            ax_left.axvline(step, color="gray", ls=":", lw=1)
            ax_left.text(step, ymax, name, rotation=90, fontsize=7,
                         color="gray", ha="right", va="top")

    # Right panel: lambda for all three views.
    series = {
        "all-to-all": lambda_aa,
        "forward (per-token avg)": lambda_fwd,
        "reverse (per-token avg)": lambda_rev,
    }
    colors = {"all-to-all": "k", "forward (per-token avg)": "C0",
              "reverse (per-token avg)": "C3"}
    for view_name, arr in series.items():
        for si in range(arr.shape[0]):
            ax_right.plot(steps, arr[si], "-", color=colors[view_name],
                          alpha=0.25, lw=0.7)
        ax_right.plot(steps, np.nanmean(arr, axis=0), "-o",
                      color=colors[view_name], lw=2, markersize=4,
                      label=view_name)
    ax_right.set_xscale("log")
    ax_right.set_xlabel("training step")
    ax_right.set_ylabel(r"$\lambda$")
    ax_right.set_title(r"$\lambda$ through training, per view")
    ax_right.legend(loc="best", fontsize=9)
    ax_right.grid(True, ls=":", lw=0.5, alpha=0.5)
    if phase1_events:
        ymin, ymax = ax_right.get_ylim()
        for name, step in phase1_events.items():
            ax_right.axvline(step, color="gray", ls=":", lw=1)
            ax_right.text(step, ymax, name, rotation=90, fontsize=7,
                          color="gray", ha="right", va="top")

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Figure 5: per-token forward spread profiles.
# ----------------------------------------------------------------------
def plot_per_token_spread(
    result: MultiViewResult,
    output_path: str,
    view: str = "forward",
    figsize: Tuple[float, float] = (10, 6),
) -> None:
    """
    Per-token within-cluster variance profile (one curve per token in
    the view's token set). For the forward view this is the "how fast
    does context spread the bundle for this token" plot.
    """
    if view == "forward":
        flows = result.forward_flows
        tset = result.forward_set
    elif view == "reverse_actual":
        flows = result.reverse_actual_flows
        tset = result.reverse_actual_set
    elif view == "reverse_pred":
        flows = result.reverse_pred_flows
        tset = result.reverse_pred_set
    else:
        raise ValueError(f"Unknown view: {view}")

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.viridis
    if tset.token_ids.size == 0:
        ax.text(0.5, 0.5, "(empty token set)", transform=ax.transAxes, ha="center")
    else:
        # For each token, plot per-coord variance averaged across coords.
        for k, tid in enumerate(tset.token_ids):
            f = flows.get(int(tid))
            if f is None:
                continue
            sv = f.get("singular_values")  # (L, H)
            n = f.get("n_pilots", 0)
            if sv is None or n <= 1:
                continue
            # Per-coord variance from SVD: (1/(N-1)) * sum_i sigma_i^2 / H,
            # but we already have means and centered SVD, so an
            # equivalent layer-wise variance summary is mean across i of
            # singular_values^2 / (n-1) / H. We use sigma^2 / (H * (N-1))
            # to get back to E[Var_coord].
            n = max(n, 2)
            var_per_layer = (sv ** 2).sum(axis=1) / (sv.shape[1] * (n - 1))
            ax.plot(np.arange(sv.shape[0]), var_per_layer, "-",
                    color=cmap(k / max(tset.token_ids.size - 1, 1)),
                    alpha=0.7, lw=1, label=f"tid {tid} (n={tset.counts[k]})")
    ax.set_yscale("log")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel(r"per-coord variance $\langle\sigma^2_i\rangle_i$")
    ax.set_title(f"Per-token spread, view = {view}  "
                 f"(seed {result.seed}, step {result.step})")
    # Legend can get crowded; print only first/last few labels.
    handles, labels = ax.get_legend_handles_labels()
    if len(labels) > 8:
        keep = list(range(4)) + list(range(len(labels) - 4, len(labels)))
        handles = [handles[i] for i in keep]
        labels = [labels[i] for i in keep]
    ax.legend(handles, labels, loc="best", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------
# Headline figure: polished within/between ratio profile with annotations.
# ----------------------------------------------------------------------
def plot_ratio_profile_headline(
    decomposition_path: str,
    output_path: str,
    step_index: int = -1,
    figsize: Tuple[float, float] = (11, 6.5),
) -> None:
    """The within/between ratio profile, polished for publication.

    Layout:
      - Three curves: forward, reverse_actual, reverse_pred.
      - Cross-seed mean as solid line with markers; per-seed traces as
        faint background lines (the cross-seed agreement is so tight
        these typically overlap).
      - Horizontal reference line at ratio = 1 (the within=between
        boundary).
      - Annotations at three key layers per curve: t=0 (embedding state),
        the peak layer for each curve, and t=T (last block output).
      - Linear y-axis to keep the ratio = 1 reference visible. The
        previous log-y version compressed the interesting range.

    Args:
        decomposition_path: path to trajectories/decomposition.npz.
        step_index: training step to plot. Default -1 = final checkpoint.
        figsize: matplotlib figure size.
    """
    with np.load(decomposition_path) as f:
        w_fwd = f["v_within_forward"][:, step_index, :]
        b_fwd = f["v_between_forward"][:, step_index, :]
        w_ra = f["v_within_reverse_actual"][:, step_index, :]
        b_ra = f["v_between_reverse_actual"][:, step_index, :]
        w_rp = f["v_within_reverse_pred"][:, step_index, :]
        b_rp = f["v_between_reverse_pred"][:, step_index, :]
        steps = f["steps"]
        step_value = int(steps[step_index])

    def _ratio(w: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, w / b, np.nan)

    r_fwd = _ratio(w_fwd, b_fwd)
    r_ra = _ratio(w_ra, b_ra)
    r_rp = _ratio(w_rp, b_rp)

    L = r_fwd.shape[1]
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=figsize)

    curves = [
        (r_fwd, "C0", "forward — within(input) / between(input)"),
        (r_ra, "C3", "reverse, actual — within(out) / between(out)"),
        (r_rp, "C2", "reverse, predicted — within(pred) / between(pred)"),
    ]

    # Plot per-seed faint, mean bold.
    for arr, color, label in curves:
        for s in range(arr.shape[0]):
            ax.plot(layers, arr[s], "-", color=color, alpha=0.18, lw=0.8)
        mean = np.nanmean(arr, axis=0)
        ax.plot(layers, mean, "-o", color=color, lw=2.0, markersize=5,
                label=label, zorder=5)

    # Reference at ratio = 1.
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, zorder=1)
    ax.text(L - 0.5, 1.0, "  within = between", color="gray", fontsize=8,
            va="center", ha="left")

    # Annotate three key points per curve: t=0, peak, t=L (last block, t=12 if L=14).
    # Skip annotating points that are NaN.
    last_block_t = L - 2  # post-final-norm is the last index; last block is L-2.
    for arr, color, _ in curves:
        mean = np.nanmean(arr, axis=0)
        if np.all(np.isnan(mean)):
            continue
        # t = 0.
        if not np.isnan(mean[0]):
            ax.annotate(f"{mean[0]:.1f}", xy=(0, mean[0]),
                        xytext=(3, 6), textcoords="offset points",
                        color=color, fontsize=8.5, weight="bold")
        # Peak layer.
        peak_t = int(np.nanargmax(mean))
        peak_v = mean[peak_t]
        ax.annotate(f"peak={peak_v:.1f}", xy=(peak_t, peak_v),
                    xytext=(0, 10), textcoords="offset points",
                    color=color, fontsize=8.5, weight="bold", ha="center")
        # Last block output (t=L-2).
        ax.annotate(f"{mean[last_block_t]:.1f}",
                    xy=(last_block_t, mean[last_block_t]),
                    xytext=(-4, -15), textcoords="offset points",
                    color=color, fontsize=8.5, weight="bold", ha="right")

    # Forward crossover annotation (the one well-defined crossover in the data).
    # Place the label inside the chart but below the typical peak region
    # so it doesn't collide with the peak annotation of the reverse curves.
    mean_fwd = np.nanmean(r_fwd, axis=0)
    if not np.any(np.isnan(mean_fwd)):
        c, status = crossover_layer(np.nanmean(w_fwd, axis=0),
                                    np.nanmean(b_fwd, axis=0),
                                    direction="forward")
        if status == "crossover":
            ax.axvline(c, color="C0", ls="--", lw=1, alpha=0.6)
            # Place annotation low on the chart to avoid colliding with
            # the reverse curves' peak labels at the top.
            y_text = 2.5
            ax.text(c + 0.15, y_text,
                    f"forward crossover\nat t = {c:.2f}",
                    color="C0", fontsize=8.5, va="bottom", ha="left",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.8, pad=1.5))

    ax.set_xlabel("layer state index t")
    ax.set_ylabel("within / between variance ratio")
    ax.set_xlim(-0.3, L - 0.6)
    ax.set_xticks(layers)
    # y-limit: tight around the data but include 0 to make the
    # ratio = 1 boundary easy to read.
    y_max = max(np.nanmax(np.nanmean(arr, axis=0)) for arr, _, _ in curves) * 1.18
    ax.set_ylim(0, y_max)
    ax.set_title(f"Within/between variance ratio across the residual stream\n"
                 f"final checkpoint (step {step_value}), 4 seeds — "
                 f"faint = individual seeds, bold = cross-seed mean")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------
# Co-location figure: log_alpha hump and reverse-lambda dip on shared axis.
# ----------------------------------------------------------------------
def _find_extremum_region(
    y: np.ndarray, kind: str,
    drop_threshold: float = 0.30,
) -> Tuple[int, int, int]:
    """Find a local extremum and the width of its "plateau" or "trough"
    in 1D array y.

    Args:
        y: 1D array of values along training steps.
        kind: 'hump' (find maximum and surrounding plateau) or 'dip'
              (find minimum and surrounding trough).
        drop_threshold: fraction of (max - flank) at which to declare
              the plateau/trough edges. 0.30 means "edges where the
              curve has descended 30% of the way from the extremum back
              toward the overall trend."

    Returns:
        (extremum_index, left_edge_index, right_edge_index). All three
        are indices into y.

    The trend baseline is estimated as the linear interpolation between
    y[0] and y[-1] — this lets the function find a hump or dip relative
    to a monotone background, not just an absolute extremum.
    """
    n = y.size
    # Trend baseline: linear interpolation between endpoints.
    trend = np.linspace(y[0], y[-1], n)
    residual = y - trend
    if kind == "hump":
        idx_extremum = int(np.argmax(residual))
        extremum_height = residual[idx_extremum]
        threshold_height = extremum_height * (1 - drop_threshold)
        # Walk outward to find edges.
        left = idx_extremum
        while left > 0 and residual[left - 1] >= threshold_height:
            left -= 1
        right = idx_extremum
        while right < n - 1 and residual[right + 1] >= threshold_height:
            right += 1
    elif kind == "dip":
        idx_extremum = int(np.argmin(residual))
        extremum_depth = residual[idx_extremum]
        threshold_depth = extremum_depth * (1 - drop_threshold)
        left = idx_extremum
        while left > 0 and residual[left - 1] <= threshold_depth:
            left -= 1
        right = idx_extremum
        while right < n - 1 and residual[right + 1] <= threshold_depth:
            right += 1
    else:
        raise ValueError(f"kind must be 'hump' or 'dip', got {kind!r}")
    return idx_extremum, left, right


def plot_colocation(
    variance_fit_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (11, 6),
) -> None:
    """Co-location figure: all-to-all log_alpha and reverse lambda on
    the same training-step axis, with their respective hump and dip
    automatically detected and annotated.

    The visual claim is: these two quantities — log_alpha (the
    variance-scaling intercept of the all-to-all marginal) and lambda
    (the variance-growth slope of the reverse output-conditioned view)
    — see the same training-dynamic event from different geometric
    angles. log_alpha rises and plateaus; reverse lambda dips
    transiently in the same training-step window.

    Implementation:
      - Twin y-axes (left = log_alpha, right = lambda).
      - Both curves shown as cross-seed mean with faint per-seed traces.
      - The hump in log_alpha and the dip in reverse lambda are
        identified automatically by _find_extremum_region.
      - The overlap region (where both extrema are active) is shaded
        as gray, making the co-location visually unambiguous.

    Args:
        variance_fit_path: path to trajectories/variance_fit.npz.
        output_path: where to save the PNG.
        figsize: matplotlib figure size.
    """
    data = np.load(variance_fit_path)
    steps = data["steps"]
    log_alpha_aa = data["log_alpha_all_to_all"]      # (n_seeds, n_steps)
    lambda_rev = data["lambda_reverse_actual"]         # (n_seeds, n_steps)

    la_mean = np.nanmean(log_alpha_aa, axis=0)
    lr_mean = np.nanmean(lambda_rev, axis=0)

    # Auto-detect the hump in log_alpha and the dip in reverse lambda.
    la_peak_i, la_left_i, la_right_i = _find_extremum_region(la_mean, kind="hump")
    lr_dip_i, lr_left_i, lr_right_i = _find_extremum_region(lr_mean, kind="dip")

    fig, ax_la = plt.subplots(figsize=figsize)
    ax_lam = ax_la.twinx()

    # Shade the overlap of the two extremum regions.
    overlap_left_idx = max(la_left_i, lr_left_i)
    overlap_right_idx = min(la_right_i, lr_right_i)
    if overlap_left_idx <= overlap_right_idx:
        ax_la.axvspan(steps[overlap_left_idx], steps[overlap_right_idx],
                      color="gold", alpha=0.18, zorder=0,
                      label=f"co-location window\n(steps {steps[overlap_left_idx]}–"
                            f"{steps[overlap_right_idx]})")

    # log_alpha on left axis (black).
    for si in range(log_alpha_aa.shape[0]):
        ax_la.plot(steps, log_alpha_aa[si], "-", color="k",
                   alpha=0.20, lw=0.7)
    ax_la.plot(steps, la_mean, "-o", color="k", lw=2.0, markersize=4,
               label=r"all-to-all $\log\alpha$ (left axis)")
    # Mark the hump's peak.
    ax_la.plot(steps[la_peak_i], la_mean[la_peak_i], "*",
               color="k", markersize=14, zorder=6,
               markeredgecolor="white", markeredgewidth=1.0)
    ax_la.annotate(
        f"log $\\alpha$ hump peak\nat step {int(steps[la_peak_i])}",
        xy=(steps[la_peak_i], la_mean[la_peak_i]),
        xytext=(-80, 18), textcoords="offset points",
        fontsize=9, color="k", weight="bold",
        arrowprops=dict(arrowstyle="-", color="k", lw=0.7),
        bbox=dict(facecolor="white", edgecolor="k",
                  alpha=0.85, pad=2.0, lw=0.5),
    )

    # reverse lambda on right axis (red).
    for si in range(lambda_rev.shape[0]):
        ax_lam.plot(steps, lambda_rev[si], "-", color="C3",
                    alpha=0.20, lw=0.7)
    ax_lam.plot(steps, lr_mean, "-o", color="C3", lw=2.0, markersize=4,
                label=r"reverse $\lambda$ (right axis)")
    # Mark the dip's minimum.
    ax_lam.plot(steps[lr_dip_i], lr_mean[lr_dip_i], "*",
                color="C3", markersize=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1.0)
    ax_lam.annotate(
        f"reverse $\\lambda$ dip\nat step {int(steps[lr_dip_i])}",
        xy=(steps[lr_dip_i], lr_mean[lr_dip_i]),
        xytext=(40, -25), textcoords="offset points",
        fontsize=9, color="C3", weight="bold",
        arrowprops=dict(arrowstyle="-", color="C3", lw=0.7),
        bbox=dict(facecolor="white", edgecolor="C3",
                  alpha=0.85, pad=2.0, lw=0.5),
    )

    # Axes formatting.
    ax_la.set_xscale("log")
    ax_la.set_xlabel("training step")
    ax_la.set_ylabel(r"$\log\alpha$ (all-to-all)", color="k")
    ax_la.tick_params(axis="y", labelcolor="k")
    ax_lam.set_ylabel(r"$\lambda$ (reverse, per-token avg)", color="C3")
    ax_lam.tick_params(axis="y", labelcolor="C3")
    ax_la.grid(True, ls=":", lw=0.4, alpha=0.5)

    # Combined legend.
    handles_la, labels_la = ax_la.get_legend_handles_labels()
    handles_lam, labels_lam = ax_lam.get_legend_handles_labels()
    ax_la.legend(handles_la + handles_lam, labels_la + labels_lam,
                 loc="lower right", fontsize=9, framealpha=0.95)

    ax_la.set_title("Co-location of all-to-all $\\log\\alpha$ hump and "
                    "reverse-view $\\lambda$ dip\n"
                    "(faint = individual seeds; bold = cross-seed mean)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------
# Training-dynamic heatmap: within/between ratio across (step, layer).
# ----------------------------------------------------------------------
def plot_ratio_heatmap(
    decomposition_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (16, 6),
    vmin_vmax: Optional[Tuple[float, float]] = None,
) -> None:
    """Per-view heatmap of the within/between variance ratio as a
    function of (training step, layer state index).

    Three panels (forward, reverse_actual, reverse_pred). Each panel:
      x-axis: layer state index t (0 to L-1)
      y-axis: training step (log scale, low at bottom)
      color:  within/between ratio, using a symlog norm centered at 1.0

    The symlog color norm puts the "within = between" boundary at the
    midpoint of the colormap, so values below 1 (between dominates,
    bluish) and above 1 (within dominates, reddish) are equally visible.
    The forward crossover that takes place around layer 2 is thus
    immediately visible as the color transition through white.

    The ratio is averaged across seeds; cross-seed agreement is
    confirmed elsewhere (fig6 shows the per-seed traces overlap
    tightly).

    Args:
        decomposition_path: trajectories/decomposition.npz.
        output_path: where to save the PNG.
        figsize: matplotlib figure size.
        vmin_vmax: optional override for the color scale limits in (vmin, vmax).
                   Default chooses limits that span ~1/30 to 30 in ratio space.
    """
    from matplotlib.colors import SymLogNorm

    with np.load(decomposition_path) as f:
        seeds = f["seeds"]
        steps = f["steps"]
        # Each array shape: (n_seeds, n_steps, L)
        ratios = {
            "Forward (input-conditioned)": (
                f["v_within_forward"], f["v_between_forward"],
            ),
            "Reverse (actual successor)": (
                f["v_within_reverse_actual"], f["v_between_reverse_actual"],
            ),
            "Reverse (predicted successor)": (
                f["v_within_reverse_pred"], f["v_between_reverse_pred"],
            ),
        }

    n_steps = steps.size
    L = next(iter(ratios.values()))[0].shape[-1]

    # Build (n_steps, L) ratio arrays, averaged across seeds.
    def _ratio_avg(w_arr, b_arr):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b_arr > 0, w_arr / b_arr, np.nan)
        return np.nanmean(r, axis=0)  # average across seeds -> (n_steps, L)

    panels = [(name, _ratio_avg(w, b)) for name, (w, b) in ratios.items()]

    # Color scale: cover the full data range, symmetric in log around 1.
    all_finite = np.concatenate([p[1][np.isfinite(p[1])] for p in panels])
    all_pos = all_finite[all_finite > 0]
    if vmin_vmax is None:
        # Pad the range slightly to keep extremes visible.
        ratio_max = max(np.nanmax(p[1]) for p in panels)
        # Lower bound: small positive value (the forward bundle is near
        # zero at t=0; we want the entire range visible).
        ratio_min = max(np.nanmin(all_pos), 0.02)
        # Symmetric in log around 1.
        log_dist = max(abs(np.log10(ratio_min)), abs(np.log10(ratio_max)))
        vmin = 10 ** (-log_dist)
        vmax = 10 ** (log_dist)
    else:
        vmin, vmax = vmin_vmax

    # SymLogNorm with linthresh near 1 makes the "within = between"
    # boundary the visual midpoint of the colormap.
    norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10)
    cmap = "RdBu_r"

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, (title, r) in zip(axes, panels):
        # imshow expects (rows, cols) = (steps, layers); we want low
        # steps at bottom, so origin='lower'.
        # We use pcolormesh with explicit step axis to get log-spaced
        # y ticks; imshow with log y is awkward.
        layers = np.arange(L + 1) - 0.5  # cell edges
        # Step edges: log-midpoints between consecutive steps, with
        # extrapolation at the ends.
        log_steps = np.log10(steps.astype(np.float64))
        edges = np.empty(n_steps + 1)
        edges[1:-1] = (log_steps[:-1] + log_steps[1:]) / 2
        edges[0] = log_steps[0] - (log_steps[1] - log_steps[0]) / 2
        edges[-1] = log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2
        step_edges = 10 ** edges

        im = ax.pcolormesh(layers, step_edges, r, cmap=cmap, norm=norm,
                           shading="flat")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(L))
        ax.set_xlabel("layer state index t")
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_title(title)
        # Light-gray annotations of key landmarks.
        # The forward crossover at t≈2 is the most striking landmark
        # in the heatmap; mark it on the forward panel only.
        if "Forward" in title:
            ax.axvline(2, color="k", ls=":", lw=0.6, alpha=0.6)

    axes[0].set_ylabel("training step")

    # Shared colorbar on the right.
    cbar = fig.colorbar(im, ax=axes, location="right", pad=0.02,
                        shrink=0.85, aspect=25)
    cbar.set_label("within / between variance ratio")
    # Annotate the colorbar at ratio = 1 (within = between).
    cbar.ax.axhline(1.0, color="k", lw=1.0)

    fig.suptitle("Within/between variance ratio across training and depth "
                 "(cross-seed mean)")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_heatmap_delta(
    decomposition_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (16, 6),
) -> None:
    """Heatmap of the *change* in within/between ratio relative to the
    first checkpoint, per view.

    Companion to `plot_ratio_heatmap`. The raw heatmap shows the
    absolute ratio across (step, layer); this one shows
    log10(ratio(step, layer) / ratio(step_0, layer)). It emphasizes
    *training-dynamic* features: where in the (step, layer) plane the
    ratio is growing or shrinking through training, not just its
    absolute magnitude.

    This is the layer-resolved version of the co-location finding:
    if the reverse-view bulge intensifies in the same training window
    where the all-to-all log_alpha hump occurs, this plot makes that
    intensification visible as a localized hot spot.

    Args:
        decomposition_path: trajectories/decomposition.npz.
        output_path: where to save the PNG.
    """
    from matplotlib.colors import Normalize

    with np.load(decomposition_path) as f:
        steps = f["steps"]
        ratios = {
            "Forward (input-conditioned)": (
                f["v_within_forward"], f["v_between_forward"],
            ),
            "Reverse (actual successor)": (
                f["v_within_reverse_actual"], f["v_between_reverse_actual"],
            ),
            "Reverse (predicted successor)": (
                f["v_within_reverse_pred"], f["v_between_reverse_pred"],
            ),
        }
    n_steps = steps.size
    L = next(iter(ratios.values()))[0].shape[-1]

    def _delta(w_arr, b_arr):
        """Return log10(ratio / ratio_at_step0), averaged across seeds."""
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b_arr > 0, w_arr / b_arr, np.nan)
        r_mean = np.nanmean(r, axis=0)  # (n_steps, L)
        # Reference: the first checkpoint's profile. Replace 0/small
        # values with NaN so they propagate.
        r0 = r_mean[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(r_mean / np.where(r0 > 1e-6, r0, np.nan))

    panels = [(name, _delta(w, b)) for name, (w, b) in ratios.items()]

    # Symmetric color range around 0 (log of unity).
    abs_max = max(np.nanmax(np.abs(p[1])) for p in panels)
    vmin, vmax = -abs_max, abs_max
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for ax, (title, d) in zip(axes, panels):
        layers = np.arange(L + 1) - 0.5
        log_steps = np.log10(steps.astype(np.float64))
        edges = np.empty(n_steps + 1)
        edges[1:-1] = (log_steps[:-1] + log_steps[1:]) / 2
        edges[0] = log_steps[0] - (log_steps[1] - log_steps[0]) / 2
        edges[-1] = log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2
        step_edges = 10 ** edges

        im = ax.pcolormesh(layers, step_edges, d, cmap="RdBu_r", norm=norm,
                           shading="flat")
        ax.set_yscale("log")
        ax.set_xticks(np.arange(L))
        ax.set_xlabel("layer state index t")
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_title(title)

    axes[0].set_ylabel("training step")
    cbar = fig.colorbar(im, ax=axes, location="right", pad=0.02,
                        shrink=0.85, aspect=25)
    cbar.set_label(r"$\log_{10}$(ratio / ratio at first checkpoint)")
    cbar.ax.axhline(0.0, color="k", lw=1.0)

    fig.suptitle("Change in within/between ratio through training, "
                 "relative to first checkpoint")
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_heatmap_with_init(
    decomposition_path: str,
    init_baseline_path: str,
    output_path: str,
    figsize: Tuple[float, float] = (16, 6.6),
) -> None:
    """Augmented version of plot_ratio_heatmap with a 'random init'
    baseline shown as a separate visually-distinct band below the
    trained heatmap.

    The init band is rendered as a single horizontal strip below the
    main panel with a small gap, sharing the colorbar so the relative
    magnitudes are visually comparable to the trained checkpoints.
    A horizontal divider line and an "init" y-axis label make the
    discontinuity explicit.

    The visual claim: at random init, the within/between ratio is high
    and approximately monotone with depth (no forward crossover, no
    reverse mid-network bulge in its trained form). Training reshapes
    this to the structured profile shown above, with the layer-localized
    bulge surviving as a residue.

    Args:
        decomposition_path: trajectories/decomposition.npz (trained data).
        init_baseline_path: init_baseline/init_seedNNNN.npz produced by
            init_check.py.
        output_path: where to save the PNG.
        figsize: matplotlib figure size.
    """
    from matplotlib.colors import SymLogNorm
    from matplotlib.gridspec import GridSpec

    # Trained data: shape (n_seeds, n_steps, L).
    with np.load(decomposition_path) as f:
        steps = f["steps"]
        ratios = {
            "Forward (input-conditioned)": (
                f["v_within_forward"], f["v_between_forward"],
            ),
            "Reverse (actual successor)": (
                f["v_within_reverse_actual"], f["v_between_reverse_actual"],
            ),
            "Reverse (predicted successor)": (
                f["v_within_reverse_pred"], f["v_between_reverse_pred"],
            ),
        }

    # Init data: shape (L,) per view.
    with np.load(init_baseline_path) as f:
        init_seed = int(f["seed"])
        init = {
            "Forward (input-conditioned)": (
                f["v_within_forward"], f["v_between_forward"],
            ),
            "Reverse (actual successor)": (
                f["v_within_reverse_actual"], f["v_between_reverse_actual"],
            ),
            "Reverse (predicted successor)": (
                f["v_within_reverse_pred"], f["v_between_reverse_pred"],
            ),
        }

    n_steps = steps.size
    L = next(iter(ratios.values()))[0].shape[-1]

    def _ratio_avg(w_arr, b_arr):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b_arr > 0, w_arr / b_arr, np.nan)
        return np.nanmean(r, axis=0)

    def _ratio_1d(w, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, w / b, np.nan)

    trained_panels = [(name, _ratio_avg(w, b)) for name, (w, b) in ratios.items()]
    init_panels = [(name, _ratio_1d(w, b)) for name, (w, b) in init.items()]

    # Color scale: cover the union of trained and init ratios.
    all_finite = []
    for _, r in trained_panels:
        all_finite.append(r[np.isfinite(r)])
    for _, r in init_panels:
        all_finite.append(r[np.isfinite(r)])
    all_finite = np.concatenate(all_finite)
    all_pos = all_finite[all_finite > 0]
    ratio_max = float(np.max(all_pos))
    ratio_min = float(max(np.min(all_pos), 0.02))
    log_dist = max(abs(np.log10(ratio_min)), abs(np.log10(ratio_max)))
    vmin = 10 ** (-log_dist)
    vmax = 10 ** (log_dist)
    norm = SymLogNorm(linthresh=1.0, vmin=vmin, vmax=vmax, base=10)
    cmap = "RdBu_r"

    # Layout: 3 columns (one per view), 2 rows (trained heatmap, init strip).
    # The init strip is much shorter than the trained heatmap.
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig,
                  width_ratios=[1, 1, 1, 0.04],
                  height_ratios=[10, 1.0],
                  hspace=0.18, wspace=0.05)

    log_steps = np.log10(steps.astype(np.float64))
    edges = np.empty(n_steps + 1)
    edges[1:-1] = (log_steps[:-1] + log_steps[1:]) / 2
    edges[0] = log_steps[0] - (log_steps[1] - log_steps[0]) / 2
    edges[-1] = log_steps[-1] + (log_steps[-1] - log_steps[-2]) / 2
    step_edges = 10 ** edges
    layers_edges = np.arange(L + 1) - 0.5

    last_im = None
    for col, ((title, r_trained), (_, r_init)) in enumerate(
            zip(trained_panels, init_panels)):
        ax_main = fig.add_subplot(gs[0, col])
        ax_init = fig.add_subplot(gs[1, col])

        # Trained panel.
        last_im = ax_main.pcolormesh(layers_edges, step_edges, r_trained,
                                     cmap=cmap, norm=norm, shading="flat")
        ax_main.set_yscale("log")
        ax_main.set_xticks(np.arange(L))
        ax_main.set_xticklabels([])  # x-tick labels on init strip only
        ax_main.set_xlim(-0.5, L - 0.5)
        ax_main.set_title(title)
        if "Forward" in title:
            ax_main.axvline(2, color="k", ls=":", lw=0.6, alpha=0.6)
        # Suppress y-tick labels on non-leftmost panels.
        if col > 0:
            ax_main.set_yticklabels([])
            ax_main.tick_params(axis="y", which="both", length=0)

        # Init strip: single row, plotted as a 1-row pcolormesh.
        init_row = r_init[None, :]  # (1, L)
        # Use simple linear y for the init strip; we'll relabel.
        init_y_edges = np.array([0, 1])
        ax_init.pcolormesh(layers_edges, init_y_edges, init_row,
                           cmap=cmap, norm=norm, shading="flat")
        ax_init.set_xticks(np.arange(L))
        ax_init.set_xlabel("layer state index t")
        ax_init.set_xlim(-0.5, L - 0.5)
        ax_init.set_yticks([0.5])
        if col == 0:
            ax_init.set_yticklabels([f"init\n(seed {init_seed})"], fontsize=8.5)
        else:
            ax_init.set_yticklabels([])
            ax_init.tick_params(axis="y", which="both", length=0)

    # Left-column y-axis: "training step" on main panel.
    fig.axes[0].set_ylabel("training step")

    # Shared colorbar.
    cax = fig.add_subplot(gs[:, 3])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_label("within / between variance ratio")
    cbar.ax.axhline(1.0, color="k", lw=1.0)

    fig.suptitle("Within/between variance ratio across training and depth, "
                 "with random-init baseline\n"
                 "(top: cross-seed mean over training; bottom: random "
                 f"init seed {init_seed}, no training)",
                 y=0.995)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Helper: cross-seed agreement summary.
# ----------------------------------------------------------------------
def summarize_cross_seed_agreement(
    crossover_path: str,
    final_step_only: bool = True,
) -> Dict:
    """
    Compute mean and std across seeds of crossover layer at the final
    checkpoint (or averaged across all checkpoints).

    Returns a dict mapping view to (mean, std) tuple. Used for
    hypothesis S1 / F3 reporting.
    """
    data = np.load(crossover_path)
    steps = data["steps"]
    arrs = {
        "forward": data["crossover_forward"],
        "reverse_actual": data["crossover_reverse_actual"],
        "reverse_pred": data["crossover_reverse_pred"],
    }
    out = {}
    if final_step_only:
        for k, arr in arrs.items():
            final = arr[:, -1]
            out[k] = {
                "mean": float(np.nanmean(final)),
                "std": float(np.nanstd(final, ddof=1)) if np.sum(~np.isnan(final)) > 1 else 0.0,
                "n_seeds_valid": int(np.sum(~np.isnan(final))),
                "final_step": int(steps[-1]),
            }
    else:
        for k, arr in arrs.items():
            out[k] = {
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr, ddof=1)),
                "n_total_valid": int(np.sum(~np.isnan(arr))),
            }
    return out
    