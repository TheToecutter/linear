"""
Plotting for the reverse build-up project.

Produces the six figures called out in REVERSE_BUILDUP_PROPOSAL.md §4:

  Fig 4.1: Per-view discriminator dashboard
           -- side-by-side D1 (cv_trace) and D4a (per-cell kurtosis)
              across all three views, training-step trajectory.

  Fig 4.2: Reverse Mardia Z depth profile
           -- layer on x-axis, Mardia Z averaged across cells on y-axis,
              with one curve per seed and the forward profile overlaid.

  Fig 4.3: Kurtosis reconstruction comparison (the headline plot)
           -- four-bar comparison per layer:
                empirical | fwd Model-B | rev-act Model-B | rev-pred Model-B
              with shuffle-null reference bars overlaid in lighter color.

  Fig 4.4: Unembedding-subspace decomposition
           -- 2x2 panel: marginal kurt par vs perp, sweep over d_par,
              plus three representative cells.

  Fig 4.5: Reverse lambda^contract distribution (handled by
           reverse_lambda_clusters.py figure 'd_n2_*').

  Fig 4.6: Co-location summary table (text/CSV, no figure).

Usage:
    python reverse_buildup_plots.py --run-dir ../phase1_runs_gelu

Output files written to multiview/model_abc/figures/:
    reverse_buildup_fig_4_1_dashboard.png
    reverse_buildup_fig_4_2_mardia.png
    reverse_buildup_fig_4_3_reconstruction.png
    reverse_buildup_fig_4_4_subspace.png
    reverse_buildup_fig_4_6_colocation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reverse_buildup import VIEWS, output_root, figures_dir


# ----------------------------------------------------------------------
# Loaders.
# ----------------------------------------------------------------------
def _load_npz(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _load_d1(run_dir: str, view: str) -> Optional[Dict]:
    return _load_npz(os.path.join(output_root(run_dir),
                                  f"d1_token_cv_{view}.npz"))


def _load_d3(run_dir: str, view: str) -> Optional[Dict]:
    return _load_npz(os.path.join(output_root(run_dir),
                                  f"d3_per_token_fits_{view}.npz"))


def _load_d4a(run_dir: str, view: str) -> Optional[Dict]:
    return _load_npz(os.path.join(output_root(run_dir),
                                  f"d4a_kurtosis_{view}.npz"))


def _load_d4b_final(run_dir: str, view: str,
                    seeds: List[int]) -> List[Dict]:
    out = []
    subdir = os.path.join(output_root(run_dir), f"d4b_gaussianity_{view}")
    if not os.path.isdir(subdir):
        # Legacy fallback: pre-refactor forward outputs live in
        # d4b_gaussianity/ without the view suffix.
        if view == "forward":
            legacy = os.path.join(output_root(run_dir), "d4b_gaussianity")
            if os.path.isdir(legacy):
                subdir = legacy
            else:
                return out
        else:
            return out
    for fn in sorted(os.listdir(subdir)):
        if not fn.endswith(".npz"):
            continue
        path = os.path.join(subdir, fn)
        with np.load(path, allow_pickle=False) as f:
            out.append({k: f[k] for k in f.files})
    return out


def _load_d5_for_view(run_dir: str, view: str,
                       shuffled: bool = False) -> List[Dict]:
    suffix = "_shuffled" if shuffled else ""
    subdir = os.path.join(output_root(run_dir),
                          f"d5_reconstruction_{view}{suffix}")
    out = []
    if not os.path.isdir(subdir):
        # Legacy fallback: forward view's pre-refactor outputs live in
        # d5_reconstruction/ without the view suffix.
        if view == "forward" and not shuffled:
            legacy = os.path.join(output_root(run_dir), "d5_reconstruction")
            if os.path.isdir(legacy):
                subdir = legacy
            else:
                return out
        else:
            return out
    for fn in sorted(os.listdir(subdir)):
        if not fn.endswith(".npz"):
            continue
        with np.load(os.path.join(subdir, fn), allow_pickle=False) as f:
            out.append({k: f[k] for k in f.files})
    return out


def _load_subspace_results(run_dir: str, view: str = "reverse_actual") -> List[Dict]:
    subdir = os.path.join(output_root(run_dir),
                          f"d_n1_unembedding_subspace_{view}")
    out = []
    if not os.path.isdir(subdir):
        return out
    for fn in sorted(os.listdir(subdir)):
        if not fn.endswith(".npz"):
            continue
        with np.load(os.path.join(subdir, fn), allow_pickle=False) as f:
            out.append({k: f[k] for k in f.files})
    return out


# ----------------------------------------------------------------------
# Fig 4.1: per-view dashboard.
# ----------------------------------------------------------------------
def plot_fig_4_1(run_dir: str) -> None:
    """Side-by-side D1 cv_trace and D4a per-cell kurtosis across all
    three views, plotted as training-step trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey="row")
    colors_view = {
        "forward":        "#2ca02c",
        "reverse_actual": "#1f77b4",
        "reverse_pred":   "#ff7f0e",
    }

    for vi, view in enumerate(VIEWS):
        d1 = _load_d1(run_dir, view)
        d4a = _load_d4a(run_dir, view)

        # Top row: D1 cv_trace, layer-mean (interior), through training.
        ax = axes[0, vi]
        if d1 is not None:
            cv = d1["cv_trace"]                          # (S, T, L)
            steps = d1["steps"]
            L = cv.shape[-1]
            interior = slice(1, max(2, L - 1))
            cv_layer_mean = np.nanmean(cv[:, :, interior], axis=(0, 2))  # (T,)
            ax.plot(steps, cv_layer_mean, color=colors_view[view], lw=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("training step")
            ax.set_ylabel("D1 CV(trace) interior mean")
        ax.set_title(f"D1 — {view}")

        # Bottom row: D4a per-cell kurtosis, cross-cell mean of an
        # interior layer (layer L//2), through training.
        ax = axes[1, vi]
        if d4a is not None:
            kurt = d4a["kurtosis_per_token"]              # (S, T, K, L)
            steps = d4a["steps"]
            L = kurt.shape[-1]
            mid = L // 2
            k_mid = np.nanmean(kurt[:, :, :, mid], axis=(0, 2))  # (T,)
            ax.plot(steps, k_mid, color=colors_view[view], lw=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("training step")
            ax.set_ylabel(f"D4a kurt @ layer {mid}, mean over cells")
        ax.set_title(f"D4a — {view}")

    plt.suptitle("Reverse build-up dashboard: D1 and D4a across all three views",
                 y=1.02)
    plt.tight_layout()
    out_path = os.path.join(figures_dir(run_dir),
                            "reverse_buildup_fig_4_1_dashboard.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


# ----------------------------------------------------------------------
# Fig 4.2: reverse Mardia Z depth profile.
# ----------------------------------------------------------------------
def plot_fig_4_2(run_dir: str, seeds: List[int]) -> None:
    """Mardia Z averaged across cells, layer on x-axis, one curve per
    seed for reverse_actual; forward profile overlaid for comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax_idx, view in enumerate(("reverse_actual", "reverse_pred")):
        ax = axes[ax_idx]
        # Reverse view per seed.
        rev_data = _load_d4b_final(run_dir, view, seeds)
        for r in rev_data:
            Z = np.nanmean(r["mardia_z"], axis=0)        # (L,)
            ax.plot(np.arange(Z.size), Z, lw=1.2, alpha=0.8,
                    label=f"seed {int(r['seed'])}")
        # Forward overlay.
        fwd_data = _load_d4b_final(run_dir, "forward", seeds)
        if fwd_data:
            Z_fwd = np.stack([np.nanmean(r["mardia_z"], axis=0)
                              for r in fwd_data], axis=0)
            Z_fwd_mean = np.nanmean(Z_fwd, axis=0)
            ax.plot(np.arange(Z_fwd_mean.size), Z_fwd_mean,
                    color="black", ls="--", lw=2.0,
                    label="forward (seed-mean, reference)")
        ax.axhline(0.0, color="gray", ls=":", lw=0.7)
        ax.axhline(2.0, color="red", ls=":", lw=0.7,
                   label="5% rejection threshold")
        ax.set_xlabel("layer index")
        ax.set_ylabel("Mardia Z (cell-mean)")
        ax.set_title(f"Mardia Z depth profile — {view}")
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_path = os.path.join(figures_dir(run_dir),
                            "reverse_buildup_fig_4_2_mardia.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


# ----------------------------------------------------------------------
# Fig 4.3: kurtosis reconstruction comparison.
# ----------------------------------------------------------------------
def plot_fig_4_3(run_dir: str) -> None:
    """Four-bar comparison per layer:
      empirical | fwd Model-B | rev-act Model-B | rev-pred Model-B
    with shuffle-null reference bars in lighter color (where available).
    """
    # Use the seed-0 final-checkpoint result by default.
    def _final_seed0(view, shuffled=False):
        results = _load_d5_for_view(run_dir, view, shuffled=shuffled)
        if not results:
            return None
        # Pick the one with seed=0 and the largest step.
        candidates = [r for r in results if int(r["seed"]) == 0]
        if not candidates:
            candidates = results
        return max(candidates, key=lambda r: int(r["step"]))

    fwd = _final_seed0("forward")
    rev_act = _final_seed0("reverse_actual")
    rev_prd = _final_seed0("reverse_pred")
    rev_act_shuf = _final_seed0("reverse_actual", shuffled=True)
    rev_prd_shuf = _final_seed0("reverse_pred", shuffled=True)

    if fwd is None or rev_act is None or rev_prd is None:
        print("[plot] Fig 4.3 needs all three views' D5 outputs; skipping.")
        return

    L = fwd["empirical_kurt"].size
    emp = fwd["empirical_kurt"]                          # marginal is same
    fwd_B = fwd["recon_B_kurt"]
    rev_act_B = rev_act["recon_B_kurt"]
    rev_prd_B = rev_prd["recon_B_kurt"]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(L)
    width = 0.20
    ax.bar(x - 1.5 * width, emp, width=width, color="black",
           label="empirical marginal")
    ax.bar(x - 0.5 * width, fwd_B, width=width, color="#2ca02c",
           label="forward Model-B")
    ax.bar(x + 0.5 * width, rev_act_B, width=width, color="#1f77b4",
           label="reverse-actual Model-B")
    ax.bar(x + 1.5 * width, rev_prd_B, width=width, color="#ff7f0e",
           label="reverse-pred Model-B")

    # Overlay shuffle-null markers as horizontal ticks where available.
    if rev_act_shuf is not None:
        ax.plot(x + 0.5 * width, rev_act_shuf["recon_B_kurt"],
                "x", color="#1f77b4", alpha=0.6, ms=6,
                label="reverse-actual null")
    if rev_prd_shuf is not None:
        ax.plot(x + 1.5 * width, rev_prd_shuf["recon_B_kurt"],
                "x", color="#ff7f0e", alpha=0.6, ms=6,
                label="reverse-pred null")

    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax.set_xticks(x)
    ax.set_xlabel("layer index")
    ax.set_ylabel("per-coordinate excess kurtosis")
    ax.set_title("Reconstruction comparison (final checkpoint, seed 0)")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    plt.tight_layout()
    out_path = os.path.join(figures_dir(run_dir),
                            "reverse_buildup_fig_4_3_reconstruction.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


# ----------------------------------------------------------------------
# Fig 4.4: unembedding-subspace decomposition.
# ----------------------------------------------------------------------
def plot_fig_4_4(run_dir: str, view: str = "reverse_actual") -> None:
    """2x2 panel:
       (0,0) marginal kurt par vs perp, across d_par sweep, final layer
       (0,1) marginal kurt across depth, one curve per d_par (par only)
       (1,0) same as (0,1) but perp only
       (1,1) gap par - perp across depth, one curve per d_par
    """
    results = _load_subspace_results(run_dir, view=view)
    if not results:
        print(f"[plot] No subspace results found for {view}; skipping Fig 4.4.")
        return

    # Use seed 0 if present, else first.
    seed_groups = {}
    for r in results:
        seed_groups.setdefault(int(r["seed"]), []).append(r)
    target_seed = 0 if 0 in seed_groups else min(seed_groups)
    group = sorted(seed_groups[target_seed], key=lambda r: int(r["d_parallel"]))
    if not group:
        return

    d_pars = [int(r["d_parallel"]) for r in group]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.8, len(d_pars)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) bar comparison at final layer.
    ax = axes[0, 0]
    width = 0.35
    x = np.arange(len(d_pars))
    par_final = np.array([r["marg_kurt_par"][-1] for r in group])
    perp_final = np.array([r["marg_kurt_perp"][-1] for r in group])
    ax.bar(x - width / 2, par_final, width=width,
           color="#1f77b4", label="parallel (in W_U rowspan)")
    ax.bar(x + width / 2, perp_final, width=width,
           color="#ff7f0e", label="perpendicular (orthogonal complement)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in d_pars])
    ax.set_xlabel("d_parallel")
    ax.set_ylabel("excess kurtosis (final layer)")
    ax.set_title("Final-layer kurtosis: par vs perp")
    ax.legend(fontsize=9)
    ax.axhline(0.0, color="gray", ls=":", lw=0.7)

    # (0,1) parallel kurt across depth.
    ax = axes[0, 1]
    for ci, r in enumerate(group):
        ax.plot(r["layers"], r["marg_kurt_par"], color=cmap[ci],
                lw=1.5, label=f"d_par={d_pars[ci]}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("parallel kurt (marginal)")
    ax.set_title("Parallel-component kurtosis across depth")
    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax.legend(fontsize=8)

    # (1,0) perpendicular kurt across depth.
    ax = axes[1, 0]
    for ci, r in enumerate(group):
        ax.plot(r["layers"], r["marg_kurt_perp"], color=cmap[ci],
                lw=1.5, label=f"d_par={d_pars[ci]}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("perpendicular kurt (marginal)")
    ax.set_title("Perpendicular-component kurtosis across depth")
    ax.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax.legend(fontsize=8)

    # (1,1) gap.
    ax = axes[1, 1]
    for ci, r in enumerate(group):
        ax.plot(r["layers"], r["gap_par_minus_perp"], color=cmap[ci],
                lw=1.5, label=f"d_par={d_pars[ci]}")
    ax.set_xlabel("layer index")
    ax.set_ylabel("gap = par - perp")
    ax.set_title("Gap: negative = parallel more Gaussian than perp (N1 prediction)")
    ax.axhline(0.0, color="black", ls="--", lw=0.8)
    ax.legend(fontsize=8)

    plt.suptitle(f"Unembedding-subspace decomposition (view={view}, "
                 f"seed={target_seed})", y=1.00)
    plt.tight_layout()
    out_path = os.path.join(figures_dir(run_dir),
                            f"reverse_buildup_fig_4_4_subspace_{view}.png")
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {out_path}")


# ----------------------------------------------------------------------
# Fig 4.6: co-location table.
# ----------------------------------------------------------------------
def write_fig_4_6_table(run_dir: str) -> None:
    """Emit a CSV listing reverse-view training events alongside the
    Phase 1 training-dynamic anomaly steps for visual co-location
    checking. The output is intentionally short: each row is a
    (event_name, step) pair. Phase 1 events that are documented in
    INVESTIGATION_WRITEUP.md are hardcoded here; reverse events come
    from the saved D4a (kurtosis emergence) and Phase B/C verdicts.
    """
    rows = []
    # Phase 1 events as documented in INVESTIGATION_WRITEUP.md.
    rows.append({
        "event": "Phase 1: post-final-norm anomaly emergence",
        "step_range": "~400-2000",
        "source": "phase 1",
    })
    rows.append({
        "event": "Phase 1: log_alpha hump peak",
        "step_range": "~2000-4000",
        "source": "phase 1",
    })
    rows.append({
        "event": "Phase 1: Sigma-distance bump",
        "step_range": "~3000-6000",
        "source": "phase 1",
    })
    rows.append({
        "event": "Phase 1: late kurtosis rise",
        "step_range": "~5000-20000",
        "source": "phase 1",
    })
    rows.append({
        "event": "Forward Mardia Z rise threshold (Z>10), cells-mean",
        "step_range": "~2000-10000 (from INVESTIGATION §6)",
        "source": "phase 1 / forward",
    })

    # Reverse-view event: emergence of D4a kurtosis crossing a threshold.
    for view in ("reverse_actual", "reverse_pred"):
        d4a = _load_d4a(run_dir, view)
        if d4a is None:
            rows.append({
                "event": f"Reverse D4a kurtosis-rise step ({view})",
                "step_range": "n/a (no data)",
                "source": "phase A",
            })
            continue
        # Cross-seed-cross-cell mean kurtosis at an interior layer.
        k = d4a["kurtosis_per_token"]                  # (S, T, K, L)
        steps = d4a["steps"]
        L = k.shape[-1]
        mid = L // 2
        prof = np.nanmean(k[:, :, :, mid], axis=(0, 2))  # (T,)
        # First step where prof > 1.0 (matching the forward threshold
        # documented in INVESTIGATION §6).
        cross = None
        for i, v in enumerate(prof):
            if np.isfinite(v) and v > 1.0:
                cross = int(steps[i])
                break
        rows.append({
            "event": f"Reverse D4a interior-layer kurtosis > 1.0 ({view})",
            "step_range": str(cross) if cross is not None else "no crossing",
            "source": "phase A",
        })

    # Phase B/C/D verdicts.
    for phase, fn in [("B", "reverse_buildup_phase_b_verdict.json"),
                       ("C", "reverse_buildup_phase_c_verdict.json"),
                       ("D", "reverse_buildup_phase_d_verdict.json")]:
        path = os.path.join(output_root(run_dir), fn)
        if os.path.exists(path):
            with open(path, "r") as f:
                v = json.load(f)
            rows.append({
                "event": f"Phase {phase} verdict",
                "step_range": json.dumps(v, default=str)[:200],
                "source": f"phase {phase}",
            })

    out_path = os.path.join(figures_dir(run_dir),
                            "reverse_buildup_fig_4_6_colocation.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["event", "step_range", "source"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[plot] -> {out_path}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seed list; default: auto-discover.")
    args = p.parse_args()

    if args.seeds is None:
        from multiview_campaign import seeds_in_run
        seeds = seeds_in_run(args.run_dir)
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    plot_fig_4_1(args.run_dir)
    plot_fig_4_2(args.run_dir, seeds)
    plot_fig_4_3(args.run_dir)
    plot_fig_4_4(args.run_dir, view="reverse_actual")
    plot_fig_4_4(args.run_dir, view="reverse_pred")
    write_fig_4_6_table(args.run_dir)


if __name__ == "__main__":
    main()
    