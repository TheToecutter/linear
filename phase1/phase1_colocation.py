"""
Phase 1 anomaly co-location with the conditional dynamics non-monotonicity.

Question: D3's right panel showed that CV(lambda) across forward-set tokens
is non-monotonic through training, with a minimum near step 1500-2000 and
a rise back through step 24000. The Phase 1 writeup documents five
training-time anomalies in the marginal dynamics:

  1. log_alpha hump, peak in steps [4500, 5050], height [-2.20, -2.03]
  2. Post-final-norm anomaly emergence, ~step 400 -> ~step 5000,
     plateaus at -1.8 log units below the inner-layer fit
  3. Late-training kurtosis rise, ~0.35 at step 2000 to ~1.05 at
     step 24000
  4. Mid-training Sigma-distance bump, peak in steps 5000-10000,
     reaching ~0.12 normalized
  5. Boundary anomaly emergence, half-magnitude by step 2000, plateaus
     by step 5000

This script:

  1. Loads CV(lambda)(t) per seed from d3_per_token_fits.npz.
  2. Loads the Phase 1 per-checkpoint flow files
     (run_dir/seed_*/flow_analysis/flow_step_*.npz) and extracts the
     trajectory of marginal-dynamics statistics: log_alpha, mean
     kurtosis, all-to-all lambda, effective rank profile.
  3. Computes derivative trajectories: the post-final-norm anomaly
     (difference between the last-layer log-variance and the inner-layer
     fit), the Sigma-distance to convergence (Frobenius distance from
     R(t) at step k to R(t) at the final step).
  4. Plots all trajectories on a shared log-x axis with documented
     anomaly windows shaded.
  5. Reports quantitative co-location: at the step of CV(lambda)
     minimum, what are each Phase 1 anomaly's values?

If Phase 1 flow files are not found, falls back to overlaying only the
documented anomaly windows (vertical shaded bands) on the CV(lambda)
trajectory — this is the minimal version of the analysis.

Output:
    run_dir/multiview/model_abc/d8_phase1_colocation.json
    run_dir/multiview/model_abc/figures/d8_phase1_colocation.png

Usage:
    python phase1_colocation.py --run-dir ../phase1_runs_gelu
    python phase1_colocation.py --run-dir ../phase1_runs_gelu --no-phase1-data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Documented Phase 1 anomaly windows. Source: PHASE_1_WRITEUP.md §6.6.
# ----------------------------------------------------------------------
PHASE1_ANOMALIES = {
    "log_alpha_hump": {
        "label": "log_alpha hump (peak)",
        "range": (4500, 5050),
        "color": "C1",
    },
    "post_final_norm_emergence": {
        "label": "post-final-norm anomaly emergence",
        "range": (400, 5000),
        "color": "C2",
    },
    "kurtosis_rise_start": {
        "label": "late kurtosis rise (start)",
        "range": (2000, 2400),
        "color": "C3",
    },
    "sigma_distance_bump": {
        "label": "Sigma-distance bump",
        "range": (5000, 10000),
        "color": "C4",
    },
    "boundary_anomaly_plateau": {
        "label": "boundary anomaly plateau",
        "range": (5000, 5500),
        "color": "C5",
    },
}


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------
def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


def phase1_flow_dir(run_dir: str, seed: int) -> str:
    return os.path.join(run_dir, f"seed_{seed}", "flow_analysis")


# ----------------------------------------------------------------------
# Phase 1 flow file loading.
# ----------------------------------------------------------------------
def load_phase1_trajectories(
    run_dir: str, seeds: List[int],
) -> Optional[Dict]:
    """Load per-checkpoint Phase 1 flow files for each seed and stack
    scalar trajectories.

    Returns a dict with keys:
      steps: (n_steps,) common across seeds
      log_alpha:   (n_seeds, n_steps)         -- our convention
      log_alpha_paper:   (n_seeds, n_steps)   -- paper convention
      lambda_all:  (n_seeds, n_steps)         -- all-to-all lambda
      mean_kurtosis: (n_seeds, n_steps)       -- mean over layers
      effective_rank_max: (n_seeds, n_steps)  -- max over layers
      sigma_distance_to_final: (n_seeds, n_steps)  -- Frobenius distance
                                                    from R(t) at step k
                                                    to R(t) at final step,
                                                    averaged over layers and
                                                    normalized by sqrt(2*L*H)
      post_final_norm_residual: (n_seeds, n_steps) -- difference between
                                                      log-variance at the
                                                      last layer and the
                                                      linear fit through the
                                                      inner layers

    If no Phase 1 flows are found, returns None.
    """
    per_seed: Dict[int, Dict[int, str]] = {}
    for s in seeds:
        d = phase1_flow_dir(run_dir, s)
        if not os.path.isdir(d):
            continue
        files = glob.glob(os.path.join(d, "flow_step_*.npz"))
        if not files:
            continue
        m = {}
        for f in files:
            match = re.search(r"flow_step_(\d+)\.npz$", f)
            if match:
                m[int(match.group(1))] = f
        if m:
            per_seed[s] = m

    if not per_seed:
        print("[phase1] No Phase 1 flow files found on disk; "
              "will overlay anomaly windows only.")
        return None

    # Common steps across all available seeds.
    step_sets = [set(m.keys()) for m in per_seed.values()]
    common = sorted(set.intersection(*step_sets)) if step_sets else []
    if not common:
        print("[phase1] No common steps across Phase 1 seeds.")
        return None
    seeds_present = sorted(per_seed.keys())
    print(f"[phase1] Loaded Phase 1 flows for seeds {seeds_present}, "
          f"{len(common)} common steps (range {common[0]}..{common[-1]})")

    # Probe L from first available.
    sample_f = next(iter(per_seed[seeds_present[0]].values()))
    with np.load(sample_f) as zf:
        L = zf["effective_rank"].shape[0] if "effective_rank" in zf.files \
            else None

    if L is None:
        print("[phase1] Could not determine layer count from flows.")
        return None

    n_s = len(seeds_present)
    n_t = len(common)
    log_alpha = np.full((n_s, n_t), np.nan)
    log_alpha_paper = np.full((n_s, n_t), np.nan)
    lam_all = np.full((n_s, n_t), np.nan)
    lam_all_paper = np.full((n_s, n_t), np.nan)
    mean_kurt = np.full((n_s, n_t), np.nan)
    erank_max = np.full((n_s, n_t), np.nan)
    sigma_dist = np.full((n_s, n_t), np.nan)
    pfn_resid = np.full((n_s, n_t), np.nan)

    # Cache final-step singular values for each seed (for Sigma distance).
    singular_values_final = {}
    for si, s in enumerate(seeds_present):
        files = per_seed[s]
        final_step = max(files.keys())
        try:
            with np.load(files[final_step]) as zf:
                if "singular_values" in zf.files:
                    singular_values_final[s] = zf["singular_values"].astype(np.float64)
        except Exception:
            pass

    for si, s in enumerate(seeds_present):
        files = per_seed[s]
        for ti, step in enumerate(common):
            try:
                with np.load(files[step]) as zf:
                    if "log_alpha" in zf.files:
                        log_alpha[si, ti] = float(zf["log_alpha"])
                    if "log_alpha_paper" in zf.files:
                        log_alpha_paper[si, ti] = float(zf["log_alpha_paper"])
                    if "lambda" in zf.files:
                        lam_all[si, ti] = float(zf["lambda"])
                    if "lambda_paper" in zf.files:
                        lam_all_paper[si, ti] = float(zf["lambda_paper"])
                    if "kurtosis_per_layer" in zf.files:
                        mean_kurt[si, ti] = float(np.nanmean(
                            zf["kurtosis_per_layer"]))
                    if "effective_rank" in zf.files:
                        erank_max[si, ti] = float(np.nanmax(
                            zf["effective_rank"]))

                    # Basis-invariant distance to final-checkpoint flow.
                    # Phase 1's H1 convergence test uses singular value
                    # spectrum distance, not R-matrix Frobenius distance,
                    # because the latter is corrupted by sign/permutation
                    # ambiguity in the SVD between checkpoints. We
                    # therefore compute the per-layer L2 distance between
                    # log singular value spectra, averaged over layers.
                    # This is gauge-invariant and matches what Phase 1
                    # documents as the "Sigma-distance" diagnostic.
                    if ("singular_values" in zf.files and
                            s in singular_values_final):
                        sv_t = zf["singular_values"].astype(np.float64)
                        sv_f = singular_values_final[s]
                        if sv_t.shape == sv_f.shape:
                            # Per-layer log-sv-spectrum distance.
                            # Add a tiny floor to avoid log(0).
                            with np.errstate(divide="ignore"):
                                log_t = np.log(sv_t + 1e-12)
                                log_f = np.log(sv_f + 1e-12)
                            per_layer = np.sqrt(
                                np.sum((log_t - log_f) ** 2, axis=1)
                            )
                            sigma_dist[si, ti] = float(np.mean(per_layer))

                    # Post-final-norm anomaly residual: difference
                    # between log-variance at last layer and a linear fit
                    # through the inner layers (excluding both boundary
                    # layers, t=0 and the last).
                    if "singular_values" in zf.files:
                        sv = zf["singular_values"].astype(np.float64)
                        # per-layer total variance (sum of squared sv).
                        # We don't have N_pilots stored here, but the
                        # *relative* shape matters; normalize by sv[0].
                        v_per_layer = (sv ** 2).sum(axis=1)
                        if v_per_layer.size >= 4 and (v_per_layer > 0).all():
                            log_v = np.log(v_per_layer)
                            inner_idx = np.arange(1, v_per_layer.size - 1)
                            slope, intercept = np.polyfit(
                                inner_idx, log_v[inner_idx], 1)
                            last_idx = v_per_layer.size - 1
                            predicted = slope * last_idx + intercept
                            pfn_resid[si, ti] = float(
                                log_v[last_idx] - predicted)
            except Exception as e:
                pass

    return {
        "seeds": seeds_present,
        "steps": np.array(common, dtype=np.int64),
        "log_alpha": log_alpha,
        "log_alpha_paper": log_alpha_paper,
        "lambda_all": lam_all,
        "lambda_all_paper": lam_all_paper,
        "mean_kurtosis": mean_kurt,
        "effective_rank_max": erank_max,
        "sigma_distance_to_final": sigma_dist,
        "post_final_norm_residual": pfn_resid,
    }


# ----------------------------------------------------------------------
# Co-location analysis: at the CV(lambda) minimum, where are the
# Phase 1 anomalies?
# ----------------------------------------------------------------------
def compute_cv_lambda(d3: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """From d3_per_token_fits, compute CV(lambda) across tokens per
    (seed, step). Returns (steps, cv_mean_over_seeds)."""
    lam = d3["lambda_per_token"]      # (n_seeds, n_steps, n_tokens)
    steps = d3["steps"]
    n_s, n_t, n_tok = lam.shape
    cv = np.full((n_s, n_t), np.nan)
    for si in range(n_s):
        for ti in range(n_t):
            v = lam[si, ti]
            v = v[np.isfinite(v)]
            if v.size >= 2 and abs(v.mean()) > 1e-12:
                cv[si, ti] = float(v.std(ddof=1) / abs(v.mean()))
    cv_mean = np.nanmean(cv, axis=0)
    return steps, cv, cv_mean


def find_minimum(steps: np.ndarray, values: np.ndarray) -> Tuple[int, float]:
    """Return (step_at_min, value_at_min) ignoring NaN."""
    finite = np.isfinite(values)
    if not finite.any():
        return -1, float("nan")
    idx = np.nanargmin(values)
    return int(steps[idx]), float(values[idx])


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_colocation(
    run_dir: str,
    steps: np.ndarray, cv_per_seed: np.ndarray, cv_mean: np.ndarray,
    phase1: Optional[Dict],
) -> None:
    if phase1 is not None:
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
        axes_flat = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 1, figsize=(11, 5))
        axes_flat = [axes]

    def _shade_anomaly_windows(ax):
        """Shade documented Phase 1 anomaly windows with vertical bands."""
        for key, info in PHASE1_ANOMALIES.items():
            lo, hi = info["range"]
            ax.axvspan(lo, hi, color=info["color"], alpha=0.10)

    # Panel 1: CV(lambda) with anomaly windows.
    ax = axes_flat[0]
    n_seeds = cv_per_seed.shape[0]
    for s in range(n_seeds):
        ax.plot(steps, cv_per_seed[s], "-", alpha=0.45, lw=0.9,
                label=f"seed {s}" if s == 0 else None)
    ax.plot(steps, cv_mean, "k-", lw=2.5, label="mean across seeds")
    # Mark the minimum.
    min_step, min_val = find_minimum(steps, cv_mean)
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--",
                   label=f"CV min at step {min_step}")
    _shade_anomaly_windows(ax)
    # Add a legend for the anomaly windows (one entry per).
    for key, info in PHASE1_ANOMALIES.items():
        ax.plot([], [], color=info["color"], alpha=0.4, lw=8,
                label=info["label"])
    ax.set_xscale("log")
    ax.set_ylabel("CV(lambda) across forward-set tokens")
    ax.set_title("D8: conditional-dynamics universality vs Phase 1 anomaly windows")
    ax.legend(fontsize=7, loc="best", ncol=2)

    if phase1 is None:
        ax.set_xlabel("training step")
        fig.tight_layout()
        out = os.path.join(figures_dir(run_dir), "d8_phase1_colocation.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"[plot] -> {out}")
        return

    # Panel 2: log_alpha trajectory.
    ax = axes_flat[1]
    la = phase1["log_alpha"]
    la_p = phase1["log_alpha_paper"]
    p1_steps = phase1["steps"]
    for s in range(la.shape[0]):
        ax.plot(p1_steps, la[s], "-", color="C0", alpha=0.45, lw=0.9)
        ax.plot(p1_steps, la_p[s], "-", color="C1", alpha=0.45, lw=0.9)
    ax.plot(p1_steps, np.nanmean(la, axis=0), "C0-", lw=2,
            label="log_alpha (ours)")
    ax.plot(p1_steps, np.nanmean(la_p, axis=0), "C1-", lw=2,
            label="log_alpha (paper)")
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--")
    _shade_anomaly_windows(ax)
    ax.set_ylabel("log_alpha")
    ax.set_title("Phase 1: log_alpha trajectory")
    ax.legend(fontsize=8, loc="best")

    # Panel 3: all-to-all lambda trajectory, both conventions.
    # The paper convention (mean-of-log per-coord variances) is what
    # Phase 1 documents and the most natural comparison to the
    # conditional lambda values in D3. The "ours" convention
    # (log-of-mean) differs by a Jensen gap that decays from ~0.5 at
    # initialization to ~0.03 at convergence, so its trajectory shows
    # mostly the gap closing rather than lambda dynamics.
    ax = axes_flat[2]
    lam_all = phase1["lambda_all"]
    lam_all_paper = phase1.get("lambda_all_paper")
    for s in range(lam_all.shape[0]):
        ax.plot(p1_steps, lam_all[s], "-", color="C0", alpha=0.4, lw=0.9)
        if lam_all_paper is not None and np.isfinite(lam_all_paper[s]).any():
            ax.plot(p1_steps, lam_all_paper[s], "-",
                    color="C1", alpha=0.4, lw=0.9)
    ax.plot(p1_steps, np.nanmean(lam_all, axis=0), "C0-", lw=2,
            label="lambda (ours = log-of-mean)")
    if lam_all_paper is not None and np.isfinite(lam_all_paper).any():
        ax.plot(p1_steps, np.nanmean(lam_all_paper, axis=0),
                "C1-", lw=2, label="lambda (paper = mean-of-log)")
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--")
    _shade_anomaly_windows(ax)
    ax.set_ylabel("all-to-all lambda")
    ax.set_title("Phase 1: marginal lambda trajectory (both conventions)")
    ax.legend(fontsize=8, loc="best")

    # Panel 4: mean kurtosis.
    ax = axes_flat[3]
    mk = phase1["mean_kurtosis"]
    for s in range(mk.shape[0]):
        ax.plot(p1_steps, mk[s], "-", alpha=0.5, lw=0.9)
    ax.plot(p1_steps, np.nanmean(mk, axis=0), "k-", lw=2,
            label="mean across seeds")
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--")
    _shade_anomaly_windows(ax)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("mean kurtosis (over layers)")
    ax.set_title("Phase 1: marginal kurtosis trajectory")
    ax.legend(fontsize=8, loc="best")

    # Panel 5: Sigma-spectrum distance to final. Basis-invariant
    # per-layer L2 distance between log-singular-value spectra at the
    # current checkpoint vs at the final checkpoint, averaged over
    # layers. Rescaled per-seed by the step-0 value so each trajectory
    # runs from 1.0 (initial spectrum, maximally far from final) toward
    # 0.0 (at the final step). This is the Phase-1-consistent
    # convergence diagnostic; raw R-matrix Frobenius distance is
    # corrupted by sign/permutation ambiguity in the SVD between
    # checkpoints and would saturate near sqrt(2H).
    ax = axes_flat[4]
    sd = phase1["sigma_distance_to_final"]
    if np.isfinite(sd).any():
        sd_rescaled = np.full_like(sd, np.nan)
        for s_idx in range(sd.shape[0]):
            row = sd[s_idx]
            finite = np.where(np.isfinite(row))[0]
            if finite.size < 2:
                continue
            ref = row[finite[0]]
            if ref > 1e-12:
                sd_rescaled[s_idx] = row / ref
        for s in range(sd_rescaled.shape[0]):
            ax.plot(p1_steps, sd_rescaled[s], "-", alpha=0.5, lw=0.9)
        ax.plot(p1_steps, np.nanmean(sd_rescaled, axis=0), "k-", lw=2,
                label="mean across seeds")
    else:
        ax.text(0.5, 0.5,
                "singular_values not in Phase 1 flow files\n"
                "(Sigma distance not computable)",
                ha="center", va="center", transform=ax.transAxes)
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--")
    _shade_anomaly_windows(ax)
    ax.set_ylabel("||log SV(t) - log SV_final(t)||_2 / initial")
    ax.set_xlabel("training step")
    ax.set_title("Phase 1: singular-value spectrum distance to final "
                 "(basis-invariant)")
    ax.legend(fontsize=8, loc="best")

    # Panel 6: post-final-norm anomaly trajectory.
    ax = axes_flat[5]
    pfn = phase1["post_final_norm_residual"]
    if np.isfinite(pfn).any():
        for s in range(pfn.shape[0]):
            ax.plot(p1_steps, pfn[s], "-", alpha=0.5, lw=0.9)
        ax.plot(p1_steps, np.nanmean(pfn, axis=0), "k-", lw=2,
                label="mean across seeds")
    else:
        ax.text(0.5, 0.5, "singular_values not in Phase 1 flow files",
                ha="center", va="center", transform=ax.transAxes)
    if min_step > 0:
        ax.axvline(min_step, color="red", lw=1.5, ls="--")
    _shade_anomaly_windows(ax)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel("log V_last - linear_fit(inner)")
    ax.set_xlabel("training step")
    ax.set_title("Phase 1: post-final-norm anomaly trajectory")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "D8: co-location of CV(lambda) non-monotonicity with Phase 1 anomalies",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d8_phase1_colocation.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="../phase1_runs_gelu")
    ap.add_argument("--no-phase1-data", action="store_true",
                    help="Skip loading Phase 1 trajectories; "
                         "overlay only documented anomaly windows.")
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)

    # Load D3 lambdas.
    d3_path = os.path.join(output_root(args.run_dir), "d3_per_token_fits.npz")
    if not os.path.exists(d3_path):
        print(f"Missing {d3_path}; run model_abc_discriminator.py first.")
        sys.exit(1)
    d3 = np.load(d3_path)
    steps, cv_per_seed, cv_mean = compute_cv_lambda(d3)
    min_step, min_val = find_minimum(steps, cv_mean)
    print(f"CV(lambda) mean trajectory: min at step {min_step} "
          f"value {min_val:.4f}")
    seeds_d3 = d3["seeds"].tolist()

    # Load Phase 1 trajectories.
    phase1 = None
    if not args.no_phase1_data:
        phase1 = load_phase1_trajectories(args.run_dir, seeds_d3)

    # Co-location report.
    payload = {
        "cv_lambda_minimum": {
            "step": int(min_step),
            "value": float(min_val),
        },
        "phase1_anomaly_windows": {
            k: {"label": v["label"], "range": list(v["range"])}
            for k, v in PHASE1_ANOMALIES.items()
        },
        "co_location_check": {},
    }
    for key, info in PHASE1_ANOMALIES.items():
        lo, hi = info["range"]
        inside = lo <= min_step <= hi
        nearest_edge = min(abs(min_step - lo), abs(min_step - hi))
        payload["co_location_check"][key] = {
            "anomaly_range": [lo, hi],
            "cv_min_step": int(min_step),
            "cv_min_inside_window": bool(inside),
            "distance_to_nearest_edge": int(nearest_edge),
        }
        marker = "INSIDE" if inside else f"{nearest_edge} steps from edge"
        print(f"  {info['label']:<40s} window [{lo}, {hi}]  "
              f"-- CV min: {marker}")

    if phase1 is not None:
        # Where does each Phase 1 trajectory have its own extremum?
        p1_steps = phase1["steps"]
        for key in ["log_alpha", "log_alpha_paper", "lambda_all",
                    "lambda_all_paper", "mean_kurtosis",
                    "sigma_distance_to_final", "post_final_norm_residual"]:
            arr = phase1.get(key)
            if arr is None:
                continue
            arr_mean = np.nanmean(arr, axis=0)
            if not np.isfinite(arr_mean).any():
                continue
            if key in ("mean_kurtosis", "sigma_distance_to_final"):
                idx = np.nanargmax(arr_mean)
                kind = "max"
            else:
                idx = np.nanargmin(arr_mean)
                kind = "min"
            extremum_step = int(p1_steps[idx])
            extremum_val = float(arr_mean[idx])
            payload.setdefault("phase1_extrema", {})[key] = {
                "kind": kind,
                "step": extremum_step,
                "value": extremum_val,
                "distance_to_cv_min": abs(extremum_step - min_step),
            }
            print(f"  Phase 1 {key:<35s} {kind} at step {extremum_step:>6d}  "
                  f"(value {extremum_val:.4f}, "
                  f"|delta from CV min| = {abs(extremum_step - min_step)})")

    out_json = os.path.join(output_root(args.run_dir), "d8_phase1_colocation.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[json] -> {out_json}")

    plot_colocation(args.run_dir, steps, cv_per_seed, cv_mean, phase1)


if __name__ == "__main__":
    main()
    