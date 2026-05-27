"""
Phase 2 attribution analysis (§5.8 of the proposal).

For each basis-invariant statistic S and each architectural design axis
D with baseline value b and variant values {v_1, v_2, ...}:

  - Measure S at the Phase 1 GELU baseline (4 seeds → mean, std,
    1.5×std threshold).
  - Measure S at each variant (2+ seeds → mean).
  - Compute ΔS_i = S(variant i) - S(baseline).
  - Compare |ΔS_i| to the baseline 1.5×std threshold.
  - Classify:
      * |ΔS_i| < threshold for all i  → S is ROBUST to axis D.
      * Monotonic non-zero ΔS_i exceeding threshold → axis D CONTROLS S.
      * Non-monotonic or extreme-only crossing → NON_MONOTONIC.

The output is the attribution matrix:
    rows: statistics
    cols: design axes
    entries: classification (ROBUST / CONTROLS↑ / CONTROLS↓ / NON_MONOTONIC)

This module is read-only over disk: it consumes already-computed
flow_analysis/ directories produced by phase2_analyze.py.

Usage
-----
    # Full attribution matrix on real inputs:
    python3 phase2_attribution.py --out phase2_attribution.txt

    # Tier 1b cross-input-distribution attribution (FFN vs attention):
    python3 phase2_attribution.py --tier1b --out tier1b_attribution.txt

    # Per-variant detail dump:
    python3 phase2_attribution.py --detail --out phase2_detail.txt
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# These are project modules; this file lives at the project root in
# deployment.
from compare_seeds import (
    SeedBundle, load_seeds, collect_summary_rows, compute_sv_distance_trajectory,
    compute_h1_criterion,
)
from phase2_configs import (
    ALL_TIER1_VARIANTS, ALL_VARIANTS, BASELINE_LABEL, VariantSpec,
)
from phase2_launch import run_dir_for, PHASE2_ROOT, PHASE1_GELU_ROOT
from phase2_input_distributions import FLOW_SUBDIR_FOR_INPUT


# Classification labels for the attribution matrix.
ROBUST = "robust"
CONTROLS_UP = "controls↑"
CONTROLS_DOWN = "controls↓"
NON_MONOTONIC = "non-monotonic"
INSUFFICIENT = "insufficient-data"


# ----------------------------------------------------------------------
# Statistics extracted per bundle. Mirrors collect_summary_rows but lets
# us treat each row's values as a numpy array we can take a mean over
# rather than printing.
#
# NOTE: load_seeds() in compare_seeds.py refuses to bundle runs whose
# (L, H) disagree -- that's correct for a within-variant comparison. For
# Phase 2 we deliberately bundle variants of DIFFERENT (L, H), so we
# must bundle each variant separately and then compare across variants
# at the statistic level. The cross-variant comparison is intrinsic-
# geometric (scalar statistics), exactly what Phase 1 established as
# the valid level of comparison (PHASE_1_WRITEUP §7).
# ----------------------------------------------------------------------


def safe_load_one_variant(run_dirs: List[str]) -> List[SeedBundle]:
    """Bundle seeds for one variant. Refuses to mix (L, H) per
    compare_seeds.load_seeds.

    Returns empty list if no run_dirs are present yet.
    """
    if not run_dirs:
        return []
    return load_seeds(run_dirs)


@dataclass
class StatisticSummary:
    """Per-(variant, statistic) summary: just the seed values + mean.

    The std isn't useful here because we typically have 2 seeds per
    variant -- the std estimate is noisy. The baseline (4 seeds) gives
    the dispersion floor we compare against.
    """
    name: str
    values: List[float]  # per-seed final-checkpoint scalar
    mean: float

    @classmethod
    def from_row(cls, row: dict) -> "StatisticSummary":
        return cls(name=row["name"], values=row["values"], mean=row["mean"])


@dataclass
class VariantMeasurement:
    """All statistics for one (axis, variant) cell.

    Stored as a dict keyed by statistic name for easy joining against
    the baseline.
    """
    axis: str
    label: str
    axis_value: Optional[float]
    n_seeds: int
    stats: Dict[str, StatisticSummary] = field(default_factory=dict)


@dataclass
class BaselineMeasurement:
    """The Phase 1 GELU baseline: 4 seeds, full dispersion table."""
    n_seeds: int
    stats: Dict[str, dict]  # statistic name -> compare_seeds row dict


# ----------------------------------------------------------------------
# Loading per-variant measurements.
# ----------------------------------------------------------------------
def find_run_dirs_for_variant(
    variant: VariantSpec, root: str = PHASE2_ROOT,
) -> List[str]:
    """Existing run_dirs for one variant (only ones that have been
    analyzed)."""
    out = []
    for seed in variant.seeds:
        rd = run_dir_for(variant, seed, root=root)
        flow_dir = os.path.join(rd, "flow_analysis")
        if os.path.isdir(flow_dir) and os.listdir(flow_dir):
            out.append(rd)
    return out


def find_baseline_run_dirs(
    root: str = PHASE1_GELU_ROOT, flow_subdir: str = "flow_analysis",
) -> List[str]:
    """Phase 1 GELU baseline run dirs that have been analyzed."""
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not d.startswith("seed_"):
            continue
        full = os.path.join(root, d)
        flow_dir = os.path.join(full, flow_subdir)
        if os.path.isdir(flow_dir) and os.listdir(flow_dir):
            out.append(full)
    return out


def load_baseline(
    flow_subdir: str = "flow_analysis",
) -> Optional[BaselineMeasurement]:
    """Load Phase 1 GELU baseline as a BaselineMeasurement.

    flow_subdir is "flow_analysis" for the standard (real-input) baseline,
    or "flow_analysis_shuffled" / "flow_analysis_random" for the Tier 1b
    cross-input-distribution comparison.
    """
    run_dirs = find_baseline_run_dirs(flow_subdir=flow_subdir)
    if not run_dirs:
        return None
    bundles = load_seeds(run_dirs)  # NOTE: requires matching (L, H), which is fine here
    rows = collect_summary_rows(bundles)
    stats = {r["name"]: r for r in rows}
    return BaselineMeasurement(n_seeds=len(bundles), stats=stats)


def load_variant(
    variant: VariantSpec, flow_subdir: str = "flow_analysis",
) -> Optional[VariantMeasurement]:
    """Load one variant's measurements (each seed treated as independent)."""
    run_dirs = find_run_dirs_for_variant(variant)
    if not run_dirs:
        return None
    bundles = load_seeds(run_dirs)
    rows = collect_summary_rows(bundles)
    stat_dict = {r["name"]: StatisticSummary.from_row(r) for r in rows}
    return VariantMeasurement(
        axis=variant.axis, label=variant.label,
        axis_value=variant.axis_value, n_seeds=len(bundles),
        stats=stat_dict,
    )


# ----------------------------------------------------------------------
# Classification for one statistic × one axis.
# ----------------------------------------------------------------------
def classify_axis(
    baseline_mean: float,
    baseline_threshold: float,
    variant_means: List[Tuple[float, float]],
) -> Tuple[str, List[float]]:
    """Classify ROBUST / CONTROLS / NON_MONOTONIC for one statistic × axis.

    Args:
        baseline_mean: mean of the statistic at the Phase 1 GELU baseline.
        baseline_threshold: 1.5 × std at the baseline (the universality
            threshold from PHASE_1_WRITEUP).
        variant_means: list of (axis_value, mean) tuples for the variants
            varying this axis. The baseline (axis_value at the baseline
            point) is NOT included; we compute deltas relative to it.

    Returns:
        (classification, deltas) where deltas is the per-variant
        ΔS = variant_mean - baseline_mean, in the order given.
    """
    if not variant_means:
        return INSUFFICIENT, []
    if np.isnan(baseline_threshold) or baseline_threshold == 0:
        return INSUFFICIENT, [m - baseline_mean for _, m in variant_means]

    deltas = [(av, m - baseline_mean) for av, m in variant_means]
    abs_deltas = [abs(d) for _, d in deltas]

    # Threshold-crossing check.
    crossed = [ad > baseline_threshold for ad in abs_deltas]
    if not any(crossed):
        return ROBUST, [d for _, d in deltas]

    # Monotonicity check. Sort variants by axis_value; check that signed
    # delta is monotonic in axis_value. If all axis_values are missing,
    # we can't check monotonicity -- fall back to NON_MONOTONIC (which
    # really means "we can't tell").
    sortable = [(av, d) for av, d in deltas if av is not None]
    if len(sortable) >= 2:
        sortable.sort(key=lambda x: x[0])
        signed = [d for _, d in sortable]
        # Monotonicity is computed over [baseline_delta=0] ++ signed (sorted
        # by axis_value). Strictly: each step must be non-negative
        # (ascending) or non-positive (descending). Tolerance of the
        # baseline_threshold prevents tiny within-noise wobbles from
        # breaking monotonicity.
        seq = [0.0] + signed  # baseline is the implicit anchor at the
                              # baseline axis_value (which we don't include
                              # in `signed`, but in the sort order it
                              # belongs at the baseline axis_value).
        # We don't actually know where baseline_axis_value sits relative
        # to the variants. Two cases:
        #   - Variants straddle the baseline axis_value (e.g. depth L=6
        #     and L=24 around baseline L=12): the implied sequence is
        #     [variant_below, baseline=0, variant_above].
        #   - All variants on one side of the baseline (rare, but
        #     possible if a sweep is one-sided): the baseline is at
        #     one end.
        # For monotonicity testing the right thing is to check whether
        # the (signed_delta, axis_value) points -- including the baseline
        # point (0.0, baseline_axis_value) -- are monotonic. We don't
        # have baseline_axis_value here, so we use a heuristic: if the
        # signed deltas (sorted by variant axis_value, with the implicit
        # baseline=0 inserted) form a monotonic sequence regardless of
        # whether baseline sits at the start, middle, or end, we accept.
        #
        # Concretely: a sequence is "monotonic accepting baseline=0
        # somewhere" iff it is monotonic when baseline=0 is inserted at
        # SOME position. For a 2-variant sweep that simplifies to:
        # the two variant deltas can have the same sign (baseline 0
        # at an end) or opposite signs (baseline 0 between them).
        # Both are monotonic when baseline is placed appropriately.
        # So we ONLY reject monotonicity if the deltas themselves are
        # non-monotonic among the variants:
        ascending = all(
            b - a >= -baseline_threshold for a, b in zip(signed, signed[1:])
        )
        descending = all(
            b - a <= baseline_threshold for a, b in zip(signed, signed[1:])
        )
        # Direction: the slope of signed-delta vs axis-value. Positive
        # slope = statistic increases with axis value = CONTROLS_UP.
        slope = signed[-1] - signed[0]  # last variant minus first variant
        if ascending and not descending:
            return (CONTROLS_UP if slope > 0 else CONTROLS_DOWN), [d for _, d in deltas]
        elif descending and not ascending:
            return (CONTROLS_DOWN if slope < 0 else CONTROLS_UP), [d for _, d in deltas]
        elif ascending and descending:
            # Both directions allowed: all deltas equal (within threshold).
            # If we're here, at least one delta crossed the threshold
            # (otherwise we returned ROBUST above). That can only happen
            # if a single variant crossed; treat as NON_MONOTONIC.
            return NON_MONOTONIC, [d for _, d in deltas]
        else:
            return NON_MONOTONIC, [d for _, d in deltas]
    else:
        # Only one variant or no axis values: just report threshold-crossing
        # without a monotonicity label.
        return NON_MONOTONIC, [d for _, d in deltas]


# ----------------------------------------------------------------------
# Building the attribution matrix.
# ----------------------------------------------------------------------
@dataclass
class AttributionCell:
    statistic: str
    axis: str
    classification: str
    baseline_mean: float
    baseline_threshold: float
    variant_labels: List[str]
    variant_means: List[float]
    deltas: List[float]


def build_attribution_matrix(
    flow_subdir: str = "flow_analysis",
    variants: Optional[List[VariantSpec]] = None,
) -> List[AttributionCell]:
    """Build the full attribution matrix.

    For each statistic in the baseline summary table and each axis in
    the variant set, classify the effect and store one AttributionCell.

    Returns a flat list of cells; group as needed for presentation.
    """
    baseline = load_baseline(flow_subdir=flow_subdir)
    if baseline is None:
        raise RuntimeError(
            f"No baseline measurements found under {PHASE1_GELU_ROOT}/. "
            f"Run Phase 1 GELU analysis first."
        )

    if variants is None:
        variants = ALL_TIER1_VARIANTS

    # Group variants by axis.
    axes: Dict[str, List[VariantSpec]] = {}
    for v in variants:
        axes.setdefault(v.axis, []).append(v)

    # Per-axis variant measurements.
    measurements_by_axis: Dict[str, List[VariantMeasurement]] = {}
    for axis, var_list in axes.items():
        ms = []
        for v in var_list:
            m = load_variant(v, flow_subdir=flow_subdir)
            if m is not None:
                ms.append(m)
        measurements_by_axis[axis] = ms

    cells: List[AttributionCell] = []
    statistic_names = list(baseline.stats.keys())
    for stat_name in statistic_names:
        b_row = baseline.stats[stat_name]
        b_mean = b_row["mean"]
        b_thresh = b_row["threshold_1p5_std"]
        for axis, var_measurements in measurements_by_axis.items():
            if not var_measurements:
                cells.append(AttributionCell(
                    statistic=stat_name, axis=axis,
                    classification=INSUFFICIENT,
                    baseline_mean=b_mean,
                    baseline_threshold=b_thresh,
                    variant_labels=[], variant_means=[], deltas=[],
                ))
                continue
            variant_pairs = [
                (m.axis_value, m.stats[stat_name].mean)
                for m in var_measurements
                if stat_name in m.stats
            ]
            labels = [m.label for m in var_measurements if stat_name in m.stats]
            classification, deltas = classify_axis(
                baseline_mean=b_mean,
                baseline_threshold=b_thresh,
                variant_means=variant_pairs,
            )
            cells.append(AttributionCell(
                statistic=stat_name, axis=axis,
                classification=classification,
                baseline_mean=b_mean, baseline_threshold=b_thresh,
                variant_labels=labels,
                variant_means=[p[1] for p in variant_pairs],
                deltas=deltas,
            ))
    return cells


# ----------------------------------------------------------------------
# Rendering.
# ----------------------------------------------------------------------
def render_matrix_text(cells: List[AttributionCell]) -> str:
    """Render the attribution matrix as a text table.

    Rows are statistics; columns are axes; cells are classification
    labels. A separate detail section follows with deltas.
    """
    # Collect unique statistics and axes (preserve insertion order).
    stats_order, axes_order = [], []
    for c in cells:
        if c.statistic not in stats_order:
            stats_order.append(c.statistic)
        if c.axis not in axes_order:
            axes_order.append(c.axis)
    cell_lookup = {(c.statistic, c.axis): c for c in cells}

    # Column widths.
    stat_w = max(len(s) for s in stats_order) if stats_order else 0
    stat_w = max(stat_w, len("statistic"))
    axis_widths = [
        max(len(a), len(ROBUST), len(CONTROLS_UP), len(NON_MONOTONIC), 14)
        for a in axes_order
    ]

    out = []
    # Header.
    header = f"{'statistic':<{stat_w}}"
    for ax, w in zip(axes_order, axis_widths):
        header += "  " + f"{ax:<{w}}"
    out.append(header)
    out.append("-" * len(header))
    for s in stats_order:
        line = f"{s:<{stat_w}}"
        for ax, w in zip(axes_order, axis_widths):
            c = cell_lookup.get((s, ax))
            label = c.classification if c else INSUFFICIENT
            line += "  " + f"{label:<{w}}"
        out.append(line)

    # Detail section.
    out.append("")
    out.append("Δstatistic = variant_mean − baseline_mean, in units of the "
               "statistic. Crosses threshold = |Δ| > 1.5 × std(Phase 1 GELU).")
    out.append("")
    for s in stats_order:
        out.append(f"## {s}")
        # Find any cell to read baseline values from (same for all axes).
        any_cell = next(
            (cell_lookup[(s, ax)] for ax in axes_order if (s, ax) in cell_lookup),
            None,
        )
        if any_cell is not None:
            out.append(
                f"  baseline mean = {any_cell.baseline_mean:.4g}, "
                f"1.5×std threshold = {any_cell.baseline_threshold:.4g}"
            )
        for ax in axes_order:
            c = cell_lookup.get((s, ax))
            if c is None or not c.variant_labels:
                out.append(f"  axis {ax}: (no variant measurements)")
                continue
            for lab, m, d in zip(c.variant_labels, c.variant_means, c.deltas):
                crosses = (
                    "★ crosses" if not np.isnan(c.baseline_threshold)
                    and abs(d) > c.baseline_threshold else ""
                )
                out.append(
                    f"  axis {ax} [{lab}]: variant_mean = {m:.4g}, "
                    f"Δ = {d:+.4g}  {crosses}"
                )
            out.append(f"  axis {ax}: classification = {c.classification}")
        out.append("")
    return "\n".join(out)


def write_matrix_csv(cells: List[AttributionCell], path: str):
    """Write the attribution matrix as a long-format CSV.

    Columns: statistic, axis, classification, baseline_mean,
             baseline_threshold, variant_labels, variant_means, deltas.
    Lists serialized as ';'-separated strings.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "statistic", "axis", "classification",
            "baseline_mean", "baseline_threshold",
            "variant_labels", "variant_means", "deltas",
        ])
        for c in cells:
            w.writerow([
                c.statistic, c.axis, c.classification,
                f"{c.baseline_mean:.6g}",
                f"{c.baseline_threshold:.6g}",
                ";".join(c.variant_labels),
                ";".join(f"{m:.6g}" for m in c.variant_means),
                ";".join(f"{d:.6g}" for d in c.deltas),
            ])


# ----------------------------------------------------------------------
# Tier 1b cross-input-distribution comparison.
# ----------------------------------------------------------------------
def _load_tier1b_flow(run_dir: str, flow_subdir: str) -> Optional[dict]:
    """Load the single final-checkpoint flow .npz for a Tier 1b
    subdirectory. Returns the loaded flow dict, or None if the
    subdir doesn't exist or contains no flow file.

    Tier 1b writes exactly one flow_step_<N>.npz per (run, distribution),
    so we don't need the trajectory machinery in FlowSeries.
    """
    import glob
    from analyze import load_flow

    full = os.path.join(run_dir, flow_subdir)
    if not os.path.isdir(full):
        return None
    files = sorted(glob.glob(os.path.join(full, "flow_step_*.npz")))
    if not files:
        return None
    # If there's more than one, take the highest step (the final ckpt).
    def step_of(p):
        return int(os.path.basename(p).replace("flow_step_", "").replace(".npz", ""))
    return load_flow(max(files, key=step_of))


def _extract_tier1b_statistics(flow: dict) -> Dict[str, float]:
    """Pick the statistics from a single-checkpoint flow that are
    well-defined and input-dependent.

    Statistics excluded by design:
      - eval/train loss: stored from training time, not recomputed on the
        Tier 1b inputs. Including them would falsely suggest the model
        was re-evaluated.
      - H1 statistics: trajectory-derived (need >1 checkpoint).

    Statistics included: λ (ours and paper), log α (ours and paper),
    mean of per-layer kurtosis (signed and absolute), mean effective
    rank across layers, plus eff rank at first and middle layers.
    These are all computed FROM the activations at the final
    checkpoint, so they vary with input distribution.
    """
    import numpy as np
    out: Dict[str, float] = {}
    out["λ (ours)"] = float(flow["lambda"])
    out["λ (paper)"] = float(flow["lambda_paper"])
    out["log α (ours)"] = float(flow["log_alpha"])
    out["log α (paper)"] = float(flow["log_alpha_paper"])
    if "kurtosis_per_layer" in flow:
        out["<κ> (signed mean kurt)"] = float(
            np.nanmean(flow["kurtosis_per_layer"])
        )
    if "kurtosis_abs_per_layer" in flow:
        out["<|κ|> (paper kurt)"] = float(
            np.nanmean(flow["kurtosis_abs_per_layer"])
        )
    if "effective_rank" in flow:
        er = np.asarray(flow["effective_rank"], dtype=np.float64)
        out["mean effective rank"] = float(np.nanmean(er))
        out["eff rank L=0"] = float(er[0])
        out["eff rank mid"] = float(er[len(er) // 2])
    return out


def build_tier1b_table(input_distributions=("real", "shuffled", "random")):
    """For each input distribution, aggregate Tier 1b statistics across
    the Phase 1 GELU baseline seeds and return a long-format table.

    Bypasses the SeedBundle/FlowSeries machinery (which is hardcoded to
    read flow_analysis/ regardless of the requested distribution -- the
    bug that produced bit-identical columns in earlier runs). Instead
    reads each seed's flow_step_<final>.npz directly from the correct
    distribution subdirectory.

    The returned rows aggregate across all baseline seeds present:
        mean = average across seeds
        std  = sample std across seeds (ddof=1)
        n_seeds = how many baseline seeds contributed
        seed_labels = ordered list of seed identifiers (e.g. ["seed_0",
                      "seed_1", ...]) matching the order of
                      per_seed_values
        per_seed_values = list of per-seed values, in same order as
                          seed_labels

    The seed_labels / per_seed_values pairing is preserved across input
    distributions: index i in 'real' is the same seed as index i in
    'shuffled'. This lets a renderer pair them up to compute per-seed
    Δ values and check sign consistency.
    """
    import numpy as np

    rows = []
    for dist in input_distributions:
        if dist not in FLOW_SUBDIR_FOR_INPUT:
            raise ValueError(
                f"Unknown input distribution {dist!r}; expected one of "
                f"{tuple(FLOW_SUBDIR_FOR_INPUT)}."
            )
        subdir = FLOW_SUBDIR_FOR_INPUT[dist]

        # Find baseline seed dirs that have THIS distribution's analysis.
        # We deliberately also consider seed dirs that DON'T have this
        # subdir, so we can keep the seed ordering consistent across
        # distributions (the seed_labels for "shuffled" should match
        # the seed_labels for "real"). Seeds that are missing this
        # distribution get NaN in per_seed_values.
        all_seed_names: List[str] = []
        if os.path.isdir(PHASE1_GELU_ROOT):
            for d in sorted(os.listdir(PHASE1_GELU_ROOT)):
                if d.startswith("seed_") and os.path.isdir(
                    os.path.join(PHASE1_GELU_ROOT, d)
                ):
                    all_seed_names.append(d)

        if not all_seed_names:
            print(f"⚠️  No baseline data found (no seed_* directories "
                  f"under {PHASE1_GELU_ROOT}).")
            continue

        # Per-statistic per-seed values (NaN if seed lacks this dist).
        per_stat_per_seed: Dict[str, List[float]] = {}
        any_seed_present = False
        for seed_name in all_seed_names:
            seed_dir = os.path.join(PHASE1_GELU_ROOT, seed_name)
            if not os.path.isdir(os.path.join(seed_dir, subdir)):
                # Seed lacks this distribution. We'll fill NaN once we
                # know which statistics exist; can't do it now without
                # touching another seed first.
                continue
            flow = _load_tier1b_flow(seed_dir, subdir)
            if flow is None:
                continue
            any_seed_present = True
            stats = _extract_tier1b_statistics(flow)
            for name, val in stats.items():
                # Lazy-initialize the per-seed array for this statistic
                # to NaN of length all_seed_names, then fill in this seed.
                if name not in per_stat_per_seed:
                    per_stat_per_seed[name] = [float("nan")] * len(all_seed_names)
                per_stat_per_seed[name][all_seed_names.index(seed_name)] = val

        if not any_seed_present:
            print(f"⚠️  No baseline data found for input distribution "
                  f"{dist} (looked in <seed>/{subdir}/).")
            continue

        for name, vals in per_stat_per_seed.items():
            arr = np.asarray(vals, dtype=np.float64)
            n_present = int(np.sum(~np.isnan(arr)))
            mean = float(np.nanmean(arr)) if n_present > 0 else float("nan")
            std = float(np.nanstd(arr, ddof=1)) if n_present >= 2 else float("nan")
            rows.append({
                "input_distribution": dist,
                "statistic": name,
                "mean": mean,
                "std": std,
                "n_seeds": n_present,
                "seed_labels": list(all_seed_names),
                "per_seed_values": [float(v) for v in vals],
            })
    return rows


def build_per_variant_tier1b_table(
    input_distributions=("real", "shuffled", "random"),
    variants: Optional[List[VariantSpec]] = None,
    include_baseline: bool = True,
):
    """Per-variant Tier 1b table: for each variant (and optionally the
    Phase 1 GELU baseline), aggregate Tier 1b statistics across seeds
    for each input distribution.

    Same input-vs-FFN/attention question as build_tier1b_table, but
    extended to all Tier 1 variants so we can ask whether the
    FFN/attention decomposition is itself architecture-dependent.

    Returns a list of dicts, each:
        {"target": "baseline" or "<variant_label>",
         "axis":   "baseline" or "<variant_axis>",
         "axis_value": float or None,
         "input_distribution": "real" | "shuffled" | "random",
         "statistic": str,
         "mean":  float,
         "std":   float,
         "n_seeds": int}
    """
    import numpy as np

    if variants is None:
        variants = ALL_TIER1_VARIANTS

    # Build the list of (target_label, axis_label, axis_value, list-of-seed-dirs).
    targets: List[Tuple[str, str, Optional[float], List[str]]] = []
    if include_baseline:
        baseline_dirs = []
        if os.path.isdir(PHASE1_GELU_ROOT):
            for d in sorted(os.listdir(PHASE1_GELU_ROOT)):
                if d.startswith("seed_"):
                    baseline_dirs.append(os.path.join(PHASE1_GELU_ROOT, d))
        if baseline_dirs:
            targets.append(("baseline", "baseline", None, baseline_dirs))

    for v in variants:
        run_dirs = []
        for seed in v.seeds:
            rd = run_dir_for(v, seed, root=PHASE2_ROOT)
            if os.path.isdir(rd):
                run_dirs.append(rd)
        if run_dirs:
            targets.append((v.label, v.axis, v.axis_value, run_dirs))

    rows = []
    for target_label, axis_label, axis_value, seed_dirs in targets:
        # Compose stable seed names (e.g. "seed_0", "seed_1") for this
        # target. seed_dirs is already in order from the targets-builder
        # above (range(v.seeds) for variants, sorted listdir for baseline),
        # so the names just follow the seed_dir basenames.
        seed_names = [os.path.basename(sd) for sd in seed_dirs]

        for dist in input_distributions:
            if dist not in FLOW_SUBDIR_FOR_INPUT:
                raise ValueError(
                    f"Unknown input distribution {dist!r}; expected one "
                    f"of {tuple(FLOW_SUBDIR_FOR_INPUT)}."
                )
            subdir = FLOW_SUBDIR_FOR_INPUT[dist]

            # For each statistic, build a per-seed array of length
            # len(seed_dirs), with NaN where the seed lacks this dist.
            # This keeps the per_seed_values arrays aligned ACROSS
            # input distributions so a renderer can compute per-seed
            # Δ = shuffled[i] - real[i] safely.
            per_stat_per_seed: Dict[str, List[float]] = {}
            for i, seed_dir in enumerate(seed_dirs):
                if not os.path.isdir(os.path.join(seed_dir, subdir)):
                    continue
                flow = _load_tier1b_flow(seed_dir, subdir)
                if flow is None:
                    continue
                stats = _extract_tier1b_statistics(flow)
                for name, val in stats.items():
                    if name not in per_stat_per_seed:
                        per_stat_per_seed[name] = [float("nan")] * len(seed_dirs)
                    per_stat_per_seed[name][i] = val

            for name, vals in per_stat_per_seed.items():
                arr = np.asarray(vals, dtype=np.float64)
                n_present = int(np.sum(~np.isnan(arr)))
                mean = float(np.nanmean(arr)) if n_present > 0 else float("nan")
                std = float(np.nanstd(arr, ddof=1)) if n_present >= 2 else float("nan")
                rows.append({
                    "target": target_label,
                    "axis": axis_label,
                    "axis_value": axis_value,
                    "input_distribution": dist,
                    "statistic": name,
                    "mean": mean,
                    "std": std,
                    "n_seeds": n_present,
                    "seed_labels": list(seed_names),
                    "per_seed_values": [float(v) for v in vals],
                })
    return rows


def _per_seed_sign_consistency(
    rows: List[dict],
    target_filter: Optional[str] = None,
    reference_distribution: str = "real",
) -> Dict[Tuple[str, str, str], dict]:
    """Compute per-seed Δ vs reference_distribution and assess
    sign-consistency across seeds.

    For each (target, statistic, distribution) where distribution !=
    reference_distribution, computes:
        per_seed_delta[i] = value_dist[i] - value_real[i]
    then checks whether all non-NaN per-seed Δs share a sign.

    Args:
        rows: long-format rows from build_tier1b_table or
            build_per_variant_tier1b_table. Each row MUST contain
            "per_seed_values" and "seed_labels".
        target_filter: if given, only consider rows for this target
            label. If None, treat rows as baseline-only.
        reference_distribution: which distribution to compute Δ
            relative to. Default "real".

    Returns:
        dict keyed by (target, statistic, distribution), each value:
            {"per_seed_delta": [float, ...],
             "sign_consistent": True/False/None,
             "n_seeds_paired": int}
        sign_consistent is None when fewer than 2 seeds have non-NaN
        Δs (can't have consistency on <2 samples). Otherwise True iff
        all Δs share a sign (with a small tolerance to avoid flipping
        on near-zero noise).
    """
    import numpy as np

    # Re-index: (target, dist, stat) -> row.
    by_key = {}
    for r in rows:
        target = r.get("target", "baseline")  # baseline rows lack 'target'
        if target_filter is not None and target != target_filter:
            continue
        if "per_seed_values" not in r or "seed_labels" not in r:
            continue
        key = (target, r["input_distribution"], r["statistic"])
        by_key[key] = r

    out = {}
    # Identify reference (real) rows.
    ref_keys = [(t, d, s) for (t, d, s) in by_key if d == reference_distribution]
    for ref_key in ref_keys:
        t, _, stat = ref_key
        ref_row = by_key[ref_key]
        # For each OTHER distribution at the same (target, stat), pair seeds.
        for (t2, d2, s2), other in by_key.items():
            if t2 != t or s2 != stat or d2 == reference_distribution:
                continue
            if ref_row["seed_labels"] != other["seed_labels"]:
                # Seed sets disagree -- skip (shouldn't happen with our
                # NaN-fill convention, but defensive).
                continue
            deltas = []
            for ref_val, other_val in zip(
                ref_row["per_seed_values"], other["per_seed_values"]
            ):
                if np.isnan(ref_val) or np.isnan(other_val):
                    deltas.append(float("nan"))
                else:
                    deltas.append(float(other_val) - float(ref_val))
            present = [d for d in deltas if not np.isnan(d)]
            if len(present) < 2:
                sign_consistent: Optional[bool] = None
            else:
                # Use a small tolerance: treat Δ within ~1e-9 of zero
                # as ambiguous (not counted against consistency).
                eps = max(1e-9, 1e-6 * max(abs(d) for d in present))
                signs = {1 if d > eps else (-1 if d < -eps else 0)
                         for d in present}
                # Sign-consistent iff all non-zero signs agree (zeros
                # are compatible with either direction).
                nonzero_signs = signs - {0}
                sign_consistent = (len(nonzero_signs) <= 1)
            out[(t, d2, stat)] = {
                "per_seed_delta": deltas,
                "sign_consistent": sign_consistent,
                "n_seeds_paired": len(present),
            }
    return out


def render_tier1b_text(rows: List[dict]) -> str:
    """Render the baseline-only Tier 1b table grouped by statistic,
    dists as columns.

    Use this when the rows came from build_tier1b_table() (baseline
    only). For per-variant output (from build_per_variant_tier1b_table)
    use render_per_variant_tier1b_text instead.
    """
    if not rows:
        return "(no Tier 1b data found)"
    # Pivot.
    stats = []
    by_stat: Dict[str, Dict[str, dict]] = {}
    for r in rows:
        s = r["statistic"]
        if s not in by_stat:
            stats.append(s)
            by_stat[s] = {}
        by_stat[s][r["input_distribution"]] = r
    dists_present = []
    for r in rows:
        if r["input_distribution"] not in dists_present:
            dists_present.append(r["input_distribution"])

    # Per-seed sign consistency. Returns (target, dist, stat) -> info.
    # baseline rows lack a "target" field, so the helper treats them as
    # target="baseline".
    sign_info = _per_seed_sign_consistency(rows, target_filter="baseline")

    stat_w = max(len(s) for s in stats)
    stat_w = max(stat_w, len("statistic"))
    col_w = 22  # expanded for the [sign] marker

    out = []
    head = f"{'statistic':<{stat_w}}"
    for d in dists_present:
        if d == "real":
            head += "  " + f"{d + ' (mean±std)':<{col_w}}"
        else:
            head += "  " + f"{d + ' (mean±std) [sign]':<{col_w}}"
    out.append(head)
    out.append("-" * len(head))
    for s in stats:
        line = f"{s:<{stat_w}}"
        for d in dists_present:
            r = by_stat[s].get(d)
            if r is None:
                line += "  " + "(missing)".ljust(col_w)
                continue
            cell_str = f"{r['mean']:.4g} ± {r['std']:.4g}"
            if d != "real":
                info = sign_info.get(("baseline", d, s))
                if info is None or info["sign_consistent"] is None:
                    marker = "?"
                elif info["sign_consistent"]:
                    marker = "✓"
                else:
                    marker = "✗"
                cell_str += f" [{marker}]"
            line += "  " + cell_str.ljust(col_w)
        out.append(line)
    out.append("")
    out.append(
        "Sign marker on Δ vs real (per-seed Δs across baseline seeds):\n"
        "  ✓ = all seeds' per-seed Δ agree on sign\n"
        "  ✗ = per-seed Δs disagree on sign (cross-seed mean is "
        "averaging over opposite-direction effects)\n"
        "  ? = fewer than 2 paired seeds available"
    )
    out.append(
        "A ✗ marker on a small Δ means the effect is not reliable -- "
        "the input distribution interacts with seed-specific quirks "
        "to give different per-seed directions. A ✓ marker on a small Δ "
        "is genuine input-invariance."
    )
    out.append("")
    out.append(
        "Interpretation: a statistic with large Δ and ✓ across "
        "distributions attributes (partly) to attention. A statistic "
        "with small Δ and ✓ attributes (mostly) to the FFN. A small Δ "
        "with ✗ indicates the statistic mixes input-driven and "
        "seed-driven contributions of comparable size -- treat the "
        "Δ as noise rather than a real effect. See proposal §5.4."
    )
    return "\n".join(out)


def render_per_variant_tier1b_text(
    rows: List[dict],
    input_distributions=("real", "shuffled", "random"),
) -> str:
    """Render the per-variant Tier 1b table.

    For each statistic, emit a block showing each (target, distribution)
    cell as `mean ± std`. Targets are grouped by axis with the baseline
    listed first.

    The headline question this table answers: does the FFN/attention
    decomposition we see at the baseline ALSO hold at each variant?
    Read down a column for one variant's input dependence; read across
    a row to see how that input dependence changes with architecture.
    """
    if not rows:
        return "(no per-variant Tier 1b data)"

    # Index by (target, dist, stat).
    cell = {(r["target"], r["input_distribution"], r["statistic"]): r
            for r in rows}

    # Ordered target list: baseline first, then variants grouped by axis.
    targets_seen: List[Tuple[str, str]] = []  # (target_label, axis)
    for r in rows:
        key = (r["target"], r["axis"])
        if key not in targets_seen:
            targets_seen.append(key)
    # Sort: baseline first, then by axis, then by target order within axis.
    def target_sort_key(t):
        label, axis = t
        return (0 if label == "baseline" else 1, axis, label)
    targets_seen.sort(key=target_sort_key)

    # Statistics in order of first appearance.
    statistics: List[str] = []
    for r in rows:
        if r["statistic"] not in statistics:
            statistics.append(r["statistic"])

    # Per-target sign consistency. The helper computes (target, dist,
    # stat) -> {sign_consistent: ..., per_seed_delta: [...]} for each
    # target it's given via target_filter. We aggregate across targets
    # into one big dict so we can look up by (target, dist, stat).
    sign_info: Dict[Tuple[str, str, str], dict] = {}
    for target, _axis in targets_seen:
        sign_info.update(
            _per_seed_sign_consistency(rows, target_filter=target)
        )

    def _sign_marker(target, dist, stat):
        info = sign_info.get((target, dist, stat))
        if info is None or info["sign_consistent"] is None:
            return "?"
        return "✓" if info["sign_consistent"] else "✗"

    lines = []
    lines.append("Per-variant Tier 1b table")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        "Each cell: <input distribution mean> ± <seed std>. "
        "For each variant, the (shuffled − real) gap measures how much "
        "the statistic depends on inter-token structure -- i.e. the "
        "attention contribution at that variant's architecture. "
        "The [sign] marker shows whether the per-seed Δs agree on "
        "sign across the target's seeds (✓), disagree (✗), or "
        "couldn't be assessed (?, fewer than 2 paired seeds)."
    )
    lines.append("")

    # For each statistic, one block.
    for stat in statistics:
        lines.append(f"## {stat}")
        # Column widths.
        target_w = max(len(t) for t, _ in targets_seen)
        col_w = 22
        # Header.
        header = f"  {'target':<{target_w}}"
        for dist in input_distributions:
            header += f"  {dist:<{col_w}}"
        # Extra columns: Δ shuffled vs real, Δ random vs real, each
        # with a sign marker.
        header += f"  {'Δshuf/|real|':<16}{'Δrand/|real|':<16}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for target, axis in targets_seen:
            cells = [cell.get((target, d, stat)) for d in input_distributions]
            if all(c is None for c in cells):
                continue
            line = f"  {target:<{target_w}}"
            for c in cells:
                if c is None:
                    line += f"  {'—':<{col_w}}"
                else:
                    line += f"  {c['mean']:.4g} ± {c['std']:.3g}".ljust(col_w + 2)

            # Compute the Δ columns relative to real for THIS target.
            real_cell = cell.get((target, "real", stat))
            if real_cell is not None and abs(real_cell["mean"]) > 1e-12:
                denom = abs(real_cell["mean"])
                shuf_cell = cell.get((target, "shuffled", stat))
                rand_cell = cell.get((target, "random", stat))
                if shuf_cell is not None:
                    shuf_dpct = (shuf_cell["mean"] - real_cell["mean"]) / denom * 100
                    m = _sign_marker(target, "shuffled", stat)
                    line += f"  {shuf_dpct:+6.1f}% [{m}]    "
                else:
                    line += f"  {'—':<16}"
                if rand_cell is not None:
                    rand_dpct = (rand_cell["mean"] - real_cell["mean"]) / denom * 100
                    m = _sign_marker(target, "random", stat)
                    line += f"  {rand_dpct:+6.1f}% [{m}]"
                else:
                    line += f"  {'—':<16}"
            lines.append(line)
        lines.append("")

    lines.append(
        "Interpretation: Δshuf/|real| ≈ 0 means the statistic is "
        "input-invariant at that variant — FFN-driven contribution "
        "dominates. Large |Δshuf/|real|| means attention contributes "
        "substantially. The [sign] marker confirms whether the cross-seed "
        "mean is a real effect (✓) or an average over opposite-direction "
        "per-seed effects (✗). A small Δ with ✗ should be treated as "
        "noise, not invariance. Whether the FFN/attention decomposition "
        "changes across variants tells you whether the decomposition "
        "is architecture-dependent."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 attribution analysis."
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path for the attribution table (default: stdout).",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Also write a long-format CSV of the attribution cells.",
    )
    parser.add_argument(
        "--tier1b", action="store_true",
        help="Build the Tier 1b cross-input-distribution table instead "
             "of the cross-axis attribution matrix.",
    )
    parser.add_argument(
        "--tier1b_dists", type=str, default="real,shuffled,random",
        help="Comma-separated input distributions for the Tier 1b table.",
    )
    parser.add_argument(
        "--per_variant_tier1b", action="store_true",
        help="Build a per-variant Tier 1b table (baseline + every Tier 1 "
             "variant × each input distribution) instead of the baseline-"
             "only Tier 1b table or the cross-axis matrix. Use this to "
             "ask whether the FFN/attention decomposition is itself "
             "architecture-dependent.",
    )
    args = parser.parse_args()

    if args.per_variant_tier1b:
        dists = tuple(d.strip() for d in args.tier1b_dists.split(",") if d.strip())
        rows = build_per_variant_tier1b_table(input_distributions=dists)
        text = render_per_variant_tier1b_text(rows, input_distributions=dists)
    elif args.tier1b:
        dists = tuple(d.strip() for d in args.tier1b_dists.split(",") if d.strip())
        rows = build_tier1b_table(input_distributions=dists)
        text = render_tier1b_text(rows)
    else:
        cells = build_attribution_matrix()
        text = render_matrix_text(cells)
        if args.csv:
            write_matrix_csv(cells, args.csv)
            print(f"Wrote CSV: {args.csv}", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
            f.write("\n")
        print(f"Wrote: {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
    