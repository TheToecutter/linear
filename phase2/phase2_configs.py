"""
Phase 2 variant definitions: single source of truth for the architectural
ablations described in PROJECT_PROPOSAL_v2.md §5.

Each variant defines:
  - A label (string ID used for run-directory naming).
  - A ModelConfig factory (since some variants change multiple fields that
    must agree, e.g. width changes head count).
  - The list of seeds to train.
  - A tier ("1a", "1b_analysis_only", "2", or "3") indicating when it runs.
  - An "axis" string ("depth", "width", "ffn_ratio", "norm", "heads",
    "gating", "external") used by the attribution analysis to group
    variants along the single design axis they vary.

Conventions
-----------
* The Phase 1 GELU baseline (L=12, H=896, I=3648, 14 heads, GELU FFN,
  RMSNorm, full multi-head attention) is the reference data point for
  every Phase 2 axis. Phase 2 does NOT re-train it — it consumes the
  4-seed GELU bundle produced by the running Phase 1 GELU job.
* Each Phase 2 variant uses 2 seeds by default (per §6.2), with the
  adaptive third-seed protocol applied at analysis time if the 2-seed
  dispersion looks anomalous.
* Run directories are at:
    phase2_runs/<axis>/<variant_label>/seed_<n>/
  e.g. phase2_runs/depth/L06/seed_0/

Note: the ModelConfig factories assert their preconditions explicitly
(e.g. head_dim must stay at 64 across the width sweep) so that a
mis-typed variant fails loudly rather than silently producing a
wrong-shape model.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from config import ModelConfig, TrainingConfig


# ----------------------------------------------------------------------
# Phase 1 GELU baseline: the reference configuration. Phase 2 does NOT
# re-train this — it's listed here so the attribution code knows what
# the baseline looks like.
# ----------------------------------------------------------------------
def make_baseline_gelu_config() -> ModelConfig:
    """Phase 1 GELU baseline: L=12, H=896, I_swiglu=2432 → I_gelu=3648,
    14 heads × head_dim 64, RMSNorm, full multi-head, RoPE, tied embed.

    This is the L=12 / H=896 / 1.5H-ratio data point that anchors all
    three Tier 1 sweeps.
    """
    return ModelConfig(
        ffn_type="gelu",
        num_hidden_layers=12,
        hidden_size=896,
        intermediate_size=2432,  # auto-becomes 3648 inside LlamaBlock for GELU
        num_attention_heads=14,
        architecture="llama",
    )


BASELINE_LABEL = "baseline_gelu"  # the Phase 1 GELU runs


# ----------------------------------------------------------------------
# Variant descriptor.
# ----------------------------------------------------------------------
@dataclass
class VariantSpec:
    """One Phase 2 variant: a label, a config factory, and metadata.

    The config_factory is a no-arg callable rather than a stored
    ModelConfig so that defaults can't drift between definition time
    and run time, and so that the factory can re-validate its
    preconditions on every call.
    """
    label: str
    axis: str
    tier: str
    config_factory: Callable[[], ModelConfig]
    seeds: List[int] = field(default_factory=lambda: [0, 1])
    description: str = ""
    # Optional integer values for plotting/attribution-axis ordering.
    # E.g. for depth sweep, axis_value=6 for L=6, 12 for L=12, 24 for L=24.
    axis_value: Optional[float] = None

    def __post_init__(self):
        # Constructing the config at __post_init__ time has two benefits:
        # (1) preconditions inside factories fail at definition time
        # rather than at launch time, and (2) the param count is
        # available for the launch banner.
        _ = self.config_factory()


# ======================================================================
# Tier 1a: depth sweep (§5.3 A)
# ======================================================================
# Hold H, num_heads, head_dim, I, FFN choice fixed. Vary only L.
# Per §6.1 the chosen contrast is "H fixed; parameter count varies",
# so that the variance-growth law λ × L can be tested directly.
# Baseline (L=12) reuses the Phase 1 GELU 4-seed runs; here we only
# need the two non-baseline depths.

def make_depth_variant(num_layers: int) -> Callable[[], ModelConfig]:
    """Factory factory: return a config builder for the given depth."""
    def factory() -> ModelConfig:
        cfg = make_baseline_gelu_config()
        cfg.num_hidden_layers = num_layers
        return cfg
    factory.__name__ = f"make_depth_L{num_layers:02d}"
    return factory


DEPTH_VARIANTS: List[VariantSpec] = [
    VariantSpec(
        label="L06",
        axis="depth",
        tier="1a",
        config_factory=make_depth_variant(6),
        seeds=[0, 1],
        description="Shallow depth sweep: L=6, H=896 fixed.",
        axis_value=6,
    ),
    VariantSpec(
        label="L18",
        axis="depth",
        tier="1a",
        config_factory=make_depth_variant(18),
        seeds=[0, 1],
        description=(
            "Intermediate depth: L=18, H=896 fixed. Added as a fourth "
            "point on the depth axis (alongside L=6, L=12 baseline, L=24) "
            "to test whether the approximate λL conservation extrapolates "
            "monotonically or curves. The Phase 1 GELU 4-seed measurements "
            "for L=12, L=6, and L=24 gave λL = 4.10, 4.01, 4.18 (paper "
            "convention); λL at L=18 is predicted to lie between 4.10 and "
            "4.18 if the drift is monotonic."
        ),
        axis_value=18,
    ),
    VariantSpec(
        label="L24",
        axis="depth",
        tier="1a",
        config_factory=make_depth_variant(24),
        seeds=[0, 1],
        description="Deep depth sweep: L=24, H=896 fixed.",
        axis_value=24,
    ),
]


# ======================================================================
# Tier 1a: width sweep (§5.3 B)
# ======================================================================
# Hold L, ffn_type, FFN ratio fixed. Vary H. Head count scales so that
# head_dim stays at 64 (which is what the paper's variants do too).
# intermediate_size scales proportionally to keep the SwiGLU-equivalent
# ratio I_swiglu / H ≈ 2.71 (which the GELU branch then re-scales to
# 1.5×, preserving the SwiGLU-vs-GELU parameter-matching invariant).

def make_width_variant(hidden_size: int) -> Callable[[], ModelConfig]:
    """Width-sweep config builder.

    head_dim is held at 64 throughout the sweep; num_attention_heads
    = H / 64. SwiGLU intermediate_size is scaled proportionally to the
    baseline ratio I/H ≈ 2.714 (matches Llama's 8/3 convention).
    """
    head_dim = 64
    if hidden_size % head_dim != 0:
        raise ValueError(
            f"width variant: hidden_size={hidden_size} not divisible by "
            f"head_dim=64."
        )
    num_heads = hidden_size // head_dim
    # Baseline ratio: 2432 / 896 = 2.71428...  (= 8/3 rounded).
    # Scale exactly proportionally and then round to nearest multiple of
    # 64 to preserve vectorization-friendly shapes.
    baseline_H = 896
    baseline_I = 2432
    raw_I = baseline_I * hidden_size / baseline_H
    intermediate_size = int(round(raw_I / 64) * 64)

    def factory() -> ModelConfig:
        cfg = make_baseline_gelu_config()
        cfg.hidden_size = hidden_size
        cfg.num_attention_heads = num_heads
        cfg.intermediate_size = intermediate_size
        # head_dim is a @property derived from H / num_heads.
        assert cfg.head_dim == head_dim, (
            f"width variant H={hidden_size}: head_dim became "
            f"{cfg.head_dim}, expected {head_dim}."
        )
        return cfg
    factory.__name__ = f"make_width_H{hidden_size:04d}"
    return factory


WIDTH_VARIANTS: List[VariantSpec] = [
    VariantSpec(
        label="H0448",
        axis="width",
        tier="1a",
        config_factory=make_width_variant(448),
        seeds=[0, 1],
        description="Narrow width sweep: H=448 (7 heads × 64), L=12 fixed.",
        axis_value=448,
    ),
    VariantSpec(
        label="H1792",
        axis="width",
        tier="1a",
        config_factory=make_width_variant(1792),
        seeds=[0, 1],
        description="Wide width sweep: H=1792 (28 heads × 64), L=12 fixed.",
        axis_value=1792,
    ),
]


# ======================================================================
# Tier 1a: FFN intermediate-ratio sweep (§5.3 C)
# ======================================================================
# Hold L, H, ffn_type fixed. Vary FFN intermediate ratio. The ratios
# are 1.5/3/4 against H=896; the GELU FFN path inside LlamaBlock takes
# the stored SwiGLU intermediate_size and multiplies by 1.5 to get the
# actual GELU intermediate dimension. So:
#   stored I=896 → GELU actually uses I_gelu=1344  (ratio 1.5)
#   stored I=1792 → GELU actually uses I_gelu=2688 (ratio 3)  -- IS NOT 3×H
# We circumvent this by setting stored I so that 1.5 × stored_I equals
# the target ratio × H. Concretely:
#   target ratio 1.5 → I_gelu = 1.5×896 = 1344  → stored I = 896
#   target ratio 3.0 → I_gelu = 3.0×896 = 2688  → stored I = 1792
#   target ratio 4.0 → I_gelu = 4.0×896 = 3584  → stored I ≈ 2389 ≈ 2432
# The baseline GELU run already uses I_gelu = 3648 (≈ ratio 4.07×H), so
# variant ratio=4 is essentially the Phase 1 GELU baseline and need
# NOT be re-trained. We therefore train the lower-ratio variants only.

def make_ffn_ratio_variant(target_ratio: float) -> Callable[[], ModelConfig]:
    """FFN-ratio sweep builder.

    target_ratio is the desired I_gelu / H (the actual GELU intermediate
    dimension divided by the hidden size). The stored intermediate_size
    is scaled so that LlamaBlock's 1.5× expansion lands on the requested
    I_gelu, rounded to the nearest multiple of 64 for vectorization.
    """
    H = 896
    target_I_gelu = int(round(target_ratio * H / 64) * 64)
    # LlamaBlock computes I_gelu = (3 * intermediate_size) // 2,
    # so we want intermediate_size = (2 / 3) * I_gelu, rounded to a
    # multiple of 64 so the GELU I lands on a clean multiple.
    raw_I_swiglu = (2 / 3) * target_I_gelu
    stored_I = int(round(raw_I_swiglu / 64) * 64)
    # Sanity: re-derive the actual I_gelu after rounding and surface it.
    actual_I_gelu = (3 * stored_I) // 2

    def factory() -> ModelConfig:
        cfg = make_baseline_gelu_config()
        cfg.intermediate_size = stored_I
        return cfg
    factory.__name__ = f"make_ffn_ratio_{target_ratio:.1f}x".replace(".", "p")
    # Stash the resolved I for the launch banner.
    factory.stored_intermediate_size = stored_I
    factory.actual_gelu_intermediate = actual_I_gelu
    factory.target_ratio = target_ratio
    return factory


FFN_RATIO_VARIANTS: List[VariantSpec] = [
    VariantSpec(
        label="ffn_1p5x",
        axis="ffn_ratio",
        tier="1a",
        config_factory=make_ffn_ratio_variant(1.5),
        seeds=[0, 1],
        description=(
            "FFN ratio sweep: I_gelu / H = 1.5 (= 0.5 × baseline). "
            "Smaller FFN."
        ),
        axis_value=1.5,
    ),
    VariantSpec(
        label="ffn_3p0x",
        axis="ffn_ratio",
        tier="1a",
        config_factory=make_ffn_ratio_variant(3.0),
        seeds=[0, 1],
        description=(
            "FFN ratio sweep: I_gelu / H = 3.0 (= 0.74 × baseline). "
            "Mid-size FFN."
        ),
        axis_value=3.0,
    ),
    # Note: the baseline GELU run already has I_gelu / H ≈ 4.07, so the
    # "ratio = 4" variant is approximated by the Phase 1 GELU baseline
    # itself. We document this rather than re-train.
]


# ======================================================================
# Tier 2 (conditional): normalization, head count, gating one-shot
# ======================================================================
# Phase 2 is GELU-only by design: every variant the queue trains must
# use ffn_type="gelu" so its statistics are comparable to the Phase 1
# GELU baseline. The SwiGLU one-shot that previously lived here has
# been removed -- the Phase 1 SwiGLU runs in phase1_runs/ already
# provide that measurement at 4 seeds, and keeping a SwiGLU factory
# in the Tier 2 catalog created an easy footgun (someone hand-launches
# it and the resulting run silently breaks cross-variant comparability).
#
# Normalization (RMSNorm → LayerNorm) and head-count variants are
# left as TODOs until the LayerNorm code path is added to llama.py
# and head-count variation is supported as a separate axis from H.
# Adding them prematurely would risk un-validated model paths.
TIER2_VARIANTS: List[VariantSpec] = []


# ======================================================================
# All variants combined.
# ======================================================================
ALL_TIER1_VARIANTS: List[VariantSpec] = (
    DEPTH_VARIANTS + WIDTH_VARIANTS + FFN_RATIO_VARIANTS
)

ALL_VARIANTS: List[VariantSpec] = ALL_TIER1_VARIANTS + TIER2_VARIANTS


def find_variant(label: str) -> VariantSpec:
    """Look up a variant by label. Raises ValueError if not found."""
    for v in ALL_VARIANTS:
        if v.label == label:
            return v
    raise ValueError(
        f"Unknown variant label {label!r}. Known: "
        f"{[v.label for v in ALL_VARIANTS]}"
    )


def variants_by_axis(axis: str) -> List[VariantSpec]:
    """All variants varying the named design axis."""
    return [v for v in ALL_VARIANTS if v.axis == axis]


def summarize() -> str:
    """Human-readable summary of all variants and their resolved configs."""
    lines = []
    lines.append("Phase 2 variant catalog")
    lines.append("=======================")
    lines.append("")
    lines.append(f"Baseline (NOT re-trained, consumed from Phase 1 GELU runs):")
    base = make_baseline_gelu_config()
    lines.append(
        f"  {BASELINE_LABEL}: L={base.num_hidden_layers}, "
        f"H={base.hidden_size}, "
        f"I_stored={base.intermediate_size} (I_gelu={3*base.intermediate_size//2}), "
        f"heads={base.num_attention_heads}, "
        f"params ≈ {base.estimate_param_count() / 1e6:.1f}M, "
        f"4 seeds [from phase1_runs_gelu/]"
    )
    lines.append("")
    by_axis = {}
    for v in ALL_VARIANTS:
        by_axis.setdefault(v.axis, []).append(v)
    for axis, variants in by_axis.items():
        lines.append(f"Axis: {axis}")
        for v in variants:
            cfg = v.config_factory()
            params_M = cfg.estimate_param_count() / 1e6
            seed_str = ",".join(str(s) for s in v.seeds) if v.seeds else "(none yet)"
            lines.append(
                f"  [{v.tier}] {v.label}: L={cfg.num_hidden_layers}, "
                f"H={cfg.hidden_size}, "
                f"I_stored={cfg.intermediate_size}, "
                f"heads={cfg.num_attention_heads}, "
                f"ffn={cfg.ffn_type}, "
                f"params={params_M:.1f}M, seeds=[{seed_str}]"
            )
            lines.append(f"       {v.description}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    # Print the variant catalog when run directly. Useful for sanity-
    # checking that resolved configs match the proposal.
    print(summarize())
    