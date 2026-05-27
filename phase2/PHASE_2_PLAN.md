# Phase 2 plan and status

**Document status:** v0 (initial scaffolding; updated as runs complete)
**Audience:** future-self; project supervisor; downstream paper reviewer

This document tracks the Phase 2 execution plan, including the exact
configs being run, the open methodological decisions from
`PROJECT_PROPOSAL_v2.md` §6, the launch schedule, and a per-variant
status table. It is the operational complement to the proposal.

The proposal stays the scientific reference; this document stays
the operational reference. Where they disagree, the proposal wins
and this document is updated.

---

## 1. Status snapshot

| Item                                                     | Status            |
|----------------------------------------------------------|-------------------|
| Phase 1 SwiGLU (baseline run, 4 seeds)                   | Complete          |
| Phase 1 GELU (Phase 2 baseline reference, 4 seeds)       | In progress       |
| Phase 2 scaffolding (configs, launch, analyze, attrib.)  | Complete          |
| Phase 2 Tier 1a launches (depth, width, FFN-ratio)       | Blocked on §6.3   |
| Phase 2 Tier 1b runs (shuffled, random inputs)           | Blocked on baseline + 1a |
| Phase 2 attribution matrix                               | Blocked on 1a    |

The blocking dependency chain is: GELU baseline finishes → §6.3
pre-checks pass → Tier 1a training launches → Tier 1a analysis →
Tier 1b analysis → attribution matrix. Each blocked item lists
the gating condition.

---

## 2. Settled decisions

The proposal flagged two open methodological questions in §6.
Both are resolved here.

### 2.1 Depth-sweep covariate (proposal §6.1)

**Decision: hold H constant, let parameter count vary.**

The depth sweep tests the paper's claim that λ × L ≈ 5.5 is
approximately conserved. The conservation law is stated in terms of
L and λ independently, so changing L while holding everything else
fixed is the right experiment. Parameter count is reported as a
covariate alongside the result, not controlled.

L=6 ≈ 88M params, L=24 ≈ 263M params; the variation is documented.
If the depth result shows a strong λ effect, the planned follow-up
(proposal §7.2) is a parameter-matched depth sweep that confirms
the effect is depth-attributable rather than param-count-attributable.

### 2.2 Seed-count protocol (proposal §6.2)

**Decision: 2 seeds per variant by default; adaptive third seed.**

Each Tier 1 variant gets 2 seeds. After the analyzer produces
the 2-seed dispersion for each statistic, we compare to the
Phase 1 GELU 4-seed dispersion:

- If the variant's 2-seed range is within 2× the baseline 4-seed std
  on every statistic → 2 seeds suffice.
- If any statistic shows a variant-range exceeding 2× the baseline
  std → add a third seed and re-run the analysis for that variant.

Expected effective seed count: ~2.2 (some variants will trigger the
third seed; most won't).

### 2.3 Tier 1b non-trajectory measurement (added)

The proposal frames Tier 1b's input-distribution decomposition as a
final-state measurement (§5.4). We confirm this: Tier 1b runs the
analyzer on the *final* checkpoint of each variant for each input
distribution, not the full 50-checkpoint trajectory. Trajectory
measurements with three input distributions would multiply storage
by 3× and analysis compute by 3× with unclear marginal value.

---

## 3. Pre-launch checks (§3 — gates that block Tier 1a)

Three pre-checks must pass after Phase 1 GELU completes and
before Tier 1a launches.

### 3.1 Phase 1 GELU go/no-go (proposal §4.6)

Phase 2 launches iff:

1. H1 PASS on all 4 GELU seeds (ratio < 0.10).
2. Within-variant dispersion on all basis-invariant statistics is
   within 2× the SwiGLU values from `PHASE_1_WRITEUP.md`.
3. Eval loss is within 0.05 of SwiGLU.
4. Cross-seed R matrices remain unaligned in the GELU regime.

If any check fails, Phase 2 is paused for diagnosis (proposal §4.6).

### 3.2 H1792 memory-fit pre-check (added)

The H=1792 width variant is approximately 527M params — ~3.6× the
baseline. The existing training recipe (micro_batch_size=8,
grad_accum_steps=8, seq_len=1024, gradient_checkpointing on) was
sized for the 146M model; it may not fit at H=1792 on a single
5090 (32 GB VRAM).

Before launching `H1792`, run a short fit-check:

```bash
python3 phase2_memfit.py --variant H1792 --total_steps 50
```

The script trains 50 steps (`micro_batch_size=8` at first, falling
back to micro_batch_size=4 with grad_accum_steps=16 if OOM). It
reports peak VRAM and step time. If the model needs micro_batch
reduction to fit, the fit-check writes a stamped notes file
(`phase2_runs/width/H1792/MEMFIT_NOTES.txt`) documenting the
effective batch shape used for that variant. The full training
run consumes those overrides automatically.

The L=24 variant (263M params) is a milder version of the same
concern and gets the same fit-check.

### 3.3 Tier 1b loader sanity

Before Tier 1b launches on the trained models, verify the loaders
on a smoke-test pass:

```bash
python3 phase2_analyze.py --tier1b --only_variant <small_model> --total_steps 50
```

This confirms shuffled/random inputs propagate cleanly through the
analyzer (which has only ever seen real language). Expected to be
trivially fine; we run it once for paranoia.

---

## 4. Launch schedule

The proposal estimates ~12 GPU-hours per 24,000-step variant. At
1 × 5090, the schedule is:

| Phase                              | Run count | GPU-hours | Wall-clock days |
|------------------------------------|----------:|----------:|----------------:|
| Phase 1 GELU (in progress)         |         4 |        48 |               2 |
| Phase 2 Tier 1a: depth (L=6, L=24)|     2×2=4 |        48 |               2 |
| Phase 2 Tier 1a: width (H=448, H=1792)| 2×2=4 |     ~64*  |             2.5 |
| Phase 2 Tier 1a: FFN ratio (1.5x, 3.0x)| 2×2=4|        48 |               2 |
| Tier 1a analysis (per-variant)     |        — |    ~30**  |             1.5 |
| Tier 1b analysis (baseline + 1a)   |        — |     ~12** |             0.5 |
| **Tier 1a + 1b total**             |       12 |   **~210**|         **~9** |

*H=1792 is larger than the baseline and may take longer per step.
The 64-hour estimate adds a 35% buffer over the bare 48 hours.

**Analysis is CPU-heavy (SVDs at H=448/896/1792 × 50 checkpoints) but
GPU-light (forward passes only). Tier 1a analysis runs the full 50-
checkpoint pipeline per variant; Tier 1b runs only one checkpoint × 2
extra input distributions per variant.

Tier 2 (norm, heads, gating one-shot) is conditional on Tier 1a
findings — launched if Tier 1a's attribution matrix has open
questions. Estimated additional ~120 GPU-hours (~5 days) if all
Tier 2 variants run.

Tier 3 (external-validity check against Gemma-2-style) is similarly
conditional, ~24 GPU-hours.

---

## 5. Tier 1a variant catalog

See `phase2_configs.summarize()` for the live catalog. Snapshot:

```
Baseline (NOT re-trained, consumed from phase1_runs_gelu/):
  L=12, H=896, I_gelu=3648, 14 heads × 64,  ≈ 146M params, 4 seeds

Axis: depth (H=896 held)
  L06: L=6,  ≈  88M params, 2 seeds
  L24: L=24, ≈ 263M params, 2 seeds

Axis: width (L=12, head_dim=64, ratio I/H ≈ 2.71 held)
  H0448: H=448,  7 heads,  ≈  44M params, 2 seeds
  H1792: H=1792, 28 heads, ≈ 527M params, 2 seeds [memfit-check first]

Axis: ffn_ratio (L=12, H=896 held; ratio is I_gelu / H)
  ffn_1p5x: I_gelu = 1344, ≈  97M params, 2 seeds
  ffn_3p0x: I_gelu = 2688, ≈ 126M params, 2 seeds
  ffn_4p0x: ≈ baseline (I_gelu = 3648, ratio ≈ 4.07) — no separate run
```

Total Tier 1a: 6 variants × 2 seeds = 12 new training runs.

---

## 6. Per-variant status

This table is updated as variants finish. "Trained" requires
checkpoints present and run_metadata.json. "Analyzed" requires
flow_analysis/ populated for the full trajectory. "1b analyzed"
requires flow_analysis_shuffled/ AND flow_analysis_random/ for
the final checkpoint.

| Axis      | Variant   | Seed | Trained | Analyzed | 1b analyzed | Eval loss | λ (boundary-excl) | Notes |
|-----------|-----------|------|---------|----------|-------------|----------:|-------------------:|-------|
| depth     | L06       | 0    | ☐       | ☐        | ☐           |         — |                  — |       |
| depth     | L06       | 1    | ☐       | ☐        | ☐           |         — |                  — |       |
| depth     | L24       | 0    | ☐       | ☐        | ☐           |         — |                  — |       |
| depth     | L24       | 1    | ☐       | ☐        | ☐           |         — |                  — |       |
| width     | H0448     | 0    | ☐       | ☐        | ☐           |         — |                  — |       |
| width     | H0448     | 1    | ☐       | ☐        | ☐           |         — |                  — |       |
| width     | H1792     | 0    | ☐       | ☐        | ☐           |         — |                  — | memfit-check first |
| width     | H1792     | 1    | ☐       | ☐        | ☐           |         — |                  — | memfit-check first |
| ffn_ratio | ffn_1p5x  | 0    | ☐       | ☐        | ☐           |         — |                  — |       |
| ffn_ratio | ffn_1p5x  | 1    | ☐       | ☐        | ☐           |         — |                  — |       |
| ffn_ratio | ffn_3p0x  | 0    | ☐       | ☐        | ☐           |         — |                  — |       |
| ffn_ratio | ffn_3p0x  | 1    | ☐       | ☐        | ☐           |         — |                  — |       |

The baseline column (12 variants × 2 = 24 cells in the active
matrix; 4 GELU baseline seeds attach separately).

---

## 7. Operational commands

The full set of commands to run after the Phase 1 GELU pre-checks
pass. Each is independently re-runnable (skip-existing applies).

```bash
# 1. Memory-fit pre-check on the heavyweight variants.
python3 phase2_memfit.py --variant H1792 --variant L24

# 2. Launch all Tier 1a variants × 2 seeds, sequentially.
python3 phase2_launch.py --launch_tier 1a

# 3. Analyze every Tier 1a run (50 ckpts each, full pipeline).
python3 phase2_analyze.py --analyze_variants

# 4. Tier 1b: shuffled + random inputs against final checkpoints.
python3 phase2_analyze.py --tier1b

# 5. Build the cross-axis attribution matrix.
python3 phase2_attribution.py --out phase2_attribution.txt --csv phase2_attribution.csv

# 6. Build the Tier 1b cross-input-distribution table.
python3 phase2_attribution.py --tier1b --out phase2_tier1b_table.txt

# 7. Plot the attribution heatmap.
python3 phase2_plots.py --out phase2_attribution_heatmap.png
```

Single-variant operation: replace `--launch_tier 1a` with
`--variant <label> --seed <n>`.

---

## 8. Failure modes and contingency

The proposal §7 covers conceptual risks. Operational risks worth
calling out here:

**H1792 fails to fit on 5090.** Fall back via gradient-checkpointing
strategies (already on) and micro_batch reduction; if still over,
the memfit-check writes notes and the launcher applies them. If
micro_batch=1 still OOMs, we cannot run H1792 at the current
seq_len=1024 and would need to either reduce seq_len or skip the
variant. Either choice is documented in the writeup.

**Tier 1a takes > 9 days wall-clock.** The schedule is a planning
estimate; if real wall-clock substantially exceeds it, we
serialize the order so that the depth + FFN-ratio sweeps complete
first (those map to the proposal's most concrete predictions —
λL conservation and effective-rank scaling). Width sweep finishes
last.

**An attribution cell shows non-monotonic effect.** This is
captured as `non-monotonic` in the matrix and triggers either a
third seed (per §2.2) or a follow-up variant probing the
intermediate axis value (e.g. L=18 between L=12 and L=24).

**Tier 1b decomposition violates the FFN/attention prediction.**
If shuffled and real inputs give identical macro statistics, the
FFN/attention decomposition's premise is wrong (or much weaker
than the proposal expected). The result is still publishable as
a corrective finding about what the macro structure does and
does not track in input structure.

---

## 9. Deliverables checklist

By Phase 2 completion (Tier 1a + Tier 1b):

1. ☐ 12 trained Tier 1a variant runs (6 variants × 2 seeds).
2. ☐ Analyzed flow trajectories for all 12.
3. ☐ Tier 1b shuffled + random analyses for baseline + 12 variants.
4. ☐ Phase 2 attribution matrix (CSV + text).
5. ☐ Phase 2 Tier 1b cross-input-distribution table.
6. ☐ Phase 2 attribution heatmap.
7. ☐ Phase 2 final writeup (extending `PHASE_1_WRITEUP.md`).
8. ☐ Updated SwiGLU-vs-GELU comparison appendix.

Items 1–3 are the data; items 4–6 are the analysis artifacts;
item 7 is the writing.

---

*End of plan.*
