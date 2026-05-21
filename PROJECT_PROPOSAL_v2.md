# The Macro Structure of Trained Transformers: A Pilot Study at 150M Scale

## A controlled investigation of the ensemble geometry the lines-of-thought framework recovers, and what architectural choices control its structure

---

## Abstract

The lines-of-thought framework of Sarfati et al. (ICLR 2025) shows that the
residual stream of a trained transformer admits a low-dimensional
linear-Gaussian description across layers: token trajectories follow a
deterministic linear flow, modulo a Gaussian residual whose per-coordinate
variance grows as a power of layer offset. The framework is striking
because it claims a clean, structural picture of how a transformer
transforms its inputs — one that depends on relatively few intrinsic
quantities (the variance growth rate $\lambda$, a prefactor $\log\alpha$,
the effective rank profile, and a small set of others) and that the paper
suggests may be approximately universal across architectures.

We propose a pilot study at 150M parameters that pursues two
questions about this framework, in two phases.

**Phase 1** establishes the framework's within-variant reproducibility on a
single architectural variant trained for 1.57B tokens at four independent
seeds. A prior 4-seed pilot using SwiGLU activations completed during the
project's first iteration and established:

- The framework's basis-invariant statistics ($\lambda$, $\log\alpha$,
  effective rank profile, isotropy profile, trajectory smoothness, the
  boundary-layer effect, and several training-dynamic features)
  reproduce across seeds at within-variant relative spreads of 1–4%.
- Cross-seed $R$ matrices share no recoverable basis structure — different
  seeds learn functionally equivalent models along seed-specific bases
  unrelated by any orthogonal map. The framework's basis-invariant
  statistics are the appropriate level of cross-model description; the
  $R$ matrices themselves are model-internal coordinate representations
  with no canonical cross-model meaning.
- Per-coordinate residual kurtosis was the single most disperse statistic
  across seeds, with a substantial outlier in one seed that the other
  measurements did not flag. The likely mechanism is gating-induced
  per-token variability in the SwiGLU FFN inflating ensemble-level
  kurtosis.

To establish a macro-clean baseline for Phase 2's ablations, we will
**re-run Phase 1's 4-seed pilot using an ungated GELU FFN** in place of
SwiGLU. The GELU variant is parameter-matched (same per-block parameter
count) and removes a known confound from kurtosis and from variance
spectra generally. This is a controlled-baseline study, not a separate
phase: the goal is to establish noise floors in the regime where Phase 2's
ablations will be run.

**Phase 2** is a causal study of which architectural choices control which
macro-level properties of the residual stream's variance structure.
Rather than comparing standing architectures (Llama vs Gemma vs Qwen vs
DeepSeek), as the project's earlier iteration proposed, Phase 2 varies
**single design axes at a time** against the Phase 1 GELU baseline:
network depth $L$, hidden width $H$, FFN intermediate ratio, and (as
follow-ons) normalization choice and attention configuration. Each
single-axis variation isolates the mechanism by which a specific design
choice influences the macro structure, instead of confounding multiple
simultaneously-varying factors.

The reframing from "are different architectures universal?" to "what
architectural choices control the macro structure?" reflects what Phase 1
established: the basis-invariant statistics travel with task performance
across seeds; the absolute pose of the residual stream in $\mathbb{R}^H$
does not. Phase 2's goal is to identify which design choices push which
intrinsic geometric quantities, with the within-variant Phase 1 noise
floor serving as the threshold for attribution.

The original proposal's commitments to vocabulary-anchored Procrustes
alignment for cross-model comparison, to a predictability phase, and to
an intervention phase are dropped. Phase 1 established that
basis-anchored cross-model alignment is unworkable at this scale (and
likely at any scale), and the predictability and intervention phases were
premised on that alignment succeeding. They are not part of the present
work and would require different machinery if pursued in a future
project.

---

## 1. Background and motivation

### 1.1 The lines-of-thought framework

The lines-of-thought framework characterizes trained transformer language
models as approximately linear-Gaussian dynamical systems at the
population level. For each transformer layer transition $t \to t+\tau$,
Sarfati et al. fit a linear map of the form

$$\tilde{x}(t+\tau) = R(t+\tau)\,\Lambda(t,\tau)\,R(t)^\top\,x(t)$$

where $R(t)$ is a per-layer orthonormal basis recovered from the SVD of
layer-$t$ activations across a sample of token positions, and
$\Lambda(t,\tau)$ is a diagonal stretch matrix capturing how singular
values evolve between layers. The actual layer transition is the linear
prediction plus a residual $w(t,\tau)$ whose ensemble distribution is
approximately Gaussian with per-coordinate variance scaling as
$\sigma^2 \sim \alpha\,\tau^\lambda$.

Two properties of this framework matter for what follows. First, the
linear flow is a *population* object — it describes how the cluster of
all token trajectories collectively deforms between layers, not what any
single trajectory does. Second, the framework's recovered quantities
fall naturally into two groups: **basis-invariant** quantities ($\lambda$,
$\log\alpha$, the effective rank profile, the isotropy profile,
successive-layer angle profiles) that don't depend on the choice of
coordinate system in which $R(t)$ is expressed, and **basis-dependent**
quantities (the $R(t)$ matrices themselves) that do.

### 1.2 What the paper's universality claim does and doesn't establish

The paper observes that several basis-invariant statistics take similar
values across four trained models from different architectural families
(GPT-2 medium, Llama-2-7B, Mistral-7B, Pythia-12B). It interprets this as
evidence for a kind of universality: the lines-of-thought structure is
not specific to one architecture but is a generic property of trained
transformers.

This claim faces three confounds. First, the four models differ in
**architecture**, but also in **scale** (350M to 12B parameters), in
**training corpus** (WebText, RedPajama, RedPajama-1T, The Pile), and in
**training duration**. The paper holds none of these constant. Observed
similarities in $\lambda$, $\log\alpha$, etc. could be attributed to any
of these dimensions individually or to interactions. Second, the paper
reports point measurements without within-variant noise floors — there
is no notion of "how much variability would we see if we re-trained the
same model with a different seed?" Without that floor, observed
cross-model similarities can't be calibrated against within-variant
noise. Third, the paper does not address cross-model alignment in
hidden space, so comparisons of $R(t)$ across models are not done
directly; the comparisons that *are* done are on basis-invariant
scalars, which is the appropriate level for the data but obscures
whether the framework's universality is structural or merely
intrinsic-geometric.

### 1.3 A controlled pilot to disentangle these confounds

This proposal isolates the architectural axis by holding every other
variable constant. All experiments use:

- the same training corpus (FineWeb-Edu, sample-10BT subset),
- the same tokenizer (Mistral-7B-v0.1's BPE, 32,768 tokens),
- the same training recipe (AdamW with $\beta_2 = 0.95$, cosine LR
  schedule with 1000-step warmup, peak LR $3\times 10^{-4}$, batch 64
  sequences × 1024 tokens for 24,000 steps = 1.57B tokens),
- the same analysis pipeline (500 held-out chunks × 19 pilot positions
  per chunk → 9500 pilot activations per layer, 14 layer states
  spanning post-embedding through post-final-norm).

This isolates *what the architecture is doing* from confounds about
data, optimization, and scale.

The 150M scale is chosen because it *enables* the methodology this
project depends on. At this scale, on a single RTX 5090, we can run
multiple seeds per architecture (giving within-variant noise floors
the paper lacks), save 50 log-spaced checkpoints per run (giving the
training-dynamic measurements that single-snapshot studies miss), and
do single-axis ablation sweeps over depth, width, and FFN
configurations (giving causal attribution that head-to-head comparisons
of pretrained models can't provide). Phase 1's first iteration
established that the framework's claims do hold at this scale: the
blunderbuss exists, basis-invariant statistics are reproducible across
seeds, the boundary anomaly emerges during training, the mid-training
$\log\alpha$ hump replicates. The 150M scale is not "below the paper's
range" in a way that's getting in the way of any question we've asked;
it is below the paper's range in a way that lets us ask better-posed
questions than the paper's setup could.

What this scale doesn't directly address is whether the *attribution
picture we build at 150M* (which design choices control which macro
properties) extends to larger scales. That is a separate question for
a different (and substantially more expensive) study. Our results are
valid characterizations of macro structure at 150M; extending them to
frontier scale is an extrapolation we do not test.

### 1.4 Conceptual contribution

This study makes one conceptual contribution beyond what the paper
demonstrates, which became clear during Phase 1's first iteration: the
framework's basis-invariant statistics describe the *intrinsic geometric
properties* of the residual stream — the magnitudes and shape of the
variance structure — while the $R(t)$ matrices describe the *absolute
pose* of that geometry in the ambient hidden space. The intrinsic
geometry is reproducible across independent training runs of the same
architecture; the absolute pose is not. The framework's preferred
quantities are exactly the ones that factor out the absolute-pose
freedom that the training process exploits, leaving the intrinsic
functional content that any equivalent model should agree on.

This framing — what the framework measures is the *shape* of the
ensemble variance, not the *placement* of any specific token trajectory
in hidden space — sharpens the universality question. Phase 2 asks: do
different architectures, trained on the same data with the same recipe,
build the same intrinsic shape, even though each chooses its own
absolute pose? That is a well-posed question with concrete success
criteria.

### 1.5 The macro/micro decomposition and the paper's gibberish results

The framing above distinguishes between *macro* properties of the
residual stream — the ensemble's variance structure, the blunderbuss
shape — and *micro* properties — where any specific token trajectory
goes within that shape. This decomposition is supported by Sarfati et
al.'s null-testing experiments (their §4.2), which we revisit because
they are the paper's most direct evidence for the separation.

The paper feeds two kinds of input through a trained GPT-2: real
language (50-token chunks of Walden), and *gibberish* (50-token
sequences of random tokens from the vocabulary). Three findings:

- **Gibberish trajectories cluster around a path similar to language.**
  The macro bundle shape is preserved. The blunderbuss exists for
  whatever you feed in.
- **Language and gibberish trajectories are linearly separable at every
  layer.** The two input types occupy adjacent but distinguishable
  manifolds in the same hidden space. The blunderbuss has internal
  micro-structure: different input distributions sit at different points
  within the macro envelope.
- **Untrained models produce neither.** Random-init weights give
  straight, parallel trajectories with no flaring. The blunderbuss
  shape doesn't exist at initialization; it is *learned*.

These observations support our framing on three points. The macro
blunderbuss is a property of the trained model, not of its input — it
applies to language and gibberish alike. The macro shape's *cross-section*
contains directions in which input distributions are distinguishable
(the linear-separability result), which is where the micro story
lives. And the macro structure is built by training, not by the
architecture's mathematical form.

We will run a parallel experiment at our scale (§5.4) that uses
gibberish (or, more precisely, shuffled-language input) as a probe to
decompose the macro statistics into FFN-driven contributions (which
should be input-distribution-invariant) and attention-driven
contributions (which should depend on inter-token correlations of the
input). This decomposition is one of the primary additions to Phase 2.

---

## 2. Hypotheses and success thresholds

We frame the project around three concrete hypotheses and the
quantitative thresholds that define what success looks like for each.

### H1: Convergence (Phase 1, already verified for SwiGLU)

The linear flow $R(t)$ converges to a final form during training: the
Frobenius distance $D_k = \lVert R^{(k)} - R^{(\text{final})}\rVert_F$
from checkpoint $k$ to the final checkpoint is monotonically decreasing,
with the standard deviation in the last quarter of training no greater
than 10% of the total reduction $D_1 - D_K$. The Phase 1 SwiGLU pilot
showed H1 PASS on all 4 seeds with ratio $0.043 \pm 0.002$ (threshold
$0.10$). Phase 1 GELU will retest H1; we expect PASS with similar margin.

### H2: Within-variant reproducibility (Phase 1, partial)

The basis-invariant statistics are reproducible across seeds at small
relative spread. Phase 1 SwiGLU established the following baseline:

| Statistic | mean | std | $1.5\times$std | rel. spread |
|---|---:|---:|---:|---:|
| $\lambda$ (paper conv.) | 0.4261 | 0.0048 | 0.0072 | 1.1% |
| $\log\alpha$ (paper conv.) | $-3.277$ | 0.070 | 0.105 | 2.1% |
| $\lambda$ (excl. boundary) | 0.5107 | 0.0048 | 0.0072 | 0.9% |
| $\log\alpha$ (excl. boundary) | $-3.756$ | 0.073 | 0.110 | 1.9% |
| effective rank, middle layers | 492.9 | 15.5 | 23.2 | 3.1% |
| $\langle\|\kappa\|\rangle$ | 1.046 | 0.205 | 0.308 | 19.6% |
| mean isotropy | 0.157 | 0.006 | 0.009 | 3.8% |
| boundary $\Delta\log\alpha$ | $-0.478$ | 0.010 | 0.015 | 2.2% |

The $1.5 \times \text{std}$ column gives the within-variant universality
threshold: in Phase 2, differences smaller than this on a given
statistic are within-variant noise and are not interpretable as
architecture-driven. Differences larger are interpretable.

Phase 1 GELU will re-measure these dispersions. We expect $\lambda$,
$\log\alpha$, effective rank, and isotropy to be similar to the SwiGLU
values, and the kurtosis dispersion to drop substantially (the 19.6%
spread is plausibly inflated by SwiGLU's gating, see §3 below).

### H3 (revised): Causal attribution (Phase 2)

For each architectural design choice varied along a single axis (depth,
width, FFN ratio, normalization, attention configuration), the change in
each basis-invariant statistic relative to the Phase 1 GELU baseline can
be classified as:

- **Within noise** (absolute change < $1.5\times \text{std}_{\text{Phase 1}}$):
  this statistic is robust to this design choice.
- **Single-axis effect** (change ≥ $1.5\times \text{std}_{\text{Phase 1}}$,
  monotonic with the axis): this design choice controls this statistic
  in a specific direction.
- **Non-monotonic effect**: the statistic depends on this design choice
  but not monotonically, indicating either a non-trivial interaction or
  a more complex relationship that requires further analysis.

H3 is the operational success criterion for Phase 2: each of the ~14
basis-invariant statistics is mapped to its responsible design axis
(or to "robust to all tested axes"), giving a causal picture of which
architectural choices push which macro-level properties.

### Originally proposed but dropped

Two hypotheses from the previous project iteration are dropped:

- **Predictability**: the previous Phase 3 proposed forecasting the
  converged linear flow from partial training. This was scoped under the
  assumption that cross-model $R(t)$ alignment would work. Phase 1
  established that it doesn't. Predictability of basis-invariant
  scalars across training is in principle still possible but loses much
  of its interest absent an alignment framework around it.
- **Intervention**: the previous Phase 4 proposed using a predicted
  linear flow as a regularizer or diagnostic during training. Same
  reason — it required functional alignment and is no longer in scope.

If a successor project pursues these, it would need to operate in the
basis-invariant statistic space throughout, which is a substantially
different research design.

---

## 3. Why GELU is the right baseline for Phase 2

Phase 1's first iteration used SwiGLU because it is the production
standard for current frontier models (Llama 2, Llama 3, Gemma, Qwen,
DeepSeek all use SwiGLU or a close gated variant). Replicating the
production setup was the right choice for establishing whether the
framework reproduces under controlled conditions. But for **Phase 2's
ablations**, where the goal is to attribute observed effects to specific
design choices, SwiGLU introduces a confound that's worth removing.

### 3.1 The gating confound

SwiGLU computes its output as
$$\text{output} = \text{silu}(W_g x) \odot (W_u x)$$
where $W_g$ and $W_u$ are two parallel projections of the same input,
and the silu-gated product is then down-projected. The multiplicative
interaction between the two projections has a known consequence: the
output's per-token magnitude depends on the *product* of two learned
quantities, which is more variable than either alone. Specifically:

- Tokens whose pre-activation lands in regions where both $W_g x$ and
  $W_u x$ are large get amplified outputs.
- The gating creates context-sensitive expressivity at the per-token
  level, which is exactly what makes SwiGLU effective for task
  performance.
- But this per-token amplification produces higher-variance contributions
  to the residual stream at the ensemble level, and (because the
  amplification is non-Gaussian) inflates ensemble kurtosis.

Phase 1's measurement of $\langle|\kappa|\rangle = 1.05 \pm 0.21$, with
one seed reaching $1.33$, is consistent with this. Kurtosis was by far
the most disperse statistic across seeds, and the seed-1 outlier
manifested *only* in kurtosis and isotropy — not in $\lambda$,
$\log\alpha$, eval loss, or H1 ratio. This pattern fits a gating-induced
variability story: a small seed-to-seed difference in the
gating-projection alignment is amplified into a measurable kurtosis
difference, while leaving the other macro quantities relatively
unchanged.

### 3.2 What an ungated GELU baseline buys

Plain ungated GELU computes
$$\text{output} = \text{gelu}(W_u x)$$
with a single linear projection followed by the smooth nonlinearity.
The per-token amplification is monotonic in input magnitude and has no
gate-vs-up interaction. The kurtosis contribution from the FFN itself
is much smaller, and the seed-to-seed variation in FFN behavior should
correspondingly drop.

The baseline GELU model is **parameter-matched** to SwiGLU: with
intermediate size scaled to $1.5 \times I_{\text{swiglu}} = 3648$, the
per-block parameter count is identical at $\approx 6.54$ M, and total
model parameter count remains at $\approx 146.4$ M.

The trade is real: GELU is less expressive than SwiGLU at fixed
parameter count, and the GELU model will likely reach a slightly higher
eval loss than the SwiGLU model. This is acceptable because Phase 2's
question is about how the *macro structure responds to design choices*,
not about which choice produces the lowest loss. A slightly worse but
more interpretable baseline is the right trade.

### 3.3 What we expect GELU vs SwiGLU to show

For the Phase 1 GELU re-run, we predict:

- $\lambda$, $\log\alpha$, effective rank, isotropy, boundary anomaly:
  similar to SwiGLU within a few percent. These statistics describe
  ensemble-level geometry and should not be dominated by FFN gating
  details.
- $\langle|\kappa|\rangle$: substantially lower than SwiGLU (perhaps
  $0.5$–$0.8$, vs $1.05$), with much tighter cross-seed spread.
- H1 ratio: similar to SwiGLU.
- Mid-training $\log\alpha$ hump, post-final-norm anomaly: similar
  shape and timing.
- Eval loss: slightly higher than SwiGLU (perhaps 2.93–2.95 vs 2.91),
  reflecting GELU's lower expressivity at matched parameters.

If these predictions hold, Phase 2 launches against the GELU baseline
with confidence that observed cross-design-choice differences are
attributable to the design choice rather than to gating-induced noise.

If GELU surprises us — e.g., if $\lambda$ shifts substantially relative
to SwiGLU — that itself is informative: it would say that the FFN
choice has a larger macro effect than we thought, and Phase 2's
analysis would need to account for both SwiGLU and GELU baselines.

---

## 4. Phase 1: Establish the GELU baseline

### 4.1 Goal

Re-run the 4-seed Phase 1 protocol with the GELU FFN, establishing
within-variant noise floors for all basis-invariant statistics in the
macro-clean regime against which Phase 2 will compare.

### 4.2 Experimental design

| Setting | Value |
|---|---|
| Architecture | Llama-style, 12 blocks, $H = 896$, 14 attention heads |
| FFN | Plain ungated GELU, $I = 3648$ (parameter-matched to SwiGLU $I = 2432$) |
| Tokenizer | Mistral-7B-v0.1 BPE, $V = 32768$ |
| Corpus | FineWeb-Edu sample-10BT |
| Training | 24,000 steps × 64 batch × 1024 ctx = 1.57B tokens |
| Optimizer | AdamW ($\beta_2 = 0.95$), peak LR $3\mathrm{e}{-4}$, cosine schedule, 1000-step warmup |
| Mixed precision | bf16 |
| Seeds | 4 independent (random init + data shuffle) |
| Analysis | 50 log-spaced checkpoints per seed, full Phase 1 pipeline |

The training recipe is identical to the SwiGLU Phase 1 except for the
FFN choice. This preserves every other comparison axis.

The completed SwiGLU runs are kept in their existing locations (e.g.
`phase1_runs/seed_*`) and are not overwritten. GELU runs go to a
parallel `phase1_runs_gelu/seed_*` directory tree.

### 4.3 Measurements

For each seed, the analyzer produces 50 flow files (one per log-spaced
checkpoint). We compute:

- **At convergence** (final checkpoint, all 4 seeds):
  $\lambda$ and $\log\alpha$ in both paper and ours conventions, with
  and without boundary-layer exclusion; per-layer effective rank,
  kurtosis, and isotropy profiles; per-layer successive-layer angle
  profile; post-final-norm anomaly magnitude; eval loss.
- **Across training** (all 50 checkpoints, all 4 seeds):
  H1 convergence ratio; mid-training $\log\alpha$ hump location and
  peak height; emergence trajectory of the post-final-norm anomaly;
  late-training kurtosis trajectory.
- **Within-seed self-consistency**: split-half R-matrix angles as the
  analyzer's noise floor.
- **Cross-seed alignment status check**: a single confirmation run of
  the top-K subspace diagnostic (using cached activations) to verify
  that cross-seed R matrices remain unaligned in the GELU regime as
  they did in SwiGLU — i.e., that the basis-indeterminacy finding from
  the SwiGLU Phase 1 isn't somehow gated-FFN-specific.

### 4.4 Deliverables

A 4-seed dispersion table for all statistics, computed in the GELU
regime, with $1.5 \times \text{std}$ thresholds for each statistic.
This table is the **Phase 2 attribution threshold reference** — every
Phase 2 cross-design-choice difference will be compared against the
appropriate row.

### 4.5 Comparison to SwiGLU

The SwiGLU Phase 1 results are kept as a parallel baseline. The
GELU-vs-SwiGLU comparison at matched seeds gives a clean reading of
how gating affects each statistic, which is itself a Phase 2-relevant
finding. We will report this comparison as a one-shot result alongside
the Phase 1 GELU deliverables, not as a primary aim.

### 4.6 Go/no-go criterion for Phase 2

Phase 2 launches if:

1. H1 PASS on all 4 GELU seeds (ratio < 0.10).
2. Within-variant dispersion on all basis-invariant statistics is within
   2× the SwiGLU values reported above (i.e., the GELU regime is not
   unexpectedly noisy).
3. Eval loss is within 0.05 of SwiGLU (i.e., the GELU model is not
   substantially under-trained — large eval-loss differences would mean
   the comparison baseline isn't trained to the same quality and would
   confound Phase 2's interpretations).
4. The cross-seed alignment status check confirms that R matrices
   remain unaligned in the GELU regime (no surprise).

If any of these fail, Phase 2 is paused for diagnosis. Most likely
failure mode: GELU dispersion turns out higher than SwiGLU on some
statistic, which would suggest that ungating *adds* noise rather than
removing it. That would be a surprising and important finding and
would change Phase 2's protocol.

### 4.7 Resources

| Item | Cost |
|---|---|
| 4 GELU training runs | $4 \times 12 = 48$ GPU-hours |
| Analyzer (auto-runs after training) | included |
| GELU-vs-SwiGLU comparison run | $\sim 1$ GPU-hour |
| **Total** | $\sim 50$ GPU-hours $\approx 2$ days |

---

## 5. Phase 2: Causal ablations of macro structure

### 5.1 Goal

Determine which architectural design choices control which basis-invariant
macro-level properties of the residual stream's variance structure.
Each Phase 2 experiment varies a *single design axis* at a time against
the Phase 1 GELU baseline, isolating the mechanism by which that axis
influences the macro structure.

### 5.2 Why single-axis variation

The previous proposal's Phase 2 plan was to compare four standing
architectures (Llama, Gemma, Qwen, DeepSeek). Each pair differs along
*multiple* axes (attention shape, FFN choice, normalization placement,
position encoding details) simultaneously. Observed differences are
hard to attribute — if Llama and Gemma differ in $\lambda$, is it the
attention, the FFN, the normalization, or an interaction?

Single-axis variation answers attribution by construction. If we vary
depth $L$ while holding everything else fixed and observe $\lambda$
change, depth controls $\lambda$. If we vary FFN intermediate ratio
and observe effective rank change, FFN ratio controls effective rank.

The trade is external validity: these single-axis variants are
artificial points in design space that don't correspond to any
production architecture. We accept this trade because the scientific
content of "which design choice controls which property" is more
valuable than a head-to-head comparison of architectures that already
exist and are widely deployed.

### 5.3 Tier 1: Core experiments (must-do)

Three single-axis variations form the minimum-viable Phase 2:

**A. Depth sweep.** Train at $L \in \{6, 12, 24\}$ blocks, holding $H = 896$
constant and adjusting nothing else. The Phase 1 GELU baseline is $L = 12$.
Total runs: 2 new values × 2 seeds = 4 runs (plus the 4 Phase 1 GELU
seeds serving as the $L=12$ data points). This directly tests the
paper's claim that $\lambda \times L \approx 5.5$ is approximately
conserved across architectures — if the conservation is real, deeper
models should compensate with smaller $\lambda$, and shallower with
larger $\lambda$.

Predictions before running:
- If $\lambda L$ is conserved: $\lambda_{L=6} \approx 0.85$,
  $\lambda_{L=24} \approx 0.22$ (with the boundary-excluded fit).
- If $\lambda$ doesn't depend on $L$: $\lambda$ stays near the Phase 1
  value of $0.43$ at all depths, and $\lambda L$ grows linearly with $L$.
- Intermediate outcomes (partial conservation) are informative about
  the structure of the law.

**B. Width sweep.** Train at $H \in \{448, 896, 1792\}$, holding $L = 12$
and number of attention heads scaled with $H$ (head dimension 64
constant). FFN intermediate scales proportionally to maintain the
SwiGLU-equivalent ratio. Total: 2 new values × 2 seeds = 4 runs.

This tests how the effective rank profile scales with the ambient
hidden dimension. If models use a fixed fraction of available
dimensions, effective rank should scale linearly with $H$. If models
use a fixed absolute number of dimensions, effective rank should be
roughly constant.

Predictions:
- Effective rank scales linearly with $H$: residual stream "fills" a
  fixed fraction of the available dimensions.
- Effective rank saturates at some absolute value: residual stream
  uses only what it needs and ignores the rest.
- An intermediate scaling (e.g., $\sim \sqrt{H}$) would be informative
  about how the model allocates its representational capacity.

**C. FFN intermediate ratio sweep.** Train at intermediate sizes
$I \in \{1.5H, 3H, 4H\}$ (ratios of 1.5, 3, 4 against $H = 896$),
holding the FFN ungated GELU and all else constant. Total: 2 new
values × 2 seeds = 4 runs.

This tests whether the FFN's expressivity (parameter count) is what
drives the effective rank and kurtosis profiles, separately from
gating choice.

Tier 1 total: 12 new runs $\times \sim 12$ GPU-hours = $\sim 144$ GPU-hours
$\approx 6$ days.

### 5.4 Tier 1b: Input-distribution decomposition (FFN vs attention)

This experiment doesn't require new training — it runs the analyzer
with different input sets on models that are already trained. We
include it as part of Tier 1 because it directly tests the FFN/macro
vs attention/macro split that motivates much of the Phase 2 design,
and because it's cheap.

**Motivation.** FFNs operate on token positions independently — the
same nonlinear map applied to whatever the residual stream contains at
position $i$, regardless of what's at other positions. Attention, by
construction, mixes value vectors from other tokens with weights
determined by content-based query-key matching. For inputs with rich
inter-token correlations (real language), attention finds peaked
attention patterns that compose specific past tokens' contributions
into the current token's residual stream — a substantial,
context-sensitive variance contribution. For inputs with no inter-token
correlations (random or shuffled tokens), attention has nothing to
match coherently; query-key dot products are roughly uniform across
positions, and the attention output degenerates to approximately a
uniform average of past values. Approximately averaging many roughly
zero-mean value vectors gives a smaller-magnitude output than peaked
attention does.

The prediction: if we feed an already-trained model two different
input distributions (real language vs shuffled language) and measure
the basis-invariant statistics on each, the FFN's contribution to the
macro structure should be approximately the same (the FFN doesn't know
which input distribution it's seeing), while attention's contribution
should differ. The difference between the two measurements isolates
attention's input-sensitive contribution to the macro structure.

**Experimental design.** For each trained model (Phase 1 GELU
baseline, plus each of the Phase 2 Tier 1 variants), run the analyzer
on three input sets at the final checkpoint:

1. *Real language.* The standard FineWeb-Edu held-out chunks
   (500 chunks × 19 positions = 9500 pilots per layer). This is the
   reference measurement.
2. *Shuffled language.* The same 500 chunks, but with token order
   shuffled within each chunk. This preserves the marginal token
   distribution (and therefore the embedding-level distribution) but
   destroys inter-token correlations. Any macro-statistic difference
   between this and (1) attributes to attention exploiting context.
3. *Random vocabulary.* Uniformly sampled tokens from the top-K
   most-frequent vocabulary positions (matching language's token
   distribution but with no semantic correlations whatsoever). Acts
   as a stronger version of (2) for comparison.

Each input set gives a complete basis-invariant statistic profile.
The differences between (1) and (2), and between (1) and (3),
quantify how much of each macro statistic depends on the model
exploiting inter-token structure via attention.

**Predictions to test:**

- $\lambda$ (variance growth rate): largely input-independent if FFN
  is the dominant driver. Predicted difference: small (within within-
  variant noise floor).
- $\log\alpha$ (variance prefactor): expected to be lower for
  shuffled and random inputs. Predicted difference: substantial,
  perhaps $-0.2$ to $-0.5$ log units.
- Effective rank profile: most likely lower for shuffled and random,
  because attention's contribution to dimensional diversity is reduced.
- Successive-layer angle profile: probably less smooth for shuffled
  and random, because attention output is more uniform and less
  structured.
- Kurtosis profile: this one is harder to predict; gating-free GELU
  models should have similar kurtosis on all input types, but
  inter-token correlations may still affect tail behavior.

If the predictions hold, the FFN/attention decomposition is
operationally validated and gives Phase 2 a second dimension of
attribution: each macro statistic is partitioned into an
FFN-driven (input-invariant) component and an attention-driven
(input-sensitive) component.

If the predictions don't hold — e.g., if shuffled language and real
language give nearly identical macro statistics — that's a striking
finding: the macro structure is largely input-blind, and attention's
contribution to it is essentially a constant offset regardless of
input correlation structure. This would refine our understanding of
what the framework measures.

**Cost.** No new training. Running the analyzer at the final checkpoint
of each model with each input set is approximately 30 minutes on a
5090. For the Phase 1 GELU baseline (4 seeds) and the Phase 2 Tier 1
variants (12 model configurations × 2 seeds = 24 models), plus 3 input
sets each, total cost is approximately $(4 + 24) \times 3 \times 0.5
= 42$ GPU-hours. About 2 days at one 5090, but can run in parallel
with Tier 1 training as soon as each variant completes.

### 5.5 Tier 2: Follow-up experiments (conditional on Tier 1)

The following experiments are launched conditional on Tier 1 results
indicating where the interesting effects lie:

**D. Normalization choice.** RMSNorm (Phase 1 default) vs LayerNorm at
the same model size. Specifically interesting because the post-final-norm
anomaly observed in Phase 1 is likely driven by RMSNorm's scaling-only
behavior; LayerNorm's recentering may produce a different boundary effect.
1 axis change × 2 seeds = 2 runs.

**E. Attention head count.** Vary head count $\{4, 8, 14, 28\}$ at fixed
total head-dimension sum and fixed $H$. Tests whether attention head
configuration affects the macro structure. Prior expectation: it should
not affect the macro much. If confirmed, supports the macro/micro
separation (attention is more micro, FFN is more macro). 3 axis changes
× 2 seeds = 6 runs.

**F. SwiGLU vs GELU one-shot comparison.** At the Phase 1 baseline
($L=12$, $H=896$), train one 2-seed run with SwiGLU instead of GELU.
This is the "gating effect" measurement — what does adding gating to
the FFN do to each macro statistic? 1 axis change × 2 seeds = 2 runs.

Tier 2 total: 10 runs $\times \sim 12$ GPU-hours = $\sim 120$ GPU-hours
$\approx 5$ days. Decision to launch depends on Tier 1 results.

### 5.6 Tier 3: External validity check (conditional)

If Tiers 1 and 2 build up a coherent attribution picture (each macro
statistic mapped to a controlling design axis or designated robust),
launch a single check against a real production-style architecture:
Gemma-2-style with all its differences (GeGLU FFN, hybrid normalization,
sliding-window attention, logit softcapping). 1 architecture × 2 seeds
= 2 runs.

The point of this external-validity check is to see whether the
attribution picture built up from single-axis variations *predicts*
where Gemma will differ from the Phase 1 GELU baseline. If the picture
is correct, the differences should match: change FFN → change kurtosis;
change normalization → change boundary anomaly; etc. If unpredicted
differences appear, the single-axis picture is incomplete (or has
interactions we missed) and the analysis needs more work.

Tier 3: 2 runs $\times \sim 12$ GPU-hours = $\sim 24$ GPU-hours $\approx 1$
day.

### 5.7 Total Phase 2 budget

| Tier | Description | Runs | GPU-hours |
|---|---|---:|---:|
| 1 | Depth, width, FFN ratio (must-do) | 12 | 144 |
| 1b | Input-distribution decomposition (no new training) | 0 | 42 |
| 2 | Norm, heads, gating (conditional) | 10 | 120 |
| 3 | External validity (conditional) | 2 | 24 |
| **Total** | (if all tiers launched) | **24** | **330** |
| Minimum-viable | Tiers 1 + 1b | 12 | 186 |

At one 5090, the full Phase 2 (Tiers 1+1b+2+3) is approximately 14 days
of wall-clock time. Tiers 1 + 1b alone are about 8 days. Tier 1b is
cheap because it requires no new training — only analyzer runs on
already-trained models with different input sets.

### 5.8 Attribution analysis

After Phase 2 runs complete, the attribution analysis is straightforward:

For each basis-invariant statistic $S$ and each design axis $D$ with
baseline value $b$ and variant values $\{v_1, v_2, \ldots\}$:

- Measure $S$ at baseline and at each variant.
- Compute $\Delta S_i = S(\text{variant } i) - S(\text{baseline})$.
- Compare $|\Delta S_i|$ to the Phase 1 GELU within-variant threshold
  $1.5 \times \text{std}(S)$.
- Classify:
  - $|\Delta S_i| < 1.5 \times \text{std}(S)$ for all $i$: **$S$ is robust
    to axis $D$.**
  - Monotonic non-zero $\Delta S_i$ exceeding the threshold: **axis $D$
    controls $S$, in direction X.**
  - Non-monotonic or threshold-crossing-only-at-extremes: **axis $D$
    affects $S$ with non-trivial dependence.**

The result is an attribution matrix: rows are statistics, columns are
design axes, entries are robust / controls / non-monotonic. This matrix
is the deliverable of Phase 2.

---

## 6. Open methodological questions

Before Phase 2 launches, two questions need decisions.

### 6.1 Depth sweep: hold what constant?

When varying $L$, three different things could be held constant:

(a) **Hidden dimension $H$ fixed (parameter count varies).** Easiest to
   implement. Total parameter count grows linearly with $L$. The
   shallow model has fewer parameters and may underfit; the deep model
   has more and may overfit slightly. Eval loss will not be matched
   across depths.

(b) **Total parameter count fixed (width adjusts).** Hold total
   parameter count at 146M by adjusting $H$ inversely with $L$. The
   shallow model has wider $H$ and the deep model has narrower $H$.
   Eval loss roughly matched, but two axes ($L$ and $H$) are
   simultaneously varying.

(c) **Total training compute fixed (token budget adjusts).** Hold
   FLOPs/token × tokens constant. Deeper models use more FLOPs/token,
   so they get fewer tokens. Eval loss roughly matched but training
   duration varies.

The cleanest experimental design depends on the question. For "does
$\lambda L$ conservation hold?" (a) is right — the conservation law is
stated in terms of $L$ and $\lambda$ independently, so we should vary
$L$ alone. For "what's the depth-dependence of efficient training?" (b)
is right. For "what's the scaling-law-aware depth dependence?" (c) is
right.

Our chosen question is the framework's claim about $\lambda L$
conservation, so (a) is the right choice. Parameter count and eval
loss are reported as covariates, not controlled.

### 6.2 What to do about seed count

Phase 1 used 4 seeds; Phase 2 proposes 2 seeds per variant. The
trade-off is:

- 2 seeds per variant: gives a small within-variant std estimate (with
  large uncertainty on the std itself), but lets us cover more axes.
- 3 seeds per variant: better dispersion estimate per variant, but 50%
  more compute per axis.

For Phase 2's purposes, the Phase 1 GELU 4-seed dispersion is the
reference noise floor. Per-variant 2-seed runs are mainly to confirm
that each variant's within-variant noise matches the Phase 1 baseline.
A 2-seed sample is sufficient for this consistency check; for any
variant where the 2-seed dispersion seems anomalously large or small,
we add a third seed.

This is an adaptive seed-count protocol: 2 seeds per variant by
default, third seed conditional on the first two giving discordant
results. Average effective seed count likely ~2.2.

---

## 7. Risks and contingencies

### 7.1 Conceptual risks

**The framework might collapse at small scale.** Phase 1 SwiGLU
established that the framework's basic claims (H1, basis-invariant
reproducibility) hold at 150M. If Phase 1 GELU shows them collapsing
(H1 FAIL, dispersion much larger), the project pivots to "why does the
framework break at this scale?" rather than continuing Phase 2.

**The attribution picture might be empty.** It's possible all single-axis
variations produce changes within the Phase 1 noise floor, meaning no
design axis controls any statistic strongly enough to attribute. This
would be a substantive (negative) finding: the framework's intrinsic
geometry is robust to all the architectural choices we tested. The
project would still publish, just with a different headline.

**The attribution picture might be dense.** The opposite failure: every
axis affects every statistic, with no clean attributions. This would
say the macro structure is a tangled product of all design choices and
there's no clean dependency structure. We'd report the matrix and note
the interactions, but the scientific story would be less crisp.

### 7.2 Methodological risks

**Phase 1 GELU might not reproduce SwiGLU's dispersion magnitudes.**
We expect kurtosis dispersion to drop and others to stay similar. If
several statistics shift unexpectedly, we have a confounded baseline
and need to debug before launching Phase 2. The Phase 1 go/no-go
criteria (§4.6) catch this.

**The depth sweep might confound parameter count.** Held with $H$ fixed,
the depth sweep covaries with parameter count. This is documented but
not controlled. If the depth sweep produces strong effects on
$\lambda$, we add a follow-up varying $L$ at matched parameter count
(via varying $H$) to confirm the effect is depth-attributable.

**Compute budget might overrun.** At 12 GPU-hours per training run, the
budget is sensitive to training-time variation (a 20% slowdown becomes
days of additional wall-clock). Phase 1 SwiGLU ran in approximately the
expected 12 hours per seed; we anchor to that estimate.

### 7.3 Field-trajectory risks

The lines-of-thought framework is relatively new (ICLR 2025). If a
follow-up paper supersedes it with a more powerful framework before
our project publishes, our results may be less impactful. We mitigate
by framing the project around the *macro structure of the residual
stream* rather than around specifically defending the lines-of-thought
framework. The basis-invariant statistics we measure are well-defined
under any framework that characterizes the residual stream's geometry.

---

## 8. Deliverables

By the end of the project, we will produce:

1. **Phase 1 GELU 4-seed dispersion table.** Within-variant noise
   floors for all basis-invariant statistics in the macro-clean GELU
   regime.

2. **SwiGLU-vs-GELU comparison.** One-shot quantification of what
   gating adds to each macro statistic.

3. **Phase 2 attribution matrix.** Each macro statistic mapped to its
   controlling design axis or to "robust." Includes effect-size
   measurements relative to Phase 1 thresholds.

4. **Phase 1 final report.** The completed Phase 1 SwiGLU and GELU
   results, the alignment-failure finding, and the within-variant
   characterization. (Already exists in draft form for SwiGLU; will
   be updated with GELU.)

5. **Phase 2 final report.** Attribution analysis with full
   single-axis results, including the depth/width sweeps that test
   $\lambda L$ conservation directly.

6. **Reusable analyzer.** The full Phase 1 / Phase 2 measurement pipeline
   (data loading, model loading, activation collection, SVD, statistical
   fits, plot generation) packaged for use on any 150M-class Llama-style
   model.

7. **Open methodological commentary.** Notes on what the lines-of-thought
   framework establishes vs leaves ambiguous, including the
   `PAPER_CODE_REVIEW.md` document tracking specific paper-code
   discrepancies and conventions.

The combination — within-variant noise floors, gating effect,
attribution matrix, framework commentary — gives a self-contained
empirical picture of what the basis-invariant statistics describe at
150M scale and which design choices push them.

---

## 9. Relation to prior work and originality

The original lines-of-thought paper measures four pretrained models at
different scales/data/training durations and observes approximate
similarity of basis-invariant statistics. Our pilot extends this in
three ways:

- **Within-variant noise floors**: the original paper has no
  within-variant baselines; we establish them by 4-seed replication.
- **Controlled comparison**: holding training data, recipe, and scale
  constant while varying single architectural axes.
- **Attribution analysis**: mapping each macro statistic to its
  controlling design choice, rather than reporting cross-architecture
  similarity at the aggregate level.

The mechanistic interpretability literature (Anthropic, Conmy et al.,
Olsson et al.) studies transformer internals at the level of
individual circuits and features. The lines-of-thought framework, and
our pilot, operate at a *complementary* level — population-level
ensemble geometry rather than per-token circuits. Both levels are
needed for a complete picture of how trained transformers process
language; this proposal addresses only the macro level.

The neural scaling laws literature (Kaplan et al., Hoffmann et al.)
characterizes loss vs compute. Our work is finer-grained: we
characterize *how the residual stream's macro geometry depends on
architectural design choices*, separately from the scalar loss. The
scaling laws don't tell us which architectural choices push which
internal representations; the present work does.

---

## 10. Honest assessment

This is a pilot study at 150M parameters on a single workstation GPU.
That places real limits on what it can establish — though, as Phase 1
has shown, the 150M scale itself is not one of them.

What the project will and won't do:

- **At our scale**: the project produces a controlled, replicable
  characterization of the lines-of-thought framework's macro
  statistics, an attribution of those statistics to specific
  architectural design choices, and the within-variant noise floors
  that make such attribution well-posed.

- **At larger scales**: we don't test whether the attribution picture
  extends. The findings should not be extrapolated to frontier-scale
  models without a separate study. The 150M scale is appropriate for
  building a causal-attribution picture; whether the same picture
  holds at 7B is an open question that requires different (and much
  more expensive) experiments.

- **Across training recipes**: we use a single recipe (FineWeb-Edu,
  AdamW, cosine schedule). Cross-recipe robustness is not tested.
  If our attribution findings depend sensitively on the data corpus
  or optimizer choice, the analysis would need to be redone for each
  recipe. We have no reason to think they will, but we haven't
  checked.

- **Across architectural families**: Phase 2's tier-1 and tier-2
  experiments are single-axis variations within a Llama-style
  architectural neighborhood. The tier-3 external-validity check
  partially addresses how far the attribution picture transfers to
  production architectures, but it's a single test rather than a
  comprehensive cross-family survey.

- **Predictability and intervention**: the original proposal's plan to
  forecast linear flow from partial training (Phase 3) and intervene
  during training (Phase 4) is dropped. These were premised on
  cross-model R-matrix alignment, which Phase 1 established is
  unworkable. If those questions remain interesting, they would
  require a follow-up project operating entirely on basis-invariant
  scalars — a substantially different research design.

What the 150M scale *enables* that motivated the choice in the first
place: multi-seed dispersion measurements, full training-dynamic
checkpointing, and single-axis ablations across the design space.
These were not feasible at the larger scales the paper studies and
are the reason the project's attribution claims are well-posed at
all. The 150M scale is the project's enabling feature, not a
concession.

---

*End of proposal.*
