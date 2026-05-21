# Phase 1 pilot: results, findings, and protocol for Phase 2

**Document status:** v1 (final Phase 1 report)
**Audience:** thesis/paper reviewer; future-self; Phase 2 launch reference
**Pilot scope:** four-seed replication of an extended Lines of Thought analysis at 150M Llama-architecture scale
**Companion documents:** `PAPER_CODE_REVIEW.md` (v4) for paper/code-level notes on Sarfati et al. (ICLR 2025)

---

## 1. Introduction and framing

### 1.1 What Phase 1 is testing

Sarfati et al. (ICLR 2025, "Lines of Thought") describe a framework in which the
residual stream of a transformer admits a low-dimensional linear-Gaussian
description across layers. They show, for several large pretrained models
(GPT-2 medium, Llama-2-7B, Mistral-7B, Pythia-12B), that:

- The per-coordinate residual variance accumulated by predicting from layer
  $t$ to layer $t + \tau$ scales as $\sigma^2 \sim \alpha\,\tau^\lambda$, a
  power law in the layer offset.
- The residual stream geometry is recoverable from a centered SVD on
  activations at a small number of "pilot" sequence positions.
- The recovered linear flow $R(t)$ — the per-layer principal-direction basis
  — has a smooth trajectory through hidden space, with successive-layer
  rotations of roughly tens of degrees.

The paper does not address training dynamics directly: its measurements are
on fully-trained models, treating each as a single converged snapshot. It
also does not measure within-variant (seed-to-seed) reproducibility — every
model in the paper is a single public release.

Phase 1 of this pilot tests a strengthened version of the framework. We
replace a single snapshot per model with a **trajectory of 50 log-spaced
checkpoints from training**, and we replace a single seed per architecture
with **four independent seeds**. The hypotheses being tested are:

- **H1 (convergence).** As training progresses, the recovered linear flow
  $R(t)$ converges to a final form. Formally: the Frobenius distance from
  $R(t)$ at checkpoint $k$ to $R(t)$ at the final checkpoint declines
  monotonically, with the last-quarter standard deviation no larger than
  10% of the total reduction. If H1 fails, the framework describes a moving
  target rather than a learned property.

- **H2 (within-variant reproducibility).** The basis-invariant statistics
  ($\lambda$, $\log\alpha$, effective rank, kurtosis, isotropy) at
  convergence are reproducible across seeds within tight bounds. The
  spread sets the noise floor for any cross-variant comparison in Phase 2.

- **H3 (alignment viability).** The vocabulary-anchored Procrustes alignment
  the proposal commits to recovers a small residual on cross-seed pairs
  (identical architecture and recipe, different random init). If H3 fails
  here, it cannot succeed in the harder cross-variant case in Phase 2, and
  the Phase 2 comparison protocol must avoid R-matrix-level comparison.

Phase 1 succeeds when:

1. H1 has a clear PASS/FAIL verdict on all four seeds.
2. H2 is measured quantitatively, with within-variant dispersion reported
   for every basis-invariant statistic.
3. H3 is resolved — either embedding-space alignment works (PASS), or it
   doesn't and we know why and what to do about it (FAIL with diagnosis).
4. Any reproducible structural features (training-dynamic regimes,
   trajectory features, boundary-layer effects) are characterized to a
   level that allows Phase 2 to test their universality.

### 1.2 Deliberate methodological departures from Sarfati et al.

The Phase 1 pipeline differs from the paper's published code on several
points. Each departure is intentional and is documented for replicability.
The full list with rationale is in `PAPER_CODE_REVIEW.md`; the short version:

- **Layer convention.** We index $t = 0, 1, \ldots, L+1$ where layer 0 is
  the post-embedding state, layers $1, \ldots, L$ are post-block-output
  states, and layer $L+1$ is the post-final-norm state. The paper indexes
  only blocks. Our convention adds two states the paper omits (the
  embedding state, and the post-final-norm state). This matters because
  one of those added states — the post-final-norm state — has unusual
  variance behavior that the paper doesn't observe.

- **Variance-fit convention.** The paper computes $\log\alpha$ as the
  *mean of per-coordinate log-variances*; we additionally report the
  alternative convention, *log of mean per-coordinate variance*. These
  two conventions differ by a Jensen gap that is 0.50 at initialization
  and decays to 0.03 by the end of training. All headline numbers are
  reported in both conventions throughout; "paper convention" means
  mean-of-log, "ours" means log-of-mean.

- **Pilot scale.** Following the paper, we use 9,500 pilot activations
  per seed (500 held-out chunks × 19 pilot positions per chunk). The
  pilot positions are $\{50, 100, 150, \ldots, 950\}$ within each
  1024-token chunk. This is well below the paper's reported pilot scale
  (10⁵–10⁶ for the larger models), but is adequate for stable SVD
  recovery at our $H = 896$.

- **Centering.** Activations are centered per-layer before SVD. The paper
  is ambiguous on this; we follow the convention that the residual stream
  is mean-centered before recovering its principal directions.

- **Cross-seed alignment.** The paper does not perform cross-model
  alignment. The proposal commits to vocabulary-anchored Procrustes
  alignment as the cross-variant comparison mechanism in Phase 2. We
  test this mechanism in Phase 1 on the cross-seed (identical-architecture)
  case as the easiest possible validation.

### 1.3 What is *not* in scope for Phase 1

- Cross-variant comparison (Phase 2).
- Causal claims about *why* the framework's basis-invariant statistics
  hold (Phase 3 if applicable).
- Comparison to the paper's published linear-flow trajectories. We
  observe a ~30 log-unit gap between our values and the paper's released
  `trajectories.npy` for Llama-2-7B; this is an open paper-code issue,
  documented in `PAPER_CODE_REVIEW.md` §12 and unrelated to Phase 1
  results.

---

## 2. Setup

### 2.1 Architecture

The pilot model is a 146.4M-parameter Llama-style transformer trained from
scratch.

| Property | Value |
|---|---|
| Hidden size $H$ | 896 |
| Number of transformer blocks $L$ | 12 |
| Number of attention heads | 14 (head dim 64) |
| Number of KV heads | 2 (GQA) |
| FFN intermediate size | 2432 |
| Activation | SwiGLU |
| Position encoding | RoPE (base 10000) |
| Normalization | RMSNorm (pre-norm) |
| Tied embeddings | Yes |
| Vocabulary size $V$ | 32768 |
| Tokenizer | Mistral-7B-v0.1 |
| Max context | 4096 (training context 1024) |

The architecture deliberately mirrors small Llama-family models. We
chose this family rather than GPT-2 because the Llama family is what
modern pretrained models look like (RMSNorm, RoPE, SwiGLU, GQA), and
because Phase 2 plans to compare to architectural variants in the same
family. The 150M scale is the largest that fits comfortably in 24 GB
of GPU memory at batch 64 × context 1024, which is the maximum we can
train on the available hardware (one RTX 5090).

Layer states recovered by the analyzer: $L_\text{total} = 14$, comprising
the post-embedding state (layer 0), the 12 block-output states
(layers 1-12), and the post-final-norm state (layer 13). This 14-state
convention is used throughout this report.

### 2.2 Training recipe

| Property | Value |
|---|---|
| Corpus | FineWeb-Edu (sample-10BT subset) |
| Total training tokens | 1.57B |
| Total steps | 24,000 |
| Batch size | 8 micro × 8 grad-accum = 64 sequences |
| Context length | 1024 tokens (65,536 tokens/step) |
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1) |
| LR schedule | linear warmup (1000 steps) → cosine decay to 10% |
| Peak LR | 3e-4 |
| Gradient clip | 1.0 |
| Mixed precision | bf16 (autocast for the forward/backward; fp32 master weights) |
| Held-out set | 500 chunks of 1024 tokens (500K tokens) |

Each of the four seeds uses an independent random initialization (Llama
default: $\mathcal{N}(0, 0.02)$ for embedding and linear weights) and an
independent training-data shuffle order. All other hyperparameters are
identical across seeds. The data shuffle is deterministic given the seed,
which is critical for the cross-seed activation alignment to be well-posed
(see §7.3).

Each seed completed training in approximately 12 hours on a single RTX 5090.

### 2.3 Analysis pipeline

For each seed, we save 50 checkpoints at log-spaced training steps from
step 100 to step 24,000. At each checkpoint we run the analyzer:

1. **Load the model state.** Restore weights from the checkpoint.
2. **Collect activations.** Run forward inference on the 500 held-out
   chunks. At each chunk, record the residual-stream hidden state at
   the 19 pilot positions $\{50, 100, \ldots, 950\}$, for each of the
   14 layer states. Total: $500 \times 19 = 9500$ pilot activations
   per seed per checkpoint, each a vector in $\mathbb{R}^{896}$.
3. **Center per layer.** Subtract the per-layer mean.
4. **SVD per layer.** Compute the centered SVD of the $9500 \times 896$
   activation matrix at each layer state. The right singular vectors
   form the rows of the recovered linear flow $R(t)$; the per-coordinate
   residual variances $\Sigma_t$ come from the squared singular values.
5. **Pairwise residual variances.** Compute $\sigma^2_{t, t+\tau}$ — the
   variance unexplained when predicting layer $t + \tau$ from layer $t$
   by ordinary least squares — for all pairs.
6. **Fit the variance-scaling law.** Fit $\log\sigma^2 = \log\alpha +
   \lambda \log\tau$ to the diagonal of the pairwise-variance matrix (the
   endpoint variances), in both conventions.
7. **Save** the flow and its derived statistics as `flow_step_NNNNNNNN.npz`
   for downstream analysis.

The analyzer is deterministic given fixed pilots; running it twice on
the same checkpoint gives identical output. The full 4-seed analysis
produces 200 flow files plus per-seed plots in approximately 1.5 hours
of additional GPU time across the full pilot.

### 2.4 Pilot deliverables

This report covers the following measurements:

- Four seeds trained to step 24,000, eval loss in [2.9062, 2.9102]
  (range 0.004).
- 200 flow files (50 checkpoints × 4 seeds), each with the basis-invariant
  statistics in both conventions.
- Cross-seed dispersion analysis for $\lambda$, $\log\alpha$, effective
  rank, kurtosis, isotropy.
- Boundary-layer-scope sensitivity analysis with and without the embedding
  and post-final-norm states.
- Cross-seed Procrustes alignment analysis (embedding-space, with the
  full vocabulary and with a top-K-by-norm anchor subset) and
  activation-space (per-layer) variants, plus a subspace-resolution
  diagnostic that resolves the alignment failure.
- Within-seed split-half R-matrix self-consistency analysis as the
  alignment-diagnostic noise floor.

---

## 3. H1: Convergence of the linear flow

### 3.1 Criterion

Let $D_k = \lVert R^{(k)} - R^{(\text{final})}\rVert_F$ be the
summed-over-layers Frobenius distance from checkpoint $k$ to the final
checkpoint, where $R^{(k)} \in \mathbb{R}^{14 \times 896 \times 896}$ is
the stack of recovered $R$ matrices at all 14 layer states. The H1 PASS
criterion is:

$$
\frac{\operatorname{std}_{\text{last } 25\%}(D_k)}{D_1 - D_K} \le 0.10
$$

That is, after the linear flow has finished moving, the residual jitter
in its position should be no larger than 10% of the total distance
traveled during training. This is a strong criterion: it requires not
just that $D_k$ is small at the end of training, but that it is
*stably* small relative to the dynamics seen during training.

### 3.2 Results

H1 passes on all four seeds, with margins to spare:

| Seed | Last-quarter std of $D_k$ | Total reduction $D_1 - D_K$ | Ratio | Verdict |
|---|---:|---:|---:|:---:|
| seed 0 | 40.05 | 876.30 | 0.046 | PASS |
| seed 1 | 38.02 | 909.57 | 0.042 | PASS |
| seed 2 | 39.24 | 896.77 | 0.044 | PASS |
| seed 3 | 38.97 | 903.56 | 0.043 | PASS |
| **mean** | **39.07** | **896.55** | **0.0436** | — |
| **range** | **2.03** | **33.27** | **0.004** | — |

The range across seeds is 0.004 on a ratio whose threshold is 0.10 — the
H1 verdict is robust to the choice of any reasonable threshold above
about 0.06. The total reduction $D_1 - D_K \approx 897$ corresponds to
the integrated distance the linear flow travels through hidden space
during training; the residual last-quarter std of about 40 means the
flow has nearly stopped moving by the time training completes.

### 3.3 Sub-finding: loss convergence vs flow convergence

We track both the held-out eval loss $\mathcal{L}_k$ and the normalized
flow distance $D_k / D_1$ as functions of training step. Both are
monotonically decreasing, both flatten in the last quarter of training,
but they don't decline on the same schedule. The eval loss flattens
gradually; the flow distance has a more pronounced kink, decreasing
rapidly until step ~5000 and then more slowly.

This is consistent with the hypothesis that the linear flow geometry
locks in earlier than the model's full loss performance — the residual
stream's *coordinate structure* converges before its *prediction
accuracy* does. Phase 1 doesn't probe this further but the timing
relationship is worth flagging for Phase 2: comparison metrics
derived from the converged flow are stable from approximately step
~10,000 onward, well before training ends.

---

## 4. Basis-invariant statistics: within-variant dispersion

This section answers H2: how reproducible are the framework's
basis-invariant statistics across seeds at the same architecture and
recipe?

For Phase 2 cross-variant comparison, we need within-variant dispersion
on every statistic that Phase 2 will use. The Phase 1 dispersion sets
the noise floor; only differences larger than this floor (operationally,
$1.5 \times \text{std}$) can be attributed to architectural differences
in Phase 2 rather than to within-variant noise.

### 4.1 Variance scaling rate $\lambda$

At the final checkpoint, in the paper convention (mean of per-coordinate
log-variances), the variance scaling exponent $\lambda$ across seeds is:

| Seed | $\lambda$ (paper) | $\lambda$ (ours) |
|---|---:|---:|
| seed 0 | 0.4256 | 0.4418 |
| seed 1 | 0.4223 | 0.4383 |
| seed 2 | 0.4235 | 0.4395 |
| seed 3 | 0.4330 | 0.4490 |
| **mean** | **0.4261** | **0.4422** |
| **std** | **0.0048** | **0.0048** |
| **range** | **0.0107** | **0.0107** |
| **$1.5\times$std** | **0.0072** | **0.0072** |
| **relative spread** | **1.1%** | **1.1%** |

The two conventions differ by a constant offset (about $+0.016$ for
our convention vs paper) at convergence, but have identical std and
range across seeds. **$\lambda$ is reproducible to 1.1% relative across
seeds.** This is the tightest dispersion we measure for any statistic.

The seed 3 value ($\lambda_{\text{paper}} = 0.4330$) is the high-side
outlier; the other three seeds cluster in [0.4223, 0.4256]. This is
consistent with a single sample slightly outside the main cluster, not
with a bimodal within-variant distribution.

### 4.2 Variance prefactor $\log\alpha$

| Seed | $\log\alpha$ (paper) | $\log\alpha$ (ours) |
|---|---:|---:|
| seed 0 | $-3.298$ | $-3.266$ |
| seed 1 | $-3.193$ | $-3.152$ |
| seed 2 | $-3.259$ | $-3.224$ |
| seed 3 | $-3.360$ | $-3.330$ |
| **mean** | **$-3.277$** | **$-3.243$** |
| **std** | **0.070** | **0.075** |
| **range** | **0.168** | **0.178** |
| **$1.5\times$std** | **0.105** | **0.112** |

The dispersion on $\log\alpha$ is substantially larger than on $\lambda$:
about 0.07 std vs 0.005 for $\lambda$. This is expected — $\log\alpha$
is more sensitive to small differences in the residual-variance
spectrum at the high-$\tau$ end of the fit, where one or two boundary
layer states dominate (see §5).

Seed 3 is again the most extreme (most negative) and seed 1 is the
least extreme. The 4-seed standard deviation grew slightly compared
to the 3-seed measurement reported in earlier iterations: $\text{std}_{n=3}
= 0.058$ vs $\text{std}_{n=4} = 0.070$, both in the paper convention.
This reflects seed 3 sitting below the cluster of seeds 0/1/2 by
about 0.10 log units.

### 4.3 Effective rank

The effective rank is the exponential of the entropy of the
normalized squared singular value distribution: $\text{eff}(\Sigma) =
\exp\left(-\sum_i p_i \log p_i\right)$ with $p_i = \sigma_i^2 /
\sum_j \sigma_j^2$. It measures how many dimensions of the residual
stream are meaningfully active.

At the final checkpoint:

| Layer | mean eff. rank | std | range | $1.5\times$std |
|---|---:|---:|---:|---:|
| layer 0 (post-embed) | 159.84 | 3.41 | 8.08 | 5.12 |
| layer 6 (mid) | 492.92 | 15.46 | 33.55 | 23.19 |
| mean across all 14 | 376.89 | 7.83 | 16.57 | 11.75 |

The effective rank profile is heavily layer-dependent: the boundary
layers (post-embedding and post-final-norm) have effective ranks near
160 — about $H/5$ — while the middle layers reach effective ranks near
500, well over half of $H = 896$. The residual stream becomes more
isotropic as it passes through transformer blocks, then collapses back
to a lower effective rank at the post-final-norm state.

This profile shape is highly reproducible across seeds. The
range/mean ratio at the middle-layer maximum is 7%, comparable to the
$\lambda$ dispersion. At the boundary layers it is even tighter (5%).

### 4.4 Kurtosis

Per-coordinate residual kurtosis (excess kurtosis: 0 = Gaussian,
positive = heavy-tailed) is the most disperse statistic we measure
across seeds.

At the final checkpoint (paper convention, $\langle|\kappa|\rangle$):

| Seed | $\langle\kappa\rangle$ (signed) | $\langle\|\kappa\|\rangle$ |
|---|---:|---:|
| seed 0 | 0.871 | 0.871 |
| seed 1 | 1.334 | 1.334 |
| seed 2 | 1.045 | 1.045 |
| seed 3 | 0.932 | 0.932 |
| **mean** | **1.046** | **1.046** |
| **std** | **0.205** | **0.205** |
| **range** | **0.463** | **0.463** |
| **$1.5\times$std** | **0.308** | **0.308** |

The signed mean kurtosis $\langle\kappa\rangle$ and the absolute-mean
$\langle|\kappa|\rangle$ agree to four decimal places across all
seeds and all checkpoints. This indicates that per-coordinate
kurtosis is **uniformly one-sided positive** at every layer in every
seed — there are no negative-kurtosis coordinates. This rules out a
hypothesis we considered earlier, that the paper's mean-of-log
kurtosis convention vs our naive mean-kurtosis convention would
diverge because of negative-kurtosis cancellation. They don't
diverge, because there are no negative kurtosis coordinates to
cancel.

Seed 1 is the kurtosis outlier ($\langle|\kappa|\rangle = 1.334$ vs
~0.87–1.05 for seeds 0, 2, 3). The 4-seed std of 0.205 is dominated
by this seed-1 contribution; without seed 1, the 3-seed std would be
0.088. We don't have a mechanistic explanation for the seed-1
anomaly, but note that it doesn't appear in $\lambda$, $\log\alpha$,
effective rank, eval loss, or H1 ratio — only in kurtosis (and
secondarily in isotropy, see below). Seed 1's training landed at a
slightly more heavy-tailed residual distribution without affecting
other downstream behavior.

### 4.5 Isotropy

We measure isotropy as the standard deviation of $\log\sigma_i^2$
across coordinates $i$ within a layer, averaged across layers. Small
isotropy values indicate near-Gaussian-isotropic residual structure;
large values indicate strong anisotropy with a few dominant
directions.

At the final checkpoint, mean across the 14 layer states:

| Seed | mean isotropy |
|---|---:|
| seed 0 | 0.154 |
| seed 1 | 0.165 |
| seed 2 | 0.153 |
| seed 3 | 0.154 |
| **mean** | **0.157** |
| **std** | **0.006** |
| **range** | **0.012** |

Seed 1 is again the most extreme (most anisotropic), consistent with
its high kurtosis. The dispersion is small enough that for Phase 2
purposes isotropy is a useful comparator: $1.5\times\text{std} \approx
0.009$, so cross-variant differences in isotropy above about 0.01
would be informative.

### 4.6 Summary: within-variant dispersion table

The following table gives the within-variant dispersion for every
statistic we'll use in Phase 2:

| Statistic | mean | std | $1.5\times$std | relative spread |
|---|---:|---:|---:|---:|
| $\lambda$ (paper) | 0.4261 | 0.0048 | 0.0072 | 1.1% |
| $\log\alpha$ (paper) | $-3.277$ | 0.070 | 0.105 | 2.1% |
| $\lambda$ (paper, boundary-excl.) | 0.5107 | 0.0048 | 0.0072 | 0.9% |
| $\log\alpha$ (paper, boundary-excl.) | $-3.756$ | 0.073 | 0.110 | 1.9% |
| mean effective rank | 376.9 | 7.83 | 11.75 | 2.1% |
| effective rank, middle layers | 492.9 | 15.5 | 23.2 | 3.1% |
| $\langle\|\kappa\|\rangle$ (paper conv.) | 1.046 | 0.205 | 0.308 | 19.6% |
| mean isotropy | 0.157 | 0.006 | 0.009 | 3.8% |
| eval loss | 2.908 | 0.0018 | 0.0028 | 0.06% |
| H1 ratio | 0.0436 | 0.0016 | 0.0025 | 3.6% |

Two patterns stand out. The first is that $\lambda$, isotropy, and
the effective rank profile all have within-variant relative spreads of
about 1–4%; these are the best-reproducible quantities at our scale.
The second is that **kurtosis is anomalously disperse** (19.6% relative
spread), dominated by the seed-1 outlier. Phase 2 should treat
kurtosis as a soft signal — substantial cross-variant differences in
kurtosis (above about $\pm 0.3$) are informative, but smaller
differences may be within-variant noise.

---

## 5. Boundary-layer scope

This section answers a question the paper does not address: do the
embedding state (layer 0) and the post-final-norm state (layer $L+1 =
13$) participate in the same variance-scaling law as the inner layer
states?

### 5.1 Measurement

We refit the variance-scaling law $\log\sigma^2_{t+\tau} = \log\alpha +
\lambda \log(t+\tau)$ at the final checkpoint, with and without the
boundary layer states included in the fit. The difference
$\Delta\log\alpha = (\log\alpha_{\text{excluded}} -
\log\alpha_{\text{all}})$ and the corresponding $\Delta\lambda$ tell us
how much the boundary layers pull the global fit.

Paper convention, final checkpoint:

| Seed | $\log\alpha$ all-layer | $\log\alpha$ excl. boundary | $\Delta\log\alpha$ | $\Delta\lambda$ |
|---|---:|---:|---:|---:|
| seed 0 | $-3.298$ | $-3.786$ | $-0.488$ | $+0.085$ |
| seed 1 | $-3.193$ | $-3.659$ | $-0.467$ | $+0.084$ |
| seed 2 | $-3.259$ | $-3.743$ | $-0.483$ | $+0.085$ |
| seed 3 | $-3.360$ | $-3.835$ | $-0.474$ | $+0.084$ |
| **mean** | **$-3.277$** | **$-3.756$** | **$-0.478$** | **$+0.085$** |
| **range** | — | — | **0.021** | **0.001** |

The boundary effect is extremely reproducible: $\Delta\log\alpha$ has a
range of 0.021 across four independent seeds — about 4.4% of its absolute
value. The corresponding $\Delta\lambda$ has a range of just 0.001, two
orders of magnitude tighter than the $\Delta\lambda$ value itself.

### 5.2 Mechanical interpretation

Looking at the per-layer residual-variance scatter (sample shown for
seed 0; same pattern in all seeds):

- Layer 1 (post-embedding's first block output) is the lowest-$\tau$
  point. Its log-variance sits roughly 0.7 units below the inner-layer
  fit.
- Layer 13 (post-final-norm) is the highest-$\tau$ point. Its log-variance
  sits roughly 1.8 units below the inner-layer fit.
- The 11 inner layer states (layers 2–12) align tightly along a single
  line.

Both boundary layers are outliers, but the post-final-norm anomaly is
larger and acts on a larger-$\tau$ point. The fit is least-squares in
log–log, so the post-final-norm anomaly has more leverage on the slope.
Excluding both boundaries thus produces a steeper $\lambda$ (about $+0.085$
in both conventions) and a more-negative $\log\alpha$ (about $-0.48$).

This pattern is **not present at initialization**. At step 100, both
boundary layers fall close to the inner-layer line; the $\Delta\log\alpha$
is near zero. The boundary anomaly emerges during training: it is roughly
zero at step 100, reaches half its final magnitude by step 2000, and
plateaus to its final value by step 5000. This means the boundary
layer effect is a **learned phenomenon**, not a fixed structural
property of the architecture.

### 5.3 Implication for cross-architecture comparison

The boundary-excluded $\log\alpha$ at convergence is $-3.756 \pm 0.073$
across the four seeds (paper convention). Comparing to Sarfati et al.'s
published values:

| Model | $\log\alpha$ (paper, all layers) | Closes gap by |
|---|---:|---:|
| GPT-2 medium | $-0.45$ | — (architecturally distant) |
| Llama-2-7B | $-5.40$ | 22% of the all-layer gap |

The all-layer gap to Llama-2-7B is 2.12 log units; the boundary-excluded
gap is 1.64 log units. Excluding the boundary effect closes 22% of the
gap, suggesting that part of the GPT-2-vs-Llama-2 difference is
accounted for by *boundary layer geometry*, not by the inner-layer
variance scaling. The remaining 78% is presumably attributable to
scale and training duration.

The gap to GPT-2 medium *widens* under boundary exclusion (because
GPT-2 has a different architectural family — LayerNorm, learned positional
embeddings). This is consistent with the boundary effect being
specific to RMSNorm-and-RoPE architectures.

The Phase 2 comparison protocol (§8) tracks both all-layer and
boundary-excluded values as separate statistics, because they answer
different questions: all-layer is what the paper reports, boundary-excluded
is what the inner-layer variance scaling is.

### 5.4 Caveat: $\lambda \times L$ conservation

The paper observes that $\lambda \times L \approx 5.5$ across the four
architectures it tests. Our all-layer $\lambda \approx 0.42$ × $L = 12$
gives $\lambda L \approx 5.1$ — within the paper's observed range, by
coincidence.

But under boundary exclusion, $\lambda$ rises to $\approx 0.51$, giving
$\lambda L \approx 6.1$. If $\lambda L \approx 5.5$ is a real universality,
it is a universality of the *all-layer* fit, not the inner-layer fit. We
can't tell from a single architectural variant whether the paper's
observed $\lambda L$ relationship would also hold under boundary
exclusion. Phase 2 will test this directly across architectures.

---

## 6. Structural reproducible features

Beyond the at-convergence dispersion in §4, the four-seed analysis reveals
several training-dynamic features that reproduce in shape and magnitude
across seeds. Each of these is a candidate "universal" feature whose
cross-architecture replication Phase 2 will test.

### 6.1 Mid-training $\log\alpha$ hump

In all four seeds, $\log\alpha$ is *not* monotonic during training. It
starts near $-3.3$ at step 100, drops slightly to near $-3.5$ at step ~300,
then rises through a broad peak centered around step 5000, and finally
falls back to its end-of-training value of $\approx -3.28$.

| Seed | Peak step | Peak $\log\alpha$ (paper) |
|---|---:|---:|
| seed 0 | 5014 | $-2.06$ |
| seed 1 | 4483 | $-2.03$ |
| seed 2 | 5014 | $-2.05$ |
| seed 3 | 5014 | $-2.20$ |

Peak step locations agree to within one log-spaced checkpoint interval
across seeds; peak heights agree to within 0.17 log units. The hump
is a real cross-seed feature.

Mechanistically, the hump corresponds to a window in training where
the per-coordinate residual variance at large $\tau$ has its highest
value relative to the local linear trend. The residual stream
geometry is most "spread out" during this window, and then contracts
toward its final form. This is consistent with the loss-vs-flow
analysis in §3.3 — the flow's coordinate structure is still being
sculpted at step ~5000, even though the eval loss has substantially
declined by that point.

For Phase 2 this is a candidate universal feature: cross-architecture
models trained with similar recipes should show the same hump shape
and timing if the framework's predictions are scale-invariant. Phase
2 will measure peak step (relative to total training steps) and peak
height across architectures.

### 6.2 $R(t)$ trajectory geometry

The successive-layer principal angle profile — the mean top-10
principal angle between $R(t)$ and $R(t+1)$ — is highly reproducible
across seeds:

- Layer 0 → layer 1: ~80° (large rotation as we pass from the
  post-embedding state through the first block)
- Layer 1 → layer 2: ~32° (still a substantial rotation early on)
- Layers 2-10 → next: ~30° steady-state (the "trajectory" interior)
- Peak rotation around layer 10 → layer 11: ~35°
- Layer 12 → layer 13: ~8° (the post-final-norm state is geometrically
  close to the last block output)

This pattern is identical in all four seeds to within ~2-3° at every
transition. The post-final-norm angle is particularly notable — it
says that the final RMSNorm does not significantly rotate the
residual stream's principal directions; it primarily rescales them.
This is consistent with RMSNorm's mathematical definition (scaling by
a learned diagonal matrix after L2-normalization).

The R(t) trajectory geometry is a property of the trained model's
residual-stream coordinate frame, not a property of the data being
processed. As such it is also a candidate cross-variant universal.

### 6.3 Post-final-norm anomaly emergence

§5.2 noted that the post-final-norm state's log-variance sits ~1.8
log units below the inner-layer fit at convergence, while at
initialization (step 100) the gap is near zero. We measured the
emergence trajectory across all 50 checkpoints:

| Step | Post-final-norm gap (paper conv.) |
|---|---:|
| 100 | $-0.05$ |
| 300 | $-0.32$ |
| 1000 | $-0.96$ |
| 2000 | $-1.51$ |
| 5000 | $-1.80$ |
| 10000 | $-1.82$ |
| 24000 | $-1.83$ |

The anomaly reaches near-final magnitude by step ~5000, then plateaus.
This is the same timing window as the $\log\alpha$ hump peak (§6.1) and
the broad flow-convergence kink (§3.3). All three features occupy the
same "phase 2" window in training where the linear-flow geometry is
being sculpted.

### 6.4 Late-training kurtosis rise

Per-coordinate residual kurtosis bottoms out at $\approx 0.35$
around step ~2000 and rises monotonically thereafter to its
end-of-training value of $\approx 1.05$ (mean across seeds 0, 2, 3;
seed 1 reaches $\approx 1.33$, see §4.4).

For seeds 0, 2, 3 the kurtosis trajectory is smoothly monotonic. For
seed 1, the trajectory shows a sharper acceleration past step ~13,000,
ending substantially higher than the other three. The kurtosis at
intermediate steps (5000–13000) is similar across all four seeds;
the divergence is entirely in the last 11,000 training steps.

This suggests that the seed-1 kurtosis anomaly is a late-training
phenomenon — something specific to seed 1's optimization trajectory
that pushed the residual distribution toward heavier tails in the
final phase of training, without affecting other measured statistics.
We have no further mechanistic explanation. For Phase 1 purposes,
seed 1 is reported as a single-seed kurtosis outlier; the cross-seed
dispersion on $\langle|\kappa|\rangle$ that we report in §4.4 is
inclusive of this outlier.

### 6.5 Mid-training $\Sigma$-distance bump

The normalized flow-distance trajectory (Frobenius distance to the
final checkpoint, normalized to start at 1.0 at step 100 and end at
0.0 at step 24000) shows a small bump centered around steps
5000-10000. In all four seeds the normalized distance reaches
about 0.12 in this window before resuming its decline to 0.

This bump is co-located in training time with the $\log\alpha$ hump
(§6.1) and the post-final-norm anomaly emergence completion (§6.3).
It is a small effect — about 12% of the total reduction is "given
back" in this window before the final convergence — but it is
reproducible across seeds and adds further evidence that the
training process has a distinct mid-training phase in which the
residual-stream geometry is restructuring.

### 6.6 Summary of structural features

The four seeds agree on:

1. A $\log\alpha$ hump with peak in steps [4500, 5050] and peak height
   in [-2.20, -2.03].
2. An $R(t)$ trajectory geometry: ~80° at layer 0→1, ~30° steady-state,
   ~8° at layer 12→13.
3. A post-final-norm anomaly that emerges between steps ~400 and ~5000
   and plateaus at $-1.8$ log units below the inner-layer fit.
4. A late-training kurtosis rise from ~0.35 at step 2000 to ~1.05 at
   step 24000 (one seed: 1.33).
5. A mid-training normalized flow-distance bump around steps 5000-10000
   reaching ~0.12.

All five features are *not* visible at initialization and emerge
during training. They are all *training-recipe-specific* in the sense
that the FineWeb-Edu corpus and AdamW-cosine recipe drive them; we
can't tell from Phase 1 alone whether they would replicate under a
different corpus or optimizer. Phase 2 holds the recipe fixed and
varies architecture, so these features become candidate
cross-architecture universals.

---

## 7. Cross-seed R-matrix alignment fails

This section answers H3. The verdict is FAIL: the proposal's
vocabulary-anchored Procrustes alignment does not recover a usable
cross-seed correspondence between R matrices at our scale, and **no
alternative orthogonal alignment we tried works either**. The Phase 2
comparison protocol must avoid R-matrix-level comparison.

### 7.1 Embedding-space Procrustes fails on the full vocabulary

We extracted the input-embedding matrix $E \in \mathbb{R}^{32768 \times 896}$
from each seed's final checkpoint. For each ordered pair (A, B), we computed
$Q$ minimizing $\lVert E_A Q - E_B \rVert_F$ via orthogonal Procrustes,
then used $Q$ to transport $R_A$ into seed B's coordinate frame.

Result, mean across the 12 ordered pairs:

| Metric | Value | Interpretation |
|---|---:|---|
| $\rho_E$ (embedding residual) | 0.605 | terrible: 60% residual after best rotation |
| aligned R-distance | 592.7 | equal to random orthogonal baseline |
| aligned/identity R-distance | 1.000 | alignment doesn't help over no alignment |
| aligned mean angle (top-10) | 85.0° | equal to random orthogonal baseline |

All four PASS/FAIL criteria fail. The embedding-space alignment is
finding *some* $Q$, but that $Q$ is statistically indistinguishable
from a random orthogonal matrix in terms of how it transports the
R matrices.

### 7.2 The cause: undertrained rare-token contamination

The Mistral tokenizer has 32,768 BPE tokens, optimized for a 7B+
model trained on a much larger corpus. At our 150M scale and
1.57B-token training duration, only a fraction of these tokens
receive enough gradient signal to learn meaningful embeddings. The
remaining rare tokens stay near their random initialization throughout
training and contribute pure noise to the Procrustes residual.

The per-row L2 norm of each embedding distinguishes trained from
untrained tokens: initialization gives norm $\approx 0.60$
(corresponding to $0.02 \sqrt{896}$), and actively-trained
embeddings grow to norm $\approx 1.3$. Filtering to the top-$K$
tokens by per-row norm gives the following sweep, mean across pairs:

| $K$ | $\rho_E$ | Verdict |
|---|---:|---|
| 100 | 0.078 | excellent |
| 1000 | 0.095 | excellent |
| 5000 | 0.132 | good |
| 32768 (full) | 0.600 | terrible |

With $K = 1000$, embedding-space Procrustes recovers a clean alignment
(mean $\rho_E < 0.10$). The previous full-vocabulary failure was
**not a failure of the alignment procedure; it was a failure of the
anchor-set selection.** The diagnostic confirms this: the alignment
machinery works on the well-trained subset.

But — and this is the central finding of this section — fixing the
embedding alignment **does not fix the R-matrix transport**.

### 7.3 Embedding-space alignment doesn't transport R matrices

With the K=1000-token-anchor embedding alignment (mean $\rho_E = 0.10$,
PASS):

| Metric | Value |
|---|---:|
| aligned R-distance (sum over layers) | 592.7 |
| identity baseline | 592.7 |
| random-orthogonal baseline | 592.7 |
| aligned/identity | 1.000 |
| aligned/random | 1.000 |
| aligned mean angle (top-10) | 85.0° |

All three conditions (aligned, identity, random) give the same
R-distance to four significant figures. The embedding alignment
recovers a clean $Q$ that successfully aligns embeddings, but **that
$Q$ has no effect on the R matrices**.

The interpretation: the residual stream's coordinate frame at deeper
layers is no longer related to the embedding's coordinate frame by
the embedding-Procrustes $Q$. Each transformer block applies attention
and MLP transformations that rotate and stretch the residual stream
away from the embedding basis. By layer 12 of a 12-layer model, the
"coordinate frame" of the residual stream has been continuously
transformed by 12 sets of trained weights; the original embedding-space
alignment is no longer relevant.

This motivated the activation-space alternative.

### 7.4 Activation-space (per-layer) alignment also fails

The `align.py` module supports a stronger alignment: for each layer
$t$, find a separate $Q_t$ that aligns the per-layer activations
$X_t^{(A)}$ and $X_t^{(B)}$ on shared inputs. This is exactly the
generalization needed if the issue is that the residual stream
coordinate frame drifts with depth.

To enable this, we ran each seed's final-checkpoint model on the same
500 held-out chunks in the same order (eval-loader seed fixed at 0),
collected per-layer pilot activations, and cached them to
`run_dir/aligned_activations.npy`. For each ordered pair we then
ran the per-layer Procrustes alignment.

Result, mean across the 12 ordered pairs:

| Metric | Value | Verdict |
|---|---:|---|
| mean $\rho_t$ across layers | 0.620 | FAIL |
| aligned R-distance | 592.7 | FAIL — equal to random |
| aligned/random | 1.000 | FAIL |
| aligned/identity | 1.000 | FAIL — alignment doesn't help |
| aligned mean angle (top-10) | 85.0° | FAIL — equal to random |

Per-layer $\rho_t$ ranges from 0.39 (layer 1, post-first-block, best
alignment) to 0.72 (middle layers, worst alignment), with a U-shape
that's deepest at the network interior. Even at layer 1 where the
alignment is best, the R-matrix transport fails identically.

### 7.5 Top-K subspace diagnostic resolves the failure

Two hypotheses remained at this point:

- (1) Cross-seed R matrices genuinely share no basis structure, even
  at the top-1 direction.
- (2) The top few directions are shared but trailing directions
  (which dominate the Frobenius norm by count) are seed-specific
  noise.

To distinguish these, we ran the subspace-angle diagnostic. For
each ordered pair and each layer, we computed the principal angle
between the top-$K$ rows of (activation-aligned) $R_A$ and the top-$K$
rows of $R_B$, sweeping $K$ from 1 to $H = 896$. As a noise floor,
we also computed within-seed split-half angles: split a single seed's
9500 pilots into two random halves, recompute $R$ on each half, and
compare. Within-seed angles measure the analyzer's sample stability.

The mean angle across all layers, ordered pairs, and seeds:

| $K$ | cross-seed | within-seed | gap | random baseline |
|---|---:|---:|---:|---:|
| 1 | 88.9° | 7.0° | $+81.9$° | 87.0° |
| 2 | 87.8° | 9.5° | $+78.4$° | 85.6° |
| 5 | 86.6° | 13.2° | $+73.4$° | 86.5° |
| 10 | 85.0° | 17.0° | $+68.0$° | 84.8° |
| 50 | 78.4° | 24.1° | $+54.3$° | 78.4° |
| 100 | 73.1° | 26.6° | $+46.5$° | 72.7° |
| 500 | 39.4° | 20.6° | $+18.8$° | 39.3° |
| 896 (full) | 0.02° | 0.02° | 0.00° | 0.00° |

Two patterns are immediately clear:

- **Within-seed angles are small** (5–25°) across the K range. The
  analyzer is sample-stable: given 9500 pilots, the recovered $R$
  is reproducible to within ~7° at the top-1 direction when the data
  is split in half. The analyzer is doing its job.

- **Cross-seed angles are at the random baseline.** At every $K$,
  the cross-seed angle matches the random-orthonormal-subspaces
  baseline to within ~0.2°. Even at $K = 1$ — the top-1 principal
  direction of $R$, the dominant direction in the residual stream —
  cross-seed angles are 89° while within-seed split-half angles are
  7°.

The verdict is definitive. **Cross-seed R matrices share no recoverable
basis structure even at the top-1 direction.** This is hypothesis (1):
different seeds learn functionally equivalent models that organize
their hidden representations along seed-specific bases unrelated by
any orthogonal map.

### 7.6 Interpretation

The four seeds give identical:

- Eval loss (range 0.004 on 2.91)
- $\lambda$ (range 0.011 on 0.426)
- $\log\alpha$ (range 0.17 on $-3.28$)
- Mid-training $\log\alpha$ hump location and magnitude
- $R(t)$ trajectory geometry (successive-layer angles)
- Post-final-norm anomaly trajectory
- Boundary-layer effect $\Delta\log\alpha$ (range 0.021 on $-0.478$)
- Within-seed split-half analyzer noise floor

And give completely different:

- Principal directions of the residual stream at every layer (89° apart
  between any pair of seeds)
- $R(t)$ matrices themselves (Frobenius distance equal to random
  baseline)

The interpretation: **basis-invariant statistics describe shared
structure that all four seeds learn; basis-dependent quantities (the
$R$ matrices themselves) are model-internal coordinate representations
with no canonical cross-model meaning.** Different seeds reach
functionally equivalent points in model space along trajectories
through seed-specific reparameterizations of the hidden space.

This is consistent with everything we know about over-parameterized
deep networks: there are many bases in which the same function can be
expressed, and training picks one essentially at random from the
seed's initialization. What's surprising is just how *complete* the
basis indeterminacy is — even the dominant direction of variance in
the residual stream points in seed-specific places.

### 7.7 Consequence for Phase 2

The proposal's commitment to "vocabulary-anchored Procrustes alignment"
as the cross-variant comparison mechanism in Phase 2 is unworkable at
our scale, and probably at any reasonable scale. Phase 2 must use
basis-invariant statistics only. The Phase 1 dispersion measurements
in §4 and §5 give the within-variant noise floors that Phase 2 will
compare cross-variant differences against.

The next section gives the operational protocol.

---

## 8. Phase 2 comparison protocol

This is the locked-in measurement protocol for Phase 2 launch. It
specifies the statistics to be measured for each Phase 2 variant, the
within-variant noise floor for each statistic (carried over from §4–§6
above), and the universality threshold ($1.5\times\text{std}$) above
which a cross-variant difference is interpretable as a real
architectural signal rather than within-variant noise.

### 8.1 Comparison statistics

For each Phase 2 variant (architectural alternative), we run the same
Phase 1 pipeline: train from scratch with the same FineWeb-Edu recipe,
save 50 log-spaced checkpoints, run the analyzer, and extract:

| # | Statistic | Type | Where described | Phase 2 use |
|---|---|---|---|---|
| 1 | $\lambda$ (paper conv., all layers) | scalar at convergence | §4.1 | variance scaling rate |
| 2 | $\log\alpha$ (paper conv., all layers) | scalar at convergence | §4.2 | variance prefactor |
| 3 | $\lambda$ (paper conv., boundary-excl.) | scalar at convergence | §5.1 | inner-layer scaling rate |
| 4 | $\log\alpha$ (paper conv., boundary-excl.) | scalar at convergence | §5.1 | inner-layer prefactor |
| 5 | Effective rank profile | length-$L_\text{total}$ vector | §4.3 | residual stream dimensionality shape |
| 6 | Kurtosis profile | length-$L_\text{total}$ vector | §4.4, §6.4 | residual tail-heaviness shape |
| 7 | Isotropy profile | length-$L_\text{total}$ vector | §4.5 | residual anisotropy shape |
| 8 | Successive-layer angle profile | length-$(L_\text{total}-1)$ vector | §6.2 | $R(t)$ trajectory geometry |
| 9 | Post-final-norm anomaly magnitude | scalar at convergence | §5.1, §6.3 | architecture-dependent boundary effect |
| 10 | Mid-training $\log\alpha$ hump location (step / total) | scalar | §6.1 | training-dynamic universality |
| 11 | Mid-training $\log\alpha$ hump peak height | scalar | §6.1 | training-dynamic universality |
| 12 | Late-training kurtosis rise rate | scalar | §6.4 | training-dynamic universality |
| 13 | H1 ratio | scalar | §3.2 | convergence-of-flow universality |
| 14 | Eval loss at convergence | scalar | §2 | training-quality control |

Statistics 1-9 are at-convergence measurements; statistics 10-13 are
training-dynamic. Statistic 14 is a control — variants that don't
reach comparable eval loss should be excluded from the comparison
(they were under-trained, not architecturally different).

For each variant we compute a **within-variant baseline** by training
two seeds (or more if compute allows). The within-variant std of each
statistic should match the Phase 1 std reported below; if it doesn't,
the architecture has different stability properties and the cross-variant
comparison needs to use that variant's own std as the noise floor.

### 8.2 Within-variant noise floors

The noise floors below are from the Phase 1 4-seed measurements (§4).
For Phase 2, the comparison threshold is $1.5 \times \text{std}$;
cross-variant differences smaller than this are within-variant noise.

For scalar statistics:

| # | Statistic | Phase 1 mean | Phase 1 std | $1.5\times$std (threshold) |
|---|---|---:|---:|---:|
| 1 | $\lambda$ (all) | 0.4261 | 0.0048 | 0.0072 |
| 2 | $\log\alpha$ (all) | $-3.277$ | 0.070 | 0.105 |
| 3 | $\lambda$ (excl. bdry) | 0.5107 | 0.0048 | 0.0072 |
| 4 | $\log\alpha$ (excl. bdry) | $-3.756$ | 0.073 | 0.110 |
| 9 | Post-final-norm anomaly $\Delta\log\alpha$ | $-0.478$ | 0.010 | 0.015 |
| 10 | $\log\alpha$ hump peak step / total | 0.205 | 0.006 | 0.009 |
| 11 | $\log\alpha$ hump peak height | $-2.085$ | 0.080 | 0.120 |
| 13 | H1 ratio | 0.0436 | 0.0016 | 0.0025 |
| 14 | Eval loss | 2.908 | 0.0018 | 0.0028 |

For vector statistics (5-8), Phase 2 compares profile shapes
elementwise, with the threshold applied to each element of the vector.
The mean and std profiles are reported below (with values at boundary
layers and middle layers as summary statistics).

| # | Statistic | Boundary value mean | Boundary $1.5\times$std | Middle value mean | Middle $1.5\times$std |
|---|---|---:|---:|---:|---:|
| 5 | Effective rank | 159.8 | 5.1 | 492.9 | 23.2 |
| 6 | $\langle\|\kappa\|\rangle$ | 0.80 | 0.30 | 1.05 | 0.31 |
| 7 | Isotropy | 0.18 | 0.02 | 0.08 | 0.01 |
| 8 | Successive-layer angle (°) | 80 (0→1) | 5 | 30 (interior) | 3 |

### 8.3 Universality verdict criteria

For a Phase 2 variant to be classified as "universal" — i.e., matching
the Phase 1 baseline along a given statistic — its value of that
statistic must fall within $1.5 \times \text{std}_{\text{Phase 1}}$ of the
Phase 1 mean.

A variant is "fully universal" if it satisfies the criterion for **all
14 statistics**. A variant is "partially universal" if it matches on
the inner-layer statistics (1, 3, 5, 6, 7, 8) but differs on the
boundary or training-dynamic statistics. A variant fails universality
if it differs on $\lambda$ (statistic 1 or 3) by more than 0.01.

The boundary-related statistics (2, 4, 9) and the kurtosis statistic
(6) are the most likely to be architecture-dependent and should be
interpreted as informative rather than verdict-determining if they
differ.

### 8.4 What Phase 2 does not measure

The following are explicitly out of scope for Phase 2:

- **R-matrix-level comparison.** Per the alignment analysis in §7, no
  meaningful cross-architecture R-matrix comparison exists at this
  scale. Statistic 8 (successive-layer angle profile) is the only
  R-matrix-derived quantity that is basis-invariant and therefore
  cross-architecture-comparable.
- **Cross-variant principal direction analysis.** Same reason as above.
  Each variant has its own principal directions and they are not
  comparable across variants.
- **Distributional alignment** (e.g., optimal-transport metrics
  between hidden distributions). The proposal mentions this as a
  potential alternative if Procrustes fails; we don't pursue it in
  Phase 2 because the basis-invariant statistics already give a clean
  comparison protocol without it.

### 8.5 Phase 2 launch checklist

Before launching Phase 2:

1. ☐ Decide the Phase 2 variant set. Proposal commits to 6 variants
   (Llama-style, Gemma-style, Qwen-style, DeepSeek-style, two others).
   Consider reducing to 4 variants if compute is tight.
2. ☐ Decide the seed count per variant. Phase 1 used 4 seeds. Phase 2
   could use 2-3 seeds per variant given budget; the noise-floor std
   on the chosen statistics is small enough that 2 seeds per variant
   is informative.
3. ☐ Confirm the training recipe is held *identical* across variants.
   Hyperparameters that differ across variants — peak LR, optimizer,
   warmup length — are confounds and would need to be ablated separately.
4. ☐ Confirm the eval set is held identical across variants (same
   500 held-out chunks, same pilot positions).
5. ☐ Decide whether to extend to a wider context length than 1024 for
   any variant. Phase 1 used context 1024; extending introduces a
   confound.
6. ☐ Total compute budget: 6 variants × 3 seeds = 18 runs of ~12
   hours each = ~9 days of single-GPU time. Plan for 14 days
   wall-clock with checkpointing and analysis overhead.

The Phase 1 deliverables that Phase 2 must inherit are: the analyzer
(`analyze.py`, with both convention support), the validate script,
the cross-seed comparison script (`compare_seeds.py`, generalized to
cross-variant), the boundary-layer-check script
(`boundary_layer_check.py`), and the alignment-check scripts which
should be repurposed to confirm — not test — that R-matrix
comparisons remain meaningless in the cross-variant case.

---

## 9. Open questions and limitations

### 9.1 Open questions

- **Why does the seed-1 kurtosis trajectory diverge from the others
  past step ~13,000?** The other measured statistics agree across all
  four seeds at convergence; only the kurtosis (and slightly the
  isotropy) shows the seed-1 anomaly. We don't have a mechanistic
  explanation. Phase 2 may shed light on whether this is
  architecture-independent (suggesting it's a generic late-training
  optimization-trajectory branch) or architecture-dependent.

- **What is the relationship between the $\log\alpha$ hump and the
  post-final-norm anomaly?** Both emerge in the same training window
  (steps 2000-5000). Are they the same phenomenon viewed through
  different statistics? §6 documents the co-timing but doesn't pursue
  the question.

- **Does $\lambda \times L \approx 5.5$ hold across architectures
  when measured under boundary exclusion?** The paper's value comes
  from all-layer fits. Our all-layer $\lambda L \approx 5.1$
  reproduces the paper's value, but our boundary-excluded $\lambda L
  \approx 6.1$ doesn't. Phase 2 tests this directly.

- **Is the cross-seed alignment failure a 150M-specific phenomenon, or
  is it scale-invariant?** Our cross-seed R matrices share no
  recoverable basis. This is consistent with standard
  over-parameterization arguments but those arguments don't depend on
  scale; we would expect the failure to persist at larger scales.
  Phase 2 doesn't test this directly (it varies architecture, not
  scale), but a follow-on study at 350M and 1B might.

- **The 30-log-unit gap on the paper's released `trajectories.npy` for
  Llama-2-7B.** We cannot match the paper's published $\log\alpha$
  value when running our analyzer on their released trajectories. This
  is documented in `PAPER_CODE_REVIEW.md` §12 and is most likely a
  per-trajectory normalization difference in their preprocessing.
  Resolution requires correspondence with the paper authors.

### 9.2 Limitations

- **Single architectural family.** Phase 1 measures only one
  architecture (Llama-style). The within-variant dispersion measurements
  reported here are within-Llama-style dispersion. Phase 2 will measure
  whether other architecture families have the same within-variant
  dispersion (i.e., the same noise floor) or different.

- **Single training recipe.** Phase 1 measures one recipe
  (FineWeb-Edu, AdamW-cosine, 1.57B tokens). Whether the same dispersion
  bounds hold under other recipes (different corpora, different
  optimizers, different LR schedules) is not tested. Phase 2 holds
  the recipe fixed; recipe-variation is a Phase 3 question if
  applicable.

- **Single scale.** Phase 1 measures at $H = 896$, $L = 12$, 146M
  parameters. The paper's models range from 350M to 12B. Whether our
  pilot results scale appropriately is a Phase 3 question.

- **Short pilot positions.** We sample at 19 positions within each
  1024-token chunk. This is well below the paper's reported scale
  (10⁵–10⁶ pilots for the larger models). At our $H = 896$, 9500
  pilots is adequate for stable SVD, but we don't know whether
  measurements that depend on pilot-position diversity (e.g., kurtosis
  sensitivity to specific token classes) would differ at larger
  pilot scales.

- **No semantic interpretation.** This pilot quantifies the
  *geometry* of the residual stream but does not interpret what the
  recovered principal directions *mean*. The "lines of thought"
  framing in the paper hints at semantic interpretability of the
  flow; our analysis doesn't address that and the alignment failure
  in §7 suggests that any semantic interpretation would have to be
  per-model rather than universal.

---

## 10. Summary

### Headline Phase 1 findings

1. **H1 (convergence) passes on all four seeds**, with margin to spare
   (ratio 0.043 vs threshold 0.10; range across seeds 0.004).

2. **The basis-invariant statistics are reproducible** across seeds at
   1–4% relative spread for $\lambda$, $\log\alpha$, effective rank,
   isotropy, and convergence dynamics. Kurtosis is the only
   substantially-disperse statistic (19% relative spread, dominated by
   one seed).

3. **The boundary-layer effect is real and structural**, with
   $\Delta\log\alpha = -0.478 \pm 0.010$ (paper convention) across
   the four seeds — about 5% relative variation. The effect emerges
   during training (not at initialization), reaches near-final
   magnitude by step ~5000, and is the same direction (post-final-norm
   sits below the inner-layer fit) in all four seeds.

4. **Mid-training structural features replicate across seeds:** the
   $\log\alpha$ hump at step ~5000, the $R(t)$ trajectory geometry,
   the post-final-norm anomaly emergence between steps 400-5000, and
   the late-training kurtosis rise. All five features are absent at
   initialization and emerge during training.

5. **Cross-seed R-matrix alignment fails fundamentally.** Embedding-space
   Procrustes (full vocab) fails; top-K-filtered embedding-Procrustes
   succeeds at the embedding level but fails to transport R; per-layer
   activation-space Procrustes fails identically. The subspace-resolution
   diagnostic confirms that **cross-seed R matrices share no recoverable
   basis structure even at the top-1 principal direction** — different
   seeds learn functionally-equivalent models along seed-specific
   bases that don't translate via any orthogonal map.

### What this means for Phase 2

Phase 2 must compare variants only on basis-invariant statistics. The
Phase 1 noise floors (§8.2) give within-variant dispersion bounds for
all 14 Phase 2 comparison statistics. A variant is "universal" with
respect to a statistic if its value falls within $1.5 \times
\text{std}_{\text{Phase 1}}$ of the Phase 1 mean for that statistic.

The proposal's commitment to vocabulary-anchored Procrustes alignment
is dropped from Phase 2 with a documented Phase 1 justification (§7).
The framework's basis-invariant statistics — what the paper itself
uses for cross-model comparison — are the appropriate level of
abstraction for Phase 2, and they suffice.

### What Phase 1 cost

- 4 seed training runs × ~12 hours each = ~48 GPU-hours
- Analysis: ~1.5 GPU-hours per seed × 4 = ~6 GPU-hours
- Alignment analysis: ~1 GPU-hour (one-time per seed, for activation
  collection)
- Total: ~55 GPU-hours on one RTX 5090, plus development time.

### What Phase 2 will cost

- 6 variants × 3 seeds × ~12 hours each = ~216 GPU-hours of training
- Analysis: ~27 GPU-hours
- Total: ~243 GPU-hours, about 10 days of dedicated single-GPU time.

If compute is tight, reducing to 4 variants × 3 seeds (~162
GPU-hours, ~7 days) or 6 variants × 2 seeds (~144 GPU-hours, ~6 days)
keeps the dispersion estimates informative at the cost of confidence
margin on within-variant noise floors per variant.

---

*End of Phase 1 report.*
