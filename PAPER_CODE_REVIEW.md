# Notes on the Sarfati et al. "Lines of Thought" code and paper

This document captures everything learned by reading both the source code at
[github.com/rapsar/lines-of-thought](https://github.com/rapsar/lines-of-thought)
and the published paper (ICLR 2025, arXiv:2410.01545). The review was done
while the Phase 1 seeded training campaign was in flight.

The first version of this document (before the paper was read carefully)
contained several errors about what the paper says and what the
methodological discrepancies between our pipeline and theirs are. The
v2 revision corrected those errors. This v3 revision incorporates
direct measurements obtained by re-running the analyzer with both
statistic conventions (mean-of-log and log-of-mean) computed
side-by-side on seed-0's 50 checkpoints. The measurements disconfirm
v2's central hypothesis that the mean-of-log vs log-of-mean convention
accounts for most of the log α gap to the paper. Corrections to v2 are
noted explicitly in §12.

The repository is small: MATLAB scripts for the analysis, one Python
script for activation collection, and a web visualization. The paper
itself provides considerable additional context that isn't apparent from
the code alone — in particular, the exact normalization conventions
used for the figures.

---

## 1. Repository structure

```
lines-of-thought/
├── README.md                   # one-line pointer to the arXiv paper
├── CITATION.cff
├── code/
│   ├── matlab/
│   │   ├── dx_calculate.m              # residual-computation core
│   │   └── LoT_langevin_integration.m  # Langevin model simulator
│   └── python/
│       ├── pull-traj.py                # collect activations from HF model
│       └── npy2mat.py                  # NumPy → MATLAB format converter
├── fig/
│   ├── fig01.m, fig02.m, fig03.m       # paper figure scripts
│   ├── figA_llama2_7B.m                # appendix figure script
│   └── startup.m
├── txt/
│   └── walden-thoreau-pg-clean.txt     # the corpus (one book)
└── web/
    └── assets/
        ├── llama-2-7B/{trajectories,singular_vectors}.npy
        ├── llama-3.2-1B/{trajectories,singular_vectors}.npy
        ├── llama-3.2-3B/{trajectories,singular_vectors}.npy
        └── mistral-7B/{trajectories,singular_vectors}.npy
```

The `.npy` files in `web/assets/` are **released artifacts**: the exact
trajectory data the paper analyzed, in NumPy format suitable for direct
reanalysis with our pipeline.

---

## 2. The paper's primary model and headline numbers

A point of context that isn't obvious until reading §2 of the paper: the
*primary* model the paper analyzes is **GPT-2 medium** (355M parameters,
24 layers, D=1024, vocab 50257). The Llama 2/3.2 and Mistral results are
*extensions* in §4.3, not the main results.

The paper's primary headline numbers are for GPT-2 medium (§3.3):

> "Linear fitting of the logarithm of the variance yields α ≃ 0.64 and λ ≃ 0.18."

The Llama 2 7B extension reports (§4.3):

> "the parameters α and λ ... differ from those of GPT-2 (here, α ≃ −5.4, λ ≃ 0.27)."

**The "α ≃ −5.4" almost certainly means log α, not α itself.** α is a
variance prefactor in Eq. (2), `var = α exp(λ(t+τ))`, so α must be
non-negative. A negative value can only be log α. The paper switches
notation between the two contexts without flagging it. If we interpret
both consistently:

- GPT-2 medium: α ≈ 0.64, λ ≈ 0.18, **log α ≈ −0.45**
- Llama 2 7B: **log α ≈ −5.4**, λ ≈ 0.27

The paper's primary corpus is 3000-14000 pseudo-sentences from Walden
(§2), not the small sets (N=100-500) shipped in `web/assets/`. Those
shipped files are subsamples for the web visualization; the actual
paper analyses use much larger ensembles for GPT-2 medium.

---

## 3. Data collection methodology

### 3.1 The corpus

Henry David Thoreau's *Walden* (Project Gutenberg edition, cleaned).
A single book. Not multiple corpora, not held-out evaluation data, not
filtered by domain. The paper notes that Markov ran similar
text-statistics analyses on *Eugene Onegin* in 1913 — there's a
deliberate tradition of using single literary works to probe text
statistics.

Implication for our work: comparisons of our FineWeb-Edu values to the
paper's Walden values mix corpus differences with everything else.

### 3.2 Activation collection (`pull-traj.py`)

1. Load a Hugging Face pretrained model.
2. Tokenize the full Walden text.
3. Split the token stream into non-overlapping chunks of N tokens each
   (default 50).
4. For each chunk: forward pass, save the hidden state at the last token
   position at every layer.
5. Save as `(num_layers, hidden_dim, num_chunks)` `.npy`.

**Only the last-token embedding per chunk is saved.** The paper calls
this the "pilot token" (§2). Our pipeline calls a different concept
"pilot activations" — we collect 19 positions per chunk (positions 50,
100, ..., 950), giving 19 trajectories per chunk instead of 1.

The terminology overlap is unfortunate but the substance is different:
their "pilot" is "the last token whose embedding becomes the logits";
ours is "a sampled position whose hidden state we analyze." Their
single-position choice is principled — only the last position is used
for the actual loss — but it produces a particular distribution of
activation states (always end-of-context). Our multi-position sampling
broadens the distribution.

### 3.3 Trajectory sample sizes

Per the paper §2: GPT-2 medium analyses use Ns ≃ 3000-14000 pseudo-
sentences. The shipped data for the larger models is much smaller:

| Model | Layers (L) | Hidden (H) | Trajectories (N) |
|---|---|---|---|
| Llama-2-7B | 32 | 4096 | 100 |
| Llama-3.2-1B | 16 | 2048 | 500 |
| Llama-3.2-3B | 28 | 3072 | 250 |
| Mistral-7B | 32 | 4096 | (similar small N) |

For these models N < H, so the per-layer SVD is rank-bounded by N. The
paper's published statistics for these models are computed from these
small samples, while their GPT-2-medium statistics use N >> H.

Our seed-0 run collects N=9,500 per checkpoint, comparable to the
GPT-2-medium regime.

---

## 4. Analysis methodology

### 4.1 The core residual computation (`dx_calculate.m`)

```matlab
xi{t1,t2} = M2 * LM * (M1' * x1);   % linear extrapolation
dx{t1,t2} = x2 - xi{t1,t2};         % residual
```

This is the paper's Eq. (1):

$$\tilde{x}(t_2) = R(t_2)\,\Lambda(t_1, t_2)\,R(t_1)^\top x(t_1)$$

The formula has SVD sign-and-permutation ambiguity. The paper handles
this by per-layer realignment (§4.3 below), not by an alternative
formulation. Our pipeline uses OLS projection instead — mathematically
equivalent to the paper's formula under matched sign conventions,
sign-robust without them.

### 4.2 Excluded layers

In `dx_calculate.m`:

```matlab
warning('Make sure traj contains only layers output, not embeddings or layernorm outputs.')
```

And `figA_llama2_7B.m`:

```matlab
llama27B.traj = llama27B.traj(:,2:end,:);   % remove first embeddings
```

The paper notes the same exclusion in §2:

> "(This final normalization is not included in our trajectories.)"

And §4.3 explicitly identifies the last-layer anomaly in their data:

> "It seems as though the last layer is misaligned with the rest of the
> trajectories, as linear extrapolation produces an error that is much
> larger than expected."

> "The last layer anomaly is also apparent for Llama 3.2 1B..., both in
> the mean and variance of δx(t, 16)... The same pattern is observed for
> Llama 3.2 3B."

> "It is noteworthy that these three recent models feature the same
> anomaly at the last layer. The reason is not immediately evident, and
> perhaps worth investigating further."

The paper conjectures the anomaly is "an effect of re-alignment or
fine-tuning, as the first and last layers are the most exposed to
perturbations."

**This is identical to what we observe in our seed-0 plot 5:** the
post-final-norm layer (state 13 in our model) sits well below the
variance-scaling fit line. Our model isn't fine-tuned, so the conjecture
about fine-tuning doesn't apply, but the anomaly is the same shape. A
more likely structural explanation: the final RMSNorm rescales each
token's vector to unit norm (up to per-dim learned gains), which
mechanically reduces the residual stream's variance at that point in a
way the linear-flow prediction can't capture from upstream singular
values alone.

For headline figures the paper goes further:

- `fig01` uses only layers 3-26 of 32 (Mistral)
- `fig03` uses only layers 12-21 of 32

**Their analyses systematically exclude both endpoints of the residual
stream, with headline figures restricted to inner-depth slices.**

### 4.3 SVD sign/permutation realignment

In `fig02.m`:

```matlab
psi{1} = svc{1};
for t = 2:24
    [order, signs] = compute_order_and_signs(psi{t-1}, svc{t});
    psi{t} = svc{t}(:, order) .* signs;
end
```

For each layer t, the singular vectors are re-permuted and sign-flipped
to maximize inner products with the previous layer's basis. The Eq. (1)
formula then makes sense because axis-i at $t_1$ corresponds to axis-i
at $t_2$ by construction.

Our pipeline uses OLS projection. Equivalent for residual magnitudes;
preserves R(t) identity across layers less cleanly.

### 4.4 Outlier removal

`figA_llama2_7B.m` hard-codes seven outlier trajectory indices for
Llama-2 7B (`bad = [66, 223, 277, 330, 435, 538, 669]`), identified via
a 10×-MAD filter on per-layer vector norms. Our pipeline doesn't do
outlier removal.

### 4.5 Statistics computed and the normalization convention

This is the section that needed the most correction from the first
version of this document.

The paper's Fig. 4 shows three panels: `<|µ/σ|>`, `<log σ²>`, `<|κ|>`,
each as a 2D heatmap over `(t, t+τ)`. The schematic in Fig. A4 makes the
computation explicit:

> "The δx along each coordinate i form a distribution, from which one
> can extract the corresponding µᵢ, σᵢ, κᵢ (mean, variance, kurtosis).
> These 1D moments are then averaged along all coordinates i (⟨µᵢ⟩ᵢ,
> etc.), forming the value displayed in the square."

So per (t, t+τ) pair:

1. For each coordinate i ∈ {1, ..., D}, compute the per-coordinate
   statistics across the N trajectories: µᵢ, σᵢ², κᵢ.
2. The displayed `<log σ²>` is **`mean_i log(σᵢ²)`** — log applied per
   coordinate, then averaged across coordinates.
3. Similarly `<|µ/σ|>` is `mean_i |µᵢ/σᵢ|`, and `<|κ|>` is
   `mean_i |κᵢ|`.

**This is mean-of-log, not log-of-mean.** The two are different unless
all per-coordinate variances are identical. The gap between them is
exactly the *Jensen-inequality residual*, bounded below by σ²/2 where
σ is the standard deviation of log σᵢ² across coordinates — our
"isotropy" statistic.

Our analyzer historically computed `log(mean over coordinates of σ²)` —
log of mean. After we modified the analyzer to compute both conventions
simultaneously (§11, item 2, completed), we measured the actual gap
across seed-0's 50 checkpoints:

- **At step 100** (untrained): log α(ours) = −3.241, log α(paper) = −3.742.
  Gap = 0.50, isotropy = 0.165, predicted-Lognormal-gap = 0.014.
  Actual gap is 35× the Lognormal prediction, indicating the per-coord
  variance distribution is much heavier-tailed than Lognormal early in
  training.
- **At step 24000** (converged): log α(ours) = −3.266, log α(paper) = −3.298.
  Gap = 0.032, isotropy = 0.153, predicted-Lognormal-gap = 0.012.
  Actual gap is 2.7× the Lognormal prediction.
- **Across training**, the gap shrinks roughly monotonically from
  ~0.5 to ~0.03 as per-coordinate variances become more homogeneous.

**For our 150M data, the convention gap is small** — at most 0.5 log
units even at the highest-dispersion checkpoint, and ~0.03 at
convergence. This is *not* the right order of magnitude to explain the
30-log-unit discrepancy we previously observed when running our
analyzer on the paper's released `.npy` files (§6.2). The convention
gap apparently can be 30 log units on the paper's data but is only
0.03 on ours; that contrast itself is informative and is discussed in
§6.2 below.

Conversion between the two: if the per-coordinate variances σᵢ² have
some distribution P(σ²), then `log E[σ²] ≥ E[log σ²]` (Jensen's
inequality), with equality when the distribution is degenerate.

### 4.6 Centering

`dx_calculate.m` does not center the activations before computing
residuals. The paper's Fig. 4(a) shows `<|µ/σ|>` values mostly under 0.1
across (t, t+τ), validating that the linear-flow prediction is
empirically centered without explicit mean subtraction.

Our analyzer centers explicitly (`mu = x.mean(axis=0); xc = x - mu`).
This should be a no-op on data where the linear-flow prediction is
already approximately centered — which is most of the data they show —
but it's a stricter operation.

---

## 5. The Langevin model and the meaning of λ

`LoT_langevin_integration.m` confirms the paper's Eq. (3) parameterization:

```matlab
B = sqrt(alpha * lambda * exp(lambda * t)) * eye(1024)
```

The noise covariance per unit time is $\alpha \lambda e^{\lambda t}$.
Integrated over a layer transition, accumulated variance scales as
$\alpha e^{\lambda(t+\tau)}$. **Both their λ and ours are the asymptotic
slope of `log(variance)` vs `t+τ`.** They are directly comparable as
slopes regardless of how variance itself is normalized.

Their α and ours differ for several reasons (see §8 for the full
breakdown): corpus, position-sampling, training-duration, layer-scope,
and a small (≤ 0.5 log unit) contribution from the mean-of-log vs
log-of-mean convention discussed in §4.5.

---

## 6. Numerical cross-scale comparison

### 6.1 What the paper reports

| Model | L | H | λ (paper) | log α (paper, our reading) |
|---|---|---|---|---|
| GPT-2 medium | 24 | 1024 | 0.18 | ~−0.45 |
| Llama 2 7B | 32 | 4096 | 0.27 | ~−5.4 |
| Llama 3.2 1B | 16 | 2048 | (not given numerically) | (not given numerically) |
| Llama 3.2 3B | 28 | 3072 | (not given numerically) | (not given numerically) |
| Mistral 7B | 32 | 4096 | (not given numerically) | (not given numerically) |

The paper gives λ and α only for GPT-2 medium and Llama 2 7B
quantitatively. For the other models it shows the variance-scaling
heatmaps (Figs. A8-A10) without extracting numerical fit parameters.

### 6.2 What our analyzer gives on the paper's released data

Running our analyzer on the paper's released trajectory `.npy` files,
log-of-mean convention (our original convention):

| Model | L | H | N | λ (ours) | log α (ours) | mean kurt (ours) |
|---|---|---|---|---|---|---|
| Our 150M, seed 0 final | 12 | 896 | 9500 | 0.44 | −3.27 | 0.87 |
| Llama-3.2-1B (their data) | 16 | 2048 | 500 | 0.35 | −36.1 | 1.09 |
| Llama-3.2-3B (their data) | 28 | 3072 | 250 | 0.19 | −35.2 | 1.59 |
| Llama-2-7B (their data) | 32 | 4096 | 100 | 0.24 | −35.9 | 1.39 |

The log α values for the published models (−35 to −36 with our analyzer)
are not the paper's values. The paper's published log α for Llama 2 7B
is ~−5.4. The ~30-log-unit gap was hypothesized in v2 of this document
to be the mean-of-log vs log-of-mean convention difference. **That
hypothesis is now disconfirmed for our own data** and is the central
update in v3.

After modifying our analyzer to compute the paper-convention statistic
alongside our own (see §11 item 2, completed), we measured both on our
seed-0 final checkpoint:

| Model | log α (ours = log-of-mean) | log α (paper = mean-of-log) | λ (ours) | λ (paper) | `<κ>` | `<|κ|>` |
|---|---|---|---|---|---|---|
| Our 150M, seed 0 final (step 24000) | −3.266 | −3.298 | 0.4418 | 0.4256 | +0.871 | 0.871 |

**The convention gap on our converged data is 0.03 in log α and 0.016
in λ.** Even at the most dispersed checkpoint (step 100), the gap is
0.50 in log α and 0.16 in λ. Neither is anywhere near the 30-log-unit
gap observed when running our analyzer on the paper's released `.npy`
files. **The convention difference cannot explain a 30-log-unit gap on
our data**, and therefore cannot explain a 30-log-unit gap *on any
data* under the same fitting procedure — the gap is structural to the
data distribution, not to the convention.

The most likely explanations for the 30-log-unit gap on the paper's
released `.npy` files (still to be investigated):

- The released `.npy` files may be normalized differently from the data
  the paper actually fit (e.g., divided by H, or by some other
  per-trajectory scaling). The paper's fit values are reported on data
  that may have been preprocessed beyond what `pull-traj.py` produces.
- The released N=100-500 trajectory counts are far below the
  N=3000-14000 the paper says it used for GPT-2 medium fits. The fits
  on the released subsamples may not reproduce the paper's published
  numbers regardless of convention.
- A unit difference (e.g., the paper fits variances of unit-normalized
  trajectories, where unit-normalization scales σ² by ~1/H ≈ 1/4096 for
  Llama-2-7B, giving log α an offset of −log H ≈ −8.3).

None of these is the mean-of-log vs log-of-mean convention.

**Net effect on our pilot's interpretation:** our seed-0 log α and λ
are not directly comparable to the paper's published values, but
the reason is now clearer:

1. The convention difference contributes ≤ 0.5 log units across our
   training trajectory, and ~0.03 at our converged checkpoint. This
   is the smallest contributor.
2. **Corpus, position-sampling, training-duration, and layer-scope
   differences are the larger contributors.** Layer scope in particular:
   our log α fit *including* the post-final-norm and embedding layers
   is pulled down by the layer-13 anomaly (~3 log units below the fit
   line in plot 5), so excluding boundary layers — the paper's
   convention — would shift our log α upward by approximately the
   contribution of those outliers. The `--exclude_boundary_layers`
   refinement (§11, item 1) would close more of the gap than the
   convention fix did.

### 6.3 Robust cross-scale findings (basis-invariant)

Slopes are normalization-invariant, so the λ comparison is the cleanest
finding:

| Model | L | λ | λ × L |
|---|---|---|---|
| Our 150M | 12 | 0.44 | 5.3 |
| Llama 3.2 1B (our analyzer) | 16 | 0.35 | 5.6 |
| GPT-2 medium (paper) | 24 | 0.18 | 4.3 |
| Llama 3.2 3B (our analyzer) | 28 | 0.19 | 5.5 |
| Llama 2 7B (paper) | 32 | 0.27 | 8.6 |
| Llama 2 7B (our analyzer) | 32 | 0.24 | 7.7 |

**Within the Llama 3.2 family + our pilot, λ × L is approximately
conserved at ~5.5** (5.3, 5.6, 5.5). GPT-2 medium and Llama 2 7B sit
outside this range. Worth investigating whether the conservation is a
property of the Llama 3.2 family specifically, of recent post-2024
models, or of some other architectural commonality.

**λ decreases with scale within the Llama 3.2 family**: 1B → 0.35,
3B → 0.19. Our 150M continues the trend at 0.44.

### 6.4 Kurtosis levels for trained models

This is where the previous version of this document was wrong. The
paper's Figs. A6-A10 show `<|κ|>` heatmaps with the following ranges
(read from the colorbars):

- **Untrained GPT-2** (Fig. A6, the null baseline): `<|κ|>` reaches
  1.0-1.5 — strong non-Gaussianity, "indicating strong non-gaussianity"
  per the figure caption.
- **Trained Llama 2 7B** (Fig. A7): `<|κ|>` mostly in 0-0.3 range.
- **Trained Mistral 7B** (Fig. A8): `<|κ|>` mostly in 0-0.3 range.
- **Trained Llama 3.2 1B** (Fig. A9): `<|κ|>` mostly in 0-0.3 range.
- **Trained Llama 3.2 3B** (Fig. A10): `<|κ|>` mostly in 0-0.3 range.

**For trained models, the paper's published mean excess kurtosis is
roughly 0.1-0.3.** Our seed-0 final kurtosis of 0.87 is meaningfully
higher than this range. (The previous version of this document
incorrectly claimed the paper's trained models showed kurtosis 1.0-1.6
— that was the untrained-model baseline.)

Possible explanations for the higher kurtosis in our pilot:

- **Scale.** Our 150M is smaller than any model the paper studied.
- **Methodology.** We had hypothesized in v2 that the paper computes
  `mean_d |κᵢ|` (mean of absolute values of per-coordinate kurtosis)
  while we compute `mean_d κᵢ` (signed mean), and that the convention
  could account for some of the gap. After measuring both conventions
  on seed-0 (see §11 item 2, completed), **this hypothesis is
  disconfirmed for our data**: at the converged checkpoint, `<κ>` =
  +0.871 and `<|κ|>` = 0.871 agree to within 0.001. The signed mean
  and absolute mean coincide because the per-coordinate kurtosis
  distribution is essentially one-sided (positive) on our residuals —
  there is no cancellation that the absolute-value convention would
  unmask. **This eliminates kurtosis-convention as an explanation for
  our high `<|κ|>`.**
- **Position sampling.** Our 19 positions per chunk vs their 1
  last-token position. The kurtosis of residuals at different sequence
  positions may differ.
- **Training duration.** Our model is trained for ~1.57B tokens; the
  paper's models are fully trained foundation models.
- **Boundary layers.** We include the post-final-norm layer; the paper
  excludes it. This layer is exactly where high kurtosis would show up
  if the final RMSNorm makes residuals heavy-tailed.

Given the convention check is now negative, the safer claim is: **our
seed-0 `<|κ|>` of 0.871 is genuinely above the paper's trained-model
range of 0.1-0.3, with the gap attributable to scale, layer-scope, and
training-duration differences rather than to a statistical convention
artifact.** Furthermore, `<|κ|>` rises monotonically from a minimum of
~0.35 around step 2000 to 0.87 at step 24000 — the framework's
Gaussianity assumption is increasingly violated through the second
half of our training run, and this trajectory is itself robust to
convention choice.

### 6.5 Effective rank comparison

The paper's effective-rank claim (§3.2) is K₀ ≈ 256 for GPT-2 medium,
defined by KL-divergence collapse rather than by singular-value entropy.
This is **not the same statistic** as our effective rank. The paper:

1. Truncate trajectories to keep only the first K principal components.
2. Compute the resulting next-token distribution p^V_K.
3. K₀ is the smallest K where DKL(p^V_K || p^V) is 10% of baseline.

Our effective rank is `exp(entropy of normalized squared singular value
distribution)`. **A direct numerical comparison isn't valid.** We can
compare the depth profile shape (peak in mid-network, drop at ends),
which we did, but the absolute numbers measure different things.

---

## 7. Pipeline differences summary

| Aspect | Paper | Our pipeline |
|---|---|---|
| Corpus | Walden (one book) | FineWeb-Edu (broad web text) |
| Models | Pretrained HF (GPT-2 medium primarily, larger as extensions) | Trained from scratch 150M (with seed control) |
| Positions per chunk | 1 (last token, "pilot") | 19 (positions 50, 100, ..., 950) |
| Total trajectories | 3,000-14,000 for GPT-2; 100-500 for released Llama/Mistral data | 9,500 per checkpoint |
| Layers analyzed | Excludes embedding and post-final-norm; headline figures restrict to inner slices | Includes all 14 layer states |
| Sign ambiguity fix | Per-layer realignment | OLS projection |
| Centering | None (empirically validated) | Explicit mean subtraction |
| Outlier removal | Manual + 10×-MAD | None |
| Variance/kurtosis statistic | Mean across coordinates of `log σᵢ²`, `|κᵢ|` (Jensen-style) | Both conventions computed and saved as of v3 (§11, item 2 complete); paper-convention values available alongside ours-convention values per checkpoint |
| Effective rank definition | KL-divergence collapse | exp(singular-value entropy) |

---

## 8. The "α normalization mystery" — current understanding

This section's framing has now changed twice. v1 framed log α as
"an unresolved mystery." v2 framed it as "most likely explained by
mean-of-log vs log-of-mean convention." v3 reports that we have
measured both conventions side-by-side on our data and concluded that
the convention explains essentially none of the gap.

**Measured contribution of the convention difference (seed-0):**

| Checkpoint | log α (ours) | log α (paper) | Gap |
|---|---|---|---|
| Step 100 (untrained) | −3.241 | −3.742 | 0.50 |
| Step 2049 (mid-train) | −2.293 | −2.345 | 0.05 |
| Step 24000 (converged) | −3.266 | −3.298 | 0.03 |

The gap to the paper's published values for trained models (GPT-2
medium log α ≈ −0.45, Llama-2-7B log α ≈ −5.4) is in the 2-3 log unit
range; the convention difference closes ≤ 0.03 of that.

**Status of the various sub-hypotheses:**

| Hypothesis (v2) | Predicted contribution | Measured contribution (v3) | Verdict |
|---|---|---|---|
| Mean-of-log vs log-of-mean convention | "matches the magnitude" of ~30 log units | 0.03 at convergence | **Disconfirmed for our data** |
| Pooled vs mean-of-per-coord kurtosis | Could explain `<|κ|>` of 0.87 | `<κ>` = `<|κ|>` to within 0.001 | **Disconfirmed for our data** |
| Boundary-layer inclusion (us) vs exclusion (paper) | Not estimated in v2 | Not yet measured directly; layer-13 sits ~3 log units below fit line in plot 5 | **Likely substantial; refinement #1 in §11 is the next test** |
| Corpus / position-sampling / training-duration | Not separately estimable from our data | Not separately estimable from our data | **Probably contribute; cannot isolate at 150M alone** |

**The 30-log-unit gap on the paper's released data is a separate
issue.** Our log α on `web/assets/llama-2-7B/trajectories.npy` was
−35.9; the paper's published Llama-2-7B log α is ~−5.4. Now that we
know the convention difference can only explain ≤ 0.5 log units on
our own data, it cannot explain 30 units on theirs either under the
same fitting procedure. The remaining ~30-unit gap is more likely a
data-normalization artifact in the released `.npy` files (see §6.2
for hypotheses) than a statistical convention.

**Verifiable next steps:**

1. **Boundary-layer exclusion (§11, item 1).** Re-fit log α and λ on
   the saved `.npz` files with layers 0 and L excluded. If our log α
   shifts upward by ~1-2 units, this is the dominant non-convention
   contributor and explains a substantial fraction of the remaining
   gap to the paper.
2. **Email Sarfati** to ask directly about the normalization applied
   to the released `.npy` files vs the data underlying the published
   fits. The 30-unit gap on their data is now the cleanest open
   question; if it's a per-trajectory normalization (e.g., dividing
   by `H`), the answer is one paragraph.

---

## 9. The last-layer anomaly is a known paper finding

§4.3 of the paper explicitly identifies the last-layer anomaly across
Mistral, Llama 3.2 1B, and Llama 3.2 3B. The paper conjectures it's a
fine-tuning or alignment artifact, but our seed-0 pilot (trained from
scratch, no fine-tuning, no alignment) shows the same anomaly. So the
conjecture about fine-tuning is incomplete — the anomaly exists in
pre-trained-from-scratch models too.

A more likely structural explanation: the final RMSNorm (or LayerNorm
for the paper's models) rescales each token vector at the last layer,
which mechanically distorts the residual stream's variance scaling at
that specific layer. The Eq. (1) prediction extrapolates linearly from
upstream singular values and can't capture this rescaling.

This was the basis of the proposal's §10.2 disclosure that we treat the
post-final-norm layer separately. After reading the paper this
disclosure is well-founded — the paper itself identifies the anomaly and
excludes the layer from its analyses. We may want to **strengthen** the
disclosure to claim a small extension of the paper's finding: the
anomaly is present in from-scratch training too, suggesting a structural
rather than fine-tuning origin.

---

## 10. Implications for our proposal and write-up

A few methodological disclosures to add to or strengthen in proposal §10.2:

1. **Corpus difference.** Walden vs FineWeb-Edu.
2. **Position sampling difference.** 1 vs 19 positions per chunk.
3. **Layer scope difference.** The paper excludes embedding and
   post-final-norm; we include both. **The post-final-norm layer is
   exactly where the paper finds anomalies in larger trained models,
   and where we also see an outlier (plot 5). Our from-scratch model
   showing the same anomaly is itself a finding worth reporting.**
4. **Sign-ambiguity fix.** OLS projection (us) vs per-layer realignment
   (them).
5. **Statistic convention (§4.5 / §8).** The paper computes
   `mean_i log σᵢ²` over coordinates and `mean_i |κᵢ|` for kurtosis.
   Our analyzer historically computed `log(mean σ²)` and signed
   `mean κᵢ`. As of v3 of this review, the analyzer now computes
   both conventions simultaneously and stores both per checkpoint
   (§11 item 2 complete). The measured Jensen-gap between the two
   conventions on our seed-0 data is 0.03 in log α at convergence
   and 0.50 at step 100 — much smaller than the gap to the paper's
   published values, so this convention difference is now only a
   minor contributor to the gap. The remaining gap is attributable
   to corpus, position-sampling, layer-scope, and training-duration
   differences; quantitative decomposition awaits the
   `--exclude_boundary_layers` refinement (§11 item 1).

---

## 11. Possible analyzer refinements

1. **Add `--exclude_boundary_layers` flag** so we can compute paper-style
   statistics (drop layer 0 and post-final-norm). **Now elevated to the
   most important remaining refinement**: with the convention check
   (item 2) showing only 0.03 log units of contribution, boundary-layer
   inclusion is the next-best candidate to explain the residual gap to
   the paper's published log α. Layer 13 sits ~3 log units below the
   variance-scaling fit line in seed-0's plot 5; excluding it would
   pull log α upward materially. Operates on saved `.npz` files; no
   re-running on checkpoints needed.

2. **~~Add `--statistic_mode {paper, ours}` flag~~ DONE in v3 of this
   review.** The analyzer (`analyze.py`) now computes both conventions
   side-by-side per checkpoint and stores both in the saved `.npz`
   files (`log_alpha`, `lambda` for log-of-mean / signed-mean-κ;
   `log_alpha_paper`, `lambda_paper`, `kurtosis_abs_per_layer` for
   mean-of-log / mean-of-|κ|). The `--statistic_mode {ours, paper,
   both}` CLI flag on `validate_analyzer.py` controls which convention
   is displayed in the summary table; both are computed regardless.
   See `analyze.py` and the v3 measurements in §6.2 and §8.

3. **Add an SVD-realignment option** that mimics
   `compute_order_and_signs` from `fig02.m`. Lower priority — our OLS
   projection is sign-robust by construction, so the realignment would
   only affect downstream alignment-residual statistics, not log α or λ.

Items 1 and 3 operate on saved flow `.npz` files (which contain R(t),
Σ(t), µ(t), per-pair residuals); neither requires re-running on
checkpoints.

---

## 12. Revision history and corrections

### 12.1 v1 → v2 corrections

The v1 version (written from code only, before reading the paper
carefully) made the following errors that v2 corrected:

1. **v1 claimed:** "The paper's trained models show kurtosis 0.9-1.6,
   so our 0.87 is consistent."
   **v2 correction:** That range is the *untrained* baseline (Fig. A6).
   The paper's *trained* models show `<|κ|>` ≈ 0.1-0.3. Our 0.87 is
   meaningfully above the paper's trained-model range. (v2 also
   speculated this was a statistical-convention artifact; v3 finds it
   is not — see below.)

2. **v1 framed log α as "an unresolved mystery."**
   **v2 correction:** Proposed that the mean-of-log vs log-of-mean
   convention accounted for the magnitude of the discrepancy. (v3
   measured the actual gap on our data and found this proposal does
   not hold — see below.)

3. **v1 did not identify GPT-2 medium as the paper's primary model.**
   **v2 correction:** §2 of the paper explicitly says GPT-2 medium is
   the primary model; the larger Llama/Mistral models are extensions.

4. **v1 did not note that the paper explicitly identifies the last-layer
   anomaly.**
   **v2 correction:** §4.3 documents the anomaly for three models and
   conjectures fine-tuning. Our from-scratch pilot showing the same
   anomaly is informative — it argues against the fine-tuning conjecture
   and points to a structural cause (the final norm).

5. **v1 said the effective rank comparison shows 150M uses 43% of H.**
   **v2 correction:** This compares apples to oranges. The paper's
   "256 effective dimensions" is from KL-divergence collapse, not from
   singular-value entropy. The two statistics measure different things.

### 12.2 v2 → v3 corrections

The v2 version made one major hypothesis that v3 has now
disconfirmed via direct measurement. Both conventions are now
computed and saved per checkpoint as of the analyzer modification
(§11, item 2), and the measurements are reported in §4.5, §6.2,
§6.4, and §8.

1. **v2 claimed:** the mean-of-log vs log-of-mean variance convention
   "matches the magnitude of the observed discrepancy" between our log
   α and the paper's published values. The implicit prediction was
   that switching conventions on our data would close most of the gap
   to the paper's published log α values.
   **v3 correction:** the measured Jensen-gap on our seed-0 data is
   0.03 log units at the converged checkpoint (step 24000) and ≤ 0.50
   across the full training trajectory. The gap to the paper's
   published log α (≈ −0.45 for GPT-2 medium, ≈ −5.4 for Llama-2-7B)
   is much larger than this and cannot be explained by the convention
   difference alone. The convention is a *real* methodological
   difference worth disclosing, but it is the smallest of the
   contributing factors — corpus, position-sampling, training-duration,
   and layer-scope differences contribute more.

2. **v2 hypothesized** that the high `<|κ|>` of 0.87 in our seed-0
   final checkpoint might be partly explained by pooled-vs-per-coord
   kurtosis convention (analogous to the variance convention).
   **v3 correction:** on our data, the signed mean `<κ>` and the
   absolute-value mean `<|κ|>` agree to within 0.001 at every
   checkpoint. The per-coordinate kurtosis distribution is essentially
   one-sided (positive) so absolute value and signed mean coincide.
   The high `<|κ|>` is a real finding, not a convention artifact.

3. **v2 claimed** that the ~30-log-unit gap observed when running our
   analyzer on the paper's released `.npy` trajectories was "the
   mean-of-log vs log-of-mean convention difference."
   **v3 correction:** since the convention difference is bounded at
   ≤ 0.50 log units on our data (which spans a wide range of activation
   distributions across 50 checkpoints), it cannot plausibly produce a
   30-log-unit gap on the paper's data either. The 30-log-unit gap on
   their released data is more likely a per-trajectory normalization
   in the released `.npy` files distinct from the data underlying the
   paper's published fits. This is now the cleanest standing open
   question with respect to the paper's released artifacts.

---

## 13. What this review did NOT change

The paper and code review affects how we **interpret and present** our
results, not how we generate them at the training level:

- **Training: unchanged.** Our seed runs do what they would have done.
- **Analyzer numerics on our own data, ours-convention values:
  unchanged.** A small fix to `analyze.py` (handling N < H) is a no-op
  for N=9,500, H=896. That code path doesn't execute. The "ours" log α
  / λ / kurtosis values produced by the v3 analyzer are bit-for-bit
  identical to those produced by the v2 analyzer; v3 only *adds* the
  paper-convention values alongside.
- **Saved flow files, in part:** as of v3, seed-0's `.npz` files have
  been regenerated to include the paper-convention fields
  (`log_alpha_paper`, `lambda_paper`, `pairwise_mean_log_var`,
  `kurtosis_abs_per_layer`, `endpoint_mean_log_var`). The "ours"
  fields are unchanged. Seeds 1, 2, 3 will need their `.npz` files
  regenerated similarly once their checkpoints are available (or
  the new analyzer can be run from the start on those seeds).
- **Phase 1 final plots for seed-0: still valid.** They show the
  ours-convention numbers, which are unchanged. Adding paper-convention
  overlays to the plots is a `flow_series.py` / `plots.py` change still
  pending.

---

## References

- Sarfati et al. *Lines of Thought in Large Language Models*. ICLR 2025.
  [arXiv:2410.01545](https://arxiv.org/abs/2410.01545)
- Repository: [github.com/rapsar/lines-of-thought](https://github.com/rapsar/lines-of-thought)
