# Input-Conditioned Ensembles in Lines of Thought: A Mathematical Investigation

**Phase 2 supplementary investigation, multiview campaign**
**Run: phase1_runs_gelu, seeds 0–3, 50 checkpoints, final step 24000**

---

## Abstract

The "Lines of Thought" paper (Sarfati et al., ICLR 2025) shows that residual-stream activations across a held-out set of inputs form an approximately Gaussian "blunderbuss" at each layer of a trained transformer, with a covariance that grows exponentially in depth at a universal rate $\lambda$. The paper models this with a Langevin-style stochastic differential equation. We investigated whether this macroscopic description has a clean microscopic counterpart by partitioning the residual-stream ensemble on the input token at the pilot position.

The answer is negative on multiple axes. Input-conditioned bundles are strongly non-Gaussian (Mardia $Z = 25\text{–}45$ across interior layers), with within-input total variance differing across input tokens by factors of 2–4×, and per-token exponential growth rates $\lambda_i$ that vary across tokens and across training steps in a structured, non-monotonic way. The paper's universal $(\alpha, \lambda)$ are therefore marginal-level central-limit-theorem statistics, not microscopic constants.

The non-Gaussianity of the conditional bundles is real and emerges progressively during training, with baseline conditional kurtosis at an interior layer growing from $\sim 0.1$ (Phase I, rapid SVD consolidation) to $\sim 5\text{–}8$ (Phase III final, post-restructuring). We initially interpreted the non-Gaussianity as a context-mixture phenomenon: $p(x_t \mid v_i)$ being itself a Gaussian mixture over the local context in which token $v_i$ appears. A first round of sub-conditioning analyses appeared to support this. Subsequent null-controlled analyses showed this support was largely a sample-partitioning artifact: simple random labelings reduce kurtosis by amounts comparable to next-token or position partitions. After proper null correction, coarse position carries a real signal of ~10% of the baseline heavy-tail magnitude and next-token carries ~4%; together they leave roughly 85% of the non-Gaussian structure unexplained by any single-variable mixture interpretation we tested.

We document the affirmative results (the Models A/B verdict, the grammatical-role bimodality in per-token $\lambda$, the three-phase consolidation/restructuring training trajectory) alongside the negative ones (the failure of the context-mixture interpretation under proper null correction), and discuss what kinds of structure remain plausible candidates for explaining the C verdict.

---

## 1. Background and motivation

### 1.1 Residual streams as a stochastic ensemble

Transformers process tokens through a sequence of residual blocks. At each layer $t$, every token position carries a residual-stream vector $x_t \in \mathbb{R}^H$, the running sum of all attention and MLP contributions up to that depth. For a held-out evaluation set with many input chunks and many token positions per chunk, the collection of these vectors at fixed layer $t$ forms an empirical distribution — the "ensemble" of residual-stream activations at depth $t$.

The Lines of Thought paper studies this ensemble's geometry across depth. Their headline findings: (i) the ensemble at each layer is approximately a single Gaussian whose covariance grows exponentially in depth at a universal rate $\lambda$, isotropically along the directions tangent to the singular vectors of the layer-to-layer mean flow, and (ii) trajectories of individual tokens through depth can be modeled by a continuous-time Langevin equation

$$dx = A(t)\,x\,dt + \sigma(t)\,dW, \qquad \sigma\sigma^\top = \alpha e^{\lambda t} I$$

with universal $(\alpha, \lambda)$ measured from the marginal-level statistics. We call the all-token, all-position empirical distribution at layer $t$ the **all-to-all** bundle (or **marginal**).

The Langevin formulation is a *phenomenological model* of the marginal ensemble's variance growth, not a microscopic dynamical law. Transformer forward passes are deterministic given input and weights — the "noise" in the SDE represents heterogeneity across the ensemble (different inputs, different positions, different surrounding contexts) reinterpreted as fictitious process noise. The paper measures and models the marginal ensemble; it makes no direct empirical claim about sub-ensembles conditional on any particular input token.

### 1.2 The conditional ensemble question

Our multiview campaign extends the Lines of Thought framework by *partitioning* the all-to-all bundle into sub-bundles based on observable conditioning variables:

- **Forward view (input-conditioned):** fix the input token at the pilot position. The sub-bundle is $\{x_t^{(k)} : \text{input}_k = v_i\}$ for each vocabulary token $v_i$.
- **Reverse view (output-conditioned):** fix the predicted next token (argmax of the model's output distribution).
- **All-to-all view:** the full marginal, as in the paper.

By the law of total variance, the marginal's covariance decomposes exactly into within-component and between-component pieces:

$$\Sigma_{\text{marginal}}(t) = \underbrace{\sum_i \pi_i \Sigma_i(t)}_{\text{within: average of conditional covariances}} + \underbrace{\sum_i \pi_i (\mu_i(t) - \mu(t))(\mu_i(t) - \mu(t))^\top}_{\text{between: variance of conditional means}}.$$

Here $\pi_i$ is the empirical frequency of input token $v_i$, $\mu_i(t)$ and $\Sigma_i(t)$ are the conditional mean and covariance, and $\mu(t)$ is the marginal mean. This identity is bookkeeping — it must hold to numerical precision.

The interesting question is **what the conditional bundles look like**. If the marginal is Gaussian, the conditionals are constrained in their first two moments but their shape is underdetermined by the marginal alone.

### 1.3 The question

> **Is there a mathematical description for the input-conditioned ensembles $p(x_t \mid v_i)$ that has the same form as the Lines of Thought description of the marginal ensemble, and if so, how does it relate to the marginal description?**

The marginal blunderbuss is, by total-probability bookkeeping, a Gaussian Mixture Model over the vocabulary of input tokens, weighted by token frequency. The question becomes: what is the structure of the components of that mixture, and does that structure satisfy a clean closed-form description that recovers the paper's marginal model under mixing?

---

## 2. Hypothesis formation

We began from the simplest candidate unification: a GMM where each input token contributes a Gaussian sub-bundle with token-specific drift $\mu_i(t)$ and a shared isotropic exponential covariance $\alpha e^{\lambda t} I$. Under this picture, the macroscopic Gaussian observed by the paper would be the marginal of a structured fluid of $|V|$ conditional Gaussians, with the paper's universal $(\alpha, \lambda)$ identified as microscopic noise parameters governing every conditional bundle, and the per-token drifts supplying the between-component variance.

Three issues with this synthesis emerged on examination:

1. **The paper's empirical Gaussianity is about linearization residuals, not conditional bundles.** Sarfati et al. measure the distribution of $\delta x(t, \tau) = x(t+\tau) - \tilde x(t, \tau)$, where $\tilde x$ is a linear extrapolation along the all-to-all SVD basis. They do not measure or claim that $p(x_t \mid v_i)$ is Gaussian for any fixed $v_i$.

2. **The Langevin equation is a marginal-level metaphor.** The "noise" is heterogeneity treated as a fictitious process. Whether this corresponds to a real Gaussian process at the conditional level is an empirical question.

3. **The "high dimensions smear modes together" intuition is backwards.** Concentration of measure makes Gaussians with even modest mean separations *more* separable in high dimensions, not less. If the marginal looks unimodal, it is because either the between-token mean separations are small relative to within-token spread, or the within-token covariance is large and structured enough to fill the gaps along separating directions.

The reframed question: **what kind of conditional structure is consistent with the observed marginal Gaussianity, and which kind actually obtains in our trained model?**

### 2.1 Three candidate models

- **Model A — Gaussian conditionals with shared covariance.** $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \Sigma_0(t))$, same $\Sigma_0$ for every input token. Conditionals differ only in drift. Paper's universal $(\alpha, \lambda)$ live at the microscopic level.

- **Model B — Gaussian conditionals with token-dependent covariance.** $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \Sigma_i(t))$ with $\Sigma_i$ varying across tokens. Paper's universal parameters are aggregate averages over token-specific values.

- **Model C — Non-Gaussian conditionals that mix to approximate marginal Gaussianity.** Paper's microscopic description does not extend to conditionals; the SDE framework is a marginal-level phenomenology with no microscopic counterpart.

---

## 3. Experimental design

We designed five discriminators D1–D5, each addressing a different empirical signature of A vs B vs C. The campaign uses a 150M-parameter Llama-style transformer (14 residual-stream snapshots per forward pass, $H = 896$), trained from scratch on 4 seeds with 50 logarithmically-spaced checkpoints. The held-out set provides ~10,000 pilots per checkpoint, with the top-20 most frequent input tokens forming the **frozen forward set** used for per-token analysis. Per-token pilot counts in this set range from ~50 to ~400.

### 3.1 The discriminators

**D1 — Cross-token coefficient of variation of within-input variance.** Under Model A, the total within-input variance $\mathrm{tr}(\Sigma_i(t))$ is the same for every input token at each layer (modulo finite-sample noise). Under B or C, it varies.

**D2 — Principal angles between conditional SVD bases.** Under Model A with shared $\Sigma_0$, every conditional bundle spans the same subspace. Under B, the bases diverge.

**D3 — Per-token $(\alpha_i, \lambda_i)$ fits.** Model A predicts $(\alpha_i, \lambda_i)$ constant across tokens and matching the paper's marginal values. Model B predicts scatter.

**D4 — Multivariate Gaussianity of conditional bundles.** Per-coordinate excess kurtosis (D4a) and Mardia multivariate kurtosis Z-score (D4b). Both reject the Gaussianity assumption common to A and B.

**D5 — GMM reconstruction.** Sample synthetic residuals from per-token Gaussian fits, weight by empirical frequencies, and compare synthesized marginal kurtosis against directly observed marginal kurtosis. If the synthesized marginal is markedly more Gaussian than the observed marginal, conditionals must themselves be non-Gaussian.

### 3.2 Threshold calibration via bootstrap noise floor

The verdict logic requires numerical thresholds. We measured the finite-sample noise floor at the actual pilot counts by bootstrap resampling within tokens (B=100 replicates per token at seed 0, step 24000), and cross-checked against cross-seed standard deviations.

The calibration produced threshold candidates of CV($\mathrm{tr}\,\Sigma_i$) ≈ 0.115 (after trimming the newline-token outlier), CV($\lambda_i$) ≈ 0.062, multivariate kurtosis $Z$ ≈ 0.234. The D2 calibration revealed that **top-10 principal directions are sample-size-degenerate at our pilot counts** (mean self-pair angle 35.9°, vs the 90° upper bound). We restricted D2 to the top-2 directions where stability is higher, and demoted D2 from a primary verdict driver to a confirmatory diagnostic. This is a generally useful methodological point: principal-subspace discriminators in high dimensions require bootstrap-floor calibration before deployment.

---

## 4. Models A and B are refuted; Model C verdict

### 4.1 The verdict

All four seeds gave the same per-layer verdict pattern at the final checkpoint:

| Layer t   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Verdict   | B | C | C | C | C | C | C | C | C | C | C  | C  | C  | C  |

13 of 14 layers tagged "C". Layer 0 is the raw embedding where for a fixed input token every pilot has the same residual-stream vector, so the diagnostic is structurally degenerate. The interesting result is layers 1–13.

### 4.2 D1: within-input variance is strongly token-dependent (Model A refuted)

Cross-token CV of $\mathrm{tr}(\Sigma_i(t))$ sits at **0.6–0.8 across every interior layer**, with a spike to 1.7 at layer 1. The threshold is 0.115. The signal exceeds threshold by 5–7×. Different input tokens produce within-input bundles whose total variance differs by factors of 2–4×. The shared-$\Sigma_0$ picture is qualitatively wrong.

### 4.3 D3: $\lambda$ is not universal, with a non-monotonic training trajectory

Per-token $\lambda$ values at the final checkpoint show a bimodal distribution: most tokens cluster near $\lambda \approx 0.55\text{–}0.60$, with a separated lower cluster near $\lambda \approx 0.40$ (8 of 20 forward-set tokens). The all-to-all $\lambda \approx 0.36$ sits *below* every per-token value — geometrically sensible, since the marginal variance growth is dominated by the slower of the within-token and between-token components.

CV($\lambda$) across tokens at the final checkpoint is 0.14, above the 0.062 threshold. The training trajectory of CV($\lambda$) is non-monotonic: starting at ~0.20 (step 100), declining to ~0.09 (step 1500–2000), rising back to ~0.14 (step 24000). This is one of the more interesting individual results of the campaign and is taken up in §6.

### 4.4 D4: conditional bundles are not Gaussian (Model B refuted)

D4a (per-coordinate kurtosis in each token's own SVD basis) shows conditional bundles with excess kurtosis 1.5–6 across interior layers, vs the all-to-all marginal kurtosis near 0–0.5. The conditional bundles are *more* non-Gaussian than the marginal.

D4b (Mardia multivariate kurtosis in a shared 32-dimensional PCA subspace) gives Mardia $Z$ values of **25–45** at most interior layers, vs the $|Z| = 2$ rejection threshold at 5%. Rejection of multivariate Gaussianity is overwhelming.

### 4.5 D5: the GMM reconstruction confirms

We take per-token empirically measured means and covariances, weight by empirical frequencies, and sample from the resulting mixture:

- **Empirical marginal:** excess kurtosis 5–15 across interior layers 1–7.
- **Model A reconstruction** (shared $\Sigma_0$): excess kurtosis ≈ 0 throughout.
- **Model B reconstruction** (per-token $\Sigma_i$): excess kurtosis 0.3–0.7 throughout.

Both Gaussian-mixture reconstructions produce nearly Gaussian marginals. The actual marginal has substantial excess kurtosis. The conditional bundles cannot be Gaussian — Gaussian-fitting them and remixing produces a markedly more Gaussian object than the data shows.

### 4.6 The Model C verdict, established

Putting D1, D3, D4, D5 together: the marginal Gaussian observed by Sarfati et al. is **not** decomposable as a Gaussian mixture over input-token-conditioned components. The conditionals have token-dependent variance, token-dependent exponential growth rate, are themselves substantially non-Gaussian, and aggregating any Gaussian fit of them does not reproduce the empirical marginal's higher-order structure.

The micro–macro relationship is real and bookkeeping-exact via the law of total variance, but the *form* of the conditionals is not a direct extension of the marginal description. There is no $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \alpha e^{\lambda t} I)$ that adequately describes any conditional bundle.

**The Lines of Thought framework's marginal universality is a central-limit-theorem phenomenon over a population of structurally distinct, non-Gaussian conditional bundles.**

---

## 5. The grammatical-role bimodality

The D3 bimodal $\lambda$ distribution is interesting enough to characterize in detail. We ran k-means (k=2) on per-token $\lambda$ values averaged across seeds at the final checkpoint, decoded the cluster members using the Mistral-7B tokenizer, tested cross-seed stability, and applied Welch's t-tests at each layer comparing the clusters on trace, effective rank, and kurtosis profiles.

### 5.1 Cluster membership

- **Low cluster** ($\lambda \approx 0.45$, 8 tokens): `\n`, `.`, `-`, `0`, `1`, `2`, `s` (likely the orphan possessive/contraction fragment), and a likely BPE space marker. *Structural/sub-word tokens*: terminators, digits, and sub-lexical fragments.
- **High cluster** ($\lambda \approx 0.58$, 12 tokens): `the`, `of`, `and`, `to`, `a`, `in`, `is`, `that`, `are`, `for`, plus `,` and `(`. *Connective tokens*: function words plus the internal punctuation that functions syntactically like glue rather than as a terminator.

The distinction is not "punctuation vs words" but a more refined grammatical-role split: tokens that *terminate or fragment* a syntactic unit (low cluster) vs tokens that *connect within* a unit (high cluster). The comma and open-parenthesis in the high cluster fit this interpretation; the period in the low cluster is a strong terminator.

### 5.2 Cross-seed stability

17 of 20 tokens have stable cluster assignment across all four seeds. The 3 unstable tokens (`.`, `s`, and the BPE space marker) all sit near the cluster boundary at $\lambda \approx 0.48\text{–}0.50$. Unambiguous cluster members (`\n`, `-`, digits in the low cluster; all 12 high-cluster tokens) are stable. The bimodality is a real structural distinction with a small fuzzy boundary region.

### 5.3 Effective rank: the sharper discriminator

The bimodality manifests *more strongly in effective rank than in $\lambda$*. The Welch t-test for effective rank by cluster is significant ($p < 0.05$) at every interior layer 1-9, and highly significant ($p < 0.02$) at layers 1-4. At layer 1, low-cluster mean effective rank is 6.4 vs high-cluster 19.8 ($p < 10^{-4}$). At layer 5, 37.7 vs 63.7 ($p \approx 0.017$). **Structural-token conditional bundles occupy roughly half the residual-stream dimensions that function-word bundles occupy across the entire model depth.**

Geometrically, lower $\lambda$ and lower effective rank are likely two views of the same phenomenon: bundles confined to a lower-dimensional subspace cannot expand as quickly in trace terms because the directions of expansion are fewer.

### 5.4 Training-time emergence

The bimodality emerges sharply between training steps ~300 and ~1000 and continues to tighten through step 24000. Cohen's $d$ (standardized cluster separation) is essentially zero through step 200, climbs from step 300, reaches $d \approx 1.5$ by step 1000, $d \approx 3$ by step 10000, $d \approx 4$ by step 24000. The clusters do not separate by centroid drift; rather, within-cluster spread shrinks as training proceeds, sharpening the standardized separation. The clusters *crystallize* during training.

### 5.5 Kurtosis is not discriminative

Per-layer Welch t-tests on the kurtosis profile are non-significant at every layer (all $p > 0.1$). Both clusters are equally non-Gaussian. The cluster distinction is about *which subspace* the conditional bundle occupies, not about *whether* it is Gaussian. The Model C non-Gaussianity is a general feature of conditional bundles, not specific to one grammatical class.

### 5.6 Implication

The Lines of Thought framework's universal $\lambda$ is not only marginalized over a population of token-specific values (as established in §4.3) — that population has *legible grammatical structure*. Function words and structural markers form distinct dynamical classes in the residual stream geometry. The bulk of the marginal $\lambda$ value comes from the larger function-word cluster; structural tokens are a smaller minority cluster pulling the marginal value down.

---

## 6. Three training phases of residual-stream geometry

The CV($\lambda$) trajectory's mid-training minimum near step 1500 is not isolated. Loading Phase 1's per-checkpoint flow files for all four seeds and overlaying their marginal trajectories — log-$\alpha$, all-to-all $\lambda$, mean kurtosis, post-final-norm boundary residual, and a basis-invariant singular-value spectrum distance to the final checkpoint — reveals three distinct phases of residual-stream geometry development.

### 6.1 The phases

- **Phase I, steps 100–1500: rapid SVD geometry consolidation.** The singular value spectrum drops 75% of the way to its final form. Marginal kurtosis falls toward its minimum. CV($\lambda$) falls toward its minimum. Post-final-norm anomaly grows from $-3.6$ to $-1.8$. Per-token $\lambda$ clusters have not yet crystallized.

- **Phase II, steps 1500–5000: consolidated mid-training plateau.** $\Sigma$-distance to final stays in 0.15–0.20. Marginal kurtosis at its minimum (~0.24–0.30). CV($\lambda$) at its minimum (~0.10). Post-final-norm anomaly plateaus at $-1.8$. log-$\alpha$ reaches its hump peak.

- **Phase III, steps 5000–24000: late-training restructuring.** $\Sigma$-distance shows a brief rebound (the Phase 1 documented bump) and then resumes declining. Marginal kurtosis rises substantially (heavier tails). CV($\lambda$) rises back (per-token $\lambda$ values diverging into clusters). log-$\alpha$ descends from its hump. Post-final-norm anomaly resumes growing more negative. Grammatical clusters crystallize.

### 6.2 Quantitative co-location

The CV($\lambda$) minimum at step 1465 sits exactly at the elbow between Phase I and Phase II — the point where rapid early convergence completes and the plateau begins. The marginal kurtosis minimum is ~500 steps earlier in the same elbow region. The post-final-norm anomaly's plateau begins ~500 steps later. Three independent diagnostics — conditional-dynamics universality, marginal Gaussianity, and boundary anomaly emergence completion — co-locate within ~500 training steps at the start of Phase II.

### 6.3 Implication for the central question

The Lines of Thought paper measures fully-trained models. Their measurements fall in our Phase III, the late-training restructuring phase. The paper's "approximately Gaussian marginal" describes a *post-restructured* state, not the cleanest Gaussian-like state the model passes through (Phase II).

The Model C conditional non-Gaussianity we documented in §4 is *also* a Phase III phenomenon. Conditional bundles are nearly Gaussian during Phase II and develop their heavy tails progressively through Phase III. The C verdict is not an inevitable property of any trained transformer; it is a property of the *post-consolidation* trained transformer, alongside the rise in marginal kurtosis and the divergence of per-token $\lambda$ values.

---

## 7. The context-mixture interpretation and its failure under null correction

This section is the most uncomfortable part of the writeup, because it documents a substantive interpretive claim that the analysis initially appeared to support and that subsequent null-controlled analysis substantially refuted. We present the trajectory as it unfolded.

### 7.1 The provisional context-mixture hypothesis

The natural interpretation of the Model C verdict was that $p(x_t \mid v_i)$ is itself a mixture — over the local context in which the input token appears. The same token "the" preceded by "of" or by a period sits in different regions of residual space; aggregating across contexts produces heavy tails and non-Gaussian shape even if each context-specific sub-bundle is individually Gaussian.

This was testable using the augmented activation files, which record per pilot: the input token, the actual next token in text, the model's predicted next token, and the pilot's position in the chunk.

### 7.2 First sub-conditioning result and its apparent support

We partitioned each input-conditioned bundle three ways and recomputed aggregate kurtosis on each sub-bundle: by next-token, by predicted-token, by position. Sub-bundles with fewer than 20 pilots were dropped, with sample-size-weighted means across sub-bundles giving the aggregate.

The result *appeared* to confirm the context-mixture hypothesis. Sub-conditioning on next-token collapsed aggregate kurtosis from baseline ~2.5 at layer 1 to ~0.3, and from ~1.0 at layer 5 to ~0.1. For the newline token (id 13), baseline kurtosis 7.66 at layer 7 collapsed to 0.02 when sub-conditioned on next-token. Sub-conditioning on predicted token reduced kurtosis less; sub-conditioning on position reduced kurtosis comparably to next-token.

### 7.3 The next-vs-position identifiability problem (initial framing)

The position result was unexpected and prompted a joint-partition analysis. NMI(next; position) across forward-set tokens ranged 0.38–0.69, with median 0.55, in the original fixed-pilot-position scheme. The joint-partition test could not be run due to sample-size shattering (zero sub-bundles with ≥20 samples at the joint resolution). We attributed the apparent equivalence of next-token and position to entanglement of the two variables in the pilot construction, and proposed that regenerating stage A with randomized pilot positions per chunk would disentangle them and identify the causal variable.

### 7.4 The stage A regeneration and the first surprise

We regenerated augmented activations with per-sequence randomized pilot positions at four representative checkpoints (one per training phase), maintaining the same total pilot count per checkpoint. The new files use 1023 unique positions (one per available chunk index) with each pilot independently drawn.

A direct check of NMI(next; position) on the new files gave **0.96**, *higher* than the original 0.55. With 1023 position buckets and ~19,500 pilots, each (token, position) cell has roughly one sample, so position trivially "determines" next-token at fine resolution. Coarsening position into quantile buckets reduced NMI smoothly: at 5 buckets, NMI was 0.22, matched to the original scheme's resolution at 20 buckets.

Running the cross-phase sub-conditioning analysis on the new files with coarse-binned position, baseline conditional kurtosis at layer 7 grew progressively across phases (0.12 → 0.48 → 2.78 → 5.91), confirming §6's training-time emergence. *Sub-conditioning on position reduced kurtosis more strongly than sub-conditioning on next-token at every phase*, by factors of 1.3× to 16×. This appeared to overturn the original §7.2 reading and identify position rather than next-token as the relevant mixture coordinate.

### 7.5 The null-control reckoning

Before publishing the position-vs-next-token result, we asked a question that should have been asked from the start: how much of the apparent partition effect is a sample-partitioning *artifact*? Kurtosis is a fourth-moment statistic with a known downward bias under sample-size reduction. Splitting a heavy-tailed sample into $k$ subgroups and computing within-subgroup kurtosis systematically underestimates the original kurtosis, *even when the subgroups are formed by random labeling rather than any meaningful partition*.

The diagnostic is the matched random-labels null: for each input token, assign random integer labels in $[0, k)$ to pilots, compute the same weighted-mean kurtosis, repeat with multiple random seeds and average. The "real signal" of a partition variable is the residual kurtosis reduction beyond what random labels achieve at the same $k$.

#### Position null control

We swept bin counts $k \in \{2, 3, 4, 5, 7, 10, 15, 20, 30\}$ at all four checkpoints, with 10 independent random labelings per $k$. The result is striking:

At Phase III final, layer 7 (baseline kurtosis 5.91):

| $k$ | quantile binning reduction | random null reduction | real signal |
|---|---|---|---|
| 2 | 3.56 | 2.77 | 0.79 |
| 3 | 4.50 | 3.62 | 0.88 |
| 5 | 5.13 | 4.63 | 0.50 |
| 10 | 5.42 | 5.24 | 0.18 |
| 20 | 5.56 | 5.53 | 0.03 |
| 30 | 5.67 | 5.59 | 0.09 |

The real position signal peaks at $k=3$ with a value of 0.88, then *declines* as we add more bins because the random null catches up. By $k=20$, 96% of the apparent position effect is sample-partitioning artifact.

#### Next-token null control

Next-token's effective bin count per input token is the number of distinct next-tokens with ≥20 pilots, which averages 4.0 at the final checkpoint with the relaxed threshold. The matched random null assigns random labels in $[0, K_i)$ for each input token, with $K_i$ matching that token's next-token effective bin count, averaged over 20 independent replicates.

At Phase III final, layer 7 (baseline kurtosis 5.78):

- next-token reduction (uncorrected): 4.27
- random null reduction (matched $k \approx 4$): 4.02
- **real signal: 0.25**

Cross-phase, the real next-token signal at layer 7 is:

| Phase | baseline | next reduction | null reduction | real signal |
|---|---|---|---|---|
| Phase I | 0.20 | 0.29 | 0.07 | +0.22 |
| Phase II | 0.55 | 0.11 | 0.17 | −0.06 |
| Phase III mid | 2.32 | 1.15 | 1.36 | −0.21 |
| Phase III final | 5.78 | 4.27 | 4.02 | +0.25 |

The real signal **oscillates sign across phases** and never exceeds 0.3. When a measured "real effect" oscillates sign through training, it is not a real effect — it is noise around zero.

### 7.6 What the null-controlled analyses tell us

Combining the position and next-token null controls:

| Variable | Real signal at Phase III final (layer 7) | Fraction of baseline (~5.8 to 5.9) explained |
|---|---|---|
| Position (coarse, $k=2\text{–}3$) | ~0.8 | ~13% |
| Next-token (matched $k \approx 4$) | ~0.25 (statistically indistinguishable from 0) | ~4% |

Both real signals are small. The original §7.2 sub-conditioning result that next-token "collapsed" the conditional bundle to near-Gaussian was almost entirely a sample-partitioning artifact. The §7.4 regeneration finding that position is a more effective mixture coordinate was also largely artifact; the real position signal is modest and lives at very coarse resolution (2–3 bins).

The honest claim is: the Model C heavy-tail structure of $p(x_t \mid v_i)$ is **not** explained by Gaussian-mixture decomposition along either of the two single-variable context partitions we tested. Position carries a small real signal at coarse resolution, accounting for roughly 10–13% of the heavy-tail magnitude. Next-token, after null correction, carries no signal we can robustly distinguish from zero.

The retracted claims are:

- The original §7.2 framing that the C verdict represents "context mixture, identifiable up to the (next, position) joint distribution."
- The post-regeneration §7.4 claim that position is "the natural mixture coordinate."
- The Finding (in earlier writeup drafts) that "context mixture, parameterized either as next-token or position, restores Gaussianity at the level of $p(x_t \mid v_i, c)$."

What we retain:

- The Model C verdict itself (§4.4–4.6) — the conditional bundles are non-Gaussian by clean multivariate diagnostics that do not depend on sample partitioning.
- The §5 grammatical bimodality, which is independent of any partition mechanism.
- The §6 three-phase training trajectory, which is also independent.
- A small (∼10%), real, coarse positional component of the heavy tails. This is a true finding, but a much weaker claim than was previously made.

---

## 8. Where this leaves the central question

The investigation set out to ask whether there is a mathematical description of $p(x_t \mid v_i)$ in the same form the Lines of Thought paper uses for the marginal. The answer is multi-layered:

1. **Models A and B are refuted.** No Gaussian conditional description, with shared or token-dependent covariance, recovers the empirical marginal or matches the conditional Gaussianity diagnostics.

2. **The conditional bundles are genuinely non-Gaussian.** Multivariate Gaussianity is rejected at Mardia $Z = 25$–$45$, and GMM reconstruction from Gaussian fits produces marginals 5–15× more Gaussian than observed.

3. **The non-Gaussianity emerges during training.** It is essentially absent at Phase I (baseline conditional kurtosis ~0.1), modest at Phase II (~0.5), and full-magnitude only at Phase III (~5–8). The Lines of Thought paper's measurements describe the Phase III state.

4. **The simplest mixture interpretations of the non-Gaussianity do not survive null correction.** Neither next-token nor position partitions explain the bulk of the heavy-tail structure. Coarse position carries a small real signal (~10% of magnitude); next-token carries essentially none.

5. **The per-token $\lambda$ has legible grammatical structure** (§5), with function-word vs structural-token clusters that crystallize during training and differ on effective rank by ~2×.

6. **The training trajectory has a three-phase structure** (§6) in which Phase II is the brief consolidated window where the Lines of Thought framework's universality is most cleanly satisfied, and Phase III is where both the conditional non-Gaussianity and the paper's measured marginal state develop together.

The Lines of Thought framework describes a snapshot of a particular training phase (Phase III), in which the marginal happens to be approximately Gaussian by central-limit-theorem averaging over a population of structurally distinct, non-Gaussian conditional bundles. The microscopic counterpart of the paper's macroscopic universal $(\alpha, \lambda)$ does not exist in any of the forms we tested. What the heavy tails of the conditionals actually *are* — what generates them, what structure they have — remains an open question.

---

## 9. What the heavy tails might actually be

After null correction, the bulk of the Model C heavy-tail magnitude is *not* explained by Gaussian-mixture decomposition along next-token or position. Three live possibilities remain for what does explain it.

### 9.1 Possibility 1: Intrinsic heavy-tailed conditionals, not a mixture

The conditional bundle $p(x_t \mid v_i)$ may genuinely be a single heavy-tailed (e.g., multivariate t or elliptical) distribution rather than a Gaussian mixture. Under this hypothesis, no partitioning of the bundle will restore Gaussianity, because the bundle is fundamentally non-Gaussian at every scale.

This would be a more fundamental departure from the Lines of Thought framework than a mixture interpretation. The paper's Langevin SDE produces Gaussian distributions by construction (linear drift plus Brownian noise). A heavy-tailed conditional would require either a non-linear drift, a non-Gaussian noise process (e.g., $\alpha$-stable noise), or both.

**Test:** Fit a multivariate t-distribution with free degrees-of-freedom parameter $\nu$ to the conditional bundle at Phase III final. If $\nu$ is low (say $< 10$) and the t-fit substantially outperforms a Gaussian fit on likelihood, intrinsic heavy-tail is supported. If $\nu$ is high ($> 30$) and the t-fit barely improves on Gaussian, this possibility is weakened.

### 9.2 Possibility 2: Rare-extreme-context mixture

A small fraction of pilots may be in unusual contexts (rare next-tokens, document boundaries, syntactic discontinuities, special characters) that contribute extreme residual-stream activations. If 95% of pilots are in routine contexts producing a roughly Gaussian bundle and 5% are in extreme contexts producing very different activations, the conditional distribution is technically a mixture, but the "rare extreme" sub-component has too few samples to be recovered by any naive partition.

**Test:** Trimmed kurtosis. Remove the most extreme 5% of pilots per input token (e.g., highest Mahalanobis distance from the per-token mean) and recompute baseline kurtosis. If kurtosis collapses substantially, this possibility is supported: the heavy tails come from a small subset of extreme contexts, and the bulk of the bundle is approximately Gaussian once the outliers are removed.

### 9.3 Possibility 3: High-order context dependency

The relevant mixture variable may be a function of multiple positions rather than any single-position partition we tested. The 2-gram (prev_token, next_token), the 3-gram window around the pilot, the syntactic class of the surrounding sentence, the semantic field of the document — any of these could be the relevant mixture coordinate. The single-position partitions we tested (next-token, position) would each be partial proxies for this richer mixture index and would each fail to recover Gaussianity individually.

**Test:** Re-extract the original chunk text for each pilot and partition on (prev_token, next_token) joint identity, or longer-window functions. Requires re-running stage A with prev_token saved alongside next_token (currently the augmented files do not save prev_token), or matching pilots back to the original tokenized chunks via stored chunk identifiers and positions.

---

## 10. Suggested follow-up directions

### 10.1 Trimmed kurtosis (Possibility 2)

The cheapest test of the three. Take the augmented files we already have (original fixed-position scheme is fine for this — no need for randomized files), and for each input token in the forward set:

- Compute per-pilot Mahalanobis distance from the per-token mean at each layer.
- Remove the highest-distance 1%, 5%, 10% of pilots.
- Recompute baseline kurtosis on the trimmed bundle.

If the kurtosis at Phase III final drops from ~6 to ~1 or below after trimming 5% of pilots, the rare-extreme-context hypothesis is strongly supported. If it remains at ~5 even after trimming 10%, this possibility is refuted and we are left with Possibilities 1 or 3.

Expected runtime: 1–2 hours. No additional data collection needed.

### 10.2 Multivariate t-distribution fit (Possibility 1)

Fit a multivariate t-distribution with free degrees-of-freedom $\nu$ to each conditional bundle at Phase III final (and ideally at the other three phases for the trajectory view). Compare log-likelihood to a multivariate Gaussian fit, report best-fit $\nu$ per token, and check whether $\nu$ varies systematically across the grammatical clusters from §5.

If best-fit $\nu \in [3, 10]$ across most tokens and the t-fit gives substantially higher log-likelihood than the Gaussian fit, intrinsic heavy-tail is supported. If $\nu$ is generally large ($> 30$), the bundle is essentially Gaussian and the kurtosis is coming from a small fraction of outliers (Possibility 2 reinterpretation).

Expected runtime: 2–4 hours, dominated by per-token MLE fitting with EM. Requires no new data.

### 10.3 n-gram partition (Possibility 3)

Augment stage A to save prev_token alongside the existing next_token field, regenerate at the four checkpoints with this extension, then run sub-conditioning on (prev, next) joint and on prev alone, both with proper null controls.

Sample-size constraints will be harder than for single-variable partitions: a typical (input, prev, next) cell may have only a handful of pilots even with the relaxed min_subbundle threshold. The analysis will need to focus on the few input tokens with hundreds of pilots (e.g., `the`, `of`) where joint cells can reach the minimum.

Expected runtime: similar to the existing stage A regeneration (10 minutes of inference) plus a follow-up analysis pass (~30 minutes).

### 10.4 Cross-checkpoint Models A/B/C verdict

The Models A/B/C discriminator was run only at the final checkpoint. Given §6's finding that conditional non-Gaussianity is a Phase III phenomenon, an interesting auxiliary result would be to run the full discriminator suite at the four representative checkpoints (or even all 50) and report at which step each layer transitions from A/B (no clear non-Gaussianity) to C (firmly non-Gaussian). This would extend §6's "Phase III is where the structure develops" story with checkpoint-level resolution on the discriminator outputs themselves.

Expected runtime: 1–2 hours per checkpoint × 4 checkpoints, or full sweep ~1 day if run over all 50.

### 10.5 Recommended sequence

If we were to extend the investigation, the natural order is 10.1 (trimmed kurtosis, ~1–2 hours, can decisively rule in or rule out the rare-extreme hypothesis) → 10.2 (multivariate t fit, ~2–4 hours, characterizes the bundle shape if 10.1 ruled out rare-extreme) → 10.3 (n-gram partition, ~1 hour data + follow-up if 10.2 left the structure unexplained). 10.4 is a parallel auxiliary that strengthens §6 regardless of the outcomes of 10.1–10.3.

A reasonable stopping point is after 10.1: if trimmed kurtosis substantially reduces the heavy tails, the C verdict is fully characterized (rare-extreme-context mixture); if it does not, we have a substantive negative result and the investigation as it stands documents a real gap in the Lines of Thought framework's microscopic extension.
