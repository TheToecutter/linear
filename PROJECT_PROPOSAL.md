# The Linear Flow of Trained Transformers: A Pilot Study at 150M Scale

## A self-contained empirical investigation of convergence, universality, predictability, and exploitation of the population-level linear structure of transformer language models

---

## Abstract

The lines-of-thought framework of Sarfati et al. (ICLR 2025) shows that trained transformer language models have approximately linear-Gaussian per-layer dynamics: token trajectories through depth follow a deterministic linear flow, modulo a stochastic residual whose statistics describe per-input deviations from that flow. The linear flow itself is recovered from the SVD of population activation statistics and represents the *average* residual-stream transformation each layer produces over the model's input distribution.

We propose a pilot study at 150M parameters, conducted entirely on a single-workstation GPU, characterizing four things this framework leaves open. First, *convergence*: does the recovered linear flow stabilize as a function of training compute, and how does its convergence relate to loss-curve convergence? Second, *universality*: across four modern consensus architectures — Llama-style, Gemma-style, Qwen-style, and DeepSeek-style with MLA — trained on a fixed corpus to similar quality, does the linear flow converge to a shared asymptote? Third, *predictability*: can the converged linear flow be forecast from partial training, accurately enough to be useful? Fourth, *exploitation*: can a predicted linear flow be applied during training, as a diagnostic monitor, regularization target, or initialization signal, to improve outcomes?

A central methodological contribution is the use of vocabulary-anchored Procrustes alignment to make cross-model comparisons of the linear flow well-posed. Because different models have different learned embedding matrices, their residual streams formally live in different vector spaces — direct comparison of recovered SVD bases is ill-defined. Vocabulary-anchored alignment, using the shared tokenizer as a Rosetta stone, provides a principled way to transport one model's coordinate system into another's, after which the linear flows can be quantitatively compared.

This is a complete, self-contained pilot study, not a precursor to a frontier-scale project. The pilot's central question is whether the lines-of-thought framework's claims hold under controlled comparison at this scale, and whether they can be exploited. Generalization to frontier scale is a natural follow-up but is not part of the present work.

---

## 1. Background

### 1.1 The lines-of-thought framework

The lines-of-thought framework characterizes trained transformer language models as approximately linear-Gaussian dynamical systems at the population level. For each transformer layer transition $t \to t+\tau$, Sarfati et al. fit a linear map

$$\tilde{x}(t+\tau) = R(t+\tau)\,\Lambda(t,\tau)\,R(t)^\top\,x(t)$$

where $R(t)$ is a per-layer orthonormal basis obtained from the SVD of layer-$t$ activations across the corpus, and $\Lambda(t,\tau) = \text{diag}(\sigma_i(t+\tau)/\sigma_i(t))$ is a diagonal stretch matrix capturing how singular values evolve between layers. The actual layer transition is $x(t+\tau) = \tilde{x}(t+\tau) + w(t,\tau)$, with $w(t,\tau)$ a residual whose ensemble distribution is approximately Gaussian, zero-mean, isotropic, with variance scaling as $\exp(\lambda(t+\tau))$.

Two properties of this framework matter for the present proposal. First, the linear term $R(t+\tau)\,\Lambda(t,\tau)\,R(t)^\top$ is a *population* object — it describes how the cluster of all token trajectories collectively deforms between layers, not what any single trajectory does. The paper is explicit that individual trajectories are deterministic given the prompt (Section 3.3, footnote 7), and that the Gaussian residual is the ensemble distribution of how those deterministic trajectories deviate from the population linear fit. Second, the linear flow arises empirically from SVD of activations — there is no a priori derivation. It is whatever it is for any given trained model.

Mechanically, the linear flow can be understood as the input-distribution average of the per-layer Jacobian. The transformer's layer function $f_t : x_t \mapsto x_{t+1}$ is deterministic but nonlinear; its Jacobian at any specific input is locally well-approximated by $I + L_t(x)$ with $L_t(x)$ a context-dependent low-rank update. The expectation of this Jacobian over the input distribution, $\overline{L_t}$, gives the *average* residual update structure at layer $t$, and the SVD of the resulting linear map at successive layers gives the $R(t)$ and $\Sigma(t)$ that the lines-of-thought analysis recovers. The linear flow is well-defined in the limit of infinite training data as the averaged Jacobian over the asymptotic activation distribution — a population statistic of the architecture-corpus pair.

### 1.2 Which claims are non-trivial, and which follow from CLT-style arguments

Before discussing what Sarfati et al. did and did not establish, it is worth distinguishing which parts of the lines-of-thought framework are genuine empirical content about trained neural networks and which are essentially consequences of central-limit-theorem-style reasoning applied to high-dimensional vector arithmetic. This distinction matters because the universality claims tested in this proposal would be far less interesting if they reduced to "CLT applies in high dimensions" — which holds for any system aggregating many small contributions in a Euclidean vector space.

We partition the framework's claims into two groups.

**Claims with substantial CLT content.** The most CLT-flavored claim is that residual deviations $w(t,\tau)$ from the linear-flow prediction are approximately Gaussian. The residual at any pair of layers is the accumulated effect of many small per-layer perturbations, each of which can be viewed as a sum of contributions from different attention heads, FFN gates, and so on. The central limit theorem predicts that such an accumulation, in high dimensions, will look approximately Gaussian regardless of the underlying microscopic details. Approximate coordinate-wise isotropy of these residuals is similarly expected: in a high-dimensional space with no preferred directions, the residual contributions average out to something roughly isotropic. *Finding Gaussian, isotropic residuals in any sufficiently large trained transformer is therefore not strongly informative on its own* — it is what we should expect for any architecture that aggregates many bounded contributions through depth.

**Claims that are genuine empirical content about trained networks.** Several other framework claims do not follow from CLT and represent specific facts about how trained models organize their computation:

- *Low-dimensional cluster structure.* The paper observes (Figure 1 and Figure 2) that trajectories occupy a roughly 256-dimensional manifold within Llama-2-7B's 4096-dimensional hidden space — about 6% of the ambient dimensionality. A CLT-driven random-contributions process would produce high-dimensional Gaussian noise filling the hidden space roughly uniformly, not concentration on a low-dimensional manifold. The observed low-dimensionality is a property of what the trained model has *learned* to represent; it is not a generic consequence of aggregating noise.

- *Coherent rotation of the principal-direction basis with depth.* The paper's Figure 2(a) shows that the angle between principal directions at successive layers is small and grows smoothly with $|t_1 - t_2|$. A CLT-style random-walk through layer transformations would produce uncorrelated or rapidly-decorrelating principal directions across layers, not smooth coherent rotation. The smooth rotation is a signature that *successive layers act on trajectories in correlated, structured ways* — a trained organization, not an emergent statistical artifact.

- *Approximate linearity of the flow over many layers.* The paper validates linear extrapolation from layer $t$ to layer $t+\tau$ for $\tau$ up to ~10 layers in their 24-layer models, with extrapolated trajectories indistinguishable from true trajectories by a linear classifier (50-60% accuracy, near chance). Random per-layer contributions would produce predictions that degrade rapidly with $\tau$; the observed long-range predictability says that *the per-layer transformations compose into something approximately linear over significant depth*. This is a strong claim about the trained network's structure that does not follow from CLT.

- *Exponential variance scaling.* The framework's claim that residual variance scales as $\sigma^2 \sim \alpha e^{\lambda(t+\tau)}$ is geometrically specific. CLT-style random walk in a vector space would produce *linear* variance growth with depth (Brownian motion). Exponential growth requires that per-layer noise *scales with current state magnitude* — a multiplicative rather than additive noise structure. The observed exponential scaling reflects how the architecture composes contributions; the specific rate $\lambda$ is an empirical quantity that varies across models and is *not* fixed by general statistical considerations.

- *The specific value of the variance scaling rate $\lambda$, and how it varies across architecture and training.* This is, of course, only meaningful as an empirical observation about specific models. It is not predicted by any general theory.

The contributions of the present pilot study lie entirely within the second group. The universality questions we test concern: whether the cluster dimensionality is shared across architectures (does the training-organized low-dimensional structure converge to the same effective dimension regardless of architectural choices?); whether the coherent rotation trajectories agree across architectures (does the "shape" of how the basis evolves with depth match across variants?); whether the variance scaling rate $\lambda$ is universal across architectures (is the per-layer noise growth rate determined by the corpus and task, or by architecture-specific implementation choices?); whether the linear-flow predictability horizon is comparable across architectures (does the extent of approximate linearity in the dynamics generalize?).

These are all empirical questions about the trained organization of the network, not about high-dimensional statistical mechanics. A reviewer skeptical that the lines-of-thought framework is testing anything beyond CLT artifacts should observe that *we do not test or claim universality of the Gaussian residual property*. We accept Gaussianity as a CLT-driven feature that any sufficiently-trained transformer should exhibit. The universality we test concerns the structural-organization properties that *only* arise from training and that are *not* predicted by general statistical considerations.

When reporting Phase 2 results, we explicitly partition our universality findings according to this distinction, so that the substantive claims are clearly identified and CLT-driven properties are reported separately as expected baselines rather than as findings.

### 1.3 What the paper established, and what it did not

Because the present proposal extends and complements the lines-of-thought paper, we want to be precise about which claims are theirs and which are ours to test.

Sarfati et al. established the following:

- *Descriptive claim across five trained models*. The linear-Gaussian decomposition holds for GPT-2, Llama-2-7B, Mistral-7B, Llama-3.2-1B, and Llama-3.2-3B. Trajectories cluster on a low-dimensional manifold; linear extrapolation from layer $t$ to $t+\tau$ produces predictions that cannot be distinguished from true trajectories by a linear classifier (50-60% accuracy, near chance); residuals are Gaussian, zero-mean, isotropic, with exponential variance scaling.
- *Anomalies, qualitatively noted*. Last-layer behavior in Mistral, Llama-3.2-1B, and Llama-3.2-3B deviates from the framework's predictions, possibly due to fine-tuning or alignment. The first few layers also show deviations, conjectured to arise from similar effects.
- *Numerical parameter values for two models*. Reported values are $\alpha \simeq 0.64, \lambda \simeq 0.18$ for GPT-2 and $\alpha \simeq -5.4, \lambda \simeq 0.27$ for Llama-2-7B. Adjusting for the difference between these numbers (the second is likely $\log\alpha$ given that variance must be positive), the parameters reflect both scale and architecture differences. The variance scaling rate $\lambda$ differs by ~50%.
- *Null tests*. (a) Gibberish inputs follow a similar pattern but live on a separable manifold from natural language. (b) An untrained reinitialized model does *not* exhibit the lines-of-thought structure — variance doesn't scale exponentially, distributions are non-Gaussian. So training is what produces the structure.

The paper did NOT establish:

- *Controlled comparison across architectures at fixed scale and corpus*. The five tested models vary in size, training corpus, and architectural details simultaneously. No controlled cross-architecture comparison exists.
- *Characterization of convergence with training compute*. Only finished models are analyzed; no $L^{(K)}$ vs $K$ curves.
- *Quantitative cross-model agreement of the linear flow*. The paper claims qualitative pattern repetition; the parameter values they do report (for GPT-2 and Llama-2-7B) actually *differ* substantially, which is consistent with universality holding at the structural level but not at the quantitative level.
- *Reproducibility across seeds*. No multi-seed analysis.
- *Predictability of the asymptotic linear flow from partial training*. No extrapolation experiments.
- *Interventions during training*. No use of the linear flow as a training target, regularization, or initialization signal.
- *Sensitivity to corpus*. No same-architecture-different-corpus comparison.

### 1.4 The basis-and-vector-space problem for cross-model comparison

A subtle but central issue arises when trying to compare recovered linear flows across models. Each trained model has its own learned embedding matrix $E^{(A)}$, which means *the residual streams of two different models live in formally different vector spaces*. Even if both models have the same hidden dimension $H$, the canonical map between their two $H$-dimensional spaces is not given by anything intrinsic to either model.

This means raw, element-wise comparison of $R^{(A)}(t)$ to $R^{(B)}(t)$ is ill-defined. Each $R^{(M)}(t)$ is an orthonormal matrix on a space determined by model $M$'s embedding choice. They are not coordinatized in commensurate ways.

Two implications follow. First, when the lines-of-thought paper observes that recovered parameters $\alpha, \lambda$ differ across models, some of that difference may reflect the different embedding spaces rather than different computations. Second, any test of "universality" of the linear flow must address how the comparison is made well-posed.

There are two routes. The *basis-invariant route* compares quantities that don't depend on coordinate choice — singular value spectra $\Sigma(t)$, manifold dimensionalities, exponential scaling rates, Gaussianity diagnostics. These are computable per-model and directly comparable. The *aligned route* uses the shared tokenizer to construct a vocabulary-anchored map between the two embedding spaces (via orthogonal Procrustes alignment), transports one model's $R(t)$ into the other's coordinate system, and compares directly. The aligned route is more powerful (it can detect agreement at the level of *what* directions the linear flow uses, not just *how* they're distributed in magnitude), but it depends on the alignment quality.

We use both. The basis-invariant route is robust and provides the headline universality measurements. The aligned route, where it succeeds (where the alignment residual is small), provides a sharper level of comparison.

### 1.5 The pilot scale and its rationale

This proposal scopes everything to 150-million-parameter models. The choice is motivated by three considerations.

First, *feasibility*. At 150M, full Chinchilla-scaled training from scratch (~3B tokens) takes roughly half a day on a single RTX 5090; at half Chinchilla (~1.5B tokens, sufficient for the convergence and universality measurements at this scale) it takes ~6-7 hours, measured directly via a smoke-test calibration. This means we can train tens of models from scratch within the project's timeframe — enough for multi-seed variance characterization, controlled architectural comparison, and intervention experiments with proper paired-run designs. None of these are feasible at 7B+ scale on workstation hardware.

Second, *experimental control*. Training from scratch lets us control the corpus, the architecture, the random seed, and the checkpoint frequency. Publicly-available pretrained models give us none of these. Phase 1's convergence analysis benefits from being able to set dense checkpoints (50-100 per training run) anywhere we want, which no publicly-released model supports.

Third, *scope honesty*. Frontier-scale generalization is a separate question and should be a separate project. Our pilot's findings will be about whether the lines-of-thought picture holds at 150M and what it implies for training at that scale. If results warrant a frontier-scale follow-up, it can be proposed and resourced separately, with the pilot's findings as motivation.

There is one important caveat about the pilot's scope. The Sarfati et al. paper validated its claims at 1B-7B scale. At 150M, the per-layer compute is more limited, the residual stream has fewer directions (~768 vs 4096), and the layer count is lower (~12 vs 32). Whether the linear-Gaussian description is empirically tight at 150M is itself one of the pilot's questions, addressed in Phase 1. If the description fails at this scale, that's a strong negative result for extending the framework downward but doesn't bear on its applicability at the scale Sarfati et al. studied. *Positive findings at 150M would transfer with some confidence to larger scales; negative findings would transfer less cleanly.*

### 1.6 Canonical objects and families of comparison metrics

The framework involves several mathematical objects that the proposal distinguishes carefully, to avoid conflating them in later sections. For each trained model $M$, the recovered linear flow consists of:

- *Per-layer-state quantities*, one per layer $t \in \{0, 1, \ldots, L\}$:
  - $R^{(M)}(t) \in \mathbb{R}^{H \times H}$ — orthonormal basis at layer $t$ (rows are principal directions)
  - $\Sigma^{(M)}(t) \in \mathbb{R}^H_{\geq 0}$ — singular value spectrum at layer $t$
  - $\mu^{(M)}(t) \in \mathbb{R}^H$ — per-layer activation mean
- *Pairwise-transition quantities*, one per $(t, \tau)$ pair with $t + \tau \leq L$:
  - $\Lambda^{(M)}(t, \tau) = \text{diag}(\Sigma^{(M)}(t+\tau) / \Sigma^{(M)}(t))$ — diagonal stretch
  - $A^{(M)}_{t,\tau} = R^{(M)}(t+\tau)\,\Lambda^{(M)}(t,\tau)\,R^{(M)}(t)^\top$ — full linear transition operator
- *Scalar summary parameters*:
  - $\lambda^{(M)}$ — variance scaling rate (slope of $\log \text{var}(w)$ vs $t+\tau$)
  - $\log\alpha^{(M)}$ — variance prefactor (intercept of the same fit)

Comparison metrics fall into four families, each defined on different objects and each requiring different prerequisites:

1. *Spectral metrics* — distances between $\Sigma^{(M)}(t)$ spectra. Basis-invariant. No alignment needed. Examples: log-spectrum L2 distance, effective rank distance.
2. *Scalar metrics* — differences in $\lambda^{(M)}$, $\log\alpha^{(M)}$, mean kurtosis, mean isotropy. Basis-invariant. No alignment needed.
3. *Subspace metrics* — distances between $R^{(M)}(t)$ bases. Require alignment to be well-posed (see §1.4). After alignment, use Frobenius distance or principal angles.
4. *Transition operator metrics* — distances between $A^{(M)}_{t,\tau}$ operators. Require *two-endpoint* alignment ($Q_t$ and $Q_{t+\tau}$). After alignment, use operator norm or Frobenius distance.

Families 1 and 2 are basis-invariant and form the headline universality measurements (per §1.4). Families 3 and 4 are basis-specific and require alignment; they provide sharper conditional measurements. When the proposal references "comparing linear flows," the specific metric family is named.

---

## 2. Hypotheses and goals

The proposed work has four nested hypotheses, with go/no-go decision points between phases so resources are committed to later phases only if earlier ones have made the case.

**Hypothesis 1 (Convergence).** For a fixed architecture and corpus trained at 150M, the recovered linear flow $L^{(K)} = (R^{(K)}(t), \Lambda^{(K)}(t,\tau), \sigma_t^{2,(K)})$ as a function of training compute $K$ converges — under both basis-invariant metrics and (where applicable) seed-replica alignment — to a limit $L^{(\infty)}$, and the rate of convergence is characterizable. The convergence rate may be faster or slower than loss-curve convergence; both possibilities are informative.

**Hypothesis 2 (Universality).** Across the four chosen architectural variants — Llama-style, Gemma-style, Qwen-style, DeepSeek-style with MLA — trained on the same corpus at 150M to similar quality, the asymptotic linear flows are universal in the following senses, in decreasing strength:

- *2a (Structural)*: The functional form (Gaussian residuals, exponential variance scaling, smooth singular-vector rotation) holds across all four variants. *Note*: per §1.2, the Gaussian residual part of this claim is essentially CLT-driven and is expected to hold for any sufficiently-trained transformer regardless of architecture. The substantive components of 2a are the *exponential* variance scaling (not linear, as a random walk would give) and the *smooth, structured* singular-vector rotation (not chaotic decorrelation across layers). We report 2a's CLT-baseline and structural components separately.
- *2b (Basis-invariant)*: The substantive basis-invariant statistics — singular value spectra $\Sigma(t)$, manifold dimensionality (effective rank, not just ambient dimension), variance scaling rate $\lambda$, and the depth profile of these quantities — converge across variants to a shared profile up to small fluctuation. These are *trained-organization* properties not predicted by general statistical considerations; this is the substantive level of the universality claim.
- *2c (Aligned)*: After vocabulary-anchored Procrustes alignment of the embedding spaces and corresponding per-layer activation alignments, the transported linear flows agree quantitatively in the specific principal directions used by each layer.

The Sarfati et al. paper established *2a* across five uncontrolled models (qualitative form, not quantitative). *2b* and *2c* are the meaningful empirical strengthenings to test, and *2b* is the level at which the universality claim has its primary substantive content.

**Hypothesis 2' (Corpus universality, secondary).** For a fixed architecture (Llama-style), trained at 150M on several broad natural-language corpora (FineWeb-Edu, RedPajama, The Pile), the asymptotic linear flow is universal in senses 2a-2c. Tests corpus-driven vs architecture-driven variation; the corpus comparison is secondary because the corpora overlap substantially in their bulk statistics, so the strongest variation we can probe is across these specific dataset choices, not across genuinely different "kinds of language."

**Hypothesis 3 (Predictability).** $L^{(\infty)}$ can be predicted from partial training (say, the first 30-50% of training compute) to substantially higher accuracy than the latest-checkpoint baseline. Prediction is performed in basis-invariant metrics primarily; aligned-coordinate prediction is a secondary target.

**Hypothesis 4 (Exploitation).** A predicted $L^{(\infty)}$ can be applied during training — as a diagnostic monitor, regularization target, or spectral initialization signal — to produce a measurable improvement in training efficiency or final model quality.

The strongest result the pilot could produce is positive findings on all four hypotheses. A natural and substantive result is positive on H1, H2b (basis-invariant universality), and H3, with H2c and H4 as further information. A clean negative result on H2 — even on H2b — would be substantively important: it would mean the lines-of-thought framework's universality is not just "qualitative repetition" but does *not* extend to basis-invariant convergence, which would sharpen what the framework actually establishes.

### 2.1 Quantitative success thresholds

To make hypothesis evaluation operational rather than purely interpretive, the proposal commits to provisional success thresholds upfront. These numbers are *provisional* — the noise scale of within-seed variance is unknown until Phase 1 measurements arrive, and the thresholds will likely be refined once we have that data. But committing to specific numbers now prevents post-hoc threshold-fitting and lets a reader of the eventual report immediately see which findings cleared the bar.

**Hypothesis 1 (Convergence) — success criterion.** The linear flow $L^{(K)}$ has *converged* if, for the basis-invariant Σ-spectrum distance metric of §1.6 family 1, the last 25% of training shows distance-to-final values whose standard deviation is less than 10% of the total reduction observed across the full training trajectory. Concretely: if the flow distance starts at 100 (arbitrary units) at step 0 and ends near 0 at the final step, the last 25% of checkpoints should fluctuate within a band of ≤10 units.

**Hypothesis 2b (Basis-invariant universality) — success criterion.** The four architectures show basis-invariant universality if, for each basis-invariant metric (Σ-spectrum distance, λ difference, effective rank profile distance), the *across-variant* mean pairwise distance is at most **1.5×** the *within-variant* seed dispersion. The factor 1.5 is the provisional threshold. With three seeds per variant we have a measurable within-variant noise floor; differences much smaller than this floor are statistically indistinguishable from seed variation, while differences much larger indicate genuine architectural divergence. The 1.5× factor allows modest across-variant variation while distinguishing it from noise.

**Hypothesis 2c (Aligned-coordinate universality) — success criterion.** Aligned-coordinate universality holds if (a) the cross-architecture vocabulary-anchored alignment residual ratio $\rho_t$ is below **0.15** for the layer states where comparison is being made (the alignment itself is meaningful), AND (b) the post-alignment Frobenius distance between transported $R(t)$ matrices is at most **1.5×** the within-variant seed dispersion of the same quantity. Both conditions are required because (a) alone doesn't show universality, just consistent vocabulary geometry; (b) alone doesn't show the model dynamics agree if the underlying spaces don't.

**Hypothesis 3 (Predictability) — success criterion.** A predictor of $L^{(\infty)}$ trained on the first 50% of checkpoints succeeds if it reduces the basis-invariant distance to the actual final flow by at least **30%** relative to the latest-checkpoint baseline (which predicts $L^{(\infty)} = L^{(K_n)}$ for the latest available $K_n$). The 30% threshold makes the predictor non-trivially useful while being achievable if the trajectory has any extrapolable structure.

**Hypothesis 4 (Exploitation) — success criterion.** A successful intervention (A or B) must produce a measurable improvement in matched paired training runs: either a reduction in final eval loss at fixed compute of at least **2%** (about 0.06 loss units at our late-training loss of ~3.0), or a reduction in compute needed to reach a fixed loss threshold of at least **10%**. Both effects are small but measurable above seed-to-seed noise (which the multi-seed paired design characterizes). Improvements below these thresholds may exist but would not justify the framework's complexity as a training tool.

These thresholds will be tightened or relaxed once Phase 1 noise floors are characterized. The thresholds are also asymmetric in an important way: clearing them is a positive result, but failing to clear them is *not* automatically a negative result. A negative result requires showing across-variant distance is *much larger* than within-variant (say, >3×), not just larger than 1.5×. The intermediate zone is "inconclusive at the pilot's statistical power."

---

## 3. The four architectural variants

The pilot trains models at 150M parameters across four architectural variants, chosen to span the meaningful axes of modern transformer design while remaining within the post-2023 consensus core. All four use the same tokenizer, the same vocabulary, the same hidden dimension (896), the same depth (12 layers), and the same training corpus, hyperparameters, and compute budget. They differ in specifically named architectural choices that map to distinct design axes.

**Variant A — Llama-style (reference).** Pre-RMSNorm only; RoPE positional encoding; SwiGLU FFN; full multi-head causal attention (14 heads × 64 head-dim = 896 hidden). Matches the Llama-2 / Mistral / Llama-3 broad template at small scale. Serves as the reference variant. We do not use grouped-query attention (GQA) for the reference at this scale — at 14 attention heads, GQA would require introducing yet another design choice (the K/V head count) that's orthogonal to the framework questions we're testing.

**Variant B — Gemma-style.** Hybrid pre+post RMSNorm; RoPE; GeGLU FFN; alternating sliding-window and full attention; attention logit softcap (50) and final logit softcap (30). Tests the *normalization placement* axis (hybrid vs pre-only), the *activation function* axis (GeGLU vs SwiGLU), and the *attention-score bounding* axis (softcap vs no softcap). The sliding-window pattern (window 4096) is faithfully implemented but functionally inert at our seq_len=1024 — see §10.2.

**Variant C — Qwen-style.** Pre-RMSNorm; QK-Norm in the attention mechanism (RMSNorm applied per-head to Q and K before RoPE); RoPE; SwiGLU FFN; full multi-head attention. Tests the *attention-input stabilization* axis (QK-Norm) in isolation. This is the smallest deviation from the Llama baseline; including it tests whether small stabilization tricks affect the recovered linear flow.

**Variant D — DeepSeek-style with simplified MLA.** Pre-RMSNorm; RoPE; SwiGLU FFN; Multi-head Latent Attention (MLA) instead of full multi-head attention. The K and V tensors are constructed via two-stage projection from a small latent (dim 96), making them rank-constrained to live in a 96-dimensional subspace of the 896-dimensional hidden space. The Q tensor is similarly constructed via a wider latent (dim 192). Tests the *attention rank-constraint* axis (low-rank K/V/Q vs unconstrained). We use a simplified MLA implementation without the decoupled-RoPE inference-cache machinery — see §10.2.

These four variants span four design axes (normalization placement, activation function, attention-score bounding, attention rank structure). They are recognizable to reviewers and reflect actual choices made in widely-deployed foundation models. The differences are large enough to be informative if they produce different linear flows, and consistent enough with each other that all four can be expected to train reliably at 150M scale.

A fifth "negative control" variant (an older pre-consensus architecture, e.g., GPT-2 style with LayerNorm, learned positional embeddings, GeLU MLP) was considered and rejected. At 150M, pre-consensus architectures may not train reliably to the same quality, which would confound the comparison — we'd be measuring training quality differences rather than architectural differences. The four-variant comparison remains internally consistent without this fifth point.

---

## 4. Phase 0: Pipeline validation (completed)

Before launching the multi-seed Phase 1 training campaign, the analysis pipeline must be validated end-to-end so that bugs do not corrupt the Phase 1 results. Phase 0 consists of three validation activities, all of which have been completed before the main campaign begins.

**(a) Unit tests on synthetic data with known ground truth.** The linear-flow analyzer is tested against synthetic activations generated by a known linear-Gaussian process. The synthetic data uses smoothly-rotating per-layer bases (small Cayley rotations between successive layers, matching the smoothness empirically observed in trained models per Sarfati et al. Fig 2(a)), known singular value spectra, and isotropic Gaussian noise with controlled variance scaling. The analyzer must recover the spectra (to 1%), the variance scaling rate $\lambda$ (in sign and within factor-2 of the per-step injected $\lambda$, the difference reflecting the envelope-vs-per-step distinction inherent in accumulated noise), excess kurtosis near zero, and isotropy near zero. All checks pass.

The synthetic-data testing surfaced two bugs that would otherwise have produced silent corruption of real results. *First:* naively applying the paper's element-wise scaling formula $\tilde{x}(t+\tau) = R(t+\tau)\Lambda(t,\tau)R(t)^\top x(t)$ to recover the linear flow gives residuals that depend strongly on SVD sign convention. The signs of recovered singular vectors are arbitrary, and predicting via element-wise scaling implicitly assumes axis-i at layer $t$ aligns with axis-i at layer $t+\tau$, which the SVD doesn't enforce. The fix is to compute the linear-flow prediction via ordinary least-squares regression $X_{\text{target}} \approx X_t A$ for each $(t, \text{target})$ pair, which is mathematically equivalent to the paper's formula under matched sign conventions but is sign-robust under any. *Second:* naively computing Gaussianity diagnostics by concatenating residuals from multiple source layers produces a mixture-of-Gaussians artifact (residuals from different sources have different variances; their union has spurious excess kurtosis). The fix is to compute kurtosis per (source, target) pair and average. Both fixes are in the production analyzer.

**(b) End-to-end smoke test with real training.** A 200-step training run on the production architecture (146.4M-parameter Llama-style model, FineWeb-Edu corpus, RTX 5090 with bfloat16 autocast) produced 50 checkpoints across the training trajectory. The smoke test verified that: model and training infrastructure work end-to-end; bfloat16 mixed precision does not produce NaNs or instability; gradient checkpointing reduces memory without affecting forward outputs (verified by unit test); per-step throughput is ~65k tokens/sec steady-state; VRAM peaks at well under 32 GB headroom; checkpoint format is correct; and the linear-flow analyzer can be run against the saved checkpoints to recover all framework quantities without errors. The smoke test's 200-step trained model is far from convergence — it serves as a software pipeline validation, not a scientific result.

**(c) Independent calibration of analyzer values across smoke-test checkpoints.** The recovered $\lambda$, $\log\alpha$, effective rank, kurtosis, and isotropy across the 50 smoke-test checkpoints are inspected for sanity (no NaNs, no infinite values, monotonic-ish trends as the model trains) and for self-consistency (the basis-invariant statistics that should be set early in training, such as embedding-layer effective rank, settle quickly; deeper-layer statistics continue evolving). This provides a small but real test of the analyzer's stability on real model outputs.

Phase 0 has identified no surviving showstopper bugs. The pipeline is ready for the multi-seed Phase 1 campaign.

---

## 5. Phase 1: Convergence measurement

### 5.1 Goal

Determine whether the linear flow converges with training compute at 150M, characterize the rate of convergence, and compare it to loss-curve convergence. Establish reproducibility across seeds.

### 5.2 Experimental design

Train multiple instances of the Llama-style baseline (Variant A) from scratch, Chinchilla-scaled, on a fixed broad corpus (FineWeb-Edu). Specifically:

- 4-6 independent runs with different random seeds.
- Same architecture, same corpus, same hyperparameters, same total compute.
- Dense checkpoint saving: 50-100 checkpoints distributed across training (logarithmically spaced for the early phase, then linearly for the late phase).
- Reserve a held-out subset of the corpus for evaluation throughout training.

For each checkpoint, compute the linear flow: run inference on a fixed evaluation corpus (~10M tokens), accumulate per-layer activations, perform per-layer SVD to recover $R^{(K)}(t)$ and $\Sigma^{(K)}(t)$, derive $\Lambda^{(K)}(t,\tau)$, fit the residual variance model. Store the per-checkpoint $L^{(K)}$ for offline analysis.

### 5.3 Metrics

Convergence is measured via three complementary distance metrics on $L^{(K)}$:

1. *Basis-invariant distances*: changes in singular value spectra $\|\log\Sigma^{(K)}(t) - \log\Sigma^{(K_\text{final})}(t)\|$, changes in scaling parameters $|\lambda^{(K)} - \lambda^{(K_\text{final})}|$, changes in manifold dimensionality (effective rank of activation covariance), changes in Gaussianity diagnostics. These can be computed per-checkpoint without alignment.

2. *Self-alignment distances*: for the same model trained with multiple seeds, the recovered $R^{(K)}(t)$ matrices live in *formally different* spaces (different learned embeddings produced by different seeds). To compare them, perform vocabulary-anchored Procrustes alignment (§5.4) and compute aligned-coordinate distance. This is a within-architecture test of whether seed replicas converge to "the same place" up to coordinate choice.

3. *Loss convergence*: held-out loss curves on the evaluation corpus, plotted alongside the linear-flow distance curves.

### 5.4 Vocabulary-anchored alignment procedure

Even seed replicas of the same architecture have different learned embedding matrices and therefore live in different spaces. The alignment procedure that makes their $R(t)$ matrices comparable is the same one used in Phase 2 for cross-architecture comparisons, so we develop it here:

Given two models A and B with the same tokenizer (so vocabulary is exactly shared) and possibly different hidden dimensions $H_A, H_B$:

**Step 1: Embedding-space alignment.** Solve the orthogonal Procrustes problem

$$Q_E = \arg\min_{Q : Q^\top Q = I} \|E^{(A)} Q - E^{(B)}\|_F$$

If $H_A = H_B$, $Q_E$ is an $H \times H$ orthogonal matrix. If $H_A \neq H_B$, use the rectangular Procrustes solution (top singular vectors of $E^{(A)\top} E^{(B)}$). Report the alignment residual ratio $\rho_E = \|E^{(A)} Q_E - E^{(B)}\|_F / \|E^{(B)}\|_F$ — this is itself a measurement of how well the two embedding spaces correspond at the vocabulary level.

**Step 2: Per-layer activation alignment.** Run both models on a shared input corpus (~1M tokens, same input set for both). Accumulate per-layer activations $X_t^{(A)}$ and $X_t^{(B)}$, where each row is the model's hidden state at one (input, position) pair. The shared input set makes corresponding rows refer to the same (input, position) — the activation-level Rosetta stone.

For each layer $t$, solve

$$Q_t = \arg\min_{Q : Q^\top Q = I} \|X_t^{(A)} Q - X_t^{(B)}\|_F$$

with alignment residual $\rho_t = \|X_t^{(A)} Q_t - X_t^{(B)}\|_F / \|X_t^{(B)}\|_F$.

**Step 3: Aligned linear flow comparison.** Two distinct objects can be transported and compared, and the proposal distinguishes them carefully because they have different alignment structures.

*The per-layer basis $R^{(M)}(t)$.* This is a within-layer object — an orthonormal basis on layer $t$'s hidden space. To express it in another model's coordinate system, conjugate by the layer-$t$ alignment only:

$$\tilde{R}^{(A \to B)}(t) = Q_t^\top R^{(A)}(t) Q_t$$

After transport, $\tilde{R}^{(A \to B)}(t)$ and $R^{(B)}(t)$ both live in B's coordinate system at layer $t$, and Frobenius distance $\|\tilde{R}^{(A \to B)}(t) - R^{(B)}(t)\|_F$ is a meaningful comparison. We typically restrict to the top-$k$ principal directions (rows of $R$) to focus on the dominant flow structure.

*The cross-layer transition operator $A^{(M)}_{t,\tau} = R^{(M)}(t+\tau)\,\Lambda^{(M)}(t,\tau)\,R^{(M)}(t)^\top$.* This is the actual linear map sending layer-$t$ activations to predicted layer-$(t+\tau)$ activations. It spans two layers, so its transport involves the alignments at both endpoints:

$$\tilde{A}^{(A \to B)}_{t,\tau} = Q_{t+\tau}^\top A^{(A)}_{t,\tau} Q_t$$

After transport, $\tilde{A}^{(A \to B)}_{t,\tau}$ and $A^{(B)}_{t,\tau}$ both act on B's coordinate system from layer $t$ to layer $t+\tau$, and operator-norm or Frobenius comparisons are well-posed.

Distinguishing these two transports matters because they answer different questions: $R(t)$ comparison asks "do the two models use similar principal directions at depth $t$?"; $A_{t,\tau}$ comparison asks "do the two models implement similar transitions from $t$ to $t+\tau$?" Both are part of the universality picture and both are reported.

**Reporting both alignment quality and post-alignment agreement.** The alignment residuals $\rho_E$ and $\{\rho_t\}_t$ tell us how well the spaces correspond at all; the post-alignment distance $d_R$ tells us whether, given the alignment, the linear flows agree. Both are necessary to report — a small $d_R$ is only meaningful if the corresponding $\rho$ is also small.

### 5.5 What the data would look like under different outcomes

If Hypothesis 1 (convergence) holds, plotting $d(L^{(K)}, L^{(K_\text{final})})$ against $K$ for each seed shows monotonic decay. Plotting the across-seed dispersion shows how reproducible the trajectory is. We expect the basis-invariant distances to be the cleanest signal; the aligned-coordinate distances will be noisier (alignment quality varies) but should also decrease.

A particularly informative comparison: plot $d(L^{(K)}, L^{(K_\text{final})})$ alongside loss curves. If the linear flow stabilizes substantially *earlier* than loss does, that is a strong empirical finding on its own — it would mean the structural content of the model is determined early in training, even though refinement of specific outputs continues much longer. The opposite finding (linear flow continues to drift after loss has plateaued) would suggest the linear flow is not capturing the model's settled structure and would qualify the convergence hypothesis.

### 5.6 Go/no-go criterion for Phase 2

Phase 2 is justified if Phase 1 shows that (a) the linear flow converges in basis-invariant metrics monotonically and reproducibly across seeds, and (b) the alignment procedure produces meaningful alignment residuals (small enough that aligned-coordinate comparisons are interpretable). If either fails, Phase 2's premise — that there is a well-defined and alignable asymptotic flow to compare across architectures — is undermined and the project should pivot.

### 5.7 Resources

A smoke-test training run (200 steps, otherwise identical to the production config) on our RTX 5090 measured **steady-state throughput of ~65,000 tokens/sec** at micro-batch 8, sequence length 1024, gradient accumulation 8. At this throughput, a full 24,000-step run (~1.57B tokens) completes in **~6-7 hours** of wall-clock. Four to six independent seed runs for Phase 1 therefore require **~2-3 days of training compute** rather than the 1-2 weeks our earlier estimate suggested. Total Phase 1 elapsed time, including analysis, is **3-5 weeks** of which 2-3 days is automated training and the rest is analysis and reporting.

*Analysis pipeline design.* The per-checkpoint analysis does not store activations to disk. For each checkpoint, ~10k pilot-token activations are accumulated in RAM (~500 MB across 14 layer states), per-layer SVDs are computed in fp32 (~20 seconds CPU time per checkpoint), and only the summary $\{R(t), \Sigma(t), \mu(t)\}$ plus scalar diagnostics are persisted to disk (~30 MB per checkpoint via `.npz`). Total disk footprint for a Phase 1 seed-run's analysis output is ~1.5 GB. The pipeline is effectively streaming — no persistent activation storage required, and SVD-at-scale concerns (randomized SVD, streaming covariance) do not arise at the present scale.

---

## 6. Phase 2: Universality

### 6.1 Goal

Test whether the asymptotic linear flow is universal across the four architectural variants (architectural universality) and across multiple broad corpora (corpus universality). The architectural test is the primary; corpus is secondary.

### 6.2 Approach: architectural universality (primary)

Train all four variants on the same corpus (FineWeb-Edu) at 150M, with identical training hyperparameters and the same total token budget. Run multiple seeds per variant — ideally 3 per variant, for 12 total runs — to characterize within-variant variance and distinguish it from across-variant variance.

*Operational criterion for "matching" the four variants.* "Matched final quality" is ambiguous because architectural choices interact with hyperparameters in ways that can produce different final losses at the same compute budget. We specify the matching criterion concretely:

- *Primary comparison: matched compute.* All four variants are trained on the same fixed token budget (~1.5B tokens) with the same per-token compute spent (same micro-batch, sequence length, optimizer steps, learning-rate schedule). This is the most controlled comparison and is the headline result. We report each variant's achieved final eval loss alongside the flow comparisons so the reader can judge whether the matched-compute results are also matched-quality.
- *Secondary comparison: matched loss.* For each variant, identify the checkpoint where eval loss crosses a shared target (e.g., the lowest eval loss achieved by the *worst-performing* variant at the matched-compute endpoint). Compute the flow at these checkpoints and compare across variants. This factors out training-quality differences from architectural differences and answers a different question: "given equally good models, do the four architectures organize their dynamics the same way?"

If matched-compute and matched-loss comparisons give similar conclusions, the universality finding is robust. If they diverge, that itself is informative — it tells us that some apparent architectural difference in the linear flow is actually a training-quality artifact.

For each trained model, recover $L = (R(t), \Lambda(t,\tau), \alpha, \lambda)$ and store. For each pair of variants $(M_1, M_2)$, perform the alignment procedure of §5.4 and compute (a) alignment residuals, (b) basis-invariant distances, (c) aligned-coordinate distances. The comparison is run at both the matched-compute and matched-loss checkpoints per the criterion above.

### 6.3 What "universality" looks like across the three levels

Following the hypothesis structure, with §1.2's CLT-vs-substantive distinction applied throughout:

**Structural universality (2a), reported as two separate findings.** First, the *CLT-baseline component*: Gaussianity tests on residuals across variants. We expect all four variants to pass this test; failure would indicate a deeper problem (e.g., the framework's first-order linear-Gaussian description doesn't apply at this scale or for this architecture). We report this as a sanity check rather than as a universality finding per se. Second, the *substantive structural component*: does the *exponential* (not linear) variance scaling hold across variants, and does the singular-vector basis rotate *smoothly* (not chaotically) with depth? These are the parts of 2a that reflect trained organization rather than CLT artifacts, and they are the parts that would meaningfully fail if some variant organized its computation very differently.

**Basis-invariant universality (2b) — the primary substantive test.** Compare $\Sigma(t)$ singular value spectra (depth profiles, not just summary statistics), $\lambda$ scaling rates, effective rank trajectories, and the principal-angle profile of how $R(t)$ rotates with depth, across the four variants. Within-variant seed dispersion is the noise floor; across-variant differences must exceed this to be meaningful. These are the trained-organization properties that the CLT does not predict. If they converge across variants, universality holds at the substantive level. If they differ, the framework's pattern is real but its specific quantitative content is architecture-contingent.

Plot, for each metric, the distribution of values across (variant, seed) pairs. If across-variant differences are comparable to or smaller than within-variant differences, basis-invariant universality holds. If across-variant differences are substantially larger, it does not.

**Aligned-coordinate universality (2c).** Apply the alignment procedure pairwise across variants. Two reportings: (1) the alignment quality itself — how well does the vocabulary-anchored Procrustes map relate the embedding spaces? If alignment residuals are large, the variants disagree about basic vocabulary geometry, which is itself a measurement. (2) Given the alignment, how close are the transported $R(t)$ matrices? Small post-alignment distance means aligned-coordinate universality holds.

### 6.4 Approach: corpus universality (secondary)

For a fixed architecture (Llama-style), train at 150M on three different corpora — FineWeb-Edu (broad web), The Pile (broad with significant code/academic), and RedPajama (broad with different mixing ratios). The corpora overlap substantially in their bulk content but differ in composition.

This tests how much of the linear flow is determined by architecture-architecture vs corpus-corpus variation. If corpus differences produce flow differences comparable to architecture differences, the framework is roughly equally sensitive to both. If corpus differences are much smaller than architecture differences, the framework is more architecture-sensitive. If corpus differences are much larger, more corpus-sensitive.

We do not include genuinely narrow corpora (code-only, single-language) because the resulting flow could be radically different in ways that don't reflect the central question (is the modern foundation-model corpus regime convergent?). The contrast across these three broad corpora is the most policy-relevant contrast.

### 6.5 What the outcomes mean

Several outcomes are possible and each is informative:

*Outcome 1: Strong universality.* All three levels (2a, 2b, 2c) hold across all four variants. Linear flows converge to essentially the same object up to coordinate choice. This would be a substantive positive finding — the lines-of-thought framework's universality is real and quantitative, not just qualitative.

*Outcome 2: Universality at structural and basis-invariant levels, but not at aligned-coordinate level.* Linear flows agree about magnitude profiles and scaling rates, but the specific directions differ even after alignment. This would mean the variants reach the same "abstract" computation but instantiate it in genuinely different ways. The intuition that "good models do the same thing" is rescued at the abstract level but fails at the concrete level.

*Outcome 3: Structural universality only.* The form of the decomposition holds, but the quantitative basis-invariant statistics differ across variants. This would mean the lines-of-thought framework is descriptively useful but not predictive across architectures — each architecture has its own asymptotic flow, and there's no shared limit.

*Outcome 4: Structural universality fails.* The decomposition doesn't hold for some variants — the residuals aren't Gaussian, the variance doesn't scale exponentially, or some other prediction of the framework breaks. This would be a substantive negative result for extending the framework downward in scale.

### 6.6 Go/no-go criterion for Phase 3

Phase 3 (predictability) is justified if Phase 2 establishes that there is a stable, characterizable asymptotic flow per (variant, corpus) pair, with within-variant variance much smaller than across-variant variance. Otherwise, the prediction question doesn't have a well-defined target.

### 6.7 Resources

12 architectural runs + 6 corpus runs = 18 total training runs at ~7 hours each = ~5-6 days of training. Plus all per-model inference, alignment computation, and analysis. Total Phase 2 elapsed time: **8-12 weeks**, of which less than a week is automated training and the rest is analysis. Phase 2 is the longest single phase of the project because the cross-architecture comparison and alignment work is analytically rich, not because training is slow.

---

## 7. Phase 3: Predictability

### 7.1 Goal

Given the partial-training linear flow recovered at checkpoints from the first 30-50% of training, predict the asymptotic $L^{(\infty)}$ that would be recovered if training were continued. Compare against the "latest checkpoint" baseline.

### 7.2 Approach

Use the data already collected in Phases 1 and 2: for each of the trained models, we have a checkpoint series $\{L^{(K_i)}\}_i$. The prediction problem is, given $\{L^{(K_1)}, ..., L^{(K_n)}\}$, predict $L^{(K_\text{final})}$.

Three approaches, in order of increasing assumption strength:

**Direct extrapolation.** Fit a smooth functional form (e.g., $L(K) = L^{(\infty)} + A K^{-\beta}$) to the per-parameter values and read off the asymptote. Most natural for scalar parameters like $\alpha, \lambda$ and for the basis-invariant statistics. For matrix-valued parameters like $R(t)$, extrapolate per-singular-vector or per-element and reassemble.

**Cross-model prediction.** With multiple fully-trained models in the same architecture-corpus class (the seed replicas from Phase 1 and the multi-seed variants from Phase 2), use these $L^{(K_\text{final})}$ values to learn the "manifold of fully-trained linear flows." Project partial-training observations onto this manifold.

**Population-fit prediction (conditional on Hypothesis 2 holding).** If basis-invariant universality holds at Phase 2, then $L^{(\infty)}$ for any specific variant at 150M is approximately the same as the variant-averaged $L^{(\infty)}$. Use this as a prior, refined by the specific variant's partial-training trajectory.

### 7.3 Evaluation

For each model in our dataset, hold out the final 50% of its checkpoints. Train the predictor on the first 50% and predict the linear flow at the final checkpoint. Compare to the actual final flow using the basis-invariant and aligned distance metrics from Phase 1.

Baseline to beat: predict $L^{(\infty)} = L^{(K_n)}$, i.e., assume the linear flow at the latest partial checkpoint is already converged. The predictor must substantially beat this baseline to be useful.

### 7.4 Go/no-go criterion for Phase 4

Phase 4 (intervention) requires that the predictor be useful enough that "applying the prediction during training" can plausibly help. If the predictor's accuracy is similar to the baseline, intervention is unlikely to help and Phase 4 should be scoped to its diagnostic-only version.

### 7.5 Resources

Phase 3 is primarily analysis on data already collected. **2-4 weeks** of elapsed analyst time, minimal additional compute.

---

## 8. Phase 4: Intervention

### 8.1 Goal

If Phases 1-3 establish that $L^{(\infty)}$ is predictable, test whether using the prediction during training improves outcomes.

### 8.2 Two primary interventions, plus an exploratory third deferred to future work

**Intervention A: Diagnostic monitoring.** During training, periodically measure the current linear flow and compute its distance from predicted $L^{(\infty)}$. Use this as a training-progress indicator orthogonal to loss curves. The hypothesis: diverging distance trajectories indicate pathological training configurations; normal training shows monotonic decrease in distance.

This is the cheapest and lowest-risk intervention. It can be added to any training run without affecting training dynamics. Implementation cost: ~2-3 weeks. Even if it doesn't reveal anything actionable, it provides information about training that doesn't otherwise exist.

**Intervention B: Linear-flow regularization.** Add to the training loss a term penalizing the distance between the current linear flow and predicted $L^{(\infty)}$. The hypothesis: this guides training toward configurations consistent with the asymptote, potentially avoiding pathological transients.

The regularization is applied at the level of basis-invariant statistics (singular value spectra and scaling rates), not at the level of $R(t)$ matrices directly, because the latter would require maintaining alignment between current and predicted coordinate systems during training, which is expensive. The basis-invariant version is much cheaper and directly testable.

Risk: too much regularization prevents the model from doing what it needs to do; too little has no effect. Strength is a hyperparameter to tune.

**Intervention C (exploratory, deferred to future work): Spectral initialization.** In principle, the predicted $L^{(\infty)}$ could be used to initialize the model so its initial per-layer averaged Jacobians match the predicted asymptotic structure — conceptually, "start from the converged spectrum." We note this intervention but do not include it in Phase 4 of the pilot. The inverse problem of mapping from a target averaged Jacobian to actual weight matrices is non-trivial at the architectural complexity of modern transformers (RMSNorm scaling, RoPE-modulated attention, SwiGLU non-linearities all enter the per-layer Jacobian in non-separable ways), and a principled implementation requires methodology development we don't want to commit to within the pilot's scope. If Interventions A and B produce positive results, spectral initialization becomes a natural and well-motivated follow-up project; if not, the underlying premise (that the population flow is a useful training target) is in question and Intervention C wouldn't help either.

### 8.3 Evaluation

For each of Interventions A and B, run matched paired training runs at 150M — one with intervention, one without — at fixed compute budgets. Multiple seeds per condition (3-5) to characterize variance.

Compare: final loss at fixed training compute; training compute required to reach a fixed loss threshold; perplexity on held-out corpora; performance on standard downstream tasks (commonsense reasoning, code completion, etc.).

A consistent improvement of the intervention version, beyond the variance characterized by seed replicas, is the success criterion.

### 8.4 Resources

Intervention A: ~2-3 weeks integration; near-zero ongoing compute.

Intervention B: 6-10 training runs (paired comparison with multiple seeds) at ~7 hr per run = 2-3 days of training, plus analysis. **Total Phase 4 elapsed time: 6-10 weeks** (the per-run training time is now small; analysis and iteration on regularization hyperparameters dominate).

---

## 9. Expected outcomes

### 9.1 Positive outcomes

If all hypotheses hold, the pilot produces:

- Empirical characterization of how the linear flow converges with training compute at 150M.
- Quantitative test of universality across four modern architectures, using vocabulary-anchored alignment for principled cross-model comparison.
- A predictor for the asymptotic linear flow from partial training, with calibrated accuracy.
- Training-efficiency improvement(s) from one or more of the interventions.

Each is a substantive empirical finding on its own.

### 9.2 Informative mixed outcomes

The hypotheses are independent enough that partial confirmations are still informative:

- *Convergence holds; universality fails.* The framework characterizes individual models well but doesn't generalize across architectures. Useful for diagnostic purposes within a single model family.
- *Universality at basis-invariant level only.* Linear flows agree about scale and structure but not about specific directions. Resolves the "two good models should do the same thing" intuition by saying they do — at the abstract level.
- *Universality holds; prediction doesn't work.* Asymptotic flows exist and agree, but partial-training trajectories don't carry enough information to forecast them. Limits the practical use of the framework but is interesting empirically.
- *Prediction works; intervention doesn't.* The linear flow is a *consequence* of training rather than something that *causes* training to proceed; steering it doesn't accelerate training.

### 9.3 Negative outcomes

Several negative outcomes are themselves contributions to the literature:

- *Linear flow doesn't converge.* The flow keeps drifting throughout training. This contradicts the framework's implicit framing of "settled" trained models having a fixed flow.
- *Structural universality fails at 150M.* The lines-of-thought decomposition isn't tight at this scale. This bounds the framework's applicability to scales above 150M.
- *Universality fails even at basis-invariant level.* Different architectures find genuinely different asymptotic flows. This sharpens what "universality" can mean.

The project structure ensures these are detected at the earliest phase where they manifest, before significant resources are committed to dependent phases.

---

## 10. Risks

### 10.1 Conceptual risks

**The framework's claims may reduce to CLT artifacts under careful inspection.** A reviewer might argue that the lines-of-thought decomposition holds in trained transformers simply because *any* system aggregating many small contributions in a high-dimensional Euclidean space will display approximately Gaussian deviations and approximately low-dimensional structure — that is, the framework measures a generic property of high-dimensional vector arithmetic rather than a specific property of trained networks. Section 1.2 addresses this directly: we partition the framework's claims into CLT-baseline components (Gaussianity, approximate isotropy of residuals) and substantive components (the *specific* exponential scaling rate, low manifold dimensionality, coherent rotation of principal directions with depth, long-range linear predictability). Universality is tested at the substantive level. Findings about CLT-baseline properties are reported as expected baselines rather than as universality claims. This pre-empts the critique by making the distinction explicit in the experimental design rather than retroactively in the results discussion.

**The linear flow may be too crude.** The linear-Gaussian model is a first-order approximation; higher-order structure (correlations across layers, non-Gaussian tails) may be where interesting variation lives. We mitigate by including residual-variance and non-Gaussianity diagnostics in Phase 1 — if these are not flat across models, we have direct evidence of where the framework's first-order description is incomplete.

**Per-input information may live entirely in the residuals.** The linear flow is by construction a *population* statistic. The per-input deviations $w(t,\tau)$ are what carry input-specific information. If improving the population flow (via prediction-based intervention) doesn't help model quality because all the useful per-input variation is in the residuals, Phase 4 interventions will fail. This is part of what Phase 4 tests.

**The 150M scale may be below the framework's regime.** Sarfati et al. validated at 1B-7B. If 150M is below where the linear-Gaussian description applies tightly, Phase 1's structural diagnostics will fail. This is detectable early and would be a substantive negative result for the framework's lower scale limit.

**Vocabulary-anchored alignment may not be the right tool.** Orthogonal Procrustes assumes that the embedding spaces correspond up to an orthogonal transformation. This may not hold — embedding spaces could disagree in nonlinear ways. The alignment residual $\rho$ is itself a measurement of this assumption's validity; if it's consistently large, the aligned-coordinate analyses become less informative but the basis-invariant ones are unaffected.

### 10.2 Methodological risks

**Variance characterization may be insufficient.** Three seeds per variant gives a sample size of 3 — small. Distinguishing real architectural differences from seed variance requires that across-variant differences be substantially larger than within-variant. If the differences are comparable, we'd need more seeds, which is feasible at 150M but costs additional compute time.

**Training quality matching across variants is hard.** "Same final quality" is operationalized as "same final held-out loss within some tolerance," but architectural choices interact with hyperparameters in ways that may make matching quality across all four variants nontrivial. We accept this and report both the achieved quality and the resulting flow comparisons, with the caveat that some variation may reflect quality differences rather than architectural ones.

**MLA at 150M with short context is in an unusual regime.** MLA's inference-time benefits don't manifest at this scale; we include it for structural comparison of the attention computation. Reviewers may legitimately ask whether this comparison generalizes to MLA's intended operating regime. The honest answer: we test a structural difference at small scale; generalization to MLA's full-scale regime is a separate question.

**Our DeepSeek-style variant uses simplified MLA (no decoupled RoPE).** DeepSeek-V3's reference MLA implementation splits Q and K into a non-positional part (derived from the latent) and a positional part (separately projected, with RoPE applied to the positional part only). This split exists to make the inference KV cache compatible with RoPE — the position-dependent rotation can't be precomputed into the cached latent. Our pilot doesn't run inference at scale and doesn't benefit from the decoupled scheme. We use *simplified MLA*: down-project to latent, normalize, up-project to per-head Q/K/V, apply RoPE directly to the up-projected Q/K. This preserves the core architectural feature of MLA (rank-constrained K and V derived from a low-dimensional latent) while removing the decoupled-RoPE inference-cache machinery. We disclose this so a reviewer checking against DeepSeek-V3's reference implementation knows the deviation. The simplification doesn't affect the structural property we want to study — that K and V live in a rank-$d_c$ subspace of the hidden space — which is what we hypothesize will show up in the recovered flow.

**The post-final-norm "layer 14" state is qualitatively different from block-output states.** Our model has 12 transformer blocks producing 13 layer states (post-embedding + 12 block outputs) plus one additional state after the final RMSNorm before the LM head. The post-final-norm state is what feeds the LM head — it has had RMSNorm applied and may exhibit the "last-layer anomaly" Sarfati et al. note in their tested models (Mistral, Llama-3.2-1B, Llama-3.2-3B all show anomalous last-layer statistics). We capture all 14 layer states identically in the analyzer, but in reporting we treat the post-final-norm state separately and don't include it in trajectory-level analyses (e.g., curve-level smoothness) where its anomalous nature would distort cross-layer trends.

**Our linear-flow prediction uses ordinary least-squares regression, not the paper's element-wise scaling formula.** The paper writes the prediction as $\tilde{x}(t+\tau) = R(t+\tau)\Lambda(t,\tau)R(t)^\top x(t)$, which is mathematically a specific linear map but implementationally relies on the SVD axes at $t$ and $t+\tau$ being sign-consistent and order-matched. NumPy's SVD provides no such guarantee — signs of singular vectors are arbitrary per-vector, and naively applying the paper's formula gives residuals 100-1000× larger than the truth due to sign-flip artifacts (verified in Phase 0 synthetic-data testing). Our analyzer instead computes the prediction $X_{t+\tau} \approx X_t A$ via ordinary least squares, which is mathematically equivalent to the paper's formula under matched sign conventions but is sign-robust under any. The recovered $\{R(t), \Sigma(t)\}$ are still stored for alignment and reporting purposes; they're just not the path to the residual prediction. This is a methodological deviation worth disclosing; we believe it's a strict improvement (the paper's variance/kurtosis statistics implicitly assume sign-matched predictions and the OLS form produces those), but a reviewer should be aware.

**Gemma's sliding-window attention is functionally inert at our pilot scale.** Gemma-2 alternates between layers with sliding-window attention (window 4096) and layers with full attention. At our pilot's sequence length of 1024, the sliding window is wider than any sequence, so the sliding-window layers behave identically to full-attention layers. We implement the alternating pattern faithfully (even-index layers configured as sliding-window, odd-index layers as full) so the architecture is reproducible from the config, but Phase 2's "Gemma vs Llama" comparison effectively tests Gemma's hybrid pre+post normalization, GeGLU MLP, and attention logit softcap, not its sliding-window design. The sliding-window pattern would only become active at seq_len > 4096, which is outside our pilot's scope. Frontier-scale follow-up work with longer contexts would test that part of Gemma's design.

**Findings are conditional on FineWeb-Edu.** The pilot's primary corpus is FineWeb-Edu (10BT sample). Phase 2 includes a secondary corpus comparison across FineWeb-Edu, The Pile, and RedPajama, but these three corpora overlap substantially in their bulk web content. Findings about universality are therefore conditional on the "broad modern English web corpus" regime. Whether the same universality holds for genuinely different corpora (code-only, single-language non-English, synthetic-data-only) is outside the pilot's scope and would require separate validation.

### 10.3 Risks specific to the field's trajectory

**Synthetic data is changing the corpus.** Frontier-scale training increasingly uses synthetic data. Our pilot trains on naturally-occurring corpora (FineWeb-Edu, The Pile, RedPajama), so our results characterize the "natural language" regime. Whether the findings transfer to synthetic-data training is an empirical question outside the pilot's scope.

**Architectural innovation may continue.** New attention variants, normalization schemes, or non-transformer architectures may appear during the project. Our four variants are a snapshot of the post-2024 consensus and would need extension to characterize newer architectures.

---

## 11. Resource summary

**Hardware.** A single workstation with one NVIDIA RTX 5090 (32 GB GDDR7, 1.79 TB/s memory bandwidth), 64 CPU cores, and 512 GB system RAM. Sufficient SSD storage (~500 GB) for model checkpoints (~600 MB × ~50 checkpoints × ~30 runs = ~1 TB max, with intermediate runs pruned to final checkpoints once analyzed) and analysis artifacts (~30 MB × ~1500 flow files = ~50 GB). Activations are not persisted — they are accumulated in RAM, SVD'd, and discarded per checkpoint.

**Compute timeline by phase** (revised based on smoke-test measurement of ~65k tokens/sec on the RTX 5090):

| Phase | Description | Training time | Total elapsed |
|-------|-------------|---------------|---------------|
| 0 | Pipeline validation (already complete) | (smoke test, 5 min training) | (done) |
| 1 | Convergence | 4-6 runs × 7 hr = 28-42 hr (~2 days) | 3-5 weeks |
| 2 | Universality | 18 runs × 7 hr = 126 hr (~5-6 days) | 8-12 weeks |
| 3 | Predictability | (analysis only) | 2-4 weeks |
| 4 | Intervention | 12-20 runs × 7 hr = 84-140 hr (~4-6 days) | 8-14 weeks |
| **Total** | | **~3 weeks training compute** | **5-8 months elapsed** |

Note that training compute is no longer the binding constraint. The dominant elapsed time is analysis, plotting, interpretation, and (in Phases 3-4) iterative method development. Training is compute-light enough that we could increase per-variant seed counts beyond the planned 3 if early Phase 2 results show high within-variant variance.

**Personnel.** One researcher operating the workstation. Most phases involve substantial automated training time during which the researcher is not actively engaged. Estimated active researcher time: ~4-6 person-months, spread over the project's 5-8 month elapsed time.

**Data and infrastructure.** Public natural-language corpora (FineWeb-Edu, The Pile, RedPajama) for training. Standard transformer training infrastructure (PyTorch). Standard tokenizer (a single tokenizer used across all variants — likely Llama-3-style 32k or 128k BPE — to ensure vocabulary alignment is exact, not approximate).

**Software deliverables (open-source).** Implementation of the four architectural variants at 150M; the lines-of-thought analysis pipeline including vocabulary-anchored Procrustes alignment; the predictor module; the intervention implementations (A, B, C). A reusable codebase that other researchers can apply to their own models.

---

## 12. Deliverables

**Empirical contributions:**

- Characterization of linear-flow convergence with training compute at 150M, including comparison to loss-curve convergence.
- Quantitative test of universality across four modern transformer architectures via vocabulary-anchored alignment, reporting both alignment quality and post-alignment agreement.
- Test of corpus universality across three broad natural-language corpora.
- A predictor for the asymptotic linear flow from partial training, with calibrated accuracy on held-out checkpoints.
- Training intervention results (one or more of: diagnostic monitoring outcomes, regularization-based training acceleration, spectral-initialization speedup).

**Methodological contributions:**

- A principled vocabulary-anchored alignment procedure for comparing population-level structures across LLMs with different embedding spaces, applicable beyond this specific project.
- A multi-level universality framework (structural, basis-invariant, aligned-coordinate) that disentangles what kinds of cross-model agreement different observations imply.

**Software artifacts:**

- The four 150M architectural variants, implemented faithfully.
- The lines-of-thought analysis pipeline with the alignment procedure.
- Trained checkpoint series for all phases, released openly.

**A comprehensive technical report** summarizing findings across all four phases, framed as a pilot study and explicitly identifying questions that frontier-scale follow-up work would need to address.

---

## 13. Relation to prior work

**Sarfati et al. (ICLR 2025) — Lines of Thought in Large Language Models.** Direct foundation of this work. The pilot study extends Sarfati et al. in four specific directions: (1) controlled comparison across architectures at fixed scale and corpus, (2) characterization of convergence with training compute, (3) introduction of vocabulary-anchored alignment for well-posed cross-model comparison, (4) tests of predictability and intervention.

**Scaling laws (Kaplan et al. 2020, Hoffmann et al. 2022).** These works characterize how loss and downstream performance scale with compute, data, and parameters. Our work characterizes a different population-level object (the linear flow) along similar axes (compute, architecture, corpus). The frameworks are complementary; scaling laws describe what trained models *do*, our work describes part of *how they do it*.

**Mechanistic interpretability (Elhage et al., Bricken et al., on transformer circuits and superposition).** Works on what individual heads and circuits compute. The lines-of-thought framework operates at a different level — population statistics rather than individual computational components. They are complementary lenses on trained models.

**Representational similarity analysis (CKA, SVCCA, RSA).** A substantial literature compares neural network representations using kernel- or correlation-based similarity scores. *Centered Kernel Alignment* (Kornblith et al. 2019) compares pairs of representations via a similarity *scalar* derived from kernelized inner products; it is widely used and basis-invariant. *Singular Vector Canonical Correlation Analysis* (Raghu et al. 2017) compares pairs of representations via the canonical correlations between their leading singular subspaces. Both tools tell you "model A's representation at layer $t$ is similar to model B's representation at layer $t$" but neither produces an explicit *map* from one model's hidden space to the other's.

Our work requires that map. The linear flow $R^{(M)}(t)$ at each layer is an orthonormal matrix in model $M$'s hidden space; to test whether two models' flows are quantitatively the same, we need to transport one into the other's coordinate system and then perform a direct matrix comparison (Frobenius distance, principal angles after transport). Vocabulary-anchored Procrustes gives us this map; CKA and SVCCA do not. They are complementary tools for a different but related question.

Practically, we report basis-invariant universality (the §1.6 family-1 and family-2 metrics) alongside the aligned-coordinate metrics; the basis-invariant metrics function as something analogous to what CKA/SVCCA would tell us at a per-layer level, while the aligned-coordinate metrics give the sharper conditional finding that those methods cannot produce.

---

## 14. Honest assessment

The pilot study could succeed at three nested levels:

*At the lowest level*, Phase 1 alone characterizes how the linear flow converges with training at 150M, with multi-seed reproducibility. This is a clean empirical contribution achievable in 5-8 weeks.

*At the middle level*, Phases 1-3 add architectural universality and predictability, producing a controlled empirical map of what trained 150M models look like at the population-flow level. This requires the full 5-month investment but produces a substantial unified result.

*At the highest level*, Phase 4's interventions add an applied dimension — either positive (training-efficiency improvements) or negative (interventions don't help, characterizing the linear flow as a consequence rather than cause of training).

We are reasonably confident that Hypothesis 1 (convergence) and the structural part of Hypothesis 2 (2a) will hold, based on Sarfati et al.'s findings at larger scale. We are less certain about basis-invariant universality (2b), still less about aligned-coordinate universality (2c), and genuinely uncertain about Hypothesis 3 (predictability) and Hypothesis 4 (exploitation). The phase structure ensures that each hypothesis is tested before resources are committed to depending on it.

The intrinsic scope limitation — 150M is below the regime Sarfati et al. validated, so their qualitative findings may not transfer downward — is named explicitly. Positive findings at 150M would transfer to larger scales with some confidence; negative findings would transfer less cleanly, and a frontier-scale follow-up would be needed to confirm them in the regime where Sarfati et al.'s claims were originally established.

The proposal commits to a complete, self-contained pilot at a single scale, with all four phases conducted on a single workstation in 5-8 months. Frontier-scale generalization is acknowledged as a natural follow-up but is explicitly outside the present scope.

---

## References

Sarfati, R., Liu, T. J. B., Boullé, N., & Earls, C. J. (2025). Lines of Thought in Large Language Models. *International Conference on Learning Representations (ICLR).*

Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. *arXiv preprint.*

Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models. *NeurIPS.*

Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *Meta AI.*

Gemma Team (Google). (2024). Gemma 2: Improving Open Language Models at a Practical Size.

Qwen Team. (2025). Qwen3 Technical Report.

DeepSeek-AI. (2025). DeepSeek-V3 Technical Report.

Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Anthropic Technical Report.*

Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. *Anthropic Technical Report.*

Schönemann, P. H. (1966). A generalized solution of the orthogonal Procrustes problem. *Psychometrika.*

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *International Conference on Machine Learning (ICML).*

Raghu, M., Gilmer, J., Yosinski, J., & Sohl-Dickstein, J. (2017). SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability. *NeurIPS.*
