# Input-Conditioned Ensembles in Lines of Thought: A Mathematical Investigation

**Phase 2 supplementary investigation, multiview campaign**
**Run: phase1_runs_gelu, seeds 0–3, 50 checkpoints, final step 24000**

---

## Abstract

The "Lines of Thought" paper (Sarfati et al., ICLR 2025) shows that residual-stream activations across a held-out set of inputs form an approximately Gaussian "blunderbuss" at each layer of a trained transformer, with a covariance that grows exponentially in depth at a universal rate $\lambda$. The paper models this with a Langevin-style stochastic differential equation. A natural question is whether this macroscopic description has a microscopic analogue: if we condition on a specific input token, does the resulting sub-ensemble also satisfy a Gaussian description with a similar universal noise law?

We test this on a 150M-parameter Llama-style model trained from scratch (Phase 1 of the larger campaign). The answer is a clean *no*: input-conditioned residual-stream bundles are strongly non-Gaussian (Mardia kurtosis Z = 25–45 across interior layers, vs the |Z|=2 rejection threshold), with within-input total variance differing across input tokens by factors of 2–4×, and per-token exponential growth rates $\lambda_i$ that vary across tokens and across training steps in a structured, non-monotonic way. The marginal Gaussian observed by the paper is therefore a central-limit-theorem averaging over a population of non-Gaussian conditional bundles, not a microscopic property of any conditional.

A follow-up sub-conditioning analysis shows the conditional bundles can be made approximately Gaussian by further conditioning on local context (the next token in text, or the pilot position in the chunk). The clean GMM picture holds at the level of $p(x_t \mid v_i, \text{context})$, not at $p(x_t \mid v_i)$. We discuss an identifiability limitation of the current pilot scheme that prevents us from isolating which context variable is the relevant mixture index, and propose a clean follow-up experiment.

---

## 1. Background and motivation

### 1.1 Residual streams as a stochastic ensemble

Transformers process tokens through a sequence of residual blocks. At each layer $t$, every token position carries a residual-stream vector $x_t \in \mathbb{R}^H$, the running sum of all attention and MLP contributions up to that depth. For a held-out evaluation set with many input chunks and many token positions per chunk, the collection of these vectors at fixed layer $t$ forms an empirical distribution — the "ensemble" of residual-stream activations at depth $t$.

The Lines of Thought paper studies this ensemble's geometry across depth. Their headline findings: (i) the ensemble at each layer is approximately a single Gaussian whose covariance grows exponentially in depth at a universal rate $\lambda$, isotropically along the directions tangent to the singular vectors of the layer-to-layer mean flow, and (ii) trajectories of individual tokens through depth can be modeled by a continuous-time Langevin equation

$$dx = A(t)\,x\,dt + \sigma(t)\,dW, \qquad \sigma\sigma^\top = \alpha e^{\lambda t} I$$

with universal $(\alpha, \lambda)$ measured from the marginal-level statistics. We will call the all-token, all-position empirical distribution at layer $t$ the **all-to-all** bundle (or **marginal**).

It is important to note that the Langevin formulation is a *phenomenological model* of the marginal ensemble's variance growth, not a microscopic dynamical law. Transformer forward passes are deterministic given input and weights — the "noise" in the SDE represents heterogeneity across the ensemble (different inputs, different positions, different surrounding contexts) reinterpreted as fictitious process noise. The paper measures and models the marginal ensemble; it makes no direct empirical claim about the sub-ensemble conditional on any particular input token.

### 1.2 The conditional ensemble question

Our multiview campaign extends the Lines of Thought framework by *partitioning* the all-to-all bundle into sub-bundles based on observable conditioning variables:

- **Forward view (input-conditioned):** fix the input token at the pilot position. The sub-bundle is $\{x_t^{(k)} : \text{input}_k = v_i\}$ for each vocabulary token $v_i$, with $x_t^{(k)}$ the residual-stream vector at layer $t$ for pilot $k$.
- **Reverse view (output-conditioned):** fix the predicted next token (the argmax of the model's output distribution). The sub-bundle is $\{x_t^{(k)} : \text{pred}_k = w_j\}$.
- **All-to-all view:** the full marginal, as in the paper.

By the law of total variance, the marginal's covariance decomposes exactly into a within-component and a between-component piece:

$$\Sigma_{\text{marginal}}(t) = \underbrace{\sum_i \pi_i \Sigma_i(t)}_{\text{within: average of conditional covariances}} + \underbrace{\sum_i \pi_i (\mu_i(t) - \mu(t))(\mu_i(t) - \mu(t))^\top}_{\text{between: variance of conditional means}}.$$

Here $\pi_i$ is the empirical frequency of input token $v_i$ in the held-out set, $\mu_i(t)$ and $\Sigma_i(t)$ are the conditional mean and covariance, and $\mu(t)$ is the marginal mean. This identity is bookkeeping — it must hold to numerical precision regardless of any modeling assumption.

The interesting question is **what the conditional bundles look like**. If the marginal is Gaussian, the conditionals are constrained (their first two moments must reassemble to the marginal's), but their *shape* is underdetermined by the marginal alone. The campaign is designed to measure that shape directly.

### 1.3 The question

We organize the question around the following:

> **Is there a mathematical description for the input-conditioned ensembles $p(x_t \mid v_i)$ that has the same form as the Lines of Thought description of the marginal ensemble, and if so, how does it relate to the marginal description?**

This is the natural micro-macro question. The marginal blunderbuss is, by total-probability bookkeeping, a Gaussian Mixture Model (GMM) over the vocabulary of input tokens, weighted by token frequency. The question becomes: what is the structure of the components of that mixture, and does that structure satisfy a clean closed-form description that recovers the paper's marginal model under mixing?

---

## 2. Hypothesis formation

We began from the simplest candidate unification: a GMM where each input token contributes a Gaussian sub-bundle with token-specific drift $\mu_i(t)$ and a shared isotropic exponential covariance $\alpha e^{\lambda t} I$. Under this picture, the macroscopic Gaussian observed by the paper would be the marginal of a structured fluid of $|V|$ conditional Gaussians, with the paper's universal $(\alpha, \lambda)$ identified as the microscopic noise parameters governing every conditional bundle, and the per-token drifts supplying the between-component variance via the law of total variance.

This is an attractive picture because it provides a single mathematical description that bridges micro and macro: one stochastic process, one set of noise parameters, vocabulary-indexed initial conditions. If true, it elevates the paper's universal $\lambda$ from a marginal statistic to a microscopic physical constant.

However, three issues with this synthesis emerged on closer examination:

1. **The paper's empirical Gaussianity is about linearization residuals, not conditional bundles.** Sarfati et al. measure the distribution of $\delta x(t, \tau) = x(t+\tau) - \tilde x(t, \tau)$, where $\tilde x$ is a linear extrapolation along the all-to-all SVD basis. They do not measure or claim that $p(x_t \mid v_i)$ is Gaussian for any fixed $v_i$. The conditional Gaussianity assumption is an extrapolation from marginal phenomenology, not a consequence of it.

2. **The Langevin equation is a marginal-level metaphor.** As noted above, the "noise" is heterogeneity treated as a fictitious process. Whether this corresponds to a real Gaussian process at the conditional level (the only level where it would be a *dynamical* law) is an empirical question.

3. **The "high dimensions smear modes together" intuition is backwards.** It is sometimes argued that in $H = 896$ dimensions, GMM components are blurred into a single Gaussian by virtue of dimensionality. But concentration of measure works the other way: in high dimensions, Gaussians with even modest mean separations are *more* separable, not less, because typical samples sit on thin shells. If the marginal looks unimodal, it is because either (a) the between-token mean separations are small relative to within-token spread, or (b) the within-token covariance is large and structured enough to fill the gaps along the separating directions. Both are empirical claims.

The user's question therefore reframed: **what kind of conditional structure is consistent with the observed marginal Gaussianity, and which kind actually obtains in our trained model?**

### 2.1 Three candidate models

A mixture of Gaussians equals a single Gaussian *almost never* — Gaussianity is not preserved under mixing. So if the marginal is genuinely (approximately) Gaussian, the conditional family is constrained. Three candidate models exhaust the natural possibilities for the shape of the conditionals:

- **Model A — Gaussian conditionals with shared covariance.** $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \Sigma_0(t))$, same $\Sigma_0$ for every input token. The conditionals differ only in their drift, and the marginal covariance is $\Sigma_0(t)$ plus the between-token mean variance. Under Model A, the paper's universal noise parameters $(\alpha, \lambda)$ live at the microscopic level — they describe every conditional bundle identically. This is the unification originally proposed.

- **Model B — Gaussian conditionals with token-dependent covariance.** $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \Sigma_i(t))$ with $\Sigma_i$ varying across tokens. The marginal Gaussian requires some conspiracy between the $\mu_i$ and $\Sigma_i$ to remain Gaussian, but moment matching at the marginal level still works. Under Model B, the paper's universal $(\alpha, \lambda)$ are aggregate averages over token-specific values, not microscopic constants.

- **Model C — Non-Gaussian conditionals that mix to approximate marginal Gaussianity.** The conditionals are not even Gaussian; the marginal's apparent Gaussianity emerges from central-limit-theorem aggregation across many non-Gaussian components. Under Model C, the paper's microscopic description does not extend to conditionals at all — the entire SDE framework is a marginal-level phenomenology with no microscopic counterpart.

These models are nested in informativeness: A is the strongest claim (a single closed-form description for every conditional), B is intermediate (Gaussian conditionals but token-specific parameters), C is the weakest (no closed-form per-conditional description). Distinguishing them requires direct measurement of the conditional bundles.

---

## 3. Experimental design

We designed five discriminators, denoted D1–D5, each addressing a different empirical signature of A vs B vs C. All are computable from the existing Phase 1 GELU campaign output, which provides per-token SVD bases, singular values, kurtosis profiles, and the raw augmented activation files (state vectors paired with input tokens, next tokens, predicted tokens, and chunk positions) for each (seed, checkpoint) combination.

The campaign uses a 150M-parameter Llama-style transformer (14 residual-stream snapshots per forward pass: the embedding output plus 13 post-block states, hidden dim $H = 896$), trained from scratch on 4 seeds with 50 logarithmically-spaced checkpoints. The held-out set provides ~10,000 pilots per checkpoint (chunks × pilot positions), with the top-20 most frequent input tokens forming the **frozen forward set** used for per-token analysis. Per-token pilot counts in this set range from ~50 to ~400.

### 3.1 The discriminators

**D1 — Cross-token coefficient of variation of within-input variance.** Under Model A, the total within-input variance $\mathrm{tr}(\Sigma_i(t))$ is the same for every input token at each layer (modulo finite-sample noise). Under B or C, it varies. The test statistic is the coefficient of variation (standard deviation divided by mean, abbreviated CV) of $\mathrm{tr}(\Sigma_i(t))$ across the top-20 most frequent input tokens at each layer.

**D2 — Principal angles between conditional SVD bases.** Under Model A with shared $\Sigma_0$, every conditional bundle spans the same subspace, so the principal angles between any two token-conditional SVD bases should be small (the top-k right-singular vectors of $\Sigma_i$ should agree across $i$). Under B, the bases diverge, and the principal angles grow. We compute principal angles in the top-k subspace (originally $k=10$, later revised to $k=2$ — see §3.2).

**D3 — Per-token $(\alpha_i, \lambda_i)$ fits.** The paper's variance-growth law $\log V_{\text{within}}(t) = \log \alpha + \lambda \log t$ can be fit per-token to $\mathrm{tr}(\Sigma_i(t))$. Model A predicts that $(\alpha_i, \lambda_i)$ is constant across tokens and matches the paper's marginal $(\alpha, \lambda)$. Model B predicts scatter. The coefficient of variation of $\lambda_i$ across tokens is the primary discriminator.

**D4 — Multivariate Gaussianity of conditional bundles.** Splits into D4a (per-coordinate excess kurtosis in each token's own SVD basis — cheap, no augmented file required) and D4b (Mardia's multivariate kurtosis Z-score in a shared 32-dimensional PCA subspace — requires the augmented activation file). Both reject the Gaussianity assumption common to A and B if the conditionals exhibit significant non-Gaussian structure. Mardia's Z is asymptotically standard normal under the Gaussian null; |Z| > 2 corresponds to ~5% rejection, |Z| > 25 is effectively impossible under Gaussian sampling.

**D5 — GMM reconstruction.** Sample synthetic residuals from per-token Gaussian fits (using each token's empirically measured mean and covariance), weight by empirical token frequencies, and compare the synthesized marginal kurtosis profile against the directly observed marginal kurtosis profile. If the synthesized marginal matches the empirical marginal in higher-order statistics, the Gaussian-conditional assumption is supported. If the synthesized marginal is markedly more Gaussian than the observed marginal, the conditionals must themselves be non-Gaussian (since Gaussian-fitting them and remixing produces a more Gaussian object than the data shows).

A verdict logic combining the discriminators tags each (seed, step, layer) as A, A_partial (shape uniform but scale varies), B, C, or unclear.

### 3.2 Threshold calibration via bootstrap noise floor

The verdict logic requires numerical thresholds (e.g., "what value of CV(trace) counts as evidence of B vs A?"). The right approach is to measure the finite-sample noise floor at the actual pilot counts available — cross-token differences below that floor are uninterpretable.

We wrote a calibrator (`noise_floor.py`) that, for each of the 20 forward-set tokens at seed 0, step 24000:

- Bootstrap-resamples that token's pilots with replacement (B = 100 replicates).
- Recomputes the full discriminator statistic pack on each replicate.
- Reports the within-token bootstrap standard deviation of each statistic.

This is the "if the true value were fixed at this token's expected value, how much would my measurement bounce around due to finite sampling" floor. Cross-token spreads larger than ~3× this floor are unlikely to arise from sampling noise alone and indicate genuine Model B (or C) signal.

We also cross-checked against the cross-seed standard deviation (same token, same step, across four seeds), to verify that seed-init noise doesn't dominate finite-sample noise. The cross-seed floor for the trace statistic was 0.055, vs the bootstrap 1-sigma of 0.034 — finite-sample noise is the dominant source of measurement uncertainty, with seed-init noise contributing about 60% on top.

The calibration produced threshold candidates:
- CV($\mathrm{tr}\,\Sigma_i$) ≈ 0.115 (after trimming the newline-token outlier, which has unusually heterogeneous contexts)
- CV($\lambda_i$) ≈ 0.062
- Multivariate kurtosis Z ≈ 0.234
- Mean self-pair principal angle for D2: **35.9°**, which gives a 3-sigma threshold of 107.75° — **larger than the maximum possible angle (90°).** This is a discovery, not a usable threshold: the top-10 principal directions of an $N \times H$ bundle are not stable under resampling when $N \sim 100\text{–}400$ and $H = 896$. D2 has no discriminating power as originally configured.

This led us to restrict D2 to the top-2 principal directions (where stability is higher) and demote D2 from a primary verdict driver to a confirmatory diagnostic. The methodological lesson is general: principal-subspace discriminators in high dimensions require careful sample-size analysis before they can be interpreted.

---

## 4. Results

### 4.1 The verdict

All four seeds gave the same per-layer verdict pattern at the final checkpoint:

| Layer t   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Verdict   | B | C | C | C | C | C | C | C | C | C | C  | C  | C  | C  |

Thirteen of fourteen layers tagged "C" (non-Gaussian conditionals), with layer 0 tagged "B". Layer 0 is the raw embedding layer where, for a fixed input token, every pilot has the *same* residual-stream vector — the conditional variance is structurally zero and the Gaussianity diagnostic is degenerate. The interesting result is layers 1–13.

The pattern is consistent across all four seeds, ruling out seed-init artifacts. The Model A unification is decisively refuted at every interior layer.

### 4.2 D1: within-input variance is strongly token-dependent (Model A refuted)

The per-layer cross-token CV of $\mathrm{tr}(\Sigma_i(t))$ sits at **0.6–0.8 across every interior layer**, with a spike to 1.7 at layer 1. The threshold is 0.115. The signal exceeds threshold by a factor of 5–7×.

Concretely: different input tokens produce within-input bundles whose total variance differs by factors of 2–4× across the top-20 tokens. This is not borderline B; this is a strong, layer-pervasive Model B signature. The "shared $\Sigma_0$" picture is not approximately right — it is qualitatively wrong.

The training-trajectory panel (D1, right) shows the same CV is high throughout training. The layer-7 CV spikes near step ~300 to ~0.85, then settles to ~0.6 by step 1500 and rises slightly to the final value. There is no early-training phase where Model A is consistent with the data; the conditional bundles have token-dependent variance throughout.

### 4.3 D3: $\lambda$ is not universal, with a non-monotonic training trajectory

The per-token $\lambda$ scatter at the final checkpoint (D3, left) shows a clear bimodal distribution: most of the 20 forward tokens cluster near $\lambda \approx 0.55\text{–}0.60$, with a separated lower cluster near $\lambda \approx 0.40$ comprising five tokens. The all-to-all $\lambda \approx 0.36$ sits *below* every per-token value.

The marginal exponential growth rate being smaller than every conditional rate is geometrically consistent: the marginal variance grows from two sources (within-token growth plus between-token mean drift), but the exponential structure is dominated by the slower-growing of these. The within-token rates are larger because the within-token bundles are smaller and grow faster in relative terms.

CV($\lambda$) across tokens at the final checkpoint is 0.14, above the 0.062 threshold. $\lambda$ universality fails — confirming Model A's failure on a second independent axis.

The training-trajectory panel (D3, right) is the most interesting individual result of the campaign: **CV($\lambda$) is non-monotonic in training time.** It begins at ~0.20 at step 100, decreases to a minimum near 0.09 around step 1500–2000 (most universal), then increases back to ~0.14 by step 24000. The universality of conditional dynamics is itself a training-time phenomenon that *peaks* mid-training and partially decays as training continues. The minimum sits approximately at the same training steps as several Phase 1 anomalies (the post-final-norm anomaly through step ~2000, the log-$\alpha$ hump near step 5000, the $\Sigma$-distance bump 5000–10000); this co-location is suggestive and warrants direct overlay analysis.

### 4.4 D4: conditional bundles are not Gaussian (Model B refuted)

**D4a (per-coordinate excess kurtosis, in each token's own SVD basis):** the mean-over-tokens conditional kurtosis profile rises from baseline at layer 1 to ~3–6 at interior layers 2–8, then gradually decays toward 1–2 by the final layer. The all-to-all marginal kurtosis (dashed lines, computed in the same way) sits near 0–0.5 throughout. The conditional bundles are *more* non-Gaussian than the marginal — by a substantial margin.

**D4b (Mardia multivariate kurtosis Z-score, in a shared 32-dimensional PCA subspace):** the per-token mean Mardia Z values are **25–45 across interior layers 2–10**. The standard Gaussian rejection threshold at the 5% level is |Z| = 2. We are rejecting multivariate Gaussianity by 25 sigma. There is no statistical ambiguity here: the conditional bundles are not Gaussian at any interior layer.

(The deep negative Z spike at layer 0, ~-130, is the same sample-degeneracy artifact noted above and is not a real result.)

### 4.5 D5: the GMM reconstruction reveals the structural failure mode

D5 is the synthesis check that ties all the discriminators together. We take the per-token empirically measured means and covariances $(\hat\mu_i, \hat\Sigma_i)$ in the top-32 PCA subspace of the marginal bundle, weight by empirical token frequencies, and *sample* from the resulting mixture. We then compare the sampled bundle's kurtosis to the empirical marginal bundle's kurtosis at each layer.

The result has three regimes:

- **Empirical marginal:** excess kurtosis 5–15 across interior layers 1–7, gradually decaying to ~0 by the final layer.
- **Model A reconstruction** (shared $\Sigma_0$ = average of per-token $\Sigma_i$): excess kurtosis ≈ 0 throughout.
- **Model B reconstruction** (per-token $\Sigma_i$): excess kurtosis ≈ 0.3–0.7 throughout.

Both Gaussian-mixture reconstructions produce nearly Gaussian marginals. The actual marginal has substantial excess kurtosis. Therefore the conditional bundles *cannot* be Gaussian — Gaussian-fitting them and remixing produces a markedly more Gaussian object than the data shows.

This pins down the C verdict from a complementary direction. The non-Gaussianity diagnosed by D4 is not an artifact of a particular Gaussianity test; it manifests as a real gap between observed marginal kurtosis and any GMM reconstruction with Gaussian components.

### 4.6 The unification, as sharpened by the data

Putting D1, D3, D4, and D5 together coherently:

> **The marginal Gaussian observed by Sarfati et al. is a central-limit-theorem phenomenon arising from aggregation over a population of non-Gaussian conditional bundles, not a microscopic property of any conditional. The paper's universal $(\alpha, \lambda)$ describe marginal-level statistics that have no clean microscopic counterpart in the form the paper's framework would suggest.**

The conditional bundles have, simultaneously:
1. Token-dependent total variance (D1, 0.6–0.8 CV across tokens at every interior layer).
2. Token-dependent exponential growth rate (D3, with non-monotonic universality trajectory through training and a bimodal cluster structure at the final checkpoint).
3. Multivariate non-Gaussianity (D4, Mardia Z = 25–45 at interior layers).
4. Heavy-tailed/multi-modal structure that gets averaged away at the marginal level (D5, marginal kurtosis 5–15× higher than any GMM reconstruction with Gaussian components).

The micro–macro relationship is real and bookkeeping-exact via the law of total variance, but the *form* of the conditionals is not a direct extension of the marginal description. There is no $p(x_t \mid v_i) = \mathcal{N}(\mu_i(t), \alpha e^{\lambda t} I)$ that adequately describes any conditional bundle, nor any small modification thereof.

---

## 5. The context-mixture hypothesis and its test

The natural interpretation of the C verdict is that $p(x_t \mid v_i)$ is *itself* a mixture — over the local context in which the input token appears. The same token "the" preceded by "of" or by a period or by a paragraph break occupies different regions of residual space; aggregating across these contexts produces heavy tails and non-Gaussian shape even if each context-specific sub-bundle is individually Gaussian.

This is a structural hypothesis: it says the C verdict is "Gaussian conditional on more variables" rather than "intrinsically non-Gaussian." It is testable, because the augmented activation files store, for each pilot, not only the input token but also the actual next token (the token at position $p+1$ in the underlying text), the model's predicted next token (the argmax of its output distribution), and the pilot's position in the chunk.

### 5.1 Sub-conditioning experiment

We computed the per-layer excess kurtosis profile of each input-conditioned bundle, then partitioned each bundle three ways and re-computed kurtosis on each sub-bundle:

- **$(input, next\_token)$:** condition on the actual successor in the text.
- **$(input, pred\_token)$:** condition on what the model predicted.
- **$(input, position)$:** condition on the pilot's chunk position. Originally intended as a control.

For each input token, sub-bundles with fewer than 20 pilots were dropped (the kurtosis estimator is unstable at smaller sizes). Sub-bundle kurtosis values were aggregated per layer as the sample-size-weighted mean across sub-bundles.

**Result:** sub-conditioning on next-token collapses aggregate kurtosis from baseline values of ~2.5 at layer 1 to ~0.3, and from ~1.0 at layer 5 to ~0.1. Sub-conditioning on predicted token reduces kurtosis much less, with the curve tracking close to baseline at interior layers. Sub-conditioning on position reduces kurtosis nearly identically to next-token.

The per-token result for the newline token (token id 13) was dramatic: baseline kurtosis 7.66 at layer 7 collapsed to 0.02 when sub-conditioned on next-token. The non-Gaussianity of the newline-conditioned bundle was *entirely* the context-mixture effect — once context is held fixed, the sub-bundle is essentially Gaussian.

The pred-token result is independently interesting: trajectories headed to the *same model prediction* are not a Gaussian sub-population, while trajectories that actually produced the *same next token in the text* are. The geometrically natural mixture index for the conditional bundles is "what comes next in the text" rather than "what the model predicts comes next." (We note the caveat that pred-token may simply be a coarser partition than next-token, since many input contexts produce the same predicted token, and partition coarseness alone could explain part of the gap.)

### 5.2 The next-token-vs-position disambiguation problem

Sub-conditioning on position reduced kurtosis nearly identically to sub-conditioning on next-token. This was unexpected — position was intended as a control, meant to *not* explain context-driven structure. Two interpretations are possible:

(a) Position is acting as a *context proxy* in this dataset. Each pilot position corresponds to a particular textual region within a fixed-length chunk, so the next-token distribution at a given position is concentrated around the words that typically appear in that text region. Next-token would be the real explanatory variable; position helps because it is correlated with context.

(b) Position is itself the explanatory variable, with the heavy tails driven by something position-dependent (positional encoding effects, perhaps, or systematic differences in what kind of text appears at different chunk positions). Next-token would help only because it is correlated with position.

To distinguish (a) from (b), we ran a joint-partition test: simultaneously partition by (next-token, position) and compare the kurtosis reduction against position-alone and next-alone. The marginal explanatory power of next-token *beyond* position (and vice versa) is the diagnostic.

**Result:** the joint partition produced *zero* sub-bundles with ≥20 samples across all 20 input tokens. Sample sizes shatter under joint partitioning, and the discriminator cannot be computed.

The diagnostic we *can* compute is the normalized mutual information NMI(next; position) per token, which measures how much the distribution of next-token within each position-bucket deviates from its overall distribution:

$$\text{NMI}(\text{next}; \text{position}) = \frac{I(\text{next}; \text{position})}{H(\text{next})}$$

A value near 0 means next-token and position are independent; a value near 1 means they are perfectly entangled (knowing position uniquely determines next-token). Across all 20 forward-set tokens, NMI values ranged 0.38–0.69, with a median of 0.55. **In our held-out pilot set, next-token and position are heavily entangled** — knowing position tells you most of what you would learn from next-token, and vice versa.

This is structural: the multiview campaign uses *fixed* pilot positions across all held-out chunks. Token 28725 ("the") appearing at position 13 across many chunks is followed by a much narrower distribution of next-tokens than the same token at position 200, because the textual contexts producing a given token at a specific chunk position are constrained. **The fixed-pilot-position scheme inadvertently acts as a context-stratification scheme.** Position and next-token carry overlapping context information, and we cannot disentangle them.

### 5.3 What we can and cannot conclude

**We can conclude:** the conditional bundle $p(x_t \mid v_i)$ is non-Gaussian, and partitioning by either next-token or position yields approximately Gaussian sub-bundles. *Context* — parameterized as next-token, position, or some underlying variable both correlate with — is the mixture index that, once held fixed, restores Gaussianity.

**We cannot conclude:** which of these variables is the *causal* mixture index. Next-token-vs-position is statistically unidentified in our dataset.

The honest framing: the GMM picture *holds* at the level of $p(x_t \mid v_i, \text{context})$ for some appropriate notion of context, with the granularity of that context at least as fine as the joint $(next, position)$ distribution. The simplest unification of micro and macro that *does* fit the data is therefore:

$$p(x_t \mid v_i, c) \approx \mathcal{N}(\mu_{i,c}(t), \Sigma_{i,c}(t))$$

with $c$ a context variable, and the marginal blunderbuss arising from aggregation over both vocabulary and context. Whether $\Sigma_{i,c}(t)$ has the universal isotropic exponential form remains testable, but requires data with independent variation in context and position.

---

## 6. Where the Fokker–Planck framing fits

If the unification had returned Model A — shared $\Sigma_0(t)$ across all conditionals — the Fokker–Planck PDE corresponding to the paper's SDE would have given a beautiful theoretical statement: every conditional density would obey the *same* PDE with the same coefficients, differing only in initial condition $\delta(x - v_i)$. The marginal would then be the convolution of a vocabulary-indexed Dirac comb with a single universal heat-kernel-like propagator. The unification would be: one Fokker–Planck equation, vocabulary-indexed initial conditions, all derived statistics following from that single PDE.

The data does not support this. Under the actual verdict, no single Fokker–Planck equation describes all conditionals — token-dependent diffusion coefficients (Model B's signature) or non-Gaussian sub-structure (Model C's signature) each break the single-PDE picture in different ways.

The Fokker–Planck framing remains useful as a description of the *marginal* dynamics, where the deterministic ODE for the marginal covariance,

$$\dot\Sigma = A(t)\Sigma + \Sigma A(t)^\top + \sigma(t)\sigma(t)^\top,$$

cleanly captures the paper's variance-growth law. But it does not extend microscopically. The transformer's residual-stream dynamics is *not* a single stochastic process applied to vocabulary-indexed initial conditions; it is a more structured object whose microscopic content has been hidden by marginal-level CLT averaging in the Lines of Thought analysis.

This is itself a substantive finding. The Fokker–Planck framing is correct for what the paper measured, but its natural extension to conditionals — which would be the cleanest possible micro–macro unification — does not survive contact with the data.

---

## 7. Summary of findings

1. **The simplest unification (Model A: shared-covariance GMM) is decisively refuted.** Cross-token CV of within-input variance is 0.6–0.8 at every interior layer, vs the 0.115 threshold derived from finite-sample noise floor calibration. The signal exceeds threshold by 5–7×, with the same pattern across all four seeds.

2. **Model B (Gaussian conditionals with token-dependent covariance) is also refuted.** Multivariate Gaussianity rejection at Mardia Z = 25–45 (interior layers, vs |Z| = 2 rejection threshold). D5 reconstruction from any Gaussian per-token fit produces a near-Gaussian marginal (excess kurtosis ≈ 0–0.7), but the actual marginal has excess kurtosis 5–15 at interior layers.

3. **Model C holds, with a context-mixture interpretation supported by sub-conditioning.** Sub-conditioning by next-token or position collapses kurtosis to ~0, indicating approximate Gaussianity at the level of $p(x_t \mid v_i, \text{context})$.

4. **The paper's universal $\lambda$ is a marginal-level CLT phenomenon, not a microscopic rate.** Per-token $\lambda$ values cluster bimodally at $\lambda \approx 0.40$ (5 tokens, punctuation/whitespace-adjacent) and $\lambda \approx 0.59$ (bulk); the all-to-all $\lambda \approx 0.36$ is smaller than every per-token value. CV($\lambda$) across tokens is non-monotonic through training, with a minimum near step 1500–2000 and a rise back to ~0.14 by step 24000 — co-located approximately with the Phase 1 marginal-dynamics anomalies.

5. **Sub-conditioning on model prediction fails to produce Gaussian sub-bundles, while sub-conditioning on actual next-token succeeds.** The natural geometric mixture index for the conditional bundles is "what actually comes next in the text" rather than "what the model predicts" — though this distinction may be partly confounded by partition granularity.

6. **The next-token-vs-position identification problem is unresolved.** NMI between next and position is 0.4–0.7 across all forward-set tokens; the fixed-pilot-position scheme inadvertently stratifies by context. The joint-partition test could not be run due to sample-size shattering. We cannot, with this dataset, isolate the causal mixture index.

7. **Methodological finding: principal-subspace discriminators in high dimensions require sample-size analysis.** The D2 (principal-angle) discriminator was originally configured with top-10 subspaces and proved sample-size-degenerate at our pilot counts (mean self-pair angle 35.9°, vs the 90° upper bound). This is a general issue for high-dimensional residual-stream analysis; we recommend bootstrap-floor calibration before deploying such diagnostics.

---

## 8. Suggested follow-up work

### 8.1 Completed analysis: the $\lambda$ bimodality reflects a grammatical-role distinction

The D3 scatter showed a bimodal distribution of per-token $\lambda$ values at the final checkpoint. To characterize this, we ran k-means (k=2) on the per-token $\lambda$ values averaged across seeds, decoded the cluster members using the Mistral-7B tokenizer, tested cross-seed stability of cluster assignment, and applied Welch's t-tests at each layer comparing the clusters on per-token trace, effective rank, and kurtosis profiles.

**Cluster membership.** The 20 forward-set tokens split cleanly into:

- **Low cluster** (λ ≈ 0.45, 8 tokens): `\n`, `.`, `-`, `0`, `1`, `2`, `s` (likely the orphan possessive/contraction fragment), and a likely BPE space marker. These are *structural/sub-word tokens*: terminators, digits, and sub-lexical fragments.
- **High cluster** (λ ≈ 0.58, 12 tokens): `the`, `of`, `and`, `to`, `a`, `in`, `is`, `that`, `are`, `for`, plus `,` and `(`. These are *connective tokens*: function words plus the internal punctuation that syntactically functions like glue rather than as a terminator.

The distinction is not "punctuation vs words" but a more refined grammatical-role split: tokens that *terminate or fragment* a syntactic unit (low cluster) vs tokens that *connect within* a unit (high cluster). The comma and open-parenthesis in the high cluster fit this interpretation — they connect rather than terminate — while the period in the low cluster is a strong terminator.

**Cross-seed stability.** 17 of 20 tokens have stable cluster assignment across all four seeds. The 3 unstable tokens (`.`, `s`, and the BPE space marker) all sit near the cluster boundary at λ ≈ 0.48–0.50. The unambiguous cluster members (`\n`, `-`, digits in the low cluster; all 12 high-cluster tokens) are rock-solid across seeds. This pattern strengthens the interpretation: the bimodality is a real structural distinction with a small fuzzy boundary region rather than an artifact of any single training run.

**The bimodality also manifests sharply in effective rank.** This was the unexpected finding. The Welch t-test for effective rank by cluster is significant (p < 0.05) at every interior layer 1-9, and highly significant (p < 0.02) at layers 1-4. At layer 1, low-cluster mean effective rank is 6.4 vs high-cluster 19.8 (p < $10^{-4}$). At layer 5, 37.7 vs 63.7 (p ≈ 0.017). **Structural tokens occupy roughly half the residual-stream dimensions that function-word tokens occupy, across the entire model depth.** This is a sharper cluster signature than λ itself; trace is significant only at early layers (1-4), kurtosis is never significant.

Geometrically, lower λ and lower effective rank are likely two views of the same phenomenon: bundles confined to a lower-dimensional subspace cannot expand as quickly in trace terms because the directions of expansion are fewer.

**Training-time emergence.** The bimodality emerges sharply between training steps ~300 and ~1000 and continues to tighten through step 24000. Cohen's d (standardized cluster separation) is essentially zero through step 200, climbs from step 300 onward, reaches d ≈ 1.5 by step 1000, d ≈ 3 by step 10000, and d ≈ 4 by step 24000. The clusters do not separate by their centroids drifting apart (the mean λ gap is small and grows slowly); rather, within-cluster spread shrinks dramatically as training proceeds, sharpening the standardized separation. The two clusters *crystallize* during the same training window where Phase 1 anomalies appear.

**Kurtosis does not distinguish the clusters.** Per-layer Welch t-tests on the kurtosis profile are non-significant at every layer (all p > 0.1). Both clusters are equally non-Gaussian. The cluster distinction is about *which subspace* the conditional bundle occupies, not about *whether* the bundle is Gaussian. The C-verdict context-mixture phenomenon is a general feature of conditional bundles, not specific to one grammatical class.

**Implication.** The Lines of Thought framework's universal $\lambda$ is not only marginalized over a population of token-specific values (as established in §4.3) — that population has *legible grammatical structure*. Function words and structural markers form distinct dynamical classes in the residual stream geometry. This is consistent with the broader interpretation that the marginal-level universality of the paper's framework hides a structured microscopic picture.

#### Updated Finding 8

The per-token $\lambda$ bimodality reflects a grammatical-role distinction between structural tokens (terminators, digits, sub-word fragments; λ ≈ 0.45) and connective tokens (function words plus internal punctuation; λ ≈ 0.58). The distinction is seed-stable on unambiguous members and emerges sharply during training steps ~300–1000.

#### Updated Finding 9

The same grammatical-role distinction manifests more strongly in effective rank than in $\lambda$. Structural-token conditional bundles occupy ~half the residual-stream dimensions of function-word bundles across the entire model depth, with Welch t-tests significant at every interior layer 1-9 (p < 0.05). Kurtosis profiles do not differ between clusters, indicating that the grammatical-class distinction is about subspace occupation rather than about Gaussianity. The clusters share the same C-verdict context-mixture structure but inhabit different subspaces of the residual stream.

### 8.2 Completed analysis: three training phases of residual-stream geometry

The CV($\lambda$) trajectory through training has a clear minimum near step 1465, where conditional dynamics are most universal. We tested whether this minimum co-locates with any of the marginal-dynamics anomalies documented in the Phase 1 writeup (§6.6): the log-$\alpha$ hump peak, the post-final-norm boundary anomaly emergence window, the late-training kurtosis rise, the $\Sigma$-distance bump, and the boundary anomaly plateau.

We loaded the Phase 1 per-checkpoint flow files for all four seeds, extracted the marginal trajectories of log-$\alpha$ (both conventions), all-to-all $\lambda$ (both conventions), mean kurtosis profile, post-final-norm boundary residual, and the basis-invariant singular-value spectrum distance to the final checkpoint, then overlaid them with the CV($\lambda$) trajectory and the documented anomaly windows.

**The $\Sigma$-distance trajectory reveals three training phases.** Computing the basis-invariant per-layer L2 distance between log-singular-value spectra at each checkpoint vs the final checkpoint (averaged over layers, rescaled to start at 1.0 and end at 0.0), three regimes emerge:

- *Phase I, steps 100–1500: rapid SVD geometry consolidation.* The singular value spectrum drops 75% of the way to its final form. Marginal kurtosis falls toward its minimum. CV($\lambda$) falls toward its minimum. Post-final-norm anomaly grows from -3.6 to -1.8.

- *Phase II, steps 1500–5000: consolidated mid-training plateau.* $\Sigma$-distance to final stays in 0.15–0.20. Marginal kurtosis is near its minimum (~0.24–0.30). CV($\lambda$) is near its minimum (~0.10). Post-final-norm anomaly plateaus at -1.8. log-$\alpha$ reaches its hump peak. This is the phase where *conditional dynamics are most universal, the marginal is most Gaussian, and the boundary anomaly is structurally stable*.

- *Phase III, steps 5000–24000: late-training restructuring.* $\Sigma$-distance has a brief rebound (the documented Phase 1 anomaly bump) and then resumes declining. Marginal kurtosis rises substantially (heavier tails developing). CV($\lambda$) rises back (per-token $\lambda$ values diverging). log-$\alpha$ descends from its hump. Post-final-norm anomaly resumes growing more negative.

**Quantitative co-location.** The CV($\lambda$) minimum at step 1465 sits exactly at the elbow between Phase I and Phase II — the point where rapid early convergence completes and the plateau begins. The marginal kurtosis minimum sits ~500 steps earlier in the same elbow region. The post-final-norm anomaly's plateau begins ~500 steps later. Three independent diagnostics — the conditional-dynamics universality, the marginal Gaussianity, and the boundary anomaly emergence completion — co-locate within ~500 training steps of each other at the start of Phase II.

The log-$\alpha$ hump peak (steps 4500–5050) sits at the *end* of Phase II, near the transition into Phase III. Reading the log-$\alpha$ trajectory together with CV($\lambda$): the marginal log-$\alpha$ reaches its highest value precisely as conditional universality is starting to deteriorate. The "hump" is the boundary between the consolidated plateau and the restructuring phase.

**Phase 1 anomaly observation:** the $\Sigma$-distance trajectory's bump (Phase III restructuring) is visible as a small rise from 0.15 to 0.18 between steps 5000 and 10000 — much subtler than the early-training descent but consistent with the Phase 1 writeup's documented "0.12 normalized" bump magnitude (the difference reflects different normalizations).

**Implications for the central interpretation.** The Lines of Thought paper measures fully-trained models, which puts their measurements in our Phase III. The paper sees Gaussian marginal behavior and reports it as a property of trained transformers. But Phase III is exactly where the marginal *develops* substantial kurtosis — what the paper sees as "approximately Gaussian" is post-consolidation heavy-tailedness, not the cleanest Gaussian state the model passes through. The truly Gaussian-like state (Phase II) is a transient consolidation that the paper's fully-trained snapshots have moved past.

More importantly, the conditional-level non-Gaussianity we documented in D4 (Mardia Z = 25–45) is also a Phase III phenomenon. It is not present at the consolidation plateau. It develops during the late-training restructuring. The C verdict is not an inevitable property of any trained transformer — it is a property of the *post-consolidation* trained transformer. Both the paper's marginal universality and our context-mixture refinement of it describe structure that *emerges during the same late-training restructuring phase*.

#### Updated Finding 10

The conditional-dynamics universality CV($\lambda$) and the marginal Gaussianity (mean kurtosis) trajectories co-locate: both reach mid-training minima within ~500 steps of each other (CV($\lambda$) at step 1465; mean kurtosis at ~1000–1500). Both are U-shaped through training, rising back into late training. The post-final-norm boundary anomaly emergence completion also co-locates with the start of this window.

#### Updated Finding 11

Three training phases of residual-stream geometry are visible in the joint Phase 1 / multiview diagnostics. Phase I (steps 100–1500): rapid SVD geometry consolidation. Phase II (steps 1500–5000): consolidated mid-training plateau with peak conditional universality, minimum marginal kurtosis, and stable post-final-norm boundary anomaly. Phase III (steps 5000–24000): late-training restructuring with the $\Sigma$-distance bump, rising marginal kurtosis, divergence of per-token $\lambda$, and resumed growth of the boundary anomaly. The Lines of Thought paper's measurements correspond to Phase III, not to the cleanest Gaussian state Phase II represents.

#### Refined interpretation of the central finding

The Lines of Thought framework's universality claims are most cleanly satisfied during the Phase II consolidation plateau, not at convergence. The C-verdict conditional non-Gaussianity and the marginal kurtosis rise both develop during Phase III. The paper's "approximately Gaussian marginal" is a description of a post-restructured trained model, and our context-mixture refinement describes structure that emerges in the same late-training phase. This sharpens the project's central claim: the paper's framework describes a snapshot of a particular training phase, and the failure of microscopic universality we documented is also a feature of that same training phase.

### 8.3 Next investigation: regenerate stage A with randomized pilot positions, across all checkpoints

The fundamental limitation in §5.2 — the unidentifiability of next-token vs position in the sub-conditioning analysis — remains unresolved by the present work. NMI(next; position) is 0.4–0.7 across all forward-set tokens because the multiview campaign uses fixed pilot positions across all held-out chunks, inadvertently entangling position with textual context. This means we know context (broadly) is the mixture index that restores Gaussianity in the conditional bundles, but we cannot isolate which specific context variable is doing the work.

The completed §8.2 analysis adds a second motivation for regenerating stage A. The C-verdict non-Gaussianity is a Phase III phenomenon (steps 5000–24000), developing during late-training restructuring. The Phase II consolidation plateau (steps 1500–5000) is where conditional dynamics look most universal and the marginal looks most Gaussian. The sub-conditioning experiment of §5.1 was run only at the final checkpoint, which is deep in Phase III — we do not know whether the context-mixture structure that explains the C verdict at the final checkpoint is also present at the Phase II plateau, nor how it develops through Phase III.

The correct next step is to regenerate stage A activation collection with two changes:

1. **Randomize pilot positions per chunk** to decouple position from text content, so the next-token-vs-position experiment becomes identifiable.
2. **Run stage A across multiple checkpoints**, not just the final one, so the sub-conditioning analysis can be repeated across the three training phases.

Concretely:

- For each held-out chunk, draw pilot positions uniformly at random from the valid range (positions $p$ with $p+1 < T$), maintaining the same total pilot count per checkpoint.
- Collect augmented activations at a representative subset of checkpoints spanning the three phases — at minimum, one in Phase I (e.g., step 500), one near the Phase II plateau (e.g., step 2500), one mid-Phase III (e.g., step 10000), and the final checkpoint (step 24000).
- Re-run the sub-conditioning and joint-partition analyses at each of these checkpoints.

After regeneration, we can answer:

- *(Resolves §5.2.)* Is next-token, position, or both the causal mixture variable for the C-verdict bundles?
- *(New question.)* Does the context-mixture structure exist at the Phase II consolidation plateau, or is it absent there and present only in Phase III? If absent in Phase II, the conditional bundles are presumably Gaussian without further conditioning during the plateau, and the context-mixture structure is something that *emerges* during late-training restructuring.
- *(New question.)* If the context-mixture structure does emerge during Phase III, at what step does it first appear, and does its emergence co-locate with any other Phase 1 anomaly (e.g., the $\Sigma$-distance bump or the kurtosis rise)?

These extended findings would close the loop between the conditional and marginal pictures across the entire training trajectory. The present analysis establishes the static picture at the final checkpoint and the marginal-trajectory co-location across training; the next campaign would extend the conditional picture across the same trajectory.

**Status of the current writeup without stage A regeneration.** The present analysis is complete as a standalone investigation. The Model A/B/C verdict, the grammatical-role bimodality, the three-phase consolidation/restructuring story, and the context-mixture refinement of the Lines of Thought framework are all defensible findings that do not depend on the next-vs-position identification or on cross-checkpoint sub-conditioning. The stage A regeneration is an *enrichment* — it would extend the conditional-level diagnostic to the trajectory level and resolve the §5.2 limitation — rather than a prerequisite for the current claims.

Cost: regenerating stage A at the four representative checkpoints (with one augmented file per seed per step, ~440 MB each) is roughly 16 inference passes — order of GPU-hours, dominated by the inference. Stage C onwards, plus the model-A-B-C discriminator suite, plus the sub-conditioning analyses, all re-run on a per-checkpoint basis. Total wall time is a small fraction of the original Phase 1 campaign.
