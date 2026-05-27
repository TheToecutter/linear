# The Forward-View Investigation: Input-Conditioned Residual-Stream Ensembles in a 150M Transformer

**A first-principles report on conditional ensemble structure beneath the Lines of Thought framework**

---

## 0. Reading guide

This report is self-contained. Readers familiar with Sarfati et al.'s "Lines of Thought in Large Language Models" (ICLR 2025) will recognize §1 as background, but everything else is internal to this investigation and assumes no prior context. Readers unfamiliar with the Lines of Thought paper should still be able to follow the report; §1 reproduces the necessary framing.

The structure is:

- **§1** reproduces the Lines of Thought paper's framing and identifies the question it leaves open about microscopic structure.
- **§2** introduces the multiview campaign — a three-way partition of the residual-stream ensemble — and motivates the forward view as the natural unit of microscopic investigation.
- **§3** states the central question of this report precisely.
- **§4** formulates three concrete candidate models for the conditional ensemble's form.
- **§5** describes the experimental setup, the model under study, and the data we collected.
- **§6** designs five discriminators that distinguish the three candidate models.
- **§7–§9** present the primary results: Models A and B are refuted, Model C is confirmed, the per-token dynamics have legible grammatical structure, and the whole picture is a Phase III training phenomenon.
- **§10** documents an interpretive wrong turn — a context-mixture interpretation that did not survive proper null controls — and the methodological lesson it teaches.
- **§11–§12** present the resolution: the conditional non-Gaussianity is driven by a small fraction of structured extreme-context pilots whose context signatures crystallize during training.
- **§13** synthesizes the full picture and answers the central question.
- **§14** lists limitations and open directions.

---

## 1. Background: Lines of Thought and its open question

### 1.1 What residual streams look like in aggregate

A transformer processes an input sequence of tokens through a stack of $L$ residual blocks. At each block boundary, every token position carries an internal vector $x \in \mathbb{R}^H$ — the **residual stream** — which is the cumulative sum of all attention and MLP contributions read by the next block. The residual stream is the primary representational substrate of the model; attention, MLP, and the final language-modeling head all read from it and write back into it.

For a fixed input chunk processed once through a trained model, the residual stream at layer $t$ at position $p$ is a specific point in $\mathbb{R}^H$. Sample many input chunks from a held-out corpus, and take many positions within each chunk, and the collection of these vectors at fixed layer $t$ forms an empirical distribution — the **residual stream ensemble at depth $t$**. We will call this the **all-to-all bundle** (it pools across all inputs and all positions) or, equivalently, the **marginal**.

The Sarfati et al. paper studies how this marginal ensemble evolves with depth. Their headline findings, paraphrased:

1. **Approximate Gaussianity at each depth.** At each layer $t$, the all-to-all bundle is approximately a high-dimensional Gaussian distribution. The deviation from Gaussianity is measured via the linearization residual $\delta x(t, \tau) = x(t+\tau) - \tilde x(t, \tau)$, where $\tilde x$ is a linear extrapolation along the all-to-all singular-vector basis; the empirical distribution of $\delta x$ is well-approximated by a multivariate Gaussian.

2. **Exponential isotropic covariance growth.** The covariance of the all-to-all bundle grows exponentially in depth, isotropically along the directions tangent to the singular vectors of the layer-to-layer mean flow:
$$\Sigma_{\text{marginal}}(t) \approx \alpha e^{\lambda t} I_{\text{tangent}}$$
with $\alpha$ and $\lambda$ measured empirically from the bundle.

3. **Universality.** The values of $\lambda$ they measure are remarkably stable across models of very different sizes and architectures (Llama-7B, Mistral-7B, etc.) and across very different corpora. This is the "universality" claim of the paper.

4. **Langevin formulation.** They model individual-token trajectories through depth using a continuous-time stochastic differential equation:
$$dx = A(t)\,x\,dt + \sigma(t)\,dW, \qquad \sigma(t)\sigma(t)^\top = \alpha e^{\lambda t} I_{\text{tangent}}$$
with the universal $(\alpha, \lambda)$ playing the role of noise process parameters.

### 1.2 What the Langevin formulation does and does not claim

The Langevin equation in (4) is a **phenomenological model of the marginal ensemble**, not a microscopic dynamical law. Transformer forward passes are deterministic given the input and weights — there is no actual stochastic process producing noise at each layer. The "noise" term $\sigma(t)\,dW$ in the SDE represents the *heterogeneity across the ensemble* (different inputs, different positions, different surrounding contexts produce different residual-stream vectors) reinterpreted as if it were a fictitious driving process.

This is a perfectly legitimate modeling move at the level of the marginal: if the marginal is approximately Gaussian and its covariance grows exponentially, then a linear-drift-plus-isotropic-noise SDE reproduces those marginal statistics. But the SDE does **not** make any claim about what happens conditional on a particular input. The paper measures, models, and discusses only marginal statistics.

### 1.3 The open question

The Lines of Thought paper establishes a clean macroscopic description: marginal Gaussianity, universal exponential covariance growth, a tidy SDE formulation. It is natural to ask whether there is a corresponding **microscopic** description — whether the macroscopic behavior reflects an underlying structure that has the same mathematical form at the level of individual conditional sub-ensembles.

There are several reasons this question is interesting independent of the paper. First, the Langevin SDE for the marginal is consistent with several mutually exclusive microscopic stories — for example, "every input produces the same Gaussian noise process" vs "different inputs produce different Gaussian noise processes that happen to average to a Gaussian" vs "no input produces a Gaussian noise process at all, but the average over inputs is Gaussian by central-limit-theorem." Each of these would have different consequences for interpretability and for what mathematical framework should be used to describe individual-input behavior.

Second, the law of total variance gives an exact decomposition of the marginal covariance into within-component and between-component pieces:
$$\Sigma_{\text{marginal}}(t) = \underbrace{\sum_i \pi_i \Sigma_i(t)}_{\text{average of conditional covariances}} + \underbrace{\sum_i \pi_i (\mu_i(t) - \mu(t))(\mu_i(t) - \mu(t))^\top}_{\text{covariance of conditional means}}$$
where the sum runs over some partition of the ensemble into sub-components, $\pi_i$ is the empirical mass of component $i$, and $\mu_i, \Sigma_i$ are the conditional moments. This is an identity; it must hold to numerical precision. The question is not whether the decomposition holds but what each piece looks like, and whether the pieces are themselves Gaussian or have some other tractable form.

Third — and most consequentially for this investigation — Sarfati et al.'s universality claim is for marginal statistics. Whether the underlying microscopic structure is also universal, or whether different conditional sub-ensembles have very different microscopic behavior that happens to wash out under marginalization, has direct bearing on the interpretation of the universality result.

This report investigates the microscopic structure for one natural choice of partition, defined in the next section.

---

## 2. The multiview campaign and the three views

### 2.1 Definitions

The all-to-all bundle pools across all inputs and all positions. A natural microscopic decomposition is to partition the bundle along an observable variable that distinguishes pilots. Three choices stand out:

- **Forward view (input-conditioned).** Pick a vocabulary token $v_i$. The forward view at $v_i$ is the sub-bundle of all pilots whose input token at the pilot position is $v_i$:
$$\mathcal{F}_i(t) = \{x_t^{(k)} : \text{input}_k = v_i\}$$
This is the natural decomposition for asking "what does the residual stream look like once we condition on the input being a specific token?"

- **Reverse view (output-conditioned).** Pick a vocabulary token $v_j$. The reverse view at $v_j$ is the sub-bundle of all pilots whose *predicted* next token (argmax of the model's output distribution) is $v_j$:
$$\mathcal{R}_j(t) = \{x_t^{(k)} : \text{argmax of model output at pilot } k = v_j\}$$
This is the natural decomposition for asking "what does the residual stream look like once we condition on the model's output decision?"

- **All-to-all view.** The full marginal, as in the Lines of Thought paper:
$$\mathcal{M}(t) = \{x_t^{(k)} : \text{all pilots } k\}$$

By the law of total variance, the marginal's first two moments decompose exactly into within-component and between-component pieces along either the forward or the reverse partition. This is bookkeeping; what makes the question interesting is the **shape** of the conditional bundles beyond their first two moments.

### 2.2 Why "forward" first

This report covers the forward view. The reverse view is the subject of a companion report. We chose to start with the forward view for several reasons:

1. **Natural causality.** The input token causes the residual stream; conditioning on the cause is the cleanest microscopic decomposition.
2. **Tractable observability.** The input token is read directly from the chunk text; no model output is needed.
3. **Vocabulary structure.** The forward view's partition labels are vocabulary tokens, which have linguistic structure (parts of speech, function vs content, punctuation vs words) that may map onto microscopic structure. Any such structure would be interesting.

### 2.3 Notation

Throughout this report, $v_i$ denotes a vocabulary token and $\mathcal{F}_i(t)$ denotes the forward view at token $v_i$ at layer $t$. $\mu_i(t)$ and $\Sigma_i(t)$ are the per-token mean and covariance. $\pi_i$ is the empirical frequency of token $v_i$ in the held-out set. $H$ is the residual-stream hidden dimension (896 for the model we study). $L$ is the number of residual-stream snapshots per forward pass (14 in our setup: the embedding output, plus the output after each of the 13 residual blocks, plus the post-final-layernorm output).

---

## 3. The central question

> **Does $p(x_t \mid v_i)$ — the forward view at token $v_i$ at layer $t$ — admit a mathematical description in the same form that Sarfati et al. apply to the marginal? If so, how does that description relate to the marginal $(\alpha, \lambda)$? If not, what is the actual form of the conditional ensemble, and what kind of structure beneath the marginal would explain the success of the Lines of Thought description at the macroscopic level?**

The cleanest possible answer would be: each conditional $p(x_t \mid v_i)$ is a Gaussian with the same covariance $\alpha e^{\lambda t} I$ as the marginal, differing across input tokens only in the drift $\mu_i(t)$. The marginal Gaussianity would then be the marginal of a structured fluid of Gaussian sub-bundles, the universality of $\alpha$ and $\lambda$ would be a microscopic property of every conditional bundle, and the Langevin SDE would lift cleanly from a marginal description to a per-token description.

We will see that this is not what happens. The empirical answer requires unpacking the candidate models more carefully, which is the subject of §4.

---

## 4. Three candidate models

We formalize three concrete candidate models for the structure of $p(x_t \mid v_i)$. The first two are clean enough to be falsifiable; the third is the residual hypothesis.

### 4.1 Model A: Gaussian conditionals with shared covariance

$$p(x_t \mid v_i) = \mathcal{N}\bigl(\mu_i(t), \Sigma_0(t)\bigr)$$

The conditional bundle at every input token is Gaussian. The covariance $\Sigma_0(t)$ is the same for every token; only the drift $\mu_i(t)$ varies. The paper's marginal universal $(\alpha, \lambda)$ live at the microscopic level: $\Sigma_0(t) = \alpha e^{\lambda t} I_{\text{tangent}}$, the same isotropic exponential covariance for every conditional.

Under Model A, the marginal's between-component piece is the empirical variance of the conditional means $\mu_i(t)$, weighted by $\pi_i$. The within-component piece is just $\Sigma_0(t)$. The marginal is therefore a mixture of Gaussians sharing covariance.

A Gaussian mixture with shared covariance and well-separated means is generally not itself Gaussian — it has multiple peaks. For the marginal to nevertheless appear Gaussian, the conditional means $\mu_i(t)$ must be packed tightly enough relative to $\Sigma_0$ that the mixture's peaks blur into one. The condition is roughly $\|\mu_i - \mu_j\| \lesssim \sqrt{\mathrm{tr}(\Sigma_0)}$ for typical token pairs.

### 4.2 Model B: Gaussian conditionals with token-dependent covariance

$$p(x_t \mid v_i) = \mathcal{N}\bigl(\mu_i(t), \Sigma_i(t)\bigr)$$

The conditional bundle at every input token is Gaussian, but the covariance $\Sigma_i(t)$ varies across tokens. The paper's marginal $(\alpha, \lambda)$ are aggregate statistics: $\alpha$ is some average of per-token $\alpha_i$, and similarly for $\lambda$.

Model B is the more permissive Gaussian story. It accommodates the observation that different input tokens probably do not produce identical residual-stream "spread" — some tokens (e.g., common function words appearing in many syntactic contexts) might be expected to produce broader bundles than others (e.g., rare specific punctuation appearing in narrow contexts) — while preserving the comforting feature that each conditional bundle is still a single Gaussian.

Under Model B, the marginal is a Gaussian mixture with both varying means and varying covariances. It can still approximate Gaussianity for the marginal if the mean and covariance variation across tokens is not too dramatic relative to the average within-token spread.

### 4.3 Model C: non-Gaussian conditionals

$$p(x_t \mid v_i) \text{ is not Gaussian for any } v_i$$

The conditional bundle is not Gaussian. The marginal Gaussianity observed by Sarfati et al. is then a **central-limit-theorem phenomenon**: the marginal is the average of $|V|$ (vocabulary-size) non-Gaussian conditional bundles, and the averaging concentrates the marginal toward Gaussianity even though no individual conditional has that form.

Under Model C, the paper's Langevin SDE is a marginal-level fiction with no microscopic counterpart. The marginal $(\alpha, \lambda)$ are emergent properties of the averaging procedure, not microscopic constants.

### 4.4 What can distinguish them

Models A, B, and C make different predictions about observable statistics of the forward view:

- **Cross-token variation in within-input total variance ($\mathrm{tr}\,\Sigma_i$).** Model A: zero (modulo finite-sample noise). Model B and C: non-zero.
- **Cross-token variation in conditional SVD basis.** Model A: zero (all conditional bundles span the same subspace). Model B and C: non-zero.
- **Cross-token variation in per-token exponential growth rate $\lambda_i$.** Model A: zero. Model B: variable, but the average matches the paper's marginal $\lambda$. Model C: indeterminate (the exponential-growth fit may not even be a good description of any individual conditional).
- **Multivariate Gaussianity of a conditional bundle.** Model A and B: passes Gaussianity tests. Model C: fails.
- **GMM reconstruction of the marginal from per-token Gaussian fits.** Model A and B: reproduces the marginal's higher-order statistics. Model C: produces a markedly more Gaussian object than the empirical marginal.

These give us the design space for discriminators, addressed in §6.

---

## 5. Experimental setup

### 5.1 The model

We trained a 150-million-parameter Llama-style transformer from scratch for the express purpose of this investigation, in order to have access to all checkpoints, all internal states, and full control over hyperparameters and training data. The architecture is a standard pre-norm Llama variant with hidden size $H = 896$, 13 transformer blocks (residual layers), 14 attention heads with head dimension 64, MLP intermediate size 2401 (giving the 8/3 expansion ratio characteristic of Llama with the SwiGLU adjustment), rotary positional embeddings, and the Mistral-7B-v0.1 tokenizer (vocabulary size 32000).

The model is trained on a 1.57-billion-token subset of FineWeb-Edu, packed into 1024-token chunks (~10.7 million training chunks total), with the standard causal language-modeling objective. We use AdamW, cosine-annealed learning rate starting at 3e-4, warmup over the first 1000 steps, gradient clipping at 1.0, and a batch size that gives effective tokens-per-step in the 65k range. Training runs for 24,000 optimizer steps.

We train **four random seeds**, identical except for the PRNG seed driving weight initialization and data shuffling. Saving four runs gives us cross-seed variance estimates for every measurement.

### 5.2 Checkpointing schedule

We save **50 checkpoints per seed** at logarithmically spaced training steps from step 100 to step 24,000. The log spacing concentrates resolution in the early training period where rapid geometry change is expected, with sparser sampling in late training where the geometry stabilizes. The 50 checkpoint steps for seed 0 are (with minor variation across seeds due to discretization):
100, 113, 129, 146, 167, ..., 5,000, 5,700, ..., 24,000.

Four specific checkpoints play recurring roles in the analysis as **representative of distinct training phases** (see §9 for the phase structure):

- **Step 479** — early training, "Phase I"
- **Step 2,563** — mid-training plateau, "Phase II"
- **Step 9,809** — mid-late training, "Phase III mid"
- **Step 24,000** — final, "Phase III final"

### 5.3 Held-out evaluation set

The held-out set is 500 chunks of 1024 tokens each (~500k tokens) drawn from FineWeb-Edu, disjoint from the training set, with a fixed random seed shared across all four training seeds so that every (seed, checkpoint) measurement uses the same evaluation chunks.

### 5.4 Pilot sampling

A **pilot** is a (chunk, position) pair where we record the residual-stream vector at each of the 14 layer snapshots, together with metadata. By default we sample 20 fixed positions per chunk (positions 50, 100, 150, ..., 1000), giving ~10,000 pilots per (seed, checkpoint). For one specific follow-up experiment we use randomized positions instead of the fixed positions; this is documented when it matters.

For each pilot we save:

- The 14 residual-stream vectors of dimension $H = 896$ each (one per layer snapshot).
- The input token id at the pilot position.
- The actual next token id at position pilot+1 in the chunk.
- The model's predicted next token id (argmax of the output distribution at the pilot position).
- The pilot position within its chunk.
- For one specialized follow-up file, the previous token id at position pilot-1.

These are stored as compressed NPZ files, one per (seed, checkpoint). The pilots are partitioned in software at analysis time according to whichever view (forward, reverse, all-to-all) and conditioning variable is needed.

### 5.5 The frozen forward set

The forward view requires that some specific tokens are designated for per-token analysis. We compute the top-20 most frequent input tokens in the held-out set and freeze this set as the **frozen forward set**. Per-token pilot counts in this set range from ~50 (for the lowest-frequency tokens in the top-20) to ~400 (for the highest-frequency tokens like `the`, `,`, `.`). The frozen forward set is the same across all four seeds and all 50 checkpoints, by construction.

Decoded with the Mistral tokenizer, the frozen forward set contains the following 20 tokens (ordered by frequency rank, approximately): `the`, `,`, `\n`, `.`, `of`, `and`, `to`, `a`, `in`, `is`, `that`, `are`, `for`, `'`, `s`, `0`, `1`, `2`, `(`, `-`, plus the BPE space marker that the tokenizer uses as a prefix on some encoded forms.

### 5.6 The multiview pipeline

At each (seed, checkpoint), the pipeline runs three stages:

- **Stage A**: load model weights, forward-pass the held-out chunks, collect pilots into the augmented NPZ file described above.
- **Stage B**: load the augmented file, partition the pilots into the forward, reverse, and all-to-all views, and compute per-view-per-token covariances, means, and standard scalar diagnostics (effective rank, kurtosis profile, etc.).
- **Stage C**: combine the Stage B outputs into a `MultiViewResult` per (seed, checkpoint), saved as `multi_view_step_NNNNNNNN.npz`. This is the basic analysis-ready unit.

The total data footprint is ~30 GB per seed for all 50 checkpoints.

### 5.7 What "kurtosis" means in this report

We use **excess kurtosis** throughout, defined as $\kappa = \mathbb{E}[(x - \mu)^4] / \sigma^4 - 3$ for a univariate distribution. Multivariate kurtosis comes in several forms; we use two:

- **D4a — per-coordinate excess kurtosis in a chosen basis.** Project each pilot into a basis, compute univariate excess kurtosis along each basis direction, take the mean across coordinates. The basis varies across analyses; we will be explicit when reporting numbers.
- **D4b — Mardia's multivariate kurtosis Z-score.** A standard multivariate normality test based on the fourth Mahalanobis moment. Z-values above ~2 reject Gaussianity at the 5% level.

A point on conventions: the absolute kurtosis numbers depend on the basis. Two different "conventions" appear in this report:

- The **per-token-SVD basis convention** uses each input token's own SVD basis for its own conditional bundle. This is the most natural choice for per-token analysis and is what the multiview pipeline's standard outputs use. Kurtosis numbers in this convention are typically in the 1-3 range at interior layers for our model.
- The **shared-PCA convention** uses a single 32-dimensional PCA basis computed from the full pilot pool, applied to every input token. This is the natural choice for cross-token comparison and is what the follow-up experiments (D11-D16) use. Kurtosis numbers in this convention are typically in the 4-8 range at interior layers for the same data.

The difference reflects how heavily the basis aligns with the heaviest tails. We will be explicit about which convention is in use whenever absolute numbers matter.

---

## 6. Experimental design: five discriminators

We designed five discriminators D1-D5 to distinguish Models A, B, and C, each addressing a different empirical signature.

### 6.1 D1 — Cross-token CV of within-input total variance

For each input token in the frozen forward set, compute $\mathrm{tr}(\Sigma_i(t))$ at each layer $t$. Under Model A, this quantity is the same for every token at fixed $t$ (modulo finite-sample noise). Under Model B or C, it varies. We report the coefficient of variation (CV = std / mean) across the 20 frozen-forward-set tokens at each layer.

### 6.2 D2 — Principal angles between conditional SVD bases

For each input token, compute the top-$k$ singular vectors of $\Sigma_i(t)$. Under Model A with shared $\Sigma_0$, all conditional bundles span the same top-$k$ subspace, so the principal angles between any two tokens' bases are zero (modulo finite-sample noise). Under Model B and C, they are non-zero.

A subtle issue with D2 is that finite-sample SVD estimates are noisy: even when two distributions have the same true covariance, their empirical top-$k$ bases at finite $n$ will differ by some non-zero angle. We must calibrate the noise floor by bootstrap resampling (see §6.7).

### 6.3 D3 — Per-token $(\alpha_i, \lambda_i)$ fits

For each input token, apply the same exponential-growth fitting procedure that Sarfati et al. apply to the all-to-all bundle: fit $\mathrm{tr}(\Sigma_i(t)) \approx \alpha_i e^{\lambda_i t}$ across layers $t$ in some interior range. Under Model A, $\alpha_i$ and $\lambda_i$ are the same for every token and equal to the marginal values. Under Model B, they vary across tokens. We report the cross-token CV of $\lambda$ and visualize the distribution of per-token $\lambda$ values.

### 6.4 D4 — Multivariate Gaussianity of conditional bundles

Under Model A and B, every conditional bundle is Gaussian; under Model C, none of them are. We test in two ways:

- **D4a** — for each input token, project the conditional bundle into a 32-PC subspace (using either the per-token basis or the shared basis as discussed in §5.7), compute the mean per-coordinate excess kurtosis, and report it per layer.
- **D4b** — for each input token, compute Mardia's multivariate kurtosis Z-score in the shared 32-PC subspace.

### 6.5 D5 — GMM reconstruction

Take the empirically measured per-token means and covariances $\hat\mu_i, \hat\Sigma_i$ from the forward view. Construct a Gaussian mixture by sampling, at each layer, $N$ synthetic pilots: for pilot $k$, draw a token index $i$ from the empirical frequencies $\pi_i$, then sample $x_t^{(k)} \sim \mathcal{N}(\hat\mu_i(t), \hat\Sigma_i(t))$. Compute the kurtosis of this synthetic marginal and compare to the kurtosis of the empirical marginal.

Under Model A and B, the per-token Gaussian fits are accurate and the synthetic marginal reproduces the empirical marginal's higher-order statistics. Under Model C, the Gaussian fits are wrong (they capture only the first two moments of a non-Gaussian object), and the synthetic marginal is markedly more Gaussian than the empirical marginal.

### 6.6 Why these particular five

These five discriminators were chosen to be (a) independent — they rely on different aspects of the data, so agreement among them is meaningful; (b) interpretable — each maps onto a specific feature of the candidate models; and (c) computable at our sample sizes. D2 in particular required careful sample-size handling as discussed in §6.7.

### 6.7 Bootstrap noise-floor calibration

Several of the discriminators (notably D1, D2, D3) require numerical thresholds to convert measurements into verdicts. We measured the finite-sample noise floor for each discriminator at the actual pilot counts we use, by bootstrap resampling within each token (B = 100 replicates per token at seed 0, step 24000) and observing the empirical spread of the discriminator output under within-bundle resampling. We also cross-checked against the cross-seed standard deviation of each discriminator.

The calibration produced these thresholds (anything significantly above the threshold is a real signal):

- D1: CV($\mathrm{tr}\,\Sigma_i$) noise floor ≈ 0.115 (after trimming the newline-token outlier whose covariance is anomalously large; see §7.1).
- D3: CV($\lambda$) noise floor ≈ 0.062.
- D4b: Mardia Z noise floor ≈ 0.234.

The D2 calibration produced an unexpected and important result: at our pilot counts (~150–400 per input token in the frozen forward set), the **top-10 principal directions are sample-size-degenerate**. Even when two bundles are drawn from the same population, the empirical mean self-pair principal angle in the top-10 subspace is 35.9° out of a possible 90°. This is far from zero. The angles only become reliably small in the top-2 subspace, where the bootstrap noise floor is ~10°. We therefore restricted D2 to the top-2 directions and demoted it from a primary verdict driver to a confirmatory diagnostic.

This is a methodological point worth flagging: principal-subspace discriminators in high dimensions require bootstrap-floor calibration before use. Naive comparisons against 90° will overstate dissimilarity at finite sample sizes.

---

## 7. Primary results: Models A and B are refuted

We report the discriminator outcomes at the final checkpoint (step 24,000), seed 0, in the standard multiview convention. Cross-seed consistency is documented at the end of this section.

### 7.1 D1 — within-input variance is strongly token-dependent

Cross-token CV of $\mathrm{tr}(\Sigma_i(t))$ at the final checkpoint sits in the range **0.6–0.8 across every interior layer**, with a spike to 1.7 at layer 1.

The threshold is 0.115. The signal exceeds threshold by **5–7×**.

The newline token (`\n`, id 13) is an outlier in absolute variance: its $\mathrm{tr}(\Sigma_i)$ is roughly 5–10× that of the other forward-set tokens. We exclude it from the trimmed CV calculation (which is what the 0.115 threshold and 0.6–0.8 numbers refer to); the untrimmed CV would be even larger.

Looking at concrete numbers: at layer 5, the trace values across the frozen forward set range from ~3 (for `(`, the smallest) to ~12 (for `,`, the largest) in the model's natural units, a factor of 4× spread. Different input tokens produce within-input bundles whose total variance differs by factors of 2–4× routinely. The shared-$\Sigma_0$ picture is qualitatively wrong.

**Conclusion: Model A is refuted by D1.**

### 7.2 D2 — bundle subspaces are different (but the diagnostic is partly sample-size-limited)

In the top-2 subspace (where the bootstrap floor is reliable), principal angles between conditional bundles' subspaces are routinely 30–60°, vs the bootstrap floor of ~10°. The bases are clearly different.

This is consistent with Model B and C and inconsistent with Model A.

### 7.3 D3 — $\lambda$ is not universal

Fitting $\mathrm{tr}(\Sigma_i(t)) \approx \alpha_i e^{\lambda_i t}$ across interior layers per token, we find at the final checkpoint:

- The all-to-all (marginal) $\lambda$ is **0.36**.
- The per-token $\lambda$ values for the frozen forward set range from **0.40 to 0.62**, with a clear bimodal distribution: most tokens cluster around $\lambda \approx 0.55$–$0.60$, with a separated lower cluster around $\lambda \approx 0.40$–$0.48$.
- The all-to-all $\lambda$ sits **below every per-token value**. This is geometrically sensible: the marginal trace grows at the rate of its slowest-growing component (the eigendirection along which most of the trace mass lies grows slowest), which is the within-token component in our case.

Cross-token CV of $\lambda$ is **0.14**, above the 0.062 noise floor.

The bimodal structure of per-token $\lambda$ — not just variable values but a clear cluster split — is interesting enough that we devote §8 to characterizing it.

**Conclusion: Model A is further refuted by D3. Model B remains possible at this stage.**

### 7.4 D4 — conditional bundles are not Gaussian

D4a per-coordinate kurtosis (in the per-token-SVD-basis convention) for the forward-set conditional bundles at interior layers ranges from 1.5 to 6, vs the all-to-all marginal kurtosis of 0–0.5 in the same convention. **The conditional bundles are markedly more non-Gaussian than the marginal.**

D4b Mardia multivariate kurtosis Z-scores (in the shared 32-PC subspace) are in the range **25–45 across interior layers**, vs the rejection threshold of |Z| = 2 at the 5% level. **Multivariate Gaussianity is rejected at overwhelming significance for every interior conditional bundle.**

**Conclusion: Model B is refuted by D4.**

### 7.5 D5 — GMM reconstruction confirms

Synthetic marginals from per-token Gaussian fits:

- **Empirical marginal kurtosis** (shared-PCA convention, interior layers 1–7): 5 to 15.
- **Model A reconstruction** (shared $\hat\Sigma_0$ across tokens, varying $\hat\mu_i$): ~0.
- **Model B reconstruction** (per-token $\hat\Sigma_i$, varying $\hat\mu_i$): 0.3 to 0.7.

Both Gaussian-mixture reconstructions produce near-Gaussian marginals; the empirical marginal has 5–15× more kurtosis. **Gaussian fits to the conditional bundles cannot reproduce the marginal's higher-order structure.**

**Conclusion: Model C is the survivor.**

### 7.6 The verdict, per-layer and per-seed

We can summarize per layer, taking the disjunction of "any of D1, D3, D4 exceeds threshold" to indicate the verdict:

| Layer t   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|-----------|---|---|---|---|---|---|---|---|---|---|----|----|----|----|
| Verdict   | B | C | C | C | C | C | C | C | C | C | C  | C  | C  | C  |

The verdict is "C" at every interior layer 1–13.

Layer 0 is the raw embedding output, where for fixed input token every pilot has the same residual vector by definition. The discriminator there is structurally degenerate: there is no within-token variance to compare. The "B" tag at layer 0 reflects this degeneracy and should be read as "not informative" rather than as evidence against Model C at that layer.

The same pattern obtains at every seed: all four seeds give 13 of 14 layers tagged "C", with layer 0 the only exception. Cross-seed agreement is exact.

### 7.7 Statement of the C verdict

The marginal Gaussian observed by Sarfati et al. for our 150M model is **not** decomposable as a Gaussian mixture over input-token-conditioned components. The conditional bundles have token-dependent variance (D1), token-dependent exponential growth rate (D3), are themselves substantially non-Gaussian (D4), and aggregating any Gaussian fit of them does not reproduce the empirical marginal's higher-order structure (D5).

The macroscopic Gaussianity is therefore a central-limit-theorem phenomenon over a population of structurally distinct, non-Gaussian conditional bundles. The paper's Langevin SDE is a marginal-level description with no direct microscopic counterpart in the form predicted by Models A or B.

This is the headline result. §8–§13 unpack what the actual microscopic structure looks like.

---

## 8. The per-token $\lambda$ has legible grammatical structure

The bimodal distribution of per-token $\lambda$ values from §7.3 is striking enough to deserve its own characterization. We ran k-means (k=2) on the per-token $\lambda$ vector averaged across seeds at the final checkpoint, decoded the cluster members using the Mistral tokenizer, tested cross-seed cluster stability, and applied Welch's t-tests at each layer comparing the two clusters on additional summary statistics.

### 8.1 Cluster membership

The two clusters correspond to a recognizable grammatical distinction.

- **Low cluster** ($\lambda \approx 0.45$, 8 tokens): `\n`, `.`, `-`, `0`, `1`, `2`, `s`, and the BPE space marker. These are **structural/sub-word tokens**: terminators (newline, period), separators (hyphen), digits, and sub-lexical fragments (the orphan `s` likely arises from possessives and contractions; the BPE space marker is a token-internal whitespace artifact of the BPE encoding).

- **High cluster** ($\lambda \approx 0.58$, 12 tokens): `the`, `of`, `and`, `to`, `a`, `in`, `is`, `that`, `are`, `for`, `,`, `(`. These are **connective tokens**: function words plus the within-sentence punctuation (comma, open parenthesis) that operates syntactically as a connector rather than as a terminator.

The split is not "punctuation vs words" or "content vs function" in any naive sense. The comma is in the high cluster (it connects within a sentence) while the period is in the low cluster (it terminates a sentence). The open parenthesis is in the high cluster; the close parenthesis is not in the frozen forward set but would presumably be in the low cluster.

The grammatical reading: the distinction is between tokens that **terminate or fragment** a syntactic unit (low cluster) and tokens that **connect within** a syntactic unit (high cluster).

### 8.2 Cross-seed stability

17 of 20 tokens have stable cluster assignment across all four seeds. The 3 unstable tokens (`.`, `s`, and the BPE space marker) sit right at the cluster boundary at $\lambda \approx 0.48$–$0.50$. Unambiguous cluster members (the digits, `\n`, `-` in the low cluster; all 12 function-word tokens in the high cluster) are stable across all seeds.

The bimodality is a real structural distinction with a small fuzzy boundary region, not a seed-specific artifact.

### 8.3 Effective rank is the sharper discriminator

Welch t-tests per layer on additional summary statistics reveal that **the cluster distinction is much sharper in effective rank than in $\lambda$**. Effective rank (the participation ratio of the bundle's eigenvalues) measures how many directions the conditional bundle effectively spans.

At layer 1: low-cluster mean effective rank 6.4 vs high-cluster 19.8 ($p < 10^{-4}$).
At layer 5: 37.7 vs 63.7 ($p \approx 0.017$).
The effective rank Welch test is significant at $p < 0.05$ for every interior layer 1–9.

**Structural-token conditional bundles occupy roughly half the residual-stream dimensions that function-word bundles occupy across the entire model depth.**

Geometrically this is intuitive. Lower $\lambda$ and lower effective rank are likely two views of the same phenomenon: a bundle confined to a lower-dimensional subspace cannot expand its trace as rapidly because there are fewer directions of expansion available.

### 8.4 Kurtosis is not class-discriminative (but heavy-tailedness is — see §12)

Per-layer Welch t-tests on the kurtosis profile by cluster are non-significant at every layer (all $p > 0.1$). Both clusters are equally non-Gaussian, when averaged across all pilots within a class.

We will see in §12 that this averaged measure hides a real class distinction: the *severity* of conditional heavy-tailedness, measured via the best-fit degrees of freedom of a multivariate t-distribution, is grammatical-class-dependent. The structural-token bundles have substantially heavier tails than the function-word bundles. The §8.4 negative result is for the per-coordinate kurtosis specifically; the §12 positive result is for the multivariate t-fit, which is a more powerful test of heavy-tailedness shape.

### 8.5 Training-time emergence

The bimodality emerges sharply between training steps **300 and 1000** and continues to tighten through step 24,000. Cohen's $d$ (standardized cluster separation in $\lambda$) is essentially zero through step 200, climbs from step 300, reaches $d \approx 1.5$ by step 1000, $d \approx 3$ by step 10,000, $d \approx 4$ by step 24,000.

The clusters do not separate by centroid drift; rather, within-cluster spread shrinks as training proceeds, sharpening the standardized separation. **The grammatical clusters crystallize during training.**

### 8.6 Implication

The Lines of Thought framework's universal $\lambda$ is not only marginalized over a population of token-specific values (as established in §7.3) — that population has **legible grammatical structure**. Function words and structural markers form distinct dynamical classes in the residual-stream geometry. The bulk of the marginal $\lambda$ value comes from the larger function-word cluster (12 tokens at $\lambda \approx 0.58$); structural tokens are a smaller minority cluster (8 tokens at $\lambda \approx 0.45$) that pulls the marginal value down.

If the paper's universal $\lambda$ value reflects an underlying microscopic universality, our data suggests that universality is at the level of *grammatical token class* rather than at the level of individual tokens. Whether the structural-cluster $\lambda$ and the function-word-cluster $\lambda$ are themselves universal across models is an interesting open question.

---

## 9. Three training phases of residual-stream geometry

The CV($\lambda$) trajectory has a mid-training **minimum** near step 1500. This is unexpected if one thinks of training as monotonically converging — it means per-token $\lambda$ values are at their *most universal* at mid-training, then diverge again in late training. We loaded all per-checkpoint flow files for all four seeds and overlaid their marginal trajectories, looking for whether other diagnostics co-locate with the CV($\lambda$) minimum.

They do. Three independent diagnostics co-locate within ~500 training steps at the elbow between two distinct training regimes, and a separate transition occurs around step 5000.

### 9.1 The three phases

- **Phase I, steps 100–1500 — rapid SVD geometry consolidation.** The singular value spectrum of the all-to-all bundle drops 75% of the way to its final form. Marginal kurtosis falls toward its minimum. CV($\lambda$) falls toward its minimum. Post-final-layernorm boundary residual (the deviation of the post-final-norm bundle from the pre-final-norm bundle, a quantity that has a structured trajectory we have characterized elsewhere) grows from $-3.6$ to $-1.8$. Per-token $\lambda$ clusters have not yet crystallized (§8.5).

- **Phase II, steps 1500–5000 — consolidated mid-training plateau.** The basis-invariant singular-value spectrum distance from each checkpoint to the final checkpoint stays in 0.15–0.20 (slowly declining). Marginal kurtosis is at its minimum (~0.24–0.30). CV($\lambda$) is at its minimum (~0.10). Post-final-norm boundary residual plateaus at $-1.8$. Per-token log-$\alpha$ reaches its hump peak (the absolute amplitude of the exponential covariance growth is at its largest in this window).

- **Phase III, steps 5000–24000 — late-training restructuring.** The basis-invariant distance to final shows a brief rebound and then resumes declining. Marginal kurtosis rises substantially (heavier tails develop). CV($\lambda$) rises back (per-token $\lambda$ values diverging into clusters). Per-token log-$\alpha$ descends from its hump. Post-final-norm boundary residual resumes growing more negative. Grammatical clusters crystallize (§8). Structured extreme-context tails crystallize (§12).

### 9.2 Quantitative co-location of phase transitions

The CV($\lambda$) minimum at step 1465 sits exactly at the elbow between Phase I and Phase II. The marginal kurtosis minimum is ~500 steps earlier in the same elbow region. The post-final-norm boundary residual's plateau begins ~500 steps later. **Three independent diagnostics — conditional-dynamics universality, marginal Gaussianity, and boundary-anomaly emergence completion — co-locate within ~500 training steps at the Phase I/II elbow.**

The Phase II/III transition is less crisp temporally — Phase III is a long restructuring rather than a sudden change — but the rebound in the basis-invariant distance to final, and the onset of CV($\lambda$) increase, mark its beginning around step 5000.

### 9.3 What "Phase III" means for the Lines of Thought framework

The Lines of Thought paper measures fully-trained models. Their measurements fall in our Phase III — the late-training restructuring phase. **The paper's "approximately Gaussian marginal" describes a post-restructuring state, not the cleanest Gaussian-like state the model passes through.** Phase II is when the marginal is *most* Gaussian (lowest kurtosis), the per-token $\lambda$ values are *most* universal (lowest CV), and the boundary anomaly is at its plateau. Phase II is also short (steps 1500–5000), so any paper measuring fully-trained models will miss it.

The Model C conditional non-Gaussianity from §7 is *also* a Phase III phenomenon. Conditional bundles are nearly Gaussian during Phase II (mean conditional kurtosis ~0.5 in the shared-PCA convention) and develop their heavy tails progressively through Phase III (reaching ~5–8 by step 24,000). The Model C verdict is not an inevitable property of a trained transformer; it is a property of the post-consolidation trained transformer, alongside the rise in marginal kurtosis and the divergence of per-token $\lambda$ values.

### 9.4 The discriminator outputs at each phase

For completeness, we re-ran the D1, D3, D4a discriminators at the four representative checkpoints across all four seeds. The per-layer verdict pattern is **identical at all four phases**: `B C C C C C C C C C C C C C` (B at layer 0, C at all interior layers). The C verdict holds throughout training, not just at the end.

What *does* vary across phases is the **magnitude** of the underlying signals. Conditional kurtosis at layer 7 (shared-PCA convention) is ~0.1 at Phase I, ~0.5 at Phase II, ~2.8 at Phase III mid, and ~5.9 at Phase III final. The C verdict is qualitative; the quantitative severity of the non-Gaussianity is what emerges during training.

---

## 10. An interpretive wrong turn: the context-mixture hypothesis and its failure under null correction

This section documents an interpretive thread that did not survive scrutiny. We include it because (a) it explains where some early-draft claims came from, (b) the methodological lesson generalizes to any sample-partitioning analysis of higher moments, and (c) it motivates §11–§12 by explaining what is *not* the explanation for the Model C non-Gaussianity.

Readers interested only in the final answer can skip to §11; this section's key takeaway is "single-variable mixture interpretations of the C verdict do not survive proper null controls."

### 10.1 The provisional hypothesis

The natural first interpretation of Model C is that $p(x_t \mid v_i)$ is itself a mixture — over the local context in which the input token appears. The same token `the` preceded by `of` or by a period sits in different syntactic and semantic contexts; aggregating across contexts produces heavy tails and non-Gaussian shape even if each context-specific sub-bundle is individually Gaussian.

This hypothesis is testable. The augmented activation files record per pilot the input token, the actual next token in text, the model's predicted next token, and the pilot's position in the chunk. If the C verdict reflects context mixture, then sub-conditioning on a sufficient context variable should "collapse" each forward-view bundle to a collection of Gaussian sub-bundles, observable as a dramatic reduction in conditional kurtosis.

### 10.2 First-round sub-conditioning result

We partitioned each input-conditioned bundle three ways and recomputed aggregate kurtosis on each sub-bundle: by next-token in text, by predicted-token (argmax of the model's output), and by pilot position. We required at least 20 pilots per sub-bundle to compute its kurtosis meaningfully, and computed a sample-size-weighted mean across surviving sub-bundles.

The first-round numbers looked excellent. Sub-conditioning on next-token appeared to reduce aggregate conditional kurtosis from baseline ~2.5 at layer 1 to ~0.3, and from ~1.0 at layer 5 to ~0.1. For the newline token specifically, baseline kurtosis 7.66 at layer 7 (per-token-SVD basis) appeared to collapse to 0.02 under next-token sub-conditioning.

The numbers seemed to confirm the context-mixture hypothesis with striking strength. Position sub-conditioning gave comparable reductions. Predicted-token sub-conditioning gave smaller but still substantial reductions. We initially concluded that the C verdict reflected context mixture identifiable up to the joint (next, position) distribution.

### 10.3 The null-control reckoning

Before publishing the result, we asked the question that should have been asked from the start: **how much of the apparent partition effect is sample-partitioning artifact?**

Kurtosis is a fourth-moment statistic with a well-known downward bias under sample-size reduction. Splitting a heavy-tailed sample into $k$ subgroups and computing within-subgroup kurtosis systematically underestimates the original kurtosis, *even when the subgroups are formed by random labeling rather than by any meaningful partition*. The bias scales with $k$: more subgroups means more reduction.

The right diagnostic is a **matched random-labels null**. For each input token, assign random integer labels in $\{0, 1, \ldots, k-1\}$ to its pilots, compute the same weighted-mean kurtosis using these random labels as the partition, average over many random-label replicates, and compare to the kurtosis under the *real* partition at the same $k$.

The **real signal** of a partition variable is the kurtosis reduction beyond what random labels at the same $k$ achieve.

### 10.4 The position null control

We swept the position bin count $k \in \{2, 3, 5, 10, 20, 30\}$ at all four representative checkpoints, with 10 independent random labelings per $k$. The result at Phase III final, layer 7 (shared-PCA convention), with baseline kurtosis 5.91:

| Bin count $k$ | Quantile binning reduction | Random null reduction | Real signal |
|---|---|---|---|
| 2 | 3.56 | 2.77 | **0.79** |
| 3 | 4.50 | 3.62 | **0.88** |
| 5 | 5.13 | 4.63 | 0.50 |
| 10 | 5.42 | 5.24 | 0.18 |
| 20 | 5.56 | 5.53 | 0.03 |
| 30 | 5.67 | 5.59 | 0.09 |

The real position signal **peaks at $k=3$ around 0.88, then declines** as we add more bins because the random null catches up. By $k=20$, **96% of the apparent position effect is sample-partitioning artifact** — only ~0.03 of the original 5.56 reduction is real.

Position carries a small real signal at very coarse resolution (~14% of baseline kurtosis explained), and essentially nothing at fine resolution.

### 10.5 The next-token null control

Next-token's effective bin count varies per input token; the average across the frozen forward set is ~4 valid next-token sub-bundles per input token (at min-subbundle = 20). The matched random null at $k=4$, averaged over 20 replicates:

| Phase | Baseline | Next reduction | Null reduction | Real signal |
|---|---|---|---|---|
| Phase I | 0.20 | 0.29 | 0.07 | +0.22 |
| Phase II | 0.55 | 0.11 | 0.17 | −0.06 |
| Phase III mid | 2.32 | 1.15 | 1.36 | −0.21 |
| Phase III final | 5.78 | 4.27 | 4.02 | +0.25 |

The real signal **oscillates sign across training phases** and never exceeds 0.3.

A measured "real effect" that oscillates sign through training is not a real effect — it is noise around zero. Next-token, after null correction, carries no signal we can robustly distinguish from zero in this analysis.

### 10.6 Retraction

The first-round sub-conditioning result (§10.2) was almost entirely a sample-partitioning artifact. We retract the following claims that appeared in early drafts of this investigation:

1. That the Model C non-Gaussianity is decomposable as a context mixture identifiable along next-token or position.
2. That next-token sub-conditioning collapses the conditional bundle to Gaussian sub-bundles.
3. That position is "the natural mixture coordinate" for the C verdict.

After null correction, what survives is:

1. The Model C verdict from §7, which uses multivariate Gaussianity diagnostics that do not depend on sample partitioning.
2. The §8 grammatical bimodality, which is also partition-independent.
3. The §9 three-phase training trajectory, also partition-independent.
4. A small (~14% of baseline kurtosis), real, coarse-position signal. This is genuine but a much smaller claim than was made before.

### 10.7 Methodological lesson

**Kurtosis-reduction sub-conditioning analyses must include a matched random-labels null control.** Without it, any partition variable will appear to "explain" a portion of the heavy tails roughly proportional to its bin count, even when the variable carries no actual information about the distribution. The same lesson applies to any analysis that estimates higher moments after sample partitioning.

A related lesson concerns the pilot sampling scheme. Our default fixed-position scheme (20 evenly spaced positions per chunk) inadvertently entangles position with surrounding-context statistics: position 50 is almost always near the start of a chunk and tends to follow specific tokens; position 1000 is near the end and follows different tokens. This entanglement makes single-variable position and next-token partitions hard to interpret as causally distinct. We addressed this in a follow-up by regenerating pilots with randomized positions; that follow-up confirmed the §10.4–10.5 numerical conclusions and reinforced the §10.6 retraction.

---

## 11. Joint bigram partition: Possibility 3 is structurally untestable

If single-variable context partitions do not explain the heavy tails, the next natural candidate is a **joint bigram partition** over (prev_token, next_token). The C verdict might reflect a mixture indexed by local bigram context that any single-variable partition would fail to capture.

We extended the stage A pipeline at the four representative checkpoints to also save prev_token alongside next_token (the original augmented files saved input, next, predicted, and position, but not previous). This produced a separate set of `augmented_step_NNNNNNNN_ngram.npz` files containing the additional prev_ids field.

We then ran joint (prev, next) sub-conditioning with proper matched random-labels nulls, restricted to high-coverage input tokens (those with $\geq 200$ pilots) at min-subbundle = 15, on the four representative checkpoints.

### 11.1 The result

**The joint partition is not computable at any sample size we have access to.**

Of the four input tokens passing the coverage filter at every phase (these are the most frequent forward-set tokens: `the`, `,`, `.`, `of`), **zero have any (prev, next) joint cell with $\geq 15$ pilots**. The marginal next-token partition produces 2 valid sub-bundles for 2 of those tokens. The marginal prev-token partition produces 4 valid sub-bundles for 1 token. The joint partition produces zero valid sub-bundles at any phase.

### 11.2 The structural reason

The reason is the vocabulary-squared joint space. For an input token $v_i$ with 200–400 pilots in our held-out set, the empirical distribution of (prev_token, next_token) pairs is concentrated on a long tail of singletons. Even the most common bigram around any given input token does not reach 15 occurrences.

Increasing pilot counts by an order of magnitude (or more) per input token would be necessary to make this partition computable. That is beyond the scope of the current dataset. Alternatively, the joint variable could be coarsened by clustering tokens into grammatical classes before partitioning, but this would introduce a new step whose own validity would need calibration.

### 11.3 What this tells us

The reduced individual-partition results (next-token real signal ~0.56 at Phase III final from 2 contributing tokens; prev-token real signal ~0.37 from 1 token, after coverage filtering) are too fragile to support meaningful interpretation beyond the small partial signals already documented in §10.

**Possibility 3 — that the conditional non-Gaussianity reflects a learned bigram-context mixture — cannot be ruled out, but cannot be directly tested with this dataset.** The next-token and prev-token partial signals leave room for some bigram structure to exist, but the bigram partition itself is uncomputable. We will see in §12 that an indirect approach to characterizing the local context of the heavy-tail-driving pilots is possible, and it yields legible bigram-related structure even though the formal partition analysis is structurally blocked.

---

## 12. The resolution: structured rare extreme contexts

After §10–§11 ruled out single-variable Gaussian-mixture explanations and showed the joint bigram partition to be structurally untestable, we tested an alternative hypothesis: **the C verdict heavy tails are driven by a small fraction of extreme-context pilots, not by any tractable Gaussian-mixture decomposition.**

Under this hypothesis, most pilots in a conditional bundle live in a Gaussian core, and a small fraction (~5%) of pilots in unusual contexts produce essentially all of the heavy-tail signature. The conditional bundle is then "Gaussian core plus a sparse extreme tail," and the extreme tail is too small a fraction of the data to be picked up by partition-style analyses that require many pilots per sub-bundle.

We tested this via three complementary analyses: trimmed kurtosis (§12.1), multivariate t-distribution fit (§12.2), and direct characterization of the extreme pilots' contexts (§12.3).

### 12.1 Trimmed kurtosis (D12)

For each input token in the frozen forward set at each of the four representative checkpoints, we computed per-pilot Mahalanobis distance from the per-token mean (in a shared 32-PC subspace, with each pilot's extremity defined as the max Mahalanobis distance across interior layers 2–10), then trimmed the most extreme pilots by some fraction $f$ and recomputed conditional kurtosis on the remaining pilots.

The diagnostic is the ratio of trimmed kurtosis to baseline kurtosis as a function of $f$. Under the rare-extreme hypothesis, this ratio drops rapidly with small $f$ and reaches near zero at moderate $f$. Under an intrinsic heavy-tailed hypothesis (no small-fraction outlier structure, the whole bundle is just heavy-tailed), the ratio decreases gradually and continuously.

The numerical result at Phase III final, layer 7 (shared-PCA convention), baseline kurtosis 0.84:

| Trim fraction $f$ | Trimmed kurtosis | Ratio to baseline |
|---|---|---|
| 0% | 0.84 | 1.00 |
| 1% | 0.73 | 0.86 |
| **5%** | **0.25** | **0.30** |
| 10% | 0.23 | 0.27 |
| 20% | −0.01 | −0.01 |
| 33% | −0.04 | −0.05 |

**Trimming 5% of the most extreme pilots reduces conditional kurtosis to 30% of baseline.** At 10% trim, kurtosis drops to noise level (0.23, close to zero). At 20% trim, it crosses through zero into slightly negative territory. At 33% trim, it remains slightly negative.

The same pattern obtains at Phase III mid (5% trim → 0.31 of baseline). At Phase I and Phase II, the baseline kurtosis is too small (0.10 and 0.12 respectively) to be informative under trimming.

The shape of the trim curve is the signature of a Gaussian-core-plus-sparse-extreme-tail distribution. The near-collapse between 1% and 5% trim, followed by a clean sign-flip by 20% trim, is what you expect if ~5% of the pilots are well-separated from a Gaussian core.

**Conclusion: ~95% of the C-verdict heavy-tail magnitude is concentrated in ~5% of pilots.**

### 12.2 Multivariate t-distribution fit (D13)

The trimmed-kurtosis result is consistent with two related but distinct microscopic stories:

(a) The bundle is genuinely a Gaussian core plus a small fraction of outliers from a different distribution.
(b) The bundle is a single moderately heavy-tailed distribution (e.g., multivariate t with finite degrees of freedom) and the trimming is just removing the heavy-tail mass.

To distinguish these, we fit a multivariate t-distribution to each conditional bundle by Expectation Conditional Maximization Either (ECME, Liu & Rubin 1995), with the degrees-of-freedom parameter $\nu$ as a free parameter. We compared the maximum log-likelihood under the t-fit to the maximum log-likelihood under a multivariate Gaussian fit; the per-pilot log-likelihood difference $\Delta\mathrm{LL}/n$ measures how much better the heavy-tailed model explains the data.

The median best-fit $\nu$ across the frozen forward set, at layer 7:

| Phase | Median $\nu$ | $\Delta\mathrm{LL}/n$ |
|---|---|---|
| Phase I | 69.5 | 0.02 |
| Phase II | 45.9 | 0.06 |
| Phase III mid | 17.9 | 0.16 |
| Phase III final | **18.6** | **0.29** |

The best-fit $\nu$ at Phase III final is **moderately heavy-tailed**, in the range $\nu \approx 18$ — heavier than Gaussian ($\nu \to \infty$) but well above the extreme heavy-tail regime ($\nu < 5$).

A multivariate t with $\nu = 18$ is well-approximated by a Gaussian-core-plus-light-extreme-tail distribution. The t-fit and the trimmed-kurtosis result are pointing at the same thing.

**More informatively**, the per-token $\nu$ at Phase III final, layer 7, **splits into two distinct sub-populations**:

- **Heavy-tailed cluster** (12 tokens with $\nu \in [5, 25]$): includes `\n` ($\nu \approx 6$), `.` ($\nu \approx 7$), `s` ($\nu \approx 8$), digits `0`/`1`/`2` ($\nu$ in 9–14). **These are all in the §8 low cluster — the structural tokens.**

- **Effectively Gaussian cluster** (5 tokens at search-bound $\nu \approx 200$): includes `for`, `are`, `that`. **These are all in the §8 high cluster — the function words.**

**The conditional heavy-tailedness is grammatical-class-dependent.** The structural tokens carry the heavier tails; the function-word tokens are often effectively Gaussian at the conditional level. This refines the §8.4 negative result on kurtosis as a class discriminator: when measured by mean per-coordinate kurtosis, the two clusters look equally non-Gaussian; when measured by best-fit t degrees-of-freedom (which is a more powerful test of distribution shape), the two clusters are sharply distinguishable.

The grammatical class also predicts conditional heavy-tailedness, not just per-token $\lambda$ and effective rank from §8. The grammatical structure of the residual-stream geometry is more pervasive than §8 alone suggested.

### 12.3 Direct characterization of the extreme pilots (D16)

If the heavy tails are driven by ~5% of pilots and the bundle's shape is grammatical-class-dependent, the natural next question is: **what makes the extreme pilots extreme?** Are they random rare events, or do they have legible structure?

For each input token in the frozen forward set at each of the four representative checkpoints, we identified the top-5% most extreme pilots (by Mahalanobis distance at layer 7 in the shared-PCA basis), then compared:

- The distribution of next_token in the extreme set vs the bulk (95%) set.
- The distribution of prev_token in the extreme set vs the bulk set.
- The distribution of pilot position in the extreme set vs the bulk set, via the KS test on positions and via quantile summaries.

The extreme set typically contains 3–20 pilots per input token at the 5% level (depending on the token's total pilot count). With these small numbers, per-token statistical power for context-distribution comparisons is limited. The aggregate view across input tokens is more informative.

#### 12.3.1 Per-token extreme-vs-bulk separation

Phase III final, layer 7, top-5% extreme ratio of Mahalanobis $d^2$ (extreme mean $d^2$ over bulk mean $d^2$):

| Input token | Decoded | $N$ total | $d^2_x/b$ ratio |
|---|---|---|---|
| 28723 | `.` | 330 | **4.31** |
| 13 | `\n` | 173 | 3.07 |
| 28725 | `,` | 389 | 2.55 |
| 298 | `to` | 183 | 2.37 |
| 304 | `and` | 191 | 2.24 |
| 272 | `the` | 366 | 2.17 |
| 264 | `a` | 143 | 2.14 |
| 349 | `is` | 94 | 2.16 |
| 302 | `of` | 239 | 2.08 |
| 28705 | `''` (BPE space) | 99 | 1.96 |

Period and newline (both §8 low-cluster structural tokens) have the largest extreme-vs-bulk separation, consistent with §12.2: structural-token bundles have the heaviest tails and consequently the largest extreme/bulk Mahalanobis ratios.

#### 12.3.2 Position skew of extreme pilots vs bulk

Phase III final, layer 7, median pilot position (extreme p50 vs bulk p50):

- **Structural tokens with extreme pilots concentrated EARLY in chunks:** `\n` (extreme p50 = 350 vs bulk p50 = 550), `.` (375 vs 500), `of` (375 vs 450). Difference of 100–200 positions.
- **Function-word tokens with extreme pilots concentrated LATE in chunks:** `,` (650 vs 500), `the` (675 vs 550), `and` (675 vs 550), `for` (800 vs 450). Difference of 150–350 positions.
- **Other tokens with extreme position skew:** input `1` (extreme p50 = 925 vs bulk 550), BPE space marker as input (900 vs 550).

The bulk distribution at each position is similar (~500–550 median across most tokens — close to the middle of the 1024-token chunk). What differs is **which positions tend to host the extreme pilots for each token class**. Structural tokens become extreme when they appear unusually early in a chunk; function words and digits become extreme when they appear unusually late.

This is not a tautological "rare positions are rare." Position 350 is not rare overall (we sample pilots from positions 50, 100, ..., 1000, so it is one of the 20 fixed positions); the bulk of newline pilots is at position 550 (in the middle of chunks), so finding most extreme-newline pilots at position 350 reflects something about which `\n`s become extreme rather than something about which positions are sampled.

#### 12.3.3 Cross-token bigram-context signatures

Pooling across all 20 forward-set input tokens, we tabulated which next-tokens and prev-tokens appear in extreme tails more often than bulk-rate. At Phase III final:

**Top next-tokens over-represented in extreme tails (pooled):**

| Next-token | Decoded | $n_{\mathrm{ext}}$ | $n_{\mathrm{bulk}}$ | Input tokens it appears in |
|---|---|---|---|---|
| 272 | `the` | 9 | 97 | **7** of 20 |
| 28705 | `''` (BPE space) | 8 | 5 | 3 of 20 |
| 297 | `in` | 3 | 9 | 2 of 20 |

The "input tokens it appears in" column is the critical diagnostic. A next-token appearing in 1 input token's extreme tail is anecdotal. A next-token appearing in **7 different input tokens' extreme tails** is a structural signature: across many input tokens $v_i$, the bigram $(v_i, \mathtt{the})$ is over-represented among extreme pilots.

**Top prev-tokens over-represented in extreme tails (pooled):**

| Prev-token | Decoded | $n_{\mathrm{ext}}$ | $n_{\mathrm{bulk}}$ | Input tokens it appears in |
|---|---|---|---|---|
| 28740 | `1` | 5 | 12 | **5** of 20 |
| 28723 | `.` | 5 | 101 | 2 of 20 |
| 349 | `is` | 3 | 10 | 2 of 20 |

The `1`-as-prev signature appears in 5 different input tokens' extreme tails. The bigram $(\mathtt{1}, v_i)$ is over-represented among extreme pilots across many input tokens $v_i$.

**The extreme pilots have legible bigram-context structure**, even though the formal joint bigram partition (§11) is uncomputable. The structure manifests as cross-input-token regularities: certain bigrams (`X → the`, `1 → X`) tend to appear in extreme tails regardless of which specific input token $v_i$ is being conditioned on.

#### 12.3.4 Cross-phase emergence of extreme-tail structure

Running the same characterization at each of the four representative checkpoints reveals that **the extreme-tail context structure crystallizes during Phase III training**, mirroring §8.5 (grammatical cluster crystallization) and §9 (general Phase III restructuring).

Tracking the `the`-as-next signature:

| Phase | $n_{\mathrm{ext}}$ | Input tokens it appears in |
|---|---|---|
| Phase I | 5 | 4 of 20 |
| Phase II | 7 | 6 of 20 |
| Phase III mid | 9 | 6 of 20 |
| Phase III final | 9 | **7 of 20** |

Tracking the `1`-as-prev signature:

| Phase | $n_{\mathrm{ext}}$ | Input tokens it appears in |
|---|---|---|
| Phase I | 5 | 3 of 20 |
| Phase II | 6 | 5 of 20 |
| Phase III mid | 8 | 5 of 20 |
| Phase III final | 5 | **5 of 20** |

The pattern in "input tokens it appears in" grows monotonically through training for the `the`-as-next signature (4 → 6 → 6 → 7). The `1`-as-prev signature is present already in Phase I (3 of 20) but stabilizes at 5 of 20 by Phase II onward.

The position-skew magnitudes also grow with phase. At Phase I, extreme-vs-bulk position differences are modest (mostly 50–150 positions). At Phase III final, they reach 200–500 positions in several cases.

**The extreme-tail context structure is learned, not inherent.** It emerges through Phase III in lockstep with the grammatical cluster crystallization, the rise in marginal and conditional kurtosis, the divergence of per-token $\lambda$ values, and the rise in best-fit t heavy-tailedness. These are not independent phenomena; they are coordinated facets of the same Phase III restructuring.

### 12.4 The combined picture

The Model C non-Gaussianity is best described as:

> The conditional bundle $p(x_t \mid v_i)$ has a Gaussian core comprising approximately 95% of pilots, plus a structured extreme tail of approximately 5% of pilots in unusual contexts. The extreme tail crystallizes during Phase III training. The extreme contexts have grammatical-class-specific signatures: structural tokens (newline, period, digits) become extreme when they appear unusually early in chunks; function words become extreme when they appear unusually late, often in specific bigram patterns ($\texttt{X} \to \texttt{the}$, $\texttt{of} \to \texttt{X}$). The heavy-tailedness of the conditional bundle is grammatical-class-dependent: structural-token conditionals have moderately heavy tails ($\nu \approx 5\text{--}14$ via multivariate t-fit), while several function-word conditionals are effectively Gaussian at the conditional level.

This picture is consistent with everything in §7–§11:

- The §7 D5 GMM reconstruction failure: a Gaussian fit of the conditional bundle averages over the extreme tail and loses the heavy structure.
- The §10 null-controlled failure of single-variable partitions: an extreme tail of only ~5% of pilots is below the resolution at which kurtosis-reduction partition analyses are reliable. Even if next-token or position carries real information about which pilots are extreme, partitioning the bundle on these variables produces sub-bundles where the extreme pilots are spread across cells and the kurtosis-reduction signal is mostly random-labels noise.
- The §11 untestability of joint bigram partitions: the extreme tail's bigram-context structure is real but spread across multiple bigram cells, none of which individually reaches the 15-pilot threshold for a within-cell kurtosis estimate.
- The §8 grammatical bimodality: the two clusters differ not just in $\lambda$ and effective rank (§8.3) but also in conditional heavy-tailedness (§12.2) and in extreme-pilot position skew direction (§12.3.2).

---

## 13. Synthesis: what we now know

We can now give a comprehensive answer to the central question from §3.

### 13.1 The level-by-level answer

**At the level of the marginal Gaussian model proposed by Sarfati et al.:** the model is approximately correct for our 150M Llama-style transformer at the final checkpoint, in the same sense that it is correct for the larger models the paper studies. The marginal residual-stream ensemble is approximately Gaussian, and its covariance grows approximately exponentially with depth.

**At the level of the simplest microscopic extension (Model A):** refuted. Conditional bundles do not have shared covariance. Different input tokens produce within-input bundles whose total variance differs by factors of 2–4×. Per-token $\lambda$ values are bimodally distributed around $0.45$ (structural cluster) and $0.58$ (function-word cluster), with the marginal $\lambda \approx 0.36$ sitting below every per-token value.

**At the level of the next-simplest microscopic extension (Model B):** refuted. Conditional bundles are not Gaussian. Mardia multivariate kurtosis Z-scores are 25–45 across interior layers. A GMM reconstruction from Gaussian per-token fits produces a marginal with 5–15× less kurtosis than the empirical marginal.

**At the residual hypothesis (Model C):** confirmed. The conditional bundles are non-Gaussian, the marginal Gaussianity is a CLT-style averaging phenomenon, and the paper's universal $(\alpha, \lambda)$ are marginal-level statistics, not microscopic constants.

**At the level of what Model C actually means:** the conditional bundles consist of a Gaussian core (~95% of pilots) plus a structured extreme tail (~5% of pilots in unusual contexts). The extreme tail has grammatical-class-dependent severity, position-skew structure, and bigram-context signatures. None of this structure is inherent in the architecture; it is learned during training, specifically during the Phase III restructuring (steps 5000–24000 in our training trajectory).

### 13.2 What the per-token $\lambda$ "universality question" looks like now

The Sarfati et al. universality claim is for the marginal $\lambda$, measured at a single value across very different models. If the universality is genuine at the marginal level, our data shows that it must be emergent: the per-token $\lambda$ values are bimodally distributed in our model, with the two clusters at roughly $0.45$ and $0.58$, neither of which equals the marginal $0.36$. The marginal $\lambda$ is some weighted aggregate of per-token values, dominated by the larger function-word cluster but pulled down by the structural cluster.

The interesting follow-up question is: are the **per-cluster** $\lambda$ values universal across models, even if the marginal value is not? If the structural cluster always sits at $\lambda \approx 0.45$ and the function-word cluster always at $\lambda \approx 0.58$ across different model scales and corpora, that would be a more meaningful universality than the marginal one. We do not have the data to answer this for our 150M model alone, but it is the natural next investigation.

### 13.3 What Sarfati et al.'s framework gets right, and what it misses

The marginal description is approximately correct and is genuinely useful for a class of questions — particularly anything related to representational capacity or to the average dynamics of residual-stream activation through depth. The CLT-style averaging that produces marginal Gaussianity is a real phenomenon and is informative about *some* aspects of trained transformers.

What the marginal description misses:

1. **Per-token dynamical heterogeneity.** Different tokens produce bundles with different total variance, different effective rank, and different exponential growth rates. The structural-vs-function-word distinction is sharp and consequential.

2. **Conditional non-Gaussianity.** The conditional bundles are not Gaussian even after the marginal is. The Langevin SDE does not lift to a per-token description.

3. **Extreme-context structure.** A small fraction of pilots produce most of the conditional heavy tails, and these pilots have legible bigram-context and position signatures that are learned during training.

4. **Training-phase structure.** The cleanest "Gaussian-like" state of the model is at mid-training (Phase II), not at the end. The paper's full-training measurements describe a post-restructuring state with substantial conditional non-Gaussianity, not the minimum-kurtosis state.

These are not refutations of the Lines of Thought framework — that framework is a marginal-level description and the marginal level is approximately correct. They are clarifications of what the marginal description does and does not capture about the underlying microscopic structure.

---

## 14. Limitations and open directions

### 14.1 Limitations

1. **Single model, single corpus, single architecture.** All findings here use one 150M-parameter Llama-style transformer trained on FineWeb-Edu. Whether the rare-extreme-context mechanism, the Phase III emergence timing, and the grammatical bimodality generalize to larger models, different architectures, and different corpora is unknown. The §8 cross-seed stability (four seeds, identical training, same dataset) tells us the findings are not artifacts of any single training run; it does not tell us they hold across models.

2. **Frozen forward set of 20 tokens.** The per-token analyses are restricted to the 20 most frequent input tokens in the held-out set. These are dominantly function words, structural tokens, and digits — exactly the tokens for which we have enough pilots to do per-token analysis. Whether the structural-vs-function-word cluster distinction extends sensibly to content words (`computer`, `building`, `algorithm`) is open. Content words have very few pilots in our held-out set; characterizing them would require either much more held-out data or much longer chunks.

3. **The joint bigram partition is structurally untestable.** §11 documents that we cannot directly test whether bigram-context mixture explains the C verdict; we only have indirect evidence (§12.3) that extreme pilots have bigram-context structure. A larger held-out set, or a coarsened bigram variable (clustering tokens by class), would be needed to test this directly.

4. **The per-pilot context window is 1 token in each direction.** Our augmented files save prev_token and next_token, but not prev_prev or next_next. If the extreme-pilot structure depends on longer context windows, we are not seeing it.

### 14.2 Open directions

1. **Mechanistic origin of specific extreme contexts.** The cross-token bigram signatures (`X → the`, `1 → X`, `. → X`) are real and grow with training, but we do not have a mechanistic account of why these particular contexts produce extreme residual-stream activations. Candidate hypotheses include syntactic discontinuity (these bigrams correspond to sentence starts, list items, numbered headings where attention patterns shift), rare-token combinations handled by memorized rather than compositional processing, or activations at the edges of high-frequency text patterns (markdown structures, headers in FineWeb-Edu). Distinguishing these would require examining the original chunk text around extreme pilots and characterizing the syntactic/semantic context qualitatively.

2. **Cross-model and cross-corpus validation.** Replicate the analysis on a larger model (1B+ parameters) and a different corpus. Test whether (a) the structural-vs-function-word grammatical bimodality is universal, (b) the rare-extreme-context mechanism replicates, (c) the Phase III emergence timing scales sensibly with training compute, (d) the per-cluster $\lambda$ values are stable across models even if the marginal $\lambda$ is.

3. **Per-cluster universality.** If the per-token $\lambda$ in our model is bimodally distributed around $0.45$ and $0.58$, and if these values are universal across models, that would be a more meaningful universality than the marginal value. Verifying or refuting this requires the cross-model replication of (2).

4. **The reverse view.** This report covers the forward view. The reverse view (output-conditioned) is the subject of a companion investigation. Whether the same Phase III restructuring, the same grammatical bimodality, and the same rare-extreme-context mechanism apply to output-conditioned sub-bundles is open. The reverse view has interesting differences from the forward view — most notably, the predicted next token is a function of the residual stream itself, so conditioning on it introduces a different causal structure.

5. **Connection to interpretability.** The grammatical clusters from §8 and the bigram signatures from §12.3 suggest that the residual-stream geometry has interpretable structure at the level of grammatical class and local context. Whether this structure connects to other interpretability findings — superposition, features, attention patterns — is open.

6. **Mathematical formalization of the Model C structure.** We have characterized the conditional bundles as "Gaussian core plus structured extreme tail" empirically. A closed-form mathematical description (e.g., a mixture of a Gaussian with a sparse non-Gaussian component, or a contaminated Gaussian model from robust statistics) might admit cleaner theoretical analysis. The multivariate t-fit results in §12.2 are a first step in this direction but the t-distribution is a smooth heavy-tailed family rather than a Gaussian-plus-sparse-outlier family.

---

## Appendices

### A. Methodological lessons

Three lessons that may be useful for related work on residual-stream geometry:

**A.1 Kurtosis-reduction sub-conditioning analyses require null control.** Random labelings reduce within-group kurtosis substantially even when the labeling carries no information. Any sub-conditioning claim must report the matched-random-labels reduction alongside the real partition's reduction. The pilot scheme also matters: fixed pilot positions can produce systematic entanglement between position and surrounding-context statistics that misleads single-variable partition analyses.

**A.2 Basis-invariant distance metrics for $\Sigma$ matrices.** The raw R-matrix Frobenius distance commonly used for cross-checkpoint covariance comparisons saturates near $\sqrt{2H}$ for any two non-identical bases due to SVD sign and permutation ambiguity. The singular-value spectrum distance (sorted differences of squared singular values, normalized by total variance) is the right tool for basis-invariant cross-checkpoint $\Sigma$ comparisons.

**A.3 Principal-subspace discriminators in high dimensions need bootstrap-floor calibration.** Top-$k$ principal angles between two subspaces estimated from $n$ samples each are far from their true values when $n \ll H$. Naive comparisons against $90°$ overstate dissimilarity. The bootstrap noise floor (within-sample resampling of the same population) gives the practical threshold for any "are these subspaces different?" claim.

### B. Data and code locations

- Multiview stage-C results: `phase1_runs_gelu/multiview/seed_S/multi_view_step_NNNNNNNN.npz`
- Augmented activations (fixed positions): `phase1_runs_gelu/multiview/seed_S/augmented_step_NNNNNNNN.npz`
- Augmented activations (randomized positions, for §10's randomization-control follow-up): `phase1_runs_gelu/multiview/seed_S/augmented_step_NNNNNNNN_random.npz`
- Augmented activations (fixed positions, with prev_token for §11–§12): `phase1_runs_gelu/multiview/seed_S/augmented_step_NNNNNNNN_ngram.npz`
- Analysis outputs and figures: `phase1_runs_gelu/multiview/model_abc/d{1-16}_*.{npz,png,json}`
- Per-script logs from overnight runs: `phase1_runs_gelu/multiview/model_abc/logs/`

### C. Verdict summary table

| Hypothesis | Verdict | Key evidence | Section |
|---|---|---|---|
| Model A (Gaussian, shared $\Sigma_0$) | Refuted | D1: cross-token CV($\mathrm{tr}\,\Sigma_i$) = 0.6–0.8 vs threshold 0.115 | §7.1 |
| Model B (Gaussian, token-dependent $\Sigma_i$) | Refuted | D4b: Mardia $Z$ = 25–45; D5: GMM reconstruction gives $\kappa \approx 0$ vs empirical 5–15 | §7.4–7.5 |
| Model C (non-Gaussian conditionals) | **Confirmed** | D4, D5 across all phases | §7.6 |
| Possibility 1 (intrinsic single-heavy-tailed distribution) | Partial | D13: best-fit $\nu \approx 18$ is moderately heavy, not extreme; per-token $\nu$ structured by grammatical class | §12.2 |
| Possibility 2 (rare extreme tail) | **Confirmed** | D12: 5% trim reduces kurtosis to 30% of baseline, 10% to noise level; D16: extreme pilots have structured bigram and position signatures | §12.1, §12.3 |
| Possibility 3 (joint bigram-context mixture) | Untestable in this dataset | D14: zero valid (prev, next) joint sub-bundles at any phase; not ruled out, but not directly testable | §11 |

### D. Glossary of internal terms used in this report

- **All-to-all bundle / marginal**: the empirical distribution of residual-stream vectors at a given layer pooling across all pilots regardless of input or position.
- **Forward view**: the per-input-token sub-bundle at a given layer.
- **Reverse view**: the per-predicted-token sub-bundle at a given layer (covered in a companion report).
- **Pilot**: a single (chunk, position) sample for which residual-stream vectors are recorded at every layer.
- **Frozen forward set**: the fixed set of 20 most-frequent input tokens used for per-token analysis.
- **Phase I / II / III**: training-phase windows defined in §9.
- **Baseline kurtosis**: the conditional bundle kurtosis without any sub-conditioning or trimming, in whichever basis convention is currently in use.
- **Real signal**: the part of an apparent sub-conditioning effect that survives subtraction of the matched random-labels null.
- **Extreme pilots**: pilots in the top-fraction $f$ of Mahalanobis distance from their per-token mean.
- **Bulk pilots**: pilots in the bottom $(1-f)$ of Mahalanobis distance.
- **Per-token-SVD basis convention**: kurtosis computed in each token's own SVD basis.
- **Shared-PCA basis convention**: kurtosis computed in a single 32-PC basis derived from the full pilot pool.
