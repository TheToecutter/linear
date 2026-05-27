# Output-Conditioned Ensembles in Lines of Thought: Shape, Scale, and the Asymmetry Between Input and Output Partitions

**Phase 2 supplementary investigation, reverse build-up campaign**
**Run: phase1_runs_gelu, seeds 0–3, 50 checkpoints, final step 24000**

---

## Abstract

The "Lines of Thought" paper (Sarfati et al., ICLR 2025) shows that residual-stream activations across a held-out set of inputs form an approximately Gaussian "blunderbuss" at each layer of a trained transformer, with covariance that grows exponentially in depth at a universal rate $\lambda$. Our earlier multiview campaign tested whether this macroscopic Gaussian admits a microscopic Gaussian-mixture decomposition along the input-token axis, and found it does not: input-conditioned bundles are heavy-tailed (multivariate Mardia $Z = 25$–$45$ across interior layers), have token-dependent variance and growth rates, and Gaussian-fit-and-remix reconstructions undershoot the empirical marginal's per-coordinate excess kurtosis by 5 to 15 units. We refer to that finding as the *forward Model C verdict*.

This work investigates the dual question: does the marginal admit a Gaussian-mixture decomposition along the *output* (successor-token) axis? We replicate the Models A/B/C discriminator suite on two reverse partitions — *reverse-actual* (the ground-truth next token in text) and *reverse-predicted* (the model's argmax prediction) — and add one new diagnostic, the unembedding-subspace decomposition, that has no forward analog. We test seven hypotheses (S1–S2, F1–F4, N1–N2) ranging from structural reproducibility through functional reconstruction quality to a new conjecture about the readout's role in regularizing conditional shape.

The findings reorganize the picture set by the forward investigation in three substantive ways. First, the Model C verdict generalizes universally: at every interior layer of both reverse views, Gaussian-mixture reconstruction undershoots the marginal's kurtosis by margins comparable to the forward case. There is no single-variable conditioning of the residual-stream bundle — input token, actual successor, or predicted successor — under which the marginal builds up as a clean Gaussian mixture. Second, the unembedding regularizes the *shape* of conditional bundles but not their *scale*: cell-wise multivariate Gaussianity (Mardia $Z$) drops monotonically from 70 at interior layers to 8 at the final layer, but cell-wise variance grows monotonically through layer 12 and contracts only at the readout step itself. Shape and scale are regularized by two distinct mechanisms acting at opposite ends of depth. Third, partitioning by the model's argmax-predicted successor produces conditional bundles whose Gaussian-mixture reconstruction is markedly *farther* from the empirical marginal than the actual-successor partition produces, with this asymmetry exhibiting the same sign in all four seeds at every interior layer. The model has internalized a residual-stream organization keyed to its own commitment structure more tightly than to the data-generating distribution.

Two basis-invariant signatures of grammatical structure replicate across the input-output axis: the predicted-successor cells separate into two clusters by readout-step variance compression — *content words* (more compressed, $\lambda^{\text{readout-step}} \approx -0.93$) versus *structural elements* — punctuation, numbers, sub-word fragments, short prepositions (less compressed, $\lambda^{\text{readout-step}} \approx -0.63$) — with 100% cross-seed cluster stability. The actual-successor cells show the same bimodality with 73.7% cross-seed stability, with the four-seed disagreement localized to the exact tokens at the grammatical boundary between "complete lexical unit" and "structural fragment." The two views thus organize different but related cuts of grammar, with the model's prediction-keyed organization being strictly more reproducible than its data-tracking organization.

We close by interpreting these findings as ruling out two refinements of the forward Model C verdict (the depth-localized story and the readout-subspace-localized story) and motivating a third (the label-structure-organized story), which in turn provides a concrete target for the pair-conditional analysis sketched in our companion Markovianity writeup. The forward and reverse views of the input-conditioned ensemble question are not interchangeable; together they discriminate among microscopic stories that either view alone leaves consistent with the same Model C verdict.

---

## 1. Background and motivation

### 1.1 Residual streams as a stochastic ensemble

Decoder-only transformers process tokens through a sequence of residual blocks. At each layer $t \in \{0, 1, \ldots, L_{\text{total}}-1\}$, every token position carries a residual-stream vector $x_t \in \mathbb{R}^H$, the running sum of all attention and MLP contributions up to that depth. For a held-out evaluation corpus containing many input sequences and many token positions per sequence, the collection of these vectors at a fixed layer $t$ forms an empirical distribution that we will call the **all-to-all bundle** or the **marginal** at depth $t$. The all-to-all bundle is observable: it is the empirical histogram of residual-stream activations on the evaluation set, with one vector per (sequence, position) pilot.

The Lines of Thought paper studies this bundle's geometry across depth. Its headline empirical findings are:

1. The all-to-all bundle at each layer is approximately a single Gaussian. The non-Gaussianity of its residuals from a low-rank linear-flow model is small.
2. The covariance of this approximate Gaussian grows exponentially in depth at a universal rate $\lambda$, with leading-order scaling
$$\mathrm{tr}(\Sigma_{\text{marg}}(t)) \approx \alpha e^{\lambda t},$$
where $\alpha$ and $\lambda$ are constants of the trained model.
3. Trajectories of individual tokens through depth can be modeled by a continuous-time stochastic differential equation of Langevin form,
$$dx_t = A(t)\,x_t\,dt + \sigma(t)\,dW_t, \qquad \sigma\sigma^\top = \alpha e^{\lambda t} I,$$
with the drift and diffusion coefficients identified by fitting to the marginal-level statistics.

The Langevin formulation is a *phenomenological model* of the marginal's variance growth, not a statement about microscopic dynamics. A transformer forward pass is deterministic given input and weights. The "noise" in the SDE represents heterogeneity across the held-out ensemble (different input sequences, different positions, different surrounding contexts) reinterpreted as fictitious process noise. The paper measures and models the marginal; it makes no direct empirical claim about the structure of conditional sub-ensembles formed by holding some observable variable fixed.

### 1.2 The conditional ensemble question

The natural extension of the Lines of Thought framework asks what the marginal looks like when partitioned on observable variables that the marginal averages over. The all-to-all bundle, by total-probability bookkeeping, is a marginal of conditional sub-bundles indexed by any such variable, with mixture weights given by the empirical frequency of the conditioning value:
$$p_{\text{marg}}(x_t) = \sum_z \pi_z \, p(x_t \mid z).$$

If $p_{\text{marg}}$ is approximately Gaussian, the conditionals $p(x_t \mid z)$ are constrained in their first two moments (via the law of total variance) but their *shape* is underdetermined. The natural unifying conjecture is that conditionals inherit the same parametric form as the marginal — Gaussian shape, with conditional means and conditional covariances satisfying the same exponential growth law — so that the macroscopic universal $(\alpha, \lambda)$ has a microscopic counterpart at the level of every individual cell.

This was the working hypothesis of the forward investigation, which partitioned the marginal on the *input token at the pilot position*. The forward investigation's conclusion was negative on every diagnostic: input-conditioned bundles are not Gaussian, do not share a common covariance, do not share a common growth rate, and do not aggregate into a Gaussian marginal under their measured frequencies. We summarized this conclusion as the *Model C verdict* (Section 1.4 of this writeup recapitulates the verdict's precise content).

### 1.3 Output conditioning as the dual question

There are at least three observable conditioning variables for which the held-out evaluation set provides a partition of the all-to-all bundle:

- **Input token** $v$: the token identity at the pilot position itself. Constitutes the *forward* view.
- **Actual successor** $w$: the token identity at the position immediately following the pilot, as it appears in the held-out text. Constitutes the *reverse-actual* view.
- **Predicted successor** $\hat w$: the model's argmax of the next-token distribution at the pilot position. Constitutes the *reverse-predicted* view.

The forward investigation tested the input-token partition exclusively. This work tests the actual-successor and predicted-successor partitions on the same evaluation set, using the same Models A/B/C discriminator infrastructure (extended to a `view` parameter so that the same code path can read any of the three conditioning variables), and adds one new diagnostic motivated by structural properties unique to the reverse view.

The reverse-view question is not a redundant check of the forward Model C verdict. The forward and reverse partitions have qualitatively different structural properties that make them genuinely independent probes of the conditional-ensemble question. Section 1.4 explains why.

### 1.4 The forward Model C verdict, recapitulated

For completeness and to fix notation, we recapitulate the forward investigation's main findings. Let $E_v = \{x_t^{(k)} : \text{input}_k = v\}$ denote the input-conditioned cell for token $v$ at layer $t$. The three nested models tested were:

- **Model A**: $p(x_t \mid v) = \mathcal{N}(\mu_v(t), \Sigma_0(t))$ — Gaussian conditionals, shared covariance, only the per-token drift differs.
- **Model B**: $p(x_t \mid v) = \mathcal{N}(\mu_v(t), \Sigma_v(t))$ — Gaussian conditionals with per-token covariance.
- **Model C**: Conditionals are not Gaussian; the marginal can still appear Gaussian by mixing.

Five discriminators (D1–D5) addressed different empirical signatures of A vs B vs C, and their joint verdict at the final checkpoint (step 24000) of a 150M-parameter Llama-style transformer with 14 residual-stream snapshots and 4 seeds was:

1. Cross-token coefficient of variation of $\mathrm{tr}(\Sigma_v(t))$ at 0.6–0.8 throughout the interior, well above the 0.115 finite-sample threshold (Model A refuted via D1).
2. Per-token log-linear fits of $\Sigma_v$ across depth give $\lambda_v$ values varying by $\sim 40\%$ across input tokens at the final checkpoint, with CV($\lambda_v$) = 0.14 above the 0.062 threshold (Model A refuted via D3).
3. Cell-wise multivariate Gaussianity (Mardia $Z$ in a shared 32-dim PCA subspace) sits at $Z = 25$–$45$ across interior layers, vs the $|Z| = 2$ rejection threshold at the 5% level (Models A and B both refuted via D4b).
4. Per-coordinate excess kurtosis on cells is in the 1.5–6 range, while the all-to-all marginal sits at 5–15; conditional bundles are *more* non-Gaussian than the marginal they average to (D4a, supporting Model C).
5. Gaussian-mixture reconstruction (sample from per-cell Gaussian fits with empirical mixture weights, compute the synthesized marginal's kurtosis) gives values 5–15 units *below* the empirical marginal kurtosis. The marginal's heavy tails are not reconstructible from Gaussian conditional fits (D5, supporting Model C).

The forward Model C verdict is that the marginal Gaussian observed by Sarfati et al. is a central-limit-theorem phenomenon over a population of structurally distinct, non-Gaussian conditional bundles — at least when those bundles are indexed by input token. The unification of macro and micro scales that the input-conditioned Gaussian-mixture conjecture would have provided does not obtain.

A separate finding from the forward investigation — that per-token $\lambda_v$ has a basis-invariant bimodal structure across input tokens, with a "connective" cluster ($\lambda \approx 0.58$: function words and intra-clause punctuation) and a "structural" cluster ($\lambda \approx 0.45$: terminators, digits, sub-word fragments) — establishes that *some* basis-invariant grammatical structure is visible in input-conditioned per-cell statistics. The reverse-view analog of this bimodality (hypothesis N2 below) is one of the diagnostics we test in this work.

A further finding — that conditional non-Gaussianity emerges during training in a three-phase structure (rapid SVD consolidation through step ~1500, a brief consolidated mid-training plateau through step ~5000, late-training restructuring through step 24000) — locates the Lines of Thought framework's measurements in Phase III. The reverse-view counterpart of this trajectory is not formally tested here but is visible in our Phase A outputs and discussed in Section 7.

A final, methodologically important finding — that simpler sub-partition mixture interpretations of the Model C verdict (next-token, position) failed null-correction tests — informs the design of the reverse view's F4 diagnostic, which is the analog at the level of full reverse partitions of the sub-conditioning experiments that did not survive null correction at the input-conditioned level. The reverse-view null protocol is mandatory (Section 3.4) precisely because the forward investigation's experience showed that ignoring the null on labels with non-uniform frequency distributions can produce kurtosis reductions large enough to be mistaken for substantive structure.

### 1.5 The asymmetry between forward and reverse boundary conditions

The forward and reverse views are not mirror images of each other in any operationally useful sense. The boundary conditions that pin the conditional bundles at the two ends of depth are enforced by structurally different mechanisms, and these differences make the reverse measurement a discriminator among multiple stories consistent with the forward Model C verdict rather than a redundant check.

**Forward boundary.** The forward view's variance is pinned to zero at $t = 0$ by a *deterministic, untrained* operation: the embedding lookup. Every pilot with input token $v$ has identically the same $x_0$, so $V_{\text{within-}v}(0) = 0$ by arithmetic. The dynamics that follow — attention folding in context, MLP nonlinearity, layer norm — produce the variance-growth and shape structure observed in $\{E_v\}_v$ across the interior layers. Non-Gaussianity at later layers is a consequence of how the network differentiates from a deterministic starting point.

**Reverse boundary.** The reverse view's boundary condition at the *late* layers is enforced by a *learned* operation: the post-final-norm map followed by the unembedding $W_U \in \mathbb{R}^{|V| \times H}$ that produces logits. For tied-embedding architectures (as in our trained model), $W_U = W_E^\top$, so the readout directions are exactly the embedding column span. All pilots with successor $w$ end in a residual-stream region whose softmax over $W_U$ assigns probability mass concentrated on $w$, but this is not a delta function. The region is a learned commitment manifold, with strictly positive within-$F_w$ variance at the final layer and a shape constrained by the geometry of the unembedding rather than by an arithmetic identity.

Four substantive consequences of this asymmetry, listed in increasing depth, motivate the reverse-view experiment:

First, the partition variable is qualitatively different. The forward partition labels pilots by what *was* — the deterministic input at the pilot position. The reverse partition labels them by what *will be* — a downstream observable that depends on the same forward pass plus subsequent context (for reverse-actual) or on the model's own internal commitment (for reverse-predicted). The two partitions induce different equivalence classes on the held-out set with no a priori reason for their Gaussian-mixture decomposability to agree.

Second, the depth at which the partition variable is "naturally" available differs. The input token is sharply available at $t = 0$ and decays in geometric salience across depth as attention folds in surrounding context. The successor token is unavailable at $t = 0$ (the model has not yet computed the prediction) and becomes geometrically salient only at depths where prediction commitment begins to form. The reverse partition should therefore look like meaningful structure at the *late* layers, where the forward partition is least meaningful.

Third, the readout subspace admits a natural decomposition. The unembedding $W_U \in \mathbb{R}^{|V| \times H}$ is low-rank relative to ambient dimension — its row span has dimension at most $\min(|V|, H)$, and in our 150M-parameter model with $H = 896$ and $|V| = 32768$, the empirical rank is $\min(896, 768) = 768$ (the embedding bottleneck imposed by tied weights). The residual-stream subspace $\mathbb{R}^H$ decomposes orthogonally into the rowspan of $W_U$ (the *readout-visible subspace* of dimension up to 768) and its orthogonal complement (the *readout-invisible subspace*, dimension at least 128 in our setting, generally larger when the embedding is rank-deficient). The forward view has no analogous learned subspace decomposition; the embedding lookup is deterministic and produces a fully-specified $x_0$ in every direction.

Fourth, the actual-vs-predicted distinction is reverse-only. There is no forward analog of "the model's belief about its input"; the input is given. But the reverse view admits two natural conditioning variables: the data's true successor (what the held-out text actually contains) and the model's argmax prediction (what the model committed to). These two variables coincide on pilots where the model predicts correctly, and diverge on pilots where it does not. The two reverse partitions are therefore probes of different objects — the data-generating distribution versus the model's internal commitment structure — and any divergence between their conditional bundles' decomposability is a measurement of how those two objects differ.

### 1.6 What the reverse measurement discriminates

The forward Model C verdict is consistent with several distinct underlying microscopic stories. Each story makes a different prediction at the reverse view, and the reverse measurement can in principle separate them. We list three.

**Story 1 (uniform-across-depth non-Gaussianity).** The marginal's heavy tails live uniformly across the residual-stream space at every layer, and conditioning on any single variable (input, actual successor, predicted successor) leaves them in place. Prediction at the reverse view: Model C verdict reproduces at every interior layer of both reverse views; Mardia $Z$ stays elevated even at the final layer; D5 reconstruction undershoots the marginal equally well across depth.

**Story 2 (depth-localized non-Gaussianity, regularized at the readout).** The marginal's heavy tails are a property of the interior layers, regularized away by the unembedding's geometric pressure as $t \to L_{\text{total}} - 1$. Prediction at the reverse view: Mardia $Z$ decreases monotonically toward the final layer (hypothesis F2 below); reverse-actual D5 reconstruction is substantially better at late layers than at the interior (F3); heavy tails at the final layer live preferentially in the readout's orthogonal complement (N1).

**Story 3 (label-structure-organized non-Gaussianity).** The conditional bundles' non-Gaussianity is a property of *which partition labels* are applied to the pilots, not of the depth at which the bundles are measured. Different labelings carve the residual stream into mixtures with structurally different decomposition profiles. Prediction at the reverse view: actual-successor and predicted-successor conditional reconstructions diverge from each other in a way that is robust across seeds, with the divergence's sign and magnitude informative about which labeling carves the bundle more cleanly (F4).

These three stories are not exhaustive and not mutually exclusive. The reverse experiment is designed to test all three simultaneously and to report which prediction(s) the data support.

### 1.7 Roadmap

The remainder of this writeup is organized as follows. Section 2 lists the formal hypotheses (S1–S2, F1–F4, N1–N2, D1), with verdict criteria and gating relationships. Section 3 describes the experimental design: the parameterization of the existing discriminator infrastructure across the three views, the new unembedding-subspace decomposition, the within-chunk shuffle-null protocol that all reverse-view kurtosis comparisons are reported against, the compute budget, and the phase structure of the campaign (Phases A–E, each phase gating decisions for the next).

Sections 4 through 7 present the results in the order they were obtained. Section 4 reports Phase A (parameterized D1, D3, D4a across the two reverse views and the forward view as idempotency check) and Phase B (D4b, the multivariate Mardia $Z$ depth profile). Section 5 reports Phase C (D5 reconstruction with shuffle null) and the F4 verdict. Section 6 reports Phases D and E: the unembedding-subspace decomposition (N1) and the reverse $\lambda^{\text{contract}}$ cluster analysis (N2), the latter requiring a reformulation of the statistic after first results showed the proposal's expected multi-layer contraction does not exist. Section 7 collects the substantive findings and reorganizes them against the three stories listed in Section 1.6.

Section 8 discusses what the findings imply for the broader research program — particularly that the pair-conditional analysis sketched in our companion Markovianity writeup is now motivated by two independent failures of single-variable conditioning to recover Gaussian mixture components, and that the F4 finding gives a concrete target for pair-conditional decomposition. Section 9 closes with limitations and open questions.

---


## 2. Hypotheses

We pre-registered seven primary hypotheses before running the experiment, plus one auxiliary hypothesis. The pre-registration distinguishes three types of hypotheses. *Structural hypotheses* (S1–S2) test that the reverse-view measurements are well-defined and reproducible across seeds; they are not informative about the substantive question but their satisfaction is a precondition for any interpretation of the others. *Functional hypotheses* (F1–F4) test whether the build-up question, framed as in the forward investigation, has the same answer for the reverse partitions as for the forward partition. *Reverse-only hypotheses* (N1–N2) test diagnostics that have no forward analog: a structural prediction about where the marginal's heavy tails live in the unembedding-induced subspace decomposition (N1), and a basis-invariant grammatical-role bimodality conjecture for reverse cells parallel to the forward $\lambda_v$ bimodality (N2). One auxiliary hypothesis (D1) concerns the training-time emergence trajectory.

Each hypothesis has an explicit verdict criterion and an explicit pre-registered prediction. The verdict criteria are operational thresholds chosen before any data was inspected; they are not adjustable after the fact. Where the verdict came out at the boundary of its criterion or in unexpected direction, we report that with the original criterion intact and discuss the substantive content separately.

### 2.1 Structural hypotheses

**S1. The reverse-view discriminator outputs are well-defined and reproducible across seeds.** For every chosen successor $w$ in the reverse token set and every $\hat w$ in the predicted-successor token set, the basis-invariant statistics — per-cell $\log\alpha_w$, $\lambda_w$, effective rank profile, per-coordinate kurtosis profile — have within-seed standard deviation comparable in fractional magnitude to the forward-view measurements documented in Section 4 of the companion forward writeup.

*Verdict criterion:* within-seed std ≤ 2× the forward-view std for the corresponding statistic at the matched layer index, for at least 80% of statistic/layer combinations.

*Pre-registered prediction:* satisfied. There is no architectural reason for reverse-view per-cell statistics to be substantially noisier than forward-view statistics on the same evaluation set.

**S2. The reverse-view law of total variance is satisfied numerically.** The reverse-actual and reverse-predicted decompositions sum to the all-to-all variance at every layer, within the small sample-size correction induced by tokens outside the chosen sets.

*Verdict criterion:* by inheritance from the existing multiview campaign, which already verified this property as part of its pre-existing `within_between_decomposition` validation.

S2 is mechanical (bookkeeping-exact via the law of total variance applied to the empirical mixture) and is included for completeness; if it were to fail it would indicate a code bug rather than an empirical finding.

### 2.2 Functional hypotheses

**F1. Reverse-actual conditionals are non-Gaussian at interior layers.** Multivariate Mardia $Z$ for the cells $F_w$ in a shared 32-dimensional PCA subspace exceeds the 5% rejection threshold ($|Z| > 2$) at most interior layers $t \in [1, L_{\text{total}} - 2]$.

*Verdict criterion:* at least 10 of the 12 interior layers (layers 1 through 12 inclusive) have cell-mean $|Z| > 5$ averaged across the reverse-actual token set.

*Pre-registered prediction:* satisfied. The forward investigation found Mardia $Z = 25$–$45$ at interior layers for input-conditioned cells; the data-generating mechanism produces non-Gaussianity through layer dynamics that should be visible in any cellwise partition of the same activations, not specifically the input-conditioned one.

**F2. Reverse-actual conditional non-Gaussianity decreases monotonically toward the final layer.** The cell-mean Mardia $Z$ profile across depth has a clear downward slope at the late layers, in contrast to the forward investigation's roughly flat interior profile. Specifically, the slope of cell-mean $Z$ over the last four layers is negative, and the final-layer $Z$ is less than 70% of the interior maximum.

*Verdict criterion:* slope($\bar Z(t)$, $t \in [L_{\text{total}}-4, L_{\text{total}}-1]$) < 0 AND $\bar Z(L_{\text{total}}-1) / \max_{t \in [1, L_{\text{total}}-2]} \bar Z(t) < 0.7$.

*Pre-registered prediction:* unclear; the proposal assigned this hypothesis as the test of Story 2 (depth-localized non-Gaussianity regularized at the readout). If F2 holds, the unembedding's geometric pressure has a measurable effect on conditional shape Gaussianity that has no forward analog. If F2 fails, Story 2 is weakened and Stories 1 and 3 are favored.

**F3. The reverse-actual Model B reconstruction undershoots the marginal less than the forward Model B reconstruction does at late layers.** Concretely, let $\kappa^{\text{emp}}(t)$ denote the empirical marginal's per-coordinate excess kurtosis at layer $t$, and $\kappa^{\text{B-fwd}}(t)$, $\kappa^{\text{B-rev-act}}(t)$ the kurtoses of the forward and reverse-actual Model B mixture samples. F3 predicts that for some range of late layers,
$$\bigl|\kappa^{\text{emp}}(t) - \kappa^{\text{B-rev-act}}(t)\bigr| < \bigl|\kappa^{\text{emp}}(t) - \kappa^{\text{B-fwd}}(t)\bigr|.$$

*Verdict criterion:* at the final-checkpoint, seed-0, late layers $t \in [9, 13]$, the inequality holds at least at three of the five layers, by a margin of at least 0.5 kurtosis units.

*Pre-registered prediction:* if Story 2 holds, F3 should hold at the late layers — the marginal's heavy tails are depth-localized away from the readout, so a partition that sees the readout boundary cleanly (reverse) should give a better reconstruction at the readout-adjacent layers than a partition tied to the boundary at the other end (forward). If Story 1 or Story 3 holds, F3 may fail.

**F4. The reverse-actual and reverse-predicted Model B reconstructions disagree at late training.** Define the predicted-vs-actual divergence
$$\Delta(t) = \kappa^{\text{B-rev-pred}}(t) - \kappa^{\text{B-rev-act}}(t).$$
For checkpoints after the Phase 1 loss-curve stabilization step (around step 4000 for this run), $\Delta(t)$ has consistent sign at the late layers across all four seeds.

*Verdict criterion:* at the final checkpoint, cross-seed sign agreement of $\Delta(t)$ over the last four layers exceeds 75%. Null absorption (the ratio of shuffled-null $\Delta$ to real $\Delta$) at those same layers does not exceed 70% (the null-correction gate, Section 3.4).

*Pre-registered prediction:* sign is open. $\Delta < 0$ (predicted-conditional reconstruction closer to Gaussian than actual-conditional) means the model's argmax labels carve cleaner mixture components and the model has internalized an organization keyed to its own commitments. $\Delta > 0$ (the reverse) is less expected and would suggest the data has hidden regularity the model has not internalized. Either sign is informative; the substantive content is in *which* sign reproduces across seeds.

### 2.3 Reverse-only hypotheses

**N1. The unembedding-subspace decomposition isolates non-Gaussianity in the readout's orthogonal complement.** Let $W_U \in \mathbb{R}^{|V| \times H}$ be the unembedding matrix and let $P_\parallel$ project onto the rowspan of $W_U$ truncated to its top $d_\parallel$ singular directions, with $P_\perp = I - P_\parallel$. Define the per-coordinate excess kurtosis "gap" as
$$g(d_\parallel; t) = \kappa^{\parallel}_{\text{marg}}(t; d_\parallel) - \kappa^{\perp}_{\text{marg}}(t; d_\parallel),$$
where $\kappa^{\parallel}$ is computed on the $d_\parallel$-dimensional projection $Z_\parallel = X V_\parallel \in \mathbb{R}^{N \times d_\parallel}$ and $\kappa^{\perp}$ is computed on the $(H - d_\parallel)$-dimensional residual restricted to its non-degenerate directions.

N1 predicts that at the final layer $t = L_{\text{total}} - 1$, the gap is substantially negative — that is, the parallel (readout-visible) component is substantially more Gaussian than the perpendicular (readout-invisible) component. Specifically, the median gap across the $d_\parallel$ sweep $\{32, 64, 128, 256\}$ is more negative than $-0.5$ at every $d_\parallel$.

*Verdict criterion:* median final-layer gap $g(d_\parallel; L_{\text{total}}-1) < -0.5$ for every $d_\parallel \in \{32, 64, 128, 256\}$.

*Pre-registered prediction:* if Story 2 holds (depth-localized non-Gaussianity regularized at the readout), N1 should also hold — the readout's regularization of conditional Gaussianity should manifest geometrically as a subspace decomposition with the parallel component cleaner. If Story 1 or Story 3 holds, N1 may fail or hold only weakly.

**N2. The reverse-view per-cell $\lambda$ has a basis-invariant bimodal structure across successors with grammatical interpretation.** Specifically, k-means clustering ($k = 2$) on per-cell reverse $\lambda$ values averaged across seeds at the final checkpoint produces two clusters whose decoded membership has grammatical interpretation analogous to the forward investigation's connective vs structural split, and cross-seed cluster membership is stable for at least 85% of tokens.

*Verdict criterion:* k-means produces two clearly separated centers (separation > 1.5 within-cluster std) AND at least 85% of tokens have consistent cluster assignment across all four seeds AND the cluster membership has post-hoc interpretable grammatical structure.

*Pre-registered prediction:* unclear. The forward bimodality is a property of input-token-conditioned variance growth, with connectives generating larger growth than structural tokens. The reverse analog would be a property of output-token-conditioned variance dynamics, but the precise grammatical content of the split is not predictable in advance; the reverse view's "natural" statistic differs from the forward view's because variance dynamics at the readout boundary are qualitatively different from variance dynamics at the embedding boundary. In particular, the proposal's first formulation of the reverse $\lambda$ statistic — a log-linear fit on the descending phase of the variance curve, $\lambda^{\text{contract}}$ — turns out to require reformulation after first results (Section 6.2).

### 2.4 Auxiliary hypothesis

**D1. The reverse-view conditional non-Gaussianity emerges during training in a phase-locked relationship with the forward-view emergence.** The forward investigation found that conditional kurtosis at interior layers grows from $\sim 0.1$ at step 100 to $\sim 5$–$8$ at step 24000, with the bulk of the growth between steps 2000 and 10000 (Phase III in the forward terminology). The reverse-view analog tracks this trajectory either in-phase (both growing together) or out-of-phase (reverse non-Gaussianity emerging earlier or later than forward).

*Verdict criterion:* comparison of cross-seed mean Mardia $Z$ trajectories for the reverse and forward views, threshold-crossing step ($Z > 10$) reported with 95% bootstrap CI.

*Pre-registered prediction:* in-phase emergence would say a single training-time process produces conditional structure visible from both directions. Out-of-phase emergence would say input-side and output-side structures are built at distinguishable training stages. Either outcome is informative.

### 2.5 Gating relationships between hypotheses

The hypotheses are not independent. Two gating relationships among them affect the campaign's experimental design:

The Phase B gate. F2 (the Mardia $Z$ depth-gradient hypothesis) acts as a gate on whether Phase D (the unembedding-subspace decomposition for N1) and the expanded Phase C (D5 on all four seeds for F4) are run at full scope. If F2 is refuted, Story 2 is weakened and N1 is unlikely to hold; we run Phase D at reduced scope (seed 0 only, all $d_\parallel$ values, documentation only) and Phase C at default scope (seed 0 only). If F2 is supported, all subsequent phases run at full scope.

The Phase C null gate. F4's verdict depends on the null absorption (Section 3.4). If the shuffle null absorbs more than 70% of the raw $\Delta$ signal at the late layers, F4 is reported as unresolved rather than as supporting either sign. This gate is mandatory and applied unconditionally.

---

## 3. Experimental design

### 3.1 Parameterization of the discriminator suite

The existing Models A/B/C discriminator (`model_abc_discriminator.py`) hardcodes the forward view in three places:

- The token-set loader returns `forward_set, _, _ = load_token_sets(run_dir)`.
- Per-cell flow access reads `r.forward_flows` from the multiview result object.
- Per-pilot partition labels read `aug["input_ids"]` from the augmented activation files.

The first engineering deliverable of this work is to lift these three hardcodes to a `view` parameter that selects between three configurations summarized in the dispatch table below.

| view              | token set                    | partition label  | applies   |
|-------------------|------------------------------|------------------|-----------|
| `forward`         | `forward_set`                | `input_ids`      | existing  |
| `reverse_actual`  | `reverse_actual_set`         | `next_ids`       | new       |
| `reverse_pred`    | `reverse_pred_set`           | `pred_ids`       | new       |

The five discriminators (D1, D3, D4a, D4b, D5) are computationally view-agnostic — they read per-cell flow files and compute scalar or array summaries with no view-specific arithmetic. The lifting is therefore mechanical: only the input loaders and the output paths change. Output paths gain a view suffix to coexist with the existing forward outputs:

```
d1_token_cv_{view}.npz
d3_per_token_fits_{view}.npz
d4a_kurtosis_{view}.npz
d4b_gaussianity_{view}/seed{S}_step{T:08d}.npz
d5_reconstruction_{view}/seed{S}_step{T:08d}.npz
```

A hard idempotency requirement: running the parameterized discriminator with `view="forward"` must reproduce the existing forward output bit-for-bit. This is verified by an integration test that loads both the legacy `d1_token_cv.npz` and the new `d1_token_cv_forward.npz` and asserts array-level equality. The integration test passed at first invocation, confirming the refactor did not perturb forward numerics. This is a useful sanity check on the campaign's overall pipeline: any drift in downstream numbers can be attributed to genuine view differences rather than to an inadvertent change in shared infrastructure.

### 3.2 Threshold and interpretation adaptations

Two discriminators require interpretation rewrites for the reverse view, even though their computations are unchanged. The forward view's $V_{\text{within-}v}(0) = 0$ identically by the deterministic embedding lookup, so cross-cell coefficient of variation at layer 0 is degenerate; for reverse views the boundary at $t = 0$ is not pinned, and cross-cell CV is well-defined (though typically large, reflecting the diversity of upstream contexts that can lead to a given successor). The interpretation of D1 cross-cell CV magnitudes therefore differs: a large value at the embedding boundary is expected for reverse views and not informative; a small value at the readout boundary (layer 13) is informative because the unembedding-induced contraction is expected to bring per-successor variance to a tight range.

The D3 per-cell exponential-fit statistic also requires reinterpretation. For forward cells, $V_{\text{within-}v}(t)$ is monotonically growing, and a log-linear fit gives a clean $\lambda_v$ summarizing the rate of growth. For reverse cells, the natural prior was that $V_{\text{within-}w}(t)$ would have an interior peak followed by a contraction phase, with the contraction-phase log-linear fit ($\lambda_w^{\text{contract}}$, fit on layers $[t_w^*, L_{\text{total}} - 1]$) being the reverse analog of $\lambda_v$. The proposal's primary reverse $\lambda$ statistic was $\lambda_w^{\text{contract}}$. As Section 6.2 documents, the data refused this expectation in an instructive way: $t_w^* = L_{\text{total}} - 2 = 12$ for every cell across every seed, leaving only two layers (12 and 13) for the log-linear fit, which is not a well-defined two-point regression. We replaced $\lambda_w^{\text{contract}}$ with $\lambda_w^{\text{readout-step}} \equiv \log V_w(L_{\text{total}}-1) - \log V_w(L_{\text{total}}-2)$, the single-layer compression at the readout step, after Section 6.2's diagnostic confirmed that the multi-layer contraction phase the proposal anticipated does not exist in the data.

### 3.3 The unembedding-subspace decomposition

A new diagnostic motivated by the reverse-only structural property described in Section 1.5: the unembedding matrix $W_U$ defines a learned subspace of the residual stream, and the marginal's higher-order moments can be decomposed into a component visible to the readout (parallel) and a component invisible to it (perpendicular).

The procedure has four steps:

1. *Extract the unembedding basis.* Load the trained model checkpoint, retrieve $W_U$ via the project's `model.get_lm_head_weight()` accessor (which handles both tied and untied embeddings — for the tied-embedding case in this work, $W_U = W_E^\top$ as a tensor reference), and compute the SVD $W_U = U_W S_W V_W^\top$. The right singular vectors $V_W \in \mathbb{R}^{H \times r}$ form an orthonormal basis for the rowspan of $W_U$ (with $r = \min(|V|, H) - \text{rank deficiency}$). The SVD is computed once per checkpoint and reused across the truncation-rank sweep.

2. *Define projectors at multiple truncation ranks.* For each $d_\parallel \in \{32, 64, 128, 256\}$, define $V_\parallel = V_W[:, :d_\parallel] \in \mathbb{R}^{H \times d_\parallel}$ (the top $d_\parallel$ right singular vectors, sorted by descending singular value). The projectors are $P_\parallel = V_\parallel V_\parallel^\top$ and $P_\perp = I - P_\parallel$. We never materialize $P_\parallel$ or $P_\perp$ as $H \times H$ matrices; instead, we project on the fly by computing $Z_\parallel = X V_\parallel$ (giving a $d_\parallel$-dimensional representation) and $X_\perp = X - Z_\parallel V_\parallel^\top$ (giving an $H$-dimensional residual with zero variance along the $d_\parallel$ parallel directions).

3. *Compute per-coordinate excess kurtosis in each subspace.* For the parallel component, kurtosis is computed on the $d_\parallel$-dimensional representation $Z_\parallel$. For the perpendicular component, kurtosis is computed on $X_\perp$ restricted to directions with non-zero variance (filtered by `v_perp > 1e-12 * v_perp.max()`), avoiding spurious zero-kurtosis contributions from the degenerate parallel directions that algebraically have zero variance in $X_\perp$. Both the empirical marginal and per-cell (successor-conditioned) kurtoses are computed.

4. *Sweep across truncation ranks.* Repeat steps 2–3 for $d_\parallel \in \{32, 64, 128, 256\}$ to characterize how the parallel-vs-perpendicular gap depends on the rank of the readout-visible subspace.

The diagnostic's primary output is the gap $g(d_\parallel; t) = \kappa^{\parallel}_{\text{marg}}(t; d_\parallel) - \kappa^{\perp}_{\text{marg}}(t; d_\parallel)$ as a function of truncation rank and depth, with hypothesis N1 predicting clear negative values at the final layer that grow in magnitude with $d_\parallel$. A flat or constant gap across depth would be a construction artifact; a depth-localized negative gap at the final layer would be the substantive finding.

### 3.4 The shuffle-null protocol

The forward investigation's Section 7 documented that simple sub-conditioning experiments on next-token and position partitions produced apparent kurtosis reductions that turned out to be largely sample-partitioning artifacts after null correction. The reverse-view F4 hypothesis — the predicted-vs-actual reconstruction divergence — is structurally similar: it compares two label assignments with similar frequency distributions but different semantic content, and is potentially vulnerable to the same artifact. The shuffle-null protocol is the mandatory mitigation.

*Within-chunk frequency-preserving permutation.* For each augmented activation file, the array of partition labels (`next_ids` for reverse-actual, `pred_ids` for reverse-predicted) is permuted within each source chunk in the held-out evaluation set. A chunk boundary is detected by finding indices where the `positions` array resets to its first pilot-position value after having advanced. Within each detected chunk, labels are permuted by a deterministic per-checkpoint stream seed (combining a top-level seed with the (training-seed, training-step) pair so each checkpoint's null is reproducible and independent). The within-chunk constraint matters: shuffling globally across all pilots from all chunks would destroy both the (pilot, label) link *and* the per-chunk label frequency profile, conflating two distinct sources of structure. Within-chunk shuffling kills only the specific link we want to test for, leaving the empirical marginal unchanged.

*Frequency-preservation sanity check.* After each shuffle, the global label frequency distribution of the shuffled array is compared to that of the original array. If the two differ by even one count, the campaign aborts with an error — the shuffle is broken or the `positions` array is malformed. This check is unconditional.

*Null-corrected F4 signal.* For each checkpoint, the D5 protocol is run twice on each reverse view: once with the real partition labels and once with the within-chunk shuffled labels. The output files coexist in parallel directories with a `_shuffled` suffix. The F4 signal is then computed in three forms:
$$\Delta^{\text{raw}}(t) = \kappa^{\text{B-rev-pred-real}}(t) - \kappa^{\text{B-rev-act-real}}(t),$$
$$\Delta^{\text{null}}(t) = \kappa^{\text{B-rev-pred-shuf}}(t) - \kappa^{\text{B-rev-act-shuf}}(t),$$
$$\Delta^{\text{corr}}(t) = \Delta^{\text{raw}}(t) - \Delta^{\text{null}}(t).$$
The R2 mitigation gate, per the proposal: if the *null absorption* defined as $\eta(t) = |\Delta^{\text{null}}(t)| / |\Delta^{\text{raw}}(t)|$ averaged over the last four layers exceeds 0.7, F4 is reported as unresolved. If $\eta < 0.7$, $\Delta^{\text{corr}}$ is the primary statistic and its cross-seed sign agreement is the F4 verdict.

### 3.5 Compute budget

The discriminator passes are dominated by I/O over the per-cell flow files plus modest computation per cell. The empirical timings on our hardware (single-GPU machine with NVMe-SSD-backed file system):

| Phase | Discriminator(s) | Scope | Time |
|-------|------------------|-------|------|
| A | D1, D3, D4a | 3 views × 4 seeds × 50 checkpoints | ~1 minute total (I/O-bound on cached MVR files) |
| B | D4b | 2 reverse views × 4 seeds × final checkpoint | ~10 minutes |
| C | D5 + shuffle null | 2 reverse views × 1 seed × final, then × 4 seeds | ~5 min / ~15 min |
| D | Unembedding-subspace | 4 seeds × final × 4 truncation ranks | ~20 minutes total |
| E | $\lambda^{\text{readout-step}}$ clusters | reuses Phase A output | ~30 seconds |

Phase A is much faster than the proposal anticipated because the multiview campaign's existing per-cell flow files cache the singular value decompositions and per-cell summary statistics that the discriminators consume; the per-cell loop in `run_d3_view` is dominated by the disk read of the flow file itself, which the operating system's page cache makes effectively free on a re-run. The total wall-clock budget for the campaign is approximately 45 minutes of computation plus checkpoint-load overhead for Phase D.

### 3.6 Phase structure and gating

The five phases of the campaign correspond to the five primary deliverables and are run in order:

**Phase A** runs the parameterized D1, D3, D4a passes on all three views, including the forward view as the idempotency check. The Phase A summary block prints D1 CV(trace) interior medians for all three views, giving an immediate cross-view comparison of how heterogeneous per-cell variances are at the interior. Phase A's output is saved per view; Phase A is the foundational phase and subsequent phases depend on its output for several derived statistics.

**Phase B** runs D4b on the two reverse views at the final checkpoint of all four seeds, computing the Mardia $Z$ depth profile and the F2 verdict. Phase B writes `reverse_buildup_phase_b_verdict.json` containing the slope, ratio, and supported/not-supported flag for the F2 criterion. Phase B's verdict is the gate on Phase D's scope.

**Phase C** runs D5 with the shuffle null on the two reverse views, default scope at seed 0 only. The `--expand-c` flag extends to all four seeds. Phase C writes `reverse_buildup_phase_c_verdict.json` containing the F4 signal, null absorption, and cross-seed sign agreement.

**Phase D** runs the unembedding-subspace decomposition (`unembedding_subspace.run_unembedding_decomposition`) at the final checkpoint, sweeping $d_\parallel \in \{32, 64, 128, 256\}$ on reverse-actual. Default scope is seed 0 only; `--expand-d` extends to all four seeds. Phase D writes `reverse_buildup_phase_d_verdict.json` with the N1 verdict.

**Phase E** runs the reverse $\lambda$ cluster analysis on the D3 output from Phase A. After Section 6.2's reformulation, the cluster statistic is $\lambda_w^{\text{readout-step}}$ when its density across cells exceeds 50%, with fallback to $\lambda_w^{\text{contract}}$ or (in the worst case) $\lambda_w$ from the full-depth log-linear fit. Phase E writes `d_n2_reverse_lambda_clusters_{view}.json` with the cluster decoded contents and cross-seed stability.

The phases are not strictly sequential in compute terms — Phase E reuses Phase A output and Phase D requires a model checkpoint independent of any other phase — but they are sequential in interpretation: each phase's verdict informs the framing of the next, and the gating decisions between them are made on the basis of the JSON verdict files.

---

## 4. Phases A and B: structural and interior-layer findings

### 4.1 Phase A: cross-view structural statistics

Phase A ran the parameterized D1, D3, and D4a discriminators on all three views (forward, reverse-actual, reverse-predicted) across all four seeds and all 50 checkpoints. The forward run reproduced the legacy `model_abc_discriminator` output bit-for-bit (verified by integration test, Section 3.1), confirming the parameterization preserves the existing forward numerics.

The headline Phase A summary at the final checkpoint, averaged across the interior layers and the four seeds:

| View              | D1 CV(trace), interior median |
|-------------------|-------------------------------|
| forward           | 0.634                         |
| reverse_actual    | 0.538                         |
| reverse_pred      | 0.830                         |

The forward CV value reproduces the result reported in the forward investigation (Section 4.2 of the companion writeup, with the "0.6–0.8 across every interior layer" finding). The two reverse values are new measurements and were not predicted in advance by the proposal.

Two substantive observations from this table.

First, reverse-actual conditional variance is the *most homogeneous* across cells of the three views, with a CV substantially below the forward value. This was unexpected. Different actual-successor cells have less variation in their total within-cell variance than different input-token cells do, at the interior of the network. A possible interpretation: the diversity of upstream contexts that can precede a given successor (the source of within-cell variance) is bounded by the contextual ambiguity of the language model task, and this contextual ambiguity is roughly constant across successors — most words can be preceded by many things, in roughly similar profusion. The diversity of *downstream contexts* that follow a given input token, by contrast, has a long-tailed distribution; "the" can be followed by almost any noun, while "the" preceded by various subjects is much narrower a set. This is speculative; what the data shows is that the reverse-actual variance is more cell-homogeneous than the forward variance, and the asymmetry is large.

Second, reverse-predicted CV is the *largest* of the three, at 0.83 — even larger than the forward CV. The model's predicted-conditional cells differ from each other in total variance by even larger factors than input-conditioned cells do. This is consistent with the model having internalized a strongly successor-keyed representation: the residual-stream variance associated with committing to one predicted token differs substantially from the variance associated with committing to another.

The training-time trajectories of these statistics (Figure 4.1 in our plot output) show further structure that distinguishes the three views. Forward D1 has a small early hump near step 300 followed by a slow rise across training. Reverse-actual D1 has a sharp early *decline* from 0.8 at step 100 to ~0.5 at step 1000, then a slow rise back to ~0.55 at step 24000 — qualitatively distinct from the forward trajectory and consistent with the early-training establishment of a cell-homogeneous successor-conditional variance. Reverse-predicted D1 has a non-monotonic shape with a large late-training rise: from a noisy ~1.2 at step 100, dropping to ~0.7 at step 1000, and climbing back to 1.3 at step 24000. The late-training rise of the predicted-conditional CV is striking and has no forward analog.

The D4a per-coordinate kurtosis trajectories at an interior layer (we report layer 7 of the 14 layers) show similar cross-view asymmetry. Forward shows the familiar three-phase trajectory documented in the forward investigation: high at step 100 (rapid SVD consolidation phase), declining to a minimum around step 2000–4000, and rising back at late training. Reverse-actual shows a more pronounced version of the same U-shape — high at step 100 (2.5), a deep minimum at step 1500 (1.4), and recovery to 1.6 at step 24000. Reverse-predicted shows a noisier trajectory at early training (where the predicted-successor distribution is itself unstable and the per-cell sample sizes vary) and the same broad U-shape at later training. The minimum of the reverse-actual U-shape co-locates with the forward Phase II plateau documented in the forward investigation's Section 6, suggesting that the same training-time consolidation event regularizes conditional Gaussianity from both input-conditioned and output-conditioned perspectives.

These cross-view trajectories satisfy hypothesis S1 — within-seed reproducibility of the reverse-view statistics is comparable in fractional magnitude to that of the forward-view statistics, with cross-seed curves indistinguishable past step 1000 for all three views. S2 is inherited from the existing multiview campaign's law-of-total-variance verification and is not re-checked here.

### 4.2 Phase B: the Mardia $Z$ depth profile and the F2 verdict

Phase B ran D4b on the two reverse views at the final checkpoint of all four seeds, computing multivariate Mardia $Z$ for each cell in a shared 32-dimensional PCA subspace. The cell-mean profile (averaged across all cells in the view's token set) gives the depth-dependent multivariate Gaussianity measure.

The Mardia $Z$ depth profiles for the two reverse views at the final checkpoint are presented as Figure 4.2 in our plot output. Both views start at high $Z$ values at the embedding boundary (reverse_actual $Z \approx 65$, reverse_predicted $Z \approx 115$) and decay monotonically to $Z \approx 8$–$10$ at the final layer. The four seeds' curves are nearly indistinguishable past layer 4 — the mechanism producing this depth gradient is robust and not seed-specific.

The forward view's profile (overlaid as a black dashed reference) is qualitatively different. The forward view starts at $Z \approx -130$ at layer 0 (the embedding layer, where forward cells are degenerate point masses by construction — the computed $Z$ at this layer is therefore not a meaningful measure of Gaussianity but an artifact of zero within-cell variance) and rises sharply to $Z \approx 45$ at layer 1, then decays roughly in parallel with the reverse views through the interior. Past layer 4, all three views' $Z$ profiles converge into a single descending trajectory that ends at $Z \approx 8$–$10$ at the final layer.

The F2 verdict, applied to reverse_actual:

- Slope of cell-mean $Z$ over the last four layers (10–13): $-0.874$. Negative as required.
- Interior maximum $Z$: 70.93 (at layer 1).
- Final-layer $Z$: 8.34.
- Ratio final / interior_max: 0.118. Well below the 0.7 threshold.

F2 is supported with very strong margins on both criteria. The slope is meaningfully negative (a typical layer-to-layer change is a decrease of ~0.9 units), and the final-layer value is well under 12% of the interior maximum. The reverse-actual conditional bundles' multivariate Gaussianity is *substantially* better at the final layer than at the interior, with the regularization occurring across the entire depth of the network rather than in a single boundary step.

The reverse-predicted profile shows the same qualitative shape with slightly higher absolute values: interior max around 115 at layer 1, final-layer value around 11. The F2-style depth gradient holds for reverse-predicted as well, though the proposal's criterion was framed for reverse-actual specifically.

A subtle but substantive aspect of the F2 result: the depth gradient is not localized to the late layers. The Mardia $Z$ decreases roughly monotonically from layer 1 through layer 13, with no clear "elbow" at the readout boundary. This suggests that the unembedding's geometric pressure on conditional Gaussianity is felt gradually across the interior of the network, not as a single boundary regularization step at the final layer. The mechanism is distributed across depth.

### 4.3 The shape-vs-scale decoupling, first observation

A finding that emerges from the comparison of Phase A and Phase B results, which we expand on in Section 6.2 below:

Phase B shows that *shape* Gaussianity (Mardia $Z$, a higher-order multivariate moment summary) decreases monotonically across depth for reverse cells, with a factor of ~10 reduction from interior to readout.

Phase A's D1 statistic, by contrast, shows that *scale* of cell-wise variance ($\text{tr}(\Sigma_w(t))$, a first-order moment summary) does not contract anywhere across the interior — the cross-cell coefficient of variation of the trace stays roughly constant from layers 4 through 12, and the trace itself grows monotonically across these layers. Section 6.2's Phase E result quantifies this more precisely: the variance peak for every reverse-actual cell across every seed sits at layer 12 (the penultimate layer), and the contraction event is concentrated in a single layer-12-to-13 step.

The shape and scale of reverse-actual cells are therefore regularized by two distinct mechanisms operating at different depths. Shape is regularized continuously across the entire interior; scale is regularized in one step at the very end. This decoupling is one of the substantive findings of the campaign and reorganizes the simple "depth-localized regularization" story (Story 2) that the proposal framed as the test of F2/F3/N1.

---

## 5. Phase C: the reconstruction comparison and the F4 verdict

### 5.1 The reconstruction comparison plot

The headline Phase C output is the per-layer, per-view Gaussian-mixture reconstruction comparison (Figure 4.3 in our plot output). For each layer, four bars are shown: the empirical marginal's per-coordinate excess kurtosis (black), the forward Model B reconstruction (green), the reverse-actual Model B reconstruction (blue), and the reverse-predicted Model B reconstruction (orange). Shuffle-null reference markers (small "×" symbols) overlay each bar, indicating the kurtosis that the shuffled-label reconstruction produced at that layer.

The substantive content of the figure is the gap between the black bars and the colored ones at each interior layer. Layer 5 is the most extreme: empirical kurtosis 14.6, forward Model B reconstruction 0.5, reverse-actual 0.8, reverse-predicted 2.6. All three reconstructions undershoot the empirical by margins comparable to or larger than the forward investigation's reported margins (5–15 kurtosis units across the interior).

The Model C verdict generalizes universally. Both reverse views show the same qualitative pattern as the forward view: empirical marginal kurtosis in the 5–15 range across interior layers, Model B reconstructions sitting near 0 (forward, reverse-actual) or 2–4 (reverse-predicted), with the gap growing in magnitude at the interior depths. There is no single-variable conditioning — input token, actual successor, or predicted successor — under which the Gaussian-mixture reconstruction reproduces the marginal's heavy tails at the layers where those heavy tails are largest.

Hypothesis F1 (reverse-actual conditionals non-Gaussian at interior layers) is supported strongly: the gap between the black bar and the reverse-actual blue bar is ≥ 8 kurtosis units at every interior layer from 2 through 7, with both Phase B's Mardia $Z$ profile (interior values 30–70) and Phase C's reconstruction undershoot confirming the verdict from independent measurements.

### 5.2 F3 is refuted

The proposal's F3 hypothesis predicted that reverse-actual Model B reconstruction at *late* layers would be closer to the empirical marginal than forward Model B reconstruction, on the grounds that the unembedding's regularization would make reverse cells more Gaussian-decomposable as $t \to L_{\text{total}}-1$.

The data refute this prediction. Across layers 9 through 13, the reverse-actual Model B and forward Model B reconstructions sit at essentially the same values (both within 0.2 kurtosis units of each other and both within 0.5 of zero). Neither is markedly closer to the empirical marginal than the other. The empirical marginal's late-layer kurtosis is itself contracting (3.2 at layer 9, 1.1 at layer 11, 0.4 at layer 13), so the absolute gap to the reconstruction is smaller at the late layers — but the *relative* difference between forward and reverse-actual reconstructions is not.

The proposal's F3 prediction was structurally tied to Story 2 (depth-localized heavy tails regularized at the readout). The F3 refutation is therefore the first of two independent disconfirmations of Story 2 (the second being the weak N1 result, Section 6.1). The marginal's heavy tails are not "depth-localized away from the readout" in the simple way Story 2 imagined. They are present at all interior depths, and they are not differentially reconstructible from one view versus another.

### 5.3 F4 is supported with the surprising direction

The reverse-predicted Model B reconstruction (orange bars in Figure 4.3) sits at substantially *higher* values than the reverse-actual reconstruction (blue bars) at every interior layer. Specifically:

| Layer | Empirical | Forward B | Reverse-actual B | Reverse-pred B |
|-------|-----------|-----------|-------------------|-----------------|
| 2     | 11.6      | 0.6       | 0.6               | 4.1             |
| 3     | 9.3       | 0.5       | 0.6               | 2.7             |
| 4     | 9.4       | 0.7       | 0.7               | 3.6             |
| 5     | 14.6      | 0.5       | 0.8               | 2.6             |
| 6     | 9.6       | 0.2       | 0.6               | 2.7             |
| 7     | 7.9       | 0.5       | 0.5               | 3.3             |
| 8     | 5.8       | 0.4       | 0.5               | 2.6             |

The reverse-predicted reconstruction is closer to the empirical marginal than the reverse-actual reconstruction at every interior layer. The mixture decomposition by *the model's argmax predictions* recovers more of the marginal's higher-order structure than the mixture decomposition by *the data's true successors* does.

The F4 verdict, applied across all four seeds: the cross-seed sign agreement of $\Delta^{\text{corr}}$ is 100% at every interior layer from 1 through 13, with the only seed-level disagreement at layer 0 (where the embedding boundary is degenerate). Three seeds (1, 2, 3) plus seed 0 all agree on positive $\Delta^{\text{corr}}$ at every interior layer. The late-layer mean null absorption is 0.318, well below the 0.7 R2 gate threshold. F4 is supported.

The positive sign of $\Delta^{\text{corr}}$ is the *less expected* of the two directions the proposal envisioned for F4. The proposal's symmetric framing assigned both signs equal a priori probability:

- $\Delta < 0$ (predicted-conditional reconstruction closer to empirical than actual-conditional) means the model's argmax labels carve cleaner mixture components than the data's labels do. Read: the model has internalized a residual-stream organization keyed to its own commitment structure.
- $\Delta > 0$ (the reverse) means the data's labels carve cleaner components. Read: the model's prediction noise is itself heavy-tailed, and argmax-conditioning groups together pilots that share a predicted token but are structurally heterogeneous in the residual stream; this heterogeneity inflates the within-cell kurtosis, which lifts the reconstruction kurtosis.

The data shows $\Delta > 0$. The argmax-conditioned cells are *more* heavy-tailed than the actual-successor-conditioned cells, and this elevated heavy-tailedness propagates through the Gaussian fit to produce a closer-to-empirical reconstruction (because the fits in cells with larger empirical kurtosis "remember" more of the marginal's tail structure even after being remixed).

The substantive content is twofold. The model has not internalized an organization that produces clean Gaussian conditional bundles when partitioned by its own commitments — the opposite is true. The predicted-successor labels carve the residual stream into cells that are *more* structurally heterogeneous than the actual-successor labels do. And the model's prediction noise is genuinely heavy-tailed in a way that's measurable through the reconstruction undershoot: the model is not committing to its predictions through a clean, narrow region of residual-stream activations, but through a heavy-tailed distribution of internal states.

### 5.4 The depth profile of $\Delta^{\text{corr}}$

The cross-seed-averaged $\Delta^{\text{corr}}$ profile across depth has a characteristic shape:

| Layer | $\Delta^{\text{corr}}$ (cross-seed mean) |
|-------|-----------------------------------------|
| 0     | $-0.34$ |
| 1     | $+0.16$ |
| 2     | $+1.62$ |
| 3     | $+2.75$ |
| 4     | $+2.68$ |
| 5     | $+2.04$ |
| 6     | $+2.17$ |
| 7     | $+2.54$ |
| 8     | $+2.34$ |
| 9     | $+1.60$ |
| 10    | $+1.02$ |
| 11    | $+0.22$ |
| 12    | $+0.12$ |
| 13    | $+0.02$ |

The signal is large in the interior (peaking at layer 3 with $\Delta^{\text{corr}} = 2.75$, sustaining ~2 units across layers 3 through 8) and decays smoothly to near zero at the final layer. The depth profile is not flat — the F4 signal is concentrated at the *interior* depths, where prediction commitment is being formed, and absent at the very last layer where the readout has imposed its uniform regularization.

The null absorption profile has the inverse shape: 0.07 in the early layers, 0.04 in the interior (very small — the real signal is overwhelmingly larger than the null), and rises to 0.40 at layer 12 and 0.75 at layer 13. At the final layer, almost all of the predicted-vs-actual difference is attributable to sample-partitioning artifact, not to label-structure content. This is mechanistically coherent: by the final layer, all pilots in either partition have collapsed onto their predicted-token commitment region, and the two label assignments produce essentially the same partition of the residual stream. The substantive F4 signal lives at the depths where the commitment is being formed, not at the depth where it has already been imposed.

### 5.5 The cross-seed reproducibility

All four seeds give the same sign of $\Delta^{\text{corr}}$ at every layer from 1 through 13, and the magnitudes are remarkably consistent. The four seeds' values at layer 3 are 2.03, 2.56, 3.47, 2.95 — a coefficient of variation of approximately 20%. At layer 5, they are 2.74, 3.15, 1.83, 2.99 — CV approximately 21%. The reproducibility is well within the within-seed noise budget for a fourth-moment statistic computed on $\sim 19{,}000$ pilots per checkpoint.

The F4 finding is therefore not a property of one trained model but a property of the four trained models we have, each with independent initialization. The model's preferential organization around its own predictions, rather than around the data's true successors, is robust across training initializations.

---

## 6. Phases D and E: subspace decomposition and the readout-step bimodality

### 6.1 Phase D: the unembedding-subspace decomposition (N1)

Phase D extracts $W_U \in \mathbb{R}^{|V| \times H}$ from each seed's final checkpoint, computes its SVD, and projects the residual-stream activations into the parallel (rowspan of $W_U$) and perpendicular subspaces for the truncation rank sweep $d_\parallel \in \{32, 64, 128, 256\}$. The per-coordinate excess kurtosis is then computed in each subspace, and the gap $g(d_\parallel; t) = \kappa^\parallel - \kappa^\perp$ is reported.

The four-seed median final-layer gap, as a function of $d_\parallel$:

| $d_\parallel$ | $g(d_\parallel; 13)$ median across 4 seeds |
|---------------|---------------------------------------------|
| 32            | $-0.098$ |
| 64            | $-0.127$ |
| 128           | $-0.175$ |
| 256           | $-0.225$ |

All four gaps are negative, in the direction N1 predicted (parallel component more Gaussian than perpendicular). All four are also small in magnitude relative to the proposal's $-0.5$ criterion. **The N1 verdict, applied strictly, is *not supported*.** The gap is non-zero and signed correctly, but small.

Two features of the data argue against dismissing this as null result and for the qualified interpretation:

*The cross-seed reproducibility is extraordinary.* The four seeds' gap values at $d_\parallel = 256$ are $-0.223$, $-0.225$, $-0.230$, $-0.220$ — agreement to two decimal places across independent training runs. The gap is a population-level property of the architecture, not a noisy estimate at single-seed scale.

*The gap scales monotonically with $d_\parallel$.* Going from rank 32 to rank 256, the gap grows in magnitude from $-0.098$ to $-0.225$, with monotone intermediate values. A random low-rank projection of the residual stream would not produce a kurtosis gap that scales coherently with the projection's rank. The monotone scaling rules out two boring interpretations:

- *Noise.* If the gap were sample-size noise, we would see scatter; we see a clean monotone with cross-seed agreement.
- *Few-direction localization.* If only the top 8 unembedding directions were special (corresponding to a handful of high-frequency vocabulary tokens), the gap at $d_\parallel = 32$ would already be near its maximum and would *shrink* as additional lower-frequency directions were folded into the parallel subspace. Instead, the gap grows with $d_\parallel$. The readout's geometric pressure on conditional Gaussianity is therefore distributed across hundreds of directions, not concentrated in a small handful.

The correct interpretation: N1's prediction is right in *direction* but small in *magnitude*. The readout does impose some regularization on the parallel-vs-perpendicular comparison, but the regularization is weak enough that the marginal's full $\sim 9$–$14$ kurtosis units at the interior layers are not isolated into the orthogonal complement. Most of the marginal's non-Gaussianity at the final layer lives equally in both subspaces. The parallel subspace is, at best, marginally cleaner.

Together with F3's refutation (Section 5.2), N1's weak-support gives two independent disconfirmations of Story 2 (the depth-localized refinement of Model C). The marginal's heavy tails are not preferentially located at any one layer or in any one subspace; they are distributed throughout the residual stream's interior representation.

### 6.2 Phase E: the readout-step bimodality (N2)

Phase E was initially formulated to test bimodality in $\lambda_w^{\text{contract}}$, the log-linear fit of $V_{\text{within-}w}(t)$ on the descending phase $[t_w^*, L_{\text{total}} - 1]$, where $t_w^*$ is the per-cell peak-variance layer. The first Phase E run produced an all-NaN cluster output: only one of 19 reverse-actual cells gave a non-NaN $\lambda^{\text{contract}}$ value, and the k-means returned $[\text{nan}, \text{nan}]$ for the cluster centers.

A diagnostic on the underlying data revealed why. For *every* cell across *every* seed, $t_w^* = 12$ — the variance peak sits at the penultimate layer, leaving only the layer-12-to-13 segment as the "descending phase." A two-point log-linear regression is not well-defined, and the contraction-fit code correctly returned NaN. Reformulating $\lambda$ for the reverse view was necessary.

The reformulation replaces $\lambda_w^{\text{contract}}$ with the single-layer compression statistic:
$$\lambda_w^{\text{readout-step}} \equiv \log V_w(L_{\text{total}}-1) - \log V_w(L_{\text{total}}-2).$$

This statistic captures exactly the contraction event that the data shows actually exists: a single-layer variance compression at the readout, not a multi-layer descending phase. Negative values indicate the readout compresses the cell's bundle; values near zero indicate no compression; positive values would indicate continued growth.

The reformulation has substantive content beyond the technical fix. *Reverse-actual within-cell variance does not contract gradually toward the readout.* It grows monotonically through layer 12, and contracts only in the single layer-12-to-13 step. The unembedding's effect on variance scale is concentrated entirely in the final transformer block (and the final-norm-plus-unembedding readout itself). This is the *shape-vs-scale decoupling* mentioned in Section 4.3: shape regularization (Mardia $Z$) is gradual across all interior layers; scale regularization (variance trace) is one-shot at the boundary.

With $\lambda^{\text{readout-step}}$, the Phase E cluster analysis runs cleanly. The results:

**Reverse-actual** ($\lambda^{\text{readout-step}}$ density 100%, cluster centers $[-0.74, -0.49]$, cross-seed stability 73.7%):

*Cluster 0 (more compressed, $\lambda \approx -0.74$, 15 tokens):* `the`, `.`, `,`, `of`, `and`, `to`, `\n`, `a`, `in`, ``, `is`, `that`, `are`, `2`, `for`.

*Cluster 1 (less compressed, $\lambda \approx -0.49$, 4 tokens):* `-`, `1`, `0`, `s`.

**Reverse-predicted** ($\lambda^{\text{readout-step}}$ density 100%, cluster centers $[-0.93, -0.63]$, cross-seed stability 100%):

*Cluster 0 (more compressed, $\lambda \approx -0.93$, 10 tokens):* `the`, `\n`, `and`, `a`, `is`, `are`, `that`, `be`, `The`, `The` (two distinct BPE ids for `The` at sentence-start vs mid-sentence).

*Cluster 1 (less compressed, $\lambda \approx -0.63$, 10 tokens):* `.`, `,`, `of`, `to`, `in`, ``, `0`, `-`, `1`, `s`.

The two views' cluster splits are grammatically interpretable but cut grammar at different boundaries.

Reverse-predicted's cluster boundary is the *content word vs structural element* line. The compressed cluster contains all the function words (`the`, `a`, `and`, `is`, `are`, `that`, `be`, `The`) plus the newline (a sentence-end marker, functioning like a content commitment). The less-compressed cluster contains punctuation marks (`.`, `,`), short prepositions (`of`, `to`, `in`), digits (`0`, `1`), the dash, the orphan possessive fragment (`s`), and the empty/whitespace token. The grammatical interpretation: when the model commits to a content-bearing word, it compresses its internal state by a factor of $\exp(-0.93) \approx 0.40$ across the readout step. When it commits to a structural element, it compresses by only $\exp(-0.63) \approx 0.53$. Content-word commitment is a higher-compression event than punctuation/structural-token commitment.

Reverse-actual's cluster boundary is the *complete lexical unit vs structural fragment* line. The compressed cluster contains content words, function words, sentence punctuation, the newline — all tokens that are "complete units when they appear" in their respective contexts. The less-compressed cluster contains only four items: the dash, the digit-fragment tokens `0` and `1`, and the orphan `s`. These are tokens that typically appear as parts of larger lexical units (`-` joining compounds, digits as fragments of multi-digit numbers, `s` as a possessive or pluralization fragment).

The two views agree on cluster-0 membership for 7 of the 19 forward-set tokens (`the`, `\n`, `and`, `a`, `is`, `are`, `that`) and on cluster-1 membership for 4 tokens (`-`, `0`, `1`, `s`). They disagree on the cluster membership of 5 tokens (`.`, `,`, `of`, `to`, `in`, ``) and 1 numeric (`2`), all of which reverse-actual puts in cluster 0 (compressed) and reverse-predicted puts in cluster 1 (less compressed).

The disagreement is itself the substantive content. The forward investigation found a bimodal $\lambda_v$ structure with a connective/structural split. The reverse-predicted bimodality is a content-word/structural-element split. The reverse-actual bimodality is a complete-unit/fragment split. *The three views recover three different basis-invariant grammatical structures from the same trained model*, each emphasizing a different aspect of how grammar organizes the residual stream.

### 6.3 The cross-seed disagreement in reverse-actual

The 73.7% cross-seed stability of reverse-actual is below the proposal's 85% criterion. Inspecting the per-seed cluster labels for the disagreeing tokens reveals that the disagreement is not random: it is localized to the exact tokens at the grammatical boundary between the two views' cluster splits.

The five tokens with cross-seed disagreement are `.`, `of`, `to`, `''` (empty/whitespace), and `2`. Their per-seed labels (seeds 0 through 3):

| Token | Seed 0 | Seed 1 | Seed 2 | Seed 3 |
|-------|--------|--------|--------|--------|
| `.`   | 1      | 1      | 0      | 0      |
| `of`  | 1      | 1      | 1      | 0      |
| `to`  | 1      | 1      | 0      | 0      |
| `''`  | 0      | 1      | 0      | 0      |
| `2`   | 1      | 1      | 0      | 0      |

Three of the five tokens (`.`, `to`, `2`) have the exact same disagreement pattern: seeds 0 and 1 assign them to cluster 1 (less compressed, with the structural fragments), and seeds 2 and 3 assign them to cluster 0 (more compressed, with the complete units). This is not random seed-level boundary noise — it is two consistent "dialects" of where the cluster boundary falls. Two of the four seeds drew the grammatical split closer to the reverse-predicted line (punctuation/prepositions structural), and two drew it closer to the complete-units line.

The fourteen stable tokens partition cleanly; the five unstable tokens occupy the exact grammatical region where the two views' interpretations diverge. The reverse-actual bimodality is therefore *not* simply a less stable version of the reverse-predicted bimodality — it is a different grammatical cut, with the four seeds split 2-vs-2 on how strictly to draw it.

The reverse-predicted 100% stability indicates that the model's prediction-keyed organization has converged to a uniquely-determined grammatical interpretation across initializations, while its data-tracking organization remains seed-sensitive at the cluster boundary. The model's commitment structure is more reproducible than its representation of what the data actually does next.

### 6.4 The N2 verdict, qualified

The strict N2 verdict criterion (85% cross-seed stability with grammatically interpretable cluster membership) is satisfied for reverse-predicted but not for reverse-actual. We report the qualified verdict: **N2 fully supported for reverse-predicted (100% stability, clean content-word/structural-element split), provisionally supported for reverse-actual (73.7% stability, complete-unit/fragment split, with the four-seed disagreement localized to grammatical-boundary tokens in two consistent dialects).**

The reverse-predicted result is the strongest possible version of N2: every token has consistent cluster assignment across all four seeds, and the grammatical interpretation of the split is clean and content-word-vs-structural-element. As a basis-invariant signature of grammatical structure in the model's representation, it is at least as strong as the forward investigation's $\lambda_v$ bimodality and arguably stronger (the forward bimodality had 85% stability with three unstable boundary tokens; reverse-predicted has 100% stability).

---

## 7. The substantive findings

The seven primary hypotheses (S1–S2, F1–F4, N1–N2) and the auxiliary hypothesis (D1) have been resolved by Phases A through E. Their verdicts:

| Hypothesis | Verdict | Substantive content |
|------------|---------|---------------------|
| S1 | Supported | Reverse-view per-cell statistics are reproducible across seeds; within-seed std is comparable to forward-view std for the corresponding statistic. |
| S2 | Inherited | Law of total variance verified by existing multiview campaign. |
| F1 | Supported | Reverse-actual conditionals are strongly non-Gaussian at interior layers (Mardia $Z = 30$–$70$; per-cell kurtosis 5–15 vs ~1 marginal). The Model C verdict generalizes from forward to reverse-actual. |
| F2 | Supported | Reverse-actual Mardia $Z$ decreases monotonically toward the final layer (slope $-0.87$, final/interior ratio 0.12). The unembedding's geometric pressure regularizes conditional shape Gaussianity gradually across the interior, with no clear elbow at any one layer. |
| F3 | Refuted (in the original formulation) | Reverse-actual Model B reconstruction is *not* closer to the empirical marginal than forward Model B reconstruction is at late layers. The "depth-localized heavy tails" story (Story 2) is wrong; non-Gaussianity is present at all interior depths and is not differentially reconstructible from one partition. |
| F4 | Supported (in the surprising direction) | $\Delta^{\text{corr}}(t) > 0$ at every interior layer, with 100% cross-seed sign agreement. The predicted-conditional Model B reconstruction is *farther* from the marginal than the actual-conditional reconstruction, meaning the argmax labels carve cells that are heavier-tailed than the true-successor labels do. The model has not internalized an organization that produces clean Gaussian decomposition keyed to its own predictions; the opposite is true. |
| N1 | Weakly supported in direction, refuted in magnitude | The final-layer parallel-vs-perpendicular kurtosis gap is consistently negative (parallel cleaner than perpendicular) and scales monotonically with $d_\parallel$, but its magnitude is small ($-0.22$ at $d_\parallel = 256$, vs the proposal criterion of $-0.5$). The readout's geometric regularization is real but weak; it does not concentrate the marginal's heavy tails into the orthogonal complement. |
| N2 | Fully supported (reverse-pred), provisionally supported (reverse-actual, with structured cross-seed disagreement) | Reverse-predicted $\lambda^{\text{readout-step}}$ has bimodal structure with 100% cross-seed stability and content-word-vs-structural-element interpretation. Reverse-actual has the same bimodality with 73.7% stability and complete-unit-vs-fragment interpretation; the cross-seed disagreement is localized to grammatically-boundary tokens in two consistent "dialects" (seeds 0&1 vs seeds 2&3). |
| D1 | Pending formal test | Reverse-view conditional kurtosis trajectories show the same broad U-shape as the forward view (U-minimum near step 1500, consistent with the forward investigation's Phase II plateau). Formal threshold-crossing comparison not computed. |

The summary in narrative form: the proposal's central conjecture — that the reverse measurement would discriminate among multiple stories consistent with the forward Model C verdict — was correct, and the discrimination came out cleanly. We now collect the three substantive findings that follow and their broader interpretation.

### 7.1 Finding 1: shape-vs-scale decoupling at the readout

The combination of F2 (supported strongly), F3 (refuted), and N1 (weakly supported in direction only) tells a coherent story about what the readout does and does not regularize.

The readout regularizes the shape of conditional bundles substantially. Per-cell Mardia $Z$ drops from 70 at the interior to 8 at the final layer — an order-of-magnitude reduction in multivariate non-Gaussianity. This regularization is not localized to the final transformer block; it is gradual across the entire interior of the network, with each layer reducing $Z$ by a roughly constant fractional amount.

The readout does not regularize the *scale* of conditional bundles in any depth-distributed way. Within-cell variance grows monotonically through layer 12 and then drops in a single layer-12-to-13 step. The unembedding-step compression $\lambda_w^{\text{readout-step}}$ is negative for every cell, with values ranging from $-1.18$ (most-compressed content words) to $-0.40$ (least-compressed fragments). The compression event is concentrated entirely in the readout itself.

The readout also does not concentrate the marginal's heavy tails into its orthogonal complement in any strong sense. The parallel-vs-perpendicular kurtosis gap is consistently negative but small ($-0.22$ at large $d_\parallel$), well below the magnitude needed to localize the marginal's $\sim 10$ kurtosis units to one subspace.

Two distinct mechanisms regularize reverse cells at the readout boundary:

*A shape regularization*, distributed across the interior depth of the network, manifesting as the gradual Mardia $Z$ depth gradient. This regularization is mechanistically responsible for F2's support but does not produce N1's strong-form result. The mechanism by which it operates is not directly visible in our data — we observe its effect (Mardia $Z$ decreases) but not its cause.

*A scale regularization*, concentrated at the readout step, manifesting as the negative $\lambda^{\text{readout-step}}$ for every cell. This regularization is mechanistically tied to the final-norm-plus-unembedding-plus-softmax stack: by the time the model produces logits, every committed-to successor has to fit through a low-rank readout, which forces the conditional bundle to contract along the readout-relevant directions.

These two mechanisms are *not* the same thing. A model could in principle regularize shape but not scale, or scale but not shape, or both, or neither. Our trained model regularizes both, but in different ways and at different depths.

### 7.2 Finding 2: label-structure-organized non-Gaussianity

The F4 finding ($\Delta^{\text{corr}}(t) > 0$ at every interior layer, 100% cross-seed sign agreement) means that the marginal's heavy tails are not just a property of the depth at which the residual stream is measured but also of *which labeling* is used to partition the residual stream into cells. Partitioning by the model's argmax predictions produces cells with more within-cell heavy tails than partitioning by the data's true successors.

A natural interpretation: the model's argmax distribution groups together pilots that share a single committed-to next token but arrive at that commitment from structurally different upstream contexts. These structurally different contexts produce structurally different residual-stream activations, and pooling them within one prediction-conditioned cell inflates the cell's higher-order moments. The data's true-successor distribution does the same averaging across the same upstream contexts, but the labels are not the model's own — they are the held-out text's continuation, which may or may not match the model's prediction.

When the model is *correct* (predicts the actual successor), the two labelings produce the same cell membership for that pilot. When the model is *wrong*, the two labelings produce different cell memberships: the pilot lands in cell $\hat w$ under reverse-predicted and in cell $w \ne \hat w$ under reverse-actual. The 19,500 pilots in our held-out evaluation are not all consistently predicted by the model; the per-pilot mismatch rate at step 24000 is approximately 40% (the model's top-1 accuracy on FineWeb-Edu at the end of Phase 1 training is approximately 60%). The F4 finding lives in this 40%.

The structurally different upstream contexts that lead to the same predicted token but different actual tokens are the source of the predicted-cells' heavy tails. The reverse-predicted partition aggregates over more such contexts per cell than the reverse-actual partition does. The reverse-actual partition is "cleaner" in the sense that it pools pilots that genuinely share a downstream observation, while the reverse-predicted partition pools pilots that share an internal commitment.

This is the third story listed in Section 1.6 — the marginal's non-Gaussianity is partly organized by which partition labels are applied to the pilots. F4 supports this story strongly and rules out the simpler "labels are interchangeable" story under which both reverse views would give the same reconstruction kurtosis.

### 7.3 Finding 3: grammatical structure replicates across the input-output axis

The forward investigation found a bimodal $\lambda_v$ structure across input tokens, with 85% cross-seed stability and a connective/structural-token grammatical interpretation. The reverse-predicted $\lambda_w^{\text{readout-step}}$ bimodality reported here has 100% cross-seed stability and a content-word/structural-element grammatical interpretation. The reverse-actual $\lambda_w^{\text{readout-step}}$ bimodality has 73.7% cross-seed stability and a complete-unit/fragment interpretation, with the four-seed disagreement localized to grammatical-boundary tokens.

The three bimodalities cut grammar at three different boundaries, but they share a common form: a basis-invariant per-cell statistic, computed from the geometry of the cell's variance dynamics, exhibits a two-cluster structure across the cells in the view's token set, with the clusters having grammatically-interpretable membership. This is a robust finding across the input-output axis: *whichever way we partition the residual stream by token identity, the resulting per-cell variance dynamics inherit grammatical structure from the partition variable*.

The three views' cluster splits do not coincide because they probe different aspects of the model's representation. The forward view's $\lambda_v$ measures how the within-input variance *grows* with depth — a property of the dynamics that follow the embedding. The reverse-actual and reverse-predicted views' $\lambda^{\text{readout-step}}$ measures how the within-output variance *contracts* at the readout — a property of the readout itself. The grammatical content of the dynamics differs because the mechanisms differ.

The reverse-predicted bimodality's 100% cross-seed stability is the cleanest result we have on basis-invariant grammatical structure. It is a finding worth reporting on its own: at the end of training, a 150M-parameter transformer trained on FineWeb-Edu has organized its residual stream such that the variance compression at the readout step partitions its argmax-predictions into a "content words" cluster (compression factor $\sim 0.40$ per layer-step) and a "structural elements" cluster (compression factor $\sim 0.53$ per layer-step), with cluster membership identical across four independent training initializations.

---

## 8. Implications for the broader research program

### 8.1 The reverse measurement closes one set of stories and opens another

The proposal framed the reverse measurement as a discriminator among three stories consistent with the forward Model C verdict. The data supports a clear verdict on each:

**Story 1 (uniform-across-depth non-Gaussianity)** predicted that conditioning on any single variable would leave the marginal's heavy tails in place at every layer of both reverse views. The data partially supports this story: F1 holds, and F3 refutes the depth-localized refinement. But the reverse views *do* show a clean Mardia $Z$ depth gradient (F2 supported), so "uniform across depth" is too strong; the *shape* component of the non-Gaussianity is depth-graded even though the *reconstruction-undershoot* component is not.

**Story 2 (depth-localized non-Gaussianity regularized at the readout)** predicted that the marginal's heavy tails would be cleanly localized at the interior depths and absent at the readout boundary, with the reverse-actual reconstruction being noticeably better at the late layers than the forward reconstruction is. The data refutes this story on two independent grounds (F3 refuted, N1 weakly supported in direction only with magnitude well below the criterion). The story is wrong.

**Story 3 (label-structure-organized non-Gaussianity)** predicted that actual-successor and predicted-successor conditional reconstructions would diverge in a way robust across seeds. The data supports this story strongly (F4 with 100% cross-seed sign agreement), and the *direction* of the divergence (positive $\Delta^{\text{corr}}$, the predicted-conditional cells being heavier-tailed than the actual-conditional cells) is the substantive finding. The marginal's heavy tails are partly a property of which labeling is applied.

Two of the three pre-registered stories are ruled out (Story 1 with qualifications, Story 2 cleanly), and one is supported with a clear sign of the divergence. The story we did not pre-register but the data points toward is a hybrid: the marginal's higher-order structure is *neither* fully uniform across depth (the shape regularization is real) *nor* localized to any single layer or subspace (F3 and N1 rule those out) *but rather* partly determined by which partition labels are applied to the pilots, with the partition variables themselves exhibiting basis-invariant grammatical structure (N2).

### 8.2 The pair-conditional analysis is now well-motivated

The original forward investigation closed with three live possibilities for what generates the Model C heavy tails (Section 9 of the forward writeup): intrinsic heavy-tailed conditionals, rare-extreme-context mixture, and high-order context dependency. The reverse build-up project did not directly test any of these — its purpose was to ask whether the forward Model C verdict generalizes to single-variable output conditioning, and to discriminate among stories about *where* the heavy tails live.

The combination of F3 refuted, N1 weakly supported, and F4 strongly supported with the surprising direction points at the third possibility (high-order context dependency) more sharply than the forward investigation alone did. The forward investigation's null-correction analysis showed that single-variable next-token or position partitions do not explain the heavy tails; the reverse build-up shows that single-variable actual-successor or predicted-successor partitions also do not explain them, and that the *difference* between two single-variable partitions (F4) is itself informative. This makes the case for a pair-conditional analysis — partitioning the marginal on the joint $(v, w)$ or $(v, \hat w)$ — substantially stronger than it was after the forward investigation.

A concrete target for pair-conditional analysis. The F4 finding gives a specific object the pair-conditional view should explain. The reverse-predicted cells $\hat F_{\hat w}$ are heavier-tailed than the reverse-actual cells $F_w$ at every interior layer. If the heavy tails decompose at the pair-conditional level — that is, if $(v, w)$ joint conditioning produces approximately Gaussian sub-cells whose mixture reproduces the marginal cleanly — the F4 asymmetry should manifest as a difference in the structure of the pair-conditional partition for actual vs predicted successors. The companion Markovianity writeup describes a pair-conditional decomposition in terms of the Markovian transition structure of the residual stream's per-layer update; the F4 finding gives a target for that decomposition to predict.

### 8.3 The shape-vs-scale decoupling as a mechanistic claim

The finding that the readout regularizes shape and scale through two different mechanisms operating at different depths is a substantive mechanistic claim about the trained transformer. It is testable in several directions that are not directly explored here:

*Architectural ablations.* Does removing the final-norm operation eliminate the scale regularization while leaving the shape regularization intact? The final norm operates pointwise on the residual stream and is not informative about cellwise distinctions, so naive expectation is yes; but the empirical result is not in our data. A clean ablation experiment would compare $\lambda^{\text{readout-step}}$ with and without final norm and report whether the magnitude changes.

*Untied embeddings.* The N1 weak result is potentially confounded by tied embeddings — the readout's rowspan is identical to the embedding's columnspan, so the parallel subspace at the readout is the same subspace the embedding produces at $t = 0$. With untied embeddings, the two subspaces differ, and the N1 gap would be measured against a learned-rather-than-inherited orthogonal complement. The R4 mitigation in the proposal addresses this concern by reporting the gap across all 14 layers; the absence of a "gap growing toward the final layer" pattern in the data is consistent with N1 being a weak rather than artifact-induced effect, but a clean untied-embedding experiment would settle the question.

*Other training stages.* The forward investigation's three-phase structure (Phase I rapid consolidation, Phase II plateau, Phase III restructuring) raises the question of when the shape regularization begins to dominate the scale regularization. Our data has a Phase A trajectory showing reverse-view kurtosis profiles across training, and the U-shape we observe (minimum at step ~1500, consistent with the forward Phase II) suggests that the shape regularization develops during Phase III alongside the marginal's heavy tails themselves. A formal study would compute the Mardia $Z$ trajectory at the final layer across all 50 checkpoints and report when the depth gradient first becomes monotonically negative.

### 8.4 The model's prediction-keyed organization as a finding

The F4 finding has independent interest beyond the F4-vs-N1-vs-F3 discrimination it was designed for. It says that the model has organized its residual stream around its own commitment structure more tightly than around the data-generating distribution, and that this organization is robust across training initializations (100% cross-seed sign agreement on $\Delta^{\text{corr}}$). The N2 finding sharpens this: reverse-predicted $\lambda^{\text{readout-step}}$ has 100% cross-seed cluster stability, while reverse-actual has 73.7% with structured cross-seed disagreement.

The model's argmax-organization is *more reproducible across initializations* than its data-tracking organization. This is consistent with a broader pattern in supervised neural training: the model's predictions converge to a relatively unique solution given enough training and capacity, while the model's representation of what the data is doing remains seed-sensitive in the directions where the data underdetermines the solution. The cross-seed dialect structure we observed in the reverse-actual cluster boundary (seeds 0&1 vs seeds 2&3) is a concrete example of this representational underspecification — two distinct, reproducible-within-pairs solutions to the question of which tokens count as "complete units" vs "fragments."

This finding is not a direct contribution to the Lines of Thought framework's residual-stream geometry, but it is a contribution to the broader question of what a trained transformer represents. The model has a more reproducible answer to "what should I predict next?" than to "what is the data going to do next?", and the difference between these answers is visible at the level of the residual stream's basis-invariant variance dynamics.

---

## 9. Limitations and open questions

### 9.1 What is not tested

Several diagnostics that would round out the picture are not formally tested here.

*The D1 training-time co-location hypothesis.* The forward investigation found that conditional non-Gaussianity emerges during Phase III, with the Mardia $Z$ rise crossing $Z > 10$ at a checkpoint that co-locates within ~500 training steps of the marginal kurtosis minimum (around step ~1500). The reverse analog — does the reverse-view Mardia $Z$ rise at the same checkpoint as the forward — is suggested by the U-shapes in the Phase A D4a trajectories but not formally tested. A simple threshold-crossing analysis on the existing Phase A output would close this.

*The pair-conditional decomposition itself.* This work motivates the pair-conditional analysis but does not perform it. The companion Markovianity writeup sketches the framework; integrating its predictions with the F4 finding from this work is a natural next step.

*The N1 result at intermediate training stages.* Phase D was run only at the final checkpoint. The training-time emergence of the readout-vs-orthogonal kurtosis gap could be informative — does the gap appear suddenly at some training stage, or does it develop gradually alongside the model's prediction accuracy? Phase D could be re-run at the four-phase representative checkpoints from the forward investigation; this is computationally cheap (Phase D takes ~5 minutes per seed per checkpoint).

*The cross-seed cluster dialects.* The 2-vs-2 dialect structure on the reverse-actual cluster boundary is a finding that deserves followup at a larger number of seeds. With only four seeds we cannot distinguish between "two stable dialects exist" and "boundary tokens flip randomly with some bias." Training 8 or 16 seeds would settle this question.

### 9.2 Limitations of the experimental design

*Model scale.* This work uses a single 150M-parameter Llama-style transformer. The Lines of Thought paper's results were reported on larger models. Whether the F4 and N2 findings reproduce at the multi-billion-parameter scale is unknown. The forward investigation's grammatical bimodality has been reproduced anecdotally in larger models (per personal communication, though not formally) and the basis-invariant character of the finding makes scale-replication plausible. The N1 weak-support result might change with scale — the unembedding's rowspan is a larger fraction of $\mathbb{R}^H$ in smaller models — but the direction of any scale dependence is hard to predict.

*Tied embeddings.* The N1 hypothesis is confounded by tied embeddings, as discussed in Section 8.3. Untied-embedding architectures (Pythia, GPT-NeoX, some Mistral variants) would give a cleaner test of N1's geometric claim.

*Single corpus.* The training corpus is FineWeb-Edu (sample-10BT). Whether the F4 finding reproduces on a different distribution of language data is not tested. Particular concern: the F4 sign might invert on a corpus where the model's prediction accuracy is much higher, since the F4 signal lives in the prediction-mismatch fraction of pilots and decays as the mismatch rate decreases. A controlled study across corpora would settle this.

*Single architecture family.* The trained model is GeLU-activated rather than the SwiGLU default for Llama-style models. The forward investigation's findings reproduce across both activation choices (per the existing project artifacts), but the reverse build-up has not been re-run on the SwiGLU variant.

### 9.3 What we are confident in versus what remains tentative

We are confident in:
- The Model C verdict generalizes to both reverse views at all interior layers.
- The Mardia $Z$ depth gradient is robust and large (factor of 10 reduction across depth).
- The F4 sign is robustly positive at every interior layer of all four seeds.
- The N2 readout-step bimodality holds with clean grammatical interpretation on both reverse views.
- The N1 gap is small in magnitude but consistent in direction and scales monotonically with $d_\parallel$.

We are tentative about:
- The mechanistic interpretation of F4 (the "model commits to predictions through a heavy-tailed distribution of internal states" framing). The data supports the sign and reproducibility of the effect; the mechanistic explanation is a hypothesis.
- The 2-vs-2 dialect structure in reverse-actual cluster boundary. With only four seeds, this is suggestive rather than confirmed.
- The generalizability beyond this single trained model. The cross-seed reproducibility within our run is strong; the cross-run reproducibility is not formally tested.

### 9.4 What this work does not claim

This work does not claim:
- That the Lines of Thought framework is "wrong." The framework describes the marginal accurately. The findings here concern the relationship between the marginal description and any microscopic conditional description, and the answer is that no clean single-variable conditional description exists.
- That a pair-conditional or higher-order conditional description necessarily exists. The pair-conditional approach is motivated but not tested; its outcome is genuinely open.
- That the F4 finding has direct interpretability implications. The F4 finding is a basis-invariant geometric statement about the residual stream. Translating it into a mechanistic interpretability claim (e.g., a circuit-level account) would require additional work outside the scope of this project.

---

## 10. Closing

The reverse build-up project extends the multiview campaign's input-conditioned Models A/B/C analysis to two output-conditioned views (actual-successor and predicted-successor) and adds one diagnostic with no forward analog (the unembedding-subspace decomposition). The seven pre-registered hypotheses are resolved by a five-phase campaign whose total compute is approximately 45 minutes.

The substantive findings reorganize the picture from the forward investigation:

The Model C verdict generalizes universally. No single-variable conditioning recovers a clean Gaussian-mixture decomposition of the marginal. This rules out the simplest microscopic refinement of the Lines of Thought framework along any single observable axis.

The readout regularizes shape and scale through two distinct mechanisms operating at different depths. Mardia $Z$ multivariate Gaussianity is regularized gradually across the entire interior of the network. Within-cell variance scale is regularized in a single layer-12-to-13 contraction step at the readout itself. These mechanisms are independent and act on different aspects of the conditional bundles' geometry.

The model's prediction-organization is more strongly imprinted on the residual stream than its data-tracking organization. The argmax-conditioned cells are heavier-tailed than the actual-successor cells at every interior layer (F4 supported with 100% cross-seed sign agreement, positive direction). Their bimodal $\lambda^{\text{readout-step}}$ structure is also more cross-seed-reproducible (100% vs 73.7%). Both findings point at a representational organization that prioritizes the model's own commitment structure over the data-generating distribution.

These results, combined with the forward investigation's prior findings, motivate moving from single-variable conditional analyses (now exhausted in three views) to pair-conditional analyses, with the F4 asymmetry as a concrete target for any pair-conditional decomposition to explain. The companion Markovianity writeup sketches the framework. The reverse build-up project closes the single-variable phase of the multiview campaign and opens the next phase with a definite next-step direction.

What the marginal's heavy tails *are*, in the sense of a clean microscopic description, remains an open question after this work. What we know now is that they are not a single-variable mixture phenomenon along any of the three axes we have tested. The Lines of Thought framework's universality is a central-limit-theorem phenomenon over a population of conditional bundles whose structure is neither Gaussian, nor simply mixed, nor straightforwardly localizable along any single observable dimension.

---

## Appendix A: Notation summary

| Symbol | Definition |
|--------|------------|
| $x_t$ | Residual-stream vector at layer $t$ of the model. |
| $L_{\text{total}}$ | Total number of layers (14 in our 150M model: embedding + 12 transformer blocks + post-final-norm = 14 distinct "states" along the residual stream). |
| $H$ | Hidden dimension (896 in our model). |
| $V_W$ | Right singular vectors of the unembedding matrix $W_U$. Columns span the rowspan of $W_U$. |
| $E_v$ | Forward (input-conditioned) cell at layer $t$ for token $v$: $\{x_t^{(k)} : \text{input}_k = v\}$. |
| $F_w$ | Reverse-actual cell at layer $t$ for successor $w$: $\{x_t^{(k)} : \text{next}_k = w\}$. |
| $\hat F_{\hat w}$ | Reverse-predicted cell at layer $t$ for predicted successor $\hat w$: $\{x_t^{(k)} : \text{pred}_k = \hat w\}$. |
| $\pi_z$ | Empirical frequency of conditioning value $z$ in the held-out pilot set. |
| $\mu_z(t), \Sigma_z(t)$ | Conditional mean and covariance at layer $t$ for cell $z$. |
| $\lambda$ | Exponential growth rate of $\mathrm{tr}(\Sigma(t))$ in depth. Marginal $\lambda$ from Lines of Thought; per-cell $\lambda_v$ for forward cells. |
| $\lambda_w^{\text{contract}}$ | Proposed reverse analog: log-linear fit on the descending phase of $V_{\text{within-}w}(t)$. Refuted by data (Section 6.2). |
| $\lambda_w^{\text{readout-step}}$ | Reformulated reverse statistic: $\log V_w(L_{\text{total}}-1) - \log V_w(L_{\text{total}}-2)$. |
| Mardia $Z$ | Standardized multivariate kurtosis test statistic. $|Z| > 2$ rejects multivariate Gaussianity at the 5% level. |
| $d_\parallel$ | Truncation rank of the unembedding-subspace decomposition. |
| $\Delta^{\text{raw}}(t)$ | F4 raw signal: $\kappa^{\text{B-rev-pred}}(t) - \kappa^{\text{B-rev-act}}(t)$. |
| $\Delta^{\text{null}}(t)$ | F4 null signal: same difference computed with within-chunk shuffled labels. |
| $\Delta^{\text{corr}}(t)$ | F4 null-corrected signal: $\Delta^{\text{raw}}(t) - \Delta^{\text{null}}(t)$. |

## Appendix B: File outputs

Output files written by the campaign to `<run-dir>/multiview/model_abc/`:

```
d1_token_cv_{forward,reverse_actual,reverse_pred}.npz
d3_per_token_fits_{forward,reverse_actual,reverse_pred}.npz
d4a_kurtosis_{forward,reverse_actual,reverse_pred}.npz
d4b_gaussianity_{reverse_actual,reverse_pred}/seed{S}_step{T:08d}.npz
d5_reconstruction_{reverse_actual,reverse_pred}/seed{S}_step{T:08d}.npz
d5_reconstruction_{reverse_actual,reverse_pred}_shuffled/seed{S}_step{T:08d}.npz
d_n1_unembedding_subspace_reverse_actual/seed{S}_step{T:08d}_d{d_par:04d}.npz
d_n2_reverse_lambda_clusters_{reverse_actual,reverse_pred}.json
reverse_buildup_phase_{b,c,d}_verdict.json
figures/reverse_buildup_fig_4_{1,2,3,4}_*.png
figures/d_n2_reverse_lambda_clusters.png
figures/reverse_buildup_fig_4_6_colocation.csv
```

The verdict JSON files are the primary machine-readable outputs and contain the headline statistics for the F2, F4, and N1 verdicts. The figure files are the primary human-readable outputs and visualize the discriminator dashboards (4.1), Mardia $Z$ profiles (4.2), reconstruction comparison (4.3), and unembedding-subspace decomposition (4.4). The N2 cluster JSON files contain decoded token lists and per-seed cluster assignments for the reverse $\lambda^{\text{readout-step}}$ bimodality analysis.

## Appendix C: Reproducibility

The campaign is fully scripted via `reverse_buildup_campaign.py`. The standard reproduction command:

```bash
python reverse_buildup_campaign.py --run-dir <run-dir>
python reverse_buildup_campaign.py --run-dir <run-dir> --phases C --expand-c
python reverse_buildup_campaign.py --run-dir <run-dir> --phases D --expand-d
python reverse_buildup_plots.py --run-dir <run-dir>
```

The first command runs Phases A, B, C (seed 0), E with default settings; subsequent commands expand to all seeds for C and D. The plots command regenerates the figures from the cached outputs.

The test suite (`pytest test_reverse_buildup.py`) verifies the parameterized discriminator's bit-identical reproduction of the legacy forward output, the shuffle null's frequency-preservation, the subspace decomposition's correctness on synthetic data with injected heavy tails, and the standalone arithmetic of the null-correction and F4-signal computations. The integration test `test_forward_parameterized_matches_existing` requires the existing forward output on disk and is gated on the `REVERSE_BUILDUP_RUN_DIR` environment variable; the other 24 tests run without project data.

Random seeds are fixed throughout: D5 sampling uses `np.random.default_rng(20260521)`; the within-chunk shuffle uses a per-checkpoint stream seed derived deterministically from `20260522 ^ (seed * 2654435761) ^ (step * 1597334677)`. The reverse-view results are therefore exactly reproducible from the same on-disk artifacts.
