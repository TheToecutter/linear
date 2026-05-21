# Multi-View Decomposition of the Residual-Stream Bundle

## A Project Proposal

This proposal describes a project that takes the basis-invariant analysis framework of Sarfati et al. (ICLR 2025) and a previous Phase 1 study, and extends it from a single-view (all-to-all ensemble) characterization to a *three-view decomposition* of the residual-stream bundle. The three views — all-to-all, input-conditioned, and output-conditioned — partition the same ensemble variance and, by the law of total variance, must sum to a single fixed total at every layer. The decomposition is the structural object of study.

The project is inference-only: it uses checkpoints from an existing trained model and does not require any additional training. Its goal is to test whether the basis-invariant framework, extended to conditional ensembles, gives a sharper functional account of what the residual stream is *doing* through training and across depth than the marginal framework alone supports.

---

## 1. Background and motivation

### 1.1 Prerequisites assumed

This proposal assumes the following are in hand:

- A single 150M-parameter Llama-style decoder-only transformer, trained from scratch on FineWeb-Edu (sample-10BT) for 24,000 steps under the Phase 1 recipe (AdamW, cosine LR schedule, batch 64 × 1024 tokens, peak LR 3e-4). Architecture follows the Llama defaults (RMSNorm, RoPE, tied embeddings, pre-norm blocks) with GELU as the FFN nonlinearity, matching the (separately rerun) Phase 1 reference configuration.
- Four independent seeds of this model trained under identical recipe with different random initialization and data-shuffle order.
- 50 log-spaced training checkpoints per seed, with full model weights on disk (not only analyzer outputs).
- A 500-chunk held-out evaluation set of 1024-token sequences from FineWeb-Edu.
- The Phase 1 analyzer pipeline producing 14-state activations (post-embedding, 12 block outputs, post-final-norm) and the standard basis-invariant statistics: $\log\alpha$, $\lambda$, effective rank profile, kurtosis profile, isotropy profile, successive-layer angle profile, post-final-norm anomaly.

This project and Phase 1 share the same model, recipe, and seeds. The multi-view analysis is a strict extension of the Phase 1 dashboard applied to the same checkpoints, not a re-run on a different configuration.

### 1.2 The framework's macro/micro decomposition

The basis-invariant framework characterizes the residual stream by the *shape* of its ensemble variance — basis-invariant scalars and profiles that don't depend on which absolute direction in $\mathbb{R}^{H}$ any particular feature is encoded along. This makes the framework's findings reproducible across seeds: the absolute pose of the bundle varies, but its intrinsic geometry does not.

The framework's limitation is that it describes the bundle's shape but not what the bundle is doing. The all-to-all ensemble pools variance from every (token, context, position) triple in the data; whatever structure exists in how the model differentiates inputs or commits to outputs is averaged into a single number per statistic per layer. The marginal view is *blind to its own structure*.

The conditional view restores the structure without leaving the basis-invariant regime. By partitioning the pilots by an externally-defined label (input token, successor token, or (input, output) pair), recomputing the same basis-invariant statistics on each subset, and using the law of total variance as bookkeeping, we obtain a finite-dimensional family of conditional macro statistics whose marginal is the original all-to-all statistic. The conditional family carries strictly more information than the marginal, and the partition structure (it sums to the marginal) constrains how the family can vary across architectures or training steps.

### 1.3 The three views

**All-to-all view (the marginal).** All 9,500 pilots from the held-out set, treated exchangeably. Variance is pooled over input token, successor token, context, and position. This is the Phase 1 view.

**Forward view (input-conditioned).** Pilots are filtered by which token sits at the pilot position. For each token $v$ in some chosen set $V$, the forward-view ensemble $E_v$ is the subset of pilots with pilot token $v$. By construction, $E_v$ has zero variance at $t = 0$ (all trajectories start at the embedding of $v$) and gains variance through subsequent layers as attention folds in context. The forward view measures the rate and dimensionality of context-driven differentiation for a fixed starting point.

**Reverse view (output-conditioned).** Pilots are filtered by the successor token — the token at position $p + 1$ in the chunk that the state at position $p$ is predicting. For each successor $w$ in some chosen set $W$, the reverse-view ensemble $F_w$ is the subset of pilots whose next token is $w$. By construction, $F_w$ has high variance at $t = 0$ (many different tokens can precede $w$) and contracts through subsequent layers as the model converges on a shared prediction. The reverse view measures the rate and dimensionality of prediction-driven convergence.

### 1.4 The law of total variance as bookkeeping

For any partition of pilots by label $z$ and per-coordinate variance $V$ at layer $t$:

$$V_{\text{all-to-all}}(t) = \mathbb{E}_z\!\left[V_{\text{within-}z}(t)\right] + \mathrm{Var}_z\!\left[\mu_t(z)\right]$$

where the first term is the average within-condition variance and the second is the variance of the per-condition means.

This identity holds at every layer separately. It says the all-to-all variance budget at each layer partitions exactly into a *within* and *between* component, given any choice of conditioning label. The forward view computes this partition for $z =$ input token; the reverse view for $z =$ successor token. Both partitions sum to the same all-to-all total at every layer, so they are *constrained decompositions* of the same fixed object, not independent measurements.

The functional content of the project is in the *layer-wise trajectory* of these partitions: at what depth does within-input variance overtake between-input variance (the moment context overwhelms input identity)? At what depth does between-output variance overtake within-output variance (the moment prediction commitment forms)? Are these depths the same? Do they stabilize across training? Do they coincide with the training-dynamic anomalies already known from Phase 1?

### 1.5 Conceptual contribution

This project's contribution is methodological: a way to bridge the macro/micro divide in the basis-invariant framework without leaving the framework's epistemic zone. The decomposition stays inside basis-invariance (no specific directions in $\mathbb{R}^{H}$ are identified as "the cat direction" or similar) but it characterizes how the basis-invariant macro statistics *partition* across micro-relevant conditions. The partition structure is itself a macro statistic — a basis-invariant family — and is a candidate for cross-architecture universality on the same terms as the original marginal statistics.

The bridge it offers is mid-altitude. It does not reduce the macro view to mechanisms (it does not say which direction encodes which feature). It does say that the all-to-all variance budget at each layer is allocatable between input-identity preservation and output-identity assembly, and the allocation curve through depth is the network's computational signature in a basis-invariant sense.

---

## 2. Hypotheses

The project's hypotheses sit at three levels: structural (does the decomposition exist and behave coherently?), functional (does it tell a mechanistic story?), and dynamical (does it evolve through training in interpretable ways?). Each hypothesis has a quantitative success criterion.

### 2.1 Structural hypotheses

**S1: The conditional ensembles are statistically well-defined and reproducible across seeds.** For each chosen condition $z$ (input or output token), the basis-invariant statistics of the conditional ensemble — $\log\alpha_z$, $\lambda_z$, effective rank profile, kurtosis profile — have within-seed std comparable in fractional magnitude to the Phase 1 all-to-all measurements (i.e., the conditional view does not introduce noise floors that swamp the signal).

*Test:* Compute conditional statistics for the top-20 most frequent input tokens and the top-20 most frequent successors, on all four seeds. Within-seed std should be ≤ 2× the Phase 1 within-seed std for the corresponding all-to-all statistic.

**S2: The law of total variance is satisfied numerically.** The forward and reverse decompositions sum to the all-to-all variance at every layer, within floating-point precision and sample-size correction.

*Test:* Compute $\mathbb{E}_z[V_{\text{within-}z}(t)] + \mathrm{Var}_z[\mu_t(z)]$ for both partitions at every layer and compare to $V_{\text{all-to-all}}(t)$. Discrepancy should be attributable entirely to (a) the finite token-set $V$ or $W$ used rather than the full vocabulary, and (b) Zipfian-weighting choices. Document the discrepancy explicitly.

### 2.2 Functional hypotheses

**F1: Forward-view variance grows with depth.** Within-input variance at layer $t$, averaged over the chosen token set, increases monotonically (or near-monotonically) with $t$ from zero at $t = 0$. The growth rate is a candidate basis-invariant signature of "attention's contribution to context differentiation."

*Test:* Fit $\log V_{\text{within}}(t) = \log\alpha_{\text{fwd}} + \lambda_{\text{fwd}} \log(t)$ for $t \geq 1$ (excluding $t=0$ where variance is zero by construction). Report $\log\alpha_{\text{fwd}}$, $\lambda_{\text{fwd}}$, and the deviation from log-linear behavior.

**F2: Reverse-view variance contracts with depth at late layers.** Within-output variance at layer $t$, averaged over the chosen successor set, decreases between some inner-layer maximum and the final layer. The contraction is concentrated in the late layers, not spread uniformly across depth.

*Test:* Identify the layer $t^*$ at which $V_{\text{within-output}}(t)$ peaks, and compute the contraction fraction $V_{\text{within-output}}(T) / V_{\text{within-output}}(t^*)$. Compare across seeds. If $t^*$ is near $T$ and the contraction fraction is meaningful (say $< 0.7$), the late-contraction picture is supported.

**F3: The within/between crossover layer exists and is reproducible.** For each view, define the crossover layer as the smallest $t$ at which the within-condition variance exceeds the between-condition variance (forward) or vice versa (reverse). The two crossover layers — forward and reverse — are well-defined at the final checkpoint, agree to within ±1 layer across seeds, and may coincide with each other (suggesting a single "decision layer").

*Test:* Compute the within/between ratio at each layer for each view. Identify the crossover. Report cross-seed agreement. Test whether forward and reverse crossovers coincide.

**F4: The basis-invariant rank profiles differ qualitatively across views.** Forward-view effective rank grows from 1 at $t=0$; reverse-view effective rank contracts toward a small number at $t=T$; all-to-all effective rank is roughly constant or slowly growing. These three profiles are qualitatively different (not just rescalings of each other).

*Test:* Plot all three effective rank profiles on the same axes, averaged across seeds. The qualitative shapes should be distinguishable.

### 2.3 Dynamical hypotheses

**D1: The crossover layers emerge during training.** At initialization or early in training, the forward and reverse crossovers are either absent (curves don't cross), at degenerate locations (layer 0 or layer $T$), or unstable. By some characteristic training step they stabilize at their convergence locations.

*Test:* Compute crossover layers at all 50 checkpoints. Track emergence trajectory. Report the training step at which crossovers first stabilize within ±1 layer of their final position.

**D2: The crossover emergence co-locates with Phase 1 training-dynamic anomalies.** The Phase 1 study found three training-dynamic features at characteristic step ranges: post-final-norm anomaly emerges through step ~2000, $\log\alpha$ hump peaks around step 5000, mid-training $\Sigma$-distance bump around steps 5000–10000. At least one of the crossover stabilization events co-locates (within a log-spaced checkpoint interval) with one of these.

*Test:* Cross-reference crossover stabilization steps with Phase 1's training-dynamic events.

**D3: The within/between partition trajectory through training is itself a reproducible signature.** Beyond crossover layer alone, the *shape* of the within/between ratio curve evolves through training in a way that is reproducible across seeds.

*Test:* Plot the within/between ratio profile (one curve per checkpoint, layer on x-axis) for each seed. The family of curves should be similar across seeds at every checkpoint.

---

## 3. Experimental design

### 3.1 Activation collection (one-time)

Modify the Phase 1 `collect_activations` function in two ways. First, alongside each (chunk, pilot_position) hidden state, save the input token id `input_ids[chunk, pilot_position]` and the successor token id `input_ids[chunk, pilot_position + 1]`. Second, optionally save the predicted next token (argmax of the logits at the pilot position under the lm_head), to support the actual-vs-predicted comparison in the reverse view.

The output of this step is a single augmented activation file per seed per checkpoint:

```
activations_step_NNNNNNNN.npz contains:
    states:      (14, 9500, 896) float32
    input_ids:   (9500,) int32   -- pilot tokens
    next_ids:    (9500,) int32   -- actual successors
    pred_ids:    (9500,) int32   -- argmax predictions
```

The compute cost is roughly equal to the Phase 1 analyzer's activation-collection step (one inference pass per checkpoint), times 50 checkpoints times 4 seeds = 200 inference passes. At ~minutes per pass this is a few hours of GPU time, well within scope.

### 3.2 Token-set selection

The full vocabulary is 32,768 tokens; the held-out set's 9,500 pilots visit only a small subset of these with usable frequency. Selection follows these rules:

- **Forward view token set $V$.** The 20 most frequent pilot tokens in the held-out set, by count of pilot occurrences. Each must have at least 50 instances. If fewer than 20 tokens meet the threshold, expand the held-out set (see §3.6) until they do.

- **Reverse view token set $W$.** The 20 most frequent successor tokens in the held-out set, with the same 50-instance threshold.

- **(Input, output) pair set $P$.** The 20 most frequent (pilot, successor) pairs, with at least 30 instances each. This is used for the optional ANOVA-style decomposition in §3.5.

The choice of 20 is a balance between sample-size adequacy and coverage breadth. Pre-registration: report sensitivity of headline results to this choice by recomputing with $|V| = |W| = 10$ and $|V| = |W| = 50$ (the latter may require held-out set expansion).

### 3.3 Conditional analyzer

The Phase 1 analyzer's `recover_linear_flow` function takes a (num_layers, N, H) activation tensor and produces the basis-invariant statistics. Wrap it in a conditional shell that takes a token filter and runs the same pipeline on the filtered subset:

```python
def conditional_flow(activations, filter_ids, target_id):
    """Filter activations to those matching target_id, then run the
    standard analyzer."""
    mask = (filter_ids == target_id)
    sub = activations[:, mask, :]
    return recover_linear_flow(sub, center=True)
```

For each view (forward, reverse, optional pair-conditional), for each chosen token in the set, run the conditional analyzer at every checkpoint of every seed. The outputs are stored as a parallel set of `flow_step_NNNNNNNN_view-VIEW_token-ID.npz` files alongside the existing all-to-all `flow_step_NNNNNNNN.npz`.

Storage cost: 20 tokens × 2 views × 50 checkpoints × 4 seeds = 8,000 small npz files per seed. Each is roughly the same size as the all-to-all flow file (a few MB). Total ~ 100 GB, manageable.

### 3.4 The within/between decomposition

For each layer $t$ and each view (forward or reverse), compute:

$$V_{\text{within}}(t) = \frac{1}{|V|} \sum_{v \in V} \frac{1}{H} \sum_{i=1}^{H} \mathrm{Var}_{k : \text{pilot}_k = v}\!\left[x^{(i)}_k(t)\right]$$

$$V_{\text{between}}(t) = \frac{1}{H} \sum_{i=1}^{H} \mathrm{Var}_{v \in V}\!\left[\mu^{(i)}_t(v)\right]$$

where $\mu^{(i)}_t(v)$ is the mean of coordinate $i$ at layer $t$ across all pilots with pilot token $v$.

By the law of total variance, on the subset of pilots whose token is in $V$, $V_{\text{within}}(t) + V_{\text{between}}(t)$ equals the per-coordinate variance of the whole subset. This should be checked numerically and compared to the all-to-all variance to quantify the contribution of tokens outside $V$ (the "long tail" of the vocabulary).

The within/between ratio $r_t = V_{\text{within}}(t) / V_{\text{between}}(t)$ is the per-layer summary statistic. The crossover layer is the smallest $t$ for which $r_t > 1$ (forward) or $r_t < 1$ (reverse).

### 3.5 (Optional) Full ANOVA decomposition

For the pair-conditional view, the variance at each layer decomposes further:

$$V_{\text{total}} = V_{\text{within-pair}} + V_{\text{input-effect}} + V_{\text{output-effect}} + V_{\text{interaction}}$$

where the four terms attribute the variance to (a) residual context within a fixed (input, output) cell, (b) input-token marginal effect, (c) successor-token marginal effect, and (d) the input-output interaction (specific bigrams behaving distinctively beyond their marginals).

This is the full bookkeeping the project's framework supports. It is listed as optional because the sample-size constraints are tightest here — common bigrams in 9,500 pilots may have only dozens of instances each, marginal for kurtosis. If the held-out set is expanded to ~50,000 pilots, the ANOVA decomposition becomes well-supported.

### 3.6 Held-out set expansion (conditional)

The current 500-chunk held-out set yields 9,500 pilots. The basic forward and reverse views can be done at this scale. The full ANOVA decomposition and sensitivity analyses with $|V|, |W| = 50$ benefit from larger samples. Expanding to 5,000 held-out chunks yields ~95,000 pilots and is a roughly 10× increase in inference cost but no training cost. Decision point: do the basic views first, look at the within-seed noise floors, then expand the held-out set if and only if the noise is dominating the signal.

### 3.7 Compute budget

| Stage | Per-checkpoint cost | Across 50 ckpts × 4 seeds | Wall clock (RTX 5090) |
|---|---|---|---|
| Augmented activation collection | ~3 min | 200 passes | ~10 hours |
| Conditional analyzer (40 conditions × 2 views) | ~30 s | 8,000 runs | ~70 hours |
| All-to-all reanalysis (already done in Phase 1) | — | — | 0 |
| Optional: held-out expansion to 95k pilots | ~30 min | 200 passes | ~100 hours |

Without the optional expansion, the full project's compute is roughly 80 hours of single-GPU time, all inference. With the expansion, roughly 180 hours. This is comparable to a single Phase 1 training run (one seed = ~12 hours of training) but generates an entire project's worth of analytic output rather than one trained model.

---

## 4. Deliverables and how they will be interpreted

This section enumerates the specific results the project will produce and the interpretation framework for each.

### 4.1 Per-view basis-invariant statistic dashboard

For each view (all-to-all, forward, reverse) at the final checkpoint, the standard Phase 1 dashboard: $\log\alpha$, $\lambda$, effective rank profile, kurtosis profile, isotropy profile, post-final-norm anomaly. For forward and reverse views these are computed as averages over the chosen token set, weighted by token frequency, with within-seed std reported.

*Interpretation.* Side-by-side comparison of the three dashboards tells the basic story of how the bundle's macro statistics differ when computed on the marginal vs on input-conditioned subsets vs on output-conditioned subsets. Differences between the dashboards are the structural signature of conditional vs marginal geometry.

### 4.2 The within/between decomposition curve

A single plot per view, per checkpoint, per seed: layer on x-axis, within-condition variance and between-condition variance on y-axis (both as fractions of the all-to-all variance at that layer). The two curves sum to ≤ 1 (with the deficit attributable to tokens outside the chosen set). The crossover point is marked.

*Interpretation.* The decomposition curve shows directly how the residual stream's variance budget at each layer partitions into "variance due to which token started here" vs "variance due to what happens after that." For the forward view, the curves invert at the crossover layer: below it, between-input dominates (the bundle's spread is mostly "which token am I"); above it, within-input dominates (the spread is mostly "what context am I in"). The crossover layer is a direct functional readout of where attention stops being differentiable from token identity.

For the reverse view, the curves invert in the other direction: at early layers, within-output dominates (predictions are not yet committed, so trajectories headed to the same successor are spread out); at late layers, between-output dominates (the model has committed, and trajectories headed to different successors are separated). The reverse crossover is the prediction-commitment layer.

### 4.3 Crossover layer trajectory through training

A plot per view, per seed: training step on x-axis (log scale), crossover layer on y-axis. Four curves per view (one per seed) overlaid. Stabilization step marked.

*Interpretation.* This shows when the model develops its input→output transition geometry. Plausible findings: (a) the crossover migrates from one end of the network to a stable interior location during training, indicating the model learns where to commit; (b) the crossover is in a degenerate location early and snaps into position at a characteristic step; (c) the crossover layer is stable across training but the magnitude of the within/between separation grows. Each pattern is a distinct mechanistic story about how the network organizes.

### 4.4 Co-location with Phase 1 training-dynamic anomalies

A summary table: each Phase 1 training-dynamic event (post-final-norm anomaly emergence, $\log\alpha$ hump peak, $\Sigma$-distance bump, late kurtosis rise) alongside the closest crossover stabilization or related event from the multi-view analysis. Co-locations within a log-spaced checkpoint interval are flagged.

*Interpretation.* If the unconditional Phase 1 anomalies co-locate with conditional-view structural events, the anomalies have a mechanistic explanation in terms of the conditional dynamics. For example, if the post-final-norm anomaly emergence (steps ~400–2000) co-locates with reverse-view contraction onset, the anomaly may be the surface signature of the model learning to commit to predictions. This is a macro→micro bridge in the dynamical direction.

### 4.5 Cross-view basis alignment

For each pair of views (all-to-all vs forward, all-to-all vs reverse, forward vs reverse), compute principal angles between the top-$k$ SVD bases at each layer. Report the mean top-10 principal angle as a profile across layers, averaged across the chosen token set for restricted views.

*Interpretation.* Small angles mean the restricted bundle's principal directions align with the global bundle's principal directions — conditional structure lives along the same axes as the marginal structure. Large angles mean the restricted bundle uses "private" directions in the residual stream that the global SVD doesn't see. Layer-dependence of the angles shows where in the network the conditional structure becomes most basis-divergent from the marginal.

### 4.6 (If pair-conditional view is run) ANOVA decomposition

A stacked-area plot per checkpoint: layer on x-axis, four bands (within-pair, input-effect, output-effect, interaction) summing to all-to-all variance.

*Interpretation.* The relative magnitudes of the four bands at each layer say how the network is allocating its representational budget. A network that does most of its work in the input-marginal band is doing token-by-token feature extraction; one that does most of its work in the output-marginal band is doing prediction assembly; one with a large interaction band is encoding bigram-specific structure in a way the marginals can't capture; one with a large within-pair band is preserving long-range context information not captured by the immediate (input, output) cell.

### 4.7 Actual-vs-predicted reverse view comparison

The reverse view can be computed conditioning on the *actual* next token (data property) or the model's *predicted* next token (model property). Both are computed; their results are compared.

*Interpretation.* When the two diverge, the model is wrong on those pilots. The predicted-conditional view tells you what the model's internal commitments look like; the actual-conditional view tells you the data-side answer. The crossover layer and contraction rates may differ between the two — for example, the model may commit cleanly to its predictions (sharp predicted-conditional contraction) but those predictions are not always right (looser actual-conditional contraction). The ratio between predicted-contraction and actual-contraction at the final layer is one summary of "prediction quality as a geometric property" — distinct from cross-entropy loss because it characterizes how much representational structure the model has built, not how often it's right.

### 4.8 Sensitivity and robustness checks

- Token-set size sensitivity: rerun headline results with $|V|, |W| = 10$ and $|V|, |W| = 50$. Report variation.
- Pilot-position sensitivity: rerun headline results restricted to pilots at positions $> 500$ (deeper-context pilots only) and pilots at positions $\leq 500$. The two should give qualitatively similar results if the analysis is not artifacted by context-length variation.
- Held-out set sensitivity: if the optional expansion is done, compare results at 9,500 pilots vs 95,000 pilots. Discrepancies indicate sample-size limitation rather than real signal.

---

## 5. Risks and mitigations

**R1: The conditional ensembles are too small for stable basis-invariant statistics.** At 9,500 pilots, even the most common token gives only a few hundred instances. Kurtosis and effective rank are sensitive to sample size in this regime.

*Mitigation:* Pre-register the within-seed noise floor measurement (hypothesis S1) as the gating step. Proceed to dynamical and functional analyses only if S1 passes. If S1 fails, expand the held-out set before continuing.

**R2: The crossover layer doesn't exist as a clean concept.** The within/between curves might not cross — within might dominate at all layers, or the curves might cross multiple times.

*Mitigation:* Define a continuous summary as fallback: the integrated signed difference $\int_t (V_{\text{within}}(t) - V_{\text{between}}(t))\, dt$ across depth. Use this in place of the crossover layer where the crossover is undefined. Report both, flag which is being used.

**R3: The decomposition is dominated by a few high-frequency tokens.** " the", " of", " and" account for a large fraction of any English text's tokens. The forward view restricted to these may not be representative.

*Mitigation:* Report results both with frequency-weighted averaging (natural) and with uniform averaging across the chosen token set (unweighted). Discrepancies between the two indicate the result is dominated by frequency rather than structure.

**R4: The actual-vs-predicted reverse view is hard to interpret cleanly when the model's accuracy varies through training.** Early-training models predict mostly badly; late-training models predict better but not perfectly.

*Mitigation:* Report the reverse-view results conditioned on actual successors as the primary view (data-side, stable). The predicted-successor variant is reported as a secondary diagnostic about the model's internal commitments. Do not attempt to interpret the comparison until the loss curve has stabilized — Phase 1 establishes the relevant step for this model.

---

## 6. Project structure and milestones

**Phase A: Setup and basic measurement (estimated 2 weeks).**

- A1: Modify `collect_activations` to capture input/successor/predicted token ids. Verify on one checkpoint of one seed.
- A2: Implement `conditional_flow` wrapper. Verify that running it with a no-op filter reproduces the Phase 1 all-to-all numbers exactly.
- A3: Run augmented activation collection across all 200 checkpoints (50 × 4 seeds).
- A4: Select token sets $V$, $W$ from the augmented data. Document selection.

**Phase B: Forward and reverse views, final checkpoint (estimated 1 week).**

- B1: Compute conditional flows for all chosen tokens at the final checkpoint of seed 0. Verify hypothesis S1 (cross-seed reproducibility by running on seeds 1, 2, 3).
- B2: Verify hypothesis S2 (numerical satisfaction of law of total variance). Document any discrepancies.
- B3: Produce final-checkpoint dashboard (§4.1) and decomposition curves (§4.2). Identify crossover layers; verify hypothesis F3.

**Phase C: Training-dynamic analysis (estimated 2 weeks).**

- C1: Compute conditional flows at all 50 checkpoints × 4 seeds.
- C2: Track crossover layers through training (§4.3). Test hypothesis D1.
- C3: Cross-reference with Phase 1 training-dynamic anomalies (§4.4). Test hypothesis D2.
- C4: Plot within/between ratio trajectory across training (§4.7). Test hypothesis D3.

**Phase D: Synthesis and (optional) extensions (estimated 1–3 weeks).**

- D1: Cross-view basis alignment analysis (§4.5).
- D2: Sensitivity checks (§4.8).
- D3 (optional): Pair-conditional ANOVA decomposition (§4.6).
- D4 (optional): Held-out set expansion if any analyses are sample-limited.
- D5: Write up.

Total estimated project length: 6–10 weeks for one researcher, single-GPU.

---

## 7. Position relative to related work

**Relative to Sarfati et al. (ICLR 2025).** This project takes the basis-invariant framework as foundational and extends it from a single-view (all-to-all) study to a three-view decomposition. The extension is methodological, not architectural — same statistics, same framework, applied to conditional subsets and reassembled via the law of total variance. The paper's null-testing experiment (gibberish vs language) is the closest precedent: it asks whether the all-to-all macro structure depends on the micro semantic content of inputs, and finds it doesn't. The multi-view decomposition is a more refined version of the same question — not "does macro depend on micro?" but "how does macro partition across micro-labeled subsets?"

**Relative to mechanistic interpretability.** This project does *not* attempt to identify which directions in the residual stream encode which features. The mechanistic-interpretability program does that (via sparse autoencoders, probing classifiers, activation patching), but it requires committing to specific directions that are seed-dependent and architecture-specific. This project deliberately stays in the basis-invariant regime, which means its results are reproducible across seeds and (in subsequent work) comparable across architectures on terms the mechanistic program cannot offer.

The two programs are complementary: mechanistic interpretability tells you what specific features mean in one model; multi-view decomposition tells you how variance budgets are allocated across input/output conditions in a way that should hold across reimplementations of the same architecture and may hold across architectures.

**Relative to the Phase 1 study.** Phase 1 established that the all-to-all basis-invariant framework gives reproducible measurements at the 150M scale and characterized several training-dynamic features. This project takes those results as input and asks what the conditional view adds. The two projects are intentionally separable: Phase 1 stands alone as a reproducibility study; this project stands alone as a methodological extension. The interaction between them is the §4.4 co-location analysis, which uses Phase 1's training-dynamic features as reference markers for the new dynamical events the multi-view analysis surfaces.

---

## 8. What success looks like

The project will be considered successful if it produces a clear, quantitative answer to each of the following questions, at within-seed noise floors that allow the answers to be trusted:

1. Does the multi-view decomposition exist as a coherent object (hypotheses S1, S2)? Expected: yes.

2. Do the forward and reverse views show qualitatively distinct variance profiles (hypothesis F4)? Expected: yes.

3. Are there well-defined crossover layers, reproducible across seeds (hypothesis F3)? Expected: yes for at least one of forward/reverse; possibly both.

4. Do the crossover layers emerge during training in a characterizable way (hypothesis D1)? Open empirical question; success means producing a clean trajectory plot regardless of its shape.

5. Do training-dynamic events in the multi-view analysis co-locate with the Phase 1 all-to-all training-dynamic events for this same model (hypothesis D2)? Open empirical question; either outcome is informative.

6. Are the basis-invariant statistics of restricted views in alignment with those of the marginal view, or do they live in different subspaces (§4.5)? Open empirical question.

The headline finding, if all hypotheses hold, is a basis-invariant mid-altitude bridge between macro and micro: the residual stream's variance budget at each layer partitions into input-identity-preservation and output-identity-assembly components in a reproducible way, and the partition's depth-trajectory is the network's computational signature in a basis-invariant, seed-independent sense.

This signature is then the natural object for a follow-on cross-architecture universality study — but that study is out of scope for this project, which restricts itself to the single Llama-style architecture and its four seeds.
