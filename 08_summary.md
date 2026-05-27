## 8. Summary

This section summarizes the paper's findings, restates what we do not
claim, and indicates the directions of future work that this paper
sets up.

### 8.1 Headline findings

We summarize the five contributions of this paper announced in §1.6
and worked through in §3-§7, restated here in compact form with
quantitative specifics.

**Finding 1: The three-view decomposition is a strict extension of
the basis-invariant framework.** We have defined a partition of the
residual-stream ensemble into three views (all-to-all, forward
input-conditioned, reverse output-conditioned with both actual and
predicted variants) related by the law of total variance, and shown
that the resulting conditional family carries strictly more
information than the marginal alone. The conditional family is
numerically tractable at 150M scale (approximately 250 seconds per
checkpoint per seed for the multi-view inference stage). The per-view
basis-invariant statistics — within-condition variance, between-condition
variance, their ratio, per-view variance-scaling exponents, per-view
effective rank profiles, per-view kurtosis profiles — are
reproducible across seeds at 2-3% relative spread, comparable to the
marginal-view reproducibility of 1-4% reported in §3.5. The
decomposition stays inside the framework's basis-invariant epistemic
zone — no specific directions in $\mathbb{R}^H$ are identified — but
exposes structural and dynamical features the marginal framework
misses.

**Finding 2: The forward crossover layer and the reverse mid-network
peak are reproducible structural features at convergence.** The forward
within/between ratio crosses 1.0 at $t_{\text{cross, fwd}} \approx 1.86$
(cross-seed mean, range $[1.75, 1.94]$ across the four seeds), within
the first two transformer blocks. Context-driven differentiation
overwhelms input identity within the first 15% of network depth. The
reverse-actual within/between ratio peaks at $t = 3$ with peak height
$18.75$ (cross-seed std $0.20$, relative spread $1.1\%$), with the
ratio remaining above 1.0 at every layer — the model never has its
between-output variance overtake its within-output variance, even at
the post-final-norm state where the ratio is $6.31$. The
reverse-predicted view has a flatter plateau-shaped peak of $11.95$
across layers $t = 2-5$, with the same qualitative shape but smaller
magnitude than reverse-actual. The per-view variance-scaling exponents
are ordered $\lambda_{\text{fwd}} = 0.535 > \lambda_{\text{a}} =
0.362 > \lambda_{\text{rev}} = 0.295$ at convergence, reproducible
across seeds at 0.6-0.9% relative spread.

**Finding 3: The reverse-view $\lambda$-dip co-locates with the marginal
$\log\alpha$ hump in training-step coordinates.** The all-to-all
$\log\alpha$ statistic exhibits a mid-training hump with peak at step
$5{,}607$ and peak height $-2.24$ (cross-seed mean). The reverse-view
$\lambda_{\text{rev}}$ exhibits a mid-training dip with minimum at
step $5{,}014$ and dip depth from peak $0.22$ to trough $0.184$
(cross-seed mean). The dip-minimum and hump-peak locations are at
adjacent log-spaced checkpoints in all four seeds. The layer-resolved
heatmap of within/between ratios confirms the co-location is
two-dimensional: the mid-network reverse-view bulge intensifies most
rapidly during steps $2{,}000-11{,}000$ at layers $t = 4-9$, the same
window in which the marginal $\log\alpha$ hump occurs. The
co-location window (steps $2{,}049-10{,}969$, bracketing both
training-dynamic events) contains the bulk of the basis-invariant
change between random initialization and convergence.

**Finding 4: Training reshapes within/between ratios from a uniformly
high initial state through differential reduction.** Random initialization
(seed 9999, no training, same architecture) produces a uniformly high
within/between ratio profile: forward ratio $2.0-8.2$ across layers,
reverse-actual ratio $16.9-72.4$ rising monotonically through depth.
Training reduces the within/between ratio at every layer, but reduces
it most strongly at the I/O boundaries (reverse-actual $t = 13$
reduced by $11.5\times$, from $72.34$ to $6.31$) and least strongly
at the mid-network (reverse-actual $t = 3$ reduced by $2.88\times$,
from $53.96$ to $18.75$). The trained-model mid-network bulge is
therefore a *residue of differential reduction*, not a peak of
learned structure. The forward crossover at $t \approx 1.86$ is
*created by training* (the forward ratio is above 1 at every layer
at random init). The reverse view's persistent within-dominance (no
reverse crossover at any layer) is *inherited from initialization*.

**Finding 5: Cross-seed Procrustes alignment reveals a three-tier
picture of basis-invariance, partial alignment at I/O boundaries, and
non-alignment at the network interior.** The three tiers: (Tier 1)
basis-invariant statistics agree across seeds at 2-3% relative spread;
(Tier 2) residual-stream activations at the I/O boundaries align
across seeds up to rotation at approximately 50% of the
scramble-null ceiling (Procrustes residual $\rho_t = 0.512$ at $t = 0$,
$0.417$ at $t = 1$, $0.636$ at $t = 13$); (Tier 3) residual-stream
activations at the mid-network layers do not align across seeds even
up to rotation (residual $0.692$ at $t = 8$, $0.700$ at $t = 9$, at
about 60-70% of the scramble-null ceiling). The three-tier picture
refines the binary "alignment fails" finding of our prior Phase 1 work
into a more precise characterization: shared vocabulary and tied
unembedding produce partial alignment at the I/O boundaries; the
network interior, where no analogous external constraint applies, is
seed-specific in $\mathbb{R}^H$ even up to rotation.

**Finding 6: Per-token covariance subspaces are nearly orthogonal
across tokens, refuting the strictest linear-Gaussian
token-independence prediction.** Principal angles between per-token
covariance subspaces (top-20 components) are at median $73.0°$ across
token pairs at convergence, vs the self-consistency null of $18.0°$
that measures sample-noise alignment for identical underlying
covariance. The gap of $55°$ is substantial, reproducible across
seeds to within $0.5°$ (cross-seed values: seed 0 73.0°, seed 1
73.5°, seed 2 73.2°, seed 3 73.2°), robust to the choice of subspace
dimensionality (pair median $79.3°$ at $k = 5$ down to $67.0°$ at $k =
50$, gap to self-null always large), and present at the earliest
training checkpoint (pair median $59.1°$ at step 100, rising to
$73.0°$ at step 24,000). The strictest linear-Gaussian formulation —
token-independent noise covariance — does not describe the trained
model. A modified linear-Gaussian framework with token-dependent
covariance remains tractable and consistent with the marginal
basis-invariant statistics we report; the strict formulation's
prediction of token-independent covariance is empirically refuted.

### 8.2 What we do not claim

We restate the principal disclaimers from §1.7 for completeness, in
the light of the findings worked through in §3-§7.

We do not claim that the residual-stream geometry recovered here
extends to substantially larger scales than 150M parameters. The
methodology is scale-independent; only the specific numerical values
($t_{\text{cross, fwd}} = 1.86$, reverse peak $18.75$ at $t = 3$,
pair median $73°$) are tied to our architecture.

We do not claim semantic meaning for the residual-stream subspaces.
The framework characterizes the variance structure; the per-token
covariance non-orthogonality finding does not say different tokens'
bundles are doing different things in any mechanistic sense.

We do not claim that the cross-seed three-tier picture is specific to
this architecture or recipe. We have measured it for one configuration
and report what we found.

We do not claim that the multi-view extension is the right extension.
Other conditional partitions are possible and may reveal different
structural features; the forward/reverse partition by input/successor
token is the simplest non-trivial choice we considered.

We do not claim that the per-token covariance non-orthogonality
finding represents a deviation from the linear-Gaussian baseline in
any interpretively significant sense beyond what we have stated. The
modified linear-Gaussian framework remains tractable and the deviation
we measure is well-characterized empirically.

We do not claim that the random-init baseline is a clean theoretical
reference point. We use it as an empirical reference against which to
calibrate training-induced changes, not as an instantiation of any
specific theoretical formulation.

We do not claim that the multi-view extension's findings replicate
under different training corpora, optimizers, normalization choices,
or embedding-tying schemes. We have measured one specific
configuration and report what we found.

### 8.3 What this paper sets up for future work

The paper establishes a methodological framework — the multi-view
decomposition of the residual-stream ensemble — and a baseline set of
empirical findings against which extensions can be measured. The
follow-up work this enables falls into several categories.

**Cross-architecture ablation studies.** Our prior Phase 1 work
identified Phase 2 as a planned cross-architecture study varying
single design axes (depth, width, FFN intermediate ratio,
normalization, attention configuration) against the GELU Phase 1
baseline. The multi-view framework provides additional cross-variant
statistics for Phase 2 to compare: the forward crossover layer, the
reverse mid-network peak height and location, the co-location
window's timing, the three-tier cross-seed pattern, the per-token
covariance pair median angle. Each of these is a candidate
cross-architecture invariant; each has a within-variant noise floor
measurable at the Phase 1 GELU baseline; each can be compared across
Phase 2 variants with the noise floor as the threshold for
attribution. The within-variant noise floors for the multi-view
statistics are reported throughout this paper alongside the
all-to-all dispersion bounds of §3.5.

**Cross-scale extension.** The methodology scales to larger models;
the bottleneck is training multiple seeds at larger scales. A natural
follow-up is to apply the multi-view extension to models in the
350M-7B parameter range (the range Sarfati et al. originally analyzed)
with multiple seeds per configuration, to test whether the structural
features we report at 150M (the $t \approx 1.86$ forward crossover,
the $t = 3$ reverse peak) scale with depth, with absolute layer
index, or with neither.

**Other conditioning choices.** §7.5 lists several conditioning
choices beyond input/successor token that respect the framework's
basis-invariance and would extend the multi-view family. Conditioning
on the previous token, on prefix length, on syntactic category, on
bigram frequency would each test whether the multi-view structural
features we report depend on the specific input/successor choice or
are more general properties of the residual-stream ensemble.

**Random-init follow-up measurements.** Our random-init analysis (§6.2)
measured within/between ratios but did not measure the per-token
covariance subspace structure at random init. The §6.5 finding that
per-token covariance non-orthogonality is already at $59°$ at step 100
suggests one of three explanations (intrinsic to architecture at
init; rapidly constructed in first ~250 gradient steps; both),
distinguishable by a random-init per-token covariance measurement.
This is a small additional experiment that would resolve a specific
interpretive ambiguity.

**Connection to mechanistic interpretability.** The multi-view
structural findings (forward crossover at $t \approx 1.86$, reverse
mid-network peak at $t = 3$, per-token covariance non-orthogonality)
are layer-resolved phenomena with potential mechanistic correlates.
A follow-up bridging the basis-invariant framework with the
mechanistic interpretability literature would test whether
specific computational circuits identified at the
mechanistic-interpretability level co-locate with the layer-resolved
structural features identified at the basis-invariant level.

**Token-level Procrustes alignment.** §6.4 reports Procrustes alignment
of marginal activations across seeds. A finer analysis would compute
per-token Procrustes — for each input token $v$, align $\mathcal{E}_v^{(A)}$
to $\mathcal{E}_v^{(B)}$ via the orthogonal Procrustes — and ask
whether the per-token alignment is better than the marginal alignment.
If yes, it would suggest the per-token covariance subspaces are
*more* reproducible across seeds than the marginal residual stream;
if no, it would suggest the per-token-level seed-specificity matches
the marginal-level seed-specificity. Either outcome would clarify
what kind of structure the cross-seed Tier 3 non-alignment is failing
to capture.

**Test of structural features across pretrained models.** A relatively
inexpensive follow-up is to apply the multi-view analyzer to existing
pretrained models (GPT-2 medium, Llama-2-7B, Mistral-7B, Pythia-12B)
that Sarfati et al. originally analyzed. The forward crossover layer,
the reverse mid-network peak, and the per-token covariance pair
median angle can all be computed on a single pretrained checkpoint
without retraining. This would test whether our structural findings
generalize to the models the original framework was developed on,
modulo the lack of within-variant noise floors at frontier scales.

### 8.4 Closing remarks

The basis-invariant framework of Sarfati et al. characterizes the
residual stream of trained transformers through a small set of
intrinsic geometric quantities computed on the marginal distribution
of pilot activations. This paper extends the framework to a three-view
decomposition that conditions on input and successor tokens, exposes
structural and dynamical features the marginal framework misses, and
refines the framework's cross-seed reproducibility claim into a
three-tier picture of intrinsic geometry vs basis-dependent absolute
pose.

The findings are basis-invariant, cross-seed reproducible, robust to
methodological choices we have varied (subspace dimensionality,
token-set composition, conditioning variant), and supported by
explicit null baselines that calibrate what "reproducible" means in
each case. The methodology is computationally tractable at 150M
scale and ready to be applied at larger scales and to other
architectures.

The interpretive significance of the findings is that the framework's
universality claim, properly understood, is a claim about *shape
universality* of the marginal residual-stream variance structure —
not about residual-stream *structural* universality at the level of
the underlying subspaces. The basis-invariant statistics extract
genuinely universal content; the underlying $R(t)$ matrices and the
per-token covariance subspaces are seed-specific and token-specific
in ways that the framework correctly factors out. The framework's
correct interpretation is therefore as a population-level
characterization of marginal statistics, not as a statement about
the absolute pose of the residual stream in hidden space.

What remains to be done is the cross-architecture extension: testing
whether the multi-view structural and dynamical findings replicate
across architectures at the within-variant noise floors this paper
establishes. The Phase 2 study referenced throughout the paper is
set up to do this, and its results will determine whether the
multi-view extension is in fact extracting architecture-universal
structure or whether the specific values we report ($t \approx 1.86$
forward crossover, $18.75$ reverse peak, $73°$ pair median angle)
are themselves architecture-specific. Either outcome would be a
meaningful refinement of what the basis-invariant framework's
universality claim actually claims.

---

*End of paper.*
