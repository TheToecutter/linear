## 7. Discussion

This section discusses the interpretive significance of the findings
reported in §3-§6. It is organized around five interpretive questions:
what the multi-view extension contributes to the basis-invariant
framework beyond the original marginal analysis (§7.1), how the
relationship between basis-invariance and seed-dependent feature
bases should be understood given the three-tier cross-seed claim
(§7.2), what the per-token covariance non-orthogonality tells us
about the linear-Gaussian framing of trained-transformer
residual-stream dynamics (§7.3), how the random-init baseline should
be interpreted as a reference point for training-induced reshaping
(§7.4), and what questions are left open by this work that future
work could address (§7.5).

Throughout this section we distinguish between two kinds of claims:
*empirical observations* (what the data shows, which is well-supported
by the measurements of §3-§6) and *interpretive framings* (how those
observations are best understood, which involves analytic choices
that other observers might disagree with). We try to keep these
separated and to flag where we are offering interpretation that goes
beyond what the data uniquely determines.

### 7.1 What the multi-view extension buys the framework

The basis-invariant framework of Sarfati et al. characterizes the
residual-stream ensemble through a small number of intrinsic geometric
quantities — $\lambda$, $\log\alpha$, effective rank, kurtosis,
isotropy — computed on the marginal (all-to-all) distribution of
pilot activations at each layer. The multi-view extension we have
described decomposes the same residual-stream ensemble into three
related views by conditioning on input token (forward view), on
actual successor token (reverse-actual view), or on predicted
successor token (reverse-predicted view). The decomposition is
constrained by the law of total variance — the within-condition and
between-condition variances of each conditional view sum to the
marginal variance at every layer — and the per-view basis-invariant
statistics are computed in the same way as the marginal statistics
but on the conditional ensembles.

What does this multi-view extension actually contribute, beyond what
the marginal framework already provides?

**More quantitative information per layer.** The most direct
contribution is just that the conditional family carries strictly
more information than the marginal alone. The marginal characterizes
each layer with a single $\lambda$, $\log\alpha$, effective rank,
kurtosis, and isotropy value. The conditional family characterizes
each layer with three sets of these quantities (one per view), plus
the per-view within and between components, plus the per-view
within/between ratios, plus the per-view variance-scaling exponents,
plus the per-view effective rank profiles. The conditional family is
constrained by the partition identity, so not all of these quantities
are independent — but the constraint is at the per-layer per-coordinate
variance level, and the derived quantities ($\lambda$ values, ratio
crossovers, peak locations, etc.) are not constrained to be the same
across views.

The structural findings of §5 demonstrate that the conditional family
contains genuinely distinct information from the marginal. The
forward crossover layer at $t \approx 1.86$, the reverse mid-network
peak at $t = 3$ with peak ratio 18.75, the per-view $\lambda$ ordering
$\lambda_{\text{fwd}} > \lambda_{\text{a}} > \lambda_{\text{rev}}$ —
these are all derivable only from the conditional decomposition, not
from the marginal alone. The conditional family is therefore a strict
extension of the framework's information content.

**Structural anomalies that the marginal misses.** The mid-network
within/between ratio peak of the reverse-actual view at $t = 3$
(height 18.75) is a structural feature that the marginal view has no
trace of. The marginal effective rank profile peaks in the mid-network
(at $t = 7-8$), and the marginal kurtosis has a spike at $t = 1$,
but neither of these is at the same layer as the reverse-view bulge.
The conditional view exposes a layer-localized geometric feature
that exists in the residual-stream ensemble but is invisible when the
ensemble is treated as a single marginal.

**Training-dynamic anomalies that the marginal misses.** The
reverse-view $\lambda$-dip during the co-location window of steps
2,000-11,000 is a training-dynamic feature with no obvious counterpart
in the marginal-view trajectories. The marginal $\log\alpha$ hump
exists at the same training-step window, but it is a different
quantity than $\lambda_{\text{rev}}$, and the dip-vs-hump distinction
is informative: the marginal records a *variance prefactor* anomaly
while the conditional records a *variance-growth slope* anomaly. The
two anomalies are at related but distinct quantities, and the
co-location of their training-step trajectories is empirical evidence
that they reflect the same underlying restructuring of the residual
stream's geometry.

**A direct measure of input-identity persistence and prediction
commitment.** The forward within/between ratio at layer $t$ directly
measures how much of the variance in the residual-stream state at
position $p$ comes from differences between input tokens vs differences
in the context of those input tokens. This is a quantity with a clear
functional interpretation — "how much does the residual stream still
reflect the input token's identity at layer $t$?" — that the marginal
view does not express. Similarly, the reverse view's within/between
ratio measures the residual stream's "prediction commitment" — the
extent to which trajectories converging on the same prediction have
collapsed onto a shared region of $\mathbb{R}^H$.

These functional quantities are basis-invariant scalars, computed
without ever identifying specific directions in $\mathbb{R}^H$ as
carrying specific meanings. The conditional decomposition therefore
adds a layer of *functional* interpretation to the framework while
respecting the basis-invariance that makes the framework's claims
cross-model-robust.

**The variance decomposition is itself a candidate cross-architecture
invariant.** If two architectures trained on the same data with the
same recipe both build similar marginal variance structures (as the
original framework claims), they may also build similar conditional
decompositions of that variance. The forward crossover layer at
$t \approx 1.86$, the reverse mid-network peak ratio at $\approx
18.75$, the per-view $\lambda$ ordering — these are candidate
universal quantities on the same footing as the marginal
basis-invariant statistics. Whether they actually transfer across
architectures is an empirical question that we do not address in
this paper but that would be a natural follow-up.

**What the multi-view extension does not buy.** We are careful not to
overstate the contribution. The multi-view extension does not provide:

- *Mechanistic interpretation* of what individual residual-stream
  directions encode. Like the original framework, the multi-view
  extension is silent on which directions in $\mathbb{R}^H$ correspond
  to which model features. The conditional decomposition factors out
  this question; it tells you about the *variance partition*, not
  about the *bases* in which that variance lives.

- *Semantic interpretation* of the conditional ensembles. The forward
  view conditions on input token identity, but two tokens that are
  semantically related (e.g., "the" and "a", or "cat" and "kitten")
  produce conditional ensembles that the framework treats as
  independent partition cells. The conditional decomposition uses
  syntactic identity (the surface-form token) as the conditioning
  label, not semantic similarity. Semantic clustering of conditioning
  labels would require a different framework.

- *Per-token bundle dynamics.* The conditional ensembles have within
  and between components, but the framework's basis-invariant
  statistics on those ensembles are still population quantities —
  they describe the *typical* per-token bundle, not what any specific
  bundle does. Per-token bundle trajectories through depth are
  outside the framework's epistemic zone.

- *Direct contact with downstream task performance.* The conditional
  decomposition's findings (forward crossover at $t \approx 1.86$,
  reverse bulge at $t = 3$) are basis-invariant structural quantities
  but are not directly tied to specific task-performance metrics.
  Whether a model with a forward crossover at a different layer would
  have measurably different downstream behavior is a question we
  cannot answer from the present measurements.

The contribution of the multi-view extension is therefore *additional
basis-invariant structural and dynamical information about the
residual-stream ensemble*, framed at the same level of abstraction as
the original framework. It is a strict extension within the
framework's existing epistemic zone, not a step into mechanistic or
semantic interpretation.

### 7.2 Basis-invariance and seed-dependent feature bases

The framework's basis-invariance is its core methodological
commitment, and the three-tier cross-seed claim of §6.4 gives a more
detailed picture of what basis-invariance is doing for cross-model
comparison than the binary "alignment fails" finding of Phase 1
allowed. We discuss the interpretation of the three-tier claim here
because it bears on a question of broader significance: what does
the framework's universality claim actually claim, and what would
it take to confirm or refute it?

**The three tiers restated.** §6.4 reported the three-tier picture:
basis-invariant statistics agree across seeds at 2-3% relative spread
(Tier 1); the embedding and readout layers' residual-stream subspaces
align across seeds up to rotation at approximately 50% of the
scramble-null ceiling (Tier 2); the mid-network layers' subspaces do
not align across seeds even up to rotation, sitting at 60-70% of the
scramble-null ceiling (Tier 3).

The three tiers describe different *levels* of cross-seed
reproducibility. Tier 1 is the *intrinsic geometric content* — the
shapes and magnitudes that characterize the residual-stream ensemble
in coordinate-free terms. Tier 2 is the *absolute pose constrained
by external geometry* — the residual-stream's position in
$\mathbb{R}^H$ at layers where shared vocabulary and tied unembedding
impose alignment. Tier 3 is the *absolute pose where no external
constraint applies* — the network interior, where the residual
stream's coordinate frame is free to be whatever the training process
selects.

**What this implies about the framework's universality claim.** The
original framework observes broadly similar basis-invariant statistics
across four pretrained large models. Sarfati et al. interpret this as
evidence of universality. But the four models differ in architecture,
scale, training corpus, and training duration. A strict reading of
their universality claim is that *the basis-invariant statistics
themselves* are similar across these models — not that the
underlying residual-stream subspaces are the same in any sense.

Our three-tier picture is consistent with that strict reading and
provides quantitative support for it. Tier 1 cross-seed reproducibility
at 2-3% is the within-variant noise floor on the basis-invariant
statistics; cross-model differences exceeding this floor would be
attributable to architecture/scale/corpus rather than to seed-level
noise. The cross-architecture work would then ask: do the differences
between $\lambda$ values across architectures exceed Tier 1 noise
floors? Do the forward crossover layers across architectures exceed
Tier 1 noise floors? These are well-posed questions that the three-tier
picture makes precise.

A *weaker* reading of universality would claim that the underlying
residual-stream subspaces are the same across models in some sense.
This reading is *not* supported by our data. Tier 3 shows that even
within a single architecture (where everything is shared except the
random seed), the mid-network subspaces are not aligned across seeds.
There is no a priori reason to expect that cross-architecture
subspaces would be more aligned than cross-seed subspaces, and every
reason to expect they would be less. Whatever universality the
framework's basis-invariant statistics describe, it does not reach
to the subspace level.

**The asymmetry between Tier 2 and Tier 3.** The Tier 2 alignment at
the I/O boundaries is mediated by the shared vocabulary and the tied
unembedding. Two models trained on the same data with the same
vocabulary will produce embedding matrices that are related by an
orthogonal transformation (up to noise from undertrained rare tokens);
their post-final-norm states will produce logits projecting onto the
same vocabulary, also constraining them to be related by orthogonal
transformation. The Tier 2 alignment is the propagation of these I/O
constraints inward through approximately the first and last attention
+MLP block.

The Tier 3 non-alignment at the network interior reflects the absence
of any analogous external constraint on the mid-network feature basis.
Two models trained from different initializations build feature bases
in $\mathbb{R}^H$ that the training process selects based on early
gradient signal — a process that is in principle stochastic at each
step (because the data is shuffled per-seed) and that does not have
any unique fixed point in coordinate-frame space. Different seeds
arrive at different coordinate frames, and no orthogonal rotation
brings them into correspondence.

The Tier 2-vs-Tier 3 asymmetry tells us that *external constraints
on the residual-stream geometry* (vocabulary, tied embeddings) are
what create cross-seed correspondence in $\mathbb{R}^H$. Where these
constraints are absent — at the network interior — there is no
correspondence.

This is consistent with a broader principle that one might call
"basis invariance up to external geometric constraints." The framework
is correctly identifying the basis-invariant statistics as the
appropriate level of cross-model abstraction. Within a single model
family with shared vocabulary, partial alignment is possible at the
I/O boundaries. Beyond that, alignment is genuinely not recoverable.

**What this means for "universality" as a goal.** The framework is
sometimes presented as offering evidence of *universal structure* in
trained-transformer residual streams. The three-tier picture suggests
that "universality" is the wrong word — what the framework is actually
documenting is *intrinsic geometric similarity* of the marginal
ensemble shape, not universal residual-stream structure. The latter
would imply more cross-model correspondence than our measurements
support; the former is what the basis-invariant statistics actually
measure.

A more precise framing of what the framework establishes: trained
transformers across a range of architectures produce residual-stream
ensembles whose *intrinsic shape statistics* (the shape and magnitude
of the variance structure, characterized in coordinate-free terms)
agree to a degree consistent with the noise floors set by within-variant
seed variation. The framework's findings are therefore evidence of
*shape universality*, not *structural universality*. The latter is
not supported, and the former is what the basis-invariant statistics
actually measure.

### 7.3 The per-token covariance finding and the linear-Gaussian framing

The strictest interpretation of the linear-Gaussian framework would
predict that the noise process driving within-input variance at each
layer is *token-independent*: a Gaussian with covariance $\Sigma_t$
that depends on the layer but not on the input token. Under this
prediction, the per-token covariance subspaces at each layer should
agree up to sample noise, with principal angles between them
indistinguishable from the self-consistency null we constructed in
§6.5.

The §6.5 findings show this prediction fails sharply. The principal
angles between per-token covariance subspaces are at $\sim 73°$ median,
substantially above the self-consistency null of $\sim 18°$ and
comparable to the random-subset null of $\sim 71°$. The per-token
covariance subspaces are nearly orthogonal across tokens, far from
the token-independent prediction. The framework's strictest
linear-Gaussian formulation does not describe the trained model.

What does this mean for the framework as a whole?

**Modified linear-Gaussian framework with token-dependent covariance.**
A natural relaxation of the strictest formulation is to allow the
noise covariance to be token-dependent: $\Sigma_t(v)$ for input token
$v$ at layer $t$. The modified framework remains Gaussian (still a
multivariate normal distribution at each layer conditional on the
input) and still tractable (the per-token marginal is a Gaussian with
its own covariance). The modified framework would predict different
per-token covariance subspaces, which is exactly what we observe.

The modified framework is consistent with the basis-invariant
statistics we report in §3-§5. The marginal $\Sigma_t$ is the
input-frequency-weighted average of the per-token $\Sigma_t(v)$
covariances, plus the variance of the per-token means (by the law
of total variance). The marginal basis-invariant statistics
($\lambda$, $\log\alpha$, effective rank, kurtosis) are computed on
the marginal $\Sigma_t$, and their value does not require any
specific claim about whether the underlying per-token covariances
are token-independent or token-dependent.

The framework's marginal statistics are therefore robust to the
distinction between the strictest and the modified linear-Gaussian
formulations. The strict formulation makes a stronger empirical
prediction (about per-token subspace alignment) that we have shown
to be false; the modified formulation makes the weaker prediction
that the marginal distribution is Gaussian, which is approximately
consistent with our measurements (the marginal kurtosis ranges from
$\sim 0.16$ to $\sim 3.87$ across layers, mostly small but with the
$t = 1$ spike indicating non-Gaussian heavy tails at one specific
layer).

**The interpretive significance of token-dependence.** That the
per-token covariances are nearly orthogonal across tokens is a
non-trivial empirical fact. It says that different input tokens'
bundles are spreading along *different directions* in $\mathbb{R}^H$,
not just along *different magnitudes* of the same direction. The
network has built per-token-specific structure into the residual
stream at the level of which subspace each token's bundle occupies.

The cross-seed reproducibility (0.5° dispersion across 4 seeds) and
the rapid emergence (already 59° at step 100) tell us this structure
is not seed-specific noise; it is a property of trained transformers
that emerges essentially from initialization and persists through
training. Whether different seeds build the *same* per-token-specific
structure, or *different* per-token-specific structures, is a
question we have not directly addressed — the per-token covariance
subspaces are seed-dependent in $\mathbb{R}^H$ (since the residual
stream's coordinate frame is seed-dependent, §6.4 Tier 3), but their
*relationship* to each other across seeds — i.e., whether seed A's
"the" subspace is the rotation of seed B's "the" subspace — is a
question that would require a token-level Procrustes alignment we
have not computed.

**What this implies for the framework's epistemic zone.** The original
framework is presented with linear-Gaussian language ("the residual
stream is approximately Gaussian, linear flow plus residual"). Our
measurements support this language at the marginal level but
contradict it at the per-token-conditional level. The framework's
correct interpretation is therefore: *the residual stream's marginal
distribution is approximately linear-Gaussian, but the per-token
conditional distributions are linear-Gaussian only in the modified
sense that each token has its own covariance.*

This is not a refutation of the framework; it is a clarification of
what level the framework's predictions apply at. The marginal
predictions are robust; the per-token predictions of the strictest
formulation are not. Researchers using the framework should be
explicit about which level their claims apply at, and should not
infer per-token predictions from marginal observations.

**The 59° starting value at step 100 is itself informative.** That
the pair-median angle is already 59° at step 100 (only ~250 gradient
steps into training) means the token-specific covariance structure is
not slowly built up over thousands of steps but is essentially present
from the beginning of meaningful training. Three explanations are
consistent with this:

1. The residual-stream's response to different input tokens is
   already differentiated at random initialization, and the
   covariance subspaces are token-specific from the start.

2. The first few hundred gradient steps rapidly build per-token
   covariance structure that then persists through training.

3. Both — there is partial token-specificity at init that the early
   training steps amplify.

We have not directly compared the random-init (seed 9999, untrained)
per-token covariance subspaces, so we cannot distinguish these. The
random-init within/between ratios reported in §6.2 are large at all
layers, which is consistent with each per-token bundle being
substantial in absolute terms even at init, but does not directly
address subspace orientation. This is a follow-up measurement that
would resolve the ambiguity.

### 7.4 Random-init as a reference point and the differential-reshaping picture

§6.2 reported the random-init baseline measurements and used them to
reinterpret the at-convergence structural findings as a differential
reshaping of an initially-uniform-high state. We discuss the
interpretive significance of this framing here, because it bears on
the question of what training is actually doing to the residual stream.

**The naive expectation: training builds new structure.** The first
intuition one might have about training is that it *builds* structure
in the residual stream — at random init the residual stream is
unstructured, and training progressively introduces the structure
that the converged model exhibits. The forward crossover at $t
\approx 1.86$, the reverse mid-network bulge, the per-view $\lambda$
ordering — these would, under the naive view, be features that
training *creates* by gradually constructing them from a flat
baseline.

**What we observe: differential reduction.** Our random-init
measurements (Table 6.3) show this naive picture is incorrect. The
random-init within/between ratios are *high* at every layer — higher
than the converged-state values at every layer. The reverse-actual
ratio at random init is 16.91 at $t = 0$, rising monotonically to
72.34 at $t = 13$; the converged-state values are 7.00 at $t = 0$
and 6.31 at $t = 13$. Training has *reduced* the ratio at every
layer, not built it from zero.

The reduction is differential across layers. At the I/O boundaries
($t = 0, 13$), training reduces the reverse-actual ratio by factors
of $\sim 2.4\times$ and $\sim 11.5\times$ respectively. At the
mid-network ($t = 3$), training reduces the ratio by only $\sim
2.9\times$. The result is the at-convergence "bulge" — a layer-resolved
profile where the mid-network ratio is largest, not because training
built a peak there but because training reduced the surrounding
layers more than it reduced the mid-network.

**The conceptual reframing.** This is a meaningful conceptual shift
in how the multi-view findings should be interpreted. The forward
crossover at $t \approx 1.86$ is created by training (because the
forward ratio at random init is above 1 at every layer, so the
crossover does not exist). The reverse mid-network bulge is not
created by training; it is a *residue* of the random-init high-monotone
profile that training preserves more strongly than it reduces the
surrounding layers. The training-induced structure is the
*differential reduction*, not the bulge itself.

This reframing matters for how the structural findings of §5 are
interpreted. The bulge at $t = 3$ is not a learned feature whose
location and magnitude tell us about what computation happens at
that layer; it is the surviving high-ratio layer in a profile that
training has otherwise flattened. The mid-network "peak" is therefore
not a peak of *learned structure* but a peak in the *residual of a
flattening process*.

The same reframing applies to other structural features in different
forms. The forward crossover is created by training in the strong
sense — the crossover does not exist at random init. The reverse
view's persistent within-dominance (no reverse crossover at any
layer) is inherited from initialization — it was already present
before training. The per-view $\lambda$ ordering at convergence may
or may not be present at init; we have not measured per-view $\lambda$
at the random-init checkpoint and this is a follow-up measurement
that would clarify the picture.

**What this implies about the role of training.** The differential-reduction
framing suggests that training does not build the residual-stream
geometry from scratch; it *shapes* a pre-existing high-variance state.
The shaping process has structural preferences — it reduces the
within/between ratio most strongly at the I/O boundaries, less so at
the mid-network — and these preferences are stable across seeds
(producing the cross-seed reproducibility we observe).

This is consistent with a broader picture in which random initialization
already provides a workable starting point for residual-stream
dynamics, and training fine-tunes the geometry rather than building it
de novo. The framework's basis-invariant statistics at convergence
reflect a *trained-state attractor* whose location in the
basis-invariant statistic space is highly reproducible across seeds,
but the *path* from random init to that attractor is mostly a process
of selective reduction rather than constructive addition.

**Caveat: the random-init baseline is not a linear-Gaussian system.**
We are careful not to overclaim that the random-init state is the
linear-Gaussian theoretical baseline against which training-induced
deviations should be measured. The random-init transformer has
random weights and is approximately but not exactly described by the
linear-Gaussian framework. The high random-init within/between ratios
we observe could reflect either (a) a genuine high-variance random-init
state that training reshapes, or (b) numerical/sampling artifacts of
the random-init configuration. We have shown via the absolute
within and between variance values (Table 6.4) that the random-init
ratios are not numerical artifacts at most layers, but we do not
claim that the random-init state is a clean theoretical baseline of
any specific kind.

What we claim is empirical: training reduces the within/between ratios
relative to a measurable random-init starting point, doing so
differentially across layers. The interpretive overlay of "training
reshapes from an approximately uniform-high initial state" is
suggestive but not formally derived from any specific theoretical
framework.

### 7.5 Open questions

The multi-view extension we have described establishes a framework
for further work. We list here a set of open questions that future
work could address, organized by category.

**Other conditioning choices.** We have used input token and successor
token as the conditioning labels for the forward and reverse views.
Many other conditioning choices are possible and may reveal different
structural features.

- **Conditioning on the previous token** (instead of input or
  successor) would describe how the residual stream represents the
  immediately-prior context. The forward and reverse views we use
  treat the input as the pilot position and the successor as the
  prediction target; conditioning on the previous token would treat
  the immediately-preceding context as a conditioning label.

- **Conditioning on prefix length or position within chunk** would
  describe how the residual stream's geometry depends on how far
  into a sequence the pilot position sits. Our analysis uses fixed
  pilot positions $\{50, 100, \ldots, 950\}$ that vary in their
  prefix length; conditioning on prefix length explicitly would
  factor this out.

- **Conditioning on syntactic or part-of-speech category** would
  group tokens that share grammatical function rather than surface
  form. This requires an external tagger but would test whether the
  residual stream's within/between structure tracks syntactic
  categories.

- **Conditioning on bigram frequency** would distinguish high-frequency
  successor predictions (where the model is confident) from
  low-frequency ones (where it is uncertain). The reverse view's
  per-successor centroids might separate more cleanly for
  high-frequency successors than for low-frequency ones.

Each of these conditionings respects the framework's basis-invariance
(the partition is defined externally to the residual-stream geometry)
and would extend the multi-view family with additional members. We
have not explored them in this paper because the input/successor
conditionings are the simplest non-trivial choices and provide a
clean partition relation to the marginal; extensions are natural
follow-up work.

**Cross-scale extension.** The 150M-parameter scale we use enables
the methodology (multiple seeds, log-spaced checkpoints, multi-view
inference at every checkpoint) but is below the paper's range. The
quantitative findings we report — the forward crossover at $t
\approx 1.86$, the reverse mid-network peak at $t = 3$ with peak
ratio 18.75, the per-token covariance median angle of $73°$ — are
specific to our 150M architecture and may or may not transfer to
larger models.

A natural follow-up is to apply the multi-view extension to larger
models at the same training-corpus scale and recipe, with multiple
seeds to maintain the within-variant noise floors. The methodology
scales straightforwardly to larger models (the conditional-view
computations are linear in $H$ and $N$); the bottleneck is training
multiple seeds at larger scales, which requires correspondingly more
compute.

The questions a cross-scale study would address: do the structural
features at convergence (forward crossover layer, reverse mid-network
peak layer) scale with depth or with absolute layer index? Does the
co-location window scale with total training duration? Does the
three-tier cross-seed claim's depth-dependence track the network
depth? These are well-posed scaling questions that the multi-view
framework supports.

**Cross-architecture extension.** Our prior Phase 1 work specifies a
planned Phase 2 study that varies single architectural axes (depth,
width, FFN intermediate ratio, normalization choice, attention
configuration) against the Phase 1 GELU baseline. Phase 2's
deliverable is whether the basis-invariant statistics agree across
architectures within the within-variant noise floors, with
disagreement attributable to specific architectural choices.

The multi-view extension provides a richer set of cross-architecture
statistics to compare: the forward crossover layer, the reverse
mid-network peak height and location, the per-view $\lambda$ ordering,
the within/between ratio profiles. Each of these is a candidate
cross-architecture invariant; each can be measured at multiple seeds
per variant; each has a within-variant noise floor measurable from
the Phase 1 GELU baseline.

The questions a cross-architecture study would address: do the
within/between ratio profiles agree across architectures within
within-variant noise floors? Does the co-location window's timing
agree across architectures? Is the three-tier cross-seed pattern
the same across architectures? These would address the framework's
universality claim with a precision that the original framework's
single-snapshot measurements cannot.

**Semantic interpretation and mechanistic interpretability.** The
multi-view extension is silent on semantic content of the residual
stream (§7.1). A natural extension is to connect the conditional-view
structural findings to mechanistic interpretability work that
identifies specific computational circuits in the residual stream.
The forward crossover at $t \approx 1.86$ might correspond to a
specific computational transition that mechanistic analysis can
characterize; the reverse mid-network peak at $t = 3$ might align
with the layer where specific prediction-relevant circuits are
located.

This connection would require bridging the basis-invariant level
(at which the multi-view extension operates) with the
specific-direction level (at which mechanistic interpretability
operates). The connection is not direct — the basis-invariant
statistics factor out the specific-direction information that
mechanistic interpretability needs — but the two methodologies could
be applied to the same model and the connections between their
findings explored.

**Relationship to other framework formalisms.** The framework we
extend is one of several recent attempts to characterize trained
transformers at the population level (rather than the
per-token-trajectory level). Other recent work uses related but
distinct frameworks — geometric analysis of feature spaces, manifold
estimation in hidden-state geometry, information-theoretic analysis
of layer-wise computation. The multi-view extension is specific to
the basis-invariant framework's language; whether the structural and
dynamical findings we report would replicate under other framework
formalisms is an open question.

**Open question on the per-token covariance finding.** The most
intriguing finding to us is the rapid emergence of per-token
covariance non-orthogonality (pair median 59° at step 100, only 14°
below its converged value of 73°). The mechanism that produces
near-orthogonal per-token covariances so rapidly is unclear. A
follow-up measurement on the random-init checkpoint (no training)
would clarify whether the structure is intrinsic to the architecture
at initialization or whether the first ~250 gradient steps construct
it. If intrinsic, the question becomes what about the random-init
architecture produces token-orthogonal covariances; if constructed,
the question becomes what about the first few hundred gradient steps
produces them so rapidly. Either answer would be informative about
how the residual stream organizes itself per-token.

### 7.6 Summary of discussion

The multi-view extension contributes additional basis-invariant
structural and dynamical information about the residual-stream
ensemble. Specifically: per-view $\lambda$ values with a
reproducible ordering, the forward crossover layer at $t \approx
1.86$, the reverse mid-network within/between ratio peak at $t = 3$,
per-view effective rank profiles with qualitatively distinct shapes,
and the co-location of the reverse-view $\lambda$-dip with the
marginal $\log\alpha$ hump in training-step coordinates. These
findings are basis-invariant and respect the framework's existing
epistemic zone.

The three-tier cross-seed claim (Tier 1: basis-invariant statistics
agree at 2-3%; Tier 2: I/O boundary subspaces aligned up to rotation
at ~50% of scramble ceiling; Tier 3: mid-network subspaces not aligned
even up to rotation) refines the framework's universality claim into
a more precise picture: what the framework documents is
shape-universality of marginal statistics, not structural-universality
of residual-stream subspaces.

The per-token covariance subspace non-orthogonality (median angle 73°
across token pairs, vs self-consistency null 18°) shows that the
strictest linear-Gaussian formulation does not describe the trained
model. A modified linear-Gaussian framework with token-dependent
covariance remains tractable and consistent with the marginal
statistics; the strict formulation's prediction of
token-independent covariance is empirically refuted.

The random-init baseline shows that training reshapes within/between
ratios from a uniform-high initial state through differential
reduction, reducing the ratio most strongly at the I/O boundaries and
least strongly at the mid-network. The at-convergence mid-network
bulge is a residue of this differential reduction, not a peak of
learned structure.

The open questions for future work span other conditioning choices,
cross-scale and cross-architecture extensions, connections to
mechanistic interpretability, and the mechanism by which per-token
covariance non-orthogonality emerges so rapidly in training. The
multi-view framework provides the methodological scaffolding for
addressing these questions; the empirical answers require further
work.

---
