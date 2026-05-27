# A Multi-View Decomposition of the Residual-Stream Ensemble

## Conditional structure, training dynamics, and the limits of cross-seed alignment in trained transformers

---

## 1. Introduction

### 1.1 What this paper extends

Sarfati et al. (ICLR 2025), in the paper they title "Lines of Thought in
Large Language Models," describe a framework in which the residual
stream of a trained transformer admits a low-dimensional linear-Gaussian
description across layers. The framework is striking in two ways. First,
it characterizes the residual-stream geometry by a small set of intrinsic
quantities — the variance-scaling exponent $\lambda$, the prefactor
$\log\alpha$, an effective rank profile, a per-coordinate kurtosis
profile, and a successive-layer angle profile — that depend only on the
*shape* of the activation cloud at each layer, not on the choice of
basis used to express it. Second, it observes that these quantities take
broadly similar values across several large pretrained models from
different architectural families (GPT-2 medium, Llama-2-7B,
Mistral-7B, Pythia-12B), which the authors interpret as evidence for a
kind of universality in trained-transformer residual-stream geometry.

For each layer transition $t \to t + \tau$, Sarfati et al. fit a linear
map of the form

$$\tilde{x}(t + \tau) = R(t + \tau)\,\Lambda(t, \tau)\,R(t)^\top x(t)$$

where $R(t) \in \mathbb{R}^{H \times H}$ is a per-layer orthonormal basis
recovered from the SVD of layer-$t$ activations across a sample of
"pilot" token positions in held-out evaluation data, and $\Lambda(t, \tau)$
is a diagonal stretch matrix capturing how singular values evolve between
layers. The actual layer transition is the linear prediction plus a
residual $w(t, \tau) = x(t + \tau) - \tilde{x}(t + \tau)$ whose ensemble
distribution is approximately Gaussian with per-coordinate variance
scaling as

$$\sigma^2(t, \tau) \approx \alpha \cdot \tau^\lambda.$$

The recovered $(\log\alpha, \lambda)$ pair is the headline summary of the
residual stream's linear-flow behavior; additional statistics — the
effective rank, the per-coordinate excess kurtosis, the isotropy, the
successive-layer principal-angle profile — describe how the residual
cloud is shaped at each layer and how it rotates between layers. Sarfati
et al. refer to these collectively as "basis-invariant" statistics
because their values are unchanged under any choice of orthonormal basis
for $\mathbb{R}^H$.

The present paper extends this framework in a specific direction: from a
single-view (marginal) characterization of the residual-stream ensemble
to a three-view decomposition that partitions the same ensemble by
conditioning variables. The motivation, the formalism, the experimental
results, and the methodological consequences of this extension are the
substance of what follows.

### 1.2 Two properties of the framework that constrain what an extension can do

Two properties of the basis-invariant framework matter for how an
extension can be designed without leaving the framework's epistemic
zone, and we make them explicit here because they recur throughout the
paper.

The first property is that the linear flow $R(t)$ and its derived
statistics are *population* objects. They describe how the cluster of
all sampled trajectories collectively deforms between layers, not what
any single trajectory does. A given pilot's activation $x_t$ contributes
one row to the data matrix from which $R(t)$ is recovered, but the
framework does not describe that row's individual behavior. Within the
framework as stated, statements about "what the residual stream does"
are statements about the ensemble's variance structure. This is a
deliberate methodological choice — it factors out per-trajectory
specifics that would otherwise vary with input and context — but it also
means the framework is silent on a question that an interpreter might
naturally ask: do *different inputs* take different paths through the
residual stream, and if so, what does the geometry of those
input-conditional paths look like?

The second property is that the basis-invariant statistics are the
appropriate level of abstraction for cross-model comparison. In prior
work (the Phase 1 study summarized in §3 of this paper) we established
that the $R(t)$ matrices themselves — the actual orthonormal bases
recovered by the SVD — share no recoverable structure across independent
training runs of the same architecture. Two seeds trained on the same
data with the same recipe produce $R$ matrices whose top-1 principal
directions are 89° apart, while within-seed split-half measurements
recover the same direction to within 7°. The cross-seed structure is
fully absent, and we have verified this is not an artifact of the
alignment procedure — embedding-space Procrustes recovers a clean
rotation that successfully aligns the embeddings (residual norm 0.10
after rotation on a top-1000-token anchor set), but that rotation has
no effect on the residual stream's principal directions at deeper
layers (R-matrix Frobenius distance after rotation equal to the
random-orthogonal baseline to four significant figures). Different
seeds learn $R$ matrices along seed-specific bases unrelated by any
orthogonal map.

This is to say: the framework's basis-invariant quantities describe a
real shared property of trained transformers, and the framework's
basis-dependent quantities (the $R$ matrices themselves) describe
seed-specific coordinate representations that have no canonical
cross-model meaning. The framework's preferred quantities — $\lambda$,
$\log\alpha$, effective rank, isotropy — are exactly the ones that
factor out the basis-dependent freedom that the training process
exploits, leaving the intrinsic functional content that any equivalent
model should agree on.

Any extension that adds to the framework must respect both properties.
It cannot identify specific directions in $\mathbb{R}^H$ as carrying
specific computational meaning (any such identification would be
seed-specific and therefore not a cross-model statement). It also
cannot describe individual trajectories without leaving the
basis-invariant regime that gives the framework its cross-model reach.
The challenge is to find an extension that adds genuinely new
information about the residual-stream geometry while staying inside the
basis-invariant regime that lets the framework's claims hold across
seeds and (the original framework hopes) across architectures.

### 1.3 What the marginal view leaves out

The framework as Sarfati et al. state it treats the residual-stream
ensemble as a single marginal distribution: all pilots from the held-out
evaluation, pooled together, are characterized by one $\lambda$, one
$\log\alpha$, one effective rank profile, one kurtosis profile per
layer. We refer to this pooled ensemble as the *all-to-all view* in what
follows. In our experimental setup we have $N = 9{,}500$ pilots per seed
per checkpoint (500 held-out chunks × 19 pilot positions per chunk), so
the all-to-all view at any layer consists of $9{,}500$ activation
vectors in $\mathbb{R}^{896}$ that the framework then summarizes by the
basis-invariant scalars and profiles listed above.

The all-to-all view obscures a structural distinction that should matter
to the framework's interpretation. The variance of the all-to-all
ensemble at any layer $t$ reflects two distinct sources. The first is
variance *across* different conditioning categories — for example, the
ensemble pools together pilots whose input token is "the", whose input
token is "and", whose input token is "machine", and so on, with each
input token's embedding occupying a different region of the hidden
space at $t = 0$ and potentially remaining distinguishable at deeper
layers. The second is variance *within* each conditioning category — for
example, the bundle of pilots all sharing input token "the" still has
internal variance because the contexts preceding the "the" differ, and
attention has folded that context information into the residual state.
The all-to-all variance pools both sources without distinguishing them.

The law of total variance partitions the all-to-all variance exactly
into these two components, given any choice of conditioning label $z$:

$$V_{\text{a}}(t) = \mathbb{E}_z\!\left[V_{\text{within-}z}(t)\right] + \mathrm{Var}_z\!\left[\mu_t(z)\right].$$

Here $V_{\text{a}}(t)$ is the all-to-all variance at layer $t$,
$V_{\text{within-}z}(t)$ is the variance computed within the subset of
pilots sharing label value $z$, and $\mu_t(z)$ is the mean of that
subset. The first term on the right is the within-condition variance
averaged over the distribution of conditions; the second is the
variance of the per-condition means across conditions. By the law of
total variance, this identity is exact at every layer for any choice
of conditioning label, modulo the contribution from conditioning labels
that the data does not represent. We discuss the boundary case of
sparsely-represented conditions in §2.5 and the resulting numerical
discrepancy in §3.3.

The original framework reports only the left-hand side. The right-hand
side decomposition is invisible in the marginal framework, yet it
describes structurally distinct functional roles. In the
*input-conditioned* (forward) view — conditioning on the input token at
the pilot position — the between-input variance is the extent to which
input token identity remains distinguishable at layer $t$, and the
within-input variance is the extent to which context has driven
trajectories sharing an input apart. In the *output-conditioned*
(reverse) view — conditioning on the successor token at position
pilot+1 — the between-output variance is the extent to which the
residual state already commits to which token comes next, and the
within-output variance is the residual ambiguity in trajectories that
converge on the same successor. These four quantities — within and
between for each of two conditioning choices — describe functionally
distinct aspects of what the residual stream is doing at each layer,
and the all-to-all variance is their constrained sum.

The functional asymmetry between the forward and reverse views is the
core of why the multi-view extension is informative. At $t = 0$ (the
post-embedding state), the forward view has zero within-input variance
by construction (all pilots sharing input token $v$ have the same
embedding $E(v)$, so their states at $t = 0$ are identical), while the
reverse view has substantial within-output variance (pilots ending up
at the same successor token started from many different inputs). At
$t = L + 1$ (the post-final-norm state), the situation reverses
qualitatively: the forward view has accumulated variance from context
that may or may not still be distinguishable across input tokens, while
the reverse view's within-output variance reflects the residual
ambiguity in predicting that specific successor. The trajectory of
how these two variances evolve through depth — and where they cross
the corresponding between-variance — is a basis-invariant description
of the functional shape of the network's computation, in a sense the
marginal view cannot express.

### 1.4 The three-view decomposition and what it does for the framework

This paper formalizes a three-view decomposition of the residual-stream
ensemble. The three views — all-to-all, forward (input-conditioned),
and reverse (output-conditioned) — are related by the law of total
variance: the within-plus-between sum of each conditional view equals
the all-to-all variance at every layer, exactly, modulo the
contribution from conditioning categories that fall outside the chosen
token set. We work with a 20-token set for each conditional view (the
top-20 most frequent input tokens for the forward view, the top-20
most frequent successor tokens for the reverse view), which together
cover approximately 18-19% of pilot positions in our held-out
evaluation set. The decomposition's identity is approximate to the
extent that the remaining 81-82% of pilots contribute to the
all-to-all variance but not to either conditional within-or-between.
We track the discrepancy explicitly in §3.3.

We further distinguish two variants of the reverse view, conditioning on
either the *actual* successor token in the held-out chunk or the
*predicted* successor token (the model's argmax prediction from the
post-final-norm state's logits). At convergence, the model's argmax
prediction agrees with the actual successor on approximately 40-45% of
pilot positions (this is the held-out top-1 accuracy of our model). On
positions where the model is correct the two reverse views agree; on
positions where the model is incorrect, the predicted-successor view
conditions on the token the model thinks comes next, while the
actual-successor view conditions on the token that actually does. Both
views are valid conditional partitions; both are reported throughout
§5-§6. The actual-successor view describes the residual stream's
relationship to the data-generating process; the predicted-successor
view describes its relationship to the model's own behavior.

The methodological character of the decomposition is in the constraint
structure, not in the additional measurements per se. The conditional
views are not independent quantities that could vary freely; the
within-plus-between identity says that they are *constrained
decompositions* of the same fixed all-to-all object. This constraint
means the conditional family carries strictly more information than the
marginal — the same total variance is allocated differently between
within and between depending on the conditioning choice — and the
allocation curve through depth becomes a basis-invariant signature of
how the residual stream's variance budget is functionally apportioned.

The decomposition stays inside basis-invariance throughout. We
characterize the within and between components of each view by the same
quantities the original framework uses on the marginal: per-coordinate
variance, log-linear variance-growth slopes, effective rank profiles,
per-coordinate kurtosis. None of these depend on specific directions in
$\mathbb{R}^H$. The conditional family is therefore a candidate for
cross-model universality on the same terms as the original marginal
statistics: if architectures trained on the same data with the same
recipe build similar marginal variance structures, they may also build
similar conditional decompositions of that variance, and the
decomposition's shape can be compared directly without requiring any
cross-model basis alignment.

We refer to this decomposition as "multi-view" rather than "two-view" or
"conditional" because it is the *family* of related ensembles, including
the marginal, that constitutes the object of study. Any one view alone
would be a less informative summary; the partition relation between
them is what gives the decomposition its functional content. We sometimes
also use the term "residual-stream geometry" in place of "lines of
thought" when referring to the object the framework characterizes,
because the framework does not actually describe individual trajectories
("lines of thought" in any narrative sense) but rather the population
geometry of the ensemble — the language is slightly more accurate to
what is measured. We cite Sarfati et al.'s original terminology when
attributing claims to them, but use the more literal terminology when
describing what is measured.

### 1.5 Experimental setup

We apply the three-view decomposition to a 150M-parameter Llama-style
decoder-only transformer, trained from scratch on FineWeb-Edu
(sample-10BT subset) for 24,000 steps at four independent seeds. Each
seed receives an independent random initialization (Llama default:
$\mathcal{N}(0, 0.02)$ for embedding and linear weights) and an
independent training-data shuffle order; all other hyperparameters are
identical across seeds. We save 50 log-spaced checkpoints per seed
from step 100 to step 24,000, and run the analyzer (described in §3) at
every checkpoint, producing 200 sets of basis-invariant measurements
and 200 sets of multi-view decompositions in total.

The 150M scale is chosen because it enables the methodology this study
depends on. At this scale, on a single RTX 5090 with 24 GB of memory,
we can run multiple independent seeds (giving within-variant noise
floors that the original framework lacks), save 50 log-spaced
checkpoints per run (giving the training-dynamic resolution that
single-snapshot studies miss), and afford the computational cost of
conditional-view inference at every checkpoint (which would be
prohibitive at frontier scale). Each seed completes training in
approximately 12 hours; the marginal analyzer takes approximately 1.5
hours per seed across all 50 checkpoints; the multi-view analyzer
takes approximately 14 seconds per checkpoint for the activation
collection stage and approximately 200-250 seconds per checkpoint for
the per-view decomposition stage, giving a total of approximately 4
hours per seed for the multi-view analysis.

The 150M scale is below the smallest model Sarfati et al. analyze
(GPT-2 medium at 350M), but our prior Phase 1 study established that
the framework's predictions hold qualitatively at this scale: the
linear flow $R(t)$ converges (H1 PASS on all four seeds with ratio
0.0436 vs threshold 0.10), the basis-invariant statistics are
reproducible across seeds at 1-4% relative spread, and the
boundary-layer anomaly and mid-training $\log\alpha$ hump that we
extend in this paper replicate in the marginal view. The 150M scale is
not "below the paper's range" in any way that obstructs the questions
we ask in this paper; it is below the paper's range in a way that lets
us ask better-posed questions than single-snapshot, single-seed
measurements at larger scale can support.

The 4-seed multi-checkpoint design gives within-variant noise floors
that the original framework cannot, since the published results in
Sarfati et al. are single-snapshot measurements on individual
pretrained-model checkpoints. Our prior Phase 1 study established that
within-seed dispersion on the marginal basis-invariant statistics is
between 1% and 4% of mean value (with per-coordinate kurtosis as the
sole exception, at 19.6% dispersion, dominated by a single seed
outlier we discuss in §3.5). We use these dispersion bounds as the
noise floor against which the conditional-view results in §5–§6 should
be calibrated.

All measurements in this paper are at the basis-invariant level.
Nothing in what follows requires interpreting individual residual-stream
directions, identifying specific principal components with specific
computational meanings, or aligning the residual stream of one seed to
another at the level of $R(t)$ matrices. The §6.4 results on cross-seed
Procrustes alignment of residual-stream *activations* (not $R(t)$
matrices) are a partial exception: we measure how well the activations
can be aligned across seeds up to rotation as a way of quantifying the
extent to which the residual-stream subspace itself, not just its
basis-invariant statistics, is reproducible. The finding is that
activation alignment partially recovers at the I/O boundaries even
though $R$-matrix alignment fails, and this nuance is the substance of
§6.4.

### 1.6 What this paper contributes

This paper contributes five findings of distinct character. We summarize
each here and indicate where the supporting detail appears. Each finding
is supported by quantitative cross-seed agreement, methodological
caveats are explicit, and the framing chosen for each is justified
against alternative framings we considered and rejected.

**Methodological — the three-view decomposition itself.** The paper's
primary contribution is the multi-view extension of the basis-invariant
framework. We make the variance decomposition exact at the conditional
level, define per-view basis-invariant statistics (within and between
per-coordinate variances, per-view $\lambda$ and $\log\alpha$
variance-scaling fits, per-view effective rank profiles, per-view
kurtosis profiles) that are constrained by the partition structure to
sum to the marginal at every layer, and demonstrate that the resulting
conditional family is well-defined, numerically tractable at this scale,
and gives a sharper functional account of the residual stream than the
marginal framework alone supports. We define both the
actual-successor and predicted-successor variants of the reverse view
and report both throughout. §2 formalizes the decomposition; §5 reports
its at-convergence findings; §6 reports its training-dynamic findings.

**Structural — the trained network's variance allocation pattern.** At
the converged network the within-input variance overtakes the
between-input variance at a sharp crossover layer that sits in the
first two transformer blocks. The crossover layer (defined as the
interpolated $t$ at which the within/between ratio crosses 1.0) is
$t \approx 1.86$ in the cross-seed mean, with per-seed values in the
range $[1.75, 1.94]$ — well below the boundary of the second block.
Context-driven differentiation overwhelms input identity within the
first 15% of network depth and stabilizes there at a within/between
ratio of approximately 3.0–3.4 through the inner layers. In the
reverse view, the within-output variance is dominant at every layer
(no reverse crossover exists), with a pronounced peak in the
mid-network around layer 3 (within/between ratio = 18.8 at the
cross-seed mean) and a slow decline through the deeper layers to a
final value of 6.3 at the post-final-norm state. The actual-successor
and predicted-successor reverse views have qualitatively the same
shape but differ in magnitude: the actual-successor peak is 18.8 and
the predicted-successor peak is 11.9, a difference of about 1.6×. The
forward and reverse views give qualitatively different profiles of
what the network is doing as a function of depth — input-identity
crossover within two layers vs persistent within-dominance throughout
— with cross-seed dispersion of 2-3% at every layer for both. §5 reports
the structural findings in detail.

**Dynamical — co-location of macro and micro events through training.**
The marginal $\log\alpha$ statistic exhibits two well-characterized
training-dynamic anomalies: a mid-training hump centered at step
$\approx 5{,}000$, and a late-training decline through training's end.
We show that the conditional views exhibit a co-located anomaly: the
reverse-view variance-growth slope $\lambda_{\text{rev}}$ dips during
the same training-step window in which the all-to-all $\log\alpha$
humps, with the lambda-dip minimum at step 5014 and the log-alpha hump
peak at step 5607 (well within one log-spaced checkpoint interval; our
checkpoints are log-spaced approximately every 12% of training). The
co-location is not just a coincidence of two scalar time series. The
layer-resolved within/between ratio for the reverse view (presented as
a heatmap in §6.1) intensifies in the same training steps and the same
layers (mid-network, $t = 4-9$) where the all-to-all anomaly is most
pronounced. The macro/micro correspondence is therefore a
two-dimensional claim — about training-step × layer — not merely a
one-dimensional claim about training-step coincidence. We characterize
the co-location window as approximately steps 2049 to 10969 (the
checkpoints adjacent to the lambda-dip minimum), during which both the
marginal $\log\alpha$ statistic and the conditional reverse-view ratio
exhibit their anomalous behavior. §6.1 reports the co-location finding.

**Boundary effects — training reshapes a uniformly high initial state.**
Random initialization produces a within/between ratio that is high at
every layer and every view; between-condition variance is small because
untrained models have not yet built representations that differentiate
themselves by input or by predicted output. At random initialization
(seed 9999, no training, same model architecture as the trained seeds),
the reverse-actual ratio reaches 72.3 at the deepest layers vs 6.3 at
convergence — an order of magnitude difference. The forward-view ratio
at $t = 1$ is 2.04 at random initialization vs 0.084 at convergence — a
24× reduction, with a sign change (random init has within > between at
$t = 1$; trained models have between > within). The forward crossover
at $t \approx 2$ does not exist at random initialization — the ratio
rises monotonically through layers from 2.0 at $t = 1$ to 8.2 at the
output, never crossing 1.0 in the trained sense. The mid-network
reverse-view bulge does exist at random initialization, but with a
qualitatively different shape than at convergence: monotonically
increasing to a high plateau (rising from 16.9 at $t = 0$ to 72.4 at
$t = 12$, with no mid-network peak) rather than peaking and declining.
Training does not *create* the bulge; it reshapes the profile, reducing
the within/between ratio most aggressively at the boundary layers (where
I/O structure is built) and less aggressively at the mid-network
(where the bulge survives). The trained-model's mid-network bulge is
therefore a residue of the initial high-ratio state, not a peak of
learned structure. We confirm via sanity checks on the absolute within
and between variance values that this finding is not a numerical
artifact of dividing two small numbers — the absolute between-variance
at random init is at most an order of magnitude smaller than at
convergence (not zero), and the within-variance values are real and
nonzero at every layer except the literal embedding state $t = 0$
where they are zero by construction for the forward view. §6.2 reports
the random-init comparison.

**Cross-seed reproducibility — basis-invariant statistics agree;
absolute pose does not.** The basis-invariant statistics reproduce
across seeds to within 2-3% at every layer of the conditional views,
consistent with what the original framework's universality claim should
imply, quantitatively measured. The same seeds, however, occupy
different subspaces of $\mathbb{R}^H$. A per-layer orthogonal Procrustes
alignment of one seed's residual-stream activations onto another's
leaves a residual norm of 56.5% of the worst-case scrambled baseline
(0.628 vs ceiling 1.112), with a clear U-shape across layers: the
residual is best (smallest, 0.417) at $t = 1$, just after the first
attention+MLP block, and worst (0.700) at the deep middle layer $t = 9$,
recovering partially at the output (0.510 at $t = 13$). The shared
vocabulary and tied unembedding constrain the embedding and readout
layers to be aligned across seeds up to rotation, but the network's
internal feature basis is not. This produces a three-tier cross-seed
claim that we discuss in §6.4: basis-invariant statistics agree across
seeds tightly (Tier 1); the embedding and readout layers are aligned
across seeds up to rotation but not exactly, with Procrustes residuals
of approximately 40-50% (Tier 2); the mid-network is not aligned even
up to rotation, with residuals of 60-70% (Tier 3). The three-tier
structure subsumes the binary Phase 1 finding that $R$-matrix alignment
fails outright, and gives a more nuanced characterization of where in
the network the alignment fails. §6.4 reports the cross-seed alignment
findings.

**Per-token covariance non-independence.** A linear-Gaussian description
in which the residual at each layer has a single token-independent noise
covariance would predict that different input tokens' bundles spread in
the same dominant directions at each layer. This prediction fails
sharply. The principal angles between the top-20 per-token covariance
subspaces are nearly orthogonal at most layers (median angle 73° across
token pairs, vs 18° for the same-token sample-noise baseline). The
failure is reproducible across seeds to within 0.5° (seed 0: 73.0°;
seed 1: 73.5°; seed 2: 73.2°; seed 3: 73.2°), robust to the choice of
subspace dimensionality from $k = 5$ through $k = 50$ (pair median
ranging from 79.3° at $k = 5$ to 67.0° at $k = 50$, in a range that is
always far above the self-consistency null which itself ranges from
23.5° to 12.5° over the same $k$ values), and present at the earliest
checkpoint we examine. At step 100, the pair median is already 59.1°
vs self-consistency null 14.7°, and it rises monotonically through
training to its final value of 73.0° at step 24,000. The framework's
marginal basis-invariant statistics agree across seeds despite each
seed building per-token covariance subspaces that are nearly orthogonal
at the individual-token level. §6.5 reports the per-token covariance
findings.

### 1.7 What this paper does not claim

We make no claim that the residual-stream geometry recovered here
extends to substantially larger scales than 150M parameters. The 150M
scale is below the smallest model Sarfati et al. analyze, and while the
framework's qualitative predictions hold at our scale, the quantitative
values of $\lambda$ and $\log\alpha$ we report differ from the paper's
published values by amounts consistent with scale dependence (our
$\lambda \approx 0.426$ vs the paper's Llama-2-7B $\lambda \approx
0.46$, our $\log\alpha \approx -3.28$ vs the paper's $-5.40$, both in
the original paper convention). Whether the three-view decomposition's
specific findings — the forward crossover at $t \approx 2$, the
mid-network reverse bulge at ratio 18.8, the per-token covariance
median angle of 73° — quantitatively transfer to larger models is a
separate empirical question we do not address. The methodological
contribution (the decomposition itself) is scale-independent; only the
specific numerical values reported are tied to our 150M architecture.

We make no semantic claim about what the residual stream encodes. The
framework characterizes the variance structure of the ensemble, not its
interpretive content. Our findings about per-token covariance
non-orthogonality do not say that different tokens' bundles are doing
*different things* in any mechanistic sense; they might be carrying
related computational content in different orthogonal bases of
$\mathbb{R}^{896}$, and our basis-invariant analysis is silent about
that distinction. We make no claim about which principal directions
correspond to which model features, no claim about whether specific
attention heads contribute disproportionately to specific reverse-view
profiles, and no claim about whether the conditional bundles correspond
to any interpretable computational quantity. The decomposition
characterizes the variance partition; its interpretive significance is
open.

We make no claim about whether the cross-seed non-alignment in
$\mathbb{R}^H$ is specific to this architecture or to this training
recipe. We have measured it for one configuration (Llama-style,
RMSNorm, RoPE, GELU FFN, tied embeddings, 12 layers, $H = 896$,
FineWeb-Edu, 24,000 steps, 1.57B tokens) and report what we found.
Whether different optimizers, different corpora, different normalization
choices, or different embedding-tying schemes would produce different
cross-seed alignment patterns is an open question. Our prior Phase 1
report identifies a planned Phase 2 study that varies single
architectural axes against this baseline, which would address some of
these questions, but Phase 2 is outside the scope of the present paper.

We make no claim that the multi-view extension *should* be the right
extension. Other conditional partitions are possible — conditioning on
prefix length, on syntactic category, on bigram frequency, on the
identity of the previous token rather than the next — and may reveal
different structural features. The forward and reverse views by input
and successor token are the simplest non-trivial conditionings that
respect the framework's basis-invariance, and the partition identity
that ties them to the marginal makes them mathematically clean, but
they are not the only viable choice. We chose them because they
correspond to natural quantities — input identity and prediction
commitment — and because the variance decomposition then has a direct
interpretation, not because we believe other conditionings would be
uninformative. We would consider extending to other conditionings a
natural follow-up to this work.

We make no claim about whether the per-token covariance non-orthogonality
finding represents a meaningful deviation from the linear-Gaussian
baseline in any interpretive sense. Our test addresses one specific
prediction of the strictest linear-Gaussian formulation — that the
noise covariance $\Sigma_t$ is token-independent — and shows that this
prediction fails. A modified linear-Gaussian framework with
token-dependent covariance $\Sigma_t(v)$ is still Gaussian and still
tractable, and the marginal basis-invariant statistics we report are
computed in a way that does not distinguish between the two
formulations. The deviation we measure is real and reproducible, but
its interpretive significance is open. We discuss the modified
linear-Gaussian formulation explicitly in §7 because it provides a
useful framing for what the framework's marginal statistics correctly
capture and what they do not.

We make no claim that the random-init baseline of §6.2 is a clean
linear-Gaussian system. A randomly-initialized transformer has random
weights and a random noise structure at each layer, and is approximately
but not exactly described by the linear-Gaussian framework. We use the
random-init measurements as a *reference point* against which to
calibrate training-induced changes, not as an instantiation of any
specific theoretical baseline. The framing "training reshapes the
within/between ratios from a uniform-high initial state" is the
empirically correct description of what we observe; the further claim
"the initial state is what the linear-Gaussian baseline predicts" is
an interpretive overlay that we offer as suggestive but do not formally
verify.

### 1.8 Document structure

The remainder of the paper is organized as follows.

§2 formalizes the basis-invariant framework, defines the three views of
the residual-stream ensemble, states the variance decomposition that
constrains them, specifies the per-view basis-invariant statistics we
will report, describes how the token sets $V$ (forward) and $W$
(reverse) are selected, and describes the orthogonal Procrustes
construction we use in §6 for cross-seed alignment. §2 is the
methodological core of the paper; the experimental sections that
follow refer back to its definitions repeatedly.

§3 describes the trained model in full architectural detail, the
training recipe, the activation-collection pipeline, and the
at-convergence findings of the marginal framework at our scale. It
establishes that the multi-view extension is being built on a baseline
faithful to the original framework's predictions, and it gives the
within-variant noise floors on the marginal statistics that calibrate
the conditional-view dispersion in §5–§6. §3 reproduces the Phase 1
results at this scale with full quantitative detail, including the
per-seed values of $\lambda$, $\log\alpha$, effective rank, kurtosis,
and isotropy, the boundary-layer effect's magnitude and reproducibility,
and the Phase 1 cross-seed alignment-failure finding that motivates
the multi-view extension's reliance on basis-invariant statistics.

§4 reports the training dynamics of the marginal framework's
statistics. The $\log\alpha$ trajectory through training (the
mid-training hump centered at step ≈ 5000, the late-training decline
through training's end), the $\lambda$ trajectory, the boundary-layer
anomaly's emergence trajectory from step 100 through plateau at step
~5000, the post-final-norm anomaly's emergence on the same schedule,
the flow-distance trajectory's mid-training bump, and the
late-training kurtosis rise. These training-dynamic features will be
the landmark events that §6 docks onto when it argues that
conditional-view anomalies co-locate with marginal-view anomalies.

§5 reports the multi-view structural findings at the final checkpoint:
the within/between variance ratios across all three views with their
crossover layers, the reverse-view mid-network bulge with full
quantitative characterization, the per-view effective rank and
kurtosis profiles, the per-token bundle dispersion analysis, and the
qualitative distinctness of the three views. All findings reported with
full cross-seed dispersion and quantitative comparison to the marginal
baseline.

§6 reports the multi-view dynamical findings. §6.1 reports the
co-location of $\lambda$-dip and $\log\alpha$-hump training-dynamic
events and the layer-resolved version of the co-location finding. §6.2
reports the random-init baseline comparison and the consequent
reframing of the bulge as a training residue. §6.3 reports the
training-dynamic heatmaps of the within/between ratio per view. §6.4
reports the cross-seed Procrustes alignment results in the multi-view
setting, including the three-tier claim about intrinsic geometry vs
absolute pose. §6.5 reports the per-token covariance non-orthogonality
results with k-robustness, cross-seed reproducibility, and
training-evolution analyses. §6 is the longest section of the paper.

§7 discusses what the multi-view extension does and does not buy the
framework, the relationship between basis-invariance and seed-dependent
feature bases, the implications for the framework's universality claim,
the relationship between the linear-Gaussian framework and the actual
empirical behavior of trained transformers (specifically the
per-token-covariance dependence), and several questions left open.
§8 summarizes the paper's contributions and indicates directions for
future work.

The paper includes 11 figures and approximately 12 tables across its
sections. The figures referenced in the text are produced by analysis
scripts that we describe in Appendix A. Code for replicating all
experimental results is available at [URL TBA]; raw measurement data
(per-seed per-checkpoint multi-view results, principal angle data,
Procrustes residuals) is at [URL TBA].

---
