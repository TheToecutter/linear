# A learned dictionary of trajectory-deviation modes emerges during training in transformer residual streams

**Document status:** v1 (comprehensive standalone report)
**Audience:** AI conference reviewer (NeurIPS / ICLR / equivalent), readers familiar with the transformer residual stream but not with prior work in this project series
**Scope:** Cross-cell low-dimensional structure of conditional-mean deviations from marginal residual-stream flows, tested in two complementary conditioning views (input-conditioned bigram pair view and output-conditioned reverse view), at the final checkpoint and across the full training trajectory of a 150M-parameter Llama-style transformer trained from scratch with four independent random seeds
**Companion documents:** This document is intended to be standalone. For prior work that this study builds on, see `MARKOVIANITY_WRITEUP.md` (the test that established the present work's prerequisite finding) and `PHASE_1_WRITEUP.md` (the basis-invariant macro statistics that motivate the framework). These are referenced when relevant but their content is recapitulated here as needed.

---

## 0. Summary

### What we did

Trained transformer language models do not just produce next-token
distributions; their internal layer-by-layer activations trace out
*trajectories* through high-dimensional residual-stream space. Sarfati
et al. (ICLR 2025, "Lines of Thought") showed that the marginal
geometry of these trajectories — pooled across all input contexts —
admits a remarkably compact description: a deterministic linear flow
plus an isotropic Gaussian noise term. Their framework recovers this
description from a small SVD on layer-wise activations and produces
basis-invariant statistics that hold across architectures.

What the Sarfati framework's noise term *hides* is exactly what
distinguishes one token's trajectory from another's. The framework
absorbs all context-dependent computation — every input-specific,
output-specific, position-specific contribution — into a single
Gaussian noise scalar per layer. A natural question follows: when we
unpack that noise term by conditioning on input or output token
identities, what structure does it have? Is the per-context
contribution organized along a small number of shared directions in
residual-stream space (a "dictionary"), or does each input/output
context drive the residual stream along its own private direction?

The present study answers this question through a *multi-view
decomposition* of conditional-mean deviations from marginal flow.
For each cell defined by either a successor token alone (the
**reverse view**) or an (input, successor) bigram pair (the **pair
view**), we compute the cell-conditional mean trajectory, subtract
the appropriate marginal mean (global for reverse, input-conditioned
for pair), and ask whether the resulting deviation tensor has
low-dimensional structure across cells.

### What we find

The study produces a two-part empirical result, reproduced cleanly
across four independent training seeds of a 150M-parameter
Llama-style transformer:

**Reverse view: a learned dictionary emerges during training.** At
the final checkpoint, the cell-conditional trajectory deviations
concentrate $\sim$8.4% more of their total energy in the leading 8
singular vectors than the within-cell shuffle null would predict
(cross-seed mean, std 0.5%, 11σ above null). The dictionary structure
is *not* present at initialization — the apparent step-100 signal is
a noise-floor artifact of tiny total deviation magnitude. Genuine
dictionary structure emerges between training steps $\sim$750 and
$\sim$5000, asymptoting at the trained level. This signature is
reproducible across seeds despite the seeds learning orthogonal
residual-stream bases, satisfying the basis-invariance criterion
that the broader lines-of-thought program rests on.

**Pair view: bigram-specific structure is small and non-monotonic.**
After subtracting the input-marginal mean (i.e., asking what the
bigram contributes *beyond* what the input alone determines), the
residual shared structure has peak energy concentration only
$\sim$2.5% above the within-input shuffle null at the final
checkpoint (~3σ). The training trajectory of this signal is
non-monotonic: it rises to a peak of $\sim$4.6% around step 2000 and
*declines* to $\sim$2.5% by step 24000. We further observe a
*leading-edge inversion*: at the lowest k, the real cumulative energy
is *below* the null (by 3–5%, ~8σ), indicating bigram-conditioned
deviations live in directions orthogonal to the dominant within-input
variance axes — i.e., bigram-specific structure is cell-specific
rather than dictionary-organized.

The contrast between the two views — a factor of ~3 in peak excess
concentration — quantifies how much of the apparent conditional
structure is input-marginal vs bigram-specific. About 70% of what
the reverse view sees is input-token structure that survives the
output filter; about 30% is genuinely bigram-conditioned. The
bigram-conditioned 30% does not form a shared dictionary; it lives
in per-cell directions.

### Why it matters

In the larger picture of the multi-view program, modified Test 3
fills in the central empirical gap: it tells us *what shape* the
context-dependent computation absorbed by the Sarfati framework's
noise term actually has. The answer:

- Mostly per-input-token (the input-marginal contribution).
- Plus a small dictionary of $\sim$8 shared modes that organize
  output-conditional trajectories across successor tokens.
- Plus a small bigram-specific contribution that is structured
  cell-specifically rather than along a shared dictionary.

Combined with the prior result that training actively *decouples*
the residual stream from Markovianity — i.e., that ~88% of the
per-layer update is not a function of the current state but of
hidden context — this completes a tripartite account of what
training adds on top of the architectural residual:

1. **Markovianity decoupling** (steps 200–2000): the network learns
   to make attention selective, so per-layer updates depend on
   context rather than on current state alone.
2. **Pair-mode bigram dictionary** (steps 400–2000, then decline):
   transient mid-training emergence of bigram-specific shared
   structure, absorbed into the input-marginal as training matures.
3. **Reverse-mode output dictionary** (steps 750–5000): emergence
   of the shared dictionary of $\sim$8 output-relevant trajectory
   modes that persists for the rest of training.

Three distinct learned-geometric phenomena on three distinct
training-time schedules. The reverse-mode dictionary is the
candidate object for further inspection: it has the basis-invariance,
the cross-seed reproducibility, and the structural compactness
required to characterize what the Sarfati framework's noise term has
been hiding all along.

The rest of this document develops these results in detail. §1
introduces the lines-of-thought framework and the multi-view program
from scratch. §2 sets up the conceptual machinery (pair-conditional
ensembles, law of total variance, action functional). §3 lays out
the pre-registered hypotheses. §4 specifies the experimental setup.
§5 specifies the test operationally. §6 reports synthetic validation.
§7 reports cross-seed final-checkpoint results. §8 reports training
trajectories. §9 recounts the experimental sequence that produced
the test in its present form — the design errors, false starts, and
diagnostic discoveries that converged on the final protocol. §10
interprets the results synthetically. §11 reshapes the remaining
multi-view tests in light of the findings. §12 lists open questions
and limitations. §13 gives a one-page summary with costs and
deliverables.

---

## 1. Background

### 1.1 The transformer residual stream and the trajectory it produces

A decoder-only transformer processes a sequence of tokens by mapping
each to an embedding vector and then refining each token's
representation through a stack of transformer blocks. Each block adds
a residual update: if $x_t \in \mathbb{R}^H$ is a token's hidden
state after block $t-1$ (with $x_0$ the post-embedding state), then

$$x_{t+1} = x_t + \mathrm{block}_t(x_t, \mathrm{context}),$$

where $\mathrm{block}_t$ produces an additive contribution combining
multi-head attention output (which reads from the contexts of all
prior tokens in the sequence) and a feed-forward subblock. The
sequence of states $x_0, x_1, \ldots, x_L$ along the layer stack
constitutes a single token's *trajectory* through residual-stream
space, often called its *line of thought*.

These trajectories are observable. For any held-out text sample, we
can record the per-layer state of every token at every position, and
ask what geometric structure this ensemble of trajectories has.

### 1.2 The lines-of-thought framework

Sarfati et al. (ICLR 2025, "Lines of Thought") proposed and empirically
validated a remarkably compact description of this ensemble. For
several large pretrained models (GPT-2 medium, Llama-2-7B, Mistral-7B,
Pythia-12B), they showed that:

- **The ensemble's covariance has low effective rank.** At each layer,
  the trajectory ensemble's covariance matrix has effective rank far
  below the ambient hidden dimension $H$ (e.g., ~256 dimensions in a
  768-dimensional space for GPT-2). The trajectories lie on a
  low-dimensional curved manifold inside $\mathbb{R}^H$.

- **The manifold's principal axes rotate smoothly across layers.**
  Computing the centered SVD of the activations at each layer gives
  an orthonormal right-singular-vector basis $R(t)$ that describes
  the manifold's principal directions at depth $t$. The trajectory
  of $R(t)$ across layers is smooth — successive layers have
  small principal angles between their bases.

- **Deviations from the average flow are approximately Gaussian and
  isotropic in the orthogonal complement to the principal axes.** If
  we project trajectory points onto the principal axes recovered by
  the SVD, the residual (orthogonal) component has approximately
  Gaussian per-coordinate statistics with isotropic variance, and
  this variance grows exponentially with depth at a rate $\lambda$.

These three observations together support a generative model of the
trajectory ensemble: a stochastic differential equation in continuous
layer-depth $t$ of the form

$$dx(t) = \underbrace{\left[\dot R(t) R(t)^\top + R(t) \dot S(t) R(t)^\top\right] x(t)}_{\text{drift}\;b(x,t)}\, dt + \underbrace{\sqrt{\alpha\, \lambda\, e^{\lambda t}}}_{\text{noise scale}\;\sigma(t)}\, dw(t),$$

where $w(t)$ is a Wiener process, $S(t) = \log \Lambda(t)$ encodes the
per-layer stretches along the principal axes, and the drift is a
linear function of $x$ entirely determined by the recovered SVD
geometry. The framework reduces the entire transformer's residual
stream geometry to a handful of parameters per layer: the principal
rotation $R(t)$, the stretches $\Lambda(t)$, the noise scale
$\sigma(t)$, and the noise exponent $\lambda$.

### 1.3 What the framework deliberately abstracts away

The framework's core methodological move is what it does with
context-dependent computation. The transformer's actual per-layer
update is

$$\Delta x_t = \mathrm{MLP}_t(x_t) + \mathrm{Attention}_t(x_t,\;\mathrm{KV\;cache}),$$

where the attention term reads from a key-value cache built from
*every preceding token in the sequence*. This is the network's
mechanism for context-conditional computation: two pilots that
happen to be at the same residual-stream state $x_t$ will receive
different $\Delta x_t$ values if they sit in different left-contexts,
because attention will read differently from their respective KV
caches.

The Sarfati framework's drift term $b(x, t)$ depends only on $x$
(and on $t$ through the precomputed $R, \Lambda$ matrices). It has
no input for context. Everything the attention term does that is
*not* recoverable from $x_t$ alone — that is, all the
context-conditional computation — is absorbed into the noise term
$\sigma(t)\, dw$.

The framework's success is that this approximation captures the
marginal ensemble geometry remarkably well. Pooled over all contexts,
the attention output averages to an approximately isotropic Gaussian
distribution with exponentially growing variance, which is exactly
what the framework's noise term assumes. The framework is therefore
*faithful to the marginal*, but it is *blind to the conditional*: by
construction, it cannot distinguish what the attention term is doing
on any specific pilot, only what it does on average over the ensemble.

The present study, and the broader multi-view program it sits
within, is about characterizing what that absorbed noise actually
contains. The lines-of-thought paper itself does not address this
question.

### 1.4 Phase 1: cross-seed reproducibility of the framework

Prior work in this project series (`PHASE_1_WRITEUP.md`) reran the
lines-of-thought analysis on a 150M-parameter Llama-style transformer
trained from scratch across four independent random seeds, with 50
log-spaced training checkpoints per seed retained. The Phase 1
findings established three properties of the framework relevant to
the present study:

- **The framework's basis-invariant statistics are reproducible
  across seeds to ~1% relative on the noise exponent $\lambda$, ~5%
  on the log-prefactor $\log\alpha$, and similar levels on
  effective rank, kurtosis, and the qualitative shape of the
  principal-rotation $R(t)$ trajectory.** This is a remarkably
  tight cross-seed agreement.

- **The framework's *coordinate basis* is not reproducible across
  seeds.** At every layer, the principal directions found by seed
  0's SVD are approximately 89° from seed 1's, indistinguishable
  from a random pair of orthonormal subspaces in $\mathbb{R}^H$.
  Each seed learns a different basis for the residual stream, but
  the basis-invariant statistics on those bases agree.

- **Several training-dynamic anomalies were found.** A peak in
  $\log\alpha$ at training step ~5000, a mid-training bump in
  $\Sigma$-distance from final values, an emergence pattern of the
  post-final-norm anomaly between steps 1000 and 5000. These were
  characterized but their mechanism was not identified.

The Phase 1 pairing "agreement on basis-invariants, disagreement on
bases" is the central puzzle the multi-view program is designed to
investigate. If the basis-invariant statistics described the *learned
function* of the network, cross-seed reproducibility would require
each seed to learn essentially the same function in different
coordinates — an implausibly strong claim. The natural alternative
is that the basis-invariant statistics measure the *architectural
residual* — the part of the per-layer update that follows
mechanically from the MLP, the residual connection, and the norm
structure — which is the same across seeds. The function would then
live in a part of the dynamics the framework doesn't see.

### 1.5 The Markovianity test (prior work in this series)

A direct test of the architectural-residual reading was done in a
companion study (`MARKOVIANITY_WRITEUP.md`). The test asks: if we
condition on the residual-stream state $x_t$ by binning it into
clusters via top-$d$ PCA and k-means, what fraction of the per-layer
update $\Delta x_t$ variance does the binning predict? Call this
fraction $R^2_{\mathrm{pos}}$.

If the lines-of-thought SDE were generatively accurate — if
$\Delta x_t$ were, in expectation, a function of $x_t$ alone — then
$R^2_{\mathrm{pos}}$ would be close to the structural ceiling
$V^{\mathrm{cap}}(t)$ set by how much of $x_t$'s variance the top-$d$
PCs capture. The Markovianity test established:

- **At the final checkpoint, $R^2_{\mathrm{pos}} \approx 0.10$**, with
  a ceiling-normalized resolved fraction of $\sim$0.70. About 12% of
  per-layer update variance is a function of $x_t$; the other 88% is
  context that does not project visibly into $x_t$ in any subspace
  the binning resolves (a dimensionality sweep confirmed saturation
  by $d = 3$–$5$).

- **Across training, $R^2_{\mathrm{pos}}$ decreases monotonically
  from ~0.57 at step 100 to ~0.10 by step ~3000**, a 5.7× reduction
  asymptoting well before eval loss converges. The
  earliest-converging geometric property of the trained model is its
  *non*-Markovianity in $x_t$.

The interpretation:

> The lines-of-thought framework describes the **architectural
> residual** of the trained transformer — the part of the per-layer
> update that mechanically follows from the MLP, the residual
> connection, and the norm structure, regardless of what the
> attention weights have learned. Training is fundamentally a
> process of *decoupling* the residual stream from this architectural
> Markovianity. The framework's noise term absorbs everything
> training adds.

This is the **H_decoupling** result: training adds context-dependent
computation that lives outside the framework's drift term. Where it
actually lives — and what shape it has — is the question that
modified Test 3 addresses.

### 1.6 The multi-view program

A separate proposal in this series (`MULTI_VIEW_PROPOSAL.md`)
described a programmatic extension of the lines-of-thought framework
from a single-view (all-to-all) characterization to a *three-view
decomposition* of the residual-stream bundle:

**All-to-all view.** The marginal ensemble: every (token, context,
position) triple in the held-out data is included, treated
exchangeably. This is the Phase 1 view. Variance is pooled over
input token identity, successor token identity, context, and
position. This is what the Sarfati framework fits.

**Forward view (input-conditioned).** Pilots are filtered by which
token sits at the pilot position. For each input token $v$, the
forward-view ensemble $E_v$ is the subset of pilots whose pilot
token is $v$. By construction, $E_v$ has zero variance at $t = 0$
(every trajectory starts at the embedding of $v$) and gains variance
through subsequent layers as attention folds in left-context. The
forward view measures the *rate and dimensionality of context-driven
differentiation* for a fixed starting point.

**Reverse view (output-conditioned).** Pilots are filtered by the
successor token — the token the network is being asked to predict.
For each successor token $w$, the reverse-view ensemble $F_w$ is the
subset of pilots whose `next_token` is $w$. By construction, $F_w$
has large variance at $t = 0$ (many different inputs can precede $w$)
and contracts through subsequent layers as the model converges on a
shared prediction. The reverse view measures the *rate and
dimensionality of prediction-driven convergence*.

The law of total variance ties the three views together: at every
layer $t$ and for any conditioning variable $z \in \{v, w\}$,

$$V_{\mathrm{all-to-all}}(t) = \mathbb{E}_z\!\left[V_{\mathrm{within}\text{-}z}(t)\right] + \mathrm{Var}_z\!\left[\mu_t(z)\right],$$

where the first term is the average within-condition variance and
the second is the variance of per-condition means. The three views
are not independent measurements; they are *constrained
decompositions* of the same marginal variance budget.

Most relevant for the present study is a fourth, joint view:

**Pair-conditional view.** Pilots are filtered by *both* input
token $v$ and successor token $w$. The pair-conditional ensemble
$E_{v,w}$ is the strict intersection of the forward and reverse
views for the corresponding $v$ and $w$. By construction, $E_{v,w}$
has zero variance at $t = 0$ (every trajectory starts at $E(v)$),
and tends toward low variance at $t = L$ (every trajectory must end
in the unembedding region that maps to $w$). The pair-conditional
view filters the trajectory ensemble to a specific bigram context.

The multi-view program enumerated five tests of the conditional
structure of the bundle. Tests 1 and 2 examined the marginal
geometry within fixed cells; Tests 3, 4, and 5 examined the
conditional drift, the per-pilot action under the SDE, and the
Onsager–Machlup bridges between fixed endpoints. The present study
is **modified Test 3**: it examines the structure of the conditional
mean as a function of cell.

### 1.7 What modified Test 3 specifically asks

The question, in plain terms: *take the average trajectory inside
each conditional cell, subtract the corresponding marginal-flow
prediction, and ask whether the residual structure is organized
along a small dictionary of shared modes across cells*.

Why this is the natural test:

The H_decoupling result established that training adds ~88% of the
per-layer update as context-dependent computation. The forward and
reverse views, in their full form, characterize how this added
computation reshapes the marginal ensemble. But the most concentrated
form of the question — does the network do *similar things* across
different (input, output) contexts, or does each context drive the
residual stream along its own private direction? — is answered by
examining the cell-conditional mean trajectories themselves.

If conditional means deviate from the marginal in a coordinated way
across cells — if a small number of "shared trajectory modes"
account for most of the deviation, with cell-specific magnitudes —
then the network's learned context-conditional computation has a
shared structural backbone. Call this **H_dictionary**.

If conditional means deviate from the marginal in essentially
private directions per cell, with no shared structure, then the
network's learned computation is fully idiosyncratic at the bigram
level. Call this **H_cell_private**.

Both outcomes are reportable; the former is more interpretable and
opens the door to a structural characterization of "what attention
adds" on top of the framework's marginal description.

### 1.8 Where this study sits

To summarize the place of the present study in the broader program:

- **Lines of Thought (Sarfati et al., ICLR 2025)** established the
  marginal framework: a linear-Gaussian SDE describing the
  ensemble-average residual-stream geometry, with all context-
  dependent computation absorbed into the noise term.
- **Phase 1** established cross-seed reproducibility of the
  framework's basis-invariant statistics on a 150M Llama-style
  model, alongside the puzzle of basis-disagreement: each seed
  learns a different basis but identical basis-invariant statistics.
- **The Markovianity result (H_decoupling)** established that the
  framework's noise term is not noise at all but learned
  context-dependent computation, that this computation accounts for
  ~88% of the per-layer update at convergence, and that training
  actively constructs it on a schedule faster than eval loss
  convergence.
- **The present study (modified Test 3)** characterizes the
  *structure* of that learned context-dependent computation by
  examining the cross-cell SVD spectrum of conditional-mean
  deviations from marginal flows, in both the reverse view (output-
  only) and the pair view (input + output).

The headline question for this study: when we look at what the
framework has been hiding as noise, does it have organized structure?
And if so, what kind of structure, on what training timescale?

---

## 2. Conceptual setup

### 2.1 The pair-conditional ensemble and the cell-mean trajectory

Fix two tokens $v$ and $w$. Consider all the pilots in the held-out
data where the pilot's current token is $v$ and the next token is
$w$. This set is the **pair-conditional ensemble** $E_{v,w}$.

For each layer $t$, the *cell-conditional mean trajectory* is

$$\bar x_{v,w}(t) = \frac{1}{|E_{v,w}|} \sum_{k \in E_{v,w}} x_t^{(k)} \in \mathbb{R}^H.$$

This is the average residual-stream state across all pilots that
share the bigram context $(v, w)$. As a function of $t$, it traces
out a sequence of $H$-dimensional vectors — a single "average
trajectory" for the bigram.

The structure of $\bar x_{v,w}(t)$ has interesting boundary properties:

- **At $t = 0$**: every trajectory starts at the embedding of $v$,
  so $\bar x_{v,w}(0) = E(v)$ exactly, regardless of $w$.
- **At $t = L$**: every trajectory ends in a residual-stream region
  that the unembedding maps to high $\Pr[\mathrm{next} = w]$, so
  $\bar x_{v,w}(L)$ lies in the small subspace the unembedding
  selects for $w$.
- **In between**: $\bar x_{v,w}(t)$ traces some path through
  residual-stream space, the precise shape of which depends on what
  the network is doing on that bigram.

A natural question is whether there is a *natural path* between
$E(v)$ and the $w$-region — a single most-likely trajectory through
the middle layers, around which all the pilots in $E_{v,w}$ cluster.
In classical mechanics, the answer would come from a principle of
least action: the path that minimizes some functional $S[x]$ is the
path the system takes, and deviations have a known probability cost.

The Sarfati SDE invites exactly such a least-action formulation,
which we develop next.

### 2.2 Action functional in the overdamped regime

Classical action $\int L\, dt$ presupposes inertia — a second-order
dynamics $m \ddot x = F$ in which kinetic energy and acceleration
are meaningful. The transformer residual stream does *not* have
inertia in this sense. The update rule is first-order: $x_{t+1} =
x_t + f_t(x_t, \mathrm{context})$. The layer output is a
displacement, not an acceleration. There is no quantity playing the
role of mass.

However, dynamics without inertia are not foreign to physics. The
**overdamped Langevin equation** describes systems in a regime of
high friction, where the inertial term $m \ddot x$ is negligible
relative to drag. Newton's equation $m \ddot x = F - \gamma \dot x +
\mathrm{noise}$ collapses to

$$\dot x = b(x, t) + \sigma(t)\, \eta(t),$$

where $b$ is the drift, $\sigma$ is the noise scale, and $\eta$ is
white noise. This regime applies to colloidal particles in fluid,
ions in solution, and gene-expression dynamics. There is no
momentum; velocity is proportional to force rather than acceleration
being so.

The lines-of-thought SDE is exactly of this form. Treating layer
depth $t$ as time, the layer increment $\Delta x_t$ is a velocity
(in continuous-depth units), and the SDE is the model for that
velocity.

In this overdamped regime, the analog of the action functional is
the **Onsager–Machlup functional**:

$$S[x] = \frac{1}{2} \int_0^T \|\dot x - b(x, t)\|^2_{\Sigma^{-1}}\, dt,$$

where the norm is taken in the metric set by the inverse noise
covariance $\Sigma^{-1}$. Up to a curvature correction that is small
and constant for our purposes, this is the negative log of the
probability density of the path $x(\cdot)$ under the SDE. Paths that
closely follow the drift accumulate low action; paths that depart
from it have high action.

For the pair-conditional question, the Onsager–Machlup functional
gives a precise object to compare against. The empirical conditional
mean $\bar x_{v,w}(t)$ is one candidate for the "natural" path
between $E(v)$ and the $w$-region; the Onsager–Machlup minimizer
with the same endpoints is another. If they agree, the SDE generates
the conditional bundle correctly. If they disagree, the SDE fails on
conditional structure even though it fits the marginal.

### 2.3 The Markovianity precondition

The Onsager–Machlup formulation rests on a precondition: the drift
$b(x, t)$ must be the correct conditional expectation of $\dot x$
given $x$. In stochastic-process language, the dynamics must be
**Markovian in $x_t$**: the distribution of $x_{t+dt}$ given the
entire history $\{x_s : s \leq t\}$ must depend only on the current
state $x_t$.

If Markovianity fails — if two pilots that happen to be at the same
$x_t$ receive systematically different $\Delta x_t$ because of
hidden context — then the SDE is a marginal fit, not a generative
description, and the action functional computes something other
than "negative log path density under the SDE."

The Markovianity test (companion document; §1.5 of the present
report) established that this precondition *fails* in our trained
model. Only ~12% of $\Delta x_t$ variance is a function of $x_t$;
the other 88% depends on hidden context. The SDE is therefore not
a generative model of trajectories, and the Onsager–Machlup action
under the marginal drift computes "Mahalanobis-weighted attention
work along the trajectory" rather than "negative log path density."

This shifts what modified Test 3 is for. Originally, in the
multi-view proposal, Test 3 was specified as comparing the empirical
$\bar x_{v,w}(t)$ to the Onsager–Machlup bridge (the most-likely
path between $E(v)$ and the $w$-region under the SDE). With the
SDE-bridge interpretation falsified, the test becomes the more direct
question: *what shape does the conditional-mean deviation from
marginal flow have, and is it organized along shared modes across
cells?* No SDE-bridge computation is needed; we use empirical
marginals as the baseline.

### 2.4 The deviation tensor

The central object of modified Test 3 is the **deviation tensor**.
For each cell $c$ in a chosen partition (reverse cells indexed by
successor token, pair cells indexed by bigram), and each layer
$t \in \{0, 1, \ldots, L\}$, define

$$d_c(t) = \bar x_c(t) - \mu_c(t),$$

where $\mu_c(t)$ is the appropriate marginal mean for cell $c$:

- **Reverse view**: $\mu(t) = \frac{1}{N} \sum_k x_t^{(k)}$, the
  all-pilots mean at layer $t$. Every reverse-mode cell uses the
  same global marginal.
- **Pair view**: $\mu_{v_c}(t) = \frac{1}{n_{v_c}} \sum_{k: v_k = v_c} x_t^{(k)}$,
  the input-marginal mean for the input token $v_c$ defining cell
  $c$. Each pair-mode cell uses the marginal for its own input
  token.

Stacking these vectors across cells, layers, and hidden dimensions
gives the **deviation tensor**:

$$D \in \mathbb{R}^{|C| \times (L+1) \times H},$$

where $|C|$ is the number of cells. The entries of $D$ are
basis-dependent (they live in the residual stream's coordinates),
but the singular value spectra of various flattenings of $D$ are
basis-invariant, in the same sense as the Phase 1 statistics.

### 2.5 The two flattenings: direction view vs trajectory view

We extract two complementary basis-invariant spectra from $D$:

**Direction view.** Reshape $D$ to $(|C| \cdot (L+1)) \times H$:
each row is a single (cell, layer) deviation vector in
$\mathbb{R}^H$. The SVD of this matrix reveals shared *directions*
in residual-stream space across (cell, layer) pairs. A small number
of dominant directions would mean the network drives trajectories
along a small dictionary of shared residual-stream axes,
irrespective of layer or cell.

**Trajectory view.** Reshape $D$ to $|C| \times ((L+1) \cdot H)$:
each row is one cell's full layer-stack-of-deviations, viewed as a
single high-dimensional vector. The SVD of this matrix reveals
shared *trajectory shapes* across cells. A small number of dominant
modes would mean different cells share common trajectory templates
(e.g., a common "rises monotonically toward the unembedding"
template, with cell-specific final magnitudes).

Both views answer the multi-view question from different angles.
The trajectory view is more informative for the dictionary
question: $H \approx 900$ but $|C| \leq 35$ in our setup, so the
trajectory view has at most 35 non-trivial singular values, making
it a tractable object. The direction view has up to $|C|(L+1) \approx
500$ singular values and is useful as a secondary check.

### 2.6 The shuffle null

To establish what spectrum the deviation tensor would have *if* the
cell labels carried no information, we compute a **shuffle null**:
permute the cell labels of pilots and recompute $D$. The shuffle
null measures the floor that finite-sample statistical fluctuations
of cell means would produce even on data with no cell-specific
structure.

The shuffle null must be mode-specific:

- **Reverse view**: a global shuffle. Permute the cell-label
  assignment of all pilots uniformly at random. The reverse-view
  marginal subtraction uses the global $\mu(t)$, which is invariant
  under permutation, so this is well-defined.

- **Pair view**: a *within-input* shuffle. Permute cell-label
  assignments only among pilots that share the same input token $v$.
  This preserves the property that each pair-cell's pilots all have
  input $v$, so the input-marginal subtraction remains valid. A
  naive global shuffle in pair mode would assign pilots to cells
  regardless of input token, breaking the input-marginal subtraction
  and producing a spurious "above-null" baseline.

The choice of within-input shuffle in pair mode is load-bearing.
Section §6.3 documents the synthetic-data discovery that a global
shuffle in pair mode falsely signals strong dictionary structure on
data with no bigram-specific signal.

### 2.7 The energy concentration statistic

We initially tried to summarize the deviation tensor's spectrum
using entropy-based **effective rank**:

$$\mathrm{ER}(\sigma) = \exp\!\left( -\sum_i p_i \log p_i \right), \qquad p_i = \sigma_i^2 / \sum_j \sigma_j^2.$$

Effective rank measures spectrum *uniformity*: ER = 1 if all energy
is in a single mode, ER = $r$ if $r$ modes have equal energy.

Effective rank turned out to be the *wrong* summary statistic for
the dictionary question. The reason: ER conflates "spectrum
concentrated at the head" with "spectrum uniformly small everywhere."
A spectrum with three large leading modes plus a long tail of small
modes can have *higher* ER than a uniformly noisy null spectrum,
because the entropy of the long tail dominates the ER calculation.
This led to misleading verdicts in our first runs (documented in §9).

The correct summary statistic for the dictionary question is
**cumulative energy concentration**:

$$E_k(\sigma) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_i \sigma_i^2}.$$

$E_k$ is monotonic from 0 to 1 and tells us what fraction of total
deviation magnitude lives in the leading $k$ modes. The dictionary
question becomes:

- For some small $k$, does $E_k^{\mathrm{real}}$ exceed
  $E_k^{\mathrm{null}}$ by an amount that is statistically
  significant and substantively large?
- If yes, at what $k$ does the excess concentration *peak*? Past
  that $k$, additional modes are no longer enriched over null; they
  belong to the cell-specific tail rather than the shared dictionary.

We call the $k$ at which $E_k^{\mathrm{real}} - E_k^{\mathrm{null}}$
peaks the **dictionary dimension**. This statistic:

- Is robust to long tails (unlike ER).
- Has a natural interpretation: "the smallest number of shared modes
  that captures all the excess concentration over null."
- Generalizes correctly to no-signal regimes (in which the peak is
  at the right tail of the spectrum, dictionary dimension is large,
  and peak excess is near zero).

### 2.8 Why empirical marginals rather than SDE-derived ones

The original multi-view proposal specified $\hat \mu_v(t)$, the
SDE-derived linear-flow extrapolation from $E(v)$ at $t = 0$, as
the baseline against which to compare $\bar x_{v,w}(t)$. We instead
use the empirical marginal mean (global for reverse; input-marginal
for pair). Reasons:

The Markovianity result established that the SDE drift only explains
~12% of $\Delta x_t$. Using the SDE-derived $\hat\mu_v$ as the
baseline would conflate "deviation from the conditional mean" with
"SDE fit error" — most of the apparent deviation would just be the
SDE missing the bulk of the per-layer dynamics, swamping the
genuine cell-specific signal.

The empirical marginal subtracts exactly what's predictable from
the conditioning variable alone, without invoking any model. In the
reverse view, $\mu(t)$ is the average trajectory of a randomly chosen
pilot; subtracting it isolates the per-output contribution. In the
pair view, $\mu_v(t)$ is the average trajectory of a pilot with
input $v$; subtracting it isolates the bigram-specific contribution
beyond what input alone determines. These are the right baselines
for asking what conditioning *adds*.

---

## 3. Hypotheses

We pre-registered four hypotheses about the deviation tensor's
cross-cell structure, plus two about its training trajectory.

### 3.1 Structural hypotheses

The four structural hypotheses are mutually exclusive at the
trajectory-view spectrum level. Each has explicit operational
criteria:

**H_dictionary.** The deviation tensor has effective rank
substantially below the shuffle null, and the peak cumulative
energy concentration $E_{k^*}^{\mathrm{real}} - E_{k^*}^{\mathrm{null}}$
exceeds 10% with significance >5σ, with dictionary dimension $k^*$
well below the cell count ($k^* \leq 0.3 |C|$). The learned function
uses a small dictionary of shared trajectory modes that cells reuse
with cell-specific magnitudes. **PASS criterion: peak excess > 10%,
peak-z > 5, $k^* / |C| \leq 0.3$**.

**H_partial_dictionary.** Peak excess concentration is between 3%
and 10% above null, significance > 3σ. Real shared structure
across cells but with substantial cell-specific tail. **PASS
criterion: peak excess > 3%, peak-z > 3, but failing H_dictionary's
criteria.**

**H_full_rank (or H_cell_private).** The deviation tensor has
effective rank at or above null, with no excess concentration at
any $k$. Each cell drives the residual stream along its own private
direction. **PASS criterion: real effective rank > null effective
rank by > 3σ, and no $k$ with positive excess concentration > 1%.**

**H_no_signal.** Real and null are statistically indistinguishable.
The conditioning token contributes no structure beyond what the
marginal already determines. **PASS criterion: peak excess < 1%,
effective rank within 2σ of null at all $k$.**

The intermediate case ("weak dictionary": peak excess 0.5–3%) we
label H_weak_dictionary; it is reported but not treated as a
positive structural finding.

### 3.2 Dynamical hypotheses

For the training-trajectory experiments, two hypotheses about the
shape of the peak-excess-vs-training-step curve:

**H_emergence.** Peak excess concentration rises monotonically (or
near-monotonically) through training, starting near zero at
initialization and asymptoting at a learned level. The dictionary
is *constructed by training*. **PASS criterion: peak excess at step
24000 is at least 4× the peak excess at step 100, with monotonic
(allowing small noise oscillations) increase across the
rapid-change window of step ~200 to ~5000.**

**H_architectural.** Peak excess concentration is flat across
training, near its final value from step 100 onward. The dictionary
is *architectural*, built into the embedding and unembedding
geometry irrespective of training. **PASS criterion: peak excess
varies by less than 30% across all checkpoints from step 100 to
step 24000.**

If both H_emergence and H_architectural fail, the trajectory shape
is non-trivial and requires interpretation (the actual outcome in
our results, anticipated as a possibility but not pre-registered as
its own hypothesis).

### 3.3 Cross-seed reproducibility

Treated as a precondition rather than a separate hypothesis: any
structural finding that does not reproduce across the four trained
seeds is dismissed. We require:

- **Peak excess agreement**: cross-seed std of peak excess at the
  final checkpoint is less than 25% of the mean (matching the
  Phase 1 cross-seed agreement on basis-invariant statistics).
- **Dictionary dimension agreement**: dictionary dimension agrees
  to within ±2 across seeds at the final checkpoint.
- **Trajectory-shape agreement**: at every checkpoint in the
  training trajectory, cross-seed std of peak excess is less than
  50% of the cross-seed mean (loosened from final-checkpoint
  agreement because per-checkpoint estimator noise is larger when
  $n_{\mathrm{shuffles}} = 20$ at all checkpoints).

If a hypothesis appears to PASS in one or two seeds but not the
others, we treat the inconsistency as evidence against the finding
rather than as a per-seed property.

### 3.4 Symmetry of the two views

We do *not* pre-register hypotheses about how the pair-mode and
reverse-mode results should compare. The comparison itself is
empirical — we treat the two modes as independent measurements of
related but distinct quantities and use their difference as a
substantive finding. A priori, three outcomes are plausible:

- *Both pass H_dictionary at similar magnitudes.* Bigram and
  reverse-only views see the same dictionary. The dictionary
  describes a fully cell-conditional structure that is not
  separable into input and output components.
- *Reverse passes H_dictionary much more strongly than pair.* The
  apparent dictionary in reverse mode is mostly input-marginal
  structure exposed through the output filter; subtracting the
  input-marginal removes most of the signal.
- *Pair passes H_dictionary much more strongly than reverse.* The
  bigram-conditioned structure is the real dictionary; reverse mode
  averages over too many input-output combinations to see it.

Each of these would have different implications for how the
network's learned function is organized. We report the actual ratio
as the substantive contrast.

---

## 4. Experimental setup

This section specifies the experimental apparatus to a level of detail
sufficient for someone to reproduce the work without consulting prior
documents in the series. The model, the training recipe, the
checkpoint inventory, the pilot collection protocol, and the cell
selection thresholds are all documented in full.

### 4.1 Architecture

A single 150M-parameter decoder-only transformer in the Llama family,
with one deliberate departure from the Llama defaults: the FFN
activation is GELU rather than SwiGLU. (This choice was made for the
broader project series to enable direct comparison with the GPT-2
medium model used in the lines-of-thought paper, which also uses
GELU. The choice is not load-bearing for the present study; the same
test could be run on a SwiGLU model and would be expected to give
qualitatively similar results.)

Architectural details:

| Property | Value |
|---|---|
| Architecture family | Llama-style decoder-only transformer |
| Hidden size $H$ | 896 |
| Number of transformer blocks $L$ | 12 |
| Total observed states per pilot | $L_\mathrm{total} = 14$ (post-embedding + 12 block outputs + post-final-norm) |
| Attention heads | 14 (head dim 64) |
| Key-value heads | 2 (grouped-query attention, GQA) |
| FFN intermediate dim | 2432 |
| FFN activation | GELU (deliberate departure from Llama-default SwiGLU) |
| Position encoding | RoPE (base 10000) |
| Normalization | RMSNorm (pre-norm) |
| Embeddings | Tied input and output embeddings |
| Vocabulary size | 32768 |
| Tokenizer | Mistral-7B-v0.1 BPE tokenizer |
| Total parameter count | ~150M |

The "$L_\mathrm{total} = 14$ states" deserves comment. The lines-of-
thought paper indexes only the $L$ block outputs as residual-stream
states. We index more aggressively: the pre-block-1 post-embedding
state (which the paper omits), each of the 12 block outputs, and the
post-final-norm state (which the paper also omits). This gives
$L_\mathrm{total} = 14$ residual-stream states per pilot. The Phase 1
work found a striking anomaly at the post-final-norm state — its
basis-invariant statistics behave unlike interior states — so we
keep it explicit rather than collapsing it into the unembedding.

For the present study, layer-by-layer structure of the deviation
tensor is reported at all 14 states. The post-final-norm state plays
an essential role in the prediction-commitment story; the
post-embedding state plays an essential role in the input-anchoring
story; we keep both.

### 4.2 Training recipe

Identical to the Phase 1 and Markovianity work:

| Property | Value |
|---|---|
| Training corpus | FineWeb-Edu sample-10BT subset |
| Tokenizer | Mistral-7B-v0.1 (same as model) |
| Total training tokens | 1.57B |
| Training steps | 24000 |
| Batch size | 64 sequences |
| Sequence length | 1024 |
| Tokens per step | 65,536 |
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$) |
| Peak learning rate | $3 \times 10^{-4}$ |
| LR schedule | Cosine decay to 10% of peak, no warmup |
| Weight decay | 0.1 |
| Dropout | 0 |
| Gradient clipping | 1.0 |

All training was done from scratch (no pretrained initialization).
Each of the four seeds uses identical hyperparameters but a
different RNG seed for parameter initialization and for data-shuffle
order. The four seeds are nominally 0, 1, 2, 3.

### 4.3 Checkpoints

Per seed, 50 checkpoints were saved at log-spaced training-step
intervals between step 100 and step 24000. For the final-checkpoint
analysis (§7), only step 24000 is used. For the training-trajectory
analysis (§8), we sub-sample 12 of the 50 saved checkpoints at
log-spaced intervals using the helper `log_spaced_subsample` that
the Markovianity work also uses. The 12 selected checkpoints are
identical across seeds (selection is on log-step value, which is
seed-independent), so cross-seed comparisons at any given step are
strictly apples-to-apples.

The selected 12 steps:

```
100, 156, 274, 428, 749, 1171, 2049, 3205, 5607, 8771, 15343, 24000
```

These steps roughly evenly partition $\log(\mathrm{step})$ from
$\log 100 \approx 2$ to $\log 24000 \approx 4.4$.

### 4.4 Pilot collection

A *pilot* is a single (token, position, context) triple in held-out
text. For each pilot, the model's per-layer residual-stream state is
recorded at all $L_\mathrm{total} = 14$ states, along with metadata:
the input token id, the next token id (the network's prediction
target), the predicted token id (the network's argmax at that
position), and the position within the chunk.

Pilots are collected by running each saved model checkpoint in
inference mode on a held-out evaluation set of 500 chunks of 1024
tokens each. From this, $N = 10{,}000$ pilots are sampled per
(seed, checkpoint) pair, with pilot positions distributed uniformly
across non-boundary positions in the chunks. The result is a
$(L_\mathrm{total}, N, H) = (14, 10{,}000, 896)$ activation tensor
per (seed, checkpoint).

The pilot file format is one .npz per (seed, checkpoint):
`phase1_runs_gelu/multiview/seed_{S}/augmented_step_{STEP:08d}.npz`
containing the activation tensor and the per-pilot metadata. Total
storage: ~3 GB per (seed, checkpoint) × 50 checkpoints × 4 seeds =
~600 GB. (The activation files are the heaviest artifacts in the
project; they exist on disk and are not regenerated for this study.)

### 4.5 The data sparsity problem

The bigram density in our held-out evaluation set is the key
quantity that determines whether pair-mode cell counts are
sufficient. We document this carefully because it drives several
design choices.

With $N = 10{,}000$ pilots drawn from chunks of 1024 tokens (so
~10 chunks of pilots) and a Mistral-tokenizer vocabulary of 32k, the
expected number of pilots per (input, output) bigram is small. In our
data:

- **Distinct input tokens observed**: 3477.
- **Distinct successor tokens observed**: 3506.
- **Distinct bigrams observed**: about 8500 (most appearing only once).
- **Bigrams with $\geq 30$ pilots**: ~7.
- **Bigrams with $\geq 15$ pilots**: ~15.
- **Bigrams with $\geq 10$ pilots**: ~25.

By contrast, single-token statistics are much denser:

- **Successor tokens with $\geq 30$ pilots**: ~35.
- **Successor tokens with $\geq 15$ pilots**: ~80.

The factor of ~5 between single-token and bigram density at the same
sample-size threshold determines that:

- **Reverse mode** can use a higher per-cell sample threshold and
  still get ~35 cells, giving good per-cell mean estimates with
  median ~57 pilots.
- **Pair mode** must use a lower sample threshold to retain any
  cells at all, and even then gives only ~15 cells with median 28
  pilots per cell.

This is the central methodological limitation of pair mode in our
setup. We could mitigate by collecting more pilots, but the existing
collection is a fixed resource for this study.

### 4.6 Cell selection algorithm

The cell selection procedure is identical for both modes up to the
choice of cell identity:

1. **Enumerate candidate cells** by tabulating pilot membership.
   For reverse mode, group pilots by `next_id`. For pair mode, group
   by (`input_id`, `next_id`).
2. **Filter by minimum count**: keep only cells with at least
   $\min\_pilots\_per\_cell$ pilots.
3. **Optional input-frequency filter** (pair mode only): if
   $\mathtt{top\_k\_tokens\_v} > 0$, restrict to cells whose input
   token is among the top-$K$ most-frequent input tokens. With
   $\mathtt{top\_k\_tokens\_v} = 0$ (our default for the production
   runs), this filter is disabled.
4. **Limit cell count**: keep at most $\mathtt{top\_k\_cells}$ cells
   by descending pilot count.

Default thresholds:

| Threshold | Reverse mode | Pair mode |
|---|---|---|
| `min_pilots_per_cell` | 30 | 15 |
| `top_k_cells` | 100 | 100 |
| `top_k_tokens_v` | 0 (n/a) | 0 (disabled) |

At these settings, our held-out data gives:

- **Reverse mode, step 24000, all seeds**: 35 cells per seed; pilot
  counts min 34, median 57, max 373; total 3430 assigned pilots
  (34.3% of $N$).
- **Pair mode, step 24000, all seeds**: 15 cells per seed; pilot
  counts min 16, median 28, max 102; total 507 assigned pilots
  (5.1% of $N$); covering 11 distinct input tokens.

A few useful observations on cross-seed identity:

- The set of selected reverse-mode cells (i.e., the 35 successor
  tokens) is **identical across all four seeds at every checkpoint**.
  Cell selection is based on pilot counts of token ids in the
  held-out data, which is the same data for all seeds.
- The pair-mode cells are also identical across seeds for the same
  reason.
- The per-cell pilot identities (which specific pilots are in each
  cell) are also identical across seeds, because pilots are sampled
  deterministically from the same held-out chunks.
- What differs across seeds is the activation values at those
  identical pilots, because each seed has different learned weights.

This is important for the basis-invariance argument: when we compare
spectra across seeds, we are comparing spectra of cell × layer × $H$
deviation tensors where the cell identities, the pilot memberships,
and the held-out positions are all identical; only the activations
differ. Differences in the spectra reflect differences in what each
seed has learned, not differences in what is being measured.

### 4.7 Marginal computation

**Reverse mode (global marginal)**:

$$\mu(t) = \frac{1}{N} \sum_{k=1}^N x_t^{(k)}.$$

This is the average residual-stream state across all $N = 10{,}000$
pilots at layer $t$. It uses all pilots, not just those assigned to
cells. Computed once per (seed, checkpoint), shared across all
reverse-mode cells.

**Pair mode (input-marginal)**:

$$\mu_v(t) = \frac{1}{n_v} \sum_{k: v_k = v} x_t^{(k)},$$

where $n_v$ is the number of pilots with input token $v$. Computed
for every distinct input token that appears in some selected cell
(11 distinct input tokens in our setup, so 11 marginals are
computed). Each pair-mode cell subtracts the marginal corresponding
to its own input token.

Implementation note: the pilots in a given cell *also* contribute to
the marginal that's subtracted from that cell's mean. This induces a
small "leakage" — a cell $(v, w)$ with $n_{v,w}$ pilots contributes
a fraction $n_{v,w}/n_v$ to its own input marginal. In our data, the
median fraction is ~5%; leakage is small but not zero. A
leave-one-out version of the marginal would eliminate this; we did
not implement it, but discuss the implications in §12.

### 4.8 Computational cost

A complete summary of the wall-clock cost of every reported run, on
a single workstation (Ryzen 9 CPU; the work is dominated by SVD and
matrix mean computations, not GPU-bound):

Per (seed, checkpoint) at default settings ($\mathtt{n\_shuffles} = 20$,
$N = 10{,}000$):

| Component | Reverse mode | Pair mode |
|---|---|---|
| Load augmented npz | < 1 sec | < 1 sec |
| Compute marginal(s) | ~2 sec | ~2 sec |
| Compute per-cell means | ~2 sec | ~1 sec |
| Compute deviation tensor | ~1 sec | ~0.5 sec |
| Direction-view SVD ($\sim 500 \times 896$) | ~5 sec | ~3 sec |
| Trajectory-view SVD ($\sim 35 \times 12{,}544$) | ~2 sec | ~1 sec |
| 20 shuffle replicates (mean + SVDs) | ~3 min | ~30 sec |
| Plot + save | ~1 sec | ~1 sec |
| **Total per (seed, checkpoint)** | **~3.5 min** | **~1 min** |

Aggregated for the runs reported in this document:

| Experiment | Wall-clock cost |
|---|---|
| Reverse mode, 4 seeds, final checkpoint only (§7.1) | ~14 min |
| Pair mode, 4 seeds, final checkpoint only (§7.2) | ~4 min |
| Reverse mode training trajectory (4 seeds × 12 ckpts, §8.1) | ~2.8 hours |
| Pair mode training trajectory (4 seeds × 12 ckpts, §8.2) | ~50 min |
| Synthetic validation (§6) | ~5 min total |
| **Grand total compute** | **~4 hours** |

Storage produced by these runs:

| Output | Approximate size |
|---|---|
| Per-checkpoint npz files (4 seeds × ~24 single-checkpoint runs) | ~250 MB total |
| Trajectory aggregated npz files (2 modes) | ~50 MB total |
| Per-checkpoint PNG plots | ~30 MB total |

This is negligible relative to the 600 GB of activation files the
study reads from.

---

## 5. Test specification

This section gives the test as a precise algorithm. A reader who
implements this in their own code from §5 alone, ignoring the rest
of the document, should be able to reproduce our results to within
numerical precision.

### 5.1 Inputs

For each (seed $s$, training step $T$) pair under test, the input is
the augmented activation file

```
augmented_step_{T:08d}.npz
```

containing:

- `states`: array of shape $(L_\mathrm{total}, N, H) = (14, 10000, 896)$,
  the per-pilot residual-stream states at every layer.
- `input_ids`: array of shape $(N,)$, integer token ids at the pilot
  position.
- `next_ids`: array of shape $(N,)$, integer token ids at position
  (pilot + 1) — the prediction target.
- `pred_ids`: array of shape $(N,)$, the model's argmax prediction
  at the pilot position (unused in this test).
- `positions`: array of shape $(N,)$, the position-within-chunk
  of each pilot (unused in this test).

Plus the configuration parameters:

- `mode` $\in \{$pair, reverse$\}$
- `min_pilots_per_cell`
- `top_k_cells`
- `top_k_tokens_v` (pair mode only)
- `n_shuffles`

### 5.2 Step 1: cell selection

**Reverse mode**: tabulate the histogram of `next_ids`. Restrict to
successor tokens with at least `min_pilots_per_cell` pilots. Sort by
descending count and keep at most `top_k_cells`.

For each retained successor token $w_c$, the cell $c$ is the set of
pilots $\{k : \mathrm{next\_id}_k = w_c\}$. Record:

- `cells[c, 0] = -1` (sentinel for "no input conditioning")
- `cells[c, 1] = w_c`
- `cell_counts[c]` = number of pilots in cell

**Pair mode**: tabulate the histogram of (`input_id`, `next_id`) pairs.
Restrict to pairs with at least `min_pilots_per_cell` pilots, then
(if `top_k_tokens_v > 0`) restrict further to pairs whose input
token is among the top-$K$ most-frequent input tokens. Sort
remaining by descending count and keep at most `top_k_cells`.

For each retained pair $(v_c, w_c)$, the cell $c$ is the set of
pilots $\{k : \mathrm{input\_id}_k = v_c\text{ and }\mathrm{next\_id}_k = w_c\}$.
Record:

- `cells[c, 0] = v_c`
- `cells[c, 1] = w_c`
- `cell_counts[c]`

In both modes, build the pilot-to-cell assignment array
`pilot_assignments[k]` $\in \{-1, 0, \ldots, |C| - 1\}$, where $-1$
means "not in any selected cell."

### 5.3 Step 2: compute marginals

**Reverse mode**:

$$\mu(t) = \frac{1}{N} \sum_{k=1}^N \mathrm{states}[t, k, :].$$

`global_marginal` has shape $(L_\mathrm{total}, H)$.

**Pair mode**: for each distinct input token $v$ appearing in some
selected cell,

$$\mu_v(t) = \frac{1}{|\{k : \mathrm{input\_id}_k = v\}|} \sum_{k : \mathrm{input\_id}_k = v} \mathrm{states}[t, k, :].$$

`input_marginal_means[v]` has shape $(L_\mathrm{total}, H)$. Store
as a dict keyed by input token id.

### 5.4 Step 3: compute per-cell means

For each cell $c$, average the states of pilots assigned to $c$:

$$\bar x_c(t) = \frac{1}{n_c} \sum_{k : \mathrm{pilot\_assignments}_k = c} \mathrm{states}[t, k, :].$$

`cell_means` has shape $(|C|, L_\mathrm{total}, H)$.

### 5.5 Step 4: compute deviation tensor

For each cell $c$, subtract the appropriate marginal:

- Reverse mode: $d_c(t) = \bar x_c(t) - \mu(t)$.
- Pair mode: $d_{v_c, w_c}(t) = \bar x_{v_c, w_c}(t) - \mu_{v_c}(t)$.

`deviation` has shape $(|C|, L_\mathrm{total}, H)$.

### 5.6 Step 5: compute spectra (direction view and trajectory view)

Reshape `deviation` to the two flattenings and compute their SVD
singular values.

**Direction view**:

```
mat_dir = deviation.reshape(|C| * L_total, H)
sv_directions = numpy.linalg.svd(mat_dir, compute_uv=False)
```

Shape: $(\min(|C| \cdot L_\mathrm{total}, H),)$.

**Trajectory view**:

```
mat_traj = deviation.reshape(|C|, L_total * H)
sv_trajectories = numpy.linalg.svd(mat_traj, compute_uv=False)
```

Shape: $(\min(|C|, L_\mathrm{total} \cdot H),) = (|C|,)$ in practice.

### 5.7 Step 6: compute effective rank and energy concentration

For each spectrum $\sigma$ (descending singular values), compute:

**Effective rank**:

$$\mathrm{ER}(\sigma) = \exp\!\left( -\sum_i p_i \log p_i \right), \qquad p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}.$$

**Cumulative energy concentration**:

$$E_k(\sigma) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_i \sigma_i^2}, \qquad k = 1, \ldots, |\sigma|.$$

Both are computed for both flattenings.

### 5.8 Step 7: shuffle null

Generate `n_shuffles` (= 20) replicates of the deviation tensor with
permuted cell labels. The permutation is mode-specific:

**Reverse mode (global shuffle)**:

For each replicate, sample a random permutation $\pi$ of the assigned
pilots, then assign `pilot_assignments_shuffled[k] =
pilot_assignments[π(k)]`. The cell membership counts are preserved
exactly (random reshuffling of which pilot goes to which cell, but
the total number per cell is invariant).

**Pair mode (within-input shuffle)**:

For each input token $v$ separately, sample a random permutation
$\pi_v$ of the assigned pilots whose input is $v$, and reshuffle
their cell labels among themselves. This preserves the property that
each cell's pilots all have input $v$, so the input-marginal
subtraction remains valid.

For each replicate, recompute step 3 (per-cell means) and steps 4–6
(deviation, spectra, effective rank, energy concentration). Stack
the per-replicate values:

```
null_sv_directions_mean = mean(sv_directions, axis=0)  # over replicates
null_sv_directions_std = std(sv_directions, axis=0)
null_energy_trajectories_mean = mean(energy_trajectories, axis=0)
null_energy_trajectories_std = std(energy_trajectories, axis=0)
null_effective_rank_trajectories_mean = mean(effective_rank, axis=0)
... etc
```

(In our implementation, the shuffle exploits a shortcut: rather than
recomputing the deviation tensor as `cell_mean - marginal` from
scratch each time, we precompute per-pilot deviations
`pilot_dev[t, k, :] = states[t, k, :] - marginal[t, :]`, where the
marginal is mode-specific. The shuffled per-cell deviation is then
just the cell-mean of these per-pilot deviations under the shuffled
assignment. This is mathematically equivalent and saves a factor of
~3 in shuffle wall-clock time.)

### 5.9 Step 8: compute dictionary dimension and verdict

The **dictionary dimension** is the smallest $k$ at which the excess
concentration $E_k^{\mathrm{real}} - E_k^{\mathrm{null}}$ peaks
(i.e., $\arg\max_k\,\mathrm{excess}(k)$). The dictionary dimension
$k^*$ is used as both:

- The single-number summary of "how many shared modes the dictionary
  has."
- The location at which we report the headline excess concentration.

The **peak excess** is
$E^* = E_{k^*}^{\mathrm{real}} - E_{k^*}^{\mathrm{null}}$.

The **peak z-score** is
$z^* = E^* / \sigma_{k^*}^{\mathrm{null}}$, where
$\sigma_{k^*}^{\mathrm{null}}$ is the cross-replicate std of
$E_{k^*}^{\mathrm{null}}$.

The verdict logic (applied to the trajectory view):

- $E^* > 10\%$ and $z^* > 5$ and $k^* / |C| \leq 0.3$: **H_dictionary PASS**.
- $E^* > 3\%$ and $z^* > 3$: **H_partial_dictionary**.
- $E^* > 0.5\%$: **H_weak_dictionary**.
- otherwise: **H_no_signal**.

The verdict is included as a string in the output for inspection;
the underlying numerical statistics (dictionary dimension, peak
excess, peak z-score, full energy concentration curves) are the
primary outputs.

### 5.10 Step 9: outputs

Per (seed, checkpoint, mode), write:

- `{mode}_seed_{S}_step_{STEP:08d}.npz` with the deviation tensor,
  cell identities, cell counts, all spectra, all energy curves,
  null statistics, verdict, dictionary dimension, peak excess,
  peak z-score, and a config block recording the parameters used.
- `{mode}_seed_{S}_step_{STEP:08d}.png` with a three-panel
  visualization: direction-view spectrum (top), trajectory-view
  spectrum (middle), trajectory-view cumulative energy concentration
  (bottom) with the dictionary dimension marked.

The script also has a multi-checkpoint training-trajectory driver
that iterates this whole pipeline across seeds × checkpoints and
emits a separate aggregated trajectory npz and a 3-panel
trajectory plot.

---

## 6. Synthetic validation

Before applying the test to the trained model's activations, we
validated it on synthetic data with known ground-truth structure.
This serves two purposes:

1. **Discrimination check.** Confirm that the test correctly
   identifies each of {dictionary, partial dictionary, full rank,
   no signal} regimes when the regime is built into the data by
   construction.
2. **Null-distribution check.** Confirm that the mode-specific
   shuffle null produces the correct baseline; in particular, that
   the within-input shuffle in pair mode is necessary and
   sufficient.

### 6.1 The four synthetic regimes

We generated synthetic activation tensors with the same shape as the
real data but with controlled deviation structure. Common parameters:

- $L = 14$ layers, $H = 64$ hidden dim (small for speed), $N = 5000$
  pilots.
- 20 distinct successor tokens, each appearing in 250 pilots.
- A common global marginal $\mu_{\mathrm{global}}(t)$ drawn from
  $\mathcal{N}(0, 0.3^2)$.
- Per-cell offsets $\beta_w \in \mathbb{R}^{L \times H}$ varying by
  regime (described below).
- Per-pilot residual $\epsilon \sim \mathcal{N}(0, 0.5^2)$.

For each pilot $k$ with successor $w$,

$$x_t^{(k)} = \mu_{\mathrm{global}}(t) + \beta_w(t) + \epsilon_k(t).$$

The four regimes differ in how $\beta_w$ is constructed:

**low_rank**: $\beta_w = \sum_{d=1}^{n_{\mathrm{share}}} c_{w,d}\,
\mathrm{dir}_d$, where $\mathrm{dir}_d$ are $n_{\mathrm{share}} = 3$
random unit vectors in $\mathbb{R}^{L \times H}$ (shared across all
cells), $c_{w,d}$ are i.i.d. Gaussian coefficients. Scaled by 1.5.
Pure $n_{\mathrm{share}}$-dimensional shared dictionary.

**long_tail**: $\beta_w = $ low_rank base + $0.2 \cdot \eta_w$,
where $\eta_w$ is a per-cell random tensor. The dictionary is real
but accompanied by a small cell-specific tail. This is the regime
we expected to see in real data.

**full_rank**: $\beta_w$ is entirely random per cell, with no
shared directions. Every cell drives the residual stream along its
own private direction.

**no_structure**: $\beta_w = 0$ for all $w$. Pure noise after
marginal subtraction. The null floor of the test.

### 6.2 Reverse-mode results across regimes

Reverse-mode driver, $\mathtt{n\_shuffles} = 10$, on 200 cells of 250
pilots each (synthetic configuration matched approximately to
reverse mode at the final checkpoint):

| Regime | dict dim | peak excess | peak z | verdict |
|---|---:|---:|---:|---|
| low_rank | 3 | +71.3% | 104 | H_dictionary PASS |
| long_tail | 3 | +8.4% | 13 | H_partial_dictionary |
| full_rank | 19 | +0.0% | 0 | H_no_signal |
| no_structure | 18 | +0.1% | <2 | H_no_signal |

Several things to note:

- **low_rank**: dictionary dimension exactly recovers
  $n_{\mathrm{share}} = 3$, peak excess is extreme (71%), verdict
  PASSES at maximum strength.
- **long_tail**: dictionary dimension still recovers 3 (the head-
  dominance statistic is robust to the per-cell tail), but peak
  excess drops to 8% because most deviation energy now lives in
  the tail. This is the regime our real data turns out to lie in.
- **full_rank**: dictionary dimension is pushed to the right tail
  of the spectrum (19, near the cell count of 20) because no $k$
  has positive excess; verdict correctly gives H_no_signal even
  though the per-cell deviations are very large. This was the
  critical test that vindicated the energy concentration statistic
  over effective rank: with effective rank, the full_rank synthetic
  regime gives ER (real) substantially *above* ER (null) — a
  positive gap that the previous verdict logic would have called
  H_full_rank PASS, but that has no shared structure at all.
  Energy concentration correctly returns H_no_signal because no
  $k$ has positive excess.
- **no_structure**: dictionary dimension is large, peak excess is
  at the noise-floor level (0.1%), verdict correctly gives
  H_no_signal.

The 71% peak excess in the pure low_rank regime gives a sense of
the upper end of the scale: a near-perfect 3-dimensional dictionary
with shape-shared, magnitude-cell-specific deviations would produce
peak excess in the 60–80% range. Our real-data peak excess of ~8%
in reverse mode places the trained model firmly in the long_tail
regime, with a real but partial dictionary.

### 6.3 Pair-mode null validation (within-input shuffle is required)

For the pair-mode null choice, we ran a separate synthetic
experiment with bigram structure of two types:

- **Input-only structure**: $\beta_{v,w} = \alpha_v$ depends only on
  input $v$, not on output $w$. No bigram-specific signal.
- **Bigram-specific structure**: $\beta_{v,w} = \alpha_v +
  \gamma_{v,w}$, with $\gamma_{v,w}$ a low-rank or full-rank
  bigram-conditional contribution.

When the synthetic data has *only* input-marginal structure (i.e.,
$\gamma = 0$), the pair-mode test should give H_no_signal: the
input-marginal subtraction has removed all the structure.

We ran the test in this regime under two different null choices:

| Null | Real $E_3$ | Null $E_3$ | excess | peak z | verdict |
|---|---:|---:|---:|---:|---|
| Within-input shuffle | 18% | 19% | -1% | -2 | H_no_signal ✓ |
| Global shuffle | 18% | 51% | -33% | <-30 | (large negative excess, undefined) |

In the global shuffle, the null spectrum is constructed by randomly
reassigning pilots to cells *regardless of their input token*. This
breaks the input-marginal subtraction: pilots in the shuffled cell
no longer all share the same input, so $\bar x_{c,\mathrm{shuffled}}
- \mu_{v_c}$ becomes a meaningless quantity. The resulting null
spectrum has spuriously high concentration in low-$k$ modes (because
the cell-mean estimator is sampling from a much wider distribution
than the per-input distribution should permit), creating a large
*false positive* signal if interpreted as a dictionary baseline.

In particular, in a more complicated case where there *is* some real
bigram-specific signal but a global shuffle null is used, the false
positive structure from the global shuffle can swamp the genuine
signal, producing either a misleading H_dictionary verdict (if the
sign convention happens to align) or a misleading H_full_rank verdict
(if it doesn't). Within-input shuffle is the unique correct null for
pair mode because it preserves the property the deviation
subtraction depends on.

This finding is what fixed the pair-mode null choice. Without
synthetic validation, the global-shuffle bug would have produced
qualitatively wrong verdicts on the real data — a serious risk we
ran for the first few iterations of the test (documented in §9).

### 6.4 What synthetic validation establishes and what it doesn't

**Establishes**:
- The test correctly discriminates between {dictionary, partial
  dictionary, full-rank cell-private, no signal} regimes on data
  with known ground truth.
- Energy concentration is the correct primary statistic; effective
  rank produces qualitatively wrong verdicts on full-rank data
  (which the present study's pair-mode data turns out to resemble).
- Dictionary dimension correctly recovers the ground-truth number
  of shared directions in pure low-rank regimes; correctly degrades
  in long-tail and full-rank regimes.
- Within-input shuffle is necessary in pair mode.

**Does not establish**:
- Behavior at the small cell counts of pair mode in our data (15
  cells vs the 20 used in synthetic). The qualitative discrimination
  holds; the noise floors may differ.
- Behavior across the training trajectory specifically (synthetic
  was final-checkpoint only).
- Behavior at the high intrinsic dimension of $x_t$ in the real
  model ($H = 896$ with effective rank ~256). Synthetic used $H = 64$
  with full rank, which is a different regime.

These limitations are reasonable for a sanity check before running
on real data. The discriminative behavior of the test holds; the
absolute magnitudes of peak excess in the real data should be
compared to the *real-data null*, not to the synthetic magnitudes.

---

## 7. Results: final checkpoint

### 7.1 Reverse mode, four seeds, step 24000

The reverse-mode test at the final training checkpoint, all four
seeds. Settings: $\min\_pilots\_per\_cell = 30$, $\mathtt{top\_k\_cells}
= 100$, $\mathtt{n\_shuffles} = 20$.

#### 7.1.1 Headline numbers

| Seed | $|C|$ | dict dim | peak excess | peak z | ER (real) | ER (null) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 35 | 9 | 8.3% | 11.1 | 14.56 | 12.40 | H_partial_dictionary |
| 1 | 35 | 8 | 8.2% | 9.1 | 14.72 | 12.52 | H_partial_dictionary |
| 2 | 35 | 8 | 9.0% | 10.9 | 14.60 | 12.93 | H_partial_dictionary |
| 3 | 35 | 9 | 7.9% | 15.7 | 14.67 | 12.34 | H_partial_dictionary |
| **mean** | **35** | **8.5** | **8.4%** | **11.7** | **14.638** | **12.547** | — |
| **std (ddof=1)** | **0** | **0.6** | **0.5%** | **2.8** | **0.071** | **0.266** | — |
| **relative spread** | 0% | 6.8% | 5.6% | 24.1% | 0.5% | 2.1% | — |

All four seeds return H_partial_dictionary with peak excess ~8% above
null, dictionary dimension ~8–9, and significance ~10–15σ. Cross-seed
relative spread on peak excess is 5.6%, comparable to the Phase 1
basis-invariant statistics' cross-seed agreement (~5% on $\log\alpha$,
~1% on $\lambda$). Dictionary dimension agrees to within ±1 across
seeds.

The reverse-mode verdict satisfies the cross-seed reproducibility
precondition (§3.3) with substantial margin.

#### 7.1.2 The cumulative energy table

Cross-seed mean cumulative energy at the leading $k$ singular values
of the trajectory view (with per-seed std in parentheses):

| $k$ | $E_k^{\mathrm{real}}$ | $E_k^{\mathrm{null}}$ | excess | peak z |
|---:|---:|---:|---:|---:|
| 1 | 25.2% (±0.4%) | 43.7% (±0.7%) | -18.5% | -18 |
| 2 | 39.9% (±0.5%) | 49.1% (±0.6%) | -9.2% | -9 |
| 3 | 50.7% (±0.5%) | 53.2% (±0.5%) | -2.4% | -2 |
| 5 | 65.0% (±0.2%) | 59.8% (±0.5%) | +5.2% | +6 |
| 8 | 76.4% (±0.2%) | 68.1% (±0.4%) | +8.3% | +11 |
| 10 | 80.9% (±0.1%) | 72.9% (±0.4%) | +8.0% | +13 |
| 15 | 88.2% (±0.1%) | 82.8% (±0.3%) | +5.4% | +13 |
| 20 | 93.1% (±0.0%) | 90.2% (±0.2%) | +2.9% | +12 |

Three features deserve attention:

**Real is below null at low $k$.** At $k = 1, 2$ the real cumulative
energy is *less* concentrated than the null by 18% and 9%
respectively. The leading singular value in real is ~250 vs ~110 in
null (a factor of 2.3× absolute), but the *relative* share of total
energy is lower because real has a much larger total deviation
budget. The pilot-conditioned cell means in real are large; the
within-cell variation pushes a lot of "tail" energy into the
spectrum that the random shuffle cannot easily replicate.

**Real crosses above null around $k = 4$.** From $k = 5$ onward, real
dominates null in cumulative energy. The crossover is the fingerprint
of a dictionary plus a tail: at low $k$, the long tail spreads the
real spectrum more broadly than the noisier null; at intermediate
$k$, real has captured its shared structure while null is still
climbing.

**The peak gap is at $k = 8$–$9$.** This is the dictionary dimension.
At $k = 8$, real has captured 76.4% of total deviation energy vs the
null's 68.1%, an 8.3% excess at 11σ. Past this point, additional
modes are no longer enriched (excess decreases monotonically to 0 at
$k = 35$). The 8 leading shared modes carry the dictionary's
contribution; everything past is the cell-specific tail.

#### 7.1.3 The trajectory-view singular value spectrum

The trajectory-view singular values (cross-seed mean):

| $k$ | $\sigma_k^{\mathrm{real}}$ | $\sigma_k^{\mathrm{null}}$ | ratio |
|---:|---:|---:|---:|
| 1 | 247 | 115 | 2.15× |
| 2 | 154 | 38 | 4.05× |
| 3 | 130 | 33 | 3.94× |
| 5 | 95 | 28 | 3.39× |
| 8 | 60 | 22 | 2.73× |
| 10 | 48 | 20 | 2.40× |
| 15 | 38 | 16 | 2.38× |
| 20 | 30 | 13 | 2.31× |
| 35 | 17 | 8 | 2.13× |

The real spectrum is approximately twice the null spectrum at every
$k$, with the largest *relative* gaps at $k = 2, 3$ (~4×) and the
largest *absolute* gaps at $k = 1, 2$ (~120 absolute units). This is
the structure expected of a long-tail dictionary: a few strong
shared modes pulling far above the null, plus a tail that decays
similarly to but consistently above the null.

#### 7.1.4 The direction-view spectrum

The direction-view spectrum (cell × layer matrix of $H$-dimensional
vectors) has 490 non-zero singular values. The real spectrum tracks
the null at all $k$, with real consistently 1.5–2× the null between
$k \approx 5$ and $k \approx 400$, converging to the null at the
spectrum's right tail.

The direction-view effective rank is 24.3 (real) vs 23.8 (null) — a
positive but small gap (~0.5 absolute, ~0.6σ). The direction view is
less informative than the trajectory view for the dictionary question,
because the cells contribute many (cell, layer) rows that share
within-cell layer-wise structure, which broadens the spectrum.

We report direction-view results in our npz outputs but the
substantive verdict is based on the trajectory view.

### 7.2 Pair mode, four seeds, step 24000

The pair-mode test at the final training checkpoint, all four seeds.
Settings: $\min\_pilots\_per\_cell = 15$, $\mathtt{top\_k\_cells} = 100$,
$\mathtt{top\_k\_tokens\_v} = 0$ (no input-frequency filter),
$\mathtt{n\_shuffles} = 20$.

#### 7.2.1 Headline numbers

| Seed | $|C|$ | dict dim | peak excess | peak z | ER (real) | ER (null) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 15 | 6 | 2.8% | 3.3 | 10.78 | 10.60 | H_weak_dictionary |
| 1 | 15 | 6 | 2.2% | 2.5 | 10.70 | 10.46 | H_weak_dictionary |
| 2 | 15 | 6 | 2.6% | 3.5 | 10.74 | 10.59 | H_weak_dictionary |
| 3 | 15 | 6 | 2.6% | 3.2 | 10.75 | 10.58 | H_weak_dictionary |
| **mean** | **15** | **6.0** | **2.5%** | **3.1** | **10.742** | **10.558** | — |
| **std (ddof=1)** | **0** | **0.0** | **0.3%** | **0.4** | **0.033** | **0.066** | — |
| **relative spread** | 0% | 0% | 9.9% | 13.9% | 0.3% | 0.6% | — |

Pair mode is much weaker than reverse: peak excess ~2.5% vs ~8.4%.
Verdict is H_weak_dictionary on all four seeds. Effective rank is
barely above null (gap 0.18, ~3σ).

Despite the smaller magnitude, the cross-seed reproducibility is
*more* tight than reverse: dictionary dimension is *exactly* 6 in
every seed (cross-seed std = 0), peak excess agrees to within 0.3%
absolute. This is informative: the pair-mode signal, however small,
is highly stable.

#### 7.2.2 The cumulative energy table — and the leading-edge inversion

The full cumulative-energy table for pair mode (cross-seed mean,
per-seed std in parentheses):

| $k$ | $E_k^{\mathrm{real}}$ | $E_k^{\mathrm{null}}$ | excess | peak z |
|---:|---:|---:|---:|---:|
| 1 | 19.6% (±0.7%) | 22.3% (±0.4%) | -2.8% | -8 |
| 2 | 37.0% (±0.6%) | 42.4% (±0.4%) | -5.4% | -8 |
| 3 | 49.7% (±0.5%) | 53.7% (±0.3%) | -4.0% | -5 |
| 5 | 70.2% (±0.3%) | 67.9% (±0.5%) | +2.3% | +3 |
| 6 | 76.0% (±0.3%) | 73.5% (±0.5%) | +2.5% | +3 |
| 8 | 83.4% (±0.3%) | 81.8% (±0.3%) | +1.7% | +3 |
| 10 | 89.7% (±0.3%) | 88.9% (±0.2%) | +0.8% | +2 |
| 15 | 100.0% | 100.0% | 0.0% | 0 |

The structure is qualitatively different from reverse mode. The
*inversion* at low $k$ is striking and stable across all four seeds:

- $E_1^{\mathrm{real}} = 19.6\% \pm 0.7\%$, $E_1^{\mathrm{null}} =
  22.3\% \pm 0.4\%$. The leading mode in real carries *less* energy
  than in null, with cross-seed reproducibility of the gap (-2.8%,
  std smaller than the gap itself).
- The peak of the inversion is at $k = 2$: real 37.0% vs null 42.4%,
  a 5.4 percentage point deficit at ~8σ.
- The cumulative real curve catches up to null around $k = 4$ and
  passes it by $k = 5$.

Crossover and peak: real exceeds null from $k = 4$ onward, with peak
positive excess of +2.5% at $k = 6$.

This is *not* the long-tail dictionary signature of reverse mode. In
reverse mode, real exceeds null from $k = 4$ onward by 5–8 percentage
points; in pair mode, the cumulative gap is mostly *negative* at low
$k$ and only weakly positive at intermediate $k$.

#### 7.2.3 Interpreting the leading-edge inversion

The leading-edge inversion has a clean mechanistic reading. The
within-input shuffle produces cells by sampling random subsets of
pilots that all share input $v$. The cell mean of such a random subset
is biased toward the *direction in which $v$'s pilots vary most* —
call this the "within-input principal axis." A different random
subset of the same $v$'s pilots, computed for a shuffled cell, will
have its mean dominated by the same axis simply because that's the
direction of maximum estimator variance.

Real cells $(v, w)$ are different. They contain pilots that share
both $v$ and $w$ — a non-random subset selected by the model's actual
predictions. The cell mean of a real $(v, w)$ cell is biased toward
whatever direction distinguishes "$v$-pilots that the model thinks
predict $w$" from "$v$-pilots overall." If this direction is
*orthogonal* to the within-input principal axis, the real cell mean
will be *less* concentrated along that axis than the within-input
shuffle is.

This is what the inversion says. The bigram-conditioned deviations
live in directions orthogonal to the dominant within-input variance
axes. Or equivalently: the bigram-conditioned function is
cell-specific (different bigrams pull the cell mean along their own
directions), not dictionary-shared (different bigrams using common
directions with different magnitudes).

The small positive excess at $k = 5$–$8$ is residual evidence of
*some* shared bigram structure across cells, but the dominant
finding is that bigram-conditioned structure is per-cell rather than
dictionary-organized.

#### 7.2.4 The pair-mode trajectory view spectrum

Cross-seed mean singular values:

| $k$ | $\sigma_k^{\mathrm{real}}$ | $\sigma_k^{\mathrm{null}}$ | ratio |
|---:|---:|---:|---:|
| 1 | 63 | 62 | 1.02× |
| 2 | 61 | 50 | 1.22× |
| 3 | 53 | 41 | 1.29× |
| 5 | 42 | 35 | 1.20× |
| 8 | 28 | 27 | 1.04× |
| 10 | 26 | 26 | 1.00× |
| 15 | 17 | 16 | 1.06× |

The real-vs-null ratio is small everywhere (1.0–1.3×), with a slight
bulge at $k = 2$–$5$. The leading singular value at $k = 1$ is
essentially equal in real and null. This is the spectrum of a "mostly
cell-private" structure with a small shared component.

### 7.3 The pair-vs-reverse contrast

Direct comparison of the two modes at the final checkpoint
(cross-seed means):

| Statistic | Reverse | Pair | Ratio | Interpretation |
|---|---:|---:|---:|---|
| Number of cells | 35 | 15 | 2.3× | bigram density limits pair |
| Dictionary dimension | 8.5 | 6.0 | 1.4× | both modes find a non-trivial shared subspace |
| Peak excess | 8.4% | 2.5% | 3.4× | reverse signal is much stronger |
| Peak z | 11.7 | 3.1 | 3.8× | reverse signal is much more significant |
| Effective rank (real) | 14.6 | 10.7 | 1.36× | both above null but reverse by much more |
| ER gap (real – null) | +2.1 | +0.18 | 11.7× | the "shared structure exists" signal |
| Verdict | H_partial_dictionary | H_weak_dictionary | — | reverse passes; pair barely passes |
| Leading-edge inversion | No (real ≥ null from $k=4$) | Yes (real < null at $k \leq 3$) | — | bigram structure is cell-specific |

The 3.4× factor between reverse and pair peak excess quantifies the
core finding:

- **About 70%** of what reverse mode sees is structure that vanishes
  when input is subtracted — i.e., input-token structure that
  survives the output filter. The reverse-mode dictionary is mostly
  describing how the trained model handles "which input typically
  precedes which output," not bigram-specific computation.

- **About 30%** survives input subtraction. This is genuinely
  bigram-conditioned structure beyond what input alone determines.
  Quantitatively small (~2.5% peak excess) but cross-seed
  reproducible to within 0.3%.

- The leading-edge inversion in pair mode further qualifies the
  surviving 30%: it lives in cell-specific directions, not in a
  shared dictionary.

In one sentence: **the network's learned conditional computation is
mostly per-input-token, with a small reverse-conditional dictionary
of shared output-relevant modes, and a much smaller bigram-specific
contribution that lives in per-cell directions.**

This is the headline empirical result of modified Test 3.

---

## 8. Results: training trajectory

The final-checkpoint analysis establishes the structure of the
trained model. The training-trajectory analysis (4 seeds × 12
log-spaced checkpoints between step 100 and step 24000) tells us
whether this structure is *learned* (emerging during training, in
which case it is a candidate object for the multi-view program) or
*architectural* (present from initialization, in which case it
would not require training to construct).

The trajectory analysis also lets us situate the modified-Test-3
findings in the broader convergence-time hierarchy of training-
dynamic phenomena documented in the Phase 1 and Markovianity work.

### 8.1 Reverse-mode trajectory

Cross-seed mean values at all 12 checkpoints (full per-seed
trajectories given in the npz output):

| Step | dict dim (mean ± std) | peak excess (mean ± std) | ER gap |
|---:|---:|---:|---:|
| 100 | 2.0 ± 0.00 | +25.3% ± 3.09% | -4.32 |
| 156 | 4.0 ± 1.41 | +4.7% ± 0.53% | -0.25 |
| 274 | 13.0 ± 0.00 | +1.8% ± 0.07% | +3.29 |
| 428 | 14.0 ± 0.82 | +2.0% ± 0.39% | +4.51 |
| 749 | 12.0 ± 0.00 | +3.4% ± 0.26% | +4.23 |
| 1171 | 10.0 ± 0.82 | +5.3% ± 0.35% | +3.45 |
| 2049 | 9.0 ± 0.00 | +7.3% ± 0.79% | +2.26 |
| 3205 | 9.0 ± 0.00 | +8.4% ± 0.35% | +1.70 |
| 5607 | 8.8 ± 0.50 | +8.7% ± 0.31% | +1.53 |
| 8771 | 8.2 ± 0.50 | +8.9% ± 0.38% | +1.58 |
| 15343 | 8.0 ± 0.00 | +8.8% ± 0.38% | +1.66 |
| 24000 | 8.5 ± 0.58 | +8.3% ± 0.46% | +2.09 |

Cross-seed agreement on peak excess is within ±1.0% (absolute) at
every checkpoint past step 156. The trajectory has three sharply
distinguishable phases.

#### 8.1.1 Phase A: noise-floor artifact at initialization (step ~100)

At step 100, the verdict is H_dictionary PASS at dict dim 2 with
peak excess 25.3%. This looks like a strong dictionary signal, but
the effective rank panel shows real ER ≈ 2.5 — *below* the null
ER ≈ 7. The deviation tensor's total magnitude is tiny: at step 100,
the model has barely trained, and trajectories conditioned on
output token are nearly identical to the marginal trajectory.

What little deviation does exist is concentrated in 1–2 trivial
modes — most likely the embedding-space directions corresponding to
the unembedding's near-random preference for $w$ over other tokens.
The "25% peak excess" at step 100 should be read as "the model has
not learned anything structured yet, but the small amount of
deviation that does exist is trivially concentrated in 1–2 modes,
which the noise-shuffled null cannot replicate."

This is the noise-floor regime. It is *not* dictionary structure in
the substantive sense; it's a regime where the total signal is small
enough that the leading-mode statistic is artifact-dominated.

Confirmation: across seeds at step 100, the dictionary dimension is
*exactly* 2 in every seed (cross-seed std = 0). This is the precise
fingerprint of "1–2 modes dominate a near-zero signal" — at higher
signal levels, the dictionary dimension would vary at least slightly
across seeds.

#### 8.1.2 Phase B: dictionary collapse (steps ~150–500)

Between step 156 and step 428, dictionary dimension *jumps* from 4
to 13 to 14, while peak excess crashes from ~5% to ~2%. This is the
transition phase: training is making the residual stream non-trivial,
the deviation tensor acquires substantial total magnitude across
many modes, but with no shared structure yet — the deviation is
spreading near-uniformly across the available residual-stream
dimensions.

Effective rank rises from 4.3 at step 156 to 8.6 at step 428 (real),
with null rising more slowly from ~5 to ~4. The real-minus-null ER
gap becomes positive (+3 to +4.5) throughout phase B, confirming the
deviation is real and structured, just not concentrated.

This phase overlaps with the rapid-fall window of the Markovianity
decoupling. The Markovianity writeup (§5.4 of that document) finds
$R^2_{\mathrm{pos}}$ falling from 0.59 at step 156 to 0.13 at step
2049. Phase B of the reverse-mode dictionary trajectory is the
period in which the model is learning to make attention selective
(driving Markovianity down) without yet organizing the attention-
induced deviation into shared trajectory modes.

#### 8.1.3 Phase C: genuine dictionary emergence (steps ~750–5000)

From step 749 onward, peak excess rises monotonically:

- step 749: 3.4%
- step 1171: 5.3%
- step 2049: 7.3%
- step 3205: 8.4%
- step 5607: 8.7%

By step 5607 the dictionary structure has reached its asymptote.

Dictionary dimension *contracts* in parallel: 12 (step 749) → 10
(1171) → 9 (2049) → 9 (3205) → 8 (15343). The "spread" structure of
phase B is consolidating into a compact dictionary; modes that
carried small amounts of structured signal in phase B are being
re-organized into a smaller set of more-strongly-loaded shared modes.

Effective rank stays roughly stable at ~14–16 in real (vs null at
~12–14) through phase C and beyond, with the real-null gap settling
around +1.5 to +2.

The phase C asymptote (peak excess ~8.4% at dict dim ~8) is the
*learned* reverse-mode dictionary. It is the part of the trained
model's output-conditional structure that is organized along a
small number of shared modes across successor tokens. The signal is
real (~11σ), learned (not present at initialization), and cross-seed
reproducible (peak excess agreement within ~5% relative across the
four seeds).

#### 8.1.4 Convergence time vs the Markovianity decoupling

A central question is whether the dictionary emergence is co-located
with or lagging the Markovianity decoupling.

The Markovianity writeup (Table in §5.4 of that document) gives the
$R^2_{\mathrm{pos}}$ trajectory:

| Step | $R^2_{\mathrm{pos}}$ |
|---:|---:|
| 100 | 0.57 |
| 156 | 0.59 |
| 274 | 0.59 |
| 428 | 0.48 |
| 749 | 0.34 |
| 1171 | 0.23 |
| 2049 | 0.13 |
| 3205 | 0.13 |
| 5607 | 0.10 |
| 24000 | 0.10 |

Side-by-side with the reverse-mode dictionary peak excess:

| Step | $R^2_{\mathrm{pos}}$ | Dictionary peak excess | Both phenomena |
|---:|---:|---:|---|
| 100 | 0.57 | (noise-floor) | initialization |
| 156–428 | 0.59 → 0.48 | (collapse, ~2%) | decoupling starts; dictionary in spread |
| 749 | 0.34 | 3.4% | decoupling well underway; dictionary emerging |
| 2049 | 0.13 | 7.3% | decoupling near asymptote; dictionary growing |
| 3205 | 0.13 | 8.4% | decoupling at asymptote; dictionary near asymptote |
| 5607 | 0.10 | 8.7% | decoupling at asymptote; dictionary at asymptote |

The Markovianity decoupling and the reverse-mode dictionary emergence
overlap in time but are not exactly co-located:

- The Markovianity R² rapid-fall completes by step ~2000–3000.
- The dictionary peak excess crosses 5% around step 1171 and reaches
  its asymptote around step 5000–5600.

The dictionary emergence lags the Markovianity decoupling by a factor
of roughly 1.5–2× in characteristic timescale. This is
mechanistically consistent with a two-step process:

1. **Step 1: Markovianity decoupling.** The network learns to make
   attention selective — so the per-layer update depends on context
   rather than on $x_t$ alone. This is the prerequisite for any
   context-conditional structure to exist in the deviation tensor.
2. **Step 2: Dictionary emergence.** Given context-dependent
   computation, the network organizes the conditional deviations
   into a small number of shared output-relevant axes. This requires
   the decoupling to have already happened and uses additional
   training to consolidate.

The dictionary cannot form before the decoupling — there's nothing
in the deviation tensor to organize — but it doesn't need to form
*immediately* after the decoupling either. It asymptotes on its own
slightly slower schedule.

### 8.2 Pair-mode trajectory

Cross-seed mean values at the same 12 checkpoints:

| Step | dict dim (mean ± std) | peak excess (mean ± std) | ER gap |
|---:|---:|---:|---:|
| 100 | 1.5 ± 0.58 | +5.6% ± 1.42% | -0.75 |
| 156 | 9.5 ± 3.70 | +0.8% ± 1.07% | +0.29 |
| 274 | 8.0 ± 2.45 | +0.5% ± 0.14% | +0.39 |
| 428 | 3.2 ± 0.50 | +4.6% ± 1.04% | -0.49 |
| 749 | 5.0 ± 0.00 | +4.0% ± 0.73% | -0.22 |
| 1171 | 5.0 ± 0.00 | +4.3% ± 0.84% | -0.17 |
| 2049 | 5.0 ± 0.00 | +4.6% ± 0.58% | -0.29 |
| 3205 | 5.2 ± 0.50 | +3.8% ± 0.92% | -0.12 |
| 5607 | 5.5 ± 0.58 | +4.1% ± 0.71% | -0.20 |
| 8771 | 5.5 ± 0.58 | +3.1% ± 0.64% | +0.02 |
| 15343 | 6.0 ± 0.00 | +2.6% ± 0.51% | +0.15 |
| 24000 | 6.0 ± 0.00 | +2.6% ± 0.25% | +0.18 |

The pair-mode trajectory has *four phases* (not three): the noise-
floor phase A and the zero-signal phase B mirror reverse mode, but
phase C is split into a rise (steps 400–2000) and a decline (steps
2000–24000) that has no counterpart in reverse mode.

#### 8.2.1 Phase A: noise-floor artifact (step 100)

Step 100 gives dict dim 1.5 with peak excess 5.6%. Same regime as
reverse mode's step 100, but the noise-floor signal is much smaller
(5.6% vs 25%) because the pair mode's input-marginal subtraction
already removes most of the trivial token-identity contribution. The
phase-A reverse > pair ratio (4–6×) is roughly the strength of the
trivial input contribution that pair mode subtracts.

#### 8.2.2 Phase B: zero signal (steps ~150–400)

Peak excess drops below 1% at steps 156 and 274. Dictionary
dimension climbs to ~9–10 (the dictionary-dimension heuristic places
the peak at the right tail of the spectrum when no real signal
exists). Effective rank gap is essentially zero.

This is the cleanest "no shared bigram structure" window in the
entire trajectory. The bigram-specific signal doesn't exist yet in
any organized form: the model has begun to differentiate trajectories
(Markovianity decoupling is in its rapid-fall phase) but has not yet
constructed any bigram-conditioned shared structure beyond what
input alone determines.

#### 8.2.3 Phase C-rise: bigram structure emerges (steps ~400–2000)

Peak excess rises sharply from 0.5% (step 274) to 4.6% (step 428) to
4.6% (step 2049). Dictionary dimension consolidates around 5. The
bigram-specific contribution emerges in this window, structured into
a roughly 5-dimensional shared subspace.

The timing of this rise is informative. The bigram-specific signal
emerges *during* the Markovianity rapid-fall window (steps 200–2000),
not after it. The pair-mode and reverse-mode dictionaries have
different temporal relationships to the decoupling:

- **Pair mode** (bigram-specific contribution): emerges during the
  decoupling.
- **Reverse mode** (output-conditional contribution): asymptotes
  after the decoupling, lagging by ~1.5–2×.

This suggests bigram-specific structure is constructed *as soon as*
attention becomes context-selective enough to support it, whereas
the broader output-conditional structure requires the decoupling to
be near-complete before it consolidates.

#### 8.2.4 Phase C-decline: bigram dictionary attenuates (steps ~2000–24000)

The most striking feature of the pair-mode trajectory: peak excess
*declines* from its peak of 4.6% around step 2049 to 2.6% at step
24000. The decline is gradual (not a crash) and reproducible across
all four seeds — every seed shows the same shape, with the peak
between steps 2049 and 5607 followed by decline.

In numbers: the ~40% relative reduction (4.6% → 2.6%) happens over
the second half of training, on a timescale not seen in the
reverse-mode trajectory (which holds steady at ~8.4% through the
same window).

We propose two non-exclusive readings.

**Reading 1: Transient mid-training organization.** Around step
~2000, when the Markovianity decoupling and reverse-mode dictionary
emergence are both still in flux, the bigram-specific contribution
has its highest *relative* concentration in shared trajectory modes.
As training continues, the network refines its bigram-conditioned
computation into per-cell private directions, attenuating the
shared-mode component. The "shared mode" component of the bigram
contribution is transient mid-training infrastructure that the
mature trained model replaces with cell-specific machinery.

**Reading 2: Input-marginal absorption.** As the reverse-mode
dictionary grows from ~5% at step 1171 to ~8.4% at step 5607, the
input-marginal mean $\mu_v(t)$ acquires substantial output-relevant
structure (because the trained input-marginal trajectory now
incorporates the average prediction-commitment for input $v$).
Subtracting a richer baseline leaves a smaller residual. The
bigram-specific signal isn't disappearing in absolute terms; it's
being absorbed into the now-richer input-marginal baseline.

Reading 2 is mechanistically tidier and matches a more direct
quantitative prediction: as $\mu_v(t)$ becomes more informative, the
residual $\bar x_{v,w}(t) - \mu_v(t)$ shrinks proportionally. The
phase-C-decline shape would emerge naturally from the reverse-mode
dictionary's growth on the same time window.

The leading-edge inversion at the final checkpoint (§7.2.2) sits
specifically in the phase-C-decline regime. The ER-gap panel shows
real ER below null ER throughout phase C-rise (negative ER gap from
step 428 through step 5607) and only edges above null in phase
C-decline (step 8771 onward). The inversion at low $k$ in the
cumulative-energy curve is the signature of "the input-marginal
absorbs the high-variance shared directions, leaving the bigram-
conditioned residual in orthogonal directions."

The two readings of phase C-decline are not mutually exclusive but
they have different testable consequences. A leave-one-out
input-marginal (computing $\mu_v$ from pilots *not* in the cell
being analyzed) would eliminate the leakage from Reading 2; if the
phase-C-decline persists under leave-one-out, Reading 1 is
strengthened. We have not performed this check.

### 8.3 Trajectory comparison

The two trajectories' peak excess curves overlap qualitatively at
phases A and B (initialization noise floor, then zero signal) but
diverge sharply through phase C:

| Step | Reverse peak | Pair peak | Ratio (R/P) | Regime |
|---:|---:|---:|---:|---|
| 100 | 25.3% | 5.6% | 4.5× | initialization, both noise-floor |
| 156 | 4.7% | 0.8% | 5.9× | both crashing |
| 274 | 1.8% | 0.5% | 3.6× | both at low |
| 428 | 2.0% | 4.6% | 0.4× | **pair > reverse: phase B vs C-rise** |
| 749 | 3.4% | 4.0% | 0.9× | crossover region |
| 1171 | 5.3% | 4.3% | 1.2× | reverse overtakes pair |
| 2049 | 7.3% | 4.6% | 1.6× | both at peak (pair plateau, reverse rising) |
| 3205 | 8.4% | 3.8% | 2.2× | reverse asymptoting, pair starting to decline |
| 5607 | 8.7% | 4.1% | 2.1× | reverse asymptoted |
| 8771 | 8.9% | 3.1% | 2.9× | pair declining |
| 15343 | 8.8% | 2.6% | 3.4× | pair at final level |
| 24000 | 8.3% | 2.6% | 3.2× | both at final |

The crossover at step 428 is notable: at this single checkpoint,
*pair-mode signal is stronger than reverse-mode signal*. Reverse
mode is in its phase B "spread" (peak excess only 2.0%) while
pair-mode bigram structure has already started to organize (peak
excess 4.6%). This is the temporal signature of "bigram structure
emerges during the decoupling, reverse structure emerges after."

After step ~1000, reverse mode dominates and the ratio grows from
~1× to ~3×. The reverse-mode dictionary continues to consolidate
through step ~5000; the pair-mode bigram structure peaks around
step ~2000 and then declines.

### 8.4 Effective rank trajectories

For completeness, the trajectory-view effective ranks themselves
(separate from the gap), cross-seed means. Final-checkpoint values
are exact from the §7 tables; intermediate values are read from the
trajectory plot's bottom panel and are approximate to one decimal
place.

**Reverse mode**:

| Step | ER (real) | ER (null) | ER gap |
|---:|---:|---:|---:|
| 100 | 2.4 | 6.7 | -4.3 |
| 274 | 7.1 | 3.8 | +3.3 |
| 1171 | 12.3 | 8.1 | +4.2 |
| 3205 | 15.8 | 13.6 | +1.7 |
| 5607 | 16.0 | 14.4 | +1.5 |
| 24000 | 14.6 | 12.5 | +2.1 |

**Pair mode**:

| Step | ER (real) | ER (null) | ER gap |
|---:|---:|---:|---:|
| 100 | 10.1 | 10.8 | -0.7 |
| 274 | 8.4 | 8.0 | +0.4 |
| 749 | 8.6 | 8.8 | -0.2 |
| 2049 | 9.9 | 10.2 | -0.3 |
| 5607 | 10.3 | 10.5 | -0.2 |
| 24000 | 10.7 | 10.6 | +0.2 |

In reverse mode, both ER (real) and ER (null) grow substantially
from step 100 (real 2.4, null 6.7) to step 5607 (real 16, null 14),
then stabilize. The gap is positive throughout phases B and C,
confirming the real deviation is more spread out across modes than
the null. (The ER gap *itself* is monotonically decreasing from
+4.5 at step 428 to +1.5 at step 5607; the dictionary peak-excess
statistic, which measures *head dominance*, is monotonically
*increasing* over this same window. These two statistics measure
different things — uniformity vs head dominance — and we now have
direct empirical evidence that they can move in opposite directions.)

In pair mode, both ER (real) and ER (null) are nearly equal
throughout training, with the gap oscillating around zero and only
edging positive at the final checkpoint. The pair-mode signal lives
almost entirely in the head of the spectrum (the small dictionary-
dimension excess), not in the tail.

---

## 9. The discovery sequence

The test described in §5 — pair view and reverse view, energy-
concentration statistic, dictionary-dimension heuristic, mode-specific
shuffle null — is the result of iterating against the actual data
through a sequence of partial designs that each gave a substantively
misleading or under-powered verdict. This section recounts that
sequence explicitly. It is included for two reasons:

1. **Methodological transparency.** The final test design has
   non-obvious choices (within-input shuffle, energy concentration
   rather than effective rank, two complementary views). The reasons
   for these choices are clearest in light of the data they were
   designed against.

2. **Reproducibility of judgment.** A reader who runs the test on
   their own model might encounter intermediate results that resemble
   one of our earlier missteps. Knowing what we did and why may help
   diagnose unfamiliar findings.

The sequence is reconstructed from notes, run outputs, and
intermediate versions of the test script.

### 9.1 Original specification: pair view only, effective rank

The original Test 3 in the multi-view proposal specified the pair-
conditional view as the central object of study. The deviation was
defined against the SDE-derived linear-flow extrapolation $\hat\mu_v(t)$,
and the structural statistic was effective rank of the deviation
tensor compared to a global shuffle null.

We implemented this first. The Markovianity result (the companion
study, completed before modified Test 3) had already established
that the SDE drift only explained ~12% of $\Delta x_t$, so we
replaced $\hat\mu_v(t)$ with the empirical input-marginal $\mu_v(t)$
to avoid contaminating the deviation with SDE fit error. This was
the first deliberate modification, made at the design stage.

The remaining test was: select pair cells with at least
$\min\_pilots\_per\_cell$ pilots, compute per-cell deviations from
$\mu_v(t)$, compute trajectory-view effective rank of the deviation
tensor, compute effective rank under a global shuffle null, and
report the gap.

### 9.2 First-run problem: pair density too low

The first production run used $\min\_pilots\_per\_cell = 30$,
matching the default of the Phase 1 conditional analyses (which were
applied to the forward view, where input-token density is plentiful).
At this threshold on seed 0 at step 24000:

- 7 pair cells selected.
- Median 38 pilots per cell.
- Total 280 pilots in selected cells (2.8% of $N$).

7 cells is too few. The trajectory-view spectrum has only 7
singular values, so the SVD has very little room to distinguish
"shared structure" from "noise." The trajectory-view effective rank
of 7 cells is bounded above by 7, and the spread between real and
null is dominated by per-cell mean estimator noise rather than
spectrum structure.

We tried two responses:

1. Relax `min_pilots_per_cell` to 15. This gave 15 cells with
   median 28 pilots each — a 2× improvement in cell count and
   roughly the same per-cell variance. Better but still small.
2. Run the same analysis on additional seeds to check whether
   anything reproduces across seeds.

Both moves were sensible but didn't address the underlying issue:
pair-mode bigram density in our held-out data is fundamentally low.
With $N = 10{,}000$ pilots, we can only see ~15 bigrams with $\geq 15$
pilots each.

### 9.3 Addition of the reverse view (sample efficiency)

The recognition that pair mode is sample-limited led to the addition
of the reverse view. The argument:

- The reverse view conditions on output token only. Top successor
  tokens have many more instances than top bigrams — common words
  like "the" appear as the next token of thousands of contexts.
- At the same $\min\_pilots\_per\_cell = 30$ threshold, reverse mode
  gives ~35 cells with median 57 pilots each — a 5× improvement in
  cell count and a 2× improvement in per-cell sample size.
- The reverse view answers a related but distinct question: rather
  than "what does the bigram add beyond the input?" it asks "what
  does the output add to the trajectory?" These are different
  questions about the same multi-view program. Reporting both is
  more informative than reporting either alone.

We added reverse mode as a `--mode` flag. The script then supported
two cell-selection routines, two marginal-computation routines, two
shuffle-null routines (initially both global), and shared everything
else.

### 9.4 First reverse-mode run: misleading effective rank verdict

The first reverse-mode run on seed 0 at step 24000 returned:

```
trajectory-view effective rank (real): 14.6 of 35 possible
trajectory-view effective rank (null): 12.4 (± 0.4)
```

The real effective rank was *above* the null effective rank by 2.2
(at ~5σ). The pre-existing verdict logic mapped this to
**H_full_rank PASS**: "each cell drives the residual stream along
its own direction; little shared structure."

This verdict was wrong in a structurally important way, but we did
not realize this from the verdict alone. The 3-panel plot revealed
the truth.

### 9.5 Looking at the spectrum: a dictionary plus a tail

When we looked at the trajectory-view singular value plot rather
than the effective-rank summary, the leading singular values were
clearly dominant:

- Real $\sigma_1$ = 247, null $\sigma_1$ = 115. Real is more than
  twice the null at the leading mode.
- Real $\sigma_2$ = 154, null $\sigma_2$ = 38. Real is **4×** the
  null at the second mode.
- Real $\sigma_3$ = 130, null $\sigma_3$ = 33. Same factor.

This is the spectrum of a *clear dictionary structure*: a few
leading modes pulling far above the null, followed by a tail that
also sits above null. The reason effective rank classified this as
"full rank" rather than "low rank with a tail" is exactly the
weakness of the effective-rank statistic identified in §2.7:

> A spectrum with three large leading modes plus a long tail of
> small modes can have *higher* ER than a uniformly noisy null
> spectrum, because the entropy of the long tail dominates the
> ER calculation.

In numbers: with 35 cells, both real and null spread their squared
singular values across 35 values. Null is more uniform (no
single mode dominates, all are small) and so has lower ER; real
has 2–3 strong leading modes but the long tail of modes 8–35 is
spread enough that the entropy of the full distribution is *higher*
than the null's. ER conflates "spectrum uniformity" with "presence
of low-dimensional structure," and these go opposite ways for the
dictionary-plus-tail case.

### 9.6 Introduction of energy concentration

We rebuilt the verdict logic around cumulative energy concentration
in the leading $k$ modes:

$$E_k(\sigma) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_i \sigma_i^2}.$$

This statistic measures head dominance directly, ignoring the tail.
A spectrum with dictionary plus tail has $E_k$ much larger than
null's $E_k$ at the dictionary's intrinsic dimension, regardless of
the tail.

Computing $E_k$ for the seed-0 data:

| $k$ | $E_k^{\mathrm{real}}$ | $E_k^{\mathrm{null}}$ | excess |
|---:|---:|---:|---:|
| 1 | 25.7% | 44.2% | -18.5% |
| 3 | 51.3% | 53.5% | -2.2% |
| 5 | 65.0% | 60.0% | +5.0% |
| 8 | 76.3% | 68.2% | +8.2% |
| 10 | 80.9% | 72.9% | +8.0% |
| 15 | 88.2% | 82.8% | +5.4% |

The signal is now unambiguous. Real concentrates 8% more of its
energy in the top 8 modes than null does — a clear low-dimensional
dictionary plus tail.

The 25%-vs-44% gap at $k = 1$ was initially confusing — real *less
concentrated* than null at the leading mode? — but it has a clean
interpretation: the real spectrum has a longer "useful tail" than
the null. Energy in the real spectrum is more widely distributed
across modes 1–35 because all 35 cells contribute genuine deviation;
energy in the null spectrum is more concentrated in the first 1–3
modes because the random shuffle's deviation is dominated by
finite-sample noise that happens to project onto a few directions.
The crossover at $k \approx 4$ is the boundary where real's
distributed structure overtakes null's concentrated noise.

We added dictionary dimension (smallest $k$ at which the excess
peaks) as the secondary statistic and rewrote the verdict logic to
use peak excess as the primary criterion.

With the new verdict:

- Seed 0, reverse mode, step 24000: peak excess 8.3% at dict dim 9,
  ~11σ. Verdict: **H_partial_dictionary** (between the H_dictionary
  threshold of 10% and the H_weak_dictionary threshold of 0.5%).

This was the substantively correct verdict. The reverse-mode signal
has a clear dictionary structure that effective rank had missed.

### 9.7 Within-input shuffle: discovered through synthetic validation

While the verdict logic was being fixed, we noticed a related issue
on synthetic data. The synthetic-validation `no_structure` regime
(zero bigram-specific contribution, $\beta_{v,w} = \alpha_v$) was
intended to give H_no_signal. With the global shuffle null, it
*passed H_dictionary* with peak excess > 30% — a strong false
positive.

The mechanism: the global shuffle assigns pilots to cells regardless
of input token, so the shuffled cells contain mixed inputs. The
shuffled cell mean then averages over multiple input embeddings,
producing a much wider distribution than any per-input subset would.
The shuffle's deviation from $\mu_v$ (the *per-input* marginal, not
the global) becomes large because each shuffled cell contains pilots
whose $\mu_v$ varies widely. This large shuffle deviation has high
energy concentration in low-$k$ modes (the input-marginal directions)
and inflates the null floor to look strong.

The fix is the **within-input shuffle**: permute cell labels only
among pilots that share input. This preserves the property that
each cell's pilots all have input $v$, keeping the input-marginal
subtraction sensible. With this fix, the `no_structure` synthetic
regime gives H_no_signal at < 0.1% peak excess, as it should.

We added the within-input shuffle as the pair-mode null, keeping the
global shuffle for reverse mode (where there's no input grouping to
preserve, so global is correct).

This was the most surprising methodological discovery of the study:
the choice of null is *not* a free parameter. A naive null can
produce false positives an order of magnitude larger than the real
signal. We needed synthetic ground truth to catch the bug.

### 9.8 Pair-mode re-run with corrected null

With both fixes in place (energy concentration verdict + within-
input shuffle), we re-ran pair mode at $\min\_pilots\_per\_cell = 15$,
$\mathtt{top\_k\_tokens\_v} = 0$, $\mathtt{n\_shuffles} = 20$ on
seed 0 at step 24000:

- 15 cells, median 28 pilots.
- Dictionary dimension: 6.
- Peak excess: 2.8% at ~3σ.
- Verdict: H_weak_dictionary.

This was the first valid pair-mode result. The bigram-specific
signal is small but cross-seed reproducible (verified subsequently
on all four seeds).

The cumulative energy table also revealed the leading-edge inversion:

| $k$ | real | null | excess |
|---:|---:|---:|---:|
| 1 | 19.2% | 22.5% | -3.4% |
| 2 | 37.3% | 42.7% | -5.4% |
| 3 | 49.8% | 53.5% | -3.6% |
| 5 | 69.9% | 67.4% | +2.4% |
| 6 | 76.0% | 73.5% | +2.5% |

Real *below* null at low $k$, with peak inversion at $k = 2$. We
initially treated this as a possible noise artifact (we had only one
seed of data, and the 15-cell setup is statistically delicate). But
cross-seed runs subsequently showed the inversion magnitude is
remarkably reproducible: $E_1^{\mathrm{real}} - E_1^{\mathrm{null}}
= -2.8\% \pm 1.0\%$ across all four seeds. Not noise.

The interpretation (§7.2.3 above) — that bigram-conditioned
deviations live in directions orthogonal to dominant within-input
variance axes — emerged from the cross-seed reproducibility plus the
observation that the inversion is specific to the pair-mode within-
input null (reverse mode has no inversion).

### 9.9 Training trajectory: the four-phase structure

The final design move was adding a training-trajectory driver: run
the test at 12 log-spaced checkpoints from step 100 to 24000 across
all four seeds. This was straightforward to implement (a wrapper
around the single-checkpoint driver with a suppressed-output mode)
but produced several substantively important findings:

- The phase-A noise-floor artifact at step 100 (peak excess 25% in
  reverse, 5.6% in pair) was initially misread as "the dictionary is
  architectural after all." The bottom-panel effective-rank trace
  ($\mathrm{ER}_{\mathrm{real}}$ < $\mathrm{ER}_{\mathrm{null}}$ at
  step 100) revealed that this was an artifact of tiny total
  deviation magnitude, not a real dictionary.
- The phase-B "spread" between steps 156 and 428 was unexpected
  initially. We had not anticipated that dictionary structure would
  emerge through a "spread then consolidate" pattern; the natural
  prior was monotonic emergence. The observation that the dictionary
  *spreads first* and only consolidates in phase C is reminiscent of
  search-then-prune phenomena in other neural-net training regimes.
- The phase-C-decline in pair mode (peak excess from 4.6% at step
  2049 down to 2.6% at step 24000) was a surprise. We had expected
  pair mode to follow the same shape as reverse mode (monotonic
  emergence to asymptote). The decline is reproducible across all
  four seeds and emerged in the second half of training, prompting
  the two-reading interpretation (transient organization vs input-
  marginal absorption) in §8.2.4.

These were not pre-registered hypotheses, but they are robust
empirical facts. The trajectory analysis was essential to producing
the comprehensive picture in §10.

### 9.10 What the discovery sequence teaches

Three meta-level observations from this sequence:

1. **The choice of summary statistic is non-trivial.** Effective
   rank seemed like a natural single-number summary of "spectrum
   shape," but it conflates head dominance with tail uniformity, and
   the dictionary structure of our data lies in the head while a
   long tail of cell-specific structure inflates the tail's
   uniformity. Energy concentration in the head is the right
   primary statistic; effective rank is useful as a secondary
   diagnostic.

2. **The choice of null is not a free parameter.** In contexts where
   the deviation depends on a conditional marginal, the shuffle null
   must preserve the structural property the conditional subtraction
   uses. A global shuffle in pair mode produces false positives an
   order of magnitude larger than the real signal. Synthetic ground
   truth catches this; running on real data alone would have left
   the bug undetected.

3. **Multi-view comparisons are more informative than single-view.**
   The reverse-mode result alone would have suggested a strong
   dictionary; the pair-mode result alone would have suggested no
   structure or weak structure depending on threshold; the
   *comparison* between the two reveals that ~70% of the apparent
   reverse dictionary is input-marginal structure, with only ~30%
   genuinely bigram-conditioned. Neither view alone tells the full
   story; the contrast does.

The final test (§5) is the product of these realizations.

---

## 10. Synthesis and interpretation

This section pulls the pieces together. Six interpretive claims, each
explicitly tied to the relevant numerical findings.

### 10.1 H_dictionary is "half-true"

The original H_dictionary hypothesis — that the learned function
organizes itself along a small dictionary of shared trajectory modes
across cells — is true *in the reverse view* and not in the pair
view.

**Reverse view evidence**: H_partial_dictionary PASS at all four
seeds, peak excess 8.4% (±0.5%), dictionary dimension 8.5 (±0.6),
~11σ above null. The 8-mode shared subspace carries 8.4% more
deviation energy than would be expected under the no-structure null.

**Pair view evidence**: H_weak_dictionary at all four seeds, peak
excess 2.5% (±0.3%), dictionary dimension 6 (±0), ~3σ above null,
plus a clear leading-edge inversion. The bigram-specific contribution
beyond input-marginal is small and structured cell-specifically
rather than dictionary-organized.

The 3.4× factor between reverse and pair peak excess is the
quantitative form of "the apparent dictionary in reverse mode is
mostly input-marginal structure." When the input-marginal is
subtracted, ~70% of the apparent dictionary signal vanishes; the
~30% that remains is genuinely bigram-conditioned but lives in
per-cell directions.

This split between the two views is mechanistically informative.
The trained network's per-pilot trajectory is largely organized by
*which input is being processed* (mostly per-input-token structure,
captured by $\mu_v(t)$), with a smaller layer of organization by
*which output is being predicted* (the reverse-mode dictionary), and
a still-smaller bigram-specific contribution that lives in
per-cell, not per-mode, directions.

### 10.2 The dictionary is learned, not architectural

H_emergence PASS for reverse mode; H_architectural FAIL.

The reverse-mode training trajectory (§8.1) settles the question.
The apparent dictionary at step 100 is a noise-floor artifact (2
modes, tiny total deviation magnitude, real ER below null ER). The
genuine dictionary emerges between step 749 and step 5607, with peak
excess growing from 3.4% to 8.7% and dictionary dimension contracting
from 12 to 8. This is a 2.5× growth in signal magnitude on a
log-spaced 10× window of training steps — the dictionary is
constructed by training, not present at initialization.

For pair mode, the same story plus the additional phase-C-decline:
H_emergence holds in phases A and C-rise (step 100 to step 2049),
then is partially reversed in phase C-decline (step 2049 to step
24000). The bigram-specific contribution is also learned (not
architectural), but with a non-monotonic late-training behavior
that has no analogue in reverse mode.

### 10.3 The four-phenomenon, three-timescale hierarchy

We can now place modified Test 3's findings in the
convergence-time hierarchy that the Markovianity work began
constructing. The phenomena, with their rapid-change windows and
asymptote points:

| Phenomenon | Window | Asymptote |
|---|---|---|
| Markovianity decoupling ($R^2_{\mathrm{pos}}$) | steps 200–2000 | ~3000 |
| Pair-mode bigram dictionary | steps 400–2000 | peak ~2000, then decline |
| Reverse-mode output dictionary | steps 750–5000 | ~5000 |
| Phase 1 $\log\alpha$ hump | peak ~5000 | ~10000 |
| Phase 1 $\Sigma$-distance bump | steps 5000–10000 | ~10000 |
| Eval loss | still declining at step 24000 | not yet |

This identifies *four* distinct training-dynamic phenomena (Markov-
decoupling, pair-dictionary, reverse-dictionary, post-final-norm
anomaly) settling on *three* distinct timescales (~3000 for
decoupling and the pair peak; ~5000 for reverse asymptote; ~10000
for the Phase 1 anomalies), all well before eval loss converges.

The phenomena are not independent. The temporal ordering is
mechanistically consistent with a causal chain:

1. **First, Markovianity decoupling.** Training learns to make
   attention selective, so the per-layer update depends on context
   rather than on $x_t$ alone. Without this, no context-conditional
   structure exists in the deviation tensor.
2. **Concurrently, pair-mode bigram dictionary emerges.** As soon as
   attention becomes context-selective, bigram-specific shared
   structure appears. It does not need to wait for the decoupling
   to complete.
3. **Then, the reverse-mode output dictionary consolidates.** Once
   the decoupling is well underway, the network organizes its
   conditional deviations into a small set of shared
   output-relevant modes. This takes slightly longer than the
   decoupling itself.
4. **In the second half of training, the pair-mode bigram structure
   declines** as the input-marginal $\mu_v$ acquires more of the
   trained structure and absorbs the bigram-specific contribution.
5. **Phase 1 anomalies (log-α hump, $\Sigma$-distance bump) appear
   later** as the final geometric refinements.
6. **Eval loss continues to fall** throughout the visible window;
   not all training-relevant work is captured by geometric statistics.

This is a more articulated training dynamics picture than any
single test would produce. The multi-view program's distinct
empirical contribution is in offering structurally complementary
views that together fill in this picture.

### 10.4 What attention is doing, geometrically

Synthesizing across the Phase 1, Markovianity, and modified Test 3
findings, we can sketch a geometric description of what the trained
attention has learned.

At convergence:

- **~12%** of $\Delta x_t$ is a function of $x_t$ alone — the
  architectural residual that the Sarfati framework fits.
- **~88%** is context-dependent computation, distributed across the
  residual stream's ~256-effective-rank structure. This is the
  "noise" the framework absorbs.
- **Of this 88%**, the cell-conditioning analyses reveal:
  - **The largest contribution is per-input-token.** Two pilots
    sharing input $v$ have trajectories that follow a roughly common
    path $\mu_v(t)$, with deviations from it that average to small
    quantities under reverse-conditional sampling.
  - **There is a small reverse-conditional dictionary** of $\sim$8
    shared trajectory modes that organize output-conditional
    deviations from the global marginal. The dictionary explains
    $\sim$8.4% more energy than null in the leading 8 modes; the
    rest of the output-conditional deviation lives in the
    cell-specific tail.
  - **The bigram-specific contribution beyond input alone is very
    small** ($\sim$2.5% peak excess at the final checkpoint, ~3σ).
    What there is lives in directions orthogonal to the dominant
    within-input variance axes — i.e., it is per-cell rather than
    per-mode shared.

In short: the trained network's learned function is mostly an
input-token-specific path through residual-stream space, modulated
by a small dictionary of shared output-relevant adjustments, plus
small bigram-specific cell-private corrections. The architectural
residual (the Sarfati linear flow) is the substrate this all sits on
top of.

### 10.5 The reverse-mode dictionary is the natural next object to inspect

Modified Test 3 measured the *existence* and *dimension* of the
reverse-mode dictionary. We have not yet looked at the dictionary's
*content* — which residual-stream directions the leading singular
vectors point along, which tokens load most heavily on each, what
the per-layer activation shape of each mode is.

This is the natural next step. Two specific questions stand out:

- **What does the dictionary's per-layer activation shape look
  like?** Each shared mode is an element of $\mathbb{R}^{L_\mathrm{total}
  \cdot H}$ (a full trajectory in residual-stream space). Reshaping
  to $\mathbb{R}^{L_\mathrm{total} \times H}$ and computing the
  per-layer norm gives a "where in the network is this mode active"
  signature. We would expect leading modes to peak at late layers
  (around the prediction-commitment transition that Phase 1 found
  between layers 11 and 12), but this should be verified.

- **Which tokens load most heavily on each mode?** Computing
  $\langle d_w, m_i \rangle$ for each successor token $w$ and each
  leading singular vector $m_i$ gives a per-token weight on each
  shared mode. Tokens that load highly on the same mode are
  cluster-related in some sense — they share a common trajectory
  template through the network. The interpretability of these
  clusters (semantic? syntactic? POS-tag? frequency-correlated?)
  is the substantive open question.

Both inspections are basis-dependent in absolute terms — each seed's
singular vectors live in that seed's particular learned basis — but
the qualitative properties (per-layer shape, token clustering) are
basis-invariant and should be reproducible across seeds. Cross-seed
mode alignment via orthogonal Procrustes would establish whether the
same dictionary modes form in each seed.

### 10.6 The lines-of-thought framework: what survives

A natural question is whether modified Test 3's findings reflect on
the lines-of-thought framework itself. We think the answer is yes,
in a way that strengthens the framework's epistemic positioning
rather than undermining it.

The framework's central claim is that the marginal ensemble geometry
of a trained transformer's residual stream admits a remarkably
compact description: a few principal directions per layer, an
isotropic-Gaussian noise term, and a small set of basis-invariant
statistics that hold across architectures. The H_decoupling result
and the modified Test 3 results together establish that *the
framework's noise term hides ~88% of the per-layer update* in our
trained model.

What survives in the framework, given this:

- **The framework correctly describes the architectural residual.**
  The ~12% of $\Delta x_t$ that is a function of $x_t$ alone is
  exactly what the framework's drift term models. The framework's
  basis-invariant statistics are reproducible across seeds because
  they measure properties of this architectural component, which is
  the same across seeds modulo random rotation.

- **The framework's basis-invariant statistics are still useful
  signatures.** $\log\alpha$, $\lambda$, effective rank, and so on
  measure properties of the marginal ensemble that hold across the
  $\sim$88% noise term and the $\sim$12% architectural component
  combined. Different architectures may differ in either the
  architectural component or in what's absorbed into the noise, and
  the framework's statistics will reflect this in basis-invariant
  ways. The Phase 1 cross-seed agreement shows the statistics are
  reliable enough to do this comparison.

- **The framework is incomplete.** It does not characterize the
  context-dependent computation in any structural way. To describe
  the learned function, the framework would need to be extended with
  conditional analyses — exactly the multi-view program that
  modified Test 3 is part of.

The dictionary structure modified Test 3 measures *is the kind of
extension the framework needs*. It is basis-invariant (the dictionary
*dimension* and *peak excess* are invariant under rotation;
individual singular vectors are not), reproducible across seeds
(within Phase-1-comparable tolerances), and structurally compact
(a single small integer plus a single small percentage). It is the
right shape for a basis-invariant macro-statistic, in the framework's
own sense.

The framework's noise term is now characterized as: roughly
input-marginal structure, modulated by an 8-dimensional shared
output-conditional dictionary, plus small per-cell bigram-specific
corrections. The framework remains a clean description of the
marginal; the multi-view extension specifies what the marginal is
abstracting over.

---

## 11. Implications for the multi-view program

The multi-view proposal enumerated five tests. Modified Test 3 is
now complete; the remaining program needs to be reshaped in light of
what this study found.

### 11.1 Tests as originally enumerated

The multi-view proposal listed five tests in increasing structural
ambition:

1. **Test 1: Pair-conditional ensemble geometry.** For each cell,
   compute its basis-invariant statistics (effective rank, $\log\alpha$,
   $\lambda$, kurtosis) within-cell. Compare to all-to-all and to the
   forward/reverse views.

2. **Test 2: Pair-conditional well-definedness.** For each cell,
   compute the within-cell noise covariance structure. Test whether
   it is consistent with the SDE's isotropic-Gaussian assumption,
   and whether different cells have different noise structures.

3. **Test 3: Conditional mean vs marginal drift.** The present study.
   (Originally specified against SDE-derived $\hat\mu_v$; modified to
   use empirical $\mu_v$ following the Markovianity result.)

4. **Test 4: Action distribution.** Per-pilot Onsager–Machlup action
   under the marginal drift; cross-cell distribution of actions.

5. **Test 5: Onsager–Machlup bridges.** Most-likely paths between
   fixed endpoints under the SDE, compared to empirical
   $\bar x_{v,w}(t)$.

### 11.2 What modified Test 3 changed about the program

Two priorities have shifted.

**The dictionary inspection (Test 3b) is the highest-value next
step.** Modified Test 3 measured the existence of the reverse-mode
dictionary. The dictionary's *content* — what the leading singular
vectors look like, which tokens load on each, how they're shaped
across layers — is now the natural next object. This is a new test
not in the original enumeration; it sits between Test 3 and Test 4.

**Test 5 (Onsager–Machlup bridges) is deprioritized.** The
Markovianity result established that the SDE is not generative for
our trained model; the bridges computed under the marginal SDE
would be the SDE's best guess at the most-likely path, but we know
this best guess explains only ~12% of $\Delta x_t$. Computing the
bridges as a test would be largely tautological — they would
recover the SDE's predictions, against which the Markovianity test
has already measured the empirical deviation. The bridges remain
useful as a *baseline* (the "what the framework predicts" path that
we then compare against $\bar x_{v,w}$), but they're not a separate
test object.

### 11.3 The reshaped program

In priority order for the next phase:

1. **Test 3b: Dictionary mode inspection.** Extract the leading
   singular vectors of the reverse-mode trajectory-view deviation
   tensor at the final checkpoint, all four seeds. For each leading
   mode, compute (a) the per-layer activation profile (norm at each
   $L_\mathrm{total}$ position), (b) the per-token loading
   ($\langle d_w, m_i \rangle$ for each successor $w$), (c) the
   cross-seed correspondence after orthogonal Procrustes alignment.
   Expected outcome: a small number of named modes ("the
   prediction-commitment mode," "the syntactic-class mode," etc.)
   with stable per-layer shapes and identifiable per-token loadings.

2. **Test 2 (revised): Within-cell noise structure projected onto
   the dictionary.** For each reverse-mode cell, compute the within-
   cell residual covariance $\Sigma_w(t)$. Project onto the
   dictionary modes from Test 3b. Cells whose within-cell noise
   concentrates in dictionary directions have a "consistent"
   structure (the same shared modes drive both deviation and noise);
   cells whose noise lives orthogonally have cell-specific noise.
   This is the natural sharpening of Test 2 in light of Test 3.

3. **Test 4 (revised): Dictionary-mode trajectories across training.**
   Compute the per-mode projection of each cell's deviation onto the
   final-checkpoint dictionary modes. Track these projections through
   training. Modes whose projections grow late are
   "dictionary-construction" modes; modes whose projections are
   present early are "architectural-or-input-marginal" modes; modes
   whose projections peak mid-training and decline match the
   pair-mode phase-C-decline pattern. This decomposes the
   dictionary emergence trajectory into per-mode trajectories.

4. **Test 1: Pair-conditional ensemble geometry.** Originally the
   first test; remains worth doing but no longer the most informative.
   The Phase 1 / Markovianity / modified Test 3 sequence has already
   established the global structural picture; per-cell statistics
   would add detail rather than fundamentally new insight.

5. **Test 5: Onsager–Machlup bridges.** Deprioritized to the bottom
   of the queue.

Estimated effort for the next phase (Tests 3b + 2 + 4 revised):

- Test 3b: ~1–2 days analysis (no new compute on the model;
  re-process the saved deviation tensors), ~30 min compute.
- Test 2 (revised): ~2–3 days analysis, ~2 hours compute.
- Test 4 (revised): ~2 days analysis, ~3 hours compute.

Total next phase: ~1 week of analysis, ~6 hours of compute. Should
produce the multi-view program's headline interpretive result.

### 11.4 Cross-architecture generalization

A separate question is whether the modified Test 3 findings reproduce
in other architectures. The basis-invariant nature of the test
results (dictionary dimension, peak excess) is the right kind of
quantity for cross-architecture comparison — every architecture
should produce its own dictionary dimension and peak excess, and the
*spread* of these values across architectures characterizes the
universality (or not) of the dictionary phenomenon.

Concrete candidates:

- **GPT-2 medium** (the lines-of-thought paper's primary target):
  ~350M parameters, 24 layers, GELU FFN. Closer to our model in
  activation than to Llama; should produce a comparable dictionary.
- **Llama-2-7B** (also in the lines-of-thought paper): 32 layers,
  SwiGLU FFN, much larger. Whether the dictionary dimension scales
  with model size is an open question.
- **Mistral-7B**: 32 layers, SwiGLU + GQA. The architecture is
  similar to ours but at 50× the scale.
- **Pythia-12B**: another architecture from the lines-of-thought
  paper.

A study running the modified Test 3 across these architectures would
test (a) whether the dictionary phenomenon is universal at scale,
(b) whether dictionary dimension scales sub-linearly, linearly, or
super-linearly with model size, and (c) whether the leading-edge
inversion in pair mode is specific to our small-cell regime or
universal.

The compute cost would be comparable to ours per architecture: ~4
hours of CPU work on the augmented activation files, plus the
larger memory footprint of storing the activation tensors for the
larger models. This is a tractable Phase 2 study.

---

## 12. Open questions and limitations

### 12.1 Open questions for further investigation

**What does the dictionary's per-layer activation profile look like?**
Reshaping each leading singular vector to $\mathbb{R}^{L_\mathrm{total}
\times H}$ and computing the per-layer norm gives a "where in the
network is this mode active" signature. We have not done this; it is
the immediate next test (Test 3b).

**Which tokens load most heavily on each mode?** The per-token
loadings $\langle d_w, m_i \rangle$ for leading modes $m_i$ partition
the 35 reverse-mode cells into clusters by mode-loading similarity.
Whether these clusters are interpretable (POS-tag groups, semantic
groups, frequency-correlated, etc.) is the substantive interpretive
question. Test 3b will address this.

**Does the dictionary phenomenon scale with model size?** The
present study used a 150M-parameter model. Whether dictionary
dimension grows with model size — and if so, sub-linearly or
super-linearly — is unknown.

**Is the leading-edge inversion in pair mode universal?** The
inversion is the cleanest cross-seed finding in pair mode but is
based on only 15 cells. Whether it persists at higher cell counts
(achievable with larger $N$ or with denser bigram coverage) is
open.

**Is the pair-mode phase-C-decline an artifact of input-marginal
absorption, or a genuine refinement?** §8.2.4 proposed two readings.
A leave-one-out input-marginal (computing $\mu_v$ from pilots not in
the cell being analyzed) would distinguish the readings. We have
not implemented this.

**Does the dictionary structure correspond to specific attention
heads?** Each attention head contributes to the context-dependent
component of $\Delta x_t$. Per-head ablation analysis (zero out one
head's output, re-measure the dictionary) would localize each mode
to a small subset of heads, identifying a direct mechanistic link
between attention patterns and the basis-invariant dictionary
statistics.

**Is the convergence-time hierarchy a robust feature of trained
transformers?** The Markovianity decoupling, pair-mode bigram
dictionary, reverse-mode output dictionary, and Phase 1 anomalies
settle on different schedules in our setup. Whether this hierarchy
generalizes across architectures, data distributions, and training
recipes is open.

### 12.2 Methodological limitations

**Cell counts are small.** Reverse mode has 35 cells; pair mode 15.
This is a hard limit on the rank of any structure the test can
detect. Dictionary dimension ~8 in reverse mode is well below the
35-cell ceiling, so the dictionary is well-resolved; but the bound
matters in principle. Larger $N$ would lift it.

**Per-cell sample sizes are modest.** Median 57 pilots per cell in
reverse, 28 in pair. The per-cell mean estimator has standard error
$\sigma_t / \sqrt{n_c}$, which at $n_c = 28$ is $\sim$19% of
$\sigma_t$. Some of the "noise" we see in the spectrum is mean-
estimator noise rather than genuine cell-specific variation. The
shuffle null partly controls for this (the null also has estimator
noise of the same magnitude), but not entirely.

**The marginal subtraction is empirical, not predictive.** Our
$\mu_v(t)$ is the empirical mean over pilots with input $v$, which
includes the pilots from the cell itself. This induces a small
"leakage" bias in pair mode: cells with high $n_{v,w}/n_v$ ratios
have their own contribution baked into the baseline. The bias is
small (median ratio ~5%) but non-zero. A leave-one-out version of
the baseline would eliminate this.

**Single architecture, single training recipe, single data.** The
four seeds differ only in init and data shuffle. Whether the
dictionary structure or the pair-mode decline generalize beyond
this setup is untested.

**The leading-edge inversion interpretation is plausible but not
directly tested.** §7.2.3 reads the inversion as "bigram-conditioned
deviations are orthogonal to dominant within-input variance axes."
This is consistent with the data but doesn't follow uniquely.
Directly projecting real cell means onto the within-input principal
directions would confirm or rule out the interpretation.

**Effective rank is a misleading summary statistic in this context.**
We documented this in §3.3 and §9. Any future test using effective
rank should be checked against energy concentration.

### 12.3 Validation this work would benefit from

The validations that would most strengthen the conclusions:

1. **Larger pilot collection.** Re-run with $N = 50{,}000$ or
   $100{,}000$ pilots. This would give ~5–10× more pair-mode cells
   and lower per-cell estimator noise. Single highest-value
   validation.

2. **Per-cell mean estimator stability.** Compare cell-mean spectra
   computed from disjoint halves of the same cell's pilots. The cell
   mean estimator should agree across halves; the leading singular
   vectors of $D$ should align across halves under orthogonal
   Procrustes.

3. **Leave-one-out marginal.** Recompute pair mode with $\mu_v(t)$
   computed excluding pilots in the cell being analyzed. Compare
   pair-mode peak excess and phase-C-decline shape under leave-one-
   out vs the full-data marginal. Tests Reading 2 vs Reading 1 of
   the phase-C-decline.

4. **Synthetic at full residual-stream scale.** All synthetic
   validation used $H = 64$ with full rank; the real model has
   $H = 896$ with effective rank ~256. A synthetic test at the
   real model's scale would confirm the test discriminates
   correctly in the high-effective-rank regime.

5. **Cross-architecture sweep.** Run the same test on GPT-2 medium,
   Llama-2-7B, Mistral-7B, Pythia-12B (the lines-of-thought paper's
   architectures). Confirms or refutes universality of the dictionary
   phenomenon.

---

## 13. Summary

### 13.1 Headline findings

1. **The reverse-mode dictionary exists.** Conditioning the cell-
   conditional mean on the output token alone (subtracting the
   global marginal) reveals an ~8-dimensional shared dictionary that
   carries ~8.4% more cumulative energy in its leading modes than
   the global-shuffle null predicts. ~11σ above null, basis-
   invariantly measurable, reproducible across four seeds to within
   ~5% relative on peak excess and ±0.6 on dictionary dimension.
   **H_partial_dictionary PASS.**

2. **The dictionary is learned, not architectural.** At
   initialization (training step 100), the apparent "dictionary"
   structure is a noise-floor artifact: 1–2 modes dominate a near-
   zero total deviation. Genuine dictionary structure emerges
   between steps ~750 and ~5000. **H_emergence PASS,
   H_architectural FAIL.**

3. **The pair-mode bigram-specific contribution is small, cell-
   specific, and non-monotonic in training.** After subtracting the
   input-marginal, residual bigram-conditioned shared structure is
   only ~2.5% peak excess at the final checkpoint, with a clear
   leading-edge inversion (real cumulative energy below null at
   $k \leq 3$) that reproduces across all four seeds. The trajectory
   shape is non-monotonic: rises to a peak of ~4.6% around step 2000
   and then declines to ~2.6% at step 24000. **H_weak_dictionary**
   throughout the mature trained regime.

4. **Most of the multi-view-conditional structure is input-
   marginal.** The 3.4× factor between reverse-mode and pair-mode
   peak excess says ~70% of what reverse mode sees is input-token
   structure exposed through the output filter; only ~30% is
   genuinely bigram-conditioned. The bigram-specific contribution is
   structured cell-specifically rather than dictionary-organized.

5. **The convergence-time hierarchy now has four phenomena on three
   timescales.** Markovianity decoupling ($R^2_{\mathrm{pos}}$
   rapid-fall, asymptote ~3000), pair-mode bigram dictionary (peak
   ~2000, then decline), reverse-mode output dictionary (asymptote
   ~5000), Phase 1 anomalies ($\log\alpha$ hump and $\Sigma$-distance
   bump, asymptote ~10000). Each enabled by the previous; eval loss
   still declining at step 24000.

### 13.2 What this means for the broader project

The lines-of-thought framework describes the architectural residual
of a trained transformer — the part of the per-layer update that
mechanically follows from the MLP, residual connection, and norm
structure. Phase 1 established its cross-seed reproducibility on a
150M Llama-style model. The Markovianity test established that
training adds ~88% of the per-layer update as context-dependent
computation absorbed by the framework's noise term. Modified Test 3
characterizes the *structure* of that learned context-dependent
computation:

- Mostly input-marginal (per-input-token).
- Plus a small dictionary of $\sim$8 shared trajectory modes that
  organize output-conditional deviations from the global marginal.
- Plus a small bigram-specific contribution that lives in per-cell
  rather than per-mode directions, partially absorbed into the
  input-marginal as training matures.

The reverse-mode dictionary is now the candidate object for the
multi-view program's headline interpretive result. It has the basis-
invariance, the cross-seed reproducibility, the structural
compactness, and the connection to training dynamics that a useful
basis-invariant macro-statistic should have. Inspecting its leading
singular vectors — which tokens load on each, what per-layer shape
each has — is the natural next test (§11.3).

### 13.3 What this work cost

Cumulative effort, including the design iterations documented in §9:

| Component | Time |
|---|---|
| Designing test, prototyping, initial pair-mode runs | ~3 days |
| Adding reverse mode, identifying effective-rank pitfall | ~1 day |
| Discovering and implementing energy-concentration statistic | ~1 day |
| Within-input shuffle null, synthetic validation | ~1 day |
| Pair-mode final-checkpoint, all four seeds | ~4 min compute |
| Reverse-mode final-checkpoint, all four seeds | ~14 min compute |
| Pair-mode training trajectory (4 × 12 ckpts) | ~50 min compute |
| Reverse-mode training trajectory (4 × 12 ckpts) | ~2.8 hours compute |
| Cross-seed analysis, interpretation | ~2 days |
| This writeup | ~2 days |
| **Total** | **~10 days, ~4 hours compute** |

Compute is negligible relative to model training (~48 GPU-hours per
seed across all checkpoints) and trivial relative to data collection
(~600 GB of activation files generated and stored separately). The
whole study runs on already-trained checkpoints on a single
workstation CPU.

### 13.4 What the next phase will cost

The recommended next-phase tests (§11.3) and their estimated costs:

| Test | Effort | Compute |
|---|---|---|
| Test 3b (dictionary mode inspection) | ~1–2 days | ~30 min |
| Test 2 revised (within-cell noise vs dictionary) | ~2–3 days | ~2 hours |
| Test 4 revised (per-mode trajectories through training) | ~2 days | ~3 hours |
| **Total** | **~1 week** | **~6 hours** |

This produces the multi-view program's headline interpretive result.

### 13.5 Final verdict (Phase 1 style)

A summary in the form of the Phase 1 verdict statements:

- **H_dictionary**: PASS in reverse mode (8.4% peak excess, 11σ,
  dim 8.5, four-seed reproducible). WEAK PASS in pair mode (2.5%
  peak excess, 3σ, dim 6, four-seed reproducible). The dictionary
  exists; it is reverse-mode in shape, partially input-marginal in
  composition, partially bigram-conditioned with cell-specific
  structure.
- **H_emergence**: PASS for reverse mode (peak excess grows from
  noise floor at step 100 through asymptote at step ~5000; 4× peak
  excess increase across the rapid-change window). PASS in phases
  A–C-rise for pair mode, with H_emergence-FAIL in phase C-decline.
- **H_architectural**: FAIL in both modes (peak excess varies by
  ~3× across training in reverse mode, ~6× in pair mode if we
  count phases A/B/C all together).
- **Cross-seed reproducibility**: PASS for reverse mode (peak excess
  agreement 5.6%, dictionary dimension ±0.6). PASS for pair mode
  (peak excess agreement 9.9%, dictionary dimension ±0).
- **The leading-edge inversion**: cross-seed reproducible (-2.8%
  ± 1.0% at $k = 1$), reading-A (orthogonal-to-within-input)
  consistent but not directly tested.

Modified Test 3 is complete. The headline finding is the existence
and emergence of the reverse-mode dictionary; the secondary finding
is the small, cell-specific, non-monotonic-in-training bigram-
specific contribution.

---

## Appendix: artifacts and reproduction

### A.1 Scripts

The complete implementation of modified Test 3 is in a single
script, `pair_deviation_test.py`. It supports both pair and reverse
modes via the `--mode` flag and includes a `--training` flag for the
multi-checkpoint trajectory driver.

Run commands for the results reported in this document:

```
# Single-checkpoint reverse mode, all four seeds:
for s in 0 1 2 3; do
    python3 pair_deviation_test.py --mode reverse \
        --seed $s --step 24000
done

# Single-checkpoint pair mode, all four seeds:
for s in 0 1 2 3; do
    python3 pair_deviation_test.py --mode pair \
        --seed $s --step 24000 \
        --min-pilots-per-cell 15 --top-k-tokens-v 0
done

# Training trajectory, reverse mode:
python3 pair_deviation_test.py --mode reverse --training

# Training trajectory, pair mode:
python3 pair_deviation_test.py --mode pair --training \
    --min-pilots-per-cell 15 --top-k-tokens-v 0
```

The training-trajectory driver auto-detects seeds (0,1,2,3 by
default) and 12 log-spaced checkpoints from the available 50.

### A.2 Saved outputs

Per-checkpoint outputs (one per (mode, seed, step) triple):

- `phase1_runs_gelu/pair_deviation/{mode}_seed_{S}_step_{STEP:08d}.npz`:
  contains `cells` (cell identities), `cell_counts`, `deviation`
  tensor of shape $(|C|, L_\mathrm{total}, H)$ as float32,
  `n_cells`, `mode`, `verdict` (string), `dictionary_dimension`,
  `peak_excess`, `peak_z`, the SVD spectra `sv_directions` and
  `sv_trajectories`, the entropy-based effective ranks
  `effective_rank_directions` and `effective_rank_trajectories`,
  the cumulative energy curves `energy_directions` and
  `energy_trajectories`, and the shuffle-null counterparts of all
  of these with `null_` prefix. Plus a `config` block recording the
  parameters used.

- `phase1_runs_gelu/pair_deviation/{mode}_seed_{S}_step_{STEP:08d}.png`:
  3-panel figure with direction-view spectrum (top), trajectory-view
  spectrum (middle), and trajectory-view cumulative energy
  concentration (bottom) with the dictionary dimension marked.

Training-trajectory aggregated outputs (one per mode):

- `phase1_runs_gelu/pair_deviation/{mode}_trajectory_seeds_0_1_2_3.npz`:
  `(n_seeds, n_steps)` scalar arrays for `dictionary_dimension`,
  `peak_excess`, `peak_z`, `effective_rank_trajectories`,
  `null_effective_rank_trajectories`, `n_cells`. Plus the full
  `(n_seeds, n_steps, n_cells_max)` energy curves
  `energy_trajectories_real` and `energy_trajectories_null`.
  Plus `seeds`, `steps`, `mode`, `config`.

- `phase1_runs_gelu/pair_deviation/{mode}_trajectory_seeds_0_1_2_3.png`:
  3-panel figure with dictionary dimension trajectory (top), peak
  excess trajectory (middle), trajectory-view effective rank
  trajectories real vs null (bottom).

### A.3 Configuration of the runs reported in this document

**§7.1 reverse final-checkpoint cross-seed:**
$\min\_pilots\_per\_cell = 30$, $\mathtt{top\_k\_cells} = 100$,
$\mathtt{n\_shuffles} = 20$, step = 24000, all four seeds.

**§7.2 pair final-checkpoint cross-seed:**
$\min\_pilots\_per\_cell = 15$, $\mathtt{top\_k\_cells} = 100$,
$\mathtt{top\_k\_tokens\_v} = 0$, $\mathtt{n\_shuffles} = 20$, step =
24000, all four seeds.

**§8.1 reverse training trajectory:**
Same as §7.1 plus 12 log-spaced checkpoints from step 100 to step
24000, all four seeds.

**§8.2 pair training trajectory:**
Same as §7.2 plus 12 log-spaced checkpoints from step 100 to step
24000, all four seeds.

### A.4 Reproducibility

To reproduce the results from scratch, given the trained model
checkpoints and the held-out chunk set:

1. Run the augmented activation collection (`multiview.py`) for
   all 4 seeds × 50 checkpoints, producing the
   `augmented_step_*.npz` files.
2. Run the four commands in §A.1.

Step 1 produces ~600 GB of activation files and takes ~20 hours on
a single GPU. Step 2 produces the analysis outputs and takes ~4
hours of CPU time as documented in §4.8.

All randomness in step 2 is seeded deterministically from
`10000 * seed + step` (for the shuffle null), so reruns produce
bit-identical npz files. The plot files are not bit-identical due
to matplotlib's font caching but contain identical data.

### A.5 Synthetic-validation reproduction

The four synthetic regimes of §6 are not persisted as separate
artifacts but can be regenerated from the inline synthetic harness
in the development notebook. The synthetic data is fast to
generate (~10 sec per regime, $H = 64$, $N = 5000$).

The synthetic validation can also be run by calling the test
functions directly from a Python session, passing in synthetic
arrays in the same format as the augmented activation files. The
discriminative behavior is deterministic given the synthetic RNG
seed; we used seed 42 in the development.

### A.6 Companion documents

For context that this writeup deliberately built up from scratch
rather than referring to:

- `PHASE_1_WRITEUP.md`: the basis-invariant macro statistics, the
  cross-seed reproducibility "agreement on basis-invariants,
  disagreement on bases" puzzle, the training-dynamic anomalies
  (log-α hump, $\Sigma$-distance bump, post-final-norm anomaly).

- `MARKOVIANITY_WRITEUP.md`: the H_decoupling result that motivates
  the present study. Specifically: ~88% of $\Delta x_t$ variance is
  *not* a function of $x_t$, the SDE drift only models ~12%, the
  decoupling emerges through training on a schedule faster than
  eval loss convergence.

- `MULTI_VIEW_PROPOSAL.md`: the original five-test enumeration that
  the present study modifies (Test 3) and §11.3 reshapes.

- `PAPER_CODE_REVIEW.md`: paper-level notes on the lines-of-thought
  paper, including identified discrepancies between the paper's
  text and its accompanying code.

These documents are not required to follow the present writeup but
provide the broader research context.

