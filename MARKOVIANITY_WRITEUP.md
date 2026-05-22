# Training decouples the transformer residual stream from Markovianity

**Document status:** v1 (initial complete report)
**Audience:** thesis/paper reviewer; future-self; multi-view program launch reference
**Scope:** four-seed test of the lines-of-thought SDE's per-trajectory Markov prediction, at the final checkpoint and across training
**Companion documents:** `PHASE_1_WRITEUP.md` (v1) for the underlying basis-invariant macro statistics this work depends on; `PAPER_CODE_REVIEW.md` (v4) for paper/code-level notes on Sarfati et al. (ICLR 2025); `MULTI_VIEW_PROPOSAL.md` for the multi-view program this work re-prioritizes

---

## 0. Summary

The Sarfati et al. lines-of-thought framework (ICLR 2025) describes
the per-token trajectory through a transformer's residual stream as a
stochastic process — a deterministic flow derived from the ensemble
SVD at each layer, plus an isotropic Gaussian noise term. Read
seriously as a generative model, this framework predicts that the
per-layer update $\Delta x_t = x_{t+1} - x_t$ should be, in
expectation, a function of the current state $x_t$ alone. In other
words, the dynamics should be **Markovian in $x_t$** — the next
state's distribution depends only on the current state, not on the
history of how the trajectory got there.

We test this prediction directly on a 150M-parameter Llama-style model
trained from scratch across four seeds (the same checkpoints used in
the Phase 1 work; details in §2). At every layer, we bin pilot states
$x_t$ by a coarse partition (top-5 principal components of $x_t$, then
k-means into 24 clusters) and ask what fraction of the update variance
that binning predicts. Call this fraction $R^2_{\mathrm{pos}}$.

Two findings, in order:

**At the final checkpoint, $R^2_{\mathrm{pos}} \approx 0.10$, with a
ceiling-normalized resolved fraction of ~0.70.** Roughly 12% of the
per-layer update variance is a function of the current residual-stream
state; the remaining 88% depends on context that does not project
visibly into $x_t$. The strong Markov reading of the SDE is wrong by
about an order of magnitude. The state-determined component lives in
a roughly 3-dimensional subspace of $x_t$ — a dimensionality sweep
shows $R^2_{\mathrm{pos}}$ saturates by $d = 3$–$5$, regardless of how
many principal components are used to bin position. Cross-seed
relative dispersion on $R^2_{\mathrm{pos}}$ is 8%, comparable to the
Phase 1 basis-invariant statistics' cross-seed agreement.

**Across training, $R^2_{\mathrm{pos}}$ decreases monotonically from
~0.57 at step 100 to ~0.10 by step ~3000.** The decoupling is
$5.7\times$, smooth, cross-seed reproducible to within $\pm 0.01$
absolute spread at every checkpoint. It is complete well before the
Phase 1 training-dynamic events (the log-α hump at step ~5000, the
mid-training Σ-distance bump) and well before eval loss converges.
The earliest-converging geometric property of the trained model is
its *non*-Markovianity in $x_t$.

The interpretation that follows: the lines-of-thought framework
describes the **architectural residual** of the trained transformer —
the part of the per-layer update that mechanically follows from the
MLP, the residual connection, and the norm structure, regardless of
what the attention weights have learned. Training is fundamentally a
process of *decoupling* the residual stream from this architectural
Markovianity. The framework's noise term absorbs everything training
adds. Its basis-invariant macro statistics are reproducible across
seeds because they measure the architecture, which is the same; the
learned function lives in the context-dependent part of $\Delta x_t$
that the framework hides as noise.

The rest of the document develops this in detail. §1 introduces the
lines-of-thought framework for readers unfamiliar with it, names the
pre-registered hypotheses for the test, and clarifies what is and is
not in scope. §2 specifies setup: architecture, training recipe, the
test pipeline, and computational cost. §3 sets up the conceptual
question (whether there is a "natural path" through the residual
stream between fixed input and output tokens) and walks through the
physics framing — the overdamped Langevin equation, the Onsager–Machlup
action functional, and the Markovianity precondition. §4 specifies the
empirical test. §5 reports the cross-seed results, the dimensionality
sweep, and the training trajectory, with full per-layer and per-seed
tables. §6 develops the interpretation: what survives of the
framework, what doesn't, and what the action functional now measures.
§7 reshapes the multi-view experimental program in light of the
result. §8 lists open questions and limitations. §9 is a one-page
summary of headline findings, costs, and next-step protocol.

---

## 1. Introduction and framing

### 1.1 What this work tests

The Sarfati et al. lines-of-thought paper makes a remarkable empirical
claim. Take many tokens, drawn from many prompts, and look at the
ensemble of their trajectories through a transformer's residual
stream. The paper shows that:

- The ensemble's covariance at each layer has low effective rank
  (~256 dimensions out of a 768-dimensional ambient space, for GPT-2).
  Trajectories live on a low-dimensional curved manifold.
- The ensemble's principal axes rotate smoothly across layers — there
  is a well-defined SVD-derived rotation $R(t)$ and stretch
  $\Lambda(t)$ that describe the average flow.
- The deviation of any individual trajectory from this average flow is
  approximately Gaussian, isotropic, and has variance that grows
  exponentially with depth at rate $\lambda$.

Together, these observations support a stochastic differential
equation (SDE) of the form

$$dx(t) = \underbrace{\left[\dot R(t) R(t)^\top + R(t) \dot S(t) R(t)^\top\right] x(t)}_{\text{drift } b(x,t)}\, dt
         + \underbrace{\sqrt{\alpha \lambda e^{\lambda t}}}_{\text{noise scale}}\, dw(t),$$

where $w(t)$ is a Wiener process, $S(t) = \log\Lambda(t)$, and the
drift is a linear function of $x$ determined entirely by the ensemble's
SVDs at each layer. The paper presents this as a compact model that
reproduces the ensemble geometry from just a handful of parameters per
layer.

The framework's key methodological move is to absorb everything the
transformer does *beyond* the ensemble-average flow into the noise
term $dw$. The drift is the linear part the ensemble's SVD recovers;
the noise is everything else. This is a defensible move if you only
want to model the ensemble's marginal geometry, but it raises an
obvious question: what does the noise term hide?

Our Phase 1 work confirmed that the framework's basis-invariant
statistics are highly reproducible across four training seeds of a
150M-parameter Llama-style model trained with identical recipe — the
cross-seed relative dispersion on $\lambda$ is about 1%; on log-α,
about 5%; on qualitative shapes, essentially identical. Importantly,
this reproducibility holds despite the fact that the seeds learn
mutually orthogonal residual-stream bases at every layer (Phase 1 §7).
The framework's statistics describe genuine shared structure that is
*not* the network's internal coordinate representation.

This is a striking pair of findings on its own, but it leaves open
the question: are the framework's statistics descriptive of *ensemble
averages only*, or do they extend to individual trajectories? The
SDE, taken seriously, makes claims about individual paths through
the residual stream. If those claims hold, the framework is a
generative model. If they don't, the framework is a statistical
summary that happens to match the ensemble's lowest moments.

The present document is the result of testing the SDE's
per-trajectory prediction directly. The mechanism the SDE proposes
for individual trajectories is that the per-layer update is, in
expectation, a function of the current state — i.e., that the
residual stream dynamics are Markovian in $x_t$. We measure how
true this is, both at the final checkpoint and across training.

### 1.2 Hypotheses

We pre-registered three hypotheses about the training-trajectory shape
of the Markov ratio $R^2_{\mathrm{pos}}$, and one outcome about its
magnitude at convergence. Each is named here so the §5 results can be
reported against named claims.

- **H_monotone.** $R^2_{\mathrm{pos}}$ rises monotonically through
  training. The drift's Markovianity in $x_t$ is something the network
  *learns* — at initialization the update is mostly noise, and
  training organizes it into a position-dependent flow. This was our
  prior expectation.

- **H_hump.** $R^2_{\mathrm{pos}}$ has a peak around step ~5000 and
  partially relaxes. This would co-locate with the Phase 1 log-α hump
  and the Phase 1 Σ-distance bump, and would be evidence that the same
  training-dynamic event that produces those statistics is also
  reshaping the per-layer update's Markov ratio.

- **H_flat.** $R^2_{\mathrm{pos}}$ is approximately constant across
  training, near its final-checkpoint value of whatever that turns
  out to be. The state-dependence of the layer update is an
  *architectural* feature, not a *learned* feature; the framework's
  basis-invariant statistics describe the architecture, not the
  function.

These are mutually exclusive predictions about the shape of
$R^2_{\mathrm{pos}}(\mathrm{step})$. They all assume the magnitude is
somewhere in the range that the d-sweep saturation analysis would
land at — but they make different predictions about whether and how
training shapes it.

Separately, we tested the **dimensional saturation hypothesis**:

- **H_ceiling.** $R^2_{\mathrm{pos}}$ is bounded below the true value
  by the ceiling effect (top-5 PCs only capture a fraction of $x_t$'s
  variance). Increasing $d$ should raise $R^2_{\mathrm{pos}}$
  proportionately, with the resolved fraction $R^2/\mathrm{ceiling}$
  remaining roughly constant.

We test H_ceiling at the final checkpoint by sweeping $d \in \{3, 5,
10, 20, 40, 80\}$ on seed 0. PASS/FAIL is decided by whether
$R^2_{\mathrm{pos}}(d)$ rises with $d$ at the same rate as the
ceiling.

These four hypotheses partition the natural answers to the question
"is the residual stream Markovian in $x_t$, and how does that change
through training?" §5 reports the result against each.

### 1.3 Scope and methodological choices

The work is conducted on the existing Phase 1 four-seed checkpoint
set. We do not retrain, do not change the architecture, and do not
collect additional pilots beyond those produced by the multi-view
collection pipeline (multiview.collect_activations_with_metadata).
This was a deliberate choice: the test is designed to use existing
infrastructure so that the cost is purely in the analysis, and so
that the cross-seed reproducibility we measure is directly comparable
to the cross-seed reproducibility of the Phase 1 basis-invariant
statistics.

The test's design choices were calibrated against the residual
stream's known structure (effective rank ~256 in an $H = 896$ ambient
space). Specifically:

- **Top-$d$ PCA dimensionality** for the position binning was chosen
  at $d = 5$ for the main run. This is small relative to the
  residual stream's intrinsic dimension, which means the ceiling on
  $R^2_{\mathrm{pos}}$ is correspondingly low — but the resolved
  fraction $R^2/\mathrm{ceiling}$ is the basis-invariant quantity that
  controls for this. The d-sweep (§5.3) tests whether higher $d$
  would have changed conclusions.

- **K-means cluster count** was chosen at $K = 24$. This balances two
  competing pressures: too few clusters means coarse position
  partition with low resolution; too many clusters means small per-
  cluster samples and high within-cluster variance estimator noise.
  At $N = 10{,}000$ pilots, $K = 24$ gives mean cluster size 417 and
  minimum cluster size typically 3–10. This is in the "small but
  non-empty" regime; the within-cluster variance estimators are
  noisy but unbiased.

- **Shuffle null** was estimated at 20 replicates for the
  final-checkpoint single-seed runs and reduced to 5 replicates for
  the cross-checkpoint training-trajectory run. This decision was
  justified by the empirical observation that the shuffle-null
  standard deviation is consistently $\sim 10^{-3}$ — about 100×
  smaller than the gap above null — so distinguishing 20 replicates
  from 5 has no effect on any conclusion.

- **Number of pilots** is $N = 10{,}000$ per seed per checkpoint, the
  same as for the multi-view augmented activation collection. This
  is comparable to the Phase 1 SVD pilot count ($9500$) and well
  above the threshold for stable k-means in 5 dimensions.

The test is intentionally cheap. A single seed at a single checkpoint
runs in under 15 minutes on a laptop CPU; the full cross-seed,
12-checkpoint training trajectory run completes in approximately 7
hours on a workstation.

### 1.4 What is not in scope

- **Cross-architecture comparison.** We test the prediction on the
  single Phase 1 architecture (150M Llama-style). Whether the
  $R^2_{\mathrm{pos}} \approx 0.10$ asymptote is itself reproducible
  across architectures, or scales with depth/width, is a Phase 2
  question and is flagged as such in §8.

- **Bigram-conditional or other multi-view-conditional R² tests.**
  The natural follow-up — within a fixed (input, output) cell, how
  Markovian is the update? — is left to a future test in the
  multi-view program (the modified Test 3 of §7).

- **Causal claim about which weight matrix is responsible for
  decoupling.** We measure the phenomenon and characterize its
  shape; we do not attempt to localize it to attention's K, Q, or V
  projections, or to identify which heads contribute most. That is
  an interpretability question and is out of scope.

- **Comparison to the lines-of-thought paper's published trajectories
  or fits.** Our test is constructed entirely from our own activation
  collection. We do not attempt to recompute the paper's pre-trained-
  model numbers under the Markov-ratio lens.

---

## 2. Setup

### 2.1 Architecture

The pilot model is the same 146.4M-parameter Llama-style transformer
used for Phase 1, trained from scratch four times with different
seeds. Reproduced here for self-containment:

| Property | Value |
|---|---|
| Hidden size $H$ | 896 |
| Number of transformer blocks $L$ | 12 |
| Number of attention heads | 14 (head dim 64) |
| Number of KV heads | 2 (GQA) |
| FFN intermediate size | 2432 |
| Activation | GELU (Phase 1 GELU variant; see Phase 1 §1.2) |
| Position encoding | RoPE (base 10000) |
| Normalization | RMSNorm (pre-norm) |
| Tied embeddings | Yes |
| Vocabulary size $V$ | 32768 |
| Tokenizer | Mistral-7B-v0.1 |
| Training context | 1024 |

Layer states recovered by the analyzer: $L_\text{total} = 14$,
comprising the post-embedding state (layer 0), the 12 block-output
states (layers 1–12), and the post-final-norm state (layer 13). Layer
*transitions* indexed in this document run $t \to t+1$ for $t \in
\{0, 1, \ldots, 12\}$, giving 13 transitions. References to the
"deep interior" transition default to $6 \to 7$ (middle of the block
stack); references to the "prediction-commitment" transition default
to $11 \to 12$ (the last block output). References to the
"post-final-norm" transition are $12 \to 13$.

### 2.2 Training recipe

Identical to Phase 1's GELU variant:

| Property | Value |
|---|---|
| Corpus | FineWeb-Edu (sample-10BT subset) |
| Total training tokens | 1.57B |
| Total steps | 24,000 |
| Batch size | 8 micro × 8 grad-accum = 64 sequences |
| Context length | 1024 tokens (65,536 tokens/step) |
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1) |
| LR schedule | linear warmup (1000 steps) → cosine decay to 10% |
| Peak LR | 3e-4 |
| Gradient clip | 1.0 |
| Mixed precision | bf16 |
| Held-out set | 500 chunks of 1024 tokens (500K tokens) |
| Seeds | 0, 1, 2, 3 (independent init + data shuffle) |

Each of the four seeds completed training in approximately 12 hours
on a single RTX 5090.

### 2.3 The test pipeline

For each seed and each checkpoint analyzed, the test runs as follows:

1. **Load augmented activations.** Read the precomputed activation
   stack from
   `phase1_runs_gelu/multiview/seed_{S}/augmented_step_{STEP:08d}.npz`,
   produced by the multi-view collection pipeline. Each file contains
   per-layer hidden states (shape $L_\text{total} \times N \times H = 14
   \times 10000 \times 896$) plus per-pilot metadata (input_id,
   next_id, pred_id, position_in_chunk). The test uses only the states
   array; the metadata is reserved for the multi-view tests.

2. **For each layer transition $t \to t+1$:**

   a. **Compute the layer increment.** $\Delta x_t^{(k)} = x_{t+1}^{(k)} - x_t^{(k)}$
      for each pilot $k = 1, \ldots, N$.

   b. **Center and PCA-project $x_t$.** Subtract the per-coordinate
      mean of $x_t$ across pilots, then compute the centered SVD of
      $x_t \in \mathbb{R}^{N \times H}$. Project each pilot onto the
      top-$d$ right singular vectors to get $z^{(k)} \in \mathbb{R}^d$.
      Also record the fraction of $x_t$'s total variance captured by
      these top-$d$ singular values; this is the **ceiling**
      $V^{\mathrm{cap}}(t)$.

   c. **K-means cluster.** Run k-means++ initialization plus up to 50
      Lloyd iterations on the $\{z^{(k)}\}$ to produce cluster
      assignments $c^{(k)} \in \{1, \ldots, K\}$. Stop early on stable
      assignments.

   d. **Within-between variance partition.** For each coordinate $i$
      of the layer increment:
      - per-cluster mean $\mu_{c,i} = \mathbb{E}_{k \in c}[\Delta x_{t, i}^{(k)}]$,
      - per-cluster within-variance contribution (sum of squared
        deviations from cluster mean),
      - aggregate to $V_{\mathrm{within}}^{(i)}(t)$ and
        $V_{\mathrm{between}}^{(i)}(t)$ at each coordinate, then
        average across coordinates to scalars $V_{\mathrm{within}}(t)$
        and $V_{\mathrm{between}}(t)$.

   e. **Compute the Markov ratio.**
      $R^2_{\mathrm{pos}}(t) = V_{\mathrm{between}}(t) / [V_{\mathrm{within}}(t) + V_{\mathrm{between}}(t)]$.

   f. **Compute shuffle null.** Repeat steps (d) and (e) with the
      cluster assignments randomly permuted, $n_{\mathrm{shuffles}}$
      times. Record the mean and standard deviation of the shuffled
      $R^2_{\mathrm{pos}}$.

3. **Save raw arrays and plot.** Persist the per-layer arrays $V$,
   $R^2$, null mean, null std, ceiling, and cluster statistics to
   npz. Generate the two-panel plot (top: absolute $R^2$ with ceiling;
   bottom: resolved fraction = $R^2 / \mathrm{ceiling}$) as a PNG.

Steps 1–3 are encapsulated in `drift_welldef_test.py` (note: the
filename retains the original "drift well-definedness" terminology;
the analysis itself is what we now call the Markov-ratio test).

For the training-trajectory version, step 1 is iterated across a
log-spaced subset of available checkpoints and the per-checkpoint
arrays are stacked into a (n_checkpoints, n_transitions) matrix for
each seed. The training-trajectory driver is `drift_welldef_training.py`.

### 2.4 Validation regime

Before running on the trained model, the test was validated against
three synthetic regimes whose ground truth is known by construction.
These cases are embedded as inline assertions in the test driver and
also documented in §4.2. They establish that the test discriminates
correctly between (a) cases where the layer update is a deterministic
function of position, (b) cases where the layer update is independent
of position, and (c) cases where the relationship exists but the
top-$d$ PCs of $x_t$ miss most of $x_t$'s structure (high-rank $x_t$).

### 2.5 Computational cost

Cost is dominated by k-means and the shuffle-null replication. Per
checkpoint per seed at the default $d = 5$, $K = 24$, $n_{\mathrm{shuffles}} = 20$:

| Component | Time on workstation |
|---|---|
| Load augmented npz | < 1 sec |
| Per-layer SVD + PCA projection (×13 transitions) | ~30 sec total |
| K-means clustering (×13 transitions) | ~1 min total |
| Within-between variance partition (×13 transitions) | ~10 sec total |
| Shuffle null × 20 (×13 transitions) | ~6 min total |
| Save and plot | < 5 sec |
| **Total per checkpoint per seed** | **~8 min** |

The single-checkpoint cross-seed run (§5.1) is 4 seeds × 8 min ≈ 30
min. The d-sweep on seed 0 (§5.3) is 6 d-values × 10 min (slightly
slower at higher d due to larger PCA) ≈ 1 hour. The cross-seed
training trajectory (§5.4) is 4 seeds × 12 checkpoints × 8 min ≈ 7
hours, completed in two sittings.

---

## 3. The conceptual setup: pair-conditional paths and the action functional

### 3.1 The pair-conditional question

The conversation that produced this work started with a precise
question about a particular subset of the trajectory ensemble. Fix
two tokens $v$ and $w$. Consider all the pilots in the data where the
current token is $v$ and the next predicted token is $w$. Call this
the **pair-conditional ensemble** $E_{v,w}$. It is the intersection
of two conditional views: pilots conditioned on input token $v$
(the multi-view "forward" view), and pilots conditioned on successor
token $w$ (the multi-view "reverse" view).

This ensemble has interesting boundary behavior. At $t = 0$ (the
post-embedding layer), every trajectory in $E_{v,w}$ starts at the
same position — the embedding of $v$ — so the ensemble has zero
spread there. At $t = L_\text{total} - 1 = 13$ (the post-final-norm
layer), every trajectory in $E_{v,w}$ ends in a region of residual-
stream space that the unembedding maps to $w$ being the highest-
probability next token, so the ensemble has low (but nonzero) spread
there. In between, the ensemble has some spread that depends on
context — different occurrences of the bigram $(v, w)$ in different
left-contexts trace different paths through the middle layers.

The natural question: is there a *single most likely path* between
the input embedding and the output region, around which all the
context-specific paths cluster? In classical mechanics, the answer
would be given by a principle of least action: the trajectory
extremizing some functional $S[x]$ is the path the system takes, and
deviations from it have a known probability cost. Is there an analog
here?

### 3.2 Inertia, overdamping, and the Onsager–Machlup action

The first thing to settle is whether "action" even makes sense for
the residual stream. Action in classical mechanics is $\int L\, dt$
with Lagrangian $L = T - V$, the difference between kinetic and
potential energies. This presupposes inertia — a second-order
dynamics, $m\ddot x = F$, where mass and acceleration are meaningful.

The transformer's residual stream does not have inertia in this
sense. The update rule is first-order:

$$x_{t+1} = x_t + f_t(x_t, \text{context}),$$

where $f_t$ is the contribution of block $t$ (its MLP and attention
output, after layer-norms). The layer output is a *displacement*, not
an acceleration. There is no $\ddot x_t$ in the update rule; there is
no quantity playing the role of mass. The reasoning "force integrated
along the trajectory equals action" doesn't apply, because action is
$\int L\, dt$ (Lagrangian integrated over time), not $\int F \cdot dx$
(force integrated along the path) — the latter is *work*, an energy
not an action.

However, dynamics without inertia are not foreign to physics. The
**overdamped Langevin equation** describes systems in a regime of
high friction, where the inertial term $m\ddot x$ becomes negligible
compared to the drag term $\gamma \dot x$. Newton's equation
$m\ddot x = F - \gamma \dot x + \text{noise}$ collapses to

$$\gamma \dot x = F(x, t) + \sqrt{2\gamma k_B T}\, \eta(t),$$

or, absorbing $\gamma$ into the units, $\dot x = b(x, t) + \sigma\,
\eta$. This is the regime of colloidal particles in fluid, ions in
solution, and gene-expression dynamics. The "force" is the drift; the
noise is environmental fluctuation. There is no momentum, no
acceleration. Force balances drag instantaneously, so velocity is
proportional to force rather than acceleration being so.

The lines-of-thought SDE is exactly of this form. The framework's
linear flow is the drift; the framework's exponentially-scaling
Gaussian term is the noise. Treating layer depth $t$ as time, the
layer increment $\Delta x_t$ is a velocity (in continuous-depth
units), and the SDE is the model for that velocity. This is the
right physical reading of the framework — it is overdamped Langevin
in depth-as-time.

In this overdamped regime, the analog of the action functional is
the **Onsager–Machlup functional**:

$$S[x] = \frac{1}{2}\int_0^T \|\dot x - b(x, t)\|^2_{\Sigma^{-1}}\, dt + (\text{curvature correction}),$$

where the norm is taken in the metric set by the inverse noise
covariance. This is, up to the curvature correction, the negative log
of the probability density of the path $x(\cdot)$ under the SDE. Paths
that minimize $S$ are the most likely; paths with large $S$ have low
probability. For our purposes the curvature correction is small and
constant, and the operative quantity is the leading term — the
squared deviation between the actual velocity and the drift, weighted
by the inverse noise covariance.

This functional has a clean interpretation. $\dot x - b(x, t)$ is
the *deviation from the drift*. The integrand penalizes deviations
quadratically: paths that closely follow the drift accumulate low
action; paths that depart from it accumulate high action. If we
expand the squared norm:

$$S[x] = \frac{1}{2}\int_0^T \left[\|\dot x\|^2_{\Sigma^{-1}} - 2\langle \dot x, b\rangle_{\Sigma^{-1}} + \|b\|^2_{\Sigma^{-1}}\right] dt,$$

the middle term is recognizably a path-integral against the drift —
the "force integrated along the path" intuition reappears, but
inside a quadratic that includes a kinetic-energy-like first term
and a force-magnitude third term. The full action is what selects
most-likely paths, not work alone. (For a conservative drift $b =
-\nabla U$, the work term integrates to $-(U(x_T) - U(x_0))$, a
boundary contribution depending only on endpoints; in that special
case the path-dependence of the action lives entirely in the first
and third terms.)

For the pair-conditional question, the action functional gives a
precise object to compare against. The empirical conditional mean
$\bar x_{v,w}(t)$ is one candidate for the "natural" path between
$E(v)$ and the $w$-region. The Onsager–Machlup minimizer with the
same endpoints is another. If they agree, the SDE generates the
conditional bundle correctly. If they disagree, the SDE fails on
conditional structure even though it fits the marginal.

### 3.3 The Markovianity precondition

But the whole programme — Onsager–Machlup, action minimization, all
of it — rests on a precondition that has not been independently
checked. The action functional uses the drift $b(x, t)$. For this
drift to correctly describe the SDE, the per-layer update must in
fact be a function of the current state. Specifically: if we knew
$x_t$ exactly, could we predict $\Delta x_t$ accurately?

In the language of stochastic processes, this is the **Markov
property** for the residual-stream dynamics. A process $x(t)$ is
Markovian if the distribution of $x(t + dt)$ given $x(t)$ does not
depend on the history of how $x$ arrived at its current value. For
the discrete layer dynamics, the equivalent statement is that the
conditional expectation $\mathbb{E}[\Delta x_t \mid x_t]$ is a
well-defined function of $x_t$, with the residual $\Delta x_t -
\mathbb{E}[\Delta x_t \mid x_t]$ behaving like noise (uncorrelated
with $x_t$ and with itself across layers).

The SDE assumes the residual stream is Markovian in $x_t$. The
framework's drift $b$ is what you would compute if this assumption
held — the SVD-recovered linear part of the ensemble's update. If
the assumption fails — if two pilots that happen to be at the same
$x_t$ receive systematically different $\Delta x_t$ because of hidden
context (left-context tokens, attention's read from the KV cache,
position-in-chunk) — then the SDE is fundamentally a marginal fit,
not a generative description. The action functional in that case
computes something other than "negative log-density of a path under
the SDE," because the SDE the action references is the wrong model.

Mechanistically, the residual stream is not exactly Markovian in
$x_t$: attention reads from the KV cache, which depends on every
previous token's history, not just the current pilot's $x_t$. So
**strict** Markovianity is known *a priori* to fail. The empirical
question is the magnitude of the failure. If the non-Markov component
is small — say 10% of $\Delta x_t$ variance — the SDE is approximately
generative and the action functional is approximately meaningful as
originally intended. If the non-Markov component is large — half or
more — the SDE is fundamentally a marginal fit and the action
functional measures something else.

This is what we test.

---

## 4. Test specification

### 4.1 The Markov ratio $R^2_{\mathrm{pos}}$

We test Markovianity by asking how well a coarse partition of states
predicts the per-layer update. The full operational definition follows.

At each layer transition $t \to t+1$, $t \in \{0, 1, \ldots,
L_\text{total} - 2\}$:

1. **Reduce position to a low-dimensional summary.** Let $X_t \in
   \mathbb{R}^{N \times H}$ be the per-pilot state matrix at layer
   $t$. Center per-coordinate: $\tilde X_t = X_t - \bar X_t$ where
   $\bar X_t$ is the per-coordinate mean across pilots. Compute the
   SVD $\tilde X_t = U_t \Sigma_t V_t^\top$. The top-$d$ rows of
   $V_t^\top$ form a $d \times H$ projection matrix; project each
   pilot's centered state to obtain $z_t^{(k)} = (V_t^\top)_{[:d]}
   \tilde x_t^{(k)} \in \mathbb{R}^d$.

   Default: $d = 5$. The dimensionality sweep (§5.3) reports
   $d \in \{3, 5, 10, 20, 40, 80\}$.

2. **Partition states.** Cluster the $\{z_t^{(k)}\}_{k=1}^N$ into
   $K$ clusters using k-means++ initialization plus Lloyd iterations.
   We use a single seed-deterministic k-means run (no multi-restart);
   k-means seed = $10000 \cdot \text{model\_seed} + t$.

   Default: $K = 24$. Empirically this gives mean cluster size ~417
   and minimum cluster size 3–10 at $N = 10{,}000$.

3. **Partition the update variance.** Define the layer increment
   $\Delta x_t^{(k)} = x_{t+1}^{(k)} - x_t^{(k)}$. For each
   coordinate $i \in \{1, \ldots, H\}$, decompose the variance of
   $\Delta x_t^{(\cdot)}_i$ across pilots into within-cluster and
   between-cluster components:

   - Per-cluster mean: $\mu_{c,i} = \frac{1}{|c|}\sum_{k \in c}
     \Delta x_{t, i}^{(k)}$.
   - Per-cluster within-variance contribution (sum of squared
     deviations from cluster mean): $W_{c,i} = \sum_{k \in c}
     (\Delta x_{t,i}^{(k)} - \mu_{c,i})^2$.
   - Within-variance, per-coordinate, divided by $N$: $V^{(i)}_{\mathrm{within}}(t) = \frac{1}{N}\sum_c W_{c,i}$.
   - Between-variance, per-coordinate, divided by $N$, weighted by
     cluster size:
     $V^{(i)}_{\mathrm{between}}(t) = \frac{1}{N}\sum_c |c| (\mu_{c,i} - \bar\mu_i)^2$
     where $\bar\mu_i = \frac{1}{N}\sum_c |c|\mu_{c,i}$.

   These satisfy the law of total variance:
   $V^{(i)}_{\mathrm{within}}(t) + V^{(i)}_{\mathrm{between}}(t) = V^{(i)}_{\mathrm{total}}(t)$,
   the total per-coordinate variance of $\Delta x_t^{(k)}_i$ across
   pilots.

   Average across coordinates to scalars:
   $V_{\mathrm{within}}(t) = \frac{1}{H}\sum_i V^{(i)}_{\mathrm{within}}(t)$
   and similarly for $V_{\mathrm{between}}$.

4. **Compute the Markov ratio.**

   $$R^2_{\mathrm{pos}}(t) = \frac{V_{\mathrm{between}}(t)}{V_{\mathrm{within}}(t) + V_{\mathrm{between}}(t)}.$$

   This is the fraction of update variance explained by the position
   binning. It is the empirical estimator of the variance fraction
   one would recover if the update were Markovian in $x_t$ and we
   observed the state up to a binning of size $K$ in a $d$-dimensional
   reduction.

5. **Null baseline.** Randomly permute the cluster assignments
   $\{c^{(k)}\}$ — keeping the cluster *sizes* but breaking the
   association between cluster id and state — and recompute step 3
   and step 4. Repeat $n_{\mathrm{shuffles}}$ times. The mean and
   standard deviation of the shuffled $R^2_{\mathrm{pos}}$ form the
   shuffle-null baseline; in our setup it is consistently around
   $0.002$ with standard deviation $\sim 10^{-3}$, regardless of
   layer, $d$, or seed.

   The gap above null is reported in units of the shuffle-null
   standard deviation; in practice this gap is $\gtrsim 100\sigma$
   everywhere on real data, so the statistical-significance question
   is trivial.

6. **Diagnostic: ceiling.** Define the **structural ceiling**
   $V^{\mathrm{cap}}(t)$ as the fraction of $x_t$'s total variance
   captured by the top-$d$ singular values:

   $$V^{\mathrm{cap}}(t) = \frac{\sum_{i=1}^d \sigma_i^2(t)}{\sum_{i=1}^H \sigma_i^2(t)},$$

   where $\sigma_i(t)$ are the singular values of $\tilde X_t$.
   This is a hard upper bound on $R^2_{\mathrm{pos}}$: if the top-$d$
   PCs only capture some fraction $V^{\mathrm{cap}}$ of $x_t$'s
   variance, the binning can only resolve structure in that fraction,
   so even when the update is a perfect function of $x_t$ the Markov
   ratio is bounded above by $V^{\mathrm{cap}}$.

   The basis-invariant, ceiling-normalized quantity that should be
   compared across $d$ and across layers is the **resolved fraction**:

   $$\mathrm{resolved}(t) = \frac{R^2_{\mathrm{pos}}(t)}{V^{\mathrm{cap}}(t)}.$$

   This answers: of the variance the binning actually resolves, what
   fraction is predicted by binned position? Note that
   $\mathrm{resolved}(t)$ can exceed 1 if the layer update concentrates
   in directions where the position distribution is narrower than
   average — see §5.2 for an example.

### 4.2 Synthetic validation

Before running on the model, we validated the test on three synthetic
regimes whose ground truth is known by construction. All synthetic
cases use $H = 64$, $N = 4000$, $d = 5$, $K = 24$, $n_{\mathrm{shuffles}} = 5$,
matching the structure of the real data on a smaller scale.

| Regime | $x_t$ structure | $\Delta x_t$ rule | Expected outcome |
|---|---|---|---|
| A | $x_t = U\ell + 0.1\eta$ with $\ell \in \mathbb{R}^8$, $U$ orthonormal | $\Delta x_t = Mx_t + 0.05\eta'$ | High R², near-ceiling resolved |
| B | $x_t \sim \mathcal{N}(0, I_H)$ | $\Delta x_t \sim \mathcal{N}(0, I_H)$ independent of $x_t$ | R² at shuffle null |
| C | $x_t \sim \mathcal{N}(0, I_H)$ (high rank) | $\Delta x_t = Mx_t + 0.05\eta'$ | Low absolute R² (ceiling limits), but resolved ≈ ground truth |

Measured outcomes:

| Regime | $R^2_{\mathrm{pos}}$ | ceiling | resolved | shuffle null |
|---:|---:|---:|---:|---:|
| A (low-rank, $\Delta x = Mx$) | 0.570 | 0.896 | 0.636 | 0.006 |
| B (independent) | 0.006 | (varies) | ≈ 0 | 0.006 |
| C (high-rank, $\Delta x = Mx$) | 0.064 | 0.096 | 0.667 | 0.006 |

The first two confirm the obvious cases — strong signal vs no signal.
The third is the case that matters most for interpreting the real
results: even when $\Delta x$ is a deterministic function of $x$, if
$x$'s intrinsic dimension exceeds $d$, the absolute $R^2$ is much
smaller than 1, but the resolved fraction still recovers the underlying
ground truth (≈ 0.67 in both A and C, both reflecting the test's
inherent resolution limits given $K = 24$ clusters and $d = 5$
projection).

The real residual stream has effective rank ~256, so we are
operationally in regime C: the absolute $R^2$ will be small even
when the update is a function of $x_t$, but the resolved fraction is
the right diagnostic to compare against ground truth.

These three cases are embedded as inline assertions in
`drift_welldef_test.py`'s test runner.

---

## 5. Results

### 5.1 Cross-seed result at the final checkpoint

We ran the test on all four trained seeds at training step 24000 (the
final checkpoint), at $d = 5$ and $K = 24$, with $N = 10{,}000$
pilots per seed, $n_{\mathrm{shuffles}} = 20$.

#### 5.1.1 Headline numbers (mean over layer transitions)

| Seed | $R^2_{\mathrm{pos}}$ | shuffle null | ceiling | resolved fraction |
|---:|---:|---:|---:|---:|
| 0 | 0.1009 | 0.0023 | 0.1403 | 0.7193 |
| 1 | 0.0985 | 0.0023 | 0.1507 | 0.6537 |
| 2 | 0.1179 | 0.0023 | 0.1526 | 0.7727 |
| 3 | 0.1004 | 0.0023 | 0.1451 | 0.6921 |
| **mean** | **0.1044** | **0.0023** | **0.1472** | **0.7093** |
| **std** | **0.0090** | **0.0000** | **0.0056** | **0.0500** |
| **relative spread** | **8.7%** | — | **3.8%** | **7.1%** |

Cross-seed agreement on $R^2$ is 8.7% relative. This is comparable to
the Phase 1 cross-seed agreement on basis-invariant statistics (1.1%
on $\lambda$, ~5% on $\log\alpha$, similar levels on others). The
shuffle null is identical across all four seeds at 0.0023 absolute;
the gap above null is ≈ 100σ in all cases, so the statistical
significance is trivial in all configurations.

The Markov ratio $R^2_{\mathrm{pos}}$ is itself a basis-invariant
statistic. It does not depend on which orthonormal basis the seed
happens to learn for the residual stream — both the PCA projection
and the k-means clustering are rotation-equivariant operations on
$x_t$, and the resulting variance ratio is unitless and invariant
under orthogonal change of basis in $\mathbb{R}^H$.

#### 5.1.2 Per-layer breakdown, seed 0

The mean across layer transitions hides substantial layer-to-layer
structure. The full per-layer table for seed 0:

| Transition ($t \to t+1$) | $R^2_{\mathrm{pos}}$ | ceiling | resolved | min cluster size |
|---:|---:|---:|---:|---:|
| 0 → 1   | 0.1416 | 0.3147 | 0.4500 | 8 |
| 1 → 2   | 0.0796 | 0.1437 | 0.5539 | 177 |
| 2 → 3   | 0.0608 | 0.1404 | 0.4330 | 3 |
| 3 → 4   | 0.0620 | 0.1242 | 0.4992 | 3 |
| 4 → 5   | 0.0627 | 0.1054 | 0.5947 | 10 |
| 5 → 6   | 0.0686 | 0.1004 | 0.6833 | 3 |
| 6 → 7   | 0.0399 | 0.1132 | 0.3527 | 3 |
| 7 → 8   | 0.0410 | 0.1003 | 0.4088 | 11 |
| 8 → 9   | 0.0648 | 0.1023 | 0.6334 | 4 |
| 9 → 10  | 0.0802 | 0.1132 | 0.7085 | 10 |
| 10 → 11 | 0.1201 | 0.1180 | 1.0177 | 10 |
| 11 → 12 | 0.2810 | 0.1331 | 2.1112 | 133 |
| 12 → 13 | 0.2073 | 0.2293 | 0.9040 | 202 |

The same pattern is present (with the same magnitudes to within ~5%)
in all four seeds. Particular features:

- **Layer 0 → 1**: highest ceiling ($V^{\mathrm{cap}} = 0.31$).
  The post-embedding state is dominated by token-identity structure;
  the top-5 PCs see a large fraction of that structure.
- **Layers 1 → 9**: low ceiling (~0.10–0.14) and low absolute $R^2$
  (0.04–0.08). Deep interior; binning doesn't resolve much of $x_t$
  and what it does resolve doesn't predict much of $\Delta x_t$
  absolutely. Resolved fraction ranges 0.35–0.71.
- **Layers 10 → 13**: ceiling rises to 0.12–0.23, $R^2$ rises sharply.
  Layer 11 → 12 is the spike: $R^2 = 0.28$ against ceiling 0.13 →
  resolved fraction = 2.1. Layer 10 → 11 also exceeds the ceiling
  (resolved = 1.02).

The seed-2 final-checkpoint result has one notable outlier: layer
$2 \to 3$ has $R^2 = 0.329$ vs ≈ 0.06 in every other seed. This is a
single-layer, single-seed anomaly that we flag but do not have a
mechanistic explanation for. It does not alter the broader picture;
the rest of seed 2's layer profile matches the others.

#### 5.1.3 Cross-seed per-layer dispersion

For comparison purposes in Phase 2, we report the per-layer
cross-seed dispersion of $R^2_{\mathrm{pos}}$ at the final checkpoint,
computed directly from the four-seed values at each layer transition:

| Transition | mean (×100) | std (×100) | relative spread |
|---:|---:|---:|---:|
| 0 → 1   | 13.78 | 0.64  | 4.7% |
| 1 → 2   | 8.48  | 0.57  | 6.8% |
| 2 → 3   | 13.20 | 13.14 | 99.5% (seed 2 outlier) |
| 3 → 4   | 5.60  | 0.41  | 7.3% |
| 4 → 5   | 5.40  | 0.66  | 12.3% |
| 5 → 6   | 4.95  | 1.31  | 26.4% |
| 6 → 7   | 4.70  | 0.62  | 13.2% |
| 7 → 8   | 4.92  | 0.89  | 18.0% |
| 8 → 9   | 5.82  | 0.91  | 15.6% |
| 9 → 10  | 7.50  | 0.77  | 10.3% |
| 10 → 11 | 12.85 | 0.82  | 6.4% |
| 11 → 12 | 28.05 | 0.31  | 1.1% |
| 12 → 13 | 20.50 | 1.11  | 5.4% |

Excluding the seed-2 layer 2→3 outlier (R² = 0.329 in seed 2 vs ≈
0.06 in every other seed), cross-seed relative spread on individual
layer transitions ranges from 1.1% (at the prediction-commitment
layer 11→12) to 26.4% (at deep-interior layer 5→6). The
prediction-commitment layer is the most reproducible single-layer
measurement; the deep-interior layers are the least reproducible.
This is consistent with the deep-interior $R^2$ values being small
in absolute terms, so a fixed-absolute estimator noise of order
$\sim 0.01$ reads as a larger relative number there. The
prediction-commitment layer's exceptional reproducibility (1.1%
relative — matching $\lambda$'s Phase 1 cross-seed agreement) is a
striking feature: every seed converges to essentially the same
absolute prediction-commitment magnitude, even though every seed
learns a different basis for the residual stream.

### 5.2 The layer profile is U-shaped

The per-layer table at §5.1.2 shows a characteristic U-shape that
appears in all four seeds:

- **Layer 0 → 1**: $R^2 \approx 0.14$, the highest of any
  pre-commit-band transition. Post-embedding $\to$ first-block-output.
- **Layers 1 → 9**: $R^2 \in [0.04, 0.09]$. The flat interior.
  Position binning predicts little of the layer update in absolute
  terms; the ceiling is also low.
- **Layers 10 → 13**: $R^2$ rises to 0.13, 0.28, 0.21. The
  prediction-commitment region.

The resolved-fraction profile is more informative than the absolute-
$R^2$ profile in the interior. Resolved fraction sits at 0.35–0.71
across the interior — the binning is mostly explaining what it can
resolve, the ceiling is just low. The interior is not "Markov-broken"
in some absolute sense; it is just that the position information that
predicts updates is a small fraction of the position's overall
variance, so binning on the top-5 PCs only captures part of it.

The resolved-fraction > 1 phenomenon at the late layers is
mechanistically informative and worth dwelling on. At layer $11 \to
12$, $R^2 = 0.28$ but ceiling = 0.13. The Markov ratio exceeds the
fraction of $x_t$'s variance the top-5 PCs see. How is this possible?

Mechanically: the top-5 PCs of $x_t$ are chosen to maximize captured
variance of $x_t$ itself. Some of $x_t$'s variance lies in directions
that contribute little to $\Delta x_t$. Conversely, $\Delta x_t$ may
concentrate along directions where $x_t$'s distribution is *narrower*
than the top-5 PCs see — directions that are not high-variance in
$x_t$ overall, but that strongly determine the update. K-means
clustering, applied to the projection onto $V$'s top-5 vectors, may
nevertheless effectively separate pilots along these high-update-
relevance directions, because they happen to correlate with the
top-5 PCs through some other structure in $x_t$.

A cleaner way to say it: the resolved fraction $R^2/\mathrm{ceiling}$
is the ratio of "fraction of $\Delta x_t$ variance captured" to
"fraction of $x_t$ variance captured," and the ratio can exceed 1
when the update is more efficiently representable in the top-5 PC
basis of $x_t$ than $x_t$ itself is. This is the geometric signature
of *prediction commitment*: the late blocks compute in a
low-dimensional subspace aligned with the dominant output-relevant
directions, and the per-layer update is more concentrated in those
directions than $x_t$'s distribution is.

This is also the same phenomenon the multi-view proposal's "reverse
view" was designed to detect: as the model approaches the unembedding,
its computation contracts onto a low-dimensional output-relevant
subspace. The Markov-ratio test sees this from a different angle —
the late layers' updates are unusually well-explained by binned
position, in the precise sense above.

### 5.3 The dimensionality sweep falsifies the ceiling hypothesis (H_ceiling)

A reasonable initial reading of $R^2 \approx 0.10$ was: the ceiling is
limiting us. The top-5 PCs of $x_t$ don't see enough of the residual
stream's ~256-effective-rank structure to detect the update's
state-dependence, so the absolute $R^2$ is structurally suppressed.
This is H_ceiling: increasing $d$ would raise $R^2$ proportionately,
with the resolved fraction remaining roughly constant.

We tested H_ceiling on seed 0 at the final checkpoint by sweeping
$d \in \{3, 5, 10, 20, 40, 80\}$. Results, mean across layer
transitions:

| $d$ | $R^2_{\mathrm{pos}}$ | ceiling | resolved | gap above null |
|---:|---:|---:|---:|---:|
| 3  | 0.0950 | 0.1121 | 0.8472 | 0.0927 |
| 5  | 0.1009 | 0.1403 | 0.7193 | 0.0986 |
| 10 | 0.1129 | 0.1896 | 0.5955 | 0.1106 |
| 20 | 0.1230 | 0.2561 | 0.4803 | 0.1207 |
| 40 | 0.1155 | 0.3446 | 0.3353 | 0.1132 |
| 80 | 0.1232 | 0.4635 | 0.2657 | 0.1209 |

**H_ceiling is FAIL.** $R^2_{\mathrm{pos}}$ saturates by $d = 5$.
Going from $d = 5$ to $d = 80$ — a 16-fold increase in PCs used to
bin position — moved $R^2$ from 0.101 to 0.123 (an increase of about
22%, well within the noise of how much the ceiling is going up at the
same time). The ceiling, meanwhile, more than tripled in the same
range (0.14 → 0.46). The resolved fraction collapses monotonically
from 0.85 to 0.27 because $R^2$ stops rising while the ceiling keeps
rising.

The reading of this saturation: the position information that
predicts the layer update lives in a roughly 3-dimensional subspace
of $x_t$. PCs 4 through 80 add variance to $x_t$ but no additional
predictive content for $\Delta x_t$. Of the residual stream's
~256-effective-rank structure, only a 3-dimensional corner of it is
update-relevant from a state-Markov standpoint. The rest is
information the model carries but does not act on through the
state-dependent component of its update rule.

The complementary reading is the harder one: the remaining ~88% of
$\Delta x_t$ variance is not a function of $x_t$ in any subspace the
binning resolves. It depends on context that does not project visibly
into the pilot's current residual-stream state. The natural candidate
is attention's read from the left-context KV cache, which depends on
every token before the pilot but is not encoded into $x_t$ in a way
the top-80 PCs can recover. Other candidates: position-in-chunk
information that is in the rotary embeddings but is small in $x_t$
itself; head-specific information that lives in a per-head subspace
not aligned with the top PCs; multi-token interaction structure that
requires the joint distribution over multiple tokens rather than $x_t$
alone.

**Verdict on the absolute claim:** the residual stream is *substantially
non-Markovian in $x_t$* at convergence. The strong reading of the
lines-of-thought SDE — that the per-layer update is, in expectation,
a function of $x_t$ alone — misses about 7/8 of the dynamics.

#### 5.3.1 Per-layer detail at $d = 80$

It is worth tracking how the layer profile changes as $d$ grows. The
per-layer $R^2_{\mathrm{pos}}$ at $d = 80$ (the largest tested):

| Transition | $R^2$ | ceiling | resolved |
|---:|---:|---:|---:|
| 0 → 1   | 0.143 | 0.604 | 0.237 |
| 1 → 2   | 0.164 | 0.582 | 0.281 |
| 2 → 3   | 0.074 | 0.499 | 0.148 |
| 3 → 4   | 0.070 | 0.439 | 0.159 |
| 4 → 5   | 0.078 | 0.418 | 0.186 |
| 5 → 6   | 0.082 | 0.413 | 0.199 |
| 6 → 7   | 0.054 | 0.408 | 0.131 |
| 7 → 8   | 0.055 | 0.391 | 0.141 |
| 8 → 9   | 0.088 | 0.380 | 0.231 |
| 9 → 10  | 0.103 | 0.400 | 0.258 |
| 10 → 11 | 0.145 | 0.434 | 0.334 |
| 11 → 12 | 0.308 | 0.466 | 0.661 |
| 12 → 13 | 0.237 | 0.601 | 0.394 |

The layer profile's *shape* is essentially identical to $d = 5$ — flat
interior, late-layer spike, embedding-layer modest bump. The
*magnitudes* of $R^2$ change very little. The ceiling rises across
the board (which makes the resolved fractions look smaller in
absolute terms) but the position-determined component itself is
saturated.

The late-layer "above-ceiling" phenomenon partially survives the
$d$ increase: at $d = 80$, $R^2 = 0.308$ vs ceiling 0.466 gives
resolved fraction 0.66 — no longer above the ceiling, but still
substantially higher than every other layer's resolved fraction. The
prediction-commitment band remains the most state-determined region
of the network at every $d$.


### 5.4 Training trajectory

The above results are snapshots of the trained model. To understand
where the 12% comes from — whether it is a property of the trained
network's learned function, or of the architecture, or some
interaction of the two — we tracked the Markov ratio across training.
Twelve log-spaced checkpoints between step 100 and step 24000, all
four seeds, $d = 5$ and $K = 24$ as before, $n_{\mathrm{shuffles}} = 5$.

The checkpoint set was selected from the 50 available Phase 1
checkpoints by log-spaced sub-sampling, anchored to include step 100
(early-training baseline), step 24000 (final), and dense coverage of
the steps where the Phase 1 anomalies are known to occur (steps
1000–10000).

#### 5.4.1 Headline result and hypothesis disposition

**$R^2_{\mathrm{pos}}$ decreases monotonically through training.**
Cross-seed mean across layer transitions:

| Step | seed 0 | seed 1 | seed 2 | seed 3 | mean | cross-seed std |
|---:|---:|---:|---:|---:|---:|---:|
| 100   | 0.554 | 0.587 | 0.561 | 0.577 | 0.5697 | 0.0150 |
| 156   | 0.586 | 0.597 | 0.590 | 0.589 | 0.5905 | 0.0047 |
| 274   | 0.530 | 0.535 | 0.535 | 0.542 | 0.5355 | 0.0049 |
| 428   | 0.395 | 0.401 | 0.394 | 0.387 | 0.3942 | 0.0057 |
| 749   | 0.263 | 0.256 | 0.255 | 0.255 | 0.2572 | 0.0039 |
| 1171  | 0.174 | 0.168 | 0.175 | 0.168 | 0.1712 | 0.0038 |
| 2049  | 0.125 | 0.128 | 0.128 | 0.125 | 0.1265 | 0.0017 |
| 3205  | 0.117 | 0.117 | 0.117 | 0.114 | 0.1163 | 0.0015 |
| 5607  | 0.112 | 0.116 | 0.114 | 0.111 | 0.1133 | 0.0022 |
| 8771  | 0.111 | 0.111 | 0.111 | 0.111 | 0.1110 | 0.0000 |
| 15343 | 0.110 | 0.104 | 0.118 | 0.104 | 0.1090 | 0.0066 |
| 24000 | 0.101 | 0.098 | 0.118 | 0.100 | 0.1043 | 0.0093 |

The fall from $\overline{R^2}_{\mathrm{pos}} = 0.590$ at step 156 (the
early-training peak) to $\overline{R^2}_{\mathrm{pos}} = 0.104$ at
step 24000 is a $5.7\times$ reduction. The fall is smooth,
monotonic, approximately log-linear in the rapid-fall window (steps
~200 to ~2000), and asymptotes by step ~3000 to its final value.

Cross-seed agreement is extraordinarily tight. The cross-seed std at
the rapid-fall midpoint (step 749) is 0.0039 absolute, or 1.5%
relative — better than the cross-seed agreement on any Phase 1
basis-invariant statistic except $\lambda$. The cross-seed std at
the final checkpoint is 0.0093 absolute (8.9% relative), driven mostly
by seed 2's slightly elevated value of 0.118 vs the other three's
~0.10. Whatever the training trajectory represents, it is a
reproducible function of training step independent of seed.

**Hypothesis disposition:**

- **H_monotone (FAIL).** $R^2_{\mathrm{pos}}$ does not rise with
  training. It falls.
- **H_hump (FAIL).** No hump or peak appears near step 5000. The
  rapid-fall phase is complete by step ~2000; at step 5607 the
  curve is in its asymptote and shows no inflection.
- **H_flat (FAIL).** $R^2_{\mathrm{pos}}$ is not flat; it varies by
  $5.7\times$ across training.
- **(unanticipated) H_decoupling (PASS).** $R^2_{\mathrm{pos}}$
  decreases monotonically through training, with the decoupling
  asymptote reached well before any other Phase 1 statistic converges.

H_decoupling is the unanticipated alternative: during training, the
layer update transitions from being predominantly state-determined to
being predominantly context-determined. This was not on the
pre-registered hypothesis list, but it is the alternative that
matches the data.

#### 5.4.2 No co-location with Phase 1 training-dynamic events

There is none. The Phase 1 log-α hump peaks at step ~5000 across all
four seeds (range 4483–5014 across seeds; Phase 1 §6.1). The $R^2$
rapid-fall phase is complete by step ~2000 and the curve is in its
asymptote at step 5000. There is no kink, inflection, or visible
feature in $R^2$ at the hump step. The mid-training Σ-distance bump
(steps 5000–10000; Phase 1 §6.5) sits entirely in the $R^2$
asymptote. The post-final-norm anomaly emergence (Phase 1 §6.3),
which reaches near-final magnitude by step ~5000, also sits in the
$R^2$ asymptote.

The trajectory does, however, fit cleanly into a hierarchy of
convergence times during training, which itself is a new observation:

- **Markovianity decoupling** ($R^2_{\mathrm{pos}}$): rapid fall
  complete by step ~2000, asymptote reached by step ~3000.
- **Linear flow geometry** (Phase 1 normalized flow-distance kink,
  §3.3): primary convergence by step ~5000.
- **Phase 1 mid-training anomalies** (log-α hump, post-final-norm
  emergence, Σ-distance bump): all peaked/completed by step ~5000.
- **Eval loss**: still declining slowly through step 24000.

These are four different aspects of the model's structure converging
on different schedules. The earliest-converging geometric property of
the trained model is its *non*-Markovianity in $x_t$ — the network
arranges for $\Delta x_t$ not to be a function of $x_t$ alone before
the other Phase 1 geometric statistics finish converging, and well
before the eval loss does.

#### 5.4.3 Differential decoupling: interior vs commit layer

The three-curve plot in `training_trajectory_seeds_0_1_2_3_d_5_k_24.png`
decomposes the decoupling by layer region. Cross-seed mean values at
the start of training (step 156 peak), at training step 5607
(approximately the Phase 1 log-α hump location), and at the final
step:

| Curve | step 156 | step 5607 | step 24000 | reduction (step 156 → final) |
|---|---:|---:|---:|---:|
| mean over all transitions | 0.590 | 0.113 | 0.104 | $5.67\times$ |
| deep interior (6 → 7) | ≈ 0.63 | ≈ 0.06 | 0.047 | ≈ $13.4\times$ |
| prediction-commit (11 → 12) | ≈ 0.70 | ≈ 0.35 | 0.281 | ≈ $2.5\times$ |

(The intermediate-step per-layer values for individual transitions
are read from the saved
`training_trajectory_seeds_0_1_2_3_d_5_k_24.npz` arrays; the table
above reports values to two-figure precision because they are
extracted from the cross-seed mean in the npz rather than from a
separately-printed log.)

Both curves fall, but at very different rates. The interior decouples
by a factor of ~13; the prediction-commitment layer by a factor of
~2.5. At convergence the prediction-commitment layer retains
approximately six times the absolute state-determinacy of the deep
interior.

The U-shape across layers reported in §5.2 is therefore not present
at initialization. The heatmap
(`training_trajectory_seeds_0_1_2_3_d_5_k_24_heatmap.png`) shows
this directly. At step 100, the entire layer stack is bright green
in the heatmap ($R^2$ in 0.5–0.7 across the column), with only a
modest brightening at the top (the future commit band is already
slightly more state-determined, but only slightly). By step ~2000
the whole stack has dimmed to the 0.10–0.30 range. By step ~5000
the characteristic U-shape — dark interior at $R^2 \approx 0.05$,
brighter commit band at $R^2 \approx 0.30$ — is fully developed and
stable through the rest of training.

So the training trajectory contains two superimposed stories:

1. **Global decoupling.** At every layer, $R^2$ falls during training.
   The network learns to make $\Delta x_t$ depend on something other
   than $x_t$ alone, at every depth.
2. **Differential restructuring.** The decoupling is faster and more
   complete at the interior than at the prediction-commitment layer.
   The U-shape we observe at convergence is what is left after a
   parallel global process is filtered through a layer-dependent
   constraint structure.

The prediction-commitment layer's relative resistance to decoupling
has a mechanistic interpretation: late-stage computation must
ultimately organize itself onto a low-dimensional, output-relevant
subspace (the cross-entropy objective and the existence of an
unembedding force this), and in that subspace the layer update is
strongly determined by the small number of $x_t$ directions that
encode output identity. The interior has no such constraint and
decouples freely.

The deep interior decouples by a factor of ~13 but doesn't go to
zero: $R^2_{\mathrm{pos}} \approx 0.047$ at convergence in the
interior, vs ~0.002 shuffle null. The residual ~0.05 is the
irreducible state-dependence that the MLP and residual addition still
contribute (these blocks are explicit functions of $x_t$ regardless
of what attention does), plus whatever attention output is correlated
with position through indirect channels.

#### 5.4.4 What is the network doing at step 100?

$R^2 = 0.57$ at step 100 reflects the initialization regime. The
model has barely trained; gradients have done little to the weights.
In this regime, attention is essentially uniform-mixing — keys and
queries are roughly random projections, so attention weights are
close to uniform over context, and the attention output at each pilot
is close to a per-pilot mean of the surrounding tokens. This mean is
itself approximately a function of $x_t$ (because $x_t$ is itself a
function of nearby token embeddings, summed by attention), so the
entire layer update is well-approximated by some block-specific map
of $x_t$. The MLP, layer-norm, and residual addition are all
deterministic blocks of $x_t$ in any case. The result: most of
$\Delta x_t$ is a function of $x_t$ at initialization, and $R^2$ is
correspondingly high.

The early-training peak at step 156 (where $R^2_{\mathrm{pos}}$
briefly rises from 0.57 to 0.59 before beginning its decline) likely
reflects the warmup phase of the learning rate schedule. The peak LR
is reached at step 1000 (linear warmup over the first 1000 steps),
so by step 156 the model is in the late-warmup regime where weights
are starting to depart from initialization but the per-block maps
are still roughly state-determined. The slight increase from 0.57 to
0.59 may reflect attention's slight sharpening relative to the
fully-random keys-and-queries baseline.

The rapid-fall window (steps ~200 to ~2000) is where training
becomes serious. The model learns to make attention output depend on
the specific identity, position, and context of left-context tokens —
making $\Delta x_t$ depend on information that is *not* encoded in
$x_t$ alone. $R^2$ falls as a direct consequence. The asymptote at
~10% is what's left when this process is complete.

The plateau onset at step ~2000 is also informative: this is roughly
the step at which the cosine LR schedule starts decaying from peak
LR. The fact that decoupling completes precisely when learning rate
starts decaying — rather than continuing through the rest of
training — suggests that the decoupling is driven by gradient
magnitude (large gradients in the early-training high-LR phase
restructure attention quickly), and that once the LR drops, further
training cannot reverse what was done.

This is consistent with — though not proof of — a "phase transition"
reading of decoupling: the network passes from the Markovian-in-$x_t$
initialization regime to the non-Markovian trained regime relatively
quickly during the high-LR phase, and the resulting non-Markovian
structure is then frozen in place.

---

## 6. Interpretation

### 6.1 The strong Markov reading of the SDE is falsified

The lines-of-thought SDE, read as a generative model for individual
trajectories, requires $\Delta x_t$ to be predominantly a function of
$x_t$ at each layer. Both the dimensionality sweep and the training
trajectory falsify this. The state-determined component captures
~12% of update variance at convergence, and the rest is context that
no amount of finer position binning can recover.

The SDE drift $b(x, t) = [\dot R R^\top + R \dot S R^\top] x$
captures the state-dependent part of the layer update reasonably well
but treats the much larger attention-output contribution as isotropic
Gaussian noise. The empirical Phase 1 result that the noise *is*
approximately Gaussian, isotropic, and exponentially scaling at the
ensemble level (Phase 1 §4.1, §4.4) is consistent with this:
averaged over an unconditioned ensemble, the attention contribution
looks like noise. But within any fixed pilot, the attention
contribution is structured, context-dependent computation — not noise.

The framework is a faithful marginal description and an inaccurate
generative description. This was always going to be the case for any
framework that absorbs attention into noise; the dimensionality sweep
quantifies the gap. The 7-in-8 of dynamics that the SDE drift misses
is the part the SDE noise term is fitting.

### 6.2 The weak Markov reading survives

Three findings constitute the surviving picture at convergence:

- The state-determined component of $\Delta x_t$ (~12%) is real,
  basis-invariantly measurable, and reproducible across four seeds
  to about 8% relative. It is a property of the architecture and
  recipe, not of the specific seed.
- It lives in a 3-dimensional subspace of $x_t$. Whatever the
  network does with state, it concentrates the work along a handful
  of dominant directions. The d-sweep saturation at $d = 3$–$5$ is
  unambiguous.
- The layer profile is U-shaped: state dominates at the embedding
  read-in (layer 0 → 1) and at the unembedding read-out (layers
  11 → 13), context dominates in the interior. The interior is
  where attention's hidden-state contribution is largest; the
  boundaries are where the residual-stream geometry is most
  determined by token identity (input) or output identity
  (prediction commitment).

The late-layer resolved-fraction > 1 phenomenon (layers 10 → 12) is
a clean geometric signature of prediction commitment: the network's
computation contracts onto a low-dimensional, dominant-PC-aligned
subspace as the model approaches the unembedding.

### 6.3 What the action functional now measures

The Onsager–Machlup action

$$S[x] = \frac{1}{2}\sum_t \|\Delta x_t - b(x_t, t)\|^2_{\Sigma_t^{-1}}$$

with $b$ the marginal-flow drift no longer measures "deviation from
the most likely path under the SDE" — it measures something different.
With $b$ accounting for only ~12% of $\Delta x_t$, the action is
dominated by the residual $\Delta x_t - b(x_t, t)$, which is
essentially the attention output at each layer.

This rebrands the action functional from "negative log-density of a
path under the SDE" to **"Mahalanobis-weighted attention work along
the trajectory."** Whether *that* quantity is informative — whether
it correlates with prediction quality, whether it distinguishes
bigrams, whether high-action pilots are interpretable as "the network
worked hard on this example" — is the empirical question for the
modified Test 4 (see §7). The quantity itself is well-defined,
computable with the data we have, and worth measuring on its own
terms regardless of its relationship to the SDE.

Explicitly: per-pilot action under the marginal-flow model is

$$S_k = \frac{1}{2}\sum_{t=0}^{L_{\text{total}} - 2} (\Delta x_t^{(k)} - b(x_t^{(k)}, t))^\top \Sigma_t^{-1} (\Delta x_t^{(k)} - b(x_t^{(k)}, t)),$$

where $\Sigma_t$ is the per-layer noise covariance fitted by the
lines-of-thought framework (Phase 1 §4.1–§4.5). $S_k$ is a scalar per
pilot, basis-invariant (the Mahalanobis weighting makes it so), and
computable from the existing per-pilot activation stacks and the
existing Phase 1 fit parameters. We do not compute it in this
document; we flag it as the next natural per-pilot quantity to
measure.

### 6.4 Training as decoupling: the framework fits the architectural residual

The training-trajectory result sharpens the falsification and changes
its direction. The 88% of $\Delta x_t$ variance that is not
state-determined at the final checkpoint started life as ~43% at
initialization. The gap grew during training. Training actively
decouples the layer update from state, and the decoupling is the
*earliest*-converging geometric feature of the model.

This reframes what the lines-of-thought SDE is fitting. It is not a
description of what the trained transformer does; it is a description
of what training could not erase. At initialization, the model is
substantially Markovian in $x_t$ ($R^2 \approx 0.57$). Training
reduces this Markovianity by ~$5.7\times$, asymptoting at $R^2
\approx 0.10$ — the residual state-dependence that the architecture's
MLP, residual connections, and norm structure mechanically contribute
regardless of what the attention weights have learned to do. The
SDE's drift fits this asymptote. The noise term hides everything
training added.

This is a stronger statement than the dimensionality sweep alone
supports. The dimensionality sweep established that the SDE misses
~88% of the per-pilot dynamics at convergence. The training trajectory
establishes that the missing component is *learned* — it is the part
of $\Delta x_t$ that training constructs by making attention
selective. The framework's macro description is dominated by the
architecturally-determined residual, not by the learned function.
The blunt summary: **the framework describes the un-trained part of
the trained model.**

#### 6.4.1 Consistency with Phase 1's cross-seed/basis indeterminacy finding

This reading is consistent with — and explains — a known feature of
the Phase 1 results. The basis-invariant macro statistics ($\lambda$,
$\log\alpha$, effective rank, kurtosis, $R(t)$ trajectory geometry)
are reproducible across seeds to about 1% relative for $\lambda$ and
similar levels for others (Phase 1 §4.1–§4.4), even though
seed-specific bases are mutually orthogonal at every layer (Phase 1
§7.5).

This pairing — agreement on basis-invariants, disagreement on bases —
is unusual. If the basis-invariant statistics described the *learned
function*, cross-seed reproducibility would require each seed to
learn essentially the same function in different coordinates — a
strong claim that would imply the learned function lives in a small,
seed-independent equivalence class of computations expressed in
different bases.

The H_decoupling reading explains the pairing without invoking that
strong claim. If the basis-invariant statistics describe the
*architectural residual* — the part of the model's per-layer behavior
that follows mechanically from the MLP, residual addition, and norm
structure — then cross-seed reproducibility is expected: every seed
has the same architecture, so the architectural residual is the
same. The bases differ because the learned function (which lives in
the part the framework doesn't see) varies across seeds. The Phase 1
"agreement on invariants, disagreement on bases" pattern becomes
the empirical signature of the architecture vs function split.

#### 6.4.2 The Phase 1 training-dynamic anomalies are architectural-residual phenomena

The Phase 1 mid-training anomalies — the log-α hump centered at step
~5000 (Phase 1 §6.1), the post-final-norm anomaly emergence
completing at step ~5000 (Phase 1 §6.3), the mid-training Σ-distance
bump in steps 5000–10000 (Phase 1 §6.5), the boundary-layer effect
(Phase 1 §5) — also fit the H_decoupling reading. They are
restructurings *inside the architectural residual*, all completed by
step ~5000. The $R^2$ decoupling is similarly an architectural-
residual phenomenon, completed by step ~3000.

None of these are co-located with each other in the strong sense
(same step, same shape), but they all occupy the same broad
"phase 2 of training" window (steps ~2000–~10000) where the
architectural-residual geometry is undergoing its primary settling.
They are different facets of the same overall process — the
architecture's residual settling into its trained configuration —
visible from different statistical angles and on different
timescales.

#### 6.4.3 The function the network actually computes is in the noise

The function the network actually computes — the input-context-
to-output-distribution mapping that makes the trained transformer
useful — is in the context-dependent part of $\Delta x_t$ that the
framework treats as noise. This is the part that *was* added by
training, and that varies across seeds (because different seeds
learn different bases for the same equivalence class of functions).
Characterizing *that* is what the multi-view program is for.

The framework's strength is that it cleanly isolates the
architectural residual, which is the noise-free part of the per-layer
dynamics. Its weakness is that it provides no mechanism for
characterizing what is in the noise. That is the gap multi-view
fills.

---

## 7. Implications for the multi-view program

This work was motivated by a larger project: the **multi-view
decomposition** of the residual-stream ensemble, in which the all-to-
all marginal is partitioned by input-token identity (forward view),
successor-token identity (reverse view), or both (pair-conditional
view). The proposal flagged five experimental tests centered on
characterizing how the conditional bundles' geometry differs from the
marginal bundle's geometry. The Markovianity result reshapes that
program.

For reference, the original tests were:

1. **Drift Markovianity (this work).** Is the per-layer update a
   function of $x_t$? Done. Answer: 12% yes at convergence, 57%
   at initialization, decoupling during training.

2. **Within-cell noise structure.** Inside a fixed pair-conditional
   cell $E_{v,w}$, is the residual $\Delta x_t - \bar{\Delta x_t}^{(v,w)}$
   Gaussian, isotropic, and exp-scaling — i.e., does the framework's
   noise envelope generalize to conditionals?

3. **Conditional mean vs marginal drift.** Does $\bar x_{v,w}(t)$
   track the marginal-flow drift integrated forward from $E(v)$,
   or deviate systematically?

4. **Action distribution.** Compute per-pilot Onsager–Machlup action
   under the marginal-SDE model; ask whether its distribution is
   informative about prediction quality.

5. **Onsager–Machlup bridges.** Solve the most-likely-path problem
   between fixed endpoints in closed form (the SDE drift is linear,
   so this is a Kalman smoother) and compare to the empirical
   pair-conditional mean.

The H_decoupling result requires reshaping each of these:

### 7.1 Test 2 (within-cell noise structure)

Still well-defined and worth running, but its result has different
stakes. If within-cell noise is non-Gaussian, that is now *expected*
— we know the noise term in the SDE is attention output, not Gaussian
noise. The interesting question becomes whether the noise structure
is *consistent across cells* or *cell-specific*. Cell-specific noise
structure would mean attention's bigram-conditioned contribution
lives in cell-specific directions of the residual stream — i.e., the
bigram determines not just where the trajectory ends up but also
*along which directions* it deviates from the marginal flow.

This is a sharper question than the original framing. It is
answerable with existing data (multi-view activations include the
metadata required to filter by bigram) at modest sample sizes — the
~30–50 instances per cell of common bigrams is enough to measure
within-cell covariance robustly, though not at high enough
sample size to measure higher-order statistics like cell-specific
kurtosis.

### 7.2 Test 3 (conditional mean vs marginal drift)

Now the most important remaining test. We expect the conditional
means to deviate from the marginal drift, and we now know the
deviation *is* the learned function — the part of the model the
framework cannot see. The quantity to measure is

$$d_{v,w}(t) = \bar x_{v,w}(t) - \int_0^t b(x, s)\, ds \bigg|_{x_0 = E(v)},$$

the cell-conditional deviation from the input-marginal-flow
prediction. The question is whether the matrix
$\{d_{v,w}(t)\}_{(v,w) \in P, t}$, stacked across many bigram cells
$P$ into a tensor and decomposed by SVD, has low effective rank in
the across-cells direction.

- **Low effective rank** would say attention's bigram-conditioned
  contribution is organized along shared axes: a small number of
  "decoupling directions" in $\mathbb{R}^H$ are reused across many
  bigrams, with cell-specific magnitudes. This would be the structural
  result the multi-view paper would be built around.
- **Full rank** would say each bigram drives the residual stream
  along its own private direction. The learned function is
  idiosyncratic at the bigram level, and the multi-view decomposition
  reveals an unstructured per-bigram contribution rather than a
  small dictionary of shared axes.

Either answer is publishable. The former is the more useful result;
the latter is the more surprising one.

### 7.3 Test 4 (action distribution)

Worth running, reinterpreted as measuring "attention work along the
trajectory" (per §6.3). Particular questions:

- **Does per-pilot action $S_k$ correlate with per-pilot cross-
  entropy at the lm_head?** A positive correlation would say
  "pilots that take expensive paths through the residual stream are
  also the ones the model predicts poorly," tying the geometric and
  prediction-quality stories together.
- **Does within-cell action distribution have heavy tails?** A small
  number of pilots requiring unusually large attention work would
  identify "hard" examples within a cell, distinct from the typical
  pilots in the cell.

Under H_decoupling there is also a training-trajectory version of
this test: does mean action grow during training in inverse
proportion to $R^2$'s fall? They are arithmetically related by
$\langle S \rangle \propto V_{\mathrm{total}}(1 - R^2)$, so the
existence of such growth is not informative — but the *shape* of the
per-pilot distribution evolving across training (e.g., growing
heavy-tailedness, growing concentration in a few pilots) would tell
us whether "attention work" becomes more or less uniform across
pilots as training progresses.

### 7.4 Test 5 (Onsager–Machlup bridges)

Deprioritized. Solving the bridge problem for an SDE that explains
only ~12% of the dynamics doesn't tell us much about the rest. The
bridges might still be useful as a baseline computation inside
Test 3 — they answer "what does the marginal flow predict the cell-
conditional mean to be?" — but they are not a separate test.

### 7.5 The reshaped multi-view program

The multi-view program asks "how does the bundle partition under
input and output conditioning?" That question is unchanged. What
changes is the framing of the *answer*: rather than "the SDE drift
and noise have conditional structure," it is now "the state-determined
and context-determined components of the layer update have completely
different status — one is architectural residual that training cannot
remove, the other is the learned function — and the multi-view
decomposition is the lens that sees the latter."

The reshaped test ordering, in priority:

1. Test 3 in its modified form (cross-cell low-rank structure in
   deviations from marginal flow). Single biggest result available.
2. Test 4 in its training-trajectory form (per-pilot action shape
   across training). Independent of Test 3, cheap to run, gives a
   different angle on the H_decoupling story.
3. Test 2 in its modified form (cross-cell consistency vs
   cell-specificity of noise structure). Smaller question but
   answerable with what we have.
4. Test 5 only as a baseline inside Test 3.

---

## 8. Open questions and limitations

### 8.1 Open questions

**What is the architectural residual asymptote in other settings?**
$R^2_{\mathrm{pos}} \approx 0.10$ is the asymptote in our setting (a
150M-parameter Llama-style GELU model trained for 24k steps on
FineWeb-Edu). Is the same number reached in other architectures, at
other scales, after other recipes? The Phase 2 architecture sweep
provides natural variation (SwiGLU vs GELU, varying depth, varying
width); running this test there is straightforward and would tell us
whether 0.10 is universal to the Llama family or specific to our
configuration.

**Does the decoupling rate depend on data?** The $R^2$ fall is
complete by step ~3000, approximately when the model has seen ~200M
tokens. Is the trajectory data-quantity-determined, step-determined,
or LR-schedule-determined? The plateau onset co-locates with the
peak-LR step, which suggests the LR schedule plays a role — but this
is not directly tested. A learning-rate-schedule variant of the
experiment (constant LR, or shorter warmup, or longer constant-LR
phase before decay) would help disentangle these.

**Does the decoupling shape depend on training-recipe choices?**
Depth and width sweeps would provide natural variation. The cleanest
test is to ask whether $R^2_{\mathrm{pos}}(\mathrm{step})$ collapses
onto a common curve when steps are renormalized by some architecture-
dependent scale (e.g., wall-clock time, or token count, or some
gradient-norm-based clock).

**Is the late-layer resolved-fraction > 1 phenomenon scale-invariant?**
The 11 → 12 transition has resolved fraction > 1 at $d = 5$, and the
phenomenon survives partially (resolved fraction 0.66, still highest
in the network) at $d = 80$. Does this prediction-commitment
signature appear in all transformer architectures, or is it specific
to architectures with tied embeddings and an unembedding head that
strongly biases the late layers?

**What is the role of attention specifically?** We attribute the
decoupling to attention becoming selective, but this is an inference
from the mechanics of the update rule rather than a direct
measurement. A direct test would zero-ablate attention in a trained
model and re-run the test; we would predict $R^2_{\mathrm{pos}}$
would rise sharply, but how sharply is empirically open.

**Does the seed-2 layer 2 → 3 spike reflect a real per-seed
difference?** The seed-2 final-checkpoint $R^2$ at layer 2 → 3 is
0.329 vs ≈ 0.06 in every other seed. This is a single-layer,
single-seed anomaly that we flag but do not have a mechanistic
explanation for. Inspecting seed 2's training-time evolution at
layer 2 → 3 specifically might reveal whether this is a late-
training drift or a feature present from earlier checkpoints.

### 8.2 Limitations

**Single architecture.** All results are on the 150M Llama-style GELU
model. Generalization beyond is open.

**Single recipe.** All seeds trained with identical hyperparameters
(same LR schedule, same data, same context length). The decoupling
shape may be recipe-specific.

**No within-cell Markov test.** We test Markovianity on the all-to-
all marginal. The natural follow-up — within a fixed (input, output)
cell, how Markovian is the update? — is not run. We expect it to be
much higher (within a fixed cell, much of the "context" that drives
non-Markovianity is held constant), but the magnitude is not
established empirically.

**No direct attention-output ablation.** The mechanistic story (the
non-Markovian component is attention) is inferred from the
mathematics of the transformer update rule rather than directly
measured by running the test on an attention-ablated model.

**Sample size for late-layer commit-band analysis.** The above-
ceiling phenomenon at layer 11 → 12 is robust across seeds, but the
interpretation (that the update concentrates in a low-dimensional
output-relevant subspace) would benefit from per-token or
per-cell analysis to confirm the subspace alignment with output
direction. With 10,000 pilots and 32,768 vocabulary, per-token
sample sizes are small.

**Cross-seed dispersion driven by single seed at final checkpoint.**
The 8.7% cross-seed relative spread on final-checkpoint
$R^2_{\mathrm{pos}}$ is driven mostly by seed 2 (0.118 vs ~0.10 in
the other three). The cross-seed spread at intermediate checkpoints
is much tighter (1.5% at step 749). If seed 2's final-checkpoint
value is treated as a single-seed late-training drift (similar to
the kurtosis-rise pattern in Phase 1 §6.4), the effective cross-seed
spread on the trained-asymptote is closer to 2% relative.

**K-means initialization sensitivity.** We use a single
seed-deterministic k-means run per measurement (no multi-restart).
Restart sensitivity is not measured. Given the tight cross-seed
agreement on $R^2_{\mathrm{pos}}$ (using different model-seed-driven
k-means seeds), restart sensitivity is likely small, but this is not
verified.

### 8.3 Validation that this work specifically would benefit from

**Reproducibility test.** Rerun the cross-seed final-checkpoint test
on a different held-out chunk set (different pilot draw). This
verifies that the cross-seed agreement is not artifact of the
particular held-out chunks.

**Effect of $K$ (cluster count).** A $K$-sweep on seed 0 at the
final checkpoint, varying $K \in \{8, 16, 24, 48, 96\}$. Expected:
$R^2_{\mathrm{pos}}$ rises with $K$ until per-cluster sample sizes
become too small. The plateau-$K$ is itself informative.

**Test on a fully-untrained model.** Rerun the test on the
randomly-initialized model (without any training). Expected:
$R^2_{\mathrm{pos}}$ slightly higher than at step 100, since gradient
updates have done literally nothing. This would establish the
"step 0" point of the trajectory more precisely.

---

## 9. Summary

### 9.1 Headline findings

1. **At convergence, the residual stream is substantially non-Markovian
   in $x_t$.** Position binning predicts ~12% of layer-update
   variance; the dimensionality sweep shows this is not a ceiling
   effect — the position-information that predicts updates lives in
   a 3-dimensional subspace, and adding more PCs adds variance to
   $x_t$ without adding update-predictive content.

2. **The layer profile is U-shaped.** State-determinacy is highest at
   the embedding read-in (layer 0 → 1) and the prediction-commit
   read-out (layers 10 → 13); it is lowest in the deep interior
   (layers 1 → 9). The late-layer band exhibits resolved fraction
   > 1 (the layer update is more concentrated in dominant-PC
   directions than $x_t$'s distribution is), a clean geometric
   signature of prediction commitment.

3. **Training actively decouples $R^2_{\mathrm{pos}}$.** The Markov
   ratio falls $5.7\times$ from initialization to convergence,
   asymptoting at step ~3000. This is the earliest-converging
   geometric property of the trained model — earlier than the linear
   flow geometry (~step 5000), earlier than the Phase 1 anomalies
   (~step 5000), well earlier than eval loss (still declining at
   step 24000).

4. **No co-location with Phase 1 training-dynamic events.** The
   $R^2$ rapid-fall phase is complete before the log-α hump and the
   Σ-distance bump. The hypothesis of a single mid-training
   restructuring event visible from all geometric angles is not
   supported; the geometric properties of the model converge on
   distinct timescales.

5. **The framework describes the architectural residual.** The
   lines-of-thought SDE's basis-invariant statistics are reproducible
   across seeds because they measure the architecture's mechanical
   contribution to the per-layer update, which is the same across
   seeds. The learned function lives in the context-dependent part of
   $\Delta x_t$ that the framework treats as noise; this is the part
   training adds, and the part where the seeds diverge.

### 9.2 What this means for the multi-view program

The multi-view decomposition was originally framed as a refinement of
the lines-of-thought framework — a sharper instrument that would
reveal conditional structure inside the bundle. The H_decoupling
result reframes it as the analytic successor. The framework fits the
architectural residual; multi-view fits what training added.

The reshaped test priority list:

1. **Cross-cell low-rank structure in deviations from marginal flow**
   (modified Test 3 of the original proposal). Stack the cell-
   conditional deviations $d_{v,w}(t)$ as a tensor across many bigram
   cells and decompose by SVD. The effective rank in the across-cells
   direction tells us whether the learned function uses shared axes
   or per-bigram private directions.

2. **Per-pilot action distribution evolution across training**
   (modified Test 4). Tracks how the per-pilot "attention work"
   distribution shape changes as the model decouples.

3. **Cross-cell consistency vs cell-specificity of noise structure**
   (modified Test 2). Smaller question, answerable with what we have.

4. Onsager–Machlup bridges only as a baseline computation inside
   Test 3.

### 9.3 What this work cost

| Component | Time |
|---|---|
| Designing and implementing the test | ~3 days (in conversation) |
| Synthetic validation | ~1 day |
| Cross-seed final-checkpoint runs (4 seeds × 8 min) | ~30 min |
| Dimensionality sweep (seed 0, 6 d-values) | ~1 hour |
| Cross-seed training trajectory (4 seeds × 12 checkpoints) | ~7 hours |
| Plotting, analysis, this writeup | ~1 day |
| **Total** | **~5 days plus ~9 hours compute** |

The compute cost is negligible compared to a Phase 1 training run
(~12 hours per seed × 4 seeds = ~48 hours of training compute to
produce the checkpoints; this analysis runs in ~9 hours and uses no
GPU). The whole investigation was conducted on already-trained
checkpoints, requires no model retraining, and could be repeated for
any newly trained model with the same multi-view augmented activation
collection in roughly the same compute budget.

### 9.4 What the next phase will cost

Modified Test 3 (cross-cell low-rank deviation analysis) requires:

- Stacking $d_{v,w}(t)$ across ~50–200 bigram cells. The data
  needed is already in the multi-view augmented activation files.
- Computing the marginal-flow integration $\hat\mu_v(t)$ for each of
  the ~50–200 input tokens $v$. This requires the Phase 1 fit
  parameters ($R(t)$, $\Lambda(t)$) which are already saved.
- SVD of a tensor of shape (n_cells, n_layers, $H$). For n_cells ≈
  100 this is a 100 × 14 × 896 tensor, well within standard memory
  budgets.
- Plotting, analysis, and writeup.

Estimated cost: ~2–3 days of work, ~1 hour of compute.

Modified Test 4 (per-pilot action distribution training trajectory)
requires:

- Computing per-pilot action $S_k$ at each of the 12 checkpoints,
  for each of 4 seeds, on the existing augmented activation files.
  This is ~40k pilot-checkpoint pairs to score.
- Joining per-pilot $S_k$ with per-pilot cross-entropy at the lm_head
  (also already available in the multi-view augmented activations).
- Plotting action distribution shape across training, scatter plots
  of $S_k$ vs cross-entropy, etc.

Estimated cost: ~2 days of work, ~3 hours of compute.

Together, the immediate next phase of empirical work should take ~1
week and contribute substantively to the multi-view paper draft.

---

## Appendix: artefacts

The scripts and saved data produced during this investigation
(filenames retain the original "drift_welldef" naming for backward
compatibility with on-disk outputs, even though the prose has moved
to the "Markovianity" framing):

### Scripts

- `drift_welldef_test.py` — single-checkpoint test, with $d$-, $K$-,
  and seed-parameterized runs. Produces a two-panel plot (absolute
  $R^2$ with ceiling, ceiling-normalized resolved fraction) and an
  npz of the raw per-layer arrays. The three synthetic validation
  cases are embedded as inline assertions and were used during
  development to verify the test discriminates correctly.

- `drift_welldef_training.py` — multi-checkpoint test across a
  log-spaced subsample of training, single seed or cross-seed.
  Produces a 3-curve plot (mean / commit-layer / interior-layer $R^2$
  vs training step), a layer-by-step heatmap, and an npz of the
  cross-(seed, step, layer) tensor.

### Saved data

For seed $S \in \{0, 1, 2, 3\}$ and training step $\mathrm{STEP}$,
the precomputed augmented activation file path is:

`phase1_runs_gelu/multiview/seed_{S}/augmented_step_{STEP:08d}.npz`

These are produced by `multiview.collect_activations_with_metadata`
and contain per-layer hidden states plus per-pilot metadata
(`states`, `input_ids`, `next_ids`, `pred_ids`, `positions`).

Test outputs are written to `phase1_runs_gelu/drift_welldef/`:

- `seed_{S}_step_{STEP:08d}_d_{D}_k_24.npz` — single-checkpoint test
  output. Contains arrays `layer_from`, `r2_pos`, `r2_pos_shuffle_mean`,
  `r2_pos_shuffle_std`, `x_var_captured`, `v_within`, `v_between`,
  `v_total`, `mean_cluster_size`, `min_cluster_size`, plus a `config`
  array recording the run parameters. The d-sweep runs use the same
  schema with varying $d$ in the filename.
- `seed_{S}_step_{STEP:08d}_d_{D}_k_24.png` — single-checkpoint
  two-panel plot.
- `training_trajectory_seeds_{S1}_{S2}_{S3}_{S4}_d_5_k_24.npz` —
  cross-seed training trajectory output. Contains `steps`, `seeds`,
  and three tensors `r2`, `null`, `ceiling` of shape (n_seeds,
  n_steps, n_transitions).
- `training_trajectory_seeds_{S1}_{S2}_{S3}_{S4}_d_5_k_24.png` —
  three-curve trajectory plot.
- `training_trajectory_seeds_{S1}_{S2}_{S3}_{S4}_d_5_k_24_heatmap.png`
  — layer-by-step heatmap.

### Configuration of the runs reported in this document

- §5.1 single-checkpoint cross-seed: $d = 5$, $K = 24$,
  $n_{\mathrm{shuffles}} = 20$, step = 24000, all four seeds.
- §5.3 d-sweep: $d \in \{3, 5, 10, 20, 40, 80\}$, $K = 24$,
  $n_{\mathrm{shuffles}} = 20$, step = 24000, seed = 0 only.
- §5.4 training trajectory: $d = 5$, $K = 24$, $n_{\mathrm{shuffles}}
  = 5$, 12 log-spaced checkpoints from step 100 to step 24000, all
  four seeds.

All runs use $N = 10{,}000$ pilots per seed per checkpoint, sampled
from the 500 held-out chunks at 19 pilot positions per chunk plus
additional pilots from the held-out expansion (matching the multi-
view augmented activation collection).
