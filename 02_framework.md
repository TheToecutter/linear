## 2. Framework and three-view decomposition

This section formalizes the basis-invariant framework that the paper
extends, defines the three views of the residual-stream ensemble,
states the variance decomposition that constrains them, specifies the
per-view basis-invariant statistics we will report, describes the
token-set selection procedure, and describes the orthogonal Procrustes
construction we use in §6 for cross-seed alignment. The model
architecture, training recipe, and analysis pipeline are deferred to
§3.

### 2.1 Notation and the residual-stream ensemble

Consider a decoder-only transformer with $L$ residual-stream blocks
and hidden width $H$. For an input chunk of tokens $\mathbf{c} = (c_1,
c_2, \ldots, c_T)$ drawn from the training corpus and a position $p$
within the chunk that we designate as a *pilot position*, the residual
stream produces a sequence of hidden states. We index these states by
a *layer-state index* $t \in \{0, 1, \ldots, L+1\}$ with the convention:

- $t = 0$: the *post-embedding state*, obtained by looking up the
  token embedding $E(c_p) \in \mathbb{R}^H$ and (for architectures
  using rotary or other absolute position encoding) applying the
  positional transformation.
- $t = 1, \ldots, L$: the *post-block-output states*. State $t$ is the
  output of block $t$ after the residual stream has passed through
  $t$ attention-plus-MLP blocks.
- $t = L + 1$: the *post-final-norm state*, obtained by applying the
  final RMSNorm (or LayerNorm) to state $L$. This is the state from
  which the model produces logits, typically by multiplying by the
  unembedding matrix (which in tied-embedding architectures is $E^\top$).

We write $x_t(\mathbf{c}, p) \in \mathbb{R}^H$ for the residual-stream
state at layer-state index $t$, chunk $\mathbf{c}$, position $p$. The
total number of layer states is $L_{\text{total}} = L + 2$. For our
$L = 12$ architecture, $L_{\text{total}} = 14$ and $t$ ranges from $0$
to $13$.

This indexing convention adds two states the original framework
typically omits — the post-embedding state $t = 0$ and the
post-final-norm state $t = L + 1$. Both states will turn out to exhibit
distinctive basis-invariant behavior that the inner-layer fit
($t = 2, \ldots, L$) misses; we make the boundary-layer effect explicit
in §3.6 and use the 14-state convention throughout. The original
framework typically indexes only blocks, which corresponds to $t = 1,
\ldots, L$ in our convention.

The framework collects $x_t(\mathbf{c}, p)$ for a fixed collection of
pilot positions across a held-out evaluation set $\mathcal{D}_{\text{eval}}$
consisting of $C$ chunks of length $T$, producing $N = C \times P$
residual-stream activation vectors per layer state, where $P$ is the
number of pilot positions used per chunk. In our experimental setup,
$C = 500$, $T = 1024$, and $P = 19$ (with pilot positions chosen at
$\{50, 100, 150, \ldots, 950\}$ within each chunk), giving $N = 9{,}500$
pilot activations per seed per checkpoint. We index the pilots by $i =
1, \ldots, N$ and use $x_t^{(i)} \in \mathbb{R}^H$ to denote the
residual-stream state of pilot $i$ at layer state $t$.

We refer to the resulting collection $\{x_t^{(i)}\}_{i=1}^N$ as the
*all-to-all ensemble* at layer $t$, with no conditioning on input,
context, position, or output. This is the object that the original
basis-invariant framework characterizes. In our extension we will also
work with *conditional ensembles* defined by restricting to subsets of
$\{1, \ldots, N\}$ that share some conditioning attribute.

For any subset $S \subseteq \{1, \ldots, N\}$ and layer state $t$,
denote by $\bar{x}_t(S) = \frac{1}{|S|}\sum_{i \in S} x_t^{(i)}$ the
mean activation of the subset. The *centered* activations are $y_t^{(i)}
= x_t^{(i)} - \bar{x}_t(S)$ for $i \in S$. Throughout this paper, all
SVDs are computed on the centered activations of the subset under
consideration; the per-coordinate variance is the variance of the
centered activations. Centering matters for the framework: the linear
flow $R(t)$ is recovered from the centered SVD, not the raw SVD, so
that the constant offset (the per-layer ensemble mean) does not
contaminate the principal directions.

### 2.2 The marginal framework's basis-invariant statistics

For the all-to-all ensemble at layer $t$, the framework computes the
following basis-invariant statistics.

**Per-layer activation matrix.** Let $X_t \in \mathbb{R}^{N \times H}$
be the matrix whose $i$-th row is $x_t^{(i)\top}$. The centered matrix
is $\tilde{X}_t = X_t - \mathbf{1}_N \bar{x}_t^\top$ where
$\mathbf{1}_N$ is the column vector of $N$ ones.

**Per-layer covariance and SVD.** The empirical per-layer covariance is
$\Sigma_t = \tilde{X}_t^\top \tilde{X}_t / N \in \mathbb{R}^{H \times H}$.
The framework computes the centered SVD $\tilde{X}_t = U_t S_t V_t^\top$
with $U_t \in \mathbb{R}^{N \times H}$, $S_t = \text{diag}(s_{t,1},
\ldots, s_{t,H})$ with $s_{t,1} \geq s_{t,2} \geq \cdots \geq s_{t,H} \geq 0$,
and $V_t \in \mathbb{R}^{H \times H}$. The recovered linear flow $R(t)$
is defined to be $V_t$ — the matrix whose columns are the right singular
vectors of the centered activations, equivalently the eigenvectors of
$\Sigma_t$ ordered by eigenvalue. The per-coordinate residual variance
(in the basis $R(t)$) is the squared singular value $\sigma^2_{t,k} =
s_{t,k}^2 / N$ for $k = 1, \ldots, H$.

**Linear flow between layers.** For each pair of layers $(t, t+\tau)$
with $\tau \geq 1$, the framework computes the linear-flow prediction
of layer $t + \tau$ from layer $t$:

$$\tilde{x}(t + \tau) = R(t + \tau) \Lambda(t, \tau) R(t)^\top x(t)$$

where $\Lambda(t, \tau)$ is a diagonal matrix whose $k$-th entry is the
square root of $\sigma^2_{t+\tau, k} / \sigma^2_{t, k}$, capturing how
the $k$-th principal direction's variance evolves between layers. The
residual $w(t, \tau) = x(t + \tau) - \tilde{x}(t + \tau)$ has empirical
per-coordinate variance that the framework fits with the power law

$$\sigma_w^2(t, \tau) \approx \alpha \tau^\lambda.$$

The fit is performed in log-log space: $\log \sigma_w^2 = \log\alpha +
\lambda \log\tau$, with $\log\alpha$ and $\lambda$ the recovered fit
parameters.

**Variance-fit convention.** Two conventions exist for how to summarize
the per-coordinate residual variance into a single scalar before fitting.
The original paper uses the *mean of log per-coordinate variance*:

$$\langle \log \sigma_w^2 \rangle_{\text{paper}}(t, \tau) = \frac{1}{H} \sum_{k=1}^H \log \sigma_{w,k}^2(t, \tau).$$

We additionally report the alternative *log of mean per-coordinate
variance*:

$$\langle \log \sigma_w^2 \rangle_{\text{ours}}(t, \tau) = \log \left( \frac{1}{H} \sum_{k=1}^H \sigma_{w,k}^2(t, \tau) \right).$$

By Jensen's inequality, $\langle \log \sigma_w^2 \rangle_{\text{paper}}
\leq \langle \log \sigma_w^2 \rangle_{\text{ours}}$, with equality when
all per-coordinate variances are equal. The two conventions differ by a
small positive offset at convergence (approximately $+0.016$ for
$\lambda$ and $+0.034$ for $\log\alpha$ in our measurements) and have
identical cross-seed dispersion. The Jensen-gap offset is approximately
$0.50$ at initialization and decays to $\approx 0.03$ by training's
end, reflecting that the per-coordinate variance becomes increasingly
uniform as training progresses. Throughout this paper we report both
conventions where the choice matters, and use "paper convention" and
"our convention" to disambiguate.

**Effective rank profile.** The effective rank at layer $t$ is

$$r_{\text{eff}}(t) = \frac{\left(\sum_{k=1}^H s_{t,k}^2\right)^2}{\sum_{k=1}^H s_{t,k}^4}.$$

This is the participation ratio of the squared singular value spectrum;
it equals $H$ when all singular values are equal (perfectly isotropic
spectrum) and equals 1 when only one singular value is nonzero (rank-1
spectrum). For a Gaussian distribution with isotropic covariance,
$r_{\text{eff}}$ equals $H$ in expectation. We report $r_{\text{eff}}(t)$
at each layer state and its cross-seed dispersion.

An equivalent and commonly-used alternative definition is the
exponential of the entropy of the normalized squared singular value
distribution:

$$r_{\text{eff, alt}}(t) = \exp\left( -\sum_{k=1}^H p_{t,k} \log p_{t,k} \right), \quad p_{t,k} = \frac{s_{t,k}^2}{\sum_{k'} s_{t,k'}^2}.$$

The two definitions agree qualitatively but not quantitatively; we use
the participation-ratio definition throughout for consistency with our
prior work and with the implementation in the released code.

**Per-coordinate excess kurtosis.** For each principal direction $k$
at layer $t$, the projected centered activations $y_{t,k}^{(i)} =
(V_t^\top y_t^{(i)})_k$ form a 1-dimensional sample. The excess
kurtosis of this sample is

$$\kappa_{t,k} = \frac{\mathbb{E}[(y_{t,k} - \bar{y}_{t,k})^4]}{(\mathbb{E}[(y_{t,k} - \bar{y}_{t,k})^2])^2} - 3.$$

We report the per-layer mean $\bar{\kappa}_t = \frac{1}{H} \sum_k
\kappa_{t,k}$ and the global mean $\langle\kappa\rangle = \frac{1}{L_{\text{total}}}
\sum_t \bar{\kappa}_t$. Both signed-mean and absolute-mean conventions
are computed; in our data they agree to four significant figures because
all per-coordinate kurtoses are positive (no negative-kurtosis coordinates
in any seed at any layer at convergence).

**Isotropy profile.** The isotropy at layer $t$ is the standard
deviation of the per-coordinate log-variances:

$$\text{iso}(t) = \text{std}_k\left( \log \sigma_{t,k}^2 \right).$$

Small isotropy values indicate that the principal-direction variances
are similar (near-isotropic distribution); large values indicate strong
anisotropy with a few directions of high variance dominating. We report
the per-layer isotropy and the global mean across layers.

**Successive-layer angle profile.** The angle between $R(t)$ and
$R(t+1)$ measures how much the principal-direction basis rotates
between consecutive layers. For each pair of consecutive layers, we
compute the principal angles between the top-$k$ columns of $R(t)$ and
$R(t+1)$ for $k = 10$ (the top-10 principal subspace) and report the
mean of these angles. The per-layer-transition angle profile is highly
reproducible across seeds and exhibits a distinctive pattern with large
rotations at the boundary transitions ($t = 0 \to 1$ and $t = L - 1 \to L$)
and a roughly constant interior.

We assume readers are familiar with the framework's standard derivations
or have access to Sarfati et al. for the full motivation of each
quantity. All of the statistics listed above are functions of the
centered per-layer activations only and are invariant under any choice
of orthonormal basis for $\mathbb{R}^H$ that the SVD might equivalently
return (e.g. column sign flips, ties in singular values). This basis
invariance is what makes them candidates for cross-model comparison.

### 2.3 The three views

A *view* of the residual-stream ensemble is a selection or partition of
the $N$ pilots into subsets, on each of which the basis-invariant
statistics of §2.2 can be recomputed. We work with three views in this
paper.

**All-to-all view (the marginal).** The full ensemble of $N$ pilots,
treated exchangeably. This recovers the original framework's
measurements. We use the subscript $\mathrm{a}$ for all-to-all
statistics: $V_{\mathrm{a}}(t)$, $\log\alpha_{\mathrm{a}}$, $\lambda_{\mathrm{a}}$,
$r_{\text{eff}, \mathrm{a}}(t)$. The all-to-all view is what §3
characterizes at convergence and what §4 tracks through training.

**Forward view (input-conditioned).** Pilots are filtered by the token
$v = c_p$ at the pilot position $p$ — what we will call the *input
token*, since it is the token at the position whose successor the model
is predicting from. For each $v$ in a chosen *forward token set*
$V \subseteq \{1, \ldots, |\text{Vocab}|\}$, the forward conditional
ensemble $\mathcal{E}_v$ is the subset of pilots with input token $v$:

$$\mathcal{E}_v = \{i \in \{1, \ldots, N\} : c_p^{(i)} = v\}, \qquad v \in V.$$

The basis-invariant statistics of $\mathcal{E}_v$ are computed on the
centered activations within the subset: the per-coordinate variance is
$V_{\mathcal{E}_v}(t) = \text{Var}_{i \in \mathcal{E}_v} [x_t^{(i)}]$,
the SVD is performed on the centered $|\mathcal{E}_v| \times H$
activation matrix, and so on.

By construction, $V_{\mathcal{E}_v}(0) = 0$ for every $v$: all pilots
sharing input token $v$ have the same post-embedding state $E(v)$, so
their variance at $t = 0$ is identically zero. (For architectures with
positional embeddings that vary across pilot positions $p$, this is
not exactly true — different pilots will have the same token embedding
but different positional embeddings — but for the RoPE architecture we
use, the post-embedding state is the unmodified token embedding, since
RoPE is applied inside attention rather than to the embeddings
directly. The variance at $t = 0$ is exactly zero in our setup.)

As $t$ increases from 0, the forward conditional ensemble gains
variance because attention folds context information (the tokens
preceding position $p$, which vary across different pilots even when
$c_p$ is fixed) into the residual state. The trajectory of
$V_{\mathcal{E}_v}(t)$ from zero through deeper layers measures the
rate and dimensionality of context-driven differentiation conditional
on a fixed starting input.

The forward view also has a *between-input* component, defined as
follows. Let $\mu_t(v) = \bar{x}_t(\mathcal{E}_v)$ be the mean
activation of the forward conditional ensemble at layer $t$. The
between-input variance is

$$V_{\text{between-fwd}}(t) = \text{Var}_{v \in V}\!\left[\mu_t(v)\right] = \frac{1}{|V|} \sum_{v \in V} \left\| \mu_t(v) - \bar{\mu}_t \right\|^2 / H$$

where $\bar{\mu}_t = \frac{1}{|V|} \sum_v \mu_t(v)$. The
between-input variance is the per-coordinate variance of the per-input
means, averaged over coordinates.

In the law of total variance, the within-input variance is the average
of the per-input within variances, weighted by the frequency of each
input:

$$V_{\text{within-fwd}}(t) = \mathbb{E}_{v}\!\left[V_{\mathcal{E}_v}(t)\right] = \frac{1}{|V|} \sum_{v \in V} \frac{|\mathcal{E}_v|}{|V_{\text{total}}|} V_{\mathcal{E}_v}(t),$$

with the weighting $|\mathcal{E}_v| / |V_{\text{total}}|$ giving more
weight to tokens that occur more often. (In our reported quantities we
use uniform weighting, $\frac{1}{|V|} \sum_v V_{\mathcal{E}_v}(t)$,
because the variance of a fixed-size sample is approximately
proportional to the sample size and uniform weighting gives more
interpretable per-token comparison. The choice affects the absolute
within/between magnitudes by a small constant factor but not their
trajectory shapes or the crossover layer location.)

**Reverse view (output-conditioned).** Pilots are filtered by the
*successor token* $w = c_{p+1}$ — the token at position $p + 1$ in the
chunk that the residual state at position $p$ is predicting. For each
$w$ in a chosen *reverse token set* $W$, the reverse conditional
ensemble $\mathcal{F}_w$ is the subset of pilots with successor token
$w$:

$$\mathcal{F}_w = \{i \in \{1, \ldots, N\} : c_{p+1}^{(i)} = w\}, \qquad w \in W.$$

By construction, $\mathcal{F}_w$ contains pilots from many different
input tokens (since many tokens can precede $w$), so its variance at
$t = 0$ is high; through subsequent layers, the model's task is to
converge on a shared prediction of $w$ from these different starting
points, so the reverse conditional ensemble may contract as the model
shapes the residual stream toward consistent output. The reverse view
measures the rate and dimensionality of prediction-driven convergence
onto a fixed endpoint.

The reverse view also has within and between components, defined
analogously:

$$V_{\text{within-rev}}(t) = \mathbb{E}_{w}\!\left[V_{\mathcal{F}_w}(t)\right],$$

$$V_{\text{between-rev}}(t) = \text{Var}_{w \in W}\!\left[\mu_t(w)\right].$$

### 2.4 The actual-vs-predicted reverse view

We work with two variants of the reverse view, distinguished by which
successor is used to condition.

**Reverse-actual view.** Conditioning on $w = c_{p+1}$, the actual
successor token from the held-out chunk. This describes the residual
stream's relationship to the data-generating process — the
ground-truth successors that the model is being trained to predict.
Conditional ensembles are $\mathcal{F}_w^{\text{actual}} = \{i :
c_{p+1}^{(i)} = w\}$.

**Reverse-predicted view.** Conditioning on $\hat{w} = \text{argmax}_w
P(w \mid x_L(\mathbf{c}, p))$, the model's argmax-predicted successor
from the post-block-$L$ state. This describes the residual stream's
relationship to the model's own behavior — the tokens the model
*thinks* come next, regardless of whether it is correct. Conditional
ensembles are $\mathcal{F}_w^{\text{pred}} = \{i : \hat{w}^{(i)} = w\}$.

The two views agree on pilots where the model's prediction is correct
($\hat{w} = c_{p+1}$) and diverge on pilots where it is not. At
convergence the model is correct on approximately 40-45% of pilot
positions (held-out top-1 accuracy), so the two reverse views differ on
55-60% of pilots. The structural and dynamical patterns we report in
§5-§6 are qualitatively the same for both reverse views, but
quantitatively the reverse-predicted view has lower within/between
ratios than the reverse-actual view at the mid-network peak (11.9 vs
18.8 at the cross-seed mean for the bulge maximum) and similar but
distinguishable training-evolution patterns.

Both views are valid conditional partitions of the pilot set and both
satisfy the variance decomposition identity. Both are reported throughout
because the comparison between them is itself informative: differences
between actual-conditioned and predicted-conditioned reverse-view
behavior reflect the gap between the data-generating process and the
model's learned distribution, and this gap shifts through training (the
predicted-view ratios are more strongly affected than the actual-view
ratios by the late-training period in which prediction accuracy is
still improving).

The reverse-predicted view has a known degenerate behavior at random
initialization: an untrained model's argmax predictions are essentially
random with respect to the input, so conditioning on $\hat{w}$ is
nearly equivalent to randomly subsetting the pilots, and the
$V_{\text{between-rev-pred}}$ at $t = 0$ is approximately zero. This
makes the within/between ratio at $t = 0$ for the reverse-predicted
view extremely large at random initialization (ratio $\approx 103$ in
our measurement, which we report in §6.2 with caveats about
numerical robustness). We flag this degeneracy explicitly because it
matters for the random-init comparison; at convergence the model's
predictions are non-degenerate and the ratio behaves stably.

### 2.5 The variance decomposition and token-set restriction

For any partition of pilots by label $z$ and any layer $t$, the law of
total variance gives the per-coordinate identity

$$V_{\mathrm{a}}(t) = \mathbb{E}_z\!\left[V_{\text{within-}z}(t)\right] + \mathrm{Var}_z\!\left[\mu_t(z)\right]$$

where $V_{\mathrm{a}}(t)$ is the all-to-all variance,
$V_{\text{within-}z}(t)$ is the variance computed within the subset of
pilots sharing label $z$, and $\mu_t(z)$ is the mean of that subset.
The first term on the right is the within-condition variance averaged
over conditions weighted by their frequency; the second is the variance
of the per-condition means.

This identity holds at every layer separately, for any choice of
conditioning label, *provided the label spans the full pilot set*. If
the label is restricted to a chosen subset (e.g., the top-20 most
frequent input tokens out of $|\text{Vocab}| = 32768$ tokens), the
identity holds only for the *subset-restricted all-to-all variance*:

$$V_{\mathrm{a}, V}(t) = \mathbb{E}_{v \in V}\!\left[V_{\mathcal{E}_v}(t)\right] + \mathrm{Var}_{v \in V}\!\left[\mu_t(v)\right]$$

where $V_{\mathrm{a}, V}(t)$ is the all-to-all variance computed on
only those pilots whose input is in $V$. We refer to $V_{\mathrm{a},
V}(t)$ as the *subset variance* to distinguish it from the
unrestricted all-to-all variance.

The unrestricted all-to-all variance differs from the subset variance
by the contributions of pilots outside $V$. Specifically:

$$V_{\mathrm{a}}(t) = \frac{|V_{\text{covered}}|}{N} V_{\mathrm{a}, V}(t) + \frac{|V_{\text{uncovered}}|}{N} V_{\mathrm{a}, \text{uncovered}}(t) + \text{(between-mean contribution)}$$

where $V_{\text{covered}}$ is the set of pilots whose input is in $V$,
$V_{\text{uncovered}}$ is the complement, and the between-mean
contribution accounts for the difference between the mean activations
of the covered and uncovered pilots. We track $|V_{\text{covered}}| / N$
explicitly throughout: for our top-20 forward token set, this fraction
is approximately 18%; for the top-20 reverse token set, approximately
19%.

The decomposition's identity is exact for the subset; the relationship
to the unrestricted all-to-all variance is approximate. Throughout this
paper, when we report "within-input variance" and "between-input
variance" without further qualification, we mean the subset-restricted
quantities, and the partition identity holds exactly. When we compare
to the unrestricted marginal, we report the subset-vs-marginal
relationship explicitly.

The functional content of the decomposition is in the *layer-wise
trajectory* of the within and between components. The questions the
decomposition was designed to ask:

1. At what layer does within-input variance overtake between-input
   variance? This is the moment context-driven differentiation
   overwhelms input identity in the residual stream. We call this the
   *forward crossover layer* and denote it $t_{\text{cross, fwd}}$.

2. At what layer does within-output variance contract enough that
   between-output variance dominates? If this happens at any layer, it
   is the moment prediction commitment forms in a basis-invariant
   sense. In our measurements, the within-output variance is dominant
   at every layer (no reverse crossover), so we instead characterize
   the reverse view by the location of its within/between *peak*
   $t_{\text{peak, rev}}$ and the magnitude of contraction from peak
   to output.

3. Are these depths the same across the two reverse view variants?
   Across seeds? Across training? Do they coincide with the
   training-dynamic anomalies of the marginal framework?

These are the structural and dynamical questions §5 and §6 address.

We summarize each view by three quantities at each layer:

- the within-condition variance $V_{\text{within-}z}(t) = \mathbb{E}_z[V_{\text{within-}z}(t)]$;
- the between-condition variance $V_{\text{between-}z}(t) = \mathrm{Var}_z[\mu_t(z)]$;
- their ratio $r_z(t) = V_{\text{within-}z}(t) / V_{\text{between-}z}(t)$.

The crossover layer of a view is the smallest $t$ at which $r_z(t)$
crosses 1.0 (forward views, in which the ratio rises through 1.0 as
$t$ increases from 0) or falls below 1.0 (reverse views, in which the
ratio is already below 1.0 at $t = 0$ if it ever crosses). For
linear-interpolation purposes we compute the crossover as the
log-linear interpolation of $r_z(t)$ between the bracketing integer
layers, since the ratio varies log-linearly in $t$ near the crossover.

### 2.6 Per-view basis-invariant statistics

On each conditional ensemble $\mathcal{E}_v$ or $\mathcal{F}_w$, the
basis-invariant statistics of §2.2 remain well-defined. We compute and
report the following per-view quantities.

**Per-view $\lambda$ and $\log\alpha$.** Fit $\log V_{\text{within-}z}(t)
= \log\alpha_z + \lambda_z \log t$ to the within-condition variance
trajectory across layers (excluding $t = 0$ for the forward view, where
variance is zero by construction). The forward $\lambda_{\text{fwd}}$
is the slope at which within-input variance grows; the reverse
$\lambda_{\text{rev}}$ is its analogue on the within-output side. Both
are reported in both the paper convention and our convention; both have
characterizable cross-seed dispersion that we report alongside.

In our measurements, $\lambda_{\text{fwd}} \approx 0.535$ and
$\lambda_{\text{rev}} \approx 0.295$ at convergence (paper convention,
averaged across seeds at the final checkpoint), substantially different
from the marginal $\lambda_{\mathrm{a}} \approx 0.362$. The
per-view $\lambda$ values are not constrained by the variance
decomposition identity to satisfy any specific relationship to the
marginal $\lambda$, but they are constrained in trajectory: the
sum-decomposition of $V_{\text{within}} + V_{\text{between}}$ must
equal $V_{\mathrm{a}}$ at every layer, which couples how
$\lambda_{\text{fwd}}$ and the between-trajectory must collectively
reproduce the all-to-all behavior.

**Per-view effective rank profile.** For each layer $t$ and each
condition $z$, compute the effective rank of the per-condition centered
activations:

$$r_{\text{eff}, z}(t) = \frac{\left(\sum_k s_{z,t,k}^2\right)^2}{\sum_k s_{z,t,k}^4}$$

where $s_{z,t,k}$ are the singular values of the centered conditional
activation matrix. Average over $z$ to get the per-view effective
rank profile $r_{\text{eff, fwd}}(t)$ or $r_{\text{eff, rev}}(t)$.

In our measurements, the per-view effective rank profiles are
*qualitatively different* functions of depth, not rescalings of each
other. The forward effective rank starts at 0 at $t = 0$ (the bundle
is a Dirac, $r_{\text{eff}} = 0$), rises to approximately 80 at the
middle layers, and remains there. The reverse effective rank starts at
approximately 80 at $t = 0$ (the within-output bundle is already
moderately spread out at the embedding) and stays approximately
constant through depth. The marginal effective rank rises from
approximately 180 at $t = 0$ to a peak of approximately 510 at
$t = 8$, then declines. The three profiles have peaks at very
different layers and very different magnitudes (§5.4 has the full
report).

**Per-view kurtosis profile.** Per-coordinate excess kurtosis of the
centered conditional activations at each layer, defined as in §2.2 but
computed on the conditional ensemble. The forward kurtosis spikes
dramatically at $t = 1$ (we measure 7.0 at $t = 1$ for the forward view
in seed 0, vs 3.9 for the marginal view at the same layer), reflecting
that the first layer's context-injection produces heavy-tailed
within-input variance. Beyond $t = 1$ the forward kurtosis declines
toward the marginal value. The reverse kurtosis profile is less
dramatic but still non-trivial.

**Per-view crossover layer.** As defined in §2.5, the layer at which the
within/between ratio crosses 1.0. For our measurements, only the
forward view has a crossover; the reverse views have no crossover
because $V_{\text{within-rev}}$ exceeds $V_{\text{between-rev}}$ at
every layer.

**Per-view within/between ratio.** The ratio $r_z(t) = V_{\text{within-}z}(t)
/ V_{\text{between-}z}(t)$, computed at every layer and reported as the
ratio profile. This is the headline statistic for the multi-view
decomposition's structural findings (§5) and dynamical findings (§6).

These conditional statistics constrain each other via the variance
decomposition (§2.5) and via the partition's relationship to the
all-to-all ensemble. The framework's basis invariance carries through
the conditioning, so cross-seed comparisons of conditional statistics
are well-posed even though cross-seed comparisons of $R(t)$ are not.
This is the central methodological point: we can compare conditional
within/between ratios across seeds, conditional crossover layers,
conditional $\lambda$ values, and conditional effective rank profiles,
all in the basis-invariant regime, and we will find that they
reproduce across seeds tightly even though the underlying conditional
$R$-matrices (which we do not compute and do not need) presumably do
not.

### 2.7 Token-set selection

The conditional ensembles $\mathcal{E}_v$ and $\mathcal{F}_w$ are
defined for any token; in practice the per-condition sample size
constrains how many we can usefully analyze. We use the *top-20 most
frequent tokens* at the pilot positions of the held-out set for both
$V$ (forward input tokens) and $W$ (reverse successor tokens),
selected as follows.

**Forward token set $V$.** We compute the frequency of each token $v$
in the position $c_p$ across all 9,500 pilots of seed 0's final
checkpoint (i.e., the marginal distribution of input tokens at the
pilot positions). We select the top 20 most frequent tokens. The
resulting set $V$ has $|V| = 20$ tokens; the per-token sample sizes
$|\mathcal{E}_v|$ range from 50 to 389 across the 20 tokens. The total
coverage is $\sum_{v \in V} |\mathcal{E}_v| = $ approximately 1700
pilots out of 9,500, or 18%.

**Reverse token set $W$.** We compute the frequency of each token $w$
in the position $c_{p+1}$ across all pilots of seed 0's final
checkpoint (i.e., the marginal distribution of successor tokens). We
select the top 20 most frequent successors. Per-token sample sizes
$|\mathcal{F}_w|$ range from 50 to approximately 380; total coverage
approximately 19%.

We use the *same* token sets across all four seeds and across all 50
checkpoints, defined from seed 0's final checkpoint. This means that
the per-seed and per-checkpoint variations we report are due to
differences in the model's response to a fixed set of input/successor
tokens, not to differences in which tokens are selected. The token
sets are chosen once and held fixed for all measurements.

We do not weight the conditional variances by the per-token sample
size when averaging across tokens to compute $V_{\text{within-fwd}}$
and $V_{\text{within-rev}}$. The uniform-weighting choice has the
property that the resulting within-variance is the average per-token
within-variance regardless of how often each token occurs. This makes
the per-token profile interpretable as "what is the typical within-token
variance," and matches how we expect the framework's universality
claim to be tested (universality across architectures presumably means
the same per-token structure, not the same token frequencies). The
choice affects absolute magnitudes by a small constant factor compared
to frequency-weighted averaging; trajectories and crossover layers
are insensitive to the choice.

We chose $|V| = |W| = 20$ as the largest set size for which every
token has at least 50 pilots in seed 0's final checkpoint, giving
adequate per-token sample size for the per-view statistics to be
stable. Reducing to $|V| = 10$ gives qualitatively identical results
with somewhat tighter within-token statistics (per-token sample sizes
range 100-389); increasing to $|V| = 50$ requires some tokens with
sample sizes below 30, which begin to show noticeable noise in their
per-token covariance estimates (relevant for the per-token covariance
analysis of §6.5). The choice $|V| = |W| = 20$ is a deliberate
compromise.

### 2.8 Cross-seed alignment via orthogonal Procrustes

For §6.4's cross-seed comparison of residual-stream subspaces, we use
the orthogonal Procrustes construction. Given two sets of activations
$X_t^{(A)} \in \mathbb{R}^{N \times H}$ and $X_t^{(B)} \in
\mathbb{R}^{N \times H}$ collected at layer $t$ from two seeds $A$ and
$B$ processing the same inputs (i.e., the $i$-th pilot in $A$ and the
$i$-th pilot in $B$ correspond to the same chunk and the same pilot
position), find the orthogonal matrix that best aligns them:

$$Q_t = \arg\min_{Q \in O(H)} \lVert X_t^{(A)} Q - X_t^{(B)} \rVert_F.$$

The closed-form solution is given by the SVD: if $U_t S_t V_t^\top =
X_t^{(A)\top} X_t^{(B)}$, then $Q_t = U_t V_t^\top$. We report the
*residual ratio*

$$\rho_t = \frac{\lVert X_t^{(A)} Q_t - X_t^{(B)} \rVert_F}{\lVert X_t^{(B)} \rVert_F}$$

as the basis-invariant measure of how well the two seeds' residual
streams align up to rotation at layer $t$. Small $\rho_t$ means the
subspaces correspond well; large $\rho_t$ means they don't.

The numerical value of $\rho_t$ is hard to interpret without reference
points. We compute two null baselines.

**Random-rotation null (floor).** Apply a fresh uniform random
orthogonal rotation $\tilde{Q}$ to $X_t^{(A)}$, then align $X_t^{(A)}
\tilde{Q}$ back to $X_t^{(A)}$ via Procrustes. The residual here is
essentially zero (modulo floating-point precision, of order $10^{-14}$
in our measurements) because the rotation is exactly invertible. The
random-rotation null is the *floor* of what perfect cross-seed alignment
would produce; the trained-pair $\rho_t$ should be measured against this
floor for "how far from perfect alignment are we."

**Random-scramble null (ceiling).** Permute the pilot ordering in
$X_t^{(B)}$ to get $X_t^{(B), \text{scrambled}}$, then align $X_t^{(A)}$
to the scrambled version. This breaks the per-pilot correspondence
between $A$ and $B$ — the $i$-th row of $X_t^{(A)}$ is now being
aligned to a row of $X_t^{(B)}$ that corresponds to a different chunk
and position. The marginal distribution of each is unchanged, but the
per-pilot correspondence that makes the alignment well-posed is broken.
The residual here is the *worst-case* "no alignment possible"
reference. We compute this baseline as the average over 3 random
permutations. In our measurements, the scramble-null $\rho_t$ is
approximately 0.9-1.3, with some variation across layers (smaller at
intermediate layers, larger at the boundary layers).

The trained-pair $\rho_t$ between two independent seeds sits between
these baselines. We report it as both an absolute number and as a
fraction of the scramble null (the *alignment-quality ratio*), which
gives a layer-by-layer measure of how close trained seeds are to
"perfect alignment" (0%) vs "no alignment" (100%). The cross-seed
comparison in §6.4 reports per-pair $\rho_t$, the mean across all 6
ordered seed pairs from our 4-seed pilot, and the two null baselines
computed on seed 0's activations (the choice of seed for null
computation does not affect the result).

### 2.9 Implementation determinism

The cross-seed Procrustes construction requires that the $i$-th pilot
in seed $A$ and the $i$-th pilot in seed $B$ correspond to the same
chunk and position. This is satisfied in our pipeline because the
held-out dataloader is deterministic and is applied with identical
ordering at every seed's final checkpoint. We verify this directly: the
input-token, successor-token, and pilot-position arrays in the
augmented activation files (§3.3) are identical across seeds, byte for
byte. We assert this equality at the start of every cross-seed
analysis script; if the assertion fails, the analysis aborts before
producing any output.

This determinism is a requirement we impose on the implementation, not
a property of any specific cross-seed pipeline. Future studies that
want to extend these results to new model variants must preserve the
pilot-correspondence determinism if they wish to apply Procrustes
alignment in the form we use.

---
