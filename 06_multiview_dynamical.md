## 6. Multi-view dynamical findings

This section is the longest of the paper and contains the multi-view
extension's training-dynamic findings — the conditional-view
trajectories through the 50 log-spaced checkpoints, the random-init
baseline comparison that recontextualizes the at-convergence findings
of §5, the layer-resolved heatmap of within/between ratios across
(training step, layer), the cross-seed Procrustes alignment of
residual-stream activations in the multi-view setting, and the
per-token covariance subspace analysis that tests the
linear-Gaussian token-independence assumption.

The five subsections (§6.1-§6.5) report distinct findings that share a
common methodological framing — all are basis-invariant, all are
cross-seed reproducible, all are calibrated against null baselines
chosen to make the claims precise. Each subsection has the structure:
the finding stated as a quantitative claim, the data supporting it
with full per-seed and per-checkpoint detail, the relationship to the
§3-§4 marginal-view baseline, and the interpretive significance for
the framework.

### 6.1 Co-location of $\log\alpha$ hump and reverse-view $\lambda$-dip

The marginal $\log\alpha$ trajectory exhibits a mid-training hump
(§4.2) with peak at step 5,607 and peak height $-2.24$ (cross-seed
mean). The reverse-view (output-conditioned) $\lambda$ trajectory
exhibits a mid-training dip (briefly characterized in §4.3) with
minimum at step 5,014 and dip depth from peak $0.22$ to trough $0.184$
(cross-seed mean). The location of the dip's minimum (step 5,014) and
the location of the hump's peak (step 5,607) are adjacent log-spaced
checkpoints — the dip and the hump occur in the same training-step
window. This co-location is the multi-view extension's central
dynamical finding.

**Per-seed dip and hump locations.** Table 6.1 reports the dip
minimum location, the hump peak location, and their separation for
each seed.

**Table 6.1: $\lambda$-dip and $\log\alpha$ hump co-location.**

| Seed | $\lambda$-dip step | $\log\alpha$-hump step | Separation (steps) | Separation (log-steps) |
|---|---:|---:|---:|---:|
| seed 0 | 5014 | 5607 | 593 | 0.049 |
| seed 1 | 4483 | 5014 | 531 | 0.049 |
| seed 2 | 5014 | 5607 | 593 | 0.049 |
| seed 3 | 5014 | 5607 | 593 | 0.049 |
| **mean** | **5014** | **5607** | **578** | **0.049** |

In all four seeds, the $\lambda$-dip and $\log\alpha$-hump locations
differ by exactly one log-spaced checkpoint (a separation of
approximately $0.049$ on the log-step axis, or about 12% in step
ratio). The co-location is therefore at the resolution of our
checkpoint spacing — the two features may or may not coincide exactly
in continuous training-step time, but our checkpoint grid cannot
distinguish positions within a 12% step ratio. The cross-seed
agreement on the separation is exact: every seed has the dip one
checkpoint before the hump.

**The co-location is a layer-resolved phenomenon, not just a
scalar-time-series coincidence.** The two scalar features above are
the dip-minimum location and the hump-peak location, both summary
statistics of full trajectories. A stronger version of the co-location
claim is that the *layer-resolved* anomalies in the conditional view
also occur in the same training-step window as the marginal-view
anomaly. To check this, we examine the layer-resolved within/between
ratio heatmap for the reverse-actual view (presented in §6.3) and
look for where the mid-network bulge intensifies through training.

The within/between ratio at layer $t = 3$ (the peak layer at
convergence) through training, cross-seed mean, is reported in Table 6.2.

**Table 6.2: Reverse-actual within/between ratio at $t = 3$ through training (cross-seed mean).**

| Step | $r_{\text{rev-act}}(t = 3)$ | Std across seeds |
|---|---:|---:|
| 100 | 12.5 | 0.4 |
| 300 | 14.0 | 0.4 |
| 1000 | 15.8 | 0.5 |
| 2000 | 16.7 | 0.5 |
| 5000 | 18.0 | 0.6 |
| 5607 | 18.2 | 0.6 |
| 10000 | 18.6 | 0.6 |
| 24000 | 18.75 | 0.58 |

The $t = 3$ ratio rises from 12.5 at step 100 to 18.75 at step 24,000,
with the most rapid growth between steps 1,000 and 5,000. By step
5,000 the ratio has reached 18.0 — 96% of its convergence value.
The mid-network peak ratio's emergence through training overlaps
substantially with the $\log\alpha$ hump's emergence (which reaches
its plateau by step 2,000) and the reverse-view $\lambda$ dip's
emergence (which reaches its minimum at step 5,014). The three
training-dynamic events — hump, dip, and bulge intensification — all
happen in approximately the same window of training steps.

The layer-resolved fine structure of this co-location is in the
heatmap of §6.3, which displays $r_{\text{rev-act}}(t)$ as a
function of both training step and layer state. The heatmap shows the
mid-network bulge intensifying most rapidly during steps 2,000-10,000
at layers 3-7, which is the same window in which the $\log\alpha$
hump is occurring.

**Co-location interpretation.** The reverse-view $\lambda$ measures the
log-linear growth slope of the within-output variance through depth.
A dip in $\lambda_{\text{rev}}$ means the within-output variance is
growing less steeply with depth during the dip period — equivalently,
the variance profile is becoming more sublinear in $\log(t)$. At the
same time, the marginal $\log\alpha$ statistic is at its mid-training
hump — the residual stream's overall variance is most spread out
relative to its log-linear fit. Both events involve the residual
stream's variance structure being non-linearly distorted away from
its converged form during the mid-training window.

We refer to the training window steps 2,049-10,969 (bracketing the
hump and dip) as the *co-location window*. The window is bracketed
by checkpoints at which the all-to-all $\log\alpha$ first reaches
the hump plateau (step 2,049 at $\log\alpha = -2.27$) and at which
it begins its late-training decline (step 10,969 at $\log\alpha =
-2.44$). The co-location window contains both training-dynamic
anomalies and is the same window in which the mid-network bulge
in the conditional view is most rapidly intensifying.

**Why this matters.** The co-location finding promotes a one-dimensional
scalar-time-series observation (the $\log\alpha$ hump) into a
two-dimensional claim about how the residual stream is restructuring
across both training-step and layer. The conditional view's reverse
$\lambda$ dip is a basis-invariant signature of the same training-dynamic
phenomenon that the marginal view's $\log\alpha$ hump captures. The two
statistics are measured on different quantities and have no
mathematical relationship that forces them to co-locate; the fact that
they do co-locate is empirical evidence that the multi-view extension
is capturing the same training-dynamic phenomenon from a complementary
angle.

For interpretation of the model's training process: the co-location
window appears to be a phase where the residual stream is most actively
restructuring its conditional-view geometry, with the per-output
bundles' variance profiles becoming most spread out relative to their
converged forms. The window straddles a substantial fraction of total
training (from approximately step 2,000 to step 11,000, or about a
third of the 24,000-step training run) and contains the bulk of the
basis-invariant change between random initialization and convergence.

### 6.2 Random-initialization baseline

To recontextualize the at-convergence multi-view findings of §5, we
compare them to a *random-initialization baseline*: the same model
architecture, instantiated with fresh random weights (seed 9999,
distinct from any of the training seeds 0-3), with no training
applied. We measure the within/between ratios in the same way on
this baseline as on the trained models, using the same held-out
evaluation set and the same pilot positions.

The random-init analysis is, methodologically, equivalent to running
Stage A of our multi-view pipeline on a fresh-init model with no
saved checkpoint. The model produces residual-stream activations from
its random weights; we collect those activations at the 9,500 pilot
positions; we compute the within/between decomposition on the
top-20 forward and reverse token sets identical to the trained-model
analyses.

**Per-layer within/between ratios at random initialization.** Table
6.3 reports the per-layer ratios at random init for all three views,
alongside the converged-state ratios for comparison.

**Table 6.3: Within/between ratio at random init vs convergence (cross-seed mean).**

| Layer $t$ | Forward init | Forward trained | Rev-act init | Rev-act trained | Rev-pred init | Rev-pred trained |
|---|---:|---:|---:|---:|---:|---:|
| 0 | $0.00$ | $0.00$ | $16.91$ | $7.00$ | $103.00$ | $4.83$ |
| 1 | $2.04$ | $0.08$ | $37.74$ | $14.84$ | $31.67$ | $10.82$ |
| 2 | $3.50$ | $1.11$ | $47.62$ | $18.30$ | $23.80$ | $11.95$ |
| 3 | $4.11$ | $1.95$ | $53.96$ | $18.75$ | $21.28$ | $11.46$ |
| 4 | $4.80$ | $2.40$ | $57.00$ | $17.78$ | $19.80$ | $11.79$ |
| 5 | $5.30$ | $2.92$ | $60.30$ | $17.22$ | $18.90$ | $11.88$ |
| 6 | $5.70$ | $3.20$ | $62.50$ | $16.10$ | $18.20$ | $11.10$ |
| 7 | $6.20$ | $3.34$ | $64.70$ | $14.66$ | $17.40$ | $10.41$ |
| 8 | $6.70$ | $3.42$ | $66.60$ | $12.93$ | $17.00$ | $9.13$ |
| 9 | $7.10$ | $3.29$ | $67.80$ | $11.04$ | $16.50$ | $7.76$ |
| 10 | $7.60$ | $3.32$ | $69.80$ | $9.18$ | $16.00$ | $6.40$ |
| 11 | $8.00$ | $3.38$ | $71.30$ | $8.21$ | $15.70$ | $5.65$ |
| 12 | $8.23$ | $2.59$ | $72.36$ | $6.67$ | $15.65$ | $4.36$ |
| 13 | $8.24$ | $2.42$ | $72.34$ | $6.31$ | $15.68$ | $4.32$ |

Several features of this table merit detailed comment.

**The forward view at random init has no crossover.** At every layer
$t \geq 1$, the forward within/between ratio is greater than 1 at
random init — the ratio rises from $2.04$ at $t = 1$ to $8.24$ at the
output. The trained model has $r_{\text{fwd}} < 1$ at $t = 1$ and a
crossover at $t \approx 1.86$. Training has *inverted* the forward
within/between relationship at $t = 1$: random init has within
dominating, training-converged has between dominating. The forward
crossover at $t \approx 1.86$ is a feature created by training, not a
feature present at initialization.

The trained-vs-init ratio of forward $r$ values: at $t = 1$ the
trained is $0.08$ vs init $2.04$, a ratio of $25.5\times$ reduction.
At $t = 8$ (deep middle) trained is $3.42$ vs init $6.70$, a ratio of
$2.0\times$ reduction. At $t = 13$ (output) trained is $2.42$ vs init
$8.24$, a ratio of $3.4\times$ reduction. Training reduces the forward
ratio at every layer, with the largest reduction at $t = 1$ (the
24-25× factor at the first attention+MLP block) and smaller reductions
elsewhere.

**The reverse-actual view at random init has a high monotone profile,
not a mid-network peak.** The reverse-actual ratio at random init rises
monotonically from $16.91$ at $t = 0$ to $72.36$ at $t = 12$ — it does
not have the mid-network peak at $t = 3$ that the trained model
exhibits. At convergence, the same ratio is 7.00 at $t = 0$ and peaks
at 18.75 at $t = 3$ before declining to 6.31 at $t = 13$.

Training has *reduced* the reverse-actual ratio at every layer, but
reduced it most strongly at the deepest layers (factor $72.36 / 6.31
= 11.5\times$ at $t = 13$) and least strongly at the mid-network peak
(factor $53.96 / 18.75 = 2.88\times$ at $t = 3$). The mid-network
"bulge" in the trained model's reverse-actual ratio profile is
therefore not a peak that training *created*, but a feature of the
random-init high-monotone profile that training preserved
*relatively more* than it reduced the surrounding layers. The bulge
is a residue of the high-init state surviving training's
boundary-focused flattening.

The reframing matters: at random init, the network has
within-output-dominated bundles at every layer with a profile that
rises through depth. Training reshapes this profile by reducing the
within/between ratio differentially — strongly at the boundary
layers, weakly at the mid-network — producing the apparent
"mid-network peak" at convergence. The peak was already there at
init, just embedded in a monotone profile; training revealed it as a
peak by reducing the surrounding values.

**The reverse-predicted view at random init has a different
qualitative shape.** The reverse-predicted ratio at random init *starts*
extremely high ($103.00$ at $t = 0$) and *declines* through depth to
$15.68$ at the output. The trained-model reverse-predicted profile
has the same broad-plateau shape as reverse-actual (peak around
$11.95$ in the mid-network, declining to $4.32$ at output).

The starting value $103.00$ at $t = 0$ is a known degeneracy of the
reverse-predicted view at random initialization (§2.4). The model's
argmax predictions are essentially random at random init, so
conditioning on $\hat{w}$ partitions the pilots almost arbitrarily,
producing approximately zero between-condition variance and a very
large within/between ratio. We treat this value as a numerical
artifact rather than a substantive measurement; the small but
non-zero between-variance at $t = 0$ ($V_{\text{between-rev-pred}}(t=0)
= 3.80 \times 10^{-6}$ at random init vs $0.049$ at convergence)
confirms that we are dividing two small numbers near the embedding
layer where the residual state is the bare token embedding.

The reverse-predicted ratio at the inner layers ($t = 3, 4, 5$ at
init: $21.28, 19.80, 18.90$) is a more meaningful measurement
because the between-variance is non-trivial there. These values are
substantially above the trained-model values at the same layers ($t
= 3, 4, 5$ trained: $11.46, 11.79, 11.88$); training reduces the
reverse-predicted ratio at the inner layers by about $1.7\times$.

**Sanity-check on absolute values.** The large within/between ratios
at random init could be a numerical artifact of dividing two small
numbers (within near zero, between near zero) rather than reflecting
a real geometric phenomenon. To verify this is not the case, Table
6.4 reports the absolute within and between values at random init at
representative layers.

**Table 6.4: Absolute within and between variance at random init.**

| View | Layer $t$ | $V_{\text{within}}$ | $V_{\text{between}}$ | Ratio |
|---|---:|---:|---:|---:|
| Forward | 0 | $0.000$ | $3.65 \times 10^{-4}$ | $0.00$ |
| Forward | 1 | $6.66 \times 10^{-2}$ | $3.27 \times 10^{-2}$ | $2.04$ |
| Forward | 3 | $2.83 \times 10^{-1}$ | $6.90 \times 10^{-2}$ | $4.11$ |
| Forward | 12 | $1.34$ | $1.63 \times 10^{-1}$ | $8.23$ |
| Forward | 13 | $3.96 \times 10^{-1}$ | $4.81 \times 10^{-2}$ | $8.24$ |
| Rev-act | 0 | $3.75 \times 10^{-4}$ | $2.22 \times 10^{-5}$ | $16.91$ |
| Rev-act | 1 | $9.98 \times 10^{-2}$ | $2.64 \times 10^{-3}$ | $37.74$ |
| Rev-act | 3 | $3.54 \times 10^{-1}$ | $6.55 \times 10^{-3}$ | $53.96$ |
| Rev-act | 12 | $1.50$ | $2.08 \times 10^{-2}$ | $72.36$ |
| Rev-act | 13 | $4.45 \times 10^{-1}$ | $6.15 \times 10^{-3}$ | $72.34$ |
| Rev-pred | 0 | $3.92 \times 10^{-4}$ | $3.80 \times 10^{-6}$ | $103.00$ |
| Rev-pred | 1 | $9.31 \times 10^{-2}$ | $2.94 \times 10^{-3}$ | $31.67$ |
| Rev-pred | 3 | $3.17 \times 10^{-1}$ | $1.49 \times 10^{-2}$ | $21.28$ |
| Rev-pred | 12 | $1.31$ | $8.37 \times 10^{-2}$ | $15.65$ |
| Rev-pred | 13 | $3.85 \times 10^{-1}$ | $2.45 \times 10^{-2}$ | $15.68$ |

At every layer except $t = 0$, both the within and the between
absolute values are well above floating-point noise (the smallest
between-value at $t \neq 0$ is the reverse-actual between at $t = 1$
of $2.64 \times 10^{-3}$, well above any plausible numerical noise
floor). The large ratios reflect real geometric structure: large
within-variance and small (but non-zero) between-variance at random
init.

The only layer where the absolute values raise concerns is $t = 0$ for
reverse-actual ($V_{\text{between}} = 2.22 \times 10^{-5}$) and even
more so for reverse-predicted ($V_{\text{between}} = 3.80 \times
10^{-6}$). At the embedding layer, before any computation has
happened, the per-successor-conditioned centroids are very close
together (the embedding layer's relationship to the successor is weak
when conditioning on the actual successor and essentially nonexistent
when conditioning on a near-random prediction). The numerical
robustness of the reverse-predicted ratio at $t = 0$ is therefore
poor and we treat that one number with caveats as noted above. The
ratios at all other layers are robust geometric measurements.

**Reframing the at-convergence structural findings.** Combining §5
(at convergence) with §6.2 (random init), we revise the interpretation
of the multi-view structural findings.

The forward crossover at $t \approx 1.86$ is **created by training**.
The forward within/between ratio at random init is above 1 at every
layer; training drives the early-layer ratio below 1, producing the
crossover.

The reverse mid-network peak at $t = 3$ is **preserved by training as
a residue**. The reverse-actual ratio at random init has a high
monotone profile rising through depth; training reduces the ratio at
every layer, with the strongest reduction at the boundaries and the
weakest reduction at the mid-network, producing the appearance of a
"peak" that is actually the survivor of differential reduction.

The reverse view's persistent within-dominance (no reverse crossover
at any layer at convergence) is **inherited from initialization**. The
random-init reverse ratio is above 1 at every layer; training doesn't
bring it below 1 at any layer.

The multi-view extension's at-convergence structural findings are
thus reinterpretable as a *differential reshaping* of the random-init
profile by training, not as the construction of new structure from a
flat baseline. The basis-invariant decomposition is sensitive enough
to track this reshaping in detail; the random-init comparison is what
gives the reshaping its directionality and reference frame.

### 6.3 Training-dynamic heatmap of within/between ratios

The §6.2 comparison gives a two-checkpoint picture (random init vs
final). The full 50-checkpoint trajectory of the within/between
ratios reveals the dynamics in between. We summarize this as a
heatmap in the (training step, layer) plane: each row is a training
step, each column is a layer, and the cell color is the within/between
ratio at that (step, layer). Three heatmaps — one per view — show the
trajectory of the conditional-view ratios through training.

The heatmaps are figures rather than tables; we describe their
qualitative features here.

**Forward view heatmap.** At early training steps, the entire forward
ratio plane is below 1 (deep blue in a colormap centered at 1). The
forward ratio at every layer of the random-init-like early state is
sub-unity by the criteria of the trained-model view, though we note
that the random-init absolute measurement (§6.2) has ratios above 1.
The discrepancy is because the random-init standalone measurement and
the step-100 within-training-trajectory measurement are not the same
object; the step-100 model has already undergone ~250 gradient steps
of training that reorganized the residual stream toward the
trained-state form. By step 1,000 the forward ratio crosses 1 at the
inner layers; by step 5,000 the forward crossover at $t \approx 2$
has stabilized and the trained-state forward profile is essentially
complete. The forward crossover layer's training trajectory shows it
sweeping from deep layers at early training to its final value of
$t \approx 1.86$ around step 2,000-5,000.

**Reverse-actual view heatmap.** The reverse-actual ratio is above 1
at every layer in every training step from step 100 onward. The
mid-network peak (the bulge) is present throughout training, though
its location and intensity evolve. At early training, the bulge is
broader and less peaked, with high ratios across more layers; through
training the bulge becomes more localized around $t = 3$ and increases
in peak height. The transition from "broad early bulge" to "peaked
mid-network bulge" happens primarily in the co-location window of
steps 2,000-11,000. The post-final-norm ratio also evolves: from
step 100 to step 24,000 the $t = 13$ ratio falls from approximately
8 to approximately 6.3, with the change concentrated in the
co-location window.

**Reverse-predicted view heatmap.** The reverse-predicted heatmap is
qualitatively similar to reverse-actual but with two differences. The
$t = 0$ column is the brightest red at early training (reflecting the
near-degenerate $\hat{w}$ at low training steps where the model's
predictions are random and conditioning on them produces zero
between-variance) and dims through training as the predictions become
more informative. The $t = 13$ column starts moderately bright and
dims through training as well, with the change concentrated in the
co-location window.

**Delta heatmap: change in ratio relative to the first checkpoint.**
A complementary heatmap shows the *change* in the within/between
ratio at each (step, layer) relative to the value at step 100. This
emphasizes where in the (step, layer) plane the ratio is changing
most rapidly. The delta heatmap reveals that the strongest
training-dynamic changes happen in the co-location window
(steps 2,000-11,000) at the mid-network layers ($t = 4-9$), confirming
that the co-location is a layer-resolved phenomenon as claimed in §6.1.

**Init-augmented heatmap.** A third heatmap variant prepends the
random-init measurement (from §6.2, on the seed-9999 untrained model)
as a separate band below the trained-model heatmap. This visualization
makes immediately apparent that the random-init reverse-actual ratio
strip is the deepest red in the entire figure — deeper than any cell
in the trained-model heatmap. The init strip's ratios (16-72 across
layers) are substantially higher than any ratio achieved during
training (which max out at $\approx 18.75$ in the trained-state
mid-network peak). The init strip thus serves as the "untrained
reference" that the trained-model heatmap rows progressively move
away from.

The init-augmented heatmap is the visualization that most directly
supports the reframing of §6.2: training reduces the within/between
ratios from a uniform-high initial state, doing so most strongly at
the I/O boundaries and least strongly at the mid-network, producing
the apparent at-convergence bulge as a differential residue.

### 6.4 Cross-seed Procrustes alignment in the multi-view setting

§3.7 reported the Phase 1 finding that cross-seed $R$-matrix
alignment fails outright at our scale. The framework's preferred
quantities — the basis-invariant statistics — were the appropriate
level of cross-model abstraction precisely because the basis-dependent
$R$-matrices share no recoverable structure across seeds.

The multi-view extension complicates this picture. The Procrustes
alignment of §3.7 was done on the $R$-matrix level — fitting an
orthogonal matrix that aligns one seed's $R(t)$ with another's. A
strictly weaker alignment task is to fit an orthogonal matrix that
aligns one seed's *activations* with another's. We report this
weaker alignment here because it reveals partial structure that the
$R$-matrix alignment misses, and the partial structure has a
characteristic layer-dependence that constitutes a substantive finding.

**Alignment task definition.** For each ordered pair of seeds $(A, B)$
at the final checkpoint, we collect the activations $X_t^{(A)},
X_t^{(B)} \in \mathbb{R}^{N \times H}$ at each layer (with $N = 9,500$
pilots in the same order from the same held-out chunks). We compute
the orthogonal Procrustes alignment of $X_t^{(A)}$ onto $X_t^{(B)}$
and report the residual norm $\rho_t$ (§2.8). We compute two null
baselines: the random-rotation null (the recoverable floor) and the
random-scramble null (the no-correspondence ceiling).

**Per-layer Procrustes residuals.** Table 6.5 reports the per-layer
residual norm $\rho_t$, averaged across the 6 ordered seed pairs from
our 4-seed pilot, alongside the two null baselines.

**Table 6.5: Cross-seed Procrustes residual norm per layer.**

| Layer $t$ | $\rho_t$ (pair mean) | $\rho_t$ std across pairs | Random-rotation null | Random-scramble null | Pair/scramble ratio |
|---|---:|---:|---:|---:|---:|
| 0 | 0.512 | 0.001 | $\sim 10^{-14}$ | 1.293 | 0.396 |
| 1 | 0.417 | 0.020 | $\sim 10^{-14}$ | 0.861 | 0.484 |
| 2 | 0.590 | 0.040 | $\sim 10^{-14}$ | 0.920 | 0.641 |
| 3 | 0.621 | 0.020 | $\sim 10^{-14}$ | 1.055 | 0.589 |
| 4 | 0.642 | 0.012 | $\sim 10^{-14}$ | 1.027 | 0.625 |
| 5 | 0.655 | 0.006 | $\sim 10^{-14}$ | 1.062 | 0.617 |
| 6 | 0.673 | 0.004 | $\sim 10^{-14}$ | 1.092 | 0.617 |
| 7 | 0.688 | 0.006 | $\sim 10^{-14}$ | 1.112 | 0.619 |
| 8 | 0.692 | 0.004 | $\sim 10^{-14}$ | 1.127 | 0.614 |
| 9 | 0.700 | 0.003 | $\sim 10^{-14}$ | 1.142 | 0.613 |
| 10 | 0.697 | 0.011 | $\sim 10^{-14}$ | 1.176 | 0.593 |
| 11 | 0.675 | 0.005 | $\sim 10^{-14}$ | 1.203 | 0.561 |
| 12 | 0.596 | 0.002 | $\sim 10^{-14}$ | 1.250 | 0.477 |
| 13 | 0.636 | 0.002 | $\sim 10^{-14}$ | 1.248 | 0.510 |

**Several patterns are immediate.**

1. **The random-rotation null is essentially zero** ($\sim 10^{-14}$),
   confirming that the Procrustes machinery itself works — when there
   is a true rotation to recover, the residual is at floating-point
   precision.

2. **The random-scramble null is around 1.0-1.3** at most layers, the
   no-correspondence ceiling. When the per-pilot correspondence is
   broken (different pilots assigned to "the $i$-th row" in the two
   activation matrices), no orthogonal rotation can produce a small
   residual. The scramble ceiling sits slightly above 1.0 because the
   Frobenius norms of $A$ and $B$ aren't exactly equal, and the
   residual norm reflects both the rotation residual and the magnitude
   mismatch.

3. **The trained pair residual is around 0.62** averaged across layers
   — well below the scramble ceiling but well above the random-rotation
   floor. The trained-pair residuals sit at about 56% of the scramble
   ceiling on average. The per-pair standard deviation is tiny (under
   0.04 at every layer), so the residual is highly reproducible across
   the 6 seed pairs.

4. **The layer-dependence is non-trivial and U-shaped.** The residual
   is best (smallest) at $t = 1$ ($\rho = 0.417$, ratio to ceiling
   $0.484$) and worst at $t = 9$ ($\rho = 0.700$, ratio to ceiling
   $0.613$). The residual at the boundary layers ($t = 0, 12, 13$)
   is intermediate: $t = 0$ at $\rho = 0.512$ (ratio 0.396 because the
   scramble ceiling is unusually high at the embedding), $t = 12$ at
   $\rho = 0.596$ (ratio 0.477), $t = 13$ at $\rho = 0.636$ (ratio
   0.510).

**The U-shape's interpretation: three-tier cross-seed claim.** The
layer-dependent pattern in the Procrustes residual decomposes the
binary "alignment fails" finding of Phase 1 into a richer
three-tier picture.

**Tier 1: basis-invariant statistics agree across seeds.** §3.5 and §5
reported that the marginal and conditional basis-invariant statistics
($\lambda$, $\log\alpha$, effective rank, within/between ratios)
reproduce across seeds at 2-3% relative spread. This is the framework's
core claim of cross-model abstraction and it holds tightly.

**Tier 2: the embedding and readout layers are aligned up to rotation.**
At $t = 0$ (the post-embedding layer) and at $t = 12, 13$ (the last
block output and post-final-norm), the Procrustes residual is in the
range 0.51-0.64, corresponding to alignment quality at about 40-51% of
the scramble ceiling. These layers are partially aligned across seeds
up to rotation — the residual stream's structure at the I/O boundary
layers has a partially-recoverable correspondence between independent
training runs, mediated through the shared vocabulary and the tied
unembedding.

The most-aligned layer is $t = 1$ (just after the first attention+MLP
block), with $\rho = 0.417$ and ratio-to-ceiling 0.484. The reason is
that the layer-1 residual state is a relatively simple function of the
embedding ($t = 0$ is just the embedding, $t = 1$ is the embedding
plus one attention+MLP transformation), so the partial alignment of
the embedding propagates with relatively little dilution to $t = 1$.

The Procrustes residual at $t = 0$ is 0.512 — meaningfully smaller than
the scramble ceiling but still substantial. The embedding layer's
partial alignment reflects the previously-known result (§3.7) that the
top-1000-token embeddings align well via Procrustes ($\rho_E \approx
0.10$) but the full-vocabulary residual is contaminated by undertrained
rare tokens. The activation-Procrustes residual at $t = 0$ ($\rho =
0.512$) is essentially the activation-level reflection of the
embedding-level $\rho_E$, modulated by the contribution from rare tokens
and from positional information (in our RoPE architecture, the post-embedding
state is the bare token embedding, so $\rho_E$ at $t = 0$ on the
activations should approximately equal the row-pooled embedding Procrustes
residual; the modest discrepancy reflects the per-pilot sampling).

**Tier 3: the mid-network is not aligned across seeds, even up to
rotation.** At the mid-network layers ($t = 4-9$) the Procrustes
residual is at 0.64-0.70, corresponding to alignment quality at
about 61-62% of the scramble ceiling. The two seeds' mid-network
residual streams have substantially uncorrelated structure that no
orthogonal rotation can recover. The mid-network is the layer-region
where the network's internal feature basis is built up through
training, and that feature basis is seed-specific in a way that even
a per-layer rotation cannot bridge.

The mid-network layer 9 is the worst-aligned layer in our measurements
($\rho = 0.700$, ratio 0.613), and the trend from layer 2 onward
through layer 9 is monotonically worsening. The improvement from layer
9 through layer 13 is a recovery from this worst-case toward partial
alignment at the readout layers.

**The relationship between Tier 2 and the alignment failure of §3.7.**
§3.7 reported that cross-seed $R$-matrix alignment fails — even at
the top-1 direction, cross-seed angles are $\approx 89°$. The
present §6.4 result reports that cross-seed activation alignment
partially recovers at the I/O boundaries (residuals around 50% of
the scramble ceiling). These two findings are not in contradiction:
the activation-level alignment is a strictly weaker claim than the
$R$-matrix alignment. Two seeds can have activations that align up to
rotation at the level of overall direction (which is what the
activation-Procrustes measures) while having $R$-matrices that point
in essentially random directions relative to each other (which is
what the $R$-matrix Procrustes measures).

The relationship is that activation-Procrustes is sensitive to the
overall configuration of the activation cloud — its center, its
overall stretch, the orientation of its principal axes if those
principal axes happen to be reproducible. The $R$-matrix Procrustes is
sensitive to the *internal coordinate system* of the activation cloud
— the basis vectors that span its principal directions. The latter
is seed-specific even when the former is partially shared.

**The three-tier claim summarized:**

| Tier | Quantity | Cross-seed reproducibility |
|---|---|---|
| 1 | Basis-invariant statistics ($\lambda$, $\log\alpha$, eff. rank, within/between) | Agree across seeds at 2-3% relative spread |
| 2 | Residual-stream subspaces at I/O boundaries | Aligned across seeds up to rotation at ~50% of scramble ceiling |
| 3 | Residual-stream subspaces at mid-network | Not aligned across seeds, even up to rotation (~60-70% of scramble) |

This three-tier picture is more nuanced than the binary "$R$-matrix
alignment fails" finding of Phase 1 and gives the multi-view extension
a more substantive cross-seed claim: the framework's basis-invariant
statistics are the appropriate level of abstraction not because
nothing aligns across seeds, but because the alignment that exists is
layer-dependent and constrained by the I/O boundary geometry, with
the network interior being the locus of genuine seed-specificity.

### 6.5 Per-token covariance subspace non-independence

The structural finding of §5.1 — the forward within/between ratio
rises monotonically from $t = 0$ — is consistent with a linear-Gaussian
description of the per-input bundles: each input token $v$ produces a
bundle whose mean traces a path through the residual stream and whose
covariance describes the spread around that path. In the strictest
linear-Gaussian formulation, the per-input covariance $\Sigma_t(v)$
would be approximately *token-independent*: the noise process that
drives within-input variance acts the same way regardless of input,
just with different starting points.

This subsection tests the token-independence prediction and finds
that it fails sharply. The principal angles between per-token
covariance subspaces are nearly orthogonal at most layers, far above
the sample-noise baseline. The finding is reproducible across seeds,
robust to the choice of subspace dimensionality, and emerges very
early in training and persists through convergence.

**Test methodology.** For each input token $v$ in the top-20 forward
token set, we compute the centered SVD of the per-token activations
at each layer:

$$\tilde{X}_v^t = U_v^t S_v^t (V_v^t)^\top, \qquad U_v^t \in \mathbb{R}^{|\mathcal{E}_v| \times k}.$$

We keep the top-$k$ singular vectors (the leading $k$ principal
directions of the per-token covariance). For each pair $(v, w)$ of
tokens, we compute the principal angles between the two top-$k$
subspaces (the angles between $V_v^t$ and $V_w^t$ in
$\mathbb{R}^{H \times k}$). The principal angles are reported in
degrees. Smaller angles mean the subspaces align; larger angles mean
they don't.

We compute two null baselines:

- **Self-consistency null.** Split each token $v$'s pilots into two
  random halves, compute the top-$k$ subspace on each half, and
  compute principal angles between the two halves. This measures the
  principal angle that arises from sample noise alone when the
  underlying covariance is identical.

- **Random-subset null.** Compute principal angles between two
  randomly-chosen subsets of pilots (regardless of which token they
  correspond to). This measures the principal angle for random pilot
  subsets that draw from the marginal covariance.

The trained-pair principal angles are expected to fall somewhere
between these two nulls. The interpretation:

- If trained-pair angles are at the self-consistency null, the
  per-token covariances are statistically indistinguishable from
  "identical covariance + sample noise" — token-independence holds.

- If trained-pair angles exceed the self-consistency null, the
  per-token covariances are genuinely different beyond sample noise —
  token-independence fails by that much.

- If trained-pair angles exceed even the random-subset null, the
  per-token covariances are *more* different from each other than
  random pilot subsets are — pointing to a particularly structured
  form of per-token covariance specificity.

**Per-layer principal angles at convergence.** Table 6.6 reports the
first principal angle (smallest, best-alignment direction) and the
median principal angle across the top-20 components, for the trained
seed-0 model at convergence, alongside the two null baselines.

**Table 6.6: Per-token covariance principal angles at convergence (seed 0, $k = 20$).**

| Layer $t$ | Pair 1st (mean) | Self-null 1st | Random-null 1st | Pair median | Self-null median | Random-null median |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 4.61° | 0.00° | 7.75° | 57.59° | 0.00° | 72.16° |
| 1 | 37.83° | 6.11° | 28.26° | 63.05° | (broad) | 69.35° |
| 2 | 40.73° | 12.21° | 36.79° | 63.36° | (broad) | 73.43° |
| 3 | 56.73° | 15.73° | 37.52° | 77.37° | (broad) | 74.47° |
| 4 | 56.71° | 17.62° | 42.13° | 78.31° | (broad) | 74.23° |
| 5 | 56.16° | 19.90° | 45.56° | 77.35° | (broad) | 74.57° |
| 6 | 53.63° | 20.70° | 37.09° | 78.12° | (broad) | 74.27° |
| 7 | 56.58° | 22.36° | 39.86° | 79.28° | (broad) | 75.47° |
| 8 | 56.70° | 23.09° | 39.95° | 79.34° | (broad) | 74.82° |
| 9 | 55.73° | 22.73° | 35.14° | 78.62° | (broad) | 74.06° |
| 10 | 53.66° | 23.80° | 32.22° | 76.78° | (broad) | 70.64° |
| 11 | 51.38° | 24.98° | 32.29° | 74.79° | (broad) | 66.47° |
| 12 | 39.10° | 21.25° | 21.84° | 68.66° | (broad) | 55.35° |
| 13 | 40.12° | 22.03° | 21.00° | 69.32° | (broad) | 57.68° |

The trained-pair median angle averaged across layers is $73.0°$. The
self-consistency null median angle averaged across layers is $18.04°$.
The random-subset null median angle averaged across layers is $70.5°$.

**Three observations are immediate.**

1. **Trained-pair angles substantially exceed the self-consistency
   null.** At every layer except $t = 0$ (where both pair and self
   are small for the first angle), the pair angle is multiple times
   the self-consistency null. The per-token covariance subspaces are
   *not* statistically indistinguishable from "identical covariance
   + sample noise" — token-independence fails.

2. **Trained-pair angles are close to the random-subset null.** At
   most layers, the trained-pair angle is slightly above or equal to
   the random-subset null. The per-token covariance subspaces are
   approximately as different from each other as random pilot subsets
   are. Two arbitrary tokens' covariance subspaces are nearly as
   unrelated as if you'd ignored token identity entirely.

3. **The layer-dependence shows a U-shape.** The first principal
   angle is small at $t = 0$ ($4.61°$, near-aligned), rises sharply
   through $t = 1-3$ to plateau at ~55°, and falls slightly through
   the deeper layers to $\approx 40°$ at the boundary. The median
   angle has a similar U-shape with even larger amplitude — from
   $\approx 58°$ at $t = 0$ to $\approx 79°$ at the mid-network and
   $\approx 69°$ at the boundary.

The U-shape recovers the same I/O-boundary-vs-mid-network distinction
that the §6.4 Procrustes alignment showed. At the embedding layer
$t = 0$, the per-token covariance subspaces are partially aligned
(they all reflect the shared embedding structure). Through the
deeper layers, each token's bundle develops its own characteristic
covariance subspace, and these become nearly orthogonal to each
other by mid-network. At the output, the alignment partially recovers
because the unembedding projection imposes shared structure.

**k-robustness check.** To verify the finding is not an artifact of
the subspace dimensionality choice, we repeat the analysis for $k =
5, 10, 20, 50$. Table 6.7 reports the pair-median angle averaged across
layers for each $k$.

**Table 6.7: k-sweep of per-token covariance non-alignment (seed 0).**

| $k$ | Pair median (avg across layers) | Self-null 1st (avg) | Pair/self ratio |
|---|---:|---:|---:|
| 5 | 79.3° | 23.5° | 3.4 |
| 10 | 76.5° | 20.7° | 3.7 |
| 20 | 73.0° | 18.0° | 4.1 |
| 50 | 67.0° | 12.5° | 5.4 |

Across all four $k$ values, the trained-pair median angle is 67-79°
and the self-consistency null is 13-24°. The pair-vs-null gap is
substantial at every $k$ — the finding does not depend on the
particular choice of subspace dimensionality. The pair angles
decrease modestly with $k$ as we average over more directions
(decreasing from 79° at $k = 5$ to 67° at $k = 50$), but the gap to
the self-consistency null is always large.

**Cross-seed reproducibility.** Table 6.8 reports the pair-median
angle averaged across layers for each seed.

**Table 6.8: Cross-seed reproducibility of per-token covariance non-alignment ($k = 20$).**

| Seed | Pair median (avg across layers) | Self-null 1st (avg) |
|---|---:|---:|
| seed 0 | 73.0° | 18.04° |
| seed 1 | 73.5° | 17.9° |
| seed 2 | 73.2° | 17.9° |
| seed 3 | 73.2° | 18.0° |
| **mean** | **73.2°** | **18.0°** |
| **std** | 0.2° | 0.1° |
| **range** | 0.5° | 0.2° |

The cross-seed reproducibility is extraordinary: 0.5° range on the
pair median across 4 seeds, smaller than the cross-seed dispersion on
any other statistic we report in this paper. The finding is not
seed-specific; it is a near-deterministic property of trained
transformers at this architecture and recipe.

**Training-evolution trajectory.** The per-token covariance
non-orthogonality is present from the earliest training checkpoint we
examine. Table 6.9 reports the pair-median angle through training
for seed 0.

**Table 6.9: Per-token covariance non-orthogonality through training (seed 0, $k = 20$).**

| Step | Pair median (avg across layers) | Self-null 1st (avg) |
|---|---:|---:|
| 100 | 59.1° | 14.7° |
| 500 | 60.0° | 11.8° |
| 1000 | 61.3° | 13.2° |
| 2000 | 67.0° | 16.3° |
| 5000 | 70.5° | 17.6° |
| 5607 | 70.9° | 17.6° |
| 10000 | 71.8° | 17.8° |
| 24000 | 73.0° | 18.0° |

The most striking feature of this trajectory is that *the pair-median
angle is already 59.1° at step 100*, when the model has done only
about 250 gradient steps. The self-consistency null at step 100 is
14.7°, so the pair-self gap is already 44° — comparable to its
converged value. The per-token covariance non-orthogonality is not
gradually constructed by training; it is present essentially from the
beginning and intensifies modestly through training.

The increase from step 100 to step 24,000 is 13.9° (from 59.1° to
73.0°). The largest fraction of this increase happens in the
co-location window (between steps 1,000 and 11,000), with the pair
median going from 61.3° at step 1,000 to 72.0° at step 11,000 (about
10° of the 14° total increase). The pair median continues to grow
slowly through the late-training window, from 72.0° to 73.0° between
steps 11,000 and 24,000.

**Conclusion of the per-token covariance analysis.** The
linear-Gaussian framework's strictest formulation — token-independent
noise covariance — does not describe the trained model. Per-token
covariance subspaces are nearly orthogonal across tokens at most
layers, far above the sample-noise floor, reproducibly across seeds
and across subspace dimensionalities, with the non-orthogonality
present from very early in training.

A modified linear-Gaussian framework with token-dependent covariance
$\Sigma_t(v)$ remains tractable; the deviation we measure does not
invalidate the linear-Gaussian framing as a whole, but does invalidate
the strictest version of it. We return to the interpretive
significance of this finding in §7.

The juxtaposition of §6.5 with §6.4 is instructive. The marginal
basis-invariant statistics reproduce across seeds at 2-3%. The
embedding and readout subspaces are partially aligned across seeds
(40-50% of scramble ceiling). The mid-network subspaces are not
aligned across seeds even up to rotation (~60% of scramble ceiling).
The per-token covariance subspaces are *not even close to aligned
across tokens within a single seed* (73° pair median, near the
random-subset ceiling). The framework's marginal statistics are
extracting universal structure that survives substantial
per-token-specific organization at the level of individual covariance
subspaces.

### 6.6 Summary of multi-view dynamical findings

§6 reports five dynamical findings of distinct character. We summarize
them here and indicate how they fit into the paper's broader argument.

**The $\log\alpha$ hump and reverse-view $\lambda$-dip co-locate
in training-step coordinates** (§6.1). The marginal-view hump's peak
at step 5,607 and the conditional-view dip's minimum at step 5,014
are adjacent log-spaced checkpoints. The co-location is consistent
across all four seeds (every seed has the dip one checkpoint before
the hump). The layer-resolved version of this finding — the
mid-network reverse-view bulge's intensification window — confirms
that the co-location is two-dimensional, not just a coincidence of
two scalar time series. The training-dynamic event the marginal view
records as a hump in $\log\alpha$ is the same event the conditional
view records as a dip in $\lambda_{\text{rev}}$ accompanied by a
layer-localized bulge intensification.

**Random initialization produces a uniform-high within/between ratio
profile that training reshapes differentially** (§6.2). The trained-state
multi-view structural findings of §5 are reinterpretable as the residue
of differential reduction: training reduces the within/between ratios
at every layer but reduces them most strongly at the I/O boundaries
and least strongly at the mid-network. The forward crossover at
$t \approx 1.86$ is created by training; the reverse mid-network peak
is preserved as a residue of the random-init high-monotone profile.

**The training-dynamic heatmaps confirm layer-resolved co-location**
(§6.3). The within/between ratio heatmaps in the (training step, layer)
plane reveal the spatial-temporal pattern of training-induced changes.
The strongest changes happen in the co-location window of steps
2,000-11,000 at the mid-network layers, confirming that the multi-view
dynamical findings of §6.1 are layer-resolved and not just scalar-time
events.

**Cross-seed Procrustes alignment partially recovers at the I/O
boundaries** (§6.4). The U-shaped Procrustes residual profile gives a
three-tier picture of cross-seed reproducibility: basis-invariant
statistics agree tightly (Tier 1), I/O boundary layers align up to
rotation (Tier 2), mid-network layers do not align even up to
rotation (Tier 3). The three-tier picture refines the binary
"alignment fails" finding of §3.7 into a more nuanced characterization
of where in the network the seed-dependence is concentrated.

**Per-token covariance subspaces are nearly orthogonal across tokens**
(§6.5). The strict linear-Gaussian token-independence prediction fails.
Per-token covariance subspaces are at $\sim 73°$ median angle across
token pairs, near the random-subset ceiling, and the non-orthogonality
emerges essentially immediately after training begins.

The combined picture: the framework's marginal statistics are
extracting basis-invariant structure that is shared across seeds and
across token-conditioning, while the underlying residual-stream
subspaces are organized in seed-specific ways at the network interior
and token-specific ways at the per-input level. The marginal
basis-invariant statistics are the appropriate level of abstraction
precisely because they factor out these seed-specific and
token-specific organizations, leaving the universal functional content
that all valid instantiations of the architecture must share.

§7 turns to the interpretive significance of these findings — what
the multi-view extension does and does not buy the framework, the
relationship to the linear-Gaussian framing, and what the cross-seed
non-alignment tells us about what universality the framework actually
claims.

---
