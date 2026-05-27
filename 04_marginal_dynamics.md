## 4. Marginal-framework training dynamics

This section reports the training dynamics of the marginal (all-to-all)
basis-invariant statistics across the 50 log-spaced checkpoints. The
original framework reports only at-convergence measurements and is
therefore silent about training dynamics; the per-checkpoint trajectories
we report here, characterized at four independent seeds, were a
contribution of our prior Phase 1 study that we reproduce in full at
the same level of quantitative detail.

The §3 at-convergence snapshot describes where the marginal statistics
*land*; this section describes how they *get there*. Several features
of the trajectory are non-trivial: the $\log\alpha$ statistic is not
monotonic but exhibits a pronounced mid-training hump that peaks
around step 5,000-6,000 and then declines through training's end; the
boundary-layer effect that is so large at convergence is essentially
absent at initialization and emerges over the first 5,000 training
steps; the per-coordinate kurtosis bottoms out around step 2,000 and
rises through the rest of training; and the recovered linear flow
$R(t)$'s convergence to its final form does not match the eval-loss
convergence schedule.

These training-dynamic features are themselves cross-seed reproducible.
We report them here because the multi-view extension's central
dynamical claim in §6.1 is that conditional-view anomalies *co-locate*
with marginal-view anomalies in training-step coordinates. Without the
marginal-framework training dynamics characterized to the level of
detail in this section, "co-location" would be an underdefined claim;
with them, co-location becomes a concrete quantitative statement about
specific training-step windows where multiple anomalies occur
simultaneously.

### 4.1 What training dynamics we report and what data they come from

The training dynamics reported in this section are computed from the
50 saved checkpoints per seed described in §3.3. At each checkpoint
and for each seed, we have the saved analyzer output containing the
basis-invariant statistics $\{R(t), \Sigma_t, \lambda, \log\alpha,
r_{\text{eff}}(t), \kappa_t, \text{iso}(t), \text{angles}(t)\}$. The
training-dynamic trajectories we report are sequences of these
quantities indexed by checkpoint step, with cross-seed dispersion
computed at every checkpoint.

We restrict this section to the marginal (all-to-all) view. The
conditional-view training dynamics, including the co-location of the
reverse-view $\lambda$-dip with the marginal $\log\alpha$ hump that
the previous paragraph alludes to, are reported in §6.1 where they
constitute the multi-view extension's central dynamical finding.

We use the paper convention (mean of per-coordinate log-variance)
throughout this section for $\log\alpha$ and $\lambda$, noting that
our convention (log of mean per-coordinate variance) gives
qualitatively identical trajectories with a small offset. The Jensen
gap between the two conventions is approximately $+0.50$ at
initialization and decays to approximately $+0.03$ by training's end;
since this gap is itself a training-dynamic quantity (it tracks how
isotropic the per-coordinate variance becomes), the two conventions
trace slightly different paths even though their endpoints differ by
a constant offset. We report the paper convention as primary because
the original framework's published values are in that convention; the
our-convention trajectories are computed and saved alongside.

The 50 log-spaced checkpoints are listed in §3.3. The log-spacing was
chosen because residual-stream geometry changes rapidly in early
training (where each doubling of training step produces visible
changes) and slowly in late training (where many additional steps
produce only small refinements). Log-spacing gives roughly uniform
resolution per decade of training step, which is appropriate for
features (the $\log\alpha$ hump, the boundary anomaly emergence, the
kurtosis rise) that occur on different timescales. The trajectories
plotted in this section's figures use log-spaced x-axes throughout.

### 4.2 The $\log\alpha$ hump

In all four seeds, $\log\alpha$ is not monotonic during training. The
trajectory has the qualitative shape: an initial value near $-3.3$ at
step 100, a slight decrease to near $-3.5$ at step approximately 300,
a substantial rise through a broad peak centered around step
5,000-6,000, and then a fall back to its end-of-training value of
approximately $-3.28$.

**Per-seed hump characterization.** Table 4.1 reports the peak location
and peak height for each seed.

**Table 4.1: $\log\alpha$ hump characterization (paper convention).**

| Seed | Peak step | Peak $\log\alpha$ | Final $\log\alpha$ | Peak-to-final drop |
|---|---:|---:|---:|---:|
| seed 0 | 5607 | $-2.24$ | $-3.298$ | $1.06$ |
| seed 1 | 5014 | $-2.21$ | $-3.193$ | $0.98$ |
| seed 2 | 5607 | $-2.23$ | $-3.259$ | $1.03$ |
| seed 3 | 5607 | $-2.28$ | $-3.360$ | $1.08$ |
| **mean** | **5607** | **$-2.24$** | **$-3.277$** | **$1.04$** |
| **range** | one checkpoint interval | $0.07$ | $0.168$ | $0.10$ |

The peak-step values are 5607 in three seeds and 5014 in one seed —
adjacent log-spaced checkpoints, so the cross-seed range on peak step
is effectively one checkpoint interval. The peak-height values agree
to within 0.07 log units across seeds. The hump is a reproducible
training-dynamic feature.

We characterize the hump as a *plateau-shaped* peak rather than a sharp
spike. The trajectory rises from $\log\alpha \approx -2.62$ at step
1,000 to its plateau height of $\log\alpha \approx -2.24$ by step
approximately 2,000, stays at the plateau height through step
approximately 8,000, and then descends through the final phase of
training to its convergence value. The plateau is broad — about 6,000
training steps wide, or roughly a quarter of total training — and the
"peak step" labels in Table 4.1 are the highest individual checkpoint
within the plateau rather than a sharp local maximum.

**Mechanical interpretation.** The hump corresponds to a window in
training where the per-coordinate residual variance at large $\tau$
has its highest value relative to the local log-linear fit. The
residual stream geometry is most "spread out" during this window, in
the sense that the variance-prefactor $\alpha$ is at its highest. The
log-linear fit to the variance-scaling law is performed across the
14 layer states at each checkpoint; what changes through training is
the magnitude of the residual variances at the largest $\tau$
(specifically $\tau = L + 1 = 13$, the post-final-norm state), which
during the hump period is briefly larger than its convergence value
before contracting to the converged value through the post-final-norm
anomaly emergence (§4.4–§4.5).

The mid-training hump is consistent with the loss-vs-flow timing
analysis: at step 5,000, the model's eval loss is approximately 3.10
(vs the final value of 2.91), so the model has not yet reached its
final prediction performance, but the linear-flow geometry's
coordinate structure has substantially developed. The hump's plateau
appears to occupy a phase of training where the residual stream is
being sculpted to its final form while the model's prediction
accuracy continues to improve.

**Why we expect the hump to be informative.** The $\log\alpha$ statistic
summarizes the residual stream's variance behavior in a single
basis-invariant scalar. A monotonic decrease (the naive expectation if
training simply tightens the residual stream) would be consistent with
"training reduces variance"; what we see instead is a non-monotonic
trajectory in which the variance prefactor first grows, then shrinks.
This implies that something distinctive happens during training between
roughly steps 2,000 and 8,000 — the residual stream goes through a
phase of expanded variance before contracting to its converged
configuration. We will see in §6.1 that the conditional-view
$\lambda$-dip co-locates exactly with this hump, providing a more
mechanistically explicit picture of what the hump represents in terms
of the multi-view decomposition.

### 4.3 The $\lambda$ trajectory

The variance-scaling exponent $\lambda$ trajectory through training is
less dramatic than the $\log\alpha$ trajectory but still non-trivial.
The all-to-all $\lambda$ (paper convention, all-layer fit) starts near
$-0.10$ at step 100, rises monotonically through training, and reaches
its convergence value of $\lambda \approx 0.426$ by step 24,000.

**Per-checkpoint $\lambda$ trajectory.** Table 4.2 gives the cross-seed
mean $\lambda$ at representative checkpoints.

**Table 4.2: All-to-all $\lambda$ trajectory through training (paper convention, cross-seed mean).**

| Step | $\lambda$ (cross-seed mean) | Std across seeds |
|---|---:|---:|
| 100 | $-0.103$ | $0.011$ |
| 300 | $-0.030$ | $0.010$ |
| 1000 | $+0.135$ | $0.009$ |
| 2000 | $+0.198$ | $0.008$ |
| 5000 | $+0.265$ | $0.007$ |
| 10000 | $+0.310$ | $0.006$ |
| 24000 | $+0.362$ | $0.005$ |

The trajectory is monotonic and approximately log-linear in training
step. Cross-seed dispersion is small at every checkpoint (std $\leq
0.011$ at every step, or relative spread $< 4\%$ except very early in
training where the absolute value is itself near zero).

The most notable feature of the all-to-all $\lambda$ trajectory is its
*early-training sign change*. At the first checkpoint (step 100), the
exponent is negative ($\lambda \approx -0.10$), meaning the
per-coordinate residual variance decreases with $\tau$ in the
power-law fit. By step approximately 250 the exponent crosses zero,
and from there it is positive and growing. The negative-$\lambda$
early-training period is brief — by step 1,000 the exponent is
already well into the positive regime — but its existence at all is
worth noting: the basis-invariant variance-scaling exponent that the
framework characterizes is not just a small positive value throughout
training, it crosses zero at a definable training step.

**Per-view $\lambda$ trajectories.** The forward and reverse views
also have $\lambda$ trajectories through training. We report these
briefly here because they appear on the same plot as the all-to-all
trajectory in our reporting; the full discussion including the
co-location of the reverse-view $\lambda$-dip with the marginal
$\log\alpha$ hump is in §6.1.

- The forward (input-conditioned) $\lambda_{\text{fwd}}$ rises
  monotonically from approximately $+0.09$ at step 100 to $+0.535$ at
  step 24,000. It is the largest of the three $\lambda$ values at
  every checkpoint, and its trajectory is approximately a vertical
  translation of the all-to-all trajectory.

- The reverse (output-conditioned) $\lambda_{\text{rev}}$ rises from
  approximately $+0.12$ at step 100 to a local maximum of approximately
  $+0.22$ at step approximately 300, then *declines* through a broad
  dip whose minimum is at step 5014 (cross-seed mean
  $\lambda_{\text{rev}} = 0.184$), and then rises again to its
  convergence value of $+0.295$ at step 24,000.

The reverse $\lambda$ trajectory is the only one of the three that is
non-monotonic in training step. The non-monotonicity has the
qualitative shape of a dip (rise from 0.12 to 0.22, fall from 0.22 to
0.18, rise from 0.18 to 0.30) and is centered on the same training
window as the $\log\alpha$ hump (the dip minimum at step 5014; the
$\log\alpha$ hump peak at step 5607). The co-location of the
$\lambda$-dip with the $\log\alpha$ hump is one of the multi-view
extension's substantive findings and is the subject of §6.1.

### 4.4 Boundary-layer effect: emergence trajectory

The boundary-layer effect described at convergence in §3.6 is a learned
phenomenon, not a fixed structural property of the architecture. At
initialization, both boundary layer states fall close to the
inner-layer line; the boundary offset $\Delta\log\alpha$ is near zero.
The anomaly emerges over the first 5,000 training steps and plateaus
to its final value.

**Per-checkpoint boundary effect.** Table 4.3 gives the cross-seed mean
$\Delta\log\alpha = \log\alpha_{\text{boundary-excluded}} -
\log\alpha_{\text{all-layer}}$ at representative checkpoints.

**Table 4.3: Boundary effect emergence trajectory (paper convention, cross-seed mean).**

| Step | $\Delta\log\alpha$ | Std across seeds |
|---|---:|---:|
| 100 | $-0.05$ | $0.04$ |
| 200 | $-0.16$ | $0.05$ |
| 400 | $-0.27$ | $0.04$ |
| 1000 | $-0.32$ | $0.04$ |
| 2000 | $-0.41$ | $0.03$ |
| 5000 | $-0.46$ | $0.02$ |
| 10000 | $-0.47$ | $0.01$ |
| 24000 | $-0.478$ | $0.009$ |

The boundary effect reaches approximately 90% of its final magnitude
by step 5,000 ($\Delta\log\alpha = -0.46$ vs final $-0.478$), about
85% by step 2,000 ($-0.41$), about 67% by step 1,000 ($-0.32$), and
about 56% by step 400 ($-0.27$). Cross-seed standard deviation
contracts through training, from $0.04-0.05$ at the earliest
checkpoints to $0.009$ at convergence — the effect becomes both
larger in magnitude and more reproducible across seeds as training
progresses.

The fact that $\Delta\log\alpha$ at step 100 is already $-0.05$ rather
than $0.0$ is a small effect — the boundary states differ from the
inner-layer line by a small amount even at the earliest checkpoint we
analyze — but it is consistent with the at-step-100 model already
having done 100 gradient updates and beginning to develop the
boundary structure. We do not have a step-0 (random initialization)
boundary measurement in our standard pipeline; the random-init
analysis we report in §6.2 is a separate measurement on a
deliberately-untrained model.

### 4.5 Post-final-norm anomaly emergence

Within the boundary-layer effect described in §4.4 and §3.6, the
post-final-norm state ($t = 13$) is the more anomalous of the two
boundary states. At convergence its log-variance sits approximately
1.8 log units below the inner-layer fit; at initialization it sits
very close to the line.

**Per-checkpoint post-final-norm gap.** Table 4.4 gives the cross-seed
mean gap between the post-final-norm log-variance and the inner-layer
fit at that $\tau$.

**Table 4.4: Post-final-norm gap emergence (paper convention, cross-seed mean).**

| Step | Post-final-norm gap | Std across seeds |
|---|---:|---:|
| 100 | $-0.05$ | $0.04$ |
| 200 | $-0.18$ | $0.04$ |
| 300 | $-0.32$ | $0.05$ |
| 500 | $-0.51$ | $0.04$ |
| 1000 | $-0.96$ | $0.04$ |
| 2000 | $-1.51$ | $0.03$ |
| 5000 | $-1.80$ | $0.02$ |
| 10000 | $-1.82$ | $0.02$ |
| 24000 | $-1.83$ | $0.01$ |

The post-final-norm anomaly reaches near-final magnitude by step
approximately 5,000, then plateaus. The trajectory is monotonic
through the entire emergence window — the gap grows steadily from
zero at the start of training to its converged value, with no
overshoot or non-monotonic features. By step 5,000 the gap has
reached approximately 98% of its final value.

The mechanical interpretation is the same as at convergence (§3.6):
the RMSNorm operation rescales the residual stream's principal
directions toward a fixed-norm shell, reducing the per-coordinate
variance at the post-norm state relative to what the inner-layer
linear flow would extrapolate. The emergence trajectory says this
rescaling effect *develops* during training rather than being a
fixed architectural property of the RMSNorm layer. The RMSNorm layer
has learned scaling parameters (one per coordinate), and the training
dynamics of the post-final-norm gap correspond to those scaling
parameters' adaptation through training.

The emergence timing of the post-final-norm anomaly co-locates with
two other features: the $\log\alpha$ hump (peak at step 5,607) and the
broad flow-convergence kink (§4.6 below). All three features occupy
the same training window — approximately steps 2,000 to 8,000 — where
the linear-flow geometry is being sculpted. This window has come to
be a recurring landmark in our analysis; the §6 multi-view findings
extend the same window with additional anomalies that the conditional
views reveal.

### 4.6 Flow-distance trajectory

The normalized flow-distance trajectory $\hat{D}_k = (D_k - D_K) / (D_1
- D_K)$, where $D_k$ is the Frobenius distance from checkpoint $k$ to
the final checkpoint, declines from $\hat{D}_1 = 1.0$ at the first
checkpoint to $\hat{D}_K = 0.0$ at the final checkpoint. The decline
is approximately monotonic but with a distinctive feature: a small
mid-training bump in which the normalized distance is briefly larger
than the local trend.

**Per-checkpoint normalized flow distance.** Table 4.5 gives the
cross-seed mean trajectory at representative checkpoints.

**Table 4.5: Normalized flow distance through training (cross-seed mean).**

| Step | $\hat{D}_k$ | Std across seeds |
|---|---:|---:|
| 100 | $1.000$ | $0.000$ |
| 200 | $0.910$ | $0.012$ |
| 500 | $0.760$ | $0.015$ |
| 1000 | $0.585$ | $0.017$ |
| 2000 | $0.350$ | $0.018$ |
| 5000 | $0.205$ | $0.014$ |
| 7000 | $0.180$ | $0.013$ |
| 10000 | $0.123$ | $0.010$ |
| 15000 | $0.061$ | $0.007$ |
| 24000 | $0.000$ | $0.000$ |

The flow-distance trajectory has a sharp early-training decline
(approximately 60% of the total distance is covered in the first
1,000 steps) and a slower late-training decline. The cross-seed
dispersion is small throughout, with std at most $0.018$ on a quantity
that ranges from 0 to 1.

**The mid-training bump.** In all four seeds, the normalized distance
$\hat{D}_k$ exhibits a small bump centered around steps 5,000-10,000.
The bump's structure: from step 5,000 to step 7,000 the trajectory
flattens (changing from 0.205 to 0.180 over 2,000 steps, while the
trend predicts a continued decline), then between step 7,000 and step
10,000 it resumes declining at a rate steeper than the trend (changing
from 0.180 to 0.123 over 3,000 steps). The net effect is that
approximately 12% of the total flow-distance is "given back" between
steps 5,000 and 7,000 before being recovered between steps 7,000 and
10,000.

The bump is small in absolute magnitude (the entire feature is within
0.03 of the local trend) but is reproducible across all four seeds and
adds further evidence that the training process has a distinct
mid-training phase in which the residual-stream geometry is
restructuring. The bump co-locates with the $\log\alpha$ hump (peak at
step 5,607), the post-final-norm anomaly's emergence completion (step
~5,000), and the reverse-view $\lambda$-dip discussed in §6.1.

**Loss-vs-flow timing.** The eval loss trajectory through training is
smoothly declining. We tabulate it at representative checkpoints for
comparison with the flow-distance trajectory:

**Table 4.6: Eval loss trajectory through training (cross-seed mean, nats).**

| Step | Eval loss | Std across seeds |
|---|---:|---:|
| 100 | 5.42 | 0.01 |
| 300 | 4.81 | 0.01 |
| 1000 | 3.95 | 0.01 |
| 2000 | 3.55 | 0.01 |
| 5000 | 3.10 | 0.01 |
| 10000 | 3.00 | 0.005 |
| 24000 | 2.908 | 0.002 |

The eval loss reaches approximately 96% of its final improvement by
step 10,000 (improvement from $5.42 - 2.91 = 2.51$ to $5.42 - 3.00 =
2.42$, or 96% of total), and the remaining 4% improvement is achieved
over the final 14,000 training steps. The flow distance reaches the
same 96% threshold ($\hat{D}_k = 0.04$) only at step approximately
18,000.

This timing mismatch — the eval loss substantially converges by step
10,000 while the residual-stream's basis-dependent coordinate
structure continues to refine through step 18,000 — is consistent
with the framework's basis-invariant statistics being functional
universal features that are determined before the model reaches its
final training trajectory, while the model's specific learned
$R(t)$ basis continues to drift afterward without further affecting
loss. The basis-invariant statistics are the part of the residual
stream that "locks in" earlier than the absolute pose.

### 4.7 Per-coordinate kurtosis trajectory

The per-coordinate residual kurtosis trajectory is the most distinctive
of the basis-invariant trajectories, exhibiting both an early-training
decrease and a late-training increase. We report it here in detail
because the seed-1 outlier behavior described at convergence in §3.5
emerges entirely in the late-training phase.

**Per-checkpoint kurtosis trajectory.** Table 4.7 gives the cross-seed
mean $\langle|\kappa|\rangle$ at representative checkpoints. We also
report each seed individually, because the seed-1 trajectory diverges
from the cluster in the final third of training.

**Table 4.7: Per-coordinate kurtosis through training (paper convention).**

| Step | Cross-seed mean | seed 0 | seed 1 | seed 2 | seed 3 |
|---|---:|---:|---:|---:|---:|
| 100 | 0.78 | 0.81 | 0.75 | 0.80 | 0.76 |
| 300 | 0.51 | 0.53 | 0.49 | 0.52 | 0.50 |
| 1000 | 0.39 | 0.41 | 0.37 | 0.40 | 0.38 |
| 2000 | 0.35 | 0.37 | 0.34 | 0.36 | 0.33 |
| 5000 | 0.43 | 0.45 | 0.43 | 0.44 | 0.40 |
| 10000 | 0.63 | 0.65 | 0.69 | 0.64 | 0.55 |
| 13000 | 0.77 | 0.78 | 0.86 | 0.78 | 0.66 |
| 17000 | 0.92 | 0.85 | 1.10 | 0.92 | 0.82 |
| 24000 | 1.046 | 0.871 | 1.334 | 1.045 | 0.932 |

The trajectory has three phases. From step 100 to step approximately
2,000 the kurtosis declines from $\approx 0.78$ to a minimum of
$\approx 0.35$. From step 2,000 to step approximately 13,000 the
kurtosis rises smoothly across all four seeds at approximately the
same rate, reaching $\approx 0.77$ by step 13,000 with small
cross-seed dispersion. From step 13,000 to step 24,000 the kurtosis
trajectories of the four seeds diverge: seed 1 accelerates more
sharply than the others, ending at $1.334$ while the others end at
$0.871-1.045$.

**Early-training decline (steps 100-2000).** The kurtosis decline from
$\approx 0.78$ to $\approx 0.35$ corresponds to the residual stream
becoming more Gaussian-like during the first few thousand training
steps. The random-initialized model has heavier-tailed per-coordinate
residuals (positive kurtosis $\approx 0.8$); training initially
*reduces* kurtosis toward the Gaussian baseline of zero excess
kurtosis. The minimum reached is $0.35$, still positive but
substantially closer to Gaussian than the initial value.

**Intermediate-training rise (steps 2000-13000).** From the kurtosis
minimum at step 2,000, the trajectory rises smoothly across all four
seeds. The rise is approximately the same magnitude in each seed
(from $\approx 0.35$ to $\approx 0.77$ over the same training window),
and cross-seed dispersion remains small through this phase ($\sigma
\approx 0.05$ at step 13,000, compared with $\sigma = 0.205$ at the
final checkpoint). This intermediate rise corresponds to the residual
stream becoming progressively more heavy-tailed as training continues,
but at the same rate across seeds.

**Late-training divergence (steps 13000-24000).** From step 13,000
onward, seed 1's kurtosis trajectory accelerates substantially more
than the other three seeds. Seed 1 rises from $0.86$ at step 13,000 to
$1.334$ at step 24,000 (an additional rise of $0.47$); the other three
seeds rise from approximately $0.74$ to approximately $0.95$ over the
same window (an additional rise of approximately $0.21$). Seed 1's
extra rise of $0.26$ over the others entirely accounts for the
$0.205$ cross-seed standard deviation reported at convergence in §3.5.

We have no mechanistic explanation for seed 1's late-training kurtosis
acceleration. The trajectory of other measured statistics in seed 1
(eval loss, $\lambda$, $\log\alpha$, effective rank, flow distance,
boundary effect, isotropy) is similar to the other seeds through the
same window; only kurtosis (and secondarily isotropy) shows the
divergence. The seed-1 behavior is consistent with a small
late-training optimization trajectory deviation that pushed the
residual distribution toward heavier tails without affecting other
basis-invariant quantities. We report this for completeness and flag
seed 1 as a kurtosis outlier throughout.

### 4.8 The unified training-dynamic timing landscape

The training-dynamic features reported in §4.2-§4.7 occupy three
distinct training-step windows that recur across the basis-invariant
statistics. We summarize the timing landscape here because it provides
the landmarks against which the §6 multi-view dynamical findings will
be calibrated.

**Early-training window (steps 100-1000): rapid early dynamics.** In
this window, all monotonic statistics undergo their fastest changes.
The flow distance falls from $\hat{D} = 1.0$ to $\hat{D} \approx 0.59$
(41% of total distance covered). The boundary effect emerges from
$\Delta\log\alpha = -0.05$ to $-0.32$ (67% of final magnitude). The
post-final-norm anomaly emerges from $-0.05$ to $-0.96$ (52% of final
magnitude). The kurtosis declines from $0.78$ to $0.39$ (the minimum
of the trajectory is at step 2,000 at $0.35$). $\lambda$ crosses zero
at approximately step 250. The eval loss falls from $5.42$ to $3.95$
(58% of total improvement).

This window is dominated by initialization-corrective behavior — the
network is moving away from its random-initialized state toward the
broad shape of a trained network, with most basis-invariant statistics
undergoing their largest derivatives in this phase.

**Mid-training window (steps 2000-10000): the "phase 2" window.** This
window contains the $\log\alpha$ hump (plateau from step 2,000 to step
8,000, peak at step 5,607 with cross-seed mean $-2.24$), the
reverse-view $\lambda$-dip (minimum at step 5,014 with cross-seed mean
$0.184$), the post-final-norm anomaly's emergence completion (from
$-1.51$ at step 2,000 to $-1.82$ at step 10,000), the boundary effect's
plateau (from $-0.41$ at step 2,000 to $-0.47$ at step 10,000), and the
flow-distance mid-training bump (between steps 5,000 and 10,000).

We refer to this as the "phase 2" window (after the residual stream's
phase of restructuring) because multiple basis-invariant statistics
exhibit non-monotonic or distinctive behavior in this window
simultaneously, suggesting a coordinated training-dynamic phase. The
window is roughly bounded by the points at which the boundary effect
crosses 85% of its final magnitude (step 2,000) and the
$\log\alpha$ hump terminates (step ~10,000). The window is
log-symmetric around step ~4,500 in log-step coordinates.

The §6 multi-view dynamical findings live in this same window. The
reverse-view $\lambda$-dip occurs at step 5,014 (within the window);
the within/between ratio heatmap analysis we report in §6 shows that
the conditional-view structural anomalies intensify during the
co-location window of approximately steps 2,049 to 10,969, which is
essentially the same window. The structural and dynamical findings of
the multi-view extension and the marginal-framework training dynamics
of this section thus describe the same training-step phenomenon from
different vantage points.

**Late-training window (steps 10000-24000): refinement and divergence.**
In this window, the basis-invariant statistics undergo their slowest
changes. The flow distance falls from $\hat{D} = 0.12$ to $0.00$
(12% of total distance, over 60% of training duration). The eval
loss falls from $3.00$ to $2.908$ (4% of total improvement). The
boundary effect refines from $-0.47$ to $-0.478$. The
$\log\alpha$ descends from $-2.24$ (the hump plateau) back toward its
convergence value of $-3.28$ (a drop of $1.04$ in this phase).

The kurtosis trajectory diverges across seeds in this window: three
seeds rise smoothly from $\approx 0.63$ at step 10,000 to $\approx
0.95$ at step 24,000, while seed 1 accelerates to $1.33$ over the
same window. The divergence is confined to this late-training phase
and to the kurtosis (and secondarily isotropy) statistic. We do not
know what is specific about seed 1's optimization trajectory in this
window.

### 4.9 What the marginal-framework training dynamics establish

The marginal training-dynamic features documented in this section are
cross-seed reproducible to within small dispersion bounds and span
three distinguishable training-step windows. The features include
both monotonic emergence trajectories (boundary effect, post-final-norm
anomaly, flow-distance convergence) and non-monotonic features
($\log\alpha$ hump, reverse-view $\lambda$-dip, mid-training
flow-distance bump, late-training kurtosis seed-1 divergence). The
non-monotonic features are the more distinctive because they cannot
be explained by simple "training reduces variance" or "training
sharpens the residual stream" pictures; they require a multi-phase
training dynamic in which the residual stream first expands or
restructures and then contracts to its converged state.

The §6 multi-view dynamical findings will use the training-step
landmarks defined here. Specifically:

1. The reverse-view $\lambda$-dip introduced in §4.3 has its minimum
   at step 5,014, within one log-spaced checkpoint of the
   $\log\alpha$ hump's peak at step 5,607. §6.1 reports this
   co-location as the multi-view extension's central dynamical
   finding.

2. The forward-view crossover layer (reported at convergence in §5)
   is essentially fully established by the end of the "phase 2"
   window — the crossover layer reaches its converged value of
   $t \approx 1.86$ by approximately step 5,000-7,000 and stays
   there. The within/between ratio's training-dynamic heatmap
   (§6.3) shows this stabilization in layer-resolved form.

3. The random-init baseline (§6.2) provides a step-0 reference point
   that complements the §4.4 step-100 measurements. The boundary
   effect at step 100 is already $-0.05$ (not zero); the random-init
   measurement at no training shows what the relevant within/between
   ratio quantities look like before any training has been applied,
   confirming that the boundary effect at step 100 is already in
   process.

4. The §3.7 cross-seed alignment failure is itself a training-dynamic
   finding in the sense that the $R(t)$ matrices that fail to align
   at convergence presumably do not fail to align at random
   initialization. We have not measured cross-seed $R$-matrix
   alignment at random initialization in this paper; the alignment
   failure at convergence is what the basis-invariant framework
   correctly handles, and §6.4's per-layer multi-view alignment
   analysis provides the at-convergence picture in more detail.

With this training-dynamic landscape established for the marginal
framework, §5 turns to the at-convergence structural findings of the
multi-view extension, and §6 returns to training dynamics in the
multi-view setting where the marginal-view landmarks of §4 are
joined by co-located conditional-view anomalies.

---
