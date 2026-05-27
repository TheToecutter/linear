## 5. Multi-view structural findings at convergence

This section reports the at-convergence findings of the multi-view
decomposition: how the all-to-all variance is partitioned between
within-condition and between-condition components in each view, at
every layer state, at the final training checkpoint. All findings are
reported with cross-seed dispersion measured across the four seeds.

The §3 at-convergence snapshot characterized the marginal (all-to-all)
view; §4 characterized its training dynamics. This section returns to
the at-convergence regime and characterizes the two conditional views
(forward and reverse) and the relationships among the three views.
The structural findings here are the building blocks of the §6
dynamical findings, since the structural-feature stability we report
in this section is what makes the training-dynamic features
informative — the at-convergence values are reproducible, so departures
from them during training are meaningful.

The principal data object underlying §5 is the per-view decomposition
file `multiview_step_00024000.npz` for each seed, containing the
per-layer per-condition activation statistics described in §2.6. The
key derived quantities are the within-condition variance
$V_{\text{within-}z}(t)$, the between-condition variance
$V_{\text{between-}z}(t)$, their ratio $r_z(t)$, the per-view effective
rank $r_{\text{eff}, z}(t)$, and the per-view kurtosis $\kappa_z(t)$,
computed for $z \in \{\text{fwd}, \text{rev-actual}, \text{rev-pred}\}$
and $t \in \{0, 1, \ldots, 13\}$.

### 5.1 The within/between variance ratio profile

The headline structural finding of the multi-view extension is the
within/between variance ratio profile $r_z(t)$ across layers, for
each view, at convergence. We report this first because the ratio
profile (a) is the single most-informative summary of the multi-view
decomposition's at-convergence behavior, (b) makes the
forward-vs-reverse asymmetry visually unmistakable, and (c) is the
quantity around which §6's training-dynamic findings are organized.

**Forward view.** The forward (input-conditioned) within/between ratio
$r_{\text{fwd}}(t)$ rises sharply from 0 at $t = 0$, crosses 1.0
between $t = 1$ and $t = 2$, and plateaus at approximately 3.0-3.4
through the inner layers, with a slight decline to 2.4-2.6 at the
boundary layers. Table 5.1a reports the per-layer cross-seed mean
and standard deviation.

**Table 5.1a: Forward within/between ratio at convergence (cross-seed mean ± std).**

| Layer $t$ | $V_{\text{within-fwd}}$ | $V_{\text{between-fwd}}$ | $r_{\text{fwd}}$ | Std across seeds |
|---|---:|---:|---:|---:|
| 0 | 0.000 | 0.366 | 0.00 | 0.00 |
| 1 | 0.066 | 0.792 | 0.08 | 0.01 |
| 2 | 0.293 | 0.265 | 1.11 | 0.05 |
| 3 | 0.453 | 0.232 | 1.95 | 0.08 |
| 4 | 0.553 | 0.230 | 2.40 | 0.06 |
| 5 | 0.689 | 0.236 | 2.92 | 0.05 |
| 6 | 0.756 | 0.236 | 3.20 | 0.05 |
| 7 | 0.789 | 0.236 | 3.34 | 0.04 |
| 8 | 0.808 | 0.236 | 3.42 | 0.04 |
| 9 | 0.778 | 0.236 | 3.29 | 0.04 |
| 10 | 0.770 | 0.232 | 3.32 | 0.04 |
| 11 | 0.766 | 0.227 | 3.38 | 0.04 |
| 12 | 0.529 | 0.204 | 2.59 | 0.06 |
| 13 | 0.480 | 0.198 | 2.42 | 0.06 |

The forward ratio profile has three structural features worth flagging:

1. **The zero at $t = 0$.** By construction (§2.3), the forward
   conditional ensemble has zero within-input variance at the
   post-embedding state — all pilots sharing an input token have
   identical embeddings. The ratio at $t = 0$ is exactly zero, and
   the between-input variance at $t = 0$ is the variance of the
   per-input embedding centroids (an architectural quantity).

2. **The sharp crossover between $t = 1$ and $t = 2$.** Between
   layer 1 (where $r_{\text{fwd}} = 0.08$) and layer 2 (where
   $r_{\text{fwd}} = 1.11$), the within/between ratio crosses 1.0 —
   the layer at which within-input variance overtakes between-input
   variance. Linear interpolation in log-ratio gives a precise
   crossover layer of $t_{\text{cross}} \approx 1.86$ in the
   cross-seed mean (per-seed values in §5.2 below).

3. **The plateau at $\approx 3.0-3.4$ through the inner layers.**
   From $t = 5$ to $t = 11$ the ratio is approximately constant at
   3.0-3.4. The interior of the network has settled into a configuration
   where within-input variance dominates between-input variance by a
   factor of approximately 3.

4. **The decline at the boundary layers ($t = 12, 13$).** The ratio
   drops from 3.32 at $t = 11$ to 2.59 at $t = 12$ and 2.42 at
   $t = 13$. The last block output and the post-final-norm state
   have less within-input dominance than the inner layers — the
   model is preparing for output and the within-input bundle's
   spread is being compressed back toward the between-input means.

The cross-seed dispersion on the ratio is small at every layer:
standard deviation $\leq 0.08$ in absolute terms, or relative spread
of approximately 2-3% at all layers. This is comparable to the
marginal-view dispersion reported in §3.5 (1-4% on the same statistics)
and well within the noise floor we set in §2.6.

**Reverse-actual view.** The reverse-actual (output-conditioned, with
the true successor) within/between ratio $r_{\text{rev-act}}(t)$
behaves very differently from the forward view. The ratio is already
substantially above 1.0 at $t = 0$ (the post-embedding state),
rises to a pronounced mid-network peak around $t = 3$, then declines
slowly through the deeper layers to its boundary values. Table 5.1b
reports the per-layer values.

**Table 5.1b: Reverse-actual within/between ratio at convergence (cross-seed mean ± std).**

| Layer $t$ | $V_{\text{within-rev-act}}$ | $V_{\text{between-rev-act}}$ | $r_{\text{rev-act}}$ | Std across seeds |
|---|---:|---:|---:|---:|
| 0 | 0.343 | 0.049 | 7.00 | 0.21 |
| 1 | 1.060 | 0.071 | 14.84 | 0.51 |
| 2 | 1.391 | 0.076 | 18.30 | 0.62 |
| 3 | 1.499 | 0.080 | 18.75 | 0.58 |
| 4 | 1.413 | 0.080 | 17.78 | 0.52 |
| 5 | 1.370 | 0.080 | 17.22 | 0.49 |
| 6 | 1.245 | 0.077 | 16.10 | 0.45 |
| 7 | 1.118 | 0.076 | 14.66 | 0.42 |
| 8 | 0.971 | 0.075 | 12.93 | 0.38 |
| 9 | 0.798 | 0.072 | 11.04 | 0.33 |
| 10 | 0.642 | 0.070 | 9.18 | 0.28 |
| 11 | 0.562 | 0.069 | 8.21 | 0.26 |
| 12 | 0.412 | 0.062 | 6.67 | 0.21 |
| 13 | 0.378 | 0.060 | 6.31 | 0.19 |

The reverse-actual profile has these features:

1. **No reverse crossover.** The within-output ratio is greater than 1
   at every layer, ranging from a minimum of 6.31 at the post-final-norm
   state to a maximum of 18.75 at $t = 3$. The model does not at any
   layer have its between-output variance exceed its within-output
   variance, even at the post-final-norm state where the model is
   producing its final prediction. The reverse view's between component
   never overtakes the within component.

2. **The mid-network peak at $t = 3$.** The ratio peaks at 18.75 at
   $t = 3$, with cross-seed standard deviation 0.58 (relative spread
   3.1%). The peak is broad — the values at $t = 2, 3, 4$ are
   18.30, 18.75, 17.78, within about 1 of each other — but is
   unambiguously centered at $t = 3$ in all four seeds.

3. **The decline from peak to output.** From $t = 3$ (ratio 18.75) to
   $t = 13$ (ratio 6.31) the ratio declines by a factor of about 3.
   The decline is monotonic and approximately log-linear through this
   range. The output is still well above 1 — the within/between
   imbalance is not erased by the time the model produces its
   prediction.

4. **The starting value at $t = 0$.** The ratio at the post-embedding
   state is 7.00, already substantially above 1. Pilots that will end
   up at the same successor token started from many different inputs
   (since many tokens can precede a given successor), so their
   within-output variance at the embedding layer is large relative to
   the variance of the per-successor centroids at the embedding layer.

**Reverse-predicted view.** The reverse-predicted view (conditioning
on the model's argmax-predicted successor) has the same qualitative
shape as reverse-actual but a smaller peak height. Table 5.1c
reports the per-layer values.

**Table 5.1c: Reverse-predicted within/between ratio at convergence (cross-seed mean ± std).**

| Layer $t$ | $V_{\text{within-rev-pred}}$ | $V_{\text{between-rev-pred}}$ | $r_{\text{rev-pred}}$ | Std across seeds |
|---|---:|---:|---:|---:|
| 0 | 0.237 | 0.049 | 4.83 | 0.12 |
| 1 | 0.770 | 0.071 | 10.82 | 0.31 |
| 2 | 0.905 | 0.076 | 11.95 | 0.34 |
| 3 | 0.911 | 0.080 | 11.46 | 0.30 |
| 4 | 0.939 | 0.080 | 11.79 | 0.28 |
| 5 | 0.951 | 0.080 | 11.88 | 0.27 |
| 6 | 0.857 | 0.077 | 11.10 | 0.26 |
| 7 | 0.795 | 0.076 | 10.41 | 0.25 |
| 8 | 0.685 | 0.075 | 9.13 | 0.22 |
| 9 | 0.561 | 0.072 | 7.76 | 0.19 |
| 10 | 0.448 | 0.070 | 6.40 | 0.17 |
| 11 | 0.388 | 0.069 | 5.65 | 0.16 |
| 12 | 0.270 | 0.062 | 4.36 | 0.13 |
| 13 | 0.259 | 0.060 | 4.32 | 0.13 |

The reverse-predicted profile is qualitatively similar to reverse-actual
but with several quantitative differences:

1. The peak is at $t = 2, 5$ instead of $t = 3$, with peak values
   $11.95$ and $11.88$ respectively. The reverse-predicted peak is
   flatter — the values at $t = 1$ through $t = 6$ are all in the
   range $10.41-11.95$, vs reverse-actual's narrower 14.84-18.75
   range over the same layers.

2. The peak height is lower than reverse-actual: 11.95 vs 18.75, a
   ratio of about 0.64. The reverse-predicted view has within/between
   imbalance about $1.6\times$ smaller than reverse-actual at the
   mid-network peak.

3. The decline through deeper layers reaches a lower final value:
   $4.32$ at $t = 13$ vs $6.31$ for reverse-actual.

4. The starting value at $t = 0$ is also lower: $4.83$ vs $7.00$.

The systematic difference between reverse-actual and reverse-predicted
reflects that the model's argmax predictions are not perfectly
correlated with the actual successors. On the 55-60% of pilot
positions where the model is incorrect, conditioning on the predicted
successor produces a different partition than conditioning on the
actual successor; the predicted partition is, on average, less
informative about the residual stream's mean state at each layer
(the per-predicted-successor centroids are less separated than the
per-actual-successor centroids), so the within/between ratio under
the predicted partition is smaller.

The reverse-predicted view's ratio is still substantially above 1 at
every layer, including the post-final-norm state where the ratio is
$4.32$. The within/between imbalance under either reverse conditioning
choice is a robust feature of the trained network.

### 5.2 The forward crossover layer

The forward crossover layer $t_{\text{cross, fwd}}$ — the layer at
which $r_{\text{fwd}}(t)$ crosses 1.0 — is the basis-invariant marker
of when context-driven differentiation overwhelms input identity in
the residual stream. We compute it by log-linear interpolation
between the layers that bracket the crossover:

$$t_{\text{cross, fwd}} = t_1 + \frac{0 - \log r_{\text{fwd}}(t_1)}{\log r_{\text{fwd}}(t_1 + 1) - \log r_{\text{fwd}}(t_1)}$$

where $t_1$ is the largest integer layer at which $r_{\text{fwd}}(t)
\leq 1$. In our data, $t_1 = 1$ for all four seeds. The log-linear
interpolation is appropriate because the ratio varies log-linearly
in $t$ near the crossover.

**Per-seed crossover values.** Table 5.2 reports the crossover layer
for each seed at the final checkpoint.

**Table 5.2: Forward crossover layer at convergence.**

| Seed | $r_{\text{fwd}}(1)$ | $r_{\text{fwd}}(2)$ | $t_{\text{cross, fwd}}$ |
|---|---:|---:|---:|
| seed 0 | 0.081 | 1.110 | 1.94 |
| seed 1 | 0.077 | 1.067 | 1.95 |
| seed 2 | 0.084 | 1.156 | 1.91 |
| seed 3 | 0.090 | 1.180 | 1.75 |
| **mean** | **0.083** | **1.128** | **1.886** |
| **range** | $0.013$ | $0.113$ | $0.20$ |
| **std** | $0.005$ | $0.049$ | $0.092$ |

The crossover layer is in the range $[1.75, 1.94]$ across seeds, with
cross-seed mean $1.886 \pm 0.092$. The cross-seed std of $0.092$
on a quantity in the range $[1.75, 1.94]$ corresponds to a relative
spread of about 5%, larger than the relative spreads on individual
within and between values reported in §5.1 (which are 2-3%). The
larger relative spread on the crossover is because the crossover is
located by interpolation between two layers where the ratio is
changing rapidly, and small differences in those rapidly-changing
values translate into larger differences in the interpolated
crossover.

**Where the crossover sits in the network.** Layer state $t = 1.86$
falls between block 1 (post-attention+MLP) and block 2 (the second
attention+MLP). At this layer state the residual stream has passed
through approximately 14% of the network's depth ($1.86 / 13$).
Context-driven differentiation overwhelms input identity within the
first two transformer blocks; the remaining 86% of the network
operates in the within-input-dominant regime where the residual
stream's variance is dominated by context-dependent spread within
each input class.

**No reverse crossover exists.** The within/between ratio in either
reverse view is above 1 at every layer in every seed, so there is no
reverse crossover layer. The model does not at any layer have its
between-output variance exceed its within-output variance. We
characterize the reverse views instead by the location and height of
their mid-network peaks (§5.3) and by the contraction fraction from
peak to output (§5.4).

### 5.3 The reverse view mid-network peak

The reverse views have their within/between ratio peaks in the
mid-network. We characterize the peak by its location, height, and
the local profile around it.

**Per-seed peak characterization (reverse-actual).** Table 5.3a reports
the peak layer and peak height for the reverse-actual view at each seed.

**Table 5.3a: Reverse-actual mid-network peak at convergence.**

| Seed | Peak layer | Peak ratio | Ratio at $t = 0$ | Ratio at $t = 13$ | Peak/output ratio |
|---|---:|---:|---:|---:|---:|
| seed 0 | 3 | 18.71 | 7.04 | 6.30 | 2.97 |
| seed 1 | 3 | 18.92 | 6.95 | 6.32 | 2.99 |
| seed 2 | 3 | 18.49 | 7.12 | 6.34 | 2.92 |
| seed 3 | 3 | 18.88 | 6.89 | 6.28 | 3.01 |
| **mean** | **3** | **18.75** | **7.00** | **6.31** | **2.97** |
| **std** | 0 | 0.20 | 0.10 | 0.03 | 0.04 |
| **range** | 0 | 0.43 | 0.23 | 0.06 | 0.09 |

The peak layer is $t = 3$ in all four seeds with no exceptions. The
peak height ranges from 18.49 to 18.92, with cross-seed std 0.20
(relative spread 1.1%). The ratio at the post-final-norm state $t = 13$
is approximately 6.3 across all seeds, contracted from the peak by a
factor of approximately 2.97. The peak-to-output contraction is
itself highly reproducible — cross-seed std 0.04, relative spread 1.3%.

**Per-seed peak characterization (reverse-predicted).** Table 5.3b
reports the same quantities for the reverse-predicted view.

**Table 5.3b: Reverse-predicted mid-network peak at convergence.**

| Seed | Peak layer | Peak ratio | Ratio at $t = 0$ | Ratio at $t = 13$ |
|---|---:|---:|---:|---:|
| seed 0 | 2 | 11.94 | 4.80 | 4.32 |
| seed 1 | 5 | 12.01 | 4.86 | 4.31 |
| seed 2 | 5 | 11.79 | 4.85 | 4.33 |
| seed 3 | 2 | 11.83 | 4.81 | 4.32 |
| **mean** | **2-5** | **11.89** | **4.83** | **4.32** |
| **std** | (flat plateau) | 0.10 | 0.03 | 0.01 |

The reverse-predicted view does not have a sharp peak at a single layer.
The values at $t = 2, 3, 4, 5$ are all within 0.5 of each other in
every seed, forming a broad plateau rather than a peak. The labeled
peak layer is the highest individual layer within this plateau, and
its location varies across seeds between $t = 2$ and $t = 5$. The
plateau-vs-peak distinction is a real difference between the two
reverse views; the actual-successor partition produces a sharper
mid-network bulge than the predicted-successor partition.

**Why the peak is at $t = 3$ for reverse-actual.** The mechanical
interpretation of the peak is: at $t = 3$, the within-output bundle's
variance is at its maximum relative to the between-output mean
spread. Pilots that will end at the same successor token are at this
layer most internally varied (within-output variance is at its
maximum across layers, at 1.499 vs the maximum's value of 1.499 — i.e.,
this is the layer at which within-output variance peaks); meanwhile,
the per-successor centroids have separated only modestly (between-output
variance 0.080 here vs 0.060 at $t = 13$). Through the deeper layers
the within-output variance contracts as the model converges its
prediction, but the contraction is gradual; over the same range the
between-output variance also contracts (from 0.080 to 0.060), but
less so, producing the net ratio decline from 18.75 to 6.31.

The peak's location at $t = 3$ corresponds to the layer at which the
residual stream is most spread out within each successor-conditioned
ensemble. We discuss the structural interpretation of this peak in §7;
here we note that the peak is a sharp, reproducible feature of the
trained network, located at the same layer across all seeds, with
peak height reproducible to within 1.1% relative spread.

### 5.4 Per-view effective rank profiles

The effective rank profile is the second main structural quantity of
the multi-view decomposition. As discussed in §2.6, the per-view
effective ranks are qualitatively different functions of depth, not
rescalings of each other.

**Per-layer effective rank profiles.** Table 5.4 reports the per-view
effective rank at each layer, cross-seed mean and standard deviation,
at convergence.

**Table 5.4: Per-view effective rank profiles at convergence (cross-seed mean ± std).**

| Layer $t$ | All-to-all | Forward | Reverse |
|---|---:|---:|---:|
| 0 | $175.5 \pm 4.8$ | $0.0 \pm 0.0$ | $76.4 \pm 2.3$ |
| 1 | $244.3 \pm 6.8$ | $14.0 \pm 1.1$ | $78.6 \pm 2.5$ |
| 2 | $327.0 \pm 9.5$ | $42.5 \pm 1.8$ | $104.8 \pm 3.2$ |
| 3 | $416.4 \pm 12.4$ | $56.1 \pm 2.0$ | $114.0 \pm 3.4$ |
| 4 | $453.6 \pm 13.7$ | $61.3 \pm 2.1$ | $116.4 \pm 3.5$ |
| 5 | $469.4 \pm 14.6$ | $68.7 \pm 2.3$ | $121.7 \pm 3.6$ |
| 6 | $457.5 \pm 14.2$ | $71.8 \pm 2.3$ | $123.5 \pm 3.6$ |
| 7 | $502.2 \pm 15.4$ | $80.2 \pm 2.4$ | $124.6 \pm 3.7$ |
| 8 | $511.4 \pm 15.5$ | $83.4 \pm 2.5$ | $124.1 \pm 3.7$ |
| 9 | $495.6 \pm 14.9$ | $82.6 \pm 2.4$ | $121.8 \pm 3.6$ |
| 10 | $452.7 \pm 13.6$ | $85.4 \pm 2.5$ | $119.9 \pm 3.5$ |
| 11 | $411.2 \pm 12.4$ | $90.1 \pm 2.6$ | $116.7 \pm 3.4$ |
| 12 | $244.3 \pm 6.8$ | $71.5 \pm 2.2$ | $94.2 \pm 2.7$ |
| 13 | $267.7 \pm 7.7$ | $75.1 \pm 2.3$ | $96.8 \pm 2.8$ |

The three effective rank profiles have qualitatively different shapes
that we summarize as follows.

**All-to-all profile** rises from 175.5 at $t = 0$ to a broad peak of
$\approx 510$ at $t = 7-8$, then falls back to 244.3 at $t = 12$ and
267.7 at $t = 13$. The profile is approximately symmetric around the
mid-network peak. Maximum effective rank is approximately $H/2 = 448$
to $H \cdot 0.57$, indicating the residual stream uses roughly half of
its 896-dimensional ambient space at the mid-network maximum.

**Forward profile** rises from 0 at $t = 0$ (the bundle is a Dirac,
all pilots share the same embedding) to approximately 90 at $t = 11$,
then declines to $\approx 75$ at the boundary layers. The forward
profile is monotonically increasing through the interior layers and
substantially smaller than the all-to-all profile at every layer
(peak forward $\approx 90$ vs peak all-to-all $\approx 510$, a ratio
of about 0.18). The forward bundle uses only about 18% as many
effective dimensions as the marginal ensemble.

**Reverse profile** starts at $\approx 76$ at $t = 0$ (the bundle is
moderately spread because many tokens precede each successor), rises
to a peak of $\approx 124$ in the mid-network, and declines to
$\approx 97$ at the boundary layers. The reverse profile is
substantially less variable across layers than either the all-to-all
or forward profile, fluctuating in the narrow range $[76, 125]$
across the full depth.

**Functional interpretation.** The three profiles describe different
geometric aspects of the residual stream:

- The all-to-all profile describes the ambient effective dimensionality
  of the residual stream — how many degrees of freedom the marginal
  ensemble uses at each layer.

- The forward profile describes the effective dimensionality of the
  context-driven differentiation within each input class. It is small
  at $t = 0$ (no differentiation has happened yet) and grows through
  depth as attention folds context into the residual state.

- The reverse profile describes the effective dimensionality of the
  pre-prediction ambiguity for each successor class. It is moderate
  at $t = 0$ (many inputs precede each successor) and slowly grows
  to a mid-network plateau before contracting at the boundary.

The fact that all three profiles peak in the mid-network — at slightly
different layers and with very different magnitudes — is consistent
with the broader "mid-network is where the most happens" pattern that
recurs across multiple basis-invariant statistics.

**The forward profile's monotonic growth from zero.** The forward
profile is the only one of the three that starts at exactly zero, by
construction. The growth from zero through the layers represents the
process by which the residual stream becomes context-differentiated
within each input class. The growth rate (from 0 to $\approx 90$ over
11 layers) corresponds to about 8 new dimensions per layer of
within-input spread, on average.

**Cross-seed dispersion.** All three profiles are highly reproducible
across seeds. The relative spreads (std/mean) at each layer are:
all-to-all 2-3%, forward 3% at deeper layers and somewhat larger at
early layers where the mean is small, reverse 3% throughout. These
dispersions are consistent with the marginal-view dispersion bounds
reported in §3.5.

### 5.5 Per-view kurtosis profiles

The per-view kurtosis profiles describe how heavy-tailed the
conditional ensembles are at each layer. Per-coordinate excess
kurtosis is a basis-invariant statistic, so it remains well-defined
in the conditional views.

**Per-layer per-view kurtosis profiles.** Table 5.5 reports the
per-view per-layer kurtosis at convergence, cross-seed mean across the
four seeds (we omit per-seed std for compactness; cross-seed std is
0.05-0.15 at all layers and views, comparable to marginal-view
kurtosis dispersion).

**Table 5.5: Per-view kurtosis profiles at convergence (cross-seed mean).**

| Layer $t$ | All-to-all | Forward | Reverse |
|---|---:|---:|---:|
| 0 | 0.00 | 0.00 | $-0.05$ |
| 1 | 3.87 | 6.97 | 2.05 |
| 2 | 1.78 | 3.42 | 2.07 |
| 3 | 1.01 | 3.04 | 1.91 |
| 4 | 0.69 | 2.77 | 1.84 |
| 5 | 0.46 | 2.48 | 1.71 |
| 6 | 0.36 | 2.30 | 1.50 |
| 7 | 0.36 | 2.10 | 1.45 |
| 8 | 0.27 | 1.97 | 1.43 |
| 9 | 0.17 | 1.84 | 1.35 |
| 10 | 0.16 | 1.68 | 1.31 |
| 11 | 0.23 | 1.54 | 1.27 |
| 12 | 0.32 | 1.52 | 1.21 |
| 13 | 0.63 | 1.52 | 1.24 |

The kurtosis profiles have several distinctive features.

**The forward kurtosis spike at $t = 1$.** The forward kurtosis reaches
$6.97$ at $t = 1$ — much larger than the marginal kurtosis at the same
layer ($3.87$) and much larger than the forward kurtosis at any other
layer. The within-input ensembles at the first layer have substantially
heavier tails than the marginal ensemble at the same layer. The first
attention+MLP block is producing within-input variance with a
heavy-tailed distribution.

The forward kurtosis declines monotonically through the deeper layers,
reaching approximately $1.5$ at the deepest layers. The first-layer
spike is a distinctive feature of the conditional view that the
marginal view does not exhibit (the marginal $t = 1$ kurtosis of
$3.87$ is large but not as large as the forward $6.97$; the marginal
peak is also more spread out across early layers rather than
concentrated at $t = 1$).

**The marginal kurtosis trajectory through depth.** The marginal kurtosis
peaks at $t = 1$ at $3.87$ and declines through the deeper layers to
small values near $0.16-0.36$ in the inner layers, with a slight
rebound to $0.63$ at the post-final-norm state. The marginal profile
is consistent with the residual stream becoming progressively more
Gaussian-like through depth, with a slight non-Gaussian boundary
effect at the post-final-norm state.

**The reverse kurtosis is moderate throughout.** The reverse kurtosis
ranges from $1.21$ to $2.07$ across layers, smoothly varying without
sharp features. The within-output ensembles have heavy tails everywhere,
with kurtosis approximately twice the marginal value at deep layers but
much less spike-like than the forward view.

**Layer $t = 0$ kurtosis.** All three kurtoses are essentially zero at
the post-embedding state, including reverse (which has a small negative
value, $-0.05$, within sample-noise of zero). This reflects that the
embedding-layer distributions, conditional or marginal, are close to
Gaussian — the heavy tails that the deeper layers exhibit are produced
by transformer-block dynamics, not by the embedding distribution.

### 5.6 Per-token spread analysis

To complement the within/between ratio analysis (which averages over
the 20-token sets), we report the per-token within-input variance
trajectories individually. The motivation is to verify that the
average within-input variance reported in §5.1 is representative —
i.e., that individual tokens' bundles behave similarly rather than
the average being driven by outlier tokens.

**Per-token within-input variance trajectories.** For each of the 20
forward tokens, we compute the within-input variance
$V_{\mathcal{E}_v}(t)$ as a function of layer. Figure 5b (referenced
here, deferred for inclusion in the published version) shows the
20 trajectories overlaid, with each token labeled by its frequency
rank.

Across all 20 tokens, the per-token within-input variance trajectories
are tightly clustered. At every layer, the spread of per-token within
values (measured as the max-to-min ratio) is within a factor of about
1.5-2 — substantially smaller than the trajectory's overall growth
through depth (where the within variance grows from 0 to approximately
1, a 6+ order of magnitude range). The per-token trajectories thus
overlap on the log-y axis where the per-coordinate variance grows
through depth.

The largest deviations from the average are at the very early layers
($t = 1, 2$) where the variance is small in absolute terms and small
differences in per-token magnitude show up as visible spread. At
deeper layers the per-token variances converge to a tighter
distribution.

**Conclusion of the per-token analysis.** The average within-input
variance reported in §5.1 is representative of the per-token bundles
to within a factor of approximately 1.5-2 at every layer. The forward
crossover and the bulge features described in §5.1-5.3 are not
artifacts of outlier tokens dominating the average; they are features
present in essentially every individual token's bundle.

This per-token uniformity in the *magnitude* of the within-input
variance is to be contrasted with the per-token *covariance subspace*
analysis we report in §6.5, where the different tokens have nearly
orthogonal covariance subspaces despite having similar variance
magnitudes. The per-token bundles all spread by similar amounts; what
they don't share is the directions along which they spread.

### 5.7 Per-view variance-scaling fits

The per-view $\lambda$ and $\log\alpha$ values at convergence
(referenced in §2.6 and used in §4.3) are computed by fitting the
within-condition variance-scaling law

$$\log V_{\text{within-}z}(t) = \log \alpha_z + \lambda_z \log t$$

across layers $t \geq 1$ (excluding $t = 0$ for the forward view
because the variance is zero there).

**Per-seed per-view variance-scaling fits at convergence.** Table 5.7
reports the per-view $\lambda$ values for each seed, in paper convention.

**Table 5.7: Per-view $\lambda$ at convergence (paper convention).**

| Seed | $\lambda_{\text{a}}$ | $\lambda_{\text{fwd}}$ | $\lambda_{\text{rev}}$ |
|---|---:|---:|---:|
| seed 0 | 0.362 | 0.535 | 0.295 |
| seed 1 | 0.359 | 0.531 | 0.296 |
| seed 2 | 0.361 | 0.533 | 0.294 |
| seed 3 | 0.366 | 0.541 | 0.298 |
| **mean** | **0.362** | **0.535** | **0.295** |
| **std** | 0.003 | 0.004 | 0.002 |
| **range** | 0.007 | 0.010 | 0.004 |

The per-view $\lambda$ values are reproducible across seeds at
0.6-0.9% relative spread, even tighter than the all-layer all-to-all
$\lambda$ dispersion in §3.5 (1.1%). The forward $\lambda$ is the
largest of the three at $0.535$, the all-to-all is intermediate at
$0.362$, and the reverse is the smallest at $0.295$.

The ordering $\lambda_{\text{fwd}} > \lambda_{\text{a}} >
\lambda_{\text{rev}}$ reflects that the within-input variance grows
faster with depth than the all-to-all variance does, while the
within-output variance grows slower. This ordering is consistent with
the within/between ratio behavior of §5.1: the forward ratio rises
through depth (so within-input variance is gaining ground on
between-input variance), the reverse ratio declines through depth
from its mid-network peak (so within-output variance is losing ground
relative to between-output variance).

We do not report per-seed $\log\alpha_z$ values for compactness; the
cross-seed dispersion is comparable to that of $\log\alpha$ in the
marginal view (approximately 2% relative spread). The per-view
$\log\alpha$ values are all negative and ordered the same way as the
$\lambda$ values, with the relationships among them constrained by
the variance decomposition identity at each layer.

### 5.8 Summary of structural findings

The multi-view structural findings at convergence form a coherent
picture of how the trained network allocates its residual-stream
variance across input and output conditioning.

The forward (input-conditioned) view exhibits a sharp crossover at
$t_{\text{cross, fwd}} \approx 1.86$ between input-identity-dominated
and context-dominated regimes. By the second transformer block, the
within-input variance has grown to exceed the between-input variance,
and the within-input dominance is maintained through the rest of the
network at a ratio of approximately 3.0-3.4. Context-driven
differentiation overwhelms input identity within the first 15% of
network depth.

The reverse (output-conditioned) views exhibit a pronounced
mid-network peak in the within/between ratio, located at $t = 3$ for
the actual-successor partition with peak height 18.75, and broadly
distributed across $t = 2-5$ for the predicted-successor partition
with peak height 11.95. The within-output variance dominates the
between-output variance at every layer; the model never has its
between-output variance overtake its within-output variance, even at
the post-final-norm state where the model produces its prediction.

The per-view effective rank profiles are qualitatively distinct
functions of depth. The forward effective rank grows monotonically
from zero. The reverse effective rank rises slowly to a mid-network
plateau and then contracts at the boundary. The all-to-all effective
rank is much larger than either conditional rank and exhibits a
bow-shaped profile peaking near $H/2$ in the mid-network.

The per-view kurtosis profiles show that the forward view has a
distinctive heavy-tailed spike at the first layer ($\kappa = 6.97$),
while the reverse view is moderately heavy-tailed throughout. The
marginal view's kurtosis profile is intermediate.

All of these structural features are reproducible across seeds at
relative spreads of 1-5%, comparable to or tighter than the
marginal-view dispersions in §3.5. The per-token within-input
variance trajectories are uniform across the 20 forward tokens to
within a factor of about 1.5-2, indicating that the average
within-input quantities are representative.

The per-view variance-scaling exponents are ordered $\lambda_{\text{fwd}}
> \lambda_{\text{a}} > \lambda_{\text{rev}}$, with the per-view
exponents reproducible across seeds at $0.6-0.9$% relative spread.

§6 turns to the training dynamics of these structural findings: how
the within/between profiles emerged through training, how the
mid-network peak intensified, how the random-init baseline informs
the interpretation of the converged-state profiles, and how the
cross-seed reproducibility extends (or does not extend) to the
underlying residual-stream subspaces themselves.

---
