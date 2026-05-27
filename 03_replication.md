## 3. Model, pipeline, and at-convergence replication

This section specifies the trained model in full architectural detail,
the training recipe, the activation-collection pipeline, and the
at-convergence findings of the marginal (all-to-all) framework at our
scale. It establishes that the multi-view extension is being built on
a baseline faithful to the original framework's predictions, and it
gives the within-variant noise floors on the marginal statistics that
calibrate the conditional-view dispersion in §5–§6.

All measurements in this section are on the all-to-all view at the
final checkpoint (step 24,000), unless otherwise specified. Section §4
reports the training dynamics — how each of these at-convergence
statistics emerged through the 24,000 training steps. Numbers reported
here come from our prior Phase 1 study reapplied to the GELU-variant
model used throughout this paper; the Phase 1 study was completed on a
SwiGLU-variant model, and we re-ran it on the GELU variant to control
for activation-function confounds in the kurtosis statistic. The two
variants give qualitatively identical results on every statistic except
kurtosis, where the SwiGLU variant had a known per-token gating-induced
inflation that the GELU variant avoids.

### 3.1 Model architecture

The trained model is a 146.4M-parameter Llama-style decoder-only
transformer. The architecture mirrors small Llama-family conventions
in all respects except for the FFN activation function, which is GELU
rather than the more common SwiGLU. The full architectural specification
appears in Table 3.1.

**Table 3.1: Model architecture.**

| Property | Value |
|---|---|
| Total parameters | 146.4M |
| Hidden width $H$ | 896 |
| Transformer blocks $L$ | 12 |
| Attention heads | 14 (head dim 64) |
| KV heads | 2 (grouped-query attention) |
| FFN intermediate size | 2432 (outer dim) |
| FFN inner expansion | 3648 = $\lfloor 1.5 \times 2432 \rfloor$ (GELU expansion factor) |
| FFN activation | GELU |
| Position encoding | RoPE with base 10000 |
| Normalization | RMSNorm (pre-norm) |
| Tied embeddings | Yes (input embedding and unembedding share weights) |
| Vocabulary size | 32,768 (Mistral-7B-v0.1 BPE) |
| Maximum context | 4096 (training context 1024) |
| Layer states recovered by analyzer | 14 ($L_{\text{total}} = L + 2$) |

A few details on the architectural choices that affect the
basis-invariant statistics we report.

**FFN activation and parameter matching.** Our prior Phase 1 study
used SwiGLU as the FFN activation. SwiGLU's gating mechanism
introduces a per-token variability in the FFN output that inflates the
per-coordinate residual kurtosis without affecting other basis-invariant
statistics, and the kurtosis dispersion in that study (19.6% relative
spread across seeds, dominated by a single outlier) was traced to this
gating-induced contamination. For the present paper we re-ran the same
4-seed pilot using GELU (ungated) as the FFN activation. The two
variants are parameter-matched: SwiGLU has FFN intermediate size 2432
with a separate gate projection, GELU has FFN intermediate size 2432
with an inner dimension of $\lfloor 1.5 \times 2432 \rfloor = 3648$
that accommodates the absence of the gate projection within the same
parameter budget. The two architectures are designed to be FLOPs-matched
to within 1%. The GELU variant gives qualitatively identical results on
$\lambda$, $\log\alpha$, effective rank, isotropy, and all other
basis-invariant statistics; we report the GELU variant's numbers
throughout this paper because they are the cleaner baseline. The SwiGLU
variant's results are reported in our prior Phase 1 study.

**Tied embeddings.** Input and output embeddings share weights:
$E_{\text{out}} = E_{\text{in}}^\top$. This is the convention used in
Llama, Mistral, and most modern small models. Tied embeddings constrain
the residual-stream geometry at the boundary layers (specifically the
post-final-norm state $t = L + 1$, from which logits are produced by
projection onto $E^\top$) in ways that the §6.4 cross-seed alignment
results reflect: shared vocabulary plus tied embeddings produces I/O
boundary alignment across seeds that is partially recovered by
Procrustes, while the network interior is not.

**Position encoding.** RoPE (rotary position embeddings) applies the
positional transformation inside attention rather than to the
embeddings directly. The post-embedding state $t = 0$ is therefore the
unmodified token embedding, with no positional information added. This
means $V_{\text{within-fwd}}(0) = 0$ for all forward conditional
ensembles exactly, not approximately. Architectures using absolute
positional embeddings (added to the token embedding at $t = 0$) would
have small but nonzero within-input variance at the embedding layer
due to position variation across pilots.

**Layer-state convention.** The 14 layer states recovered by the
analyzer comprise the post-embedding state ($t = 0$), the 12
post-block-output states ($t = 1, \ldots, 12$), and the post-final-norm
state ($t = 13$). This convention adds two states the original
framework typically omits — the post-embedding state and the
post-final-norm state — both of which exhibit basis-invariant
boundary anomalies that the inner-layer ($t = 2, \ldots, 12$) fit
misses. We use the 14-state convention throughout this paper and
report the boundary-layer effect (§3.6) explicitly.

### 3.2 Training recipe

All four seeds are trained from scratch on FineWeb-Edu (sample-10BT
subset) for 24,000 steps under a single recipe specified in Table 3.2.
Each seed receives an independent random initialization (Llama default:
$\mathcal{N}(0, 0.02)$ for embedding and linear weights) and an
independent training-data shuffle order; all other hyperparameters are
identical across seeds.

**Table 3.2: Training recipe.**

| Property | Value |
|---|---|
| Corpus | FineWeb-Edu sample-10BT subset |
| Total tokens trained | 1.572B |
| Total steps | 24,000 |
| Batch size | 8 micro-batches × 8 gradient-accumulation = 64 sequences |
| Sequence length | 1024 tokens per sequence (65,536 tokens per step) |
| Optimizer | AdamW |
| AdamW $\beta_1$ | 0.9 |
| AdamW $\beta_2$ | 0.95 |
| AdamW weight decay | 0.1 |
| Peak learning rate | $3 \times 10^{-4}$ |
| LR schedule | Linear warmup 1000 steps → cosine decay to 10% of peak |
| Gradient clipping | global $L_2$ norm clip to 1.0 |
| Mixed precision | bf16 autocast forward/backward, fp32 master weights |
| Initialization | $\mathcal{N}(0, 0.02)$ for embeddings and linear weights |
| Held-out set | 500 chunks of 1024 tokens (500K tokens) |
| Held-out tokenization | identical to training (Mistral-7B-v0.1 BPE) |

The data shuffle order is deterministic given the seed. This is required
for the cross-seed activation alignment of §6.4 to be well-posed:
corresponding pilots in different seeds must be the same input tokens
at the same positions of the same chunks. We have verified the
determinism directly by comparing the pilot input-token, successor-token,
and position arrays across seeds (they are identical byte-for-byte).

Each seed completes training in approximately 12 hours on a single
RTX 5090 GPU with 24 GB of memory. The four seeds converge to
held-out eval losses in the range $[2.9062, 2.9102]$, a range of
$0.004$ across independent training runs. The eval loss is computed
on the same 500-chunk held-out set every 100 training steps and
saved to a CSV alongside the training loss; we use this CSV as
ground truth for the eval loss values reported in this paper.

The training was conducted in early-to-mid 2025. The Phase 1 SwiGLU
variant used the same compute budget; the GELU variant we report
here was run subsequently as the controlled baseline for the
multi-view extension and for the planned Phase 2 ablation study.

### 3.3 Analysis pipeline

For each seed we save 50 checkpoints at log-spaced training steps from
step 100 to step 24,000, with checkpoint spacing approximately 12% per
step. The full list of checkpoint steps is:

100, 112, 125, 140, 156, 175, 196, 219, 245, 274, 306, 342, 383, 428,
479, 535, 599, 670, 749, 837, 937, 1047, 1171, 1310, 1465, 1638, 1832,
2049, 2292, 2563, 2866, 3205, 3584, 4009, 4483, 5014, 5607, 6271, 7013,
7842, 8771, 9809, 10969, 12268, 13719, 15343, 17159, 19189, 21460, 24000.

At each of these 50 checkpoints, for each of the 4 seeds, we apply the
analyzer as follows.

**Step 1: Restore the model.** Load the saved model weights from disk
into a freshly-instantiated `LlamaStyleTransformer` matching the
architecture of Table 3.1. Run a sanity-check forward pass on a fixed
input and verify the resulting eval loss matches the logged training
CSV value to within $10^{-8}$ (we verified this match on the GELU
restoration to within $1.13 \times 10^{-8}$).

**Step 2: Collect activations.** Run forward inference on the 500
held-out chunks with `model.eval()` and `torch.no_grad()`. At each
chunk, record the residual-stream hidden state at the 19 pilot
positions $\{50, 100, 150, \ldots, 950\}$, for each of the 14 layer
states. The hidden states are collected via forward hooks attached
between blocks (for the post-block-output states), at the embedding
layer (for the post-embedding state $t = 0$), and after the final
RMSNorm (for the post-final-norm state $t = 13$).

Total per checkpoint: $500 \times 19 = 9{,}500$ pilots, each yielding
14 layer states of dimension $H = 896$, for a total tensor of shape
$(L_{\text{total}}, N, H) = (14, 9500, 896)$, or approximately 47 MB
per checkpoint in float32 (we save in float32 for precision; bf16
would halve the size but would introduce truncation errors that
contaminate the SVD).

We also record, for each pilot, the input token $c_p$, the actual
successor token $c_{p+1}$, the model's predicted successor $\hat{w} =
\text{argmax}_w P(w \mid x_L(\mathbf{c}, p))$, and the pilot position
$p$. These are the conditioning labels for the multi-view decomposition
of §5–§6.

**Step 3: Center per layer.** Subtract the per-layer mean
$\bar{x}_t = \frac{1}{N} \sum_i x_t^{(i)}$ to get the centered
activation matrix $\tilde{X}_t$.

**Step 4: Compute the centered SVD per layer.** Compute the SVD of
$\tilde{X}_t / \sqrt{N}$ at each layer state. The right singular
vectors form $R(t)$; the squared singular values give the
per-coordinate residual variance $\sigma_{t,k}^2$. We use double-precision
SVD throughout (`numpy.linalg.svd` with `full_matrices=False`).

**Step 5: Compute pairwise residual variances.** For each ordered pair
$(t, t + \tau)$ with $\tau \geq 1$, predict $x(t + \tau)$ from $x(t)$
via ordinary least squares in the per-layer basis:

$$\tilde{x}(t + \tau) = R(t + \tau) S_{t + \tau} V_t^\top X_t / \lVert X_t \rVert_F$$

(equivalently, the OLS prediction whose residual variance is what the
framework's $\sigma^2(t, \tau)$ refers to). Compute the per-coordinate
residual variance $\sigma_{t, \tau, k}^2$ for $k = 1, \ldots, H$.

**Step 6: Fit the variance-scaling law.** Fit $\log \sigma^2 = \log
\alpha + \lambda \log \tau$ to the endpoint variances — i.e., the
$\sigma^2(0, \tau)$ for $\tau = 1, \ldots, L + 1$ — using the
sub-step procedure described in Sarfati et al. Both conventions
(mean of log per-coordinate variance; log of mean per-coordinate
variance) are computed.

**Step 7: Save the flow.** Save $R(t)$, $\Sigma_t$, $(\log\alpha,
\lambda)$ in both conventions, the per-layer effective rank, the
per-layer kurtosis, the per-layer isotropy, the successive-layer
angles, and the pairwise residual variance matrix to disk as
`flow_step_NNNNNNNN.npz`. Total per seed per checkpoint: approximately
65 MB; total across all 4 seeds × 50 checkpoints = approximately 13 GB.

**Multi-view extension.** Step 2 of the multi-view pipeline differs
slightly: in addition to the standard activation collection, we also
save the input token, successor token, predicted-successor token, and
position arrays for each pilot. The conditional-view analyses of §5–§6
take these augmented activation files as input and perform the
per-condition statistics described in §2.6.

**Determinism.** The full analyzer pipeline is deterministic given fixed
pilot positions and a fixed forward pass. Running the analyzer twice
on the same checkpoint produces byte-identical output. The full 4-seed
× 50-checkpoint sweep takes approximately 6 hours of GPU time across
the entire pilot (1.5 hours per seed). The multi-view extension adds
approximately 14 seconds per checkpoint for activation augmentation
plus 200-250 seconds per checkpoint for the per-view decomposition,
totaling approximately 4 hours per seed.

### 3.4 H1: Convergence of the linear flow

The first replication question is whether the recovered linear flow
$R(t)$ converges to a stable form during training, as the framework
implicitly assumes when it characterizes a single converged checkpoint
by its basis-invariant statistics. The original framework does not
explicitly test convergence; the convergence test was a contribution of
our prior Phase 1 study, which we replicate here.

**Convergence criterion.** Let $D_k = \lVert R^{(k)} -
R^{(\text{final})} \rVert_F$ be the summed-over-layers Frobenius
distance from checkpoint $k$ to the final checkpoint, where
$R^{(k)} \in \mathbb{R}^{14 \times 896 \times 896}$ is the stack of
recovered $R$ matrices at all 14 layer states. Convergence requires
that the residual jitter in the last quarter of training is small
relative to the total movement during training:

$$\frac{\text{std}_{\text{last } 25\%}(D_k)}{D_1 - D_K} \leq 0.10.$$

That is, after the linear flow has finished moving, the residual
jitter in its position should be no larger than 10% of the total
distance traveled during training. This is a strong criterion: it
requires not just that $D_k$ is small at the end of training, but
that it is *stably* small relative to the dynamics seen during
training.

**Result.** The H1 criterion passes on all four seeds with margin to
spare (Table 3.3).

**Table 3.3: H1 convergence test results.**

| Seed | Last-quarter std of $D_k$ | Total reduction $D_1 - D_K$ | Ratio | Verdict |
|---|---:|---:|---:|:---:|
| seed 0 | 40.05 | 876.30 | 0.046 | PASS |
| seed 1 | 38.02 | 909.57 | 0.042 | PASS |
| seed 2 | 39.24 | 896.77 | 0.044 | PASS |
| seed 3 | 38.97 | 903.56 | 0.043 | PASS |
| **mean** | **39.07** | **896.55** | **0.0436** | — |
| **std** | 0.85 | 14.04 | 0.0016 | — |
| **range** | 2.03 | 33.27 | 0.004 | — |

The cross-seed range on the H1 ratio is 0.004 on a threshold of 0.10
— the verdict is robust to the choice of any reasonable threshold
above about 0.06. The total reduction $D_1 - D_K \approx 897$
corresponds to the integrated distance the linear flow travels
through hidden space during training; the residual last-quarter
standard deviation of $\approx 40$ means the flow has nearly stopped
moving by the time training completes.

**Loss vs flow convergence timing.** We track both the held-out eval
loss $\mathcal{L}_k$ and the normalized flow distance $D_k / D_1$ as
functions of training step. Both are monotonically decreasing, both
flatten in the last quarter of training, but they do not decline on
the same schedule. The eval loss flattens gradually throughout
training, with the largest derivative in the first few thousand
steps and a smooth decay thereafter. The flow distance has a more
pronounced kink, decreasing rapidly until step approximately 5,000
and then much more slowly.

This is consistent with the hypothesis that the linear-flow geometry
locks in earlier than the model's full loss performance — the
residual stream's *coordinate structure* converges before its
*prediction accuracy* does. The flow at step 10,000 is essentially
the same as the flow at step 24,000 (normalized distance
approximately 0.10 of the way back to step 1's flow), while the eval
loss at step 10,000 is still measurably higher than at step 24,000.
We return to this timing relationship in §4 as a defining feature
of the training-dynamic landscape.

### 3.5 Basis-invariant statistics at convergence

We report the framework's headline basis-invariant statistics at the
final checkpoint, with cross-seed dispersion measured as the standard
deviation across the four seeds. Each statistic is reported in full
per-seed detail; the consolidated dispersion table is Table 3.5.

**Variance-scaling exponent $\lambda$.** At the final checkpoint:

**Table 3.4a: $\lambda$ at convergence.**

| Seed | $\lambda$ (paper convention) | $\lambda$ (our convention) |
|---|---:|---:|
| seed 0 | 0.4256 | 0.4418 |
| seed 1 | 0.4223 | 0.4383 |
| seed 2 | 0.4235 | 0.4395 |
| seed 3 | 0.4330 | 0.4490 |
| **mean** | **0.4261** | **0.4422** |
| **std** | **0.0048** | **0.0048** |
| **range** | 0.0107 | 0.0107 |
| **$1.5 \times$ std** | 0.0072 | 0.0072 |
| **relative spread** | **1.1%** | **1.1%** |

The two conventions differ by a constant offset (about $+0.016$ for
our convention vs the paper convention) at convergence, but have
identical std and range across seeds. **$\lambda$ is reproducible to
1.1% relative across seeds.** This is the tightest dispersion we
measure for any statistic in the marginal framework.

The seed 3 value ($\lambda_{\text{paper}} = 0.4330$) is the high-side
outlier; the other three seeds cluster in $[0.4223, 0.4256]$. This is
consistent with a single sample slightly outside the main cluster
(seed 3 sits about $+0.007$ above the cluster mean of $[0.4223,
0.4256]$ in paper convention), not with a bimodal within-variant
distribution.

**Variance prefactor $\log\alpha$.** At the final checkpoint:

**Table 3.4b: $\log\alpha$ at convergence.**

| Seed | $\log\alpha$ (paper convention) | $\log\alpha$ (our convention) |
|---|---:|---:|
| seed 0 | $-3.298$ | $-3.266$ |
| seed 1 | $-3.193$ | $-3.152$ |
| seed 2 | $-3.259$ | $-3.224$ |
| seed 3 | $-3.360$ | $-3.330$ |
| **mean** | **$-3.277$** | **$-3.243$** |
| **std** | **0.070** | **0.075** |
| **range** | 0.168 | 0.178 |
| **$1.5 \times$ std** | 0.105 | 0.112 |
| **relative spread** | **2.1%** | **2.3%** |

The dispersion on $\log\alpha$ is substantially larger than on
$\lambda$: about 0.07 std vs 0.005 for $\lambda$. This is expected —
$\log\alpha$ is more sensitive to small differences in the
residual-variance spectrum at the high-$\tau$ end of the fit, where
one or two boundary layer states dominate (§3.6).

Seed 3 is again the most extreme (most negative $\log\alpha$) and seed
1 is the least extreme (least negative). The 4-seed standard deviation
is 0.070 (paper convention); without seed 3 the 3-seed std would be
0.054 (also paper convention). Seed 3's contribution to the dispersion
is significant but not dominating.

**Effective rank.** The effective rank profile is heavily layer-dependent;
we report the per-layer mean across seeds at convergence:

**Table 3.4c: Effective rank profile (cross-seed mean and std at convergence).**

| Layer | Mean across seeds | Std across seeds | Range across seeds | $1.5 \times$ std |
|---|---:|---:|---:|---:|
| $t = 0$ (post-embed) | 175.5 | 4.8 | 11.2 | 7.2 |
| $t = 1$ | 244.3 | 6.8 | 14.9 | 10.2 |
| $t = 2$ | 327.0 | 9.5 | 21.7 | 14.3 |
| $t = 3$ | 416.4 | 12.4 | 27.2 | 18.6 |
| $t = 4$ | 453.6 | 13.7 | 30.6 | 20.6 |
| $t = 5$ | 469.4 | 14.6 | 33.5 | 21.9 |
| $t = 6$ | 457.5 | 14.2 | 32.4 | 21.3 |
| $t = 7$ | 502.2 | 15.4 | 32.9 | 23.1 |
| $t = 8$ | 511.4 | 15.5 | 33.3 | 23.2 |
| $t = 9$ | 495.6 | 14.9 | 32.5 | 22.3 |
| $t = 10$ | 452.7 | 13.6 | 30.3 | 20.4 |
| $t = 11$ | 411.2 | 12.4 | 27.8 | 18.6 |
| $t = 12$ | 244.3 | 6.8 | 14.9 | 10.2 |
| $t = 13$ (post-norm) | 267.7 | 7.7 | 16.8 | 11.6 |
| **mean across layers** | **376.9** | **7.8** | **16.6** | **11.7** |

The effective rank profile is heavily layer-dependent and bow-shaped: the
boundary layers (post-embedding and post-final-norm) have effective
ranks near 175-270 — about $H/3$ to $H/5$ — while the middle layers
reach effective ranks near 510, well over half of $H = 896$. The
residual stream becomes more isotropic as it passes through transformer
blocks, then collapses back to a lower effective rank at the
post-final-norm state.

This profile shape is highly reproducible across seeds. The
range/mean ratio at the middle-layer maximum is 7%, comparable to the
$\lambda$ dispersion. At the boundary layers it is even tighter (5%).

**Per-coordinate excess kurtosis.** At the final checkpoint:

**Table 3.4d: Per-coordinate excess kurtosis at convergence.**

| Seed | Signed mean $\langle\kappa\rangle$ | Absolute mean $\langle\|\kappa\|\rangle$ |
|---|---:|---:|
| seed 0 | 0.871 | 0.871 |
| seed 1 | 1.334 | 1.334 |
| seed 2 | 1.045 | 1.045 |
| seed 3 | 0.932 | 0.932 |
| **mean** | **1.046** | **1.046** |
| **std** | **0.205** | **0.205** |
| **range** | 0.463 | 0.463 |
| **$1.5 \times$ std** | 0.308 | 0.308 |
| **relative spread** | **19.6%** | **19.6%** |

The signed mean kurtosis and the absolute-mean kurtosis agree to four
decimal places across all seeds and all checkpoints. This indicates
that per-coordinate kurtosis is uniformly one-sided positive at every
layer in every seed — there are no negative-kurtosis coordinates.
This rules out a hypothesis we considered earlier (and confirmed
fully in our Phase 1 study), that the paper's mean-of-log kurtosis
convention vs our naive mean-kurtosis convention would diverge because
of negative-kurtosis cancellation. They don't diverge, because there
are no negative kurtosis coordinates to cancel.

Seed 1 is the kurtosis outlier ($\langle|\kappa|\rangle = 1.334$ vs
$0.871$–$1.045$ for seeds 0, 2, 3). The 4-seed standard deviation of
0.205 is dominated by this seed 1 contribution; without seed 1, the
3-seed standard deviation would be 0.088. We have no mechanistic
explanation for the seed 1 anomaly. Notably, it doesn't appear in
$\lambda$, $\log\alpha$, effective rank, eval loss, or H1 ratio — only
in kurtosis (and secondarily in isotropy). Seed 1's training landed at
a slightly more heavy-tailed residual distribution without affecting
other downstream behavior.

Importantly, the seed 1 kurtosis anomaly is a late-training phenomenon.
At step 5,000, all four seeds have near-identical kurtosis values
(around 0.5 in absolute terms). The kurtosis at intermediate steps
(5,000-13,000) is similar across all four seeds; the divergence
happens entirely in the last 11,000 training steps. Seeds 0, 2, 3
show a smoothly monotonic kurtosis rise from $\approx 0.35$ at step
2,000 to $\approx 1.0$ at step 24,000; seed 1 shows a sharper
acceleration past step 13,000, ending at $1.334$. We discuss this
training-dynamic behavior in §4.

**Isotropy.** We measure isotropy as the standard deviation of
$\log \sigma_{t,k}^2$ across coordinates $k$ within a layer, averaged
across layers. At the final checkpoint:

**Table 3.4e: Mean isotropy at convergence.**

| Seed | Mean isotropy (across 14 layers) |
|---|---:|
| seed 0 | 0.154 |
| seed 1 | 0.165 |
| seed 2 | 0.153 |
| seed 3 | 0.154 |
| **mean** | **0.157** |
| **std** | **0.006** |
| **range** | 0.012 |
| **relative spread** | **3.8%** |

Seed 1 is again the most extreme (most anisotropic), consistent with
its high kurtosis. The dispersion is small enough that for any
cross-architecture comparison work, isotropy is a useful comparator:
$1.5 \times \text{std} \approx 0.009$, so cross-variant differences
above about 0.01 would be informative.

**Held-out eval loss.** At the final checkpoint:

**Table 3.4f: Eval loss at convergence.**

| Seed | Eval loss (cross-entropy, nats) |
|---|---:|
| seed 0 | 2.9080 |
| seed 1 | 2.9062 |
| seed 2 | 2.9102 |
| seed 3 | 2.9078 |
| **mean** | **2.9080** |
| **std** | **0.0018** |
| **range** | 0.0040 |
| **relative spread** | **0.06%** |

The eval loss is the tightest-converging statistic, with a relative
spread of 0.06%. All four seeds reached the same prediction accuracy
to within 0.06% — this is the cleanest evidence that the four seeds
are training to functionally equivalent models, regardless of how
different their internal $R$ matrices may be.

**Consolidated dispersion table.** Table 3.5 collects all of the above.

**Table 3.5: Within-variant dispersion summary.**

| Statistic | Mean | Std | $1.5 \times$ std | Relative spread |
|---|---:|---:|---:|---:|
| $\lambda$ (paper, all-layer) | 0.4261 | 0.0048 | 0.0072 | 1.1% |
| $\log\alpha$ (paper, all-layer) | $-3.277$ | 0.070 | 0.105 | 2.1% |
| $\lambda$ (paper, boundary-excl.) | 0.5107 | 0.0048 | 0.0072 | 0.9% |
| $\log\alpha$ (paper, boundary-excl.) | $-3.756$ | 0.073 | 0.110 | 1.9% |
| Effective rank (mean across layers) | 376.9 | 7.8 | 11.7 | 2.1% |
| Effective rank (middle-layer max) | 511.4 | 15.5 | 23.2 | 3.0% |
| Per-coord excess kurtosis | 1.046 | 0.205 | 0.308 | 19.6% |
| Mean isotropy | 0.157 | 0.006 | 0.009 | 3.8% |
| Eval loss | 2.908 | 0.0018 | 0.0028 | 0.06% |
| H1 ratio | 0.0436 | 0.0016 | 0.0025 | 3.6% |

Two patterns stand out. The first is that $\lambda$, isotropy, the
effective rank profile, and eval loss all have within-variant relative
spreads of about 0.06% to 4%; these are the best-reproducible quantities
at our scale. The second is that **kurtosis is anomalously disperse**
(19.6% relative spread), dominated by the seed 1 outlier. Any
cross-architecture comparison work that uses kurtosis as a comparison
statistic should treat it as a soft signal — substantial differences in
kurtosis above approximately $\pm 0.3$ are informative, but smaller
differences may be within-variant noise.

The dispersion bounds in Table 3.5 are the within-variant noise floor
that the conditional-view dispersion in §5–§6 will be calibrated
against. A conditional-view statistic with relative spread of about
1-4% across seeds matches the marginal-view dispersion and is
considered well-reproduced; substantially larger dispersion would
indicate that the conditioning introduces noise that swamps the
signal.

### 3.6 Boundary-layer effect

The basis-invariant variance-scaling law $\sigma^2 \sim \alpha \tau^\lambda$
is approximately log-linear in $\tau$ across the inner layer states,
but two states deviate from the fit. We characterize this *boundary-layer
effect* here because it affects the at-convergence $\log\alpha$ and
$\lambda$ values, and because it is itself a basis-invariant signature
that is highly reproducible across seeds.

**Per-layer residual variance scatter at convergence.** Looking at the
per-layer endpoint variance scatter (representative for seed 0, with
qualitatively identical pattern in all four seeds):

- $t = 0$ (post-embedding, lowest-$\tau$ point at $\tau = 1$): log-variance
  sits roughly 0.7 units below the inner-layer fit. The bundle has
  just been embedded and has not yet accumulated the variance the
  inner-layer linear flow would predict.
- $t = 13$ (post-final-norm, highest-$\tau$ point at $\tau = L + 1 = 13$):
  log-variance sits roughly 1.8 units below the inner-layer fit. The
  RMSNorm operation has compressed the residual stream's variance
  substantially relative to the inner-layer extrapolation.
- The 11 inner layer states ($t = 2, \ldots, 12$) align tightly along
  a single log-linear line that gives the inner-layer fit.

Both boundary layers are outliers, but the post-final-norm anomaly is
larger (1.8 log units vs 0.7) and acts on a larger-$\tau$ point. The
fit is least-squares in log-log, so the post-final-norm anomaly has
more leverage on the slope.

**Refitting with boundary states excluded.** Excluding both boundary
states from the variance-scaling fit gives a steeper $\lambda$ and a
more-negative $\log\alpha$ (Table 3.6):

**Table 3.6: Boundary-layer effect at convergence (paper convention).**

| Seed | $\log\alpha$ all-layer | $\log\alpha$ boundary-excluded | $\Delta\log\alpha$ | $\Delta\lambda$ |
|---|---:|---:|---:|---:|
| seed 0 | $-3.298$ | $-3.786$ | $-0.488$ | $+0.085$ |
| seed 1 | $-3.193$ | $-3.659$ | $-0.467$ | $+0.084$ |
| seed 2 | $-3.259$ | $-3.743$ | $-0.483$ | $+0.085$ |
| seed 3 | $-3.360$ | $-3.835$ | $-0.474$ | $+0.084$ |
| **mean** | **$-3.277$** | **$-3.756$** | **$-0.478$** | **$+0.085$** |
| **std** | 0.070 | 0.073 | 0.009 | 0.0005 |
| **range** | 0.168 | 0.176 | 0.021 | 0.001 |

The boundary effect is extraordinarily reproducible: $\Delta\log\alpha$
has a range of 0.021 across four independent seeds — about 4.4% of its
absolute value — and the corresponding $\Delta\lambda$ has a range of
just 0.001, two orders of magnitude tighter than the $\Delta\lambda$
value itself.

**Boundary effect is learned, not architectural.** The boundary effect is
*not* present at initialization. At step 100, both boundary layers fall
close to the inner-layer line; $\Delta\log\alpha$ is near zero. The
anomaly emerges during training: it reaches half its final magnitude
by step approximately 2,000 and plateaus to its final value by step
approximately 5,000. The boundary effect is a *learned phenomenon*,
not a fixed structural property of the architecture. The training
dynamics of its emergence are reported in §4.

**Mechanical interpretation.** The post-final-norm state's variance sits
substantially below the inner-layer fit because the RMSNorm operation
rescales the residual stream toward a fixed-norm shell, reducing the
per-coordinate variance at the output relative to the inner layers.
The post-embedding state's offset is smaller and opposite in nature,
reflecting that the embedded representation has not yet undergone the
variance expansion that early layers perform.

The mechanical reading of the post-final-norm gap as "RMSNorm
rescaling" is consistent with the observed successive-layer angle
profile, which shows that the $R(t)$ rotation from $t = 12$ (last
block output) to $t = 13$ (post-final-norm) is only $\approx 8°$ — far
smaller than the $\approx 30°$ inner-layer rotations. The RMSNorm
operation primarily rescales the residual stream's principal
directions rather than rotating them.

Both boundary offsets reflect the basis-invariant geometry actually
being slightly more complicated than the strict single-power-law fit
would suggest; the framework's statistics are best understood with the
boundary contributions tracked explicitly rather than absorbed into a
global fit. We report both all-layer and boundary-excluded $\lambda$
and $\log\alpha$ throughout the paper because they answer different
questions: the all-layer values are what the original framework
reports, the boundary-excluded values are what the inner-layer
variance scaling actually is, and the difference between them is the
boundary-layer effect.

**$\lambda \times L$ scaling.** The original framework observes that
$\lambda \times L \approx 5.5$ across the four architectures it tests.
Our all-layer $\lambda \approx 0.4261 \times L = 12$ gives $\lambda L
\approx 5.11$ — within the paper's observed range, by coincidence (the
paper's value reflects the all-layer fit at substantially different
scales). But under boundary exclusion, $\lambda$ rises to $\approx
0.5107$, giving $\lambda L \approx 6.13$. If $\lambda L \approx 5.5$ is
a real universality, it is a universality of the all-layer fit, not
the inner-layer fit. We cannot tell from a single architectural variant
whether the paper's observed $\lambda L$ relationship would also hold
under boundary exclusion. Establishing whether $\lambda L$ universality
holds in either sense across architectures is a question for a
cross-architecture study.

### 3.7 Cross-seed $R$-matrix non-alignment

The framework's basis-invariant statistics are reproducible across
seeds (Table 3.5); the framework's basis-dependent quantities (the
$R(t)$ matrices themselves) are not. We summarize this finding here
because it motivates the cross-seed alignment analysis of §6.4 in the
multi-view setting. The full detail of the Phase 1 alignment-failure
analysis is in our prior work.

**Embedding-space Procrustes fails on the full vocabulary.** For each
ordered seed pair $(A, B)$ at the final checkpoint we computed the
orthogonal Procrustes alignment of the embedding matrices and used the
recovered rotation $Q$ to transport $R_A$ into seed $B$'s coordinate
frame. The mean result across the 12 ordered pairs:

| Metric | Value | Interpretation |
|---|---:|---|
| $\rho_E$ (embedding residual) | 0.605 | terrible: 60% residual after best rotation |
| Aligned R-distance | 592.7 | equal to random orthogonal baseline |
| Aligned/identity R-distance | 1.000 | alignment doesn't help over no alignment |
| Aligned mean angle (top-10) | 85.0° | equal to random orthogonal baseline |

All four metrics indicate alignment failure. The embedding-space
alignment is finding *some* rotation $Q$, but that $Q$ is statistically
indistinguishable from a random orthogonal matrix in terms of how it
transports the $R$ matrices.

**The cause: undertrained rare-token contamination.** The Mistral
tokenizer has 32,768 BPE tokens, optimized for a 7B+ model trained on
a much larger corpus. At our 150M scale and 1.57B-token training
duration, only a fraction of these tokens receive enough gradient
signal to learn meaningful embeddings. The remaining rare tokens stay
near their random initialization throughout training and contribute
pure noise to the Procrustes residual.

Filtering to the top-$K$ tokens by per-row $L_2$ norm gives the
following sweep:

| $K$ | $\rho_E$ | Verdict |
|---|---:|---|
| 100 | 0.078 | excellent |
| 1000 | 0.095 | excellent |
| 5000 | 0.132 | good |
| 32768 (full vocab) | 0.605 | terrible |

With $K = 1000$, embedding-space Procrustes recovers a clean
alignment. The full-vocabulary failure was not a failure of the
alignment procedure; it was a failure of the anchor-set selection.

**But fixing the embedding alignment does not fix the $R$-matrix
transport.** With the $K = 1000$ embedding alignment (mean $\rho_E =
0.10$, alignment quality good), the recovered $Q$ still fails to
transport $R$. All three conditions (Procrustes-aligned, identity-aligned,
random-aligned) give the same $R$-matrix Frobenius distance to four
significant figures. The embedding alignment recovers a clean $Q$ that
successfully aligns embeddings, but that $Q$ has no effect on the $R$
matrices at deeper layers.

**Activation-space alignment also fails.** For each ordered seed pair
we collected per-layer pilot activations from both seeds (using
identical held-out chunks in identical order) and ran the per-layer
Procrustes alignment described in §2.8. The per-layer alignment is
strictly more powerful than the embedding-based transport because it
finds a separate rotation at each layer. The result is still failure:
the mean per-layer residual ratio $\rho_t$ across the 12 ordered pairs
is 0.620, statistically indistinguishable from the random-orthogonal
baseline.

**Top-$K$ subspace diagnostic resolves the failure.** The per-layer
activation-Procrustes alignment is finding $\rho_t = 0.620$ at every
layer; the question is whether the failure is "all directions are
random" or "the top few directions are shared and the trailing
directions (which dominate the Frobenius norm by count) are
seed-specific noise." To distinguish these, we computed the principal
angles between the top-$K$ rows of (activation-aligned) $R_A$ and the
top-$K$ rows of $R_B$ for $K$ sweeping from 1 to $H = 896$, comparing
to the within-seed split-half noise floor:

| $K$ | Cross-seed | Within-seed (split-half) | Gap |
|---|---:|---:|---:|
| 1 | 88.9° | 7.0° | $+81.9°$ |
| 2 | 87.8° | 9.5° | $+78.4°$ |
| 5 | 86.6° | 13.2° | $+73.4°$ |
| 10 | 85.0° | 17.0° | $+68.0°$ |
| 50 | 78.4° | 24.1° | $+54.3°$ |
| 100 | 73.1° | 26.6° | $+46.5°$ |
| 500 | 39.4° | 20.6° | $+18.8°$ |
| 896 (full) | 0.02° | 0.02° | $0.00°$ |

Two patterns are clear. Within-seed angles are small (5-25°) across
the $K$ range — the analyzer is sample-stable. Cross-seed angles are
at the random-orthogonal baseline at every $K$ — even at $K = 1$, the
top-1 principal direction of $R$, cross-seed angles are 89° while
within-seed split-half angles are 7°.

**Conclusion.** Cross-seed $R$ matrices share no recoverable basis
structure even at the top-1 direction. Different seeds learn $R$
matrices along seed-specific bases unrelated by any orthogonal map.

This finding underwrites the framework's reliance on basis-invariant
statistics for cross-model comparison. The framework's preferred
quantities — $\lambda$, $\log\alpha$, effective rank, isotropy — are
the ones that factor out the absolute-pose freedom that the training
process exploits, leaving the intrinsic functional content that any
equivalent model should agree on.

We note that the Phase 1 alignment-failure result is a *binary*
finding: either alignment works or it doesn't, and in our case it
doesn't. §6.4 reports a more nuanced multi-view alignment analysis
that recovers a partial alignment at the I/O boundary layers, with the
"alignment fails entirely" framing of Phase 1 being correct for the
network interior but too strong for the boundaries. The multi-view
framing of cross-seed alignment is one of the paper's substantive
contributions and is set up here for the §6.4 discussion.

### 3.8 Summary of replication

At convergence, the framework's predictions hold at our 150M scale:

1. The linear flow $R(t)$ converges (H1 PASS on all 4 seeds with ratio
   $0.0436 \pm 0.0016$, threshold 0.10).
2. The variance-scaling law $\sigma^2 \sim \alpha \tau^\lambda$ fits
   well in the inner layers, with $\lambda \approx 0.426$ (paper
   convention, all-layer fit) reproducible across seeds at 1.1%
   relative spread.
3. The basis-invariant statistics ($\lambda$, $\log\alpha$, effective
   rank, isotropy) are reproducible across seeds at 1-4% relative
   spread.
4. Per-coordinate kurtosis is reproducible at substantially higher
   dispersion (19.6%), dominated by one seed outlier whose anomaly is
   confined to the kurtosis statistic and emerges in late training.
5. The boundary-layer effect is real, learned during training,
   reproducible across seeds to within 0.021 log units (about 4.4% of
   its absolute value), and produces a $\Delta\log\alpha = -0.478$
   shift between all-layer and boundary-excluded fits.
6. The basis-dependent $R(t)$ matrices are not reproducible across
   seeds — confirming the framework's appropriate level of abstraction
   is the basis-invariant statistics.

These at-convergence findings establish the baseline against which the
multi-view extension's findings are to be compared. §4 turns to the
training dynamics of the same marginal statistics, where the
apparently-converged surface obscures a richer set of features that
will become the landmark events of §6's co-location story.

---
