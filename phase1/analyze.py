"""
Per-checkpoint linear-flow recovery.

For a given checkpoint, this module:

  1. Loads the model and puts it on the GPU.
  2. Runs inference on the held-out evaluation corpus, collecting per-layer
     hidden states at pre-specified "pilot" token positions.
  3. Computes the lines-of-thought linear flow:
       - Per-layer SVD: R(t), Σ(t)
       - Pairwise stretches: Λ(t, τ) = diag(Σ(t+τ) / Σ(t))
       - Residuals: w(t, τ) = x(t+τ) − R(t+τ) Λ(t,τ) R(t)ᵀ x(t)
       - Variance scaling fit: log E‖w‖²/H = log α + λ (t+τ)
       - Gaussianity diagnostics: excess kurtosis, skewness, isotropy
       - Manifold dimensionality (effective rank of activation covariance)
  4. Saves the recovered flow object L^(K) to disk as a .npz file
     for downstream alignment and convergence analysis.

Why pilot-token sampling instead of all positions: per Sarfati et al.,
the linear flow is fit on individual token trajectories sampled from
within-sequence positions, not on all positions. Different positions
have different statistics (very-early positions especially), so we
pick pilot positions at well-separated, late-enough indices to get
data points that are in the "settled flow" regime.

Public entry points:
  - analyze_checkpoint(checkpoint_path, eval_loader, ...) -> dict
        Recover the linear flow from one checkpoint.
  - save_flow(flow_dict, output_path)
        Serialize a flow result to .npz.
  - load_flow(path) -> dict
        Deserialize.
  - analyze_run(run_dir, ...)
        Analyze every checkpoint in a completed training run.
"""

import os
import time
import glob
import json
from typing import List, Dict, Optional

import numpy as np
import torch

from config import ModelConfig, TrainingConfig, load_config_pair
from models import LlamaStyleTransformer, build_model


# ----------------------------------------------------------------------
# Pilot-token sampling.
# ----------------------------------------------------------------------
def default_pilot_positions(seq_len: int, stride: int = 50, start: int = 50) -> List[int]:
    """
    Pick pilot positions within a single sequence.

    Returns positions [start, start+stride, start+2*stride, ...] up to seq_len-1.
    With start=50, stride=50, seq_len=1024 this gives 19 positions:
        [50, 100, 150, ..., 950]

    Rationale:
      - start=50 skips the first 50 positions where the activation
        statistics haven't settled (per Sarfati et al.).
      - stride=50 spaces pilots far enough apart that they're roughly
        independent samples of the trajectory ensemble.
      - We deliberately don't pick the last few positions because RoPE
        + causal masking gives end-of-sequence positions a wider
        attention window over the whole sequence, which can produce
        unusual activations relative to the bulk.
    """
    positions = list(range(start, seq_len, stride))
    # Trim the final few positions if they're too close to the end.
    while positions and positions[-1] > seq_len - 50:
        positions.pop()
    return positions


# ----------------------------------------------------------------------
# Activation accumulation.
# ----------------------------------------------------------------------
@torch.no_grad()
def collect_activations(
    model,  # any *StyleTransformer with the standard forward signature
    eval_loader,
    pilot_positions: List[int],
    device: str,
    autocast_dtype,
    max_pilots: int = 100_000,
) -> np.ndarray:
    """
    Run inference on eval_loader and collect hidden states at the
    specified pilot positions.

    Returns:
        activations: np.ndarray of shape (num_layers + 2, num_pilots, H)
            in fp32. Layer 0 is post-embedding, layers 1..L are per-block
            outputs, layer L+1 is post-final-norm.

    Stops once max_pilots have been collected. With max_pilots=100k and
    19 pilots/sequence × batch 16 = 304 pilots/batch, we need ~330
    batches = ~5000 sequences. At seq_len=1024 that's ~5M evaluation
    tokens (well under our 500-chunk held-out set of 500 * 1024 ≈ 500k
    tokens).

    Wait — we only have 500 held-out chunks. With 19 pilots/chunk we
    can collect at most 500 * 19 = 9500 pilots from the held-out set.
    That's a much smaller sample than 100k. To get to 100k pilots we'd
    need to either (a) accept fewer samples, or (b) sample from the
    training set as well.

    For Phase 1 we accept the smaller sample. 9500 pilots × 896-dim
    activations is still 9500 × 896 ≈ 8.5M features per layer, more
    than enough for a stable SVD fit at H=896.
    """
    model.eval()
    num_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    n_layer_outputs = num_layers + 2  # input + L outputs + final-norm

    # Accumulator: list of lists, one per layer.
    per_layer_buffers = [[] for _ in range(n_layer_outputs)]
    collected = 0

    for batch in eval_loader:
        if collected >= max_pilots:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=autocast_dtype,
                                 enabled=(device == "cuda")):
            _, _, hidden = model(input_ids, return_hidden_states=True)

        # Each hidden[i] has shape (B, T, H). Index at pilot positions.
        # We promote to fp32 before storing so the SVD runs at full precision.
        B = input_ids.size(0)
        T = input_ids.size(1)
        # Validate pilot positions for the seq_len.
        valid_pos = [p for p in pilot_positions if p < T]
        for layer_idx, h in enumerate(hidden):
            # h shape: (B, T, H). Pick pilot rows -> (B, n_pilots, H) -> flatten.
            picked = h[:, valid_pos, :].float().cpu().numpy()
            per_layer_buffers[layer_idx].append(picked.reshape(-1, H))
        collected += B * len(valid_pos)

    # Concatenate.
    activations = np.zeros((n_layer_outputs, collected, H), dtype=np.float32)
    for layer_idx in range(n_layer_outputs):
        cat = np.concatenate(per_layer_buffers[layer_idx], axis=0)
        # Trim to exactly `collected` rows in case the last batch overshot.
        activations[layer_idx] = cat[:collected]

    return activations


# ----------------------------------------------------------------------
# Linear-flow recovery.
# ----------------------------------------------------------------------
def recover_linear_flow(
    activations: np.ndarray,
    center: bool = True,
) -> Dict:
    """
    Given (num_layers, N, H) activations, recover the lines-of-thought
    linear flow.

    Returns a dict with:
      - 'R': (num_layers, H, H) per-layer SVD basis (rows are principal directions)
      - 'singular_values': (num_layers, H) sorted descending per layer
      - 'means': (num_layers, H) per-layer activation mean (used for centering)
      - 'effective_rank': (num_layers,) entropy-based dim estimate (= exp of
                         the entropy of the normalized squared singular values)
      - 'pairwise_residual_variance': (num_layers, num_layers) where entry [t, t+τ]
            is E‖w(t,τ)‖² / H for the linear-flow extrapolation residual.
            Equal to mean_d (var_d) over coordinates. ("ours" convention.)
            Entries with t >= t+τ are NaN.
      - 'pairwise_mean_log_var': (num_layers, num_layers) where entry [t, t+τ]
            is mean_d (log var_d) over coordinates. ("paper" convention.)
            Differs from log(pairwise_residual_variance) by Jensen's inequality:
            mean_d log(var_d) ≤ log(mean_d var_d). Entries with t >= t+τ are NaN.
      - Variance scaling fit, both conventions:
          'log_alpha', 'lambda' — fit of log(mean_d var_d) vs (t+τ) (ours)
          'log_alpha_paper', 'lambda_paper' — fit of mean_d(log var_d) vs (t+τ)
        Both use the (t+τ) target layer index, marginalized over starting t.
        The paper-convention values are directly comparable to Sarfati et al.'s
        published log α and λ; the "ours" values are not (Jensen-style offset).
      - 'kurtosis_per_layer': (num_layers,) signed mean excess kurtosis of
            residual distribution at each endpoint layer, averaged over H
            dimensions and over source layers. ("ours" convention.)
      - 'kurtosis_abs_per_layer': (num_layers,) mean *absolute* excess kurtosis
            mean_d |κᵢ|, averaged over source layers. ("paper" convention —
            comparable to Sarfati et al.'s <|κ|> figures.)
      - 'isotropy_per_layer': (num_layers,) standard deviation of
            log(diagonal-of-residual-covariance) across H dimensions — 0
            means perfectly isotropic. (Convention-independent.)

    Args:
        activations: (num_layers, N, H) array. Each row of activations[t]
            is one (input, position) pair's hidden state at layer t.
        center: If True, subtract per-layer means before SVD. The paper
            does NOT explicitly center, but uncentered SVD lumps the
            mean into the first singular vector which complicates
            interpretation. We center by default and store the means
            for downstream use (alignment can apply them or not).

    Implementation note on the linear-flow prediction:
        The paper's formula $\\tilde{x}(t+\\tau) = R(t+\\tau)\\Lambda(t,\\tau)R(t)^\\top x(t)$
        with $\\Lambda = \\text{diag}(\\sigma_i(t+\\tau)/\\sigma_i(t))$ describes a
        specific linear map, but its element-wise scaling assumes the SVD
        axes at successive layers are sign-consistent and ordered the same
        way — which numpy's SVD does NOT guarantee (signs are arbitrary per
        singular vector). Naively applying the formula gives residuals
        contaminated by SVD sign-flip artifacts that have nothing to do
        with the model's dynamics.

        Two equivalent ways to compute the prediction sign-robustly:
        (1) Solve for the best linear map A_{t→target} via least squares,
            i.e., A = (X_t^T X_t)^{-1} X_t^T X_target. Always gives the
            same prediction regardless of SVD sign convention because it
            never decomposes A into rotations and scales.
        (2) Apply a canonical sign convention to R (e.g., force the largest
            magnitude entry of each row to be positive).

        We use approach (1) — least-squares regression — because it's
        provably the optimal linear predictor in MSE sense, which is what
        the paper's variance/kurtosis statistics implicitly assume.
    """
    L, N, H = activations.shape
    means = np.zeros((L, H), dtype=np.float32)
    R = np.zeros((L, H, H), dtype=np.float32)
    singular_values = np.zeros((L, H), dtype=np.float32)
    effective_rank = np.zeros(L, dtype=np.float32)
    kurtosis = np.zeros(L, dtype=np.float32)
    isotropy = np.zeros(L, dtype=np.float32)

    # Step 1: per-layer SVD.
    # CONVENTION: we store R(t) with principal directions as ROWS (matching
    # numpy's SVD Vt output). This differs from the paper's convention where
    # R has principal directions as columns. The two are related by transpose.
    # R(t) here is stored for downstream alignment analysis (Phase 2 cross-
    # model comparisons); we do NOT use it directly for prediction.
    centered_activations = np.zeros_like(activations)
    # Keep U_t from each layer's SVD; we need it for the projection-based
    # residual computation below. U_t is (N, K) where K = min(N, H).
    # In our standard pilot regime N >> H so K = H; with the paper's
    # smaller-sample data N < H so K = N.
    K = min(N, H)
    U_per_layer = np.zeros((L, N, K), dtype=np.float32)
    for t in range(L):
        x = activations[t]  # (N, H)
        mu = x.mean(axis=0) if center else np.zeros(H, dtype=np.float32)
        means[t] = mu
        xc = x - mu
        centered_activations[t] = xc
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        # u has shape (N, K), s has shape (K,), vt has shape (K, H).
        U_per_layer[t] = u
        # Singular value vector padded to length H so downstream code that
        # expects H slots keeps working.
        sv_full = np.zeros(H, dtype=np.float32)
        sv_full[:K] = s / np.sqrt(max(N, 1))  # singular value -> per-direction std
        singular_values[t] = sv_full
        # R(t) padded similarly. The first K rows are the real principal vectors;
        # the rest are zero-padding to keep shape consistent at (H, H).
        R_full = np.zeros((H, H), dtype=np.float32)
        R_full[:K] = vt
        R[t] = R_full

        # Effective rank: exp of entropy of normalized squared singular values.
        sv2 = sv_full * sv_full
        sv2_norm = sv2 / max(sv2.sum(), 1e-30)
        entropy = -np.sum(np.where(sv2_norm > 0, sv2_norm * np.log(sv2_norm), 0.0))
        effective_rank[t] = np.exp(entropy)

    # Step 2: pairwise residuals via projection onto source-layer column space.
    # 
    # The OLS prediction X_pred = X_t @ A where A = (X_t^T X_t)^{-1} X_t^T X_target
    # simplifies under SVD of X_t = U_t S_t V_t^T to:
    #     X_pred = U_t U_t^T X_target
    # i.e., the prediction is the projection of X_target onto X_t's column space.
    # We compute this efficiently as X_pred = U_t @ (U_t.T @ X_target), which is
    # two (N, H) @ (H, H) matrix multiplies — much faster than calling lstsq
    # for each (source, target) pair separately, because lstsq does an SVD
    # internally on every call (~5× total speedup).
    pairwise_var = np.full((L, L), np.nan, dtype=np.float32)
    # Paper convention: mean over coordinates of log(var_d), per (source, target).
    # Computed alongside the "ours" mean_d(var_d). The two differ by Jensen's
    # inequality and have been a source of confusion when comparing to the
    # paper's published log α — see PAPER_CODE_REVIEW.md §4.5/§8.
    pairwise_mean_log_var = np.full((L, L), np.nan, dtype=np.float32)
    residuals_by_endpoint = [[] for _ in range(L)]

    for t in range(L - 1):
        U_t = U_per_layer[t]  # (N, H)
        for target in range(t + 1, L):
            X_target = centered_activations[target]  # (N, H)
            # Project X_target onto U_t's column space (= X_t's column space).
            coords = U_t.T @ X_target           # (H, H) — coords in U_t basis
            X_pred = U_t @ coords               # (N, H) — back to ambient space
            w = X_target - X_pred               # (N, H) — residual
            sq_norm = (w * w).sum(axis=1)
            pairwise_var[t, target] = sq_norm.mean() / H
            # Per-coordinate variance (mean of w² along the N axis) then
            # mean of its log. We guard the log against zero variance
            # coordinates, which can happen when the projection exactly
            # recovers X_target along some directions.
            var_per_dim_pair = (w * w).mean(axis=0)
            pairwise_mean_log_var[t, target] = float(
                np.log(np.maximum(var_per_dim_pair, 1e-30)).mean()
            )
            residuals_by_endpoint[target].append(w)

    # Step 3: variance scaling fit. log σ² grows linearly in (t+τ).
    # Compute both conventions:
    #   "ours":  log(mean_d var_d)  — fit log(pairwise_var.mean over sources) vs end
    #   "paper": mean_d log(var_d)  — fit pairwise_mean_log_var.mean over sources vs end
    # The two fits use the same x-axis (endpoint layer index) and the same
    # marginalization (averaging across source layers at each endpoint).
    endpoint_indices = []
    endpoint_log_var = []         # "ours" convention
    endpoint_mean_log_var = []    # "paper" convention
    for end in range(1, L):
        vars_at_end = pairwise_var[:end, end]
        vars_at_end = vars_at_end[~np.isnan(vars_at_end)]
        mlv_at_end = pairwise_mean_log_var[:end, end]
        mlv_at_end = mlv_at_end[~np.isnan(mlv_at_end)]
        if len(vars_at_end) == 0:
            continue
        endpoint_indices.append(end)
        endpoint_log_var.append(np.log(vars_at_end.mean()))
        endpoint_mean_log_var.append(mlv_at_end.mean())
    endpoint_indices = np.array(endpoint_indices, dtype=np.float32)
    endpoint_log_var = np.array(endpoint_log_var, dtype=np.float32)
    endpoint_mean_log_var = np.array(endpoint_mean_log_var, dtype=np.float32)
    if len(endpoint_indices) >= 2:
        lam, log_alpha = np.polyfit(endpoint_indices, endpoint_log_var, deg=1)
        lam_paper, log_alpha_paper = np.polyfit(
            endpoint_indices, endpoint_mean_log_var, deg=1,
        )
    else:
        lam, log_alpha = float("nan"), float("nan")
        lam_paper, log_alpha_paper = float("nan"), float("nan")

    # Step 4: Gaussianity diagnostics per endpoint layer.
    # IMPORTANT: residuals from different source layers (t1, end), (t2, end)
    # have different variances even though both are Gaussian by assumption.
    # Concatenating them creates a mixture-of-Gaussians which has spurious
    # excess kurtosis. Instead, we compute the kurtosis of each (t, end)
    # pair separately *after standardizing* (subtracting mean and dividing
    # by per-dim std), then average those standardized kurtoses.
    #
    # Two conventions for aggregating per-coordinate kurtosis κᵢ to a single
    # number per endpoint:
    #   "ours":  mean_d κᵢ        — signed mean, can cancel across coords.
    #   "paper": mean_d |κᵢ|      — Sarfati et al. report <|κ|> values.
    # Both are stored.
    kurtosis_abs = np.full(L, np.nan, dtype=np.float32)
    for end in range(L):
        if not residuals_by_endpoint[end]:
            kurtosis[end] = float("nan")
            kurtosis_abs[end] = float("nan")
            isotropy[end] = float("nan")
            continue
        # Compute kurtosis per source layer separately.
        per_source_kurt = []
        per_source_kurt_abs = []
        per_source_iso = []
        for w_source in residuals_by_endpoint[end]:
            m = w_source.mean(axis=0, keepdims=True)
            w_c = w_source - m
            var_per_dim = (w_c * w_c).mean(axis=0)
            var_per_dim_safe = np.where(var_per_dim > 1e-30, var_per_dim, 1.0)
            m4 = (w_c ** 4).mean(axis=0)
            excess_kurt_per_dim = m4 / (var_per_dim_safe * var_per_dim_safe) - 3.0
            per_source_kurt.append(excess_kurt_per_dim.mean())
            per_source_kurt_abs.append(np.abs(excess_kurt_per_dim).mean())
            log_var = np.log(np.maximum(var_per_dim, 1e-30))
            per_source_iso.append(log_var.std())
        kurtosis[end] = float(np.mean(per_source_kurt))
        kurtosis_abs[end] = float(np.mean(per_source_kurt_abs))
        isotropy[end] = float(np.mean(per_source_iso))

    return {
        "R": R,
        "singular_values": singular_values,
        "means": means,
        "effective_rank": effective_rank,
        "pairwise_residual_variance": pairwise_var,
        "pairwise_mean_log_var": pairwise_mean_log_var,
        "endpoint_indices": endpoint_indices,
        "endpoint_log_var": endpoint_log_var,
        "endpoint_mean_log_var": endpoint_mean_log_var,
        "log_alpha": float(log_alpha),
        "lambda": float(lam),
        "log_alpha_paper": float(log_alpha_paper),
        "lambda_paper": float(lam_paper),
        "kurtosis_per_layer": kurtosis,
        "kurtosis_abs_per_layer": kurtosis_abs,
        "isotropy_per_layer": isotropy,
        "num_layers_total": L,
        "hidden_dim": H,
        "num_pilots": N,
        "centered": center,
    }


# ----------------------------------------------------------------------
# Top-level checkpoint analysis.
# ----------------------------------------------------------------------
def analyze_checkpoint(
    checkpoint_path: str,
    eval_loader,
    model_cfg: ModelConfig,
    device: str = "cuda",
    pilot_positions: Optional[List[int]] = None,
    max_pilots: int = 100_000,
    verbose: bool = True,
    statistic_mode: str = "both",
) -> Dict:
    """
    Load a checkpoint, run inference on eval_loader, recover the linear flow.

    Returns the flow dict (see recover_linear_flow) augmented with:
      - 'checkpoint_step': training step at which checkpoint was taken
      - 'checkpoint_path': source path
      - 'pilot_positions': positions sampled within each sequence
      - 'analysis_time_sec': wall-clock time for the analysis

    Both statistic conventions (ours / paper) are always computed and saved.
    The `statistic_mode` argument only affects which values are printed in
    the verbose summary line. Valid values: 'ours', 'paper', 'both'.
    """
    assert statistic_mode in ("ours", "paper", "both"), (
        f"statistic_mode must be one of 'ours', 'paper', 'both'; got {statistic_mode!r}"
    )
    t_start = time.time()
    if verbose:
        print(f"  Analyzing {os.path.basename(checkpoint_path)} ...")

    # Load checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Autocast dtype matches training.
    if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    # Pilot positions.
    if pilot_positions is None:
        # We can't know the eval seq_len without inspecting the loader; pick
        # a reasonable default that will be trimmed in collect_activations.
        pilot_positions = default_pilot_positions(seq_len=2048)

    # Collect activations.
    if verbose:
        print(f"    Collecting activations ...")
    activations = collect_activations(
        model=model, eval_loader=eval_loader,
        pilot_positions=pilot_positions, device=device,
        autocast_dtype=autocast_dtype, max_pilots=max_pilots,
    )
    if verbose:
        N = activations.shape[1]
        print(f"      Collected {N:,} pilot activations × "
              f"{activations.shape[0]} layers × {activations.shape[2]} dims")

    # Free model VRAM before doing CPU-bound SVDs.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    # Recover the flow.
    if verbose:
        print(f"    Computing linear flow (per-layer SVD + pairwise residuals) ...")
    flow = recover_linear_flow(activations, center=True)

    # Augment with checkpoint metadata.
    flow["checkpoint_step"] = ckpt["step"]
    flow["checkpoint_path"] = checkpoint_path
    flow["checkpoint_loss"] = ckpt.get("loss", float("nan"))
    flow["checkpoint_eval_loss"] = ckpt.get("eval_loss", float("nan"))
    flow["checkpoint_seed"] = ckpt.get("seed", -1)
    flow["pilot_positions"] = pilot_positions
    flow["analysis_time_sec"] = time.time() - t_start

    if verbose:
        eff_rank_0 = flow['effective_rank'][0]
        elapsed = flow['analysis_time_sec']
        if statistic_mode == "ours":
            print(f"    ↳ Step {ckpt['step']}: "
                  f"λ = {flow['lambda']:.4f}, "
                  f"log α = {flow['log_alpha']:.4f}, "
                  f"eff rank L0 = {eff_rank_0:.1f}, "
                  f"<κ> = {np.nanmean(flow['kurtosis_per_layer']):+.3f} "
                  f"[{elapsed:.1f}s]")
        elif statistic_mode == "paper":
            print(f"    ↳ Step {ckpt['step']}: "
                  f"λ = {flow['lambda_paper']:.4f} (paper), "
                  f"log α = {flow['log_alpha_paper']:.4f} (paper), "
                  f"eff rank L0 = {eff_rank_0:.1f}, "
                  f"<|κ|> = {np.nanmean(flow['kurtosis_abs_per_layer']):.3f} "
                  f"[{elapsed:.1f}s]")
        else:  # both
            print(f"    ↳ Step {ckpt['step']}: "
                  f"λ = {flow['lambda']:.4f} / {flow['lambda_paper']:.4f} (ours/paper), "
                  f"log α = {flow['log_alpha']:.4f} / {flow['log_alpha_paper']:.4f}, "
                  f"eff rank L0 = {eff_rank_0:.1f}, "
                  f"<κ>={np.nanmean(flow['kurtosis_per_layer']):+.3f} "
                  f"<|κ|>={np.nanmean(flow['kurtosis_abs_per_layer']):.3f} "
                  f"[{elapsed:.1f}s]")

    return flow


# ----------------------------------------------------------------------
# Disk I/O.
# ----------------------------------------------------------------------
def save_flow(flow: Dict, output_path: str):
    """Save a flow dict to .npz format.

    Keeps array-valued fields as arrays and scalar metadata as scalars.
    The .npz format preserves dtypes and is much smaller than torch.save."""
    arr_keys = {
        "R", "singular_values", "means",
        "effective_rank",
        "pairwise_residual_variance", "pairwise_mean_log_var",
        "endpoint_indices", "endpoint_log_var", "endpoint_mean_log_var",
        "kurtosis_per_layer", "kurtosis_abs_per_layer", "isotropy_per_layer",
    }
    scalar_keys = {
        "log_alpha", "lambda",
        "log_alpha_paper", "lambda_paper",
        "num_layers_total", "hidden_dim", "num_pilots",
        "centered", "checkpoint_step", "checkpoint_path", "checkpoint_loss",
        "checkpoint_eval_loss", "checkpoint_seed", "pilot_positions",
        "analysis_time_sec",
    }
    arrays = {k: flow[k] for k in arr_keys if k in flow}
    scalars = {k: flow[k] for k in scalar_keys if k in flow}
    # NumPy savez can hold scalars too, but we store them as 0-d arrays.
    for k, v in scalars.items():
        if isinstance(v, (list, tuple)):
            arrays[k] = np.array(v)
        else:
            arrays[k] = np.array(v)
    np.savez_compressed(output_path, **arrays)


def load_flow(path: str) -> Dict:
    """Load a flow .npz back into a Python dict, restoring scalar types."""
    raw = np.load(path, allow_pickle=False)
    flow = {}
    for key in raw.files:
        val = raw[key]
        # 0-d arrays representing scalars: extract.
        if val.ndim == 0:
            flow[key] = val.item()
        else:
            flow[key] = val
    return flow


# ----------------------------------------------------------------------
# Run-level orchestration: analyze every checkpoint in a completed run.
# ----------------------------------------------------------------------
def analyze_run(
    run_dir: str,
    eval_loader,
    device: str = "cuda",
    output_subdir: str = "flow_analysis",
    skip_existing: bool = True,
    max_pilots: int = 100_000,
    statistic_mode: str = "both",
):
    """
    Analyze every checkpoint in `run_dir/checkpoints/`, writing one .npz
    per checkpoint to `run_dir/<output_subdir>/`.

    Args:
        run_dir: A completed training run directory (must contain
                 'checkpoints/' and 'run_metadata.json').
        eval_loader: A DataLoader yielding evaluation batches (typically
                     made from the same held-out set as training).
        device: 'cuda' or 'cpu'.
        output_subdir: Subdirectory within run_dir to write analysis files.
        skip_existing: If True, don't reanalyze checkpoints whose output
                       already exists (useful for incremental analysis as
                       training progresses).
        max_pilots: Maximum number of pilot activations to collect per
                    checkpoint. Limited in practice by the eval set size.
        statistic_mode: Display convention for the verbose per-checkpoint
                    summary line ('ours', 'paper', or 'both'). Both
                    conventions are always computed and saved regardless;
                    this argument only controls console output.
    """
    # Load metadata.
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_cfg = ModelConfig(**metadata["model"])

    # Setup output dir.
    output_dir = os.path.join(run_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Find all checkpoints.
    checkpoint_files = sorted(glob.glob(
        os.path.join(run_dir, "checkpoints", "step_*.pt")
    ))
    if not checkpoint_files:
        print(f"  ⚠️  No checkpoints found in {run_dir}/checkpoints/")
        return

    print(f">> Analyzing {len(checkpoint_files)} checkpoints in {run_dir} ...")
    print()

    # Pilot positions: pick based on the eval loader's actual seq_len.
    # We peek at one batch to determine T.
    sample_batch = next(iter(eval_loader))
    seq_len = sample_batch["input_ids"].size(1)
    pilot_positions = default_pilot_positions(seq_len=seq_len)
    print(f"   Using {len(pilot_positions)} pilot positions per sequence "
          f"(seq_len = {seq_len})")
    print(f"   Pilot positions: {pilot_positions}")
    print()

    analyzed = 0
    skipped = 0
    for ckpt_path in checkpoint_files:
        step_str = os.path.basename(ckpt_path).replace("step_", "").replace(".pt", "")
        output_path = os.path.join(output_dir, f"flow_step_{step_str}.npz")
        if skip_existing and os.path.exists(output_path):
            skipped += 1
            continue
        flow = analyze_checkpoint(
            checkpoint_path=ckpt_path,
            eval_loader=eval_loader,
            model_cfg=model_cfg,
            device=device,
            pilot_positions=pilot_positions,
            max_pilots=max_pilots,
            verbose=True,
            statistic_mode=statistic_mode,
        )
        save_flow(flow, output_path)
        analyzed += 1

    print()
    print(f">> ✅ Analyzed {analyzed} checkpoints"
          + (f" (skipped {skipped} existing)" if skipped else ""))
    print(f"   Outputs: {output_dir}")
    