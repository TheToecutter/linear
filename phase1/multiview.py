"""
Multi-view decomposition of the residual-stream bundle.

This module extends the Phase 1 analyzer to support input-conditioned
(forward view) and output-conditioned (reverse view) ensembles, plus
the law-of-total-variance bookkeeping that ties them to the all-to-all
marginal.

The contract is:

  1. `collect_activations_with_metadata` runs one inference pass over the
     held-out set and saves, for each of N pilots, both the (L_total, H)
     hidden-state stack AND three integer tags: (input_token_id,
     next_token_id, predicted_token_id).

  2. `conditional_flow` is a thin wrapper around the Phase 1
     `analyze.recover_linear_flow`. It takes an activation tensor, a
     filter array, and a target value; it slices the activations and
     calls the standard pipeline. With a no-op filter it must reproduce
     the Phase 1 all-to-all numbers exactly.

  3. `within_between_decomposition` implements the law of total variance
     bookkeeping. Given activations and a partition label, it returns
     V_within(t), V_between(t), and the all-to-all V_total(t) for
     comparison. By the LOTV identity, V_within + V_between equals the
     V_total of the subset that the partition labels cover (i.e., the
     pilots whose label is in the chosen token set).

  4. `crossover_layer` extracts the depth at which within-condition
     variance crosses between-condition variance. Returns NaN if no
     crossover exists.

All functions are deterministic given fixed inputs. None mutate their
arguments.

Hidden state convention follows analyze.py: layers 0..L_total-1 where
0 is post-embedding, 1..num_hidden_layers are per-block outputs, and
L_total-1 is post-final-norm.
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Phase 1 entry points.
from analyze import (
    collect_activations,
    recover_linear_flow,
    default_pilot_positions,
)

# torch and project model/config types are only needed by
# collect_activations_with_metadata; importing them lazily keeps the
# rest of the module testable without a full project install.
try:
    import torch
    from config import ModelConfig
    from model import LlamaStyleTransformer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    ModelConfig = None
    LlamaStyleTransformer = None


# ----------------------------------------------------------------------
# Augmented activation collection.
# ----------------------------------------------------------------------
def collect_activations_with_metadata(
    model,
    eval_loader,
    pilot_positions: List[int],
    device: str,
    autocast_dtype,
    max_pilots: int = 100_000,
    compute_predictions: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run inference on eval_loader; collect hidden states at pilot positions
    plus the three per-pilot integer tags needed for multi-view analysis.

    The implementation mirrors `analyze.collect_activations` but additionally
    captures, for each pilot:
      - the input token id at the pilot position
      - the actual next token id (position p+1)
      - optionally the predicted next token id (argmax of logits at p)

    The "predicted" token requires running the lm_head, which the Phase 1
    analyzer skips. We turn it on by default because it adds negligible
    cost (one matmul to V=32768) but enables the actual-vs-predicted
    reverse view comparison described in the proposal §4.7.

    Returns:
        dict with keys
          'states':    (L_total, N, H) float32
          'input_ids': (N,) int32  -- token at pilot position
          'next_ids':  (N,) int32  -- token at pilot position + 1 (actual successor)
          'pred_ids':  (N,) int32  -- argmax of model's prediction at pilot position
                                       (filled with -1 if compute_predictions=False)
          'positions': (N,) int32  -- which pilot position each row came from
        These are aligned: row k of states[:, k, :] has input_ids[k],
        next_ids[k], pred_ids[k], positions[k].

    Notes on pilot_positions and the next-token constraint:
      Because we need a valid next token at position p+1, any pilot at
      position p must have p+1 < T (within the chunk). The Phase 1 default
      trims pilots near the end of the sequence; that trimming is preserved
      here. If a pilot survives trimming, p+1 is guaranteed valid because
      the default trim leaves at least ~50 tokens of margin.
    """
    if not _HAS_TORCH:
        raise RuntimeError(
            "collect_activations_with_metadata requires torch and the "
            "project's model/config modules to be importable."
        )
    return _collect_activations_with_metadata_impl(
        model, eval_loader, pilot_positions, device, autocast_dtype,
        max_pilots, compute_predictions,
    )


def _collect_activations_with_metadata_impl(
    model, eval_loader, pilot_positions, device, autocast_dtype,
    max_pilots, compute_predictions,
):
    with torch.no_grad():
        return _collect_activations_with_metadata_body(
            model, eval_loader, pilot_positions, device, autocast_dtype,
            max_pilots, compute_predictions,
        )


def _collect_activations_with_metadata_body(
    model, eval_loader, pilot_positions, device, autocast_dtype,
    max_pilots, compute_predictions,
):
    model.eval()
    num_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    n_layer_outputs = num_layers + 2  # post-embed + L block outputs + post-final-norm

    state_buffers: List[List[np.ndarray]] = [[] for _ in range(n_layer_outputs)]
    input_id_chunks: List[np.ndarray] = []
    next_id_chunks: List[np.ndarray] = []
    pred_id_chunks: List[np.ndarray] = []
    pos_chunks: List[np.ndarray] = []
    collected = 0

    for batch in eval_loader:
        if collected >= max_pilots:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)  # (B, T)
        B, T = input_ids.shape

        # Validate pilot positions for this T. We require p+1 < T so that
        # input_ids[:, p+1] (the actual successor) is defined.
        valid_pos = [p for p in pilot_positions if 0 <= p and p + 1 < T]
        if not valid_pos:
            continue
        pos_idx = torch.tensor(valid_pos, device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", dtype=autocast_dtype,
                                 enabled=(device == "cuda")):
            logits, _, hidden = model(input_ids, return_hidden_states=True)

        # Hidden states: list of (B, T, H). Slice each at the valid pilot
        # positions and promote to fp32 on CPU.
        for layer_idx, h in enumerate(hidden):
            picked = h[:, pos_idx, :].float().cpu().numpy()  # (B, n_pilots, H)
            state_buffers[layer_idx].append(picked.reshape(-1, H))

        # Pilot-position tags: input token id at each pilot.
        input_tag = input_ids[:, pos_idx].cpu().numpy().astype(np.int32)  # (B, n_pilots)
        # Successor: input_ids at position p+1.
        next_pos = pos_idx + 1
        next_tag = input_ids[:, next_pos].cpu().numpy().astype(np.int32)

        if compute_predictions:
            # Predicted next token at each pilot position: argmax of logits.
            # logits has shape (B, T, V). The prediction made at position p
            # is logits[:, p, :] argmaxed over V.
            pred_tag = logits[:, pos_idx, :].argmax(dim=-1).cpu().numpy().astype(np.int32)
        else:
            pred_tag = -1 * np.ones((B, len(valid_pos)), dtype=np.int32)

        input_id_chunks.append(input_tag.reshape(-1))
        next_id_chunks.append(next_tag.reshape(-1))
        pred_id_chunks.append(pred_tag.reshape(-1))
        pos_tile = np.tile(np.array(valid_pos, dtype=np.int32), (B,))
        pos_chunks.append(pos_tile)

        collected += B * len(valid_pos)

    # Concatenate everything.
    N = collected
    states = np.zeros((n_layer_outputs, N, H), dtype=np.float32)
    for layer_idx in range(n_layer_outputs):
        if state_buffers[layer_idx]:
            cat = np.concatenate(state_buffers[layer_idx], axis=0)
            states[layer_idx] = cat[:N]
    input_ids_arr = np.concatenate(input_id_chunks, axis=0)[:N] if input_id_chunks else np.zeros((0,), dtype=np.int32)
    next_ids_arr = np.concatenate(next_id_chunks, axis=0)[:N] if next_id_chunks else np.zeros((0,), dtype=np.int32)
    pred_ids_arr = np.concatenate(pred_id_chunks, axis=0)[:N] if pred_id_chunks else np.zeros((0,), dtype=np.int32)
    positions_arr = np.concatenate(pos_chunks, axis=0)[:N] if pos_chunks else np.zeros((0,), dtype=np.int32)

    return {
        "states": states,
        "input_ids": input_ids_arr,
        "next_ids": next_ids_arr,
        "pred_ids": pred_ids_arr,
        "positions": positions_arr,
    }


def save_augmented_activations(payload: Dict[str, np.ndarray], path: str) -> None:
    """Save the output of collect_activations_with_metadata to .npz."""
    np.savez_compressed(
        path,
        states=payload["states"],
        input_ids=payload["input_ids"],
        next_ids=payload["next_ids"],
        pred_ids=payload["pred_ids"],
        positions=payload["positions"],
    )


def load_augmented_activations(path: str) -> Dict[str, np.ndarray]:
    """Load the output of save_augmented_activations from .npz."""
    with np.load(path) as f:
        return {
            "states": f["states"],
            "input_ids": f["input_ids"],
            "next_ids": f["next_ids"],
            "pred_ids": f["pred_ids"],
            "positions": f["positions"],
        }


# ----------------------------------------------------------------------
# Token-set selection.
# ----------------------------------------------------------------------
@dataclass
class TokenSet:
    """A selected set of token ids with their pilot counts.

    Attributes:
        view:        'forward' (input tokens), 'reverse_actual' (actual
                     successors), or 'reverse_pred' (predicted successors)
        token_ids:   sorted (descending by count) list of selected token ids
        counts:      pilot count for each token, parallel to token_ids
        min_count:   the minimum-count threshold used during selection
        total_pilots: total number of pilots in the source dataset
    """
    view: str
    token_ids: np.ndarray
    counts: np.ndarray
    min_count: int
    total_pilots: int

    def coverage_fraction(self) -> float:
        """Fraction of all pilots covered by the selected tokens."""
        if self.total_pilots == 0:
            return 0.0
        return float(self.counts.sum()) / float(self.total_pilots)

    def to_dict(self) -> Dict:
        return {
            "view": self.view,
            "token_ids": self.token_ids.tolist(),
            "counts": self.counts.tolist(),
            "min_count": int(self.min_count),
            "total_pilots": int(self.total_pilots),
            "coverage_fraction": self.coverage_fraction(),
        }


def select_token_set(
    tags: np.ndarray,
    view: str,
    top_k: int = 20,
    min_count: int = 50,
) -> TokenSet:
    """
    Select the top-k most frequent tokens from a tag array, subject to a
    minimum-count threshold.

    Args:
        tags: (N,) integer array (input_ids, next_ids, or pred_ids).
        view: descriptor; one of 'forward', 'reverse_actual', 'reverse_pred'.
        top_k: maximum number of tokens to return.
        min_count: minimum pilot count for a token to be eligible.

    Returns:
        TokenSet with up to top_k tokens that each appear at least
        min_count times in `tags`, ordered by count descending. If fewer
        than top_k tokens meet the threshold, returns however many do.
    """
    if tags.size == 0:
        return TokenSet(view=view, token_ids=np.zeros(0, dtype=np.int32),
                        counts=np.zeros(0, dtype=np.int64),
                        min_count=min_count, total_pilots=0)

    uniq, counts = np.unique(tags, return_counts=True)
    # Filter by min count.
    keep = counts >= min_count
    uniq = uniq[keep]
    counts = counts[keep]
    # Sort descending by count.
    order = np.argsort(-counts)
    uniq = uniq[order][:top_k]
    counts = counts[order][:top_k]

    return TokenSet(
        view=view,
        token_ids=uniq.astype(np.int32),
        counts=counts.astype(np.int64),
        min_count=min_count,
        total_pilots=int(tags.size),
    )


# ----------------------------------------------------------------------
# Conditional flow.
# ----------------------------------------------------------------------
def conditional_flow(
    activations: np.ndarray,
    tags: np.ndarray,
    target_id: int,
    center: bool = True,
    min_pilots: int = 10,
) -> Dict:
    """
    Run the Phase 1 linear-flow analyzer on the subset of activations
    whose tag equals target_id.

    Args:
        activations: (L_total, N, H) array (the `states` from
                     collect_activations_with_metadata).
        tags: (N,) integer array used for filtering.
        target_id: include only pilots with tags == target_id.
        center: passed to recover_linear_flow.
        min_pilots: minimum number of pilots required to attempt the SVD.
                    Subsets smaller than this return a NaN placeholder.

    Returns:
        Phase 1 flow dict (see analyze.recover_linear_flow), augmented with:
          - 'target_id': the filter value used
          - 'n_pilots': number of pilots in the conditional subset
          - 'failed': True if the SVD failed or the subset was too small;
                      in that case the flow's array fields are NaN-filled
                      placeholders matching the expected shape so downstream
                      aggregation can still proceed.

    Notes:
      The SVD can fail in two ways on conditional subsets:
        1. Too few pilots: <10 instances of the target token. Skipped
           preemptively because the variance/SVD estimates would be
           meaningless anyway.
        2. LAPACK non-convergence: rare, but possible with very early
           training checkpoints where prediction-conditioned subsets can
           have near-degenerate row structure. Caught and reported as
           failed=True rather than propagated.

      With a no-op filter (e.g., tags being a constant array equal to
      target_id), this function reduces exactly to the Phase 1 analyzer
      called on the full activations, by design.
    """
    mask = (tags == target_id)
    n = int(mask.sum())

    if n < min_pilots:
        return _failed_flow_placeholder(activations.shape, target_id, n,
                                        reason="too_few_pilots")

    sub = activations[:, mask, :]
    try:
        flow = recover_linear_flow(sub, center=center)
    except np.linalg.LinAlgError as e:
        # SVD didn't converge — typically happens on very early checkpoints
        # for prediction-conditioned views where many pilots collapse to
        # the same trivial prediction. Return a placeholder so the
        # campaign can continue.
        placeholder = _failed_flow_placeholder(activations.shape, target_id, n,
                                               reason=f"svd_failed: {e}")
        return placeholder

    flow["target_id"] = int(target_id)
    flow["n_pilots"] = n
    flow["failed"] = False
    return flow


def _failed_flow_placeholder(shape: Tuple[int, int, int],
                             target_id: int, n_pilots: int,
                             reason: str = "") -> Dict:
    """Construct a NaN-filled flow dict with the same shape contract as
    a successful recover_linear_flow call. Used when conditioning
    produces a degenerate subset and the SVD cannot or should not run.
    """
    L, _, H = shape
    return {
        "R": np.full((L, H, H), np.nan, dtype=np.float32),
        "singular_values": np.full((L, H), np.nan, dtype=np.float32),
        "means": np.full((L, H), np.nan, dtype=np.float32),
        "effective_rank": np.full(L, np.nan, dtype=np.float32),
        "kurtosis_per_layer": np.full(L, np.nan, dtype=np.float32),
        "isotropy_per_layer": np.full(L, np.nan, dtype=np.float32),
        "pairwise_residual_variance": np.full((L, L), np.nan, dtype=np.float64),
        "log_alpha": float("nan"),
        "lambda": float("nan"),
        "target_id": int(target_id),
        "n_pilots": int(n_pilots),
        "failed": True,
        "failure_reason": reason,
    }


# ----------------------------------------------------------------------
# Within/between variance decomposition.
# ----------------------------------------------------------------------
@dataclass
class DecompositionResult:
    """One within/between decomposition over a chosen token set.

    All arrays are length L_total (one entry per layer state).

    Attributes:
        view:           descriptor (forward / reverse_actual / reverse_pred).
        token_ids:      (K,) the tokens used for the decomposition.
        counts:         (K,) per-token pilot counts.
        v_within:       (L,) average within-condition per-coordinate variance,
                              i.e., E_v[ (1/H) sum_i Var_{k: tag_k=v}[x_k^{(i)}(t)] ]
                              with E_v weighted by counts (frequency-weighted).
        v_between:      (L,) variance of per-condition means,
                              i.e., (1/H) sum_i Var_v[ mu_t^{(i)}(v) ]
                              with the same frequency weighting.
        v_subset_total: (L,) per-coordinate total variance of the union of
                              pilots covered by the token set. By LOTV, equals
                              v_within + v_between up to floating point.
        v_all_to_all:   (L,) per-coordinate total variance of ALL pilots
                              (including those whose tag is not in token_ids).
        within_fraction: (L,) v_within / v_subset_total, in [0, 1].
        between_fraction:(L,) v_between / v_subset_total, in [0, 1].

    The within_fraction and between_fraction sum to 1.0 at every layer
    (modulo float precision); these are the "stacked area" curves in
    proposal §4.2. The v_subset_total / v_all_to_all ratio quantifies the
    coverage cost — how much variance the long tail (tokens outside the
    set) contributes.
    """
    view: str
    token_ids: np.ndarray
    counts: np.ndarray
    v_within: np.ndarray
    v_between: np.ndarray
    v_subset_total: np.ndarray
    v_all_to_all: np.ndarray

    @property
    def within_fraction(self) -> np.ndarray:
        return _safe_divide(self.v_within, self.v_subset_total)

    @property
    def between_fraction(self) -> np.ndarray:
        return _safe_divide(self.v_between, self.v_subset_total)

    @property
    def subset_coverage(self) -> np.ndarray:
        """v_subset_total / v_all_to_all — how much of the unconditional
        variance the chosen token set's pilots account for."""
        return _safe_divide(self.v_subset_total, self.v_all_to_all)


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a, dtype=np.float64)
    nz = np.abs(b) > 1e-30
    out[nz] = a[nz] / b[nz]
    return out


def within_between_decomposition(
    activations: np.ndarray,
    tags: np.ndarray,
    token_set: TokenSet,
) -> DecompositionResult:
    """
    Decompose per-coordinate variance into within/between components
    over the given token set, at every layer.

    Implementation:
      Let S = {k : tags[k] in token_set.token_ids}. For each token v in
      the set, let n_v = #{k in S : tags[k] = v}, mu(v, t, i) = mean of
      activations[t, k, i] over k with tags[k] = v.

      Frequency-weighted (i.e., natural) within variance at coord i, layer t:
        V_w(t, i) = (1/|S|) * sum_v n_v * Var_{k:tags[k]=v}[ activations[t, k, i] ]
                  = pooled within-cluster variance, treating each pilot
                    as one unit (this equals the standard "pooled within
                    variance" decomposition).

      Frequency-weighted between variance at coord i, layer t:
        V_b(t, i) = (1/|S|) * sum_v n_v * (mu(v, t, i) - mu_S(t, i))^2
        where mu_S(t, i) is the grand mean over S.

      By the LOTV identity:
        V_w(t, i) + V_b(t, i) = (1/|S|) * sum_{k in S} (x_k - mu_S)^2
                              = sample variance over S.

      We then average V_w and V_b across i to collapse to (L,) arrays.

    Args:
        activations: (L_total, N, H) array.
        tags: (N,) integer filter array.
        token_set: which tokens to include in the partition.

    Returns:
        DecompositionResult.

    Sanity: if you pass an empty token_set, all returned arrays except
    v_all_to_all are zeros.
    """
    L, N, H = activations.shape
    token_ids = token_set.token_ids
    K = token_ids.size

    # Build subset mask in one pass.
    in_set = np.isin(tags, token_ids)
    n_subset = int(in_set.sum())

    # All-to-all variance (per layer, averaged over coords).
    # ddof=0 to match population variance convention used by LOTV.
    v_all_to_all = np.zeros(L, dtype=np.float64)
    for t in range(L):
        v_all_to_all[t] = activations[t].var(axis=0, ddof=0).mean()

    if K == 0 or n_subset == 0:
        zero = np.zeros(L, dtype=np.float64)
        return DecompositionResult(
            view=token_set.view,
            token_ids=token_ids,
            counts=token_set.counts,
            v_within=zero.copy(),
            v_between=zero.copy(),
            v_subset_total=zero.copy(),
            v_all_to_all=v_all_to_all,
        )

    # Per-coord-per-layer accumulators.
    v_within_acc = np.zeros((L, H), dtype=np.float64)
    v_between_acc = np.zeros((L, H), dtype=np.float64)
    v_subset_total = np.zeros(L, dtype=np.float64)

    # We compute everything layer by layer to bound memory.
    for t in range(L):
        sub = activations[t, in_set, :].astype(np.float64)  # (n_subset, H)
        # Subset grand mean and total variance.
        mu_S = sub.mean(axis=0)  # (H,)
        v_total_t = sub.var(axis=0, ddof=0)  # (H,)
        v_subset_total[t] = v_total_t.mean()

        # Per-token means and within-variances.
        v_w_t = np.zeros(H, dtype=np.float64)
        v_b_t = np.zeros(H, dtype=np.float64)
        # For convenience, we re-mask within the subset.
        sub_tags = tags[in_set]
        for v_id, n_v in zip(token_ids, token_set.counts):
            mask_v = (sub_tags == v_id)
            n_v_actual = int(mask_v.sum())
            if n_v_actual == 0:
                continue
            xv = sub[mask_v]  # (n_v_actual, H)
            mu_v = xv.mean(axis=0)
            # Within-cluster contribution (frequency-weighted by n_v_actual).
            within_v = xv.var(axis=0, ddof=0)  # (H,)
            v_w_t += (n_v_actual / n_subset) * within_v
            # Between-cluster contribution.
            v_b_t += (n_v_actual / n_subset) * (mu_v - mu_S) ** 2

        v_within_acc[t] = v_w_t
        v_between_acc[t] = v_b_t

    # Average across coords.
    v_within = v_within_acc.mean(axis=1)
    v_between = v_between_acc.mean(axis=1)

    return DecompositionResult(
        view=token_set.view,
        token_ids=token_ids,
        counts=token_set.counts,
        v_within=v_within,
        v_between=v_between,
        v_subset_total=v_subset_total,
        v_all_to_all=v_all_to_all,
    )


# ----------------------------------------------------------------------
# Crossover layer.
# ----------------------------------------------------------------------
def crossover_layer(
    v_within: np.ndarray,
    v_between: np.ndarray,
    direction: str,
) -> Tuple[float, str]:
    """
    Identify the depth at which within and between variances cross.

    Args:
        v_within: (L,) within-condition variance, layer by layer.
        v_between: (L,) between-condition variance, layer by layer.
        direction: 'forward' -> find smallest t where v_within(t) > v_between(t)
                                (input-id loses out to context)
                   'reverse' -> find smallest t where v_between(t) > v_within(t)
                                (output-id wins over input-residue)

    Returns:
        (crossover_t, status) where status is one of:
          'crossover'    -- found a clean crossover at t (interpolated layer index)
          'no_crossover' -- the inequality is never satisfied; returns NaN
          'always_true'  -- the inequality holds at layer 0 already; returns 0.0
          'tied'         -- equality at exactly one t (degenerate); returns that t

    The returned layer index can be fractional via linear interpolation
    between the two layers where the crossover happens. This lets us
    track sub-layer shifts in crossover location across training, which
    are easily lost if we round to integer.
    """
    if v_within.shape != v_between.shape:
        raise ValueError("v_within and v_between must have the same shape")

    if direction == "forward":
        sign = v_within - v_between
    elif direction == "reverse":
        sign = v_between - v_within
    else:
        raise ValueError(f"Unknown direction: {direction}")

    L = sign.size
    # Already satisfied at t=0?
    if sign[0] > 0:
        return 0.0, "always_true"

    # Find first sign change.
    for t in range(1, L):
        if sign[t] > 0:
            # Linear interpolation between t-1 and t.
            a, b = sign[t - 1], sign[t]
            if b - a == 0:
                return float(t), "tied"
            frac = -a / (b - a)  # in [0, 1]
            return float(t - 1) + frac, "crossover"
        if sign[t] == 0:
            return float(t), "tied"

    return float("nan"), "no_crossover"


# ----------------------------------------------------------------------
# Top-level entry: full multi-view analysis on one augmented activation set.
# ----------------------------------------------------------------------
@dataclass
class MultiViewResult:
    """Bundle of all multi-view outputs from one (seed, checkpoint).

    Attributes:
        step:             training step
        seed:             training seed
        all_to_all:       Phase 1 flow dict (the marginal)
        forward_set:      TokenSet used for the forward view
        forward_flows:    {token_id: flow_dict} per-token Phase 1 flow,
                          computed on the input-conditioned subset
        forward_decomp:   DecompositionResult for the forward view
        reverse_actual_set:    TokenSet for actual-successor reverse view
        reverse_actual_flows:  {token_id: flow_dict}
        reverse_actual_decomp: DecompositionResult
        reverse_pred_set:      TokenSet for predicted-successor reverse view
        reverse_pred_flows:    {token_id: flow_dict}
        reverse_pred_decomp:   DecompositionResult
    """
    step: int
    seed: int
    all_to_all: Dict
    forward_set: TokenSet
    forward_flows: Dict[int, Dict]
    forward_decomp: DecompositionResult
    reverse_actual_set: TokenSet
    reverse_actual_flows: Dict[int, Dict]
    reverse_actual_decomp: DecompositionResult
    reverse_pred_set: TokenSet
    reverse_pred_flows: Dict[int, Dict]
    reverse_pred_decomp: DecompositionResult


def run_multi_view(
    augmented: Dict[str, np.ndarray],
    forward_set: TokenSet,
    reverse_actual_set: TokenSet,
    reverse_pred_set: Optional[TokenSet] = None,
    step: int = -1,
    seed: int = -1,
) -> MultiViewResult:
    """
    Compute all multi-view outputs for one augmented activation set.

    Token sets are passed in (not selected here) because a project-wide
    rule of thumb is to fix the token sets ONCE — typically at the final
    checkpoint of seed 0 — and reuse them for every (seed, checkpoint).
    That keeps the per-token flows comparable across training and across
    seeds; selecting separately per checkpoint would let the set drift
    with token-frequency noise.

    Args:
        augmented: output of collect_activations_with_metadata (or its
                   loaded form).
        forward_set: input-token set; tags will be 'input_ids'.
        reverse_actual_set: successor token set; tags will be 'next_ids'.
        reverse_pred_set: predicted-successor token set; tags will be
                          'pred_ids'. If None, the predicted-successor view
                          is skipped (returns empty flows and a zero
                          DecompositionResult).
        step, seed: metadata recorded on the result.

    Returns:
        MultiViewResult.
    """
    states = augmented["states"]
    input_ids = augmented["input_ids"]
    next_ids = augmented["next_ids"]
    pred_ids = augmented["pred_ids"]

    # All-to-all (Phase 1 marginal).
    all_to_all = recover_linear_flow(states, center=True)

    # Forward view.
    forward_flows = {
        int(tid): conditional_flow(states, input_ids, int(tid))
        for tid in forward_set.token_ids
    }
    forward_decomp = within_between_decomposition(states, input_ids, forward_set)

    # Reverse-actual view.
    reverse_actual_flows = {
        int(tid): conditional_flow(states, next_ids, int(tid))
        for tid in reverse_actual_set.token_ids
    }
    reverse_actual_decomp = within_between_decomposition(states, next_ids,
                                                         reverse_actual_set)

    # Reverse-predicted view (optional).
    if reverse_pred_set is not None and reverse_pred_set.token_ids.size > 0:
        reverse_pred_flows = {
            int(tid): conditional_flow(states, pred_ids, int(tid))
            for tid in reverse_pred_set.token_ids
        }
        reverse_pred_decomp = within_between_decomposition(states, pred_ids,
                                                           reverse_pred_set)
    else:
        # Empty placeholder.
        empty_set = reverse_pred_set if reverse_pred_set is not None else TokenSet(
            view="reverse_pred", token_ids=np.zeros(0, dtype=np.int32),
            counts=np.zeros(0, dtype=np.int64), min_count=0, total_pilots=0,
        )
        reverse_pred_flows = {}
        L = states.shape[0]
        zero = np.zeros(L, dtype=np.float64)
        v_all = np.array([states[t].var(axis=0, ddof=0).mean() for t in range(L)])
        reverse_pred_decomp = DecompositionResult(
            view="reverse_pred",
            token_ids=empty_set.token_ids,
            counts=empty_set.counts,
            v_within=zero.copy(),
            v_between=zero.copy(),
            v_subset_total=zero.copy(),
            v_all_to_all=v_all,
        )
        reverse_pred_set = empty_set

    return MultiViewResult(
        step=step,
        seed=seed,
        all_to_all=all_to_all,
        forward_set=forward_set,
        forward_flows=forward_flows,
        forward_decomp=forward_decomp,
        reverse_actual_set=reverse_actual_set,
        reverse_actual_flows=reverse_actual_flows,
        reverse_actual_decomp=reverse_actual_decomp,
        reverse_pred_set=reverse_pred_set,
        reverse_pred_flows=reverse_pred_flows,
        reverse_pred_decomp=reverse_pred_decomp,
    )


# ----------------------------------------------------------------------
# Disk I/O for multi-view results.
# ----------------------------------------------------------------------
def save_multi_view_result(result: MultiViewResult, output_dir: str) -> None:
    """
    Save a MultiViewResult as a directory tree.

    Layout:
        output_dir/
          meta.json              -- step, seed, token sets, coverage summary
          all_to_all.npz         -- Phase 1 flow dict (same format as analyze.py)
          decomp_forward.npz     -- forward DecompositionResult
          decomp_reverse_actual.npz
          decomp_reverse_pred.npz
          flows_forward/<tid>.npz       -- one per-token flow
          flows_reverse_actual/<tid>.npz
          flows_reverse_pred/<tid>.npz
    """
    os.makedirs(output_dir, exist_ok=True)

    meta = {
        "step": int(result.step),
        "seed": int(result.seed),
        "forward_set": result.forward_set.to_dict(),
        "reverse_actual_set": result.reverse_actual_set.to_dict(),
        "reverse_pred_set": result.reverse_pred_set.to_dict(),
    }
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    _save_flow_dict(result.all_to_all, os.path.join(output_dir, "all_to_all.npz"))
    _save_decomp(result.forward_decomp,
                 os.path.join(output_dir, "decomp_forward.npz"))
    _save_decomp(result.reverse_actual_decomp,
                 os.path.join(output_dir, "decomp_reverse_actual.npz"))
    _save_decomp(result.reverse_pred_decomp,
                 os.path.join(output_dir, "decomp_reverse_pred.npz"))

    for view_name, flows in [
        ("forward", result.forward_flows),
        ("reverse_actual", result.reverse_actual_flows),
        ("reverse_pred", result.reverse_pred_flows),
    ]:
        sub = os.path.join(output_dir, f"flows_{view_name}")
        os.makedirs(sub, exist_ok=True)
        for tid, flow in flows.items():
            _save_flow_dict(flow, os.path.join(sub, f"{tid}.npz"))


def load_multi_view_result(input_dir: str,
                           skip_arrays: Optional[set] = None) -> MultiViewResult:
    """Reload a saved MultiViewResult.

    Args:
        input_dir: directory written by save_multi_view_result.
        skip_arrays: optional set of array names to skip when reading the
            per-flow .npz files. Skipped arrays appear in the returned
            flow dicts as None (so callers can detect their absence).
            Used by Stage D to avoid loading the huge `R` rotation
            matrices (~44 MB per per-token flow file) when only the
            scalar summaries and small arrays are needed for trajectory
            aggregation. With four seeds × 50 checkpoints × 60 token
            flows, this saves about 80 GB of disk reads.

    Returns:
        MultiViewResult. If skip_arrays is set, the corresponding fields
        within each flow dict will be None instead of np.ndarray.
    """
    with open(os.path.join(input_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    def _set_from_dict(d):
        return TokenSet(
            view=d["view"],
            token_ids=np.array(d["token_ids"], dtype=np.int32),
            counts=np.array(d["counts"], dtype=np.int64),
            min_count=int(d["min_count"]),
            total_pilots=int(d["total_pilots"]),
        )

    forward_set = _set_from_dict(meta["forward_set"])
    reverse_actual_set = _set_from_dict(meta["reverse_actual_set"])
    reverse_pred_set = _set_from_dict(meta["reverse_pred_set"])

    all_to_all = _load_flow_dict(os.path.join(input_dir, "all_to_all.npz"),
                                 skip_arrays=skip_arrays)
    forward_decomp = _load_decomp(os.path.join(input_dir, "decomp_forward.npz"),
                                  forward_set)
    reverse_actual_decomp = _load_decomp(
        os.path.join(input_dir, "decomp_reverse_actual.npz"), reverse_actual_set)
    reverse_pred_decomp = _load_decomp(
        os.path.join(input_dir, "decomp_reverse_pred.npz"), reverse_pred_set)

    def _load_flows(view_name, ts):
        sub = os.path.join(input_dir, f"flows_{view_name}")
        out = {}
        for tid in ts.token_ids:
            p = os.path.join(sub, f"{int(tid)}.npz")
            if os.path.exists(p):
                out[int(tid)] = _load_flow_dict(p, skip_arrays=skip_arrays)
        return out

    return MultiViewResult(
        step=int(meta["step"]),
        seed=int(meta["seed"]),
        all_to_all=all_to_all,
        forward_set=forward_set,
        forward_flows=_load_flows("forward", forward_set),
        forward_decomp=forward_decomp,
        reverse_actual_set=reverse_actual_set,
        reverse_actual_flows=_load_flows("reverse_actual", reverse_actual_set),
        reverse_actual_decomp=reverse_actual_decomp,
        reverse_pred_set=reverse_pred_set,
        reverse_pred_flows=_load_flows("reverse_pred", reverse_pred_set),
        reverse_pred_decomp=reverse_pred_decomp,
    )


def _save_flow_dict(flow: Dict, path: str) -> None:
    """Save a Phase 1 flow dict to .npz. Skips non-array values."""
    arrays = {}
    scalars = {}
    for k, v in flow.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, (int, float, np.integer, np.floating)):
            scalars[k] = float(v)
        elif isinstance(v, (list, tuple)):
            try:
                arr = np.asarray(v)
                if arr.dtype != object:
                    arrays[k] = arr
            except Exception:
                pass
        # We drop strings (e.g., 'checkpoint_path') for now; they're not
        # critical for downstream analysis.
    if scalars:
        arrays["_scalars_keys"] = np.array(list(scalars.keys()))
        arrays["_scalars_values"] = np.array(list(scalars.values()), dtype=np.float64)
    np.savez_compressed(path, **arrays)


def _load_flow_dict(path: str,
                    skip_arrays: Optional[set] = None) -> Dict:
    """Load a flow dict from .npz. Optionally skip specified array keys
    to save I/O and memory; skipped keys appear in the result as None."""
    skip = skip_arrays or set()
    out: Dict = {}
    with np.load(path, allow_pickle=False) as f:
        keys = list(f.keys())
        if "_scalars_keys" in keys and "_scalars_values" in keys:
            sk = f["_scalars_keys"]
            sv = f["_scalars_values"]
            for k, v in zip(sk, sv):
                out[str(k)] = float(v)
            keys = [k for k in keys if not k.startswith("_scalars_")]
        for k in keys:
            if k in skip:
                # Don't materialize this array; record None so callers
                # can detect its absence if they care.
                out[k] = None
            else:
                # Force a copy out of the .npz handle so we can close it
                # cleanly and not hold a memory map.
                out[k] = np.asarray(f[k])
    return out


def _save_decomp(d: DecompositionResult, path: str) -> None:
    np.savez_compressed(
        path,
        view=np.array(d.view),
        token_ids=d.token_ids,
        counts=d.counts,
        v_within=d.v_within,
        v_between=d.v_between,
        v_subset_total=d.v_subset_total,
        v_all_to_all=d.v_all_to_all,
    )


def _load_decomp(path: str, fallback_set: TokenSet) -> DecompositionResult:
    with np.load(path, allow_pickle=False) as f:
        return DecompositionResult(
            view=str(f["view"]),
            token_ids=f["token_ids"],
            counts=f["counts"],
            v_within=f["v_within"],
            v_between=f["v_between"],
            v_subset_total=f["v_subset_total"],
            v_all_to_all=f["v_all_to_all"],
        )
        