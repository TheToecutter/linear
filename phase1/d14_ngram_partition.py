"""
10.3 n-gram partition test.

Tests Possibility 3: high-order context dependency. The relevant
mixture variable for the C-verdict heavy tails may be a function of
multiple positions (e.g., the (prev_token, next_token) 2-gram) rather
than any single-position partition we have tested.

This script:

  1. Regenerates stage A inference at the four representative
     checkpoints with PREV_TOKEN saved alongside the existing fields.
     Output filename pattern: augmented_step_NNNNNNNN_ngram.npz.
     This is a small additional stage A pass (~10 min for 16
     (seed, step) pairs).

  2. Runs joint (prev, next) sub-conditioning on the new files at the
     four checkpoints, with a properly matched random-labels null
     control. We test partition variables {prev, next, (prev, next)
     joint}, all with null controls matched to each partition's
     effective bin count per input token.

Reading: the headline statistic is the "real signal" (null_kurt -
partition_kurt) at Phase III final, layer 7, for each partition. If
the joint (prev, next) partition has substantially higher real signal
than next alone or prev alone, the C verdict has high-order context
dependency and the mixture index needs more than a single position
variable. If joint and individual partitions all give similar small
real signals, n-gram structure is not the explanation either, and we
are left with Possibility 1 (intrinsic heavy tails).

The script defaults to fixed pilot positions (matching the original
campaign scheme), because the (prev, next) joint identifiability
question is at the per-pilot level and does not require position
randomization.

Output:
    run_dir/multiview/seed_S/augmented_step_NNNNNNNN_ngram.npz   (extended)
    run_dir/multiview/model_abc/d14_ngram_partition.npz
    run_dir/multiview/model_abc/figures/d14_ngram_partition.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

import torch

from config import ModelConfig, load_config_pair
from data import prepare_dataset, make_dataloaders
from model import LlamaStyleTransformer

from analyze import default_pilot_positions
from multiview_campaign import (
    seeds_in_run,
    checkpoints_in_seed,
    multiview_dir,
    augmented_path,
    load_token_sets,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PHASE_LABELS = {
    479:   "Phase I",
    2563:  "Phase II",
    9809:  "Phase III mid",
    24000: "Phase III final",
}


# ----------------------------------------------------------------------
# Paths.
# ----------------------------------------------------------------------
def ngram_augmented_path(run_dir: str, seed: int, step: int) -> str:
    return augmented_path(run_dir, seed, step).replace(
        ".npz", "_ngram.npz")


def output_root(run_dir: str) -> str:
    return os.path.join(run_dir, "multiview", "model_abc")


def figures_dir(run_dir: str) -> str:
    return os.path.join(output_root(run_dir), "figures")


# ----------------------------------------------------------------------
# Stage A: collect states, input, next, pred, position, *and* prev.
# Uses the fixed default pilot positions (no randomization), since the
# (prev, next) joint identifiability question is per-pilot.
# ----------------------------------------------------------------------
def collect_with_prev(
    model, eval_loader, pilot_positions: List[int],
    device: str, autocast_dtype, max_pilots: int,
) -> dict:
    model.eval()
    num_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    n_layer_outputs = num_layers + 2

    state_buffers: List[List[np.ndarray]] = [[] for _ in range(n_layer_outputs)]
    input_id_chunks, next_id_chunks, prev_id_chunks = [], [], []
    pred_id_chunks, pos_chunks = [], []
    collected = 0

    with torch.no_grad():
        for batch in eval_loader:
            if collected >= max_pilots:
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            B, T = input_ids.shape
            # Valid: need p-1 >= 0 (for prev) and p+1 < T (for next).
            valid_pos = [p for p in pilot_positions
                         if 1 <= p and p + 1 < T]
            if not valid_pos:
                continue
            pos_idx = torch.tensor(valid_pos, device=device, dtype=torch.long)

            with torch.amp.autocast("cuda", dtype=autocast_dtype,
                                     enabled=(device == "cuda")):
                logits, _, hidden = model(input_ids, return_hidden_states=True)

            for layer_idx, h in enumerate(hidden):
                picked = h[:, pos_idx, :].float().cpu().numpy()
                state_buffers[layer_idx].append(picked.reshape(-1, H))

            input_tag = input_ids[:, pos_idx].cpu().numpy().astype(np.int32)
            next_tag  = input_ids[:, pos_idx + 1].cpu().numpy().astype(np.int32)
            prev_tag  = input_ids[:, pos_idx - 1].cpu().numpy().astype(np.int32)
            pred_tag  = logits[:, pos_idx, :].argmax(dim=-1) \
                            .cpu().numpy().astype(np.int32)

            input_id_chunks.append(input_tag.reshape(-1))
            next_id_chunks.append(next_tag.reshape(-1))
            prev_id_chunks.append(prev_tag.reshape(-1))
            pred_id_chunks.append(pred_tag.reshape(-1))
            pos_chunks.append(
                np.tile(np.array(valid_pos, dtype=np.int32), (B,)))
            collected += B * len(valid_pos)

    N = collected
    states = np.zeros((n_layer_outputs, N, H), dtype=np.float32)
    for layer_idx in range(n_layer_outputs):
        if state_buffers[layer_idx]:
            cat = np.concatenate(state_buffers[layer_idx], axis=0)
            states[layer_idx] = cat[:N]
    return {
        "states":     states,
        "input_ids":  np.concatenate(input_id_chunks)[:N] if input_id_chunks
                       else np.zeros((0,), dtype=np.int32),
        "next_ids":   np.concatenate(next_id_chunks)[:N] if next_id_chunks
                       else np.zeros((0,), dtype=np.int32),
        "prev_ids":   np.concatenate(prev_id_chunks)[:N] if prev_id_chunks
                       else np.zeros((0,), dtype=np.int32),
        "pred_ids":   np.concatenate(pred_id_chunks)[:N] if pred_id_chunks
                       else np.zeros((0,), dtype=np.int32),
        "positions":  np.concatenate(pos_chunks)[:N] if pos_chunks
                       else np.zeros((0,), dtype=np.int32),
    }


def save_ngram_payload(payload: dict, path: str) -> None:
    np.savez_compressed(
        path,
        states=payload["states"],
        input_ids=payload["input_ids"],
        next_ids=payload["next_ids"],
        prev_ids=payload["prev_ids"],
        pred_ids=payload["pred_ids"],
        positions=payload["positions"],
    )


def load_ngram_payload(path: str) -> dict:
    with np.load(path) as f:
        return {
            "states":    f["states"],
            "input_ids": f["input_ids"],
            "next_ids":  f["next_ids"],
            "prev_ids":  f["prev_ids"],
            "pred_ids":  f["pred_ids"],
            "positions": f["positions"],
        }


# ----------------------------------------------------------------------
# Sub-conditioning analysis with null control.
# ----------------------------------------------------------------------
def _per_layer_kurtosis(states_sub, basis, means):
    L, n, H = states_sub.shape
    out = np.full(L, np.nan, dtype=np.float64)
    if n < 5:
        return out
    for t in range(L):
        X = states_sub[t].astype(np.float64) - means[t]
        Z = X @ basis[t].T
        var = Z.var(axis=0)
        if not np.all(var > 0):
            continue
        m4 = ((Z - Z.mean(axis=0)) ** 4).mean(axis=0)
        out[t] = float(np.mean(m4 / (var ** 2) - 3.0))
    return out


def _global_basis(states, d):
    L, N, H = states.shape
    d_eff = min(d, H, N - 1)
    basis = np.zeros((L, d_eff, H), dtype=np.float64)
    means = np.zeros((L, H), dtype=np.float64)
    for t in range(L):
        X = states[t].astype(np.float64)
        means[t] = X.mean(0)
        Xc = X - means[t]
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        basis[t] = Vt[:d_eff]
    return basis, means


def _aggregate_kurt(states_v, group_ids, basis, means, min_subbundle):
    L = states_v.shape[0]
    unique, counts = np.unique(group_ids, return_counts=True)
    keep = unique[counts >= min_subbundle]
    if keep.size == 0:
        return np.full(L, np.nan), 0
    weights, kurts = [], []
    for u in keep:
        mask = (group_ids == u)
        sub = states_v[:, mask, :]
        k = _per_layer_kurtosis(sub, basis, means)
        kurts.append(k)
        weights.append(int(mask.sum()))
    K = np.stack(kurts)
    w = np.array(weights, dtype=np.float64); w /= w.sum()
    out = np.full(L, np.nan)
    for t in range(L):
        col = K[:, t]
        v = np.isfinite(col)
        if v.sum():
            out[t] = float(np.average(col[v], weights=w[v]))
    return out, int(keep.size)


def analyze_token(
    states, input_ids, next_ids, prev_ids,
    basis, means, token_id, min_subbundle,
    n_random_reps, rng,
):
    mask = (input_ids == int(token_id))
    n_total = int(mask.sum())
    if n_total < min_subbundle * 2:
        return {"insufficient": True, "n_total": n_total}
    states_v = states[:, mask, :]
    next_v = next_ids[mask]
    prev_v = prev_ids[mask]
    L = states_v.shape[0]

    baseline = _per_layer_kurtosis(states_v, basis, means)
    next_kurt, n_next = _aggregate_kurt(
        states_v, next_v, basis, means, min_subbundle)
    prev_kurt, n_prev = _aggregate_kurt(
        states_v, prev_v, basis, means, min_subbundle)

    # Joint (prev, next): encode as pair_id = prev * (max_next+1) + next.
    # We use a deterministic encoding into a single integer key.
    M = int(max(prev_v.max(), next_v.max())) + 1
    joint_ids = prev_v.astype(np.int64) * M + next_v.astype(np.int64)
    joint_kurt, n_joint = _aggregate_kurt(
        states_v, joint_ids, basis, means, min_subbundle)

    # Matched random nulls at each variable's effective bin count.
    def _null(k_bins, n_reps):
        if k_bins < 2:
            return np.full(L, np.nan)
        reps = []
        for _ in range(n_reps):
            rl = rng.integers(0, k_bins, size=n_total, dtype=np.int32)
            kr, _ = _aggregate_kurt(
                states_v, rl, basis, means, min_subbundle)
            reps.append(kr)
        return np.nanmean(np.stack(reps), axis=0)

    null_next  = _null(n_next,  n_random_reps)
    null_prev  = _null(n_prev,  n_random_reps)
    null_joint = _null(n_joint, n_random_reps)

    return {
        "insufficient": False,
        "token_id": int(token_id),
        "n_total": n_total,
        "baseline": baseline,
        "next_kurt": next_kurt, "n_next": n_next,
        "prev_kurt": prev_kurt, "n_prev": n_prev,
        "joint_kurt": joint_kurt, "n_joint": n_joint,
        "null_next": null_next,
        "null_prev": null_prev,
        "null_joint": null_joint,
    }


def analyze_checkpoint_data(
    aug: dict, tids: np.ndarray, min_subbundle: int,
    max_pc_dim: int, n_random_reps: int, rng,
):
    states = aug["states"]
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    prev_ids = aug["prev_ids"]
    L, N, H = states.shape
    print(f"    loaded ({L=}, {N=}, {H=})")
    print(f"    computing top-{max_pc_dim} PC basis ...")
    basis, means = _global_basis(states, max_pc_dim)

    results = {}
    for tok in tids:
        r = analyze_token(
            states, input_ids, next_ids, prev_ids, basis, means,
            int(tok), min_subbundle, n_random_reps, rng,
        )
        results[int(tok)] = r

    valid = [r for r in results.values() if not r.get("insufficient", True)]
    print(f"    {len(valid)}/{tids.size} tokens with sufficient samples")
    return results


def aggregate(results: Dict) -> Dict:
    valid = [r for r in results.values() if not r.get("insufficient", True)]
    if not valid:
        return {}
    L = valid[0]["baseline"].size
    w = np.array([r["n_total"] for r in valid], dtype=np.float64)
    w /= w.sum()

    def _wm(key):
        arr = np.stack([r[key] for r in valid])
        out = np.full(L, np.nan)
        for t in range(L):
            col = arr[:, t]; v = np.isfinite(col)
            if v.sum():
                out[t] = float(np.average(col[v], weights=w[v]))
        return out

    baseline   = _wm("baseline")
    next_kurt  = _wm("next_kurt")
    prev_kurt  = _wm("prev_kurt")
    joint_kurt = _wm("joint_kurt")
    null_next  = _wm("null_next")
    null_prev  = _wm("null_prev")
    null_joint = _wm("null_joint")
    return {
        "baseline":   baseline,
        "next_kurt":  next_kurt,
        "prev_kurt":  prev_kurt,
        "joint_kurt": joint_kurt,
        "null_next":  null_next,
        "null_prev":  null_prev,
        "null_joint": null_joint,
        "real_next":  null_next  - next_kurt,
        "real_prev":  null_prev  - prev_kurt,
        "real_joint": null_joint - joint_kurt,
        "n_next_mean":  float(np.mean([r["n_next"]  for r in valid])),
        "n_prev_mean":  float(np.mean([r["n_prev"]  for r in valid])),
        "n_joint_mean": float(np.mean([r["n_joint"] for r in valid])),
        "n_valid_tokens": len(valid),
    }


# ----------------------------------------------------------------------
# Plot.
# ----------------------------------------------------------------------
def plot_results(run_dir, aggregates, seed, layer):
    steps = sorted(aggregates.keys())
    if not steps:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(steps)))

    # Panel 1: bar chart of real signals at layer_of_interest.
    ax = axes[0]
    width = 0.27
    x = np.arange(len(steps))
    real_next  = [aggregates[s]["real_next"][layer]  for s in steps]
    real_prev  = [aggregates[s]["real_prev"][layer]  for s in steps]
    real_joint = [aggregates[s]["real_joint"][layer] for s in steps]
    ax.bar(x - width, real_next,  width, label="real signal: next", color="C0")
    ax.bar(x,         real_prev,  width, label="real signal: prev", color="C1")
    ax.bar(x + width, real_joint, width, label="real signal: (prev, next)",
           color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS.get(s, str(s)) for s in steps],
                       rotation=15, fontsize=8)
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_ylabel(f"real kurtosis signal at layer {layer}")
    ax.set_title("Real signal (null-corrected) by partition")
    ax.legend(fontsize=8, loc="best")

    # Panel 2: per-layer profiles at final checkpoint.
    ax = axes[1]
    final_step = steps[-1]
    prof = aggregates[final_step]
    if prof:
        L = prof["baseline"].size
        layers = np.arange(L)
        ax.plot(layers, prof["baseline"], "k-", lw=2.5, label="baseline")
        ax.plot(layers, prof["next_kurt"],  "--", color="C0", lw=1.5,
                label="| next")
        ax.plot(layers, prof["prev_kurt"],  "--", color="C1", lw=1.5,
                label="| prev")
        ax.plot(layers, prof["joint_kurt"], "--", color="C2", lw=1.5,
                label="| (prev, next)")
        ax.plot(layers, prof["null_joint"], ":", color="gray", lw=1.5,
                label="random null at joint k")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("excess kurtosis")
    ax.set_title(
        f"Per-layer profile at {PHASE_LABELS.get(final_step, str(final_step))}")
    ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"D14: n-gram partition (Possibility 3), seed {seed}",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    out = os.path.join(figures_dir(run_dir), "d14_ngram_partition.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"[plot] -> {out}")


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------
DEFAULT_STEPS = [479, 2563, 9809, 24000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True,
                    help="Phase 1 JSON config (e.g. seed_0/run_metadata.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, nargs="*", default=DEFAULT_STEPS)
    ap.add_argument("--max-pilots", type=int, default=100_000)
    ap.add_argument("--min-subbundle", type=int, default=10)
    ap.add_argument("--max-pc-dim", type=int, default=32)
    ap.add_argument("--n-random-reps", type=int, default=20)
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-existing-aug", action="store_true", default=True,
                    help="Reuse existing _ngram.npz if present.")
    ap.add_argument("--rng-seed", type=int, default=20260521)
    args = ap.parse_args()

    os.makedirs(figures_dir(args.run_dir), exist_ok=True)
    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)

    # ---- Stage A regen (with prev_token) ----
    # Map steps to nearest available.
    avail = dict(checkpoints_in_seed(args.run_dir, args.seed))
    actual_steps = []
    for s in args.steps:
        if s in avail:
            actual_steps.append((s, avail[s]))
        else:
            nearest = min(avail.keys(),
                          key=lambda k: abs(np.log(max(k, 1)) -
                                            np.log(max(s, 1))))
            print(f"[warn] requested step {s} not present, "
                  f"using nearest {nearest}")
            actual_steps.append((nearest, avail[nearest]))

    needs_inference = []
    for step, ckpt_path in actual_steps:
        out_path = ngram_augmented_path(args.run_dir, args.seed, step)
        if args.skip_existing_aug and os.path.exists(out_path):
            print(f"[A] step {step}: ngram file exists at {out_path}, "
                  f"reusing")
        else:
            needs_inference.append((step, ckpt_path, out_path))

    if needs_inference:
        print(f"\n[A] Will run inference for {len(needs_inference)} steps")
        model_cfg, train_cfg = load_config_pair(args.config)
        _, held_out = prepare_dataset(model_cfg, train_cfg)
        _, eval_loader = make_dataloaders(held_out, held_out, train_cfg)

        if args.device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
            autocast_dtype = torch.bfloat16
        elif args.device == "cuda":
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.float32

        pilot_positions = default_pilot_positions(
            seq_len=model_cfg.max_position_embeddings)
        # Require p >= 1 (for prev) and p+1 < T-1 (for next).
        pilot_positions = [p for p in pilot_positions
                           if 1 <= p and p + 1 < model_cfg.max_position_embeddings - 1]

        for step, ckpt_path, out_path in needs_inference:
            t0 = time.time()
            print(f"[A] step {step}: loading checkpoint ...")
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            model = LlamaStyleTransformer(model_cfg).to(args.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            payload = collect_with_prev(
                model, eval_loader, pilot_positions,
                args.device, autocast_dtype, args.max_pilots,
            )
            save_ngram_payload(payload, out_path)
            N = payload["states"].shape[1]
            print(f"[A] step {step}: saved {N:,} pilots in "
                  f"{time.time()-t0:.1f}s -> {out_path}")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # ---- Analysis ----
    print(f"\n[B] Analyzing {len(actual_steps)} checkpoints")
    rng = np.random.default_rng(args.rng_seed)
    aggregates = {}
    for step, _ in actual_steps:
        print(f"\n[{step}] {PHASE_LABELS.get(step, str(step))}:")
        aug_path = ngram_augmented_path(args.run_dir, args.seed, step)
        if not os.path.exists(aug_path):
            print(f"  [skip] missing {aug_path}")
            continue
        aug = load_ngram_payload(aug_path)
        results = analyze_checkpoint_data(
            aug, tids, args.min_subbundle, args.max_pc_dim,
            args.n_random_reps, rng,
        )
        aggregates[step] = aggregate(results)
        prof = aggregates[step]
        if prof:
            t = args.layer
            print(f"    layer {t} baseline = {prof['baseline'][t]:.3f}")
            print(f"    next:   reduction={prof['baseline'][t] - prof['next_kurt'][t]:.3f}  "
                  f"null={prof['baseline'][t] - prof['null_next'][t]:.3f}  "
                  f"real={prof['real_next'][t]:.3f}  "
                  f"n_sub mean={prof['n_next_mean']:.1f}")
            print(f"    prev:   reduction={prof['baseline'][t] - prof['prev_kurt'][t]:.3f}  "
                  f"null={prof['baseline'][t] - prof['null_prev'][t]:.3f}  "
                  f"real={prof['real_prev'][t]:.3f}  "
                  f"n_sub mean={prof['n_prev_mean']:.1f}")
            print(f"    joint:  reduction={prof['baseline'][t] - prof['joint_kurt'][t]:.3f}  "
                  f"null={prof['baseline'][t] - prof['null_joint'][t]:.3f}  "
                  f"real={prof['real_joint'][t]:.3f}  "
                  f"n_sub mean={prof['n_joint_mean']:.1f}")
            print(f"    n_valid_tokens = {prof['n_valid_tokens']}")

    # Save.
    out_arrays = {
        "steps": np.array(sorted(aggregates.keys()), dtype=np.int64),
        "layer": np.int32(args.layer),
        "seed": np.int32(args.seed),
    }
    for step, prof in aggregates.items():
        if not prof:
            continue
        for key in ("baseline", "next_kurt", "prev_kurt", "joint_kurt",
                    "null_next", "null_prev", "null_joint",
                    "real_next", "real_prev", "real_joint"):
            out_arrays[f"step_{step}_{key}"] = prof[key]
    out_path = os.path.join(output_root(args.run_dir),
                            "d14_ngram_partition.npz")
    np.savez(out_path, **out_arrays)
    print(f"\n[npz] -> {out_path}")

    plot_results(args.run_dir, aggregates,
                 seed=args.seed, layer=args.layer)


if __name__ == "__main__":
    main()
