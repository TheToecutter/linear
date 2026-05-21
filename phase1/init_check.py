"""
Random-initialization baseline check for the multi-view decomposition.

Question: at training step 0 (random initialization, no training), does
the reverse-view mid-network bulge in the within/between ratio profile
already exist?

If yes, the bulge is architecturally intrinsic — a property of the
residual stream's geometry at random weights, not something learning
creates. Training intensifies it but doesn't summon it.

If no, the bulge is a learned phenomenon and our framing needs to
change accordingly.

This script:
  1. Instantiates a freshly-initialized LlamaStyleTransformer using
     the same model config as the trained runs, with a fresh seed
     (different from any of seeds 0-3).
  2. Runs the same activation-collection pipeline that Stage A uses,
     against the same held-out dataloader, with the same pilot positions.
  3. Selects token sets the same way Stage B does (top-20 by frequency).
  4. Computes the within/between decomposition for all three views.
  5. Plots the ratio profile and saves to figures/.

Idiomatic placement: project root, alongside multiview_campaign.py.

Usage:
    python3 init_check.py [--config PATH_TO_RUN_METADATA] [--seed N]

By default loads the config from seed_0/run_metadata.json and uses
seed 9999 for the model init.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config_pair, ModelConfig
from model import LlamaStyleTransformer
from data import prepare_dataset, make_dataloaders
from analyze import default_pilot_positions
from multiview import (
    collect_activations_with_metadata,
    select_token_set,
    within_between_decomposition,
)


def init_model(model_cfg: ModelConfig, seed: int, device: str = "cuda"):
    """Instantiate LlamaStyleTransformer with a fresh seed.

    The constructor calls _init_weights internally, which uses
    torch's RNG. We seed PyTorch globally before instantiation to
    get a deterministic, reproducible random initialization that's
    distinct from any of the training-time seeds used so far.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = LlamaStyleTransformer(model_cfg).to(device)
    model.eval()
    return model


def compute_ratio_profile(payload: dict) -> dict:
    """Run Stage B + within/between decomposition on a single payload.

    Returns a dict mapping view name to (within, between, ratio) arrays
    of shape (L_total,).
    """
    states = payload["states"]
    input_ids = payload["input_ids"]
    next_ids = payload["next_ids"]
    pred_ids = payload["pred_ids"]

    # Token-set selection: top-20 by frequency, matching Stage B defaults.
    fset = select_token_set(input_ids, view="forward", top_k=20, min_count=10)
    raset = select_token_set(next_ids, view="reverse_actual", top_k=20, min_count=10)
    rpset = select_token_set(pred_ids, view="reverse_pred", top_k=20, min_count=10)

    fwd = within_between_decomposition(states, input_ids, fset)
    ra = within_between_decomposition(states, next_ids, raset)
    rp = within_between_decomposition(states, pred_ids, rpset)

    def _ratio(d):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d.v_between > 0, d.v_within / d.v_between, np.nan)

    return {
        "forward": (fwd.v_within, fwd.v_between, _ratio(fwd), fset),
        "reverse_actual": (ra.v_within, ra.v_between, _ratio(ra), raset),
        "reverse_pred": (rp.v_within, rp.v_between, _ratio(rp), rpset),
    }


def load_trained_final_ratios(traj_path: str) -> Tuple[dict, np.ndarray]:
    """Load the cross-seed-mean ratio profile at the final checkpoint
    from the trained runs, for visual comparison."""
    with np.load(traj_path) as f:
        steps = f["steps"]
        w_fwd = f["v_within_forward"][:, -1, :]
        b_fwd = f["v_between_forward"][:, -1, :]
        w_ra = f["v_within_reverse_actual"][:, -1, :]
        b_ra = f["v_between_reverse_actual"][:, -1, :]
        w_rp = f["v_within_reverse_pred"][:, -1, :]
        b_rp = f["v_between_reverse_pred"][:, -1, :]

    def _mean_ratio(w, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(b > 0, w / b, np.nan)
        return np.nanmean(r, axis=0)

    return {
        "forward": _mean_ratio(w_fwd, b_fwd),
        "reverse_actual": _mean_ratio(w_ra, b_ra),
        "reverse_pred": _mean_ratio(w_rp, b_rp),
    }, int(steps[-1])


def make_plot(init_ratios: dict, trained_ratios: dict, trained_step: int,
              init_seed: int, output_path: str) -> None:
    """Plot the init-step ratio profile on top of the trained-step
    cross-seed-mean profile."""
    L = next(iter(init_ratios.values()))[2].size
    layers = np.arange(L)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    styles = {
        "forward": {"color": "C0", "label_init": "forward (init)",
                    "label_final": f"forward (trained, step {trained_step})"},
        "reverse_actual": {"color": "C3", "label_init": "reverse actual (init)",
                           "label_final": f"reverse actual (step {trained_step})"},
        "reverse_pred": {"color": "C2", "label_init": "reverse pred (init)",
                         "label_final": f"reverse pred (step {trained_step})"},
    }
    for view, info in styles.items():
        _, _, r_init, _ = init_ratios[view]
        r_final = trained_ratios[view]
        ax.plot(layers, r_init, ":", color=info["color"], lw=2.0,
                marker="o", markersize=4, alpha=0.85,
                label=info["label_init"])
        ax.plot(layers, r_final, "-", color=info["color"], lw=2.0,
                marker="s", markersize=4,
                label=info["label_final"])
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.text(L - 0.5, 1.0, "  within = between", color="gray",
            fontsize=8, va="center", ha="left")
    ax.set_xlabel("layer state index t")
    ax.set_ylabel("within / between variance ratio")
    ax.set_xticks(layers)
    ax.set_title(f"Random-init (seed {init_seed}, no training) vs trained "
                 f"(step {trained_step}, cross-seed mean)\n"
                 f"within/between ratio profile per view")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, ncol=2)
    ax.grid(True, ls=":", lw=0.4, alpha=0.5)
    ax.set_xlim(-0.3, L - 0.6)
    # y-limit: include both curves' max with headroom.
    y_max = max(
        max(np.nanmax(r) for _, _, r, _ in init_ratios.values()),
        max(np.nanmax(r) for r in trained_ratios.values()),
    ) * 1.10
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(init_ratios: dict, trained_ratios: dict, trained_step: int):
    """Print a compact table comparing init vs trained at key layers."""
    L = next(iter(init_ratios.values()))[2].size
    trained_ra = trained_ratios["reverse_actual"]
    peak_t = int(np.nanargmax(trained_ra))
    key_layers = sorted({0, 1, peak_t, L - 2, L - 1})

    # Part 1: ratio comparison.
    print()
    print(f"Ratio comparison (within / between):")
    print(f"{'view':<18} {'layer':>6} {'init':>10} {'trained':>10} {'init/trained':>14}")
    print("-" * 64)
    for view in ("forward", "reverse_actual", "reverse_pred"):
        for t in key_layers:
            r_init = init_ratios[view][2][t]
            r_final = trained_ratios[view][t]
            with np.errstate(divide="ignore", invalid="ignore"):
                fold = r_init / r_final if r_final > 0 else float("nan")
            print(f"{view:<18} {t:>6} {r_init:>10.3f} {r_final:>10.3f} {fold:>14.3f}")
        print()

    # Part 2: absolute within / between at init, to sanity-check that
    # the large ratios reflect real structure rather than dividing two
    # near-zero numbers.
    print(f"Absolute within and between variance at init (sanity check on ratios):")
    print(f"{'view':<18} {'layer':>6} {'within':>14} {'between':>14} {'ratio':>10}")
    print("-" * 68)
    for view in ("forward", "reverse_actual", "reverse_pred"):
        w_arr, b_arr, r_arr, _ = init_ratios[view]
        for t in key_layers:
            print(f"{view:<18} {t:>6} {w_arr[t]:>14.4e} {b_arr[t]:>14.4e} "
                  f"{r_arr[t]:>10.3f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../phase1_runs_gelu/seed_0/run_metadata.json",
                    help="Path to a run_metadata.json from any trained seed; "
                         "we use only its model config (not its weights).")
    ap.add_argument("--seed", type=int, default=9999,
                    help="Random seed for the fresh model initialization "
                         "(deliberately distinct from training seeds 0-3).")
    ap.add_argument("--trajectories", default="../phase1_runs_gelu/multiview/trajectories",
                    help="Trajectories directory for the trained baseline comparison.")
    ap.add_argument("--figures-dir", default="../phase1_runs_gelu/figures",
                    help="Where to write the comparison figure.")
    ap.add_argument("--max-pilots", type=int, default=10_000,
                    help="Pilot cap, matching Stage A's default.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading config from {args.config} ...")
    model_cfg, train_cfg = load_config_pair(args.config)

    print(f"Building held-out dataloader (uses cached tokenization if available) ...")
    _, held_out = prepare_dataset(model_cfg, train_cfg)
    _, eval_loader = make_dataloaders(held_out, held_out, train_cfg)

    print(f"Instantiating fresh model with init seed {args.seed} ...")
    model = init_model(model_cfg, args.seed, device=device)

    pilot_positions = default_pilot_positions(seq_len=model_cfg.max_position_embeddings)
    pilot_positions = [p for p in pilot_positions
                       if p + 1 < model_cfg.max_position_embeddings - 1]

    if device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8:
        autocast_dtype = torch.bfloat16
    elif device == "cuda":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.float32

    print(f"Collecting activations from random-init model ...")
    payload = collect_activations_with_metadata(
        model=model,
        eval_loader=eval_loader,
        pilot_positions=pilot_positions,
        device=device,
        autocast_dtype=autocast_dtype,
        max_pilots=args.max_pilots,
        compute_predictions=True,
    )
    N = payload["states"].shape[1]
    L = payload["states"].shape[0]
    print(f"   {N:,} pilots collected, {L} layer states")

    print(f"Computing within/between decomposition ...")
    init_ratios = compute_ratio_profile(payload)

    print(f"Loading trained-baseline ratios for comparison ...")
    decomp_path = os.path.join(args.trajectories, "decomposition.npz")
    if not os.path.exists(decomp_path):
        print(f"   WARNING: {decomp_path} not found; will plot init only.")
        trained_ratios = None
        trained_step = -1
    else:
        trained_ratios, trained_step = load_trained_final_ratios(decomp_path)

    if trained_ratios is not None:
        print_summary(init_ratios, trained_ratios, trained_step)

    # Save the init decomposition to disk so other plotters can reuse it
    # (e.g., the init-augmented heatmap) without re-running inference.
    init_save_dir = os.path.join(os.path.dirname(args.trajectories), "init_baseline")
    os.makedirs(init_save_dir, exist_ok=True)
    init_save_path = os.path.join(init_save_dir, f"init_seed{args.seed}.npz")
    np.savez(
        init_save_path,
        seed=np.int32(args.seed),
        v_within_forward=init_ratios["forward"][0],
        v_between_forward=init_ratios["forward"][1],
        v_within_reverse_actual=init_ratios["reverse_actual"][0],
        v_between_reverse_actual=init_ratios["reverse_actual"][1],
        v_within_reverse_pred=init_ratios["reverse_pred"][0],
        v_between_reverse_pred=init_ratios["reverse_pred"][1],
    )
    print(f"\nInit decomposition saved to {init_save_path}")

    os.makedirs(args.figures_dir, exist_ok=True)
    out_path = os.path.join(args.figures_dir, "fig9_init_vs_trained_ratio.png")
    if trained_ratios is not None:
        make_plot(init_ratios, trained_ratios, trained_step, args.seed, out_path)
    else:
        # Init-only fallback plot.
        fig, ax = plt.subplots(figsize=(10, 6))
        L = next(iter(init_ratios.values()))[2].size
        layers = np.arange(L)
        for view, color in [("forward", "C0"), ("reverse_actual", "C3"),
                            ("reverse_pred", "C2")]:
            ax.plot(layers, init_ratios[view][2], ":", marker="o",
                    color=color, label=f"{view} (init seed {args.seed})")
        ax.axhline(1.0, color="gray", ls=":", lw=1.0)
        ax.set_xlabel("layer state index t")
        ax.set_ylabel("within / between variance ratio")
        ax.legend(fontsize=9)
        ax.grid(True, ls=":", lw=0.4, alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

    print(f"\nDone. Figure written to {out_path}")


if __name__ == "__main__":
    main()
    