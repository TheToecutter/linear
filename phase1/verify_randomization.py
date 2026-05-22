"""
Sanity check: verify that the randomized augmented files actually
achieved decorrelation of next-token from position.

Loads the new randomized augmented file at the final checkpoint
(seed 0, step 24000) and reports NMI(next; position) per token in the
forward set, comparing to the original fixed-position file.

Pass criterion: NMI under the randomized scheme should be close to zero
(say < 0.05) for all tokens.

Usage:
    python verify_randomization.py --run-dir ../phase1_runs_gelu
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np

from multiview import load_augmented_activations
from multiview_campaign import (
    augmented_path,
    load_token_sets,
)


def randomized_augmented_path(run_dir: str, seed: int, step: int) -> str:
    return augmented_path(run_dir, seed, step).replace(
        ".npz", "_random.npz")


def nmi_next_position(next_ids: np.ndarray,
                      positions: np.ndarray) -> float:
    """Normalized mutual information NMI(next; position) = I/H(next)."""
    n = next_ids.size
    if n < 2:
        return float("nan")
    nu, ni = np.unique(next_ids, return_inverse=True)
    pu, pi = np.unique(positions, return_inverse=True)
    joint = np.zeros((nu.size, pu.size), dtype=np.float64)
    np.add.at(joint, (ni, pi), 1.0)
    joint /= n
    p_next = joint.sum(1)
    p_pos = joint.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(p_next, p_pos)
        ratio = np.where((joint > 0) & (denom > 0), joint / denom, 1.0)
        mi = float(np.sum(joint * np.log(np.where(ratio > 0, ratio, 1.0))))
    h_next = -float(np.sum(p_next[p_next > 0] * np.log(p_next[p_next > 0])))
    if h_next <= 0:
        return float("nan")
    return mi / h_next


def report(label: str, aug_path: str, tids: np.ndarray) -> None:
    if not os.path.exists(aug_path):
        print(f"\n{label}: missing {aug_path}")
        return
    aug = load_augmented_activations(aug_path)
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    positions = aug["positions"]
    print(f"\n{label}:  {aug_path}")
    print(f"  N total = {input_ids.size}")
    print(f"  unique positions = {np.unique(positions).size}")
    print(f"  position range = [{positions.min()}, {positions.max()}]")
    print(f"  NMI(next; position) per forward-set token:")
    nmis = []
    for tok in tids:
        mask = (input_ids == int(tok))
        if mask.sum() < 20:
            continue
        nmi = nmi_next_position(next_ids[mask], positions[mask])
        nmis.append(nmi)
        print(f"    tok {int(tok):>6d}  N={int(mask.sum()):>5d}  "
              f"NMI={nmi:.3f}")
    if nmis:
        print(f"  mean NMI across tokens: {np.mean(nmis):.3f}  "
              f"(median {np.median(nmis):.3f}, "
              f"max {np.max(nmis):.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=24000)
    args = ap.parse_args()

    forward_set, _, _ = load_token_sets(args.run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    print(f"Forward set: {tids.size} tokens")

    fixed_path = augmented_path(args.run_dir, args.seed, args.step)
    random_path = randomized_augmented_path(args.run_dir, args.seed, args.step)

    report("ORIGINAL (fixed positions)", fixed_path, tids)
    report("RANDOMIZED", random_path, tids)


if __name__ == "__main__":
    main()
