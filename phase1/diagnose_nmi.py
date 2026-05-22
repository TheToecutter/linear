"""
Diagnose the source of high NMI(next; position) in randomized stage A.

Test hypothesis: high NMI is a bucket-sparsity artifact. With 1023
unique positions and ~19,500 pilots, each (token, position) cell has
roughly one sample, so position trivially "determines" next-token.

If true: binning positions into a small number of buckets should reduce
NMI substantially. If false (position truly determines next-token even
under aggregation): NMI stays high.

Usage:
    python diagnose_nmi.py --run-dir ../phase1_runs_gelu
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import numpy as np

from multiview import load_augmented_activations
from multiview_campaign import (
    augmented_path,
    load_token_sets,
)


def nmi_next_position(next_ids: np.ndarray,
                      positions: np.ndarray) -> float:
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


def quantile_bin(positions: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin positions into n_bins quantile buckets."""
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(positions, quantiles)
    # Avoid duplicate edges (can happen with discrete data).
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros_like(positions, dtype=np.int32)
    binned = np.searchsorted(edges, positions, side="right") - 1
    return np.clip(binned, 0, len(edges) - 2).astype(np.int32)


def report(run_dir: str, seed: int, step: int, label: str,
           random: bool) -> Dict:
    forward_set, _, _ = load_token_sets(run_dir)
    tids = forward_set.token_ids.astype(np.int32)
    path = augmented_path(run_dir, seed, step)
    if random:
        path = path.replace(".npz", "_random.npz")
    if not os.path.exists(path):
        print(f"\n{label}: missing {path}")
        return {}
    aug = load_augmented_activations(path)
    input_ids = aug["input_ids"]
    next_ids = aug["next_ids"]
    positions = aug["positions"]
    print(f"\n{label}: {path}")
    print(f"  N={input_ids.size}, "
          f"unique positions={np.unique(positions).size}, "
          f"position range=[{positions.min()}, {positions.max()}]")

    # Sweep bin counts to see how NMI changes with position coarsening.
    bin_counts = [None, 100, 50, 20, 10, 5]
    results = {}
    print(f"  Mean NMI across {tids.size} forward-set tokens, varying "
          f"position binning:")
    print(f"  {'bins':>8s}  {'mean NMI':>10s}  {'median NMI':>11s}  "
          f"{'max NMI':>9s}  {'samples/bucket':>16s}")
    for bins in bin_counts:
        if bins is None:
            pos_eff = positions
            label_b = "raw"
        else:
            pos_eff = quantile_bin(positions, bins)
            label_b = str(bins)
        # Estimate samples per (token, bucket) cell across forward-set
        # tokens; report the median.
        cells_per_token = []
        nmis = []
        for tok in tids:
            mask = (input_ids == int(tok))
            if mask.sum() < 20:
                continue
            n_buckets_present = np.unique(pos_eff[mask]).size
            samples_per_bucket = mask.sum() / max(n_buckets_present, 1)
            cells_per_token.append(samples_per_bucket)
            nmi = nmi_next_position(next_ids[mask], pos_eff[mask])
            nmis.append(nmi)
        if nmis:
            print(f"  {label_b:>8s}  {np.mean(nmis):>10.3f}  "
                  f"{np.median(nmis):>11.3f}  {np.max(nmis):>9.3f}  "
                  f"{np.median(cells_per_token):>16.1f}")
            results[label_b] = {
                "mean_nmi": float(np.mean(nmis)),
                "median_nmi": float(np.median(nmis)),
                "median_samples_per_bucket": float(np.median(cells_per_token)),
            }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=24000)
    args = ap.parse_args()

    print(f"Diagnosing NMI(next; position) bucket sparsity")
    print(f"=" * 60)
    report(args.run_dir, args.seed, args.step,
           "ORIGINAL (fixed positions)", random=False)
    report(args.run_dir, args.seed, args.step,
           "RANDOMIZED (per-sequence random positions)", random=True)

    print(f"\nReading:")
    print(f"  If randomized NMI drops sharply with binning while original")
    print(f"  does not, the high randomized NMI is a bucket-sparsity artifact")
    print(f"  and binning recovers usable identifiability.")
    print(f"  If binning does not reduce randomized NMI to near-zero, the")
    print(f"  high NMI is not just sparsity -- position genuinely identifies")
    print(f"  context even under aggregation.")


if __name__ == "__main__":
    main()
