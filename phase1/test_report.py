"""
Test the plotting and reporting pipeline end-to-end with synthetic data.

Builds a fake "run directory" with 20 synthetic flow .npz files that
mimic what analyze_run() would produce, then runs report.py on it.

This catches bugs that would otherwise only surface when running on
real data:
  - file format mismatches between save_flow and load_flow
  - shape mismatches between flow fields and what the plots expect
  - colormap, legend, axis-scale issues that don't fail on synthetic data
    but would fail on degenerate edge cases

Run:  python3 test_report.py
"""

import os
import sys
import tempfile
import shutil
import json
import subprocess

import numpy as np

from analyze import save_flow
from test_analyze import make_pure_linear_trajectories


def build_fake_run(run_dir: str, num_checkpoints: int = 20, L: int = 14, H: int = 32):
    """Build a fake run directory with synthetic flow files."""
    flow_dir = os.path.join(run_dir, "flow_analysis")
    os.makedirs(flow_dir, exist_ok=True)

    # Need a run_metadata.json too (load_flow_series doesn't read it, but
    # report.py's load_flow_series doesn't require it — only its callers do).
    # We'll skip that file for this test since load_flow_series doesn't use it.

    steps = np.geomspace(10, 1000, num_checkpoints).astype(int)
    # Force monotonic increasing.
    steps = np.unique(steps)
    num_checkpoints = len(steps)

    # Simulate loss decreasing across training.
    eval_losses = 10.0 - 7.0 * (np.log(steps) - np.log(steps[0])) / (
        np.log(steps[-1]) - np.log(steps[0])
    )
    train_losses = eval_losses + np.random.default_rng(0).normal(0, 0.05, len(eval_losses))

    # Generate flows: linear flow recovered from synthetic activations.
    # We make `signal_noise_ratio` increase across training to simulate
    # the model getting better-organized over time.
    from analyze import recover_linear_flow

    rng_master = np.random.default_rng(42)
    for k, step in enumerate(steps):
        # Different signal-to-noise per checkpoint to simulate training progress.
        snr = 5.0 + 20.0 * k / num_checkpoints
        activations, _, _ = make_pure_linear_trajectories(
            L=L, N=500, H=H,
            lambda_true=0.2, log_alpha_true=-1.0,
            signal_noise_ratio=snr, rotation_angle_per_layer=0.05,
            rng_seed=int(rng_master.integers(0, 1000000)),
        )
        flow = recover_linear_flow(activations, center=True)
        flow["checkpoint_step"] = int(step)
        flow["checkpoint_path"] = f"/fake/step_{step:08d}.pt"
        flow["checkpoint_loss"] = float(train_losses[k])
        flow["checkpoint_eval_loss"] = float(eval_losses[k])
        flow["checkpoint_seed"] = 0
        flow["pilot_positions"] = [50, 100, 150]
        flow["analysis_time_sec"] = 1.0
        output_path = os.path.join(flow_dir, f"flow_step_{step:08d}.npz")
        save_flow(flow, output_path)

    return num_checkpoints


def main():
    print("Building fake run directory ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "fake_run")
        K = build_fake_run(run_dir, num_checkpoints=20, L=14, H=32)
        print(f"  ↳ Created {K} synthetic flow files in {run_dir}/flow_analysis/")
        print()

        print("Running report.py ...")
        result = subprocess.run(
            [sys.executable, "report.py", "--run_dir", run_dir],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.returncode != 0:
            print(f"❌ report.py exited with code {result.returncode}")
            sys.exit(1)

        # Verify all 8 PNGs were created.
        plots_dir = os.path.join(run_dir, "plots")
        expected = [
            "01_loss_curves.png",
            "02_basis_invariant_trajectories.png",
            "03_effective_rank_depth_profile.png",
            "04_flow_convergence.png",
            "05_variance_scaling_fit.png",
            "06_singular_value_spectra.png",
            "07_successive_layer_angles.png",
            "08_pairwise_residual_heatmap.png",
        ]
        missing = []
        for filename in expected:
            path = os.path.join(plots_dir, filename)
            if not os.path.exists(path):
                missing.append(filename)
                continue
            # Quick sanity: file should be non-trivial size (> 5 KB).
            size = os.path.getsize(path)
            if size < 5_000:
                print(f"⚠️  {filename} is suspiciously small ({size} bytes)")
            else:
                print(f"✅ {filename} ({size:,} bytes)")

        if missing:
            print(f"❌ Missing plots: {missing}")
            sys.exit(1)

        print()
        print(f"✅ All {len(expected)} plots generated successfully.")

        # Optional: keep one set of plots for inspection
        keep_dir = "/tmp/phase1_test_plots"
        if os.path.exists(keep_dir):
            shutil.rmtree(keep_dir)
        shutil.copytree(plots_dir, keep_dir)
        print(f"   (Sample plots preserved in {keep_dir} for inspection.)")


if __name__ == "__main__":
    main()
