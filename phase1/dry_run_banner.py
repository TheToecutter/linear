"""
Print the startup banner without actually running training.

Use this to verify that everything wires up before launching real training:
  - GPU is visible and identified correctly
  - Model parameter count matches
  - Memory estimate prints sensibly
  - Checkpoint schedule is correct
  - All paths in the run_dir are creatable

Run:  python3 dry_run_banner.py

If this works, the next step is a real (short) smoke test:
    python3 train.py --seed 0 --run_dir smoke_test --total_steps 200
"""

import os
import torch

from config import ModelConfig, TrainingConfig
from train import _print_startup_banner


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Force a small allocation so we see real numbers in subsequent steps.
        _ = torch.zeros(1, device=device)
    autocast_dtype = (
        torch.bfloat16 if device == "cuda"
        and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
    )

    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    run_dir = "/tmp/phase1_dry_run"
    os.makedirs(run_dir, exist_ok=True)

    _print_startup_banner(
        device=device,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        seed=0,
        run_dir=run_dir,
        autocast_dtype=autocast_dtype,
    )
    print("✅ Dry run complete. Ready for real training when you are.")


if __name__ == "__main__":
    main()
