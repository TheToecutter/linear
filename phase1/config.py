"""
Configuration objects for the Phase 1 pilot study.

Two configs are exported:

  - ModelConfig: hyperparameters of the 150M Llama-style transformer.
    Sized to ~146M parameters at H=896, L=12, I=2432 (heads=14, head_dim=64).

  - TrainingConfig: hyperparameters of the training loop — batch sizes,
    optimizer, learning rate schedule, checkpoint cadence, and the
    Phase-1-specific dense checkpoint schedule.

Why dataclasses (rather than a single class with __init__ defaults like the
older microdisllm setup)? Two reasons:
  - Multiple configs need to coexist clearly in the same process — model vs
    training vs analysis — and dataclasses make the boundaries explicit.
  - Serialization for run reproducibility: dataclasses serialize to JSON
    trivially, which is what we want for per-checkpoint metadata.

All values here are project-wide defaults. Per-run overrides go through the
TrainingConfig constructor or via run_phase1.py.
"""

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
import math


# ----------------------------------------------------------------------
# Model config: the 150M Llama-style architecture.
# ----------------------------------------------------------------------
@dataclass
class ModelConfig:
    """
    Llama-style decoder-only transformer hyperparameters.

    Architecture: pre-RMSNorm, RoPE, SwiGLU FFN, full causal self-attention.
    This is Phase 1's reference variant ("Variant A" in the proposal).
    Variants B, C, D in Phase 2 will inherit from this and modify specific
    components.

    Parameter count at defaults:
      embedding (tied): V × H = 32768 × 896 = 29.36M
      per-layer attention: 4 × H² = 4 × 896² = 3.21M
      per-layer FFN (SwiGLU): 3 × H × I = 3 × 896 × 2432 = 6.54M
      per-layer total: 9.75M
      12 layers: 117.0M
      grand total: ~146.4M
    """

    # Vocabulary. Using the Mistral-7B-v0.1 tokenizer (32000 base + special
    # tokens rounded up to 32768). Identical to Llama 2's SentencePiece BPE
    # without the gated-access requirement.
    vocab_size: int = 32768

    # Residual stream / hidden dimension. 896 = 128 × 7, divides evenly by
    # head_dim=64 → 14 attention heads. Targeting ~150M params total.
    hidden_size: int = 896

    # FFN inner dimension. SwiGLU param-count parity rule wants I ≈ 8/3 × H
    # ≈ 2389. Rounded to next clean multiple of 64 for vectorization: 2432.
    intermediate_size: int = 2432

    # Transformer block count. 12 is a conventional choice at this scale
    # (matches Pythia-160M, GPT-2 small).
    num_hidden_layers: int = 12

    # Attention heads. 14 heads × head_dim=64 = 896, matches hidden_size.
    # No grouped-query for the reference variant (full multi-head attention).
    num_attention_heads: int = 14

    # Sequence length. RoPE cache size; sequences past this fail.
    max_position_embeddings: int = 2048

    # RMSNorm epsilon (Llama 2 default).
    rms_norm_eps: float = 1e-6

    # RoPE base frequency. 10000 is the Llama 2 / Mistral default; fine at
    # 2048 context.
    rope_theta: float = 10000.0

    # Activation checkpointing: recompute per-block activations during
    # backward to save ~50% activation memory. Recommended at this scale
    # on a single 5090 (32 GB VRAM) with reasonable batch sizes.
    gradient_checkpointing: bool = True

    # Tie input embeddings to output projection. Saves V×H params and is
    # standard for Llama-family models.
    tie_embeddings: bool = True

    # Architecture variant. The Phase 1 reference is "llama". Phase 2 will
    # add "qwen" (QK-Norm), "gemma" (hybrid norm + GeGLU + sliding attn),
    # "deepseek" (MLA). All variants share this same config dataclass — the
    # architecture string dispatches to the right class via build_model().
    # When loading older run_metadata.json files that don't have this field,
    # we default to "llama" for backwards compatibility.
    architecture: str = "llama"

    # ---------- Gemma-specific fields (ignored by other architectures) ----------
    # Final logit softcap: logits = softcap * tanh(logits / softcap). Bounds
    # output logits to ±softcap. Default 30 matches Gemma-2.
    final_logit_softcap: float = 30.0
    # Attention logit softcap: applied to raw Q·K dot products before softmax,
    # within each attention head. Default 50 matches Gemma-2.
    attn_logit_softcap: float = 50.0
    # Sliding-window size for the sliding-window attention layers (the even-
    # indexed layers in Gemma's alternating pattern). 4096 matches Gemma-2.
    # At seq_len < sliding_window, sliding is functionally equivalent to full
    # attention — see the Gemma model docstring for what this implies at our
    # pilot scale (seq_len=1024 < 4096, so sliding is inert).
    sliding_window: int = 4096

    # ---------- DeepSeek/MLA-specific fields (ignored by other architectures) ----------
    # KV latent dimension: the residual stream is down-projected to this
    # dimension before being up-projected to per-head K and V. DeepSeek-V3
    # uses 512 at H=7168 (~7%). At our H=896 we scale to 96 (~11%, giving
    # a bit more room since our model is much smaller and proportionally
    # over-compressing would constrain too much).
    mla_kv_latent_dim: int = 96
    # Q latent dimension: similar down/up scheme for queries. DeepSeek-V3
    # uses 1536 at H=7168 (~21%). At H=896 we use 192 (~21%).
    mla_q_latent_dim: int = 192

    @property
    def head_dim(self) -> int:
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size={self.hidden_size} must be divisible by "
            f"num_attention_heads={self.num_attention_heads}"
        )
        return self.hidden_size // self.num_attention_heads

    def estimate_param_count(self) -> int:
        """Approximate parameter count from the hyperparameters."""
        V, H, I, L = (
            self.vocab_size, self.hidden_size,
            self.intermediate_size, self.num_hidden_layers,
        )
        embed = V * H  # tied
        per_layer = (
            4 * H * H        # QKV projection (3 × H²) + output projection (H²)
            + 3 * H * I      # SwiGLU: gate (H×I) + up (H×I) + down (I×H)
        )
        norms = 2 * H * L + H  # 2 RMSNorms per layer + final norm
        return embed + L * per_layer + norms


# ----------------------------------------------------------------------
# Training config.
# ----------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """
    Training-loop hyperparameters for Phase 1.

    Phase 1 training target: ~1.5B tokens per run (about half Chinchilla
    for a 150M model). Multi-seed Phase 1 launches 4-6 of these in
    sequence. At ~8,000 tok/sec (rough RTX 5090 throughput for 150M
    training with flash attention and gradient checkpointing), each run
    is roughly 2-3 days of compute.

    Checkpoint schedule:
      Phase 1 needs dense checkpoints to characterize L(K) convergence.
      We use a log-spaced schedule with 50 checkpoints from step 100 to
      training end, giving high resolution in early training (where
      changes are fast) and adequate sampling in late training (where
      changes are slow).
    """

    # ----- data -----
    train_seq_len: int = 1024
    # Number of held-out chunks reserved at the END of the tokenized
    # corpus for evaluation. Deterministic, not random — gives a stable
    # eval set across runs.
    held_out_chunks: int = 500

    # ----- batch sizes -----
    # Micro-batch is per-GPU; we'll use gradient accumulation to reach
    # the effective batch size. On a single 5090 with 32 GB VRAM and
    # gradient checkpointing on, micro_batch_size=8 at seq_len=1024
    # should fit comfortably for the 150M model.
    micro_batch_size: int = 8
    grad_accum_steps: int = 8  # effective batch = 64 × seq_len = 65,536 tokens

    # Held-out evaluation batch size (no SIGReg, no grad needed; can be larger).
    eval_batch_size: int = 16

    # ----- optimizer -----
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # LR schedule: linear warmup then cosine decay to a floor.
    warmup_steps: int = 1000
    lr_floor_ratio: float = 0.1  # final LR = learning_rate × this

    # ----- training duration -----
    # Total training tokens. With micro_batch=8, grad_accum=8, seq_len=1024,
    # one optimizer step processes 65,536 tokens. 1.5B tokens = ~22,900 steps.
    # We round up to 24000 steps for clean log-spacing of checkpoints.
    total_steps: int = 24000

    # ----- evaluation cadence -----
    # Run held-out eval every N optimizer steps.
    eval_every: int = 500

    # ----- checkpoint schedule -----
    # Phase 1 wants 50 log-spaced checkpoints. The schedule is computed
    # by checkpoint_schedule() below from total_steps and num_checkpoints.
    num_checkpoints: int = 50
    # Don't save before this step (very early checkpoints have nothing useful).
    first_checkpoint_step: int = 100

    # ----- logging -----
    log_every: int = 50  # console log every N optimizer steps
    csv_log_path: str = "training_log.csv"

    # ----- output -----
    checkpoint_dir: str = "checkpoints"
    # Save run metadata (model config, training config, git commit if any)
    # to a JSON file in checkpoint_dir.
    metadata_filename: str = "run_metadata.json"

    # ----- reproducibility -----
    seed: int = 0
    deterministic: bool = False  # set True for full bit-reproducibility (slower)

    def checkpoint_schedule(self) -> List[int]:
        """
        Generate the log-spaced checkpoint step schedule.

        Returns a sorted list of step indices at which to save checkpoints.
        Log-spaced from first_checkpoint_step to total_steps. Always
        includes total_steps as the last checkpoint.

        Example with num_checkpoints=50, first_checkpoint_step=100,
        total_steps=24000:
          [100, 113, 127, ..., 21300, 24000]
        """
        if self.num_checkpoints < 2:
            return [self.total_steps]
        # Log-space, then deduplicate (early steps may collide after rounding).
        log_start = math.log(self.first_checkpoint_step)
        log_end = math.log(self.total_steps)
        raw = [
            int(round(math.exp(log_start + (log_end - log_start) * i / (self.num_checkpoints - 1))))
            for i in range(self.num_checkpoints)
        ]
        # Dedup while preserving order, then ensure final step is present.
        seen = set()
        unique = []
        for step in raw:
            if step not in seen:
                seen.add(step)
                unique.append(step)
        if unique[-1] != self.total_steps:
            unique.append(self.total_steps)
        return unique


def save_config_pair(model_cfg: ModelConfig, train_cfg: TrainingConfig, path: str):
    """Serialize both configs to a JSON file. Used for run metadata."""
    payload = {
        "model": asdict(model_cfg),
        "training": asdict(train_cfg),
        # Computed quantities that aren't in the dataclass but are useful for
        # offline inspection.
        "model_param_estimate": model_cfg.estimate_param_count(),
        "model_head_dim": model_cfg.head_dim,
        "training_checkpoint_schedule": train_cfg.checkpoint_schedule(),
        "training_tokens_per_step": (
            train_cfg.micro_batch_size * train_cfg.grad_accum_steps
            * train_cfg.train_seq_len
        ),
        "training_total_tokens": (
            train_cfg.micro_batch_size * train_cfg.grad_accum_steps
            * train_cfg.train_seq_len * train_cfg.total_steps
        ),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_config_pair(path: str):
    """Deserialize a JSON file into a (ModelConfig, TrainingConfig) pair.

    Forward-compatible: silently ignores any keys in the JSON that aren't
    fields of the current ModelConfig/TrainingConfig. This means new
    fields added to the configs don't break loading of older saved runs,
    and runs saved with newer configs can be loaded by older code (minus
    the new info)."""
    with open(path) as f:
        payload = json.load(f)
    model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
    train_fields = {f.name for f in dataclasses.fields(TrainingConfig)}
    model_kwargs = {k: v for k, v in payload["model"].items() if k in model_fields}
    train_kwargs = {k: v for k, v in payload["training"].items() if k in train_fields}
    return (
        ModelConfig(**model_kwargs),
        TrainingConfig(**train_kwargs),
    )
    