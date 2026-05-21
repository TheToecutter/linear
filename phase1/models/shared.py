"""
Shared building blocks for all 150M architecture variants.

Components here are architecture-agnostic: they're used identically by
Llama-style, Qwen-style, Gemma-style, and DeepSeek-style transformers.

What's here:
  - RMSNorm: used by all variants
  - RotaryEmbedding + apply_rope: used by all attention variants
  - SwiGLUMLP: used by Llama and Qwen
  - GeluMLP: plain ungated GELU FFN; the macro-cleanliness baseline
    for Phase 2 Llama variants
  - count_parameters, estimate_training_memory_gb: architecture-agnostic utilities

What's NOT here (lives in per-architecture files):
  - Attention modules (each variant has its own attention details:
      Llama: full multi-head, Qwen: + QK-norm, Gemma: + sliding window,
      DeepSeek: MLA)
  - MLP variants other than SwiGLU (Gemma uses GeGLU)
  - The transformer block structure (norm placement differs: Llama pre-only,
      Gemma hybrid pre+post)
  - The full transformer class (each variant has architecture-specific init
      and forward logic; they share signatures but not implementation)
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# RMSNorm (used by all variants).
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).

        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Faster than LayerNorm (no mean centering), and what Llama / Mistral /
    Qwen / Gemma / DeepSeek all use. Computed in fp32 under autocast for
    numerical stability, then cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Promote to fp32 for stable variance computation.
        input_dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        x_normalized = x_fp32 * rms
        return (x_normalized * self.weight).to(input_dtype)


# ----------------------------------------------------------------------
# Rotary Position Embeddings (used by all attention variants).
# ----------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """
    Precomputes RoPE cos/sin tables.

    RoPE applies a rotation to each (2i, 2i+1) pair of query/key features
    by an angle that depends on the token position. This makes the
    Q·K dot product encode relative position rather than absolute.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Frequencies: theta_i = base^(-2i / head_dim) for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        ))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (T, head_dim/2)
        freqs = torch.outer(positions, inv_freq)
        # Duplicate each freq to cover the full head_dim (one freq per pair).
        # cos/sin will be (T, head_dim).
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        # Return (seq_len, head_dim) cos and sin in the requested dtype.
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate-half operation: split the last dim in two and swap with signs.
    For x = [a, b] (two halves), rotate_half(x) = [-b, a].
    """
    half = x.shape[-1] // 2
    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([-b, a], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    Apply RoPE to query and key tensors.

    q, k: (B, num_heads, T, head_dim)
    cos, sin: (T, head_dim)

    Returns rotated q, k of the same shapes.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ----------------------------------------------------------------------
# SwiGLU MLP (used by Llama and Qwen; Gemma uses GeGLU which lives in
# models/gemma.py).
# ----------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    """
    Llama-style SwiGLU MLP block:
        gate, up = split(x @ W_gate_up)
        return down(silu(gate) * up)

    The gate+up projection is fused into a single 2I-output linear.
    With intermediate I ≈ 8/3 × H (the SwiGLU parameter-parity recipe),
    the total FFN parameters work out to about 3 × H × I.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size, 2 * intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            intermediate_size, hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class GeluMLP(nn.Module):
    """
    Plain (un-gated) GELU MLP block:
        return down(gelu(up(x)))

    No gating. Used as a macro-cleanliness baseline for Phase 2: gated
    FFNs (SwiGLU, GeGLU) have a multiplicative interaction between two
    projections that can amplify per-token sensitivity and inflate the
    kurtosis of the residual stream. The ungated GELU produces
    "well-mannered" variance contributions that scale roughly linearly
    with input variance, making it easier to attribute observed
    blunderbuss properties to other design choices (depth, width, etc.)
    rather than to FFN-internal nonlinearity quirks.

    Parameter-parity note. SwiGLU has 3 × H × I params per block;
    plain GELU has 2 × H × I_gelu. For the two to match parameter
    counts, I_gelu = 1.5 × I_swiglu. So replacing SwiGLU at I=2432
    with GeluMLP at the parameter-matched setting requires I_gelu=3648.
    Callers are responsible for passing the correct intermediate_size.
    See LlamaBlock for how this is handled at the model-construction
    level. NOTE: This is NOT the same as Gemma's "GeGLU" — that's a
    gated FFN with GELU as the gate activation. This is ungated.
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


# ----------------------------------------------------------------------
# Parameter counting and memory estimation utilities.
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_training_memory_gb(
    model_cfg, micro_batch_size: int,
    seq_len: int, dtype_bytes: int = 2,
) -> dict:
    """
    Rough VRAM estimate for training the model, broken down by component.

    Returns a dict with keys: 'weights_gb', 'gradients_gb', 'optimizer_gb',
    'activations_gb', 'total_gb'. The optimizer estimate assumes AdamW
    (two fp32 state tensors per parameter, fp32 master copy of weights).

    Activation estimate is approximate — it accounts for the residual
    stream and per-layer activations but doesn't model every intermediate
    tensor exactly. With gradient checkpointing on, the activation cost
    is roughly the residual stream × num_layers (each layer's input is
    saved for the backward recomputation).

    Architecture-agnostic: works for any of the 150M variants since they
    all have similar parameter counts at the same (H, L, I) sizes.
    """
    P = model_cfg.estimate_param_count()
    weights_gb = P * dtype_bytes / 1e9
    gradients_gb = P * dtype_bytes / 1e9  # match weight dtype
    # AdamW: m and v in fp32 (4 bytes each) plus master weights in fp32.
    optimizer_gb = P * (4 + 4 + 4) / 1e9
    H = model_cfg.hidden_size
    L = model_cfg.num_hidden_layers
    act_per_layer = micro_batch_size * seq_len * H * dtype_bytes
    activations_gb = act_per_layer * (L + 1) / 1e9
    total_gb = weights_gb + gradients_gb + optimizer_gb + activations_gb
    return {
        "weights_gb": weights_gb,
        "gradients_gb": gradients_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "total_gb": total_gb,
    }
    