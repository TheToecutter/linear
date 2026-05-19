"""
The 150M Llama-style transformer (Variant A, reference architecture).

Architecture summary:
  - Decoder-only, pre-RMSNorm
  - Rotary Position Embeddings (RoPE) on Q and K, applied per-head
  - SwiGLU MLP (gate × up, then down projection)
  - Full multi-head causal self-attention (no GQA in the reference variant;
    GQA is a possible Phase 2 modification)
  - Tied input/output embeddings
  - Causal attention via torch.nn.functional.scaled_dot_product_attention
    (uses FlashAttention 2 on Ada/Hopper GPUs; falls back gracefully otherwise)

The architecture is identical in form to Llama 2 / Mistral / Llama 3 (just
smaller). Phase 2 will add three sibling architectures (Gemma-style,
Qwen-style, DeepSeek-style with MLA) by modifying specific components.

This file deliberately contains ONLY the model. Training, data, analysis,
alignment all live in separate modules. The model class exposes hidden
states optionally (for the analysis pipeline) and exposes nothing about
filters, controllers, or interventions — those are out of scope for the
current proposal.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig


# ----------------------------------------------------------------------
# Building blocks.
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    """
    Root-Mean-Square Layer Normalization (Zhang & Sennrich, 2019).

        RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Faster than LayerNorm (no mean centering), and what Llama / Mistral /
    Qwen all use. Computed in fp32 under autocast for numerical stability,
    then cast back to the input dtype.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        rms = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(rms + self.eps)
        return (x_normed.to(dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE; Su et al. 2021).

    Precomputes cos/sin tables of shape (max_seq_len, head_dim) up front.
    For position m and dimension pair (2i, 2i+1):
        theta_i = base^(-2i / head_dim)
        cos[m, i] = cos(m * theta_i),  sin[m, i] = sin(m * theta_i)

    During forward, returns the slices [0:seq_len] of the cached tables,
    moved to the right device/dtype.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, f"RoPE requires even head_dim, got {head_dim}"
        # Inverse frequencies, one per pair of dimensions.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Position indices 0, 1, ..., max_seq_len-1.
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: (max_seq_len, head_dim/2)
        freqs = torch.outer(t, inv_freq)
        # Cache full (max_seq_len, head_dim) cos/sin tables by duplicating
        # each frequency to match the way we split the head dimension below.
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device, dtype):
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the channels and negate, then concat.
    (x1, x2) → (-x2, x1). Combined with cos/sin multiplication, this
    implements a 2D rotation on each pair of channels."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE rotation to Q and K. V is NOT rotated.

    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE on Q and K.

    Uses torch.nn.functional.scaled_dot_product_attention which dispatches
    to FlashAttention 2 on supported GPUs (Ada Lovelace, Hopper, Blackwell).
    Causal mask is requested via `is_causal=True`, which is what enables
    the fast kernel — explicit masks fall back to slower paths.

    The QKV projection is fused into a single linear (3H × H), which is
    standard for efficiency. Output projection is H × H.
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rotary_emb = rotary_emb

        # Fused QKV projection.
        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        # Output projection.
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        # Project to Q, K, V and split.
        qkv = self.qkv_proj(x)  # (B, T, 3H)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, T, head_dim).
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K.
        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention with causal mask.
        # SDPA picks the best available kernel (FlashAttention if supported).
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )

        # Reshape back to (B, T, H).
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(attn_out)


class SwiGLUMLP(nn.Module):
    """
    Llama-style SwiGLU MLP block:
        gate, up = split(x @ W_gate_up)
        return down(silu(gate) * up)

    The gate+up projection is fused into a single 2I-output linear.
    With intermediate I ≈ 8/3 × H (the SwiGLU parameter-parity recipe),
    the total FFN parameters work out to about 3 × H × I.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class TransformerBlock(nn.Module):
    """
    A single Llama-style transformer block: pre-norm + attention + residual,
    then pre-norm + MLP + residual.

        h = x + Attention(RMSNorm(x))
        out = h + MLP(RMSNorm(h))
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = CausalSelfAttention(config, rotary_emb)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ----------------------------------------------------------------------
# The full model.
# ----------------------------------------------------------------------
class LlamaStyleTransformer(nn.Module):
    """
    150M-parameter Llama-style decoder-only transformer.

    Forward signature accepts an optional `return_hidden_states` flag.
    When True, returns the per-layer hidden states (one tensor per layer,
    plus the post-final-norm output), which the analysis pipeline uses
    to fit the lines-of-thought linear flow.

    Hidden state layout when return_hidden_states=True:
        len = num_hidden_layers + 1
        hidden_states[0]: post-embedding, pre-layer-0 (i.e., the input
                          to layer 0, after token embedding)
        hidden_states[i] for 1 ≤ i ≤ num_hidden_layers:
                          output of layer i-1 (input to layer i)
        hidden_states[num_hidden_layers]: post-final-norm, pre-lm-head
                          (i.e., final residual stream state used for
                          next-token prediction)

    The final RMSNorm IS applied to hidden_states[num_hidden_layers]; the
    layer-i outputs (i < num_hidden_layers) are PRE-norm (raw residual
    stream values), which is the right representation for SVD analysis.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (no positional embedding — RoPE handles positions).
        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared rotary embedding cache used by every layer.
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Stack of transformer blocks.
        self.blocks = nn.ModuleList([
            TransformerBlock(config, self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])

        # Final norm before the lm_head.
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head: tied to token_embed when config.tie_embeddings is True.
        # We don't allocate a separate parameter in that case; forward uses
        # F.linear(x, self.token_embed.weight).
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Standard Llama-style init: normal(0, 0.02) for linears, normal(0, 1/sqrt(H)) for embeddings.
        RMSNorm weights start at 1 (already done by nn.Parameter(torch.ones(...)))."""
        std = 0.02
        # Embeddings: small init, scaled by 1/sqrt(H) is a common alternative
        # but 0.02 matches GPT-2/Llama defaults.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_lm_head_weight(self) -> torch.Tensor:
        """Return the LM head weight matrix. Handles the tied-embedding case."""
        if self.config.tie_embeddings:
            return self.token_embed.weight
        return self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ):
        """
        Args:
            input_ids: (B, T) int64 token IDs.
            labels: (B, T) int64, or None. If provided, compute next-token CE loss
                    against labels[..., 1:] using logits[..., :-1, :]. (i.e., the
                    standard "predict next token" objective with shifted labels.)
            return_hidden_states: if True, also return the per-layer hidden state
                    list (see class docstring for layout).

        Returns:
            (logits, loss, hidden_states) where:
              logits: (B, T, V) float
              loss: scalar Tensor or None
              hidden_states: list[Tensor] or None
        """
        B, T = input_ids.shape
        assert T <= self.config.max_position_embeddings, (
            f"Sequence length {T} exceeds max_position_embeddings "
            f"{self.config.max_position_embeddings}"
        )

        x = self.token_embed(input_ids)  # (B, T, H)
        hidden_states = [x.detach()] if return_hidden_states else None

        # Layer stack. With gradient checkpointing on, we wrap each block in
        # torch.utils.checkpoint to save activation memory at the cost of
        # one extra forward pass per block during backward.
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if return_hidden_states:
                hidden_states.append(x.detach())

        # Final norm + LM head.
        x_final = self.final_norm(x)
        if return_hidden_states:
            hidden_states.append(x_final.detach())

        lm_head_weight = self.get_lm_head_weight()
        logits = F.linear(x_final, lm_head_weight)  # (B, T, V)

        loss = None
        if labels is not None:
            # Standard shifted-label next-token CE.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

        return logits, loss, hidden_states


# ----------------------------------------------------------------------
# Utilities for parameter counting and memory estimation.
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_training_memory_gb(
    model_cfg: ModelConfig, micro_batch_size: int,
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
    """
    P = model_cfg.estimate_param_count()
    weights_gb = P * dtype_bytes / 1e9
    gradients_gb = P * dtype_bytes / 1e9  # match weight dtype
    # AdamW: m and v in fp32 (4 bytes each) plus master weights in fp32.
    optimizer_gb = P * (4 + 4 + 4) / 1e9
    # Activations: with gradient checkpointing, we save the input to each
    # block (B × T × H, fp16/bf16) plus the final hidden state.
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
