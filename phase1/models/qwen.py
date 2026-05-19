"""
Qwen-style 150M transformer (Variant C).

Architectural delta from Llama-style:
  - **QK-Norm**: RMSNorm is applied to Q and K *after* the QKV projection
    and *before* RoPE. This stabilizes the attention dot-products by
    keeping Q/K magnitudes well-controlled regardless of input magnitude.
  - Everything else is identical to Llama-style: pre-RMSNorm in the block,
    SwiGLU MLP, RoPE, full multi-head attention, tied embeddings.

Reference: Qwen-3 (2024-2025), DeepSeek-V3, OpenAI's o-series scale
language models. QK-Norm originates from Henry et al. 2020
("Query-Key Normalization for Transformers") and has become the de-facto
standard for stabilizing attention in large models trained at extreme
scale.

What flow-analysis behavior we expect to differ from Llama-style:
  - The attention "logit distribution" is more uniform-magnitude across
    different inputs (the whole point of QK-norm), which may produce
    *smoother* singular-vector trajectories with depth.
  - Variance scaling rate λ may differ if QK-norm changes how each
    layer's contribution scales relative to the residual stream.
  - Within-layer structure (eff_rank, kurtosis) likely similar to Llama-style
    in the asymptote since QK-norm is mostly about training stability,
    not about the trained model's representation geometry. But Phase 1
    will tell us empirically.

Parameter count is essentially identical to Llama-style (QK-Norm adds
2 * num_heads * head_dim parameters total — negligible at 150M).
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig
from .shared import RMSNorm, RotaryEmbedding, apply_rope, SwiGLUMLP


class QwenCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with QK-Norm + RoPE.

    The QK-Norm is RMSNorm applied per-head to Q and K, separately, after
    projection and before RoPE. Each head gets its own learned scale
    (a parameter of shape (head_dim,)).

    Pseudocode of the forward pass differences from LlamaCausalSelfAttention:
        q, k, v = qkv_proj(x).chunk(3)
        q, k, v = reshape_to_heads(q, k, v)         # (B, num_heads, T, head_dim)
        q = q_norm(q)   # NEW: per-head RMSNorm
        k = k_norm(k)   # NEW: per-head RMSNorm
        q, k = apply_rope(q, k, ...)
        attn_out = SDPA(q, k, v, is_causal=True)
        return out_proj(reshape_back(attn_out))
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rotary_emb = rotary_emb

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        # QK-Norm: per-head RMSNorm on Q and K.
        # The norm is applied on the head_dim axis. We use RMSNorm rather
        # than LayerNorm to match Qwen's actual implementation.
        self.q_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ──── QK-Norm: applied per-head, before RoPE. ────
        # Q and K have shape (B, num_heads, T, head_dim). The RMSNorm
        # normalizes along the head_dim axis (the last dim), which is
        # exactly what nn.LayerNorm-like modules do.
        q = self.q_norm(q)
        k = self.k_norm(k)
        # ─────────────────────────────────────────────────

        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(attn_out)


class QwenBlock(nn.Module):
    """
    Qwen-style transformer block. Same structure as LlamaBlock, but uses
    QwenCausalSelfAttention (with QK-norm) instead of LlamaCausalSelfAttention.

        h = x + Attention_with_QK_norm(RMSNorm(x))
        out = h + SwiGLU_MLP(RMSNorm(h))
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = QwenCausalSelfAttention(config, rotary_emb)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class QwenStyleTransformer(nn.Module):
    """
    150M-parameter Qwen-style decoder-only transformer.

    Architecturally identical to LlamaStyleTransformer except for QK-Norm
    inside attention. The forward signature, hidden-state layout, and all
    public API are intentionally the same as LlamaStyleTransformer — they
    are drop-in interchangeable for the training loop and analysis
    pipeline. See LlamaStyleTransformer's docstring for details.
    """

    architecture_name = "qwen"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.blocks = nn.ModuleList([
            QwenBlock(config, self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # RMSNorm weights (including the new q_norm and k_norm) initialized
        # to 1, which is the default behavior of nn.Parameter(torch.ones).

    def get_lm_head_weight(self) -> torch.Tensor:
        if self.config.tie_embeddings:
            return self.token_embed.weight
        return self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
    ):
        B, T = input_ids.shape
        assert T <= self.config.max_position_embeddings, (
            f"Sequence length {T} exceeds max_position_embeddings "
            f"{self.config.max_position_embeddings}"
        )

        x = self.token_embed(input_ids)
        hidden_states = [x.detach()] if return_hidden_states else None

        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = gradient_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if return_hidden_states:
                hidden_states.append(x.detach())

        x_final = self.final_norm(x)
        if return_hidden_states:
            hidden_states.append(x_final.detach())

        lm_head_weight = self.get_lm_head_weight()
        logits = F.linear(x_final, lm_head_weight)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="mean",
            )

        return logits, loss, hidden_states
