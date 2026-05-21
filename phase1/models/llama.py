"""
Llama-style 150M transformer (Variant A, reference architecture).

Architecture summary:
  - Decoder-only, pre-RMSNorm
  - Rotary Position Embeddings (RoPE) on Q and K, applied per-head
  - SwiGLU MLP (gate × up, then down projection) — default ffn_type="swiglu"
  - Plain GELU MLP available as ffn_type="gelu" (parameter-matched);
    Phase 2's macro-cleanliness baseline for FFN-agnostic experiments
  - Full multi-head causal self-attention (no GQA in the reference variant)
  - Tied input/output embeddings
  - Causal attention via torch.nn.functional.scaled_dot_product_attention

The architecture is identical in form to Llama 2 / Mistral / Llama 3 (just
smaller). This is the reference variant against which Phase 2's other
three variants (Qwen, Gemma, DeepSeek-MLA) will be compared.
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig
from .shared import RMSNorm, RotaryEmbedding, apply_rope, SwiGLUMLP, GeluMLP


class LlamaCausalSelfAttention(nn.Module):
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

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(attn_out)


class LlamaBlock(nn.Module):
    """
    Llama-style transformer block: pre-RMSNorm, attention + residual,
    then pre-RMSNorm + MLP + residual.

        h = x + Attention(RMSNorm(x))
        out = h + MLP(RMSNorm(h))

    The MLP class is selected by config.ffn_type:
      - "swiglu" (default): SwiGLUMLP at intermediate_size, the
        Phase 1 reference and Llama-2/Mistral convention.
      - "gelu": plain ungated GeluMLP at 1.5 × intermediate_size,
        the macro-cleanliness baseline for Phase 2.
    Both choices use the same per-block parameter count (3 × H × I_swiglu).
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = LlamaCausalSelfAttention(config, rotary_emb)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_type = getattr(config, "ffn_type", "swiglu")
        if ffn_type == "swiglu":
            self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)
        elif ffn_type == "gelu":
            # Parameter-matched intermediate size: SwiGLU has 3HI params,
            # plain MLP has 2HI; so for parity we use I_gelu = 1.5 × I.
            gelu_intermediate = (3 * config.intermediate_size) // 2
            self.mlp = GeluMLP(config.hidden_size, gelu_intermediate)
        else:
            raise ValueError(
                f"Unknown ffn_type {ffn_type!r}. Expected 'swiglu' or 'gelu'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LlamaStyleTransformer(nn.Module):
    """
    150M-parameter Llama-style decoder-only transformer.

    Forward signature accepts an optional `return_hidden_states` flag.
    When True, returns the per-layer hidden states (one tensor per layer,
    plus the post-final-norm output), which the analysis pipeline uses
    to fit the lines-of-thought linear flow.

    Hidden state layout when return_hidden_states=True:
        len = num_hidden_layers + 2
        hidden_states[0]: post-embedding (input to layer 0)
        hidden_states[i] for 1 ≤ i ≤ num_hidden_layers:
                          output of layer i-1 (input to layer i)
        hidden_states[num_hidden_layers + 1]: post-final-norm,
                          the residual stream state used for next-token prediction

    The final RMSNorm IS applied to hidden_states[-1]; the per-layer
    outputs are PRE-norm (raw residual stream values), which is the right
    representation for SVD analysis.
    """

    architecture_name = "llama"

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
            LlamaBlock(config, self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not config.tie_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )

        self._init_weights()

    def _init_weights(self):
        """Standard Llama-style init: N(0, 0.02) for embeddings and linears."""
        std = 0.02
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
                    against labels[..., 1:] using logits[..., :-1, :].
            return_hidden_states: if True, also return the per-layer hidden state
                    list (see class docstring for layout).

        Returns:
            (logits, loss, hidden_states)
        """
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
        