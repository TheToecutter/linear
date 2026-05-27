"""
Gemma-style 150M transformer (Variant B).

Architectural deltas from Llama-style:

  1. **Hybrid pre+post RMSNorm.** Each block has FOUR RMSNorm modules
     instead of two. The structure is:

         h = x + post_attn_norm(Attention(pre_attn_norm(x)))
         out = h + post_mlp_norm(MLP(pre_mlp_norm(h)))

     The post-norm before each residual add bounds the magnitude of each
     sublayer's contribution to the residual stream, which is the main
     architectural difference from Llama-family blocks.

  2. **GeGLU MLP (instead of SwiGLU).** Uses GELU as the gate's activation
     function instead of SiLU/Swish. The fused-projection structure is
     otherwise identical:
         GeGLU(x) = GELU(W_gate x) * (W_up x)
         MLP(x)   = W_down(GeGLU(x))

  3. **Alternating sliding-window and full attention.** Even-indexed layers
     use sliding-window attention with window=4096 (Gemma-2 default); odd-
     indexed layers use full attention.

     At our pilot's sequence length of 1024, the sliding window (4096) is
     larger than every sequence, so sliding-window attention is FUNCTIONALLY
     IDENTICAL to full attention for all our experiments. We implement
     it faithfully but flag that this architectural feature is inert at
     our pilot scale. Increasing seq_len > 4096 in future work would
     activate it.

  4. **Attention logit softcap.** Inside each attention head, raw Q·K dot
     products are passed through a tanh softcap before softmax:
         scores = softcap * tanh(scores / softcap)
     This bounds attention scores per-head, with default softcap=50.

  5. **Final logit softcap.** The LM head output is bounded the same way:
         logits = softcap * tanh(logits / softcap)
     Default softcap=30. This only affects the output, not the residual
     stream, so it's invisible to the lines-of-thought analysis. We include
     it for training-dynamics fidelity to Gemma-2.

What we expect to differ from Llama in the recovered flow:

  - The hybrid post-norm is the most consequential change. The post-norm
    bounds each sublayer's contribution to the residual stream, which
    plausibly *reduces* the exponential variance scaling rate λ (since
    growth is checked at every layer) and *increases* the effective rank
    at depth (since post-normed contributions spread magnitude more
    evenly across dimensions).
  - The GeGLU vs SwiGLU change is small (different gate activation; same
    fused structure). Unlikely to substantially affect the recovered flow.
  - The attention softcap is a soft non-linearity bounding extreme scores;
    its effect on the residual stream is subtle but real.
  - The final logit softcap and sliding window do not affect the recovered
    flow at our scale.

Reference: Gemma-2 Technical Report (Google DeepMind, 2024).
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig
from .shared import RMSNorm, RotaryEmbedding, apply_rope


# ----------------------------------------------------------------------
# GeGLU MLP (Gemma uses GELU as the gate's activation function).
# ----------------------------------------------------------------------
class GeGLUMLP(nn.Module):
    """
    Gemma-style GeGLU MLP block:
        gate, up = split(x @ W_gate_up)
        return down(GELU(gate) * up)

    The only difference from SwiGLUMLP is the gate activation: GELU
    instead of SiLU. The fused gate+up projection and the down projection
    are structured identically to Llama-style SwiGLU.

    We use the GELU "tanh approximation" variant (also called gelu_pytorch_tanh)
    which is what Gemma uses in its reference implementation. The numerical
    differences from the "exact" GELU are negligible but consistent with
    Gemma's training.
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
        # GELU(approximate="tanh") matches Gemma's reference impl.
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


# ----------------------------------------------------------------------
# Attention with logit softcap and optional sliding-window mask.
# ----------------------------------------------------------------------
class GemmaCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with:
      - Optional sliding-window mask (window = config.sliding_window)
      - Attention logit softcap (softcap = config.attn_logit_softcap)
      - RoPE on Q and K (same as Llama)

    The softcap applies the tanh-based bound to raw Q·K scores BEFORE
    softmax: `scores = softcap * tanh(scores / softcap)`. This prevents
    any single attention score from dominating, which is the main effect
    Gemma-2 reports as improving training stability.

    Note on SDPA: PyTorch's scaled_dot_product_attention doesn't natively
    support the softcap operation. With softcap enabled, we fall back to
    explicit attention computation. This is slower than SDPA but
    architecturally faithful. With softcap=0 (or disabled), we'd dispatch
    to SDPA. For Gemma we always have a positive softcap, so we always
    use the explicit path.

    Args at init:
      - is_sliding: if True, use sliding-window attention with the configured
        window size. If False, use full attention. Gemma-2 alternates:
        even-index layers are sliding; odd-index layers are full.
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding,
                 is_sliding: bool):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.rotary_emb = rotary_emb
        self.is_sliding = is_sliding
        self.softcap = config.attn_logit_softcap
        # Precomputed scale matches SDPA's default.
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=False
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )

    def _build_attention_mask(self, T: int, device, dtype) -> torch.Tensor:
        """
        Build a (T, T) attention mask combining causal + (optionally)
        sliding-window restriction. Mask value of -inf prevents attention;
        0 allows it.

        For seq_len < sliding_window, the sliding-window mask is identical
        to the causal mask — sliding is inert in that regime.
        """
        # Causal: position i attends to positions 0..i.
        # mask[i, j] = 0 if j <= i, else -inf
        mask = torch.zeros(T, T, dtype=dtype, device=device)
        i = torch.arange(T, device=device)
        j = torch.arange(T, device=device)
        # Standard causal mask.
        causal_block = i.unsqueeze(1) < j.unsqueeze(0)  # (T, T), True where j > i
        if self.is_sliding:
            # Sliding-window: position i can only attend to positions in
            # [i - window + 1, i]. Combine with causal: positions j in
            # max(0, i - window + 1) .. i.
            window = self.config.sliding_window
            too_far_back = (i.unsqueeze(1) - j.unsqueeze(0)) >= window
            forbidden = causal_block | too_far_back
        else:
            forbidden = causal_block
        mask = mask.masked_fill(forbidden, float("-inf"))
        return mask  # (T, T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Explicit attention with softcap. Compute Q · K^T / sqrt(d).
        # Shape: (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T)
        #       = (B, num_heads, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply tanh softcap to raw scores BEFORE the mask + softmax.
        if self.softcap > 0:
            attn_scores = self.softcap * torch.tanh(attn_scores / self.softcap)

        # Apply causal + sliding-window mask.
        mask = self._build_attention_mask(T, x.device, attn_scores.dtype)
        # Broadcast: (T, T) → (1, 1, T, T) → (B, num_heads, T, T)
        attn_scores = attn_scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax + apply to values.
        attn_weights = F.softmax(attn_scores, dim=-1)
        # Compute attention output: (B, num_heads, T, T) @ (B, num_heads, T, head_dim)
        attn_out = torch.matmul(attn_weights, v)

        # Reshape back: (B, num_heads, T, head_dim) → (B, T, H).
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(attn_out)


# ----------------------------------------------------------------------
# Gemma block with hybrid pre+post RMSNorm.
# ----------------------------------------------------------------------
class GemmaBlock(nn.Module):
    """
    Gemma-style transformer block with hybrid pre+post RMSNorm:

        h = x + post_attn_norm(Attention(pre_attn_norm(x)))
        out = h + post_mlp_norm(MLP(pre_mlp_norm(h)))

    The post-norms bound each sublayer's contribution to the residual
    stream before the residual add. This is Gemma's main architectural
    distinction from Llama-family blocks.

    The block accepts an `is_sliding` flag determining whether its
    attention is the sliding-window variant (used by even-indexed layers)
    or full attention (used by odd-indexed layers). The block structure
    is otherwise identical regardless.
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding,
                 is_sliding: bool):
        super().__init__()
        # Pre-norm (input to sublayer).
        self.pre_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = GemmaCausalSelfAttention(config, rotary_emb, is_sliding=is_sliding)
        # Post-norm (sublayer output, before residual add).
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GeGLUMLP(config.hidden_size, config.intermediate_size)
        self.post_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention sublayer with hybrid norm.
        h = self.pre_attn_norm(x)
        h = self.attn(h)
        h = self.post_attn_norm(h)
        x = x + h
        # MLP sublayer with hybrid norm.
        h = self.pre_mlp_norm(x)
        h = self.mlp(h)
        h = self.post_mlp_norm(h)
        x = x + h
        return x


# ----------------------------------------------------------------------
# Full model with final logit softcap and alternating sliding/full layers.
# ----------------------------------------------------------------------
class GemmaStyleTransformer(nn.Module):
    """
    150M-parameter Gemma-style decoder-only transformer.

    The architecture differs from Llama in four ways (see module docstring):
    hybrid pre+post norm, GeGLU MLP, alternating sliding/full attention,
    and softcaps on attention scores and final logits.

    For the lines-of-thought analysis, the hidden-state layout is
    identical to LlamaStyleTransformer (post-embedding + per-block-output +
    post-final-norm). The block-output states are the FULL residual stream
    AFTER both the attention residual add and the MLP residual add, just
    like in Llama. The post-norms operate on sublayer outputs (bounded
    contributions before adding back to the residual stream), but the
    residual stream itself remains unconstrained — which is exactly the
    representation we want to analyze.
    """

    architecture_name = "gemma"

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.hidden_size)

        self.rotary_emb = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        # Alternating sliding/full attention: even layers (0, 2, 4, ...) are
        # sliding-window; odd layers (1, 3, 5, ...) are full attention. This
        # matches Gemma-2's layer-by-layer schedule.
        self.blocks = nn.ModuleList([
            GemmaBlock(config, self.rotary_emb, is_sliding=(i % 2 == 0))
            for i in range(config.num_hidden_layers)
        ])

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if not config.tie_embeddings:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False,
            )

        self._init_weights()

    def _init_weights(self):
        """Standard init: N(0, 0.02) for embeddings and linears."""
        std = 0.02
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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

        # Final logit softcap. Bounds logits to ±softcap. This is post-hoc
        # to the residual stream we just analyzed, so it doesn't affect
        # hidden_states but does affect the loss.
        if self.config.final_logit_softcap > 0:
            cap = self.config.final_logit_softcap
            logits = cap * torch.tanh(logits / cap)

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
