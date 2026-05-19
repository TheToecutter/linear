"""
DeepSeek-style 150M transformer (Variant D).

The architectural distinguishing feature of this variant is **Multi-head
Latent Attention (MLA)**, introduced in DeepSeek-V2 and refined in V3.
Other than MLA, this variant follows Llama-style conventions: pre-RMSNorm
block structure, SwiGLU MLP, RoPE, tied embeddings.

================ What MLA is and why we implement it ================

Standard multi-head attention has per-token projections W_Q, W_K, W_V each
of shape (H, H). At inference time it caches K and V at full rank for each
position. At large scale (H ~ 4096+, long context, batch size > 1) this
KV cache dominates inference memory.

MLA replaces the K and V projections with a two-stage scheme:

  C^{KV} = X @ W_DKV     # down-project to a small latent of dim d_c
  K      = C^{KV} @ W_UK # up-project to (num_heads, head_dim)
  V      = C^{KV} @ W_UV # up-project to (num_heads, head_dim)

A symmetric scheme is applied to Q:

  C^Q = X @ W_DQ         # down-project to latent of dim d'_c
  Q   = C^Q @ W_UQ       # up-project to (num_heads, head_dim)

The total parameters of these four projections are similar to standard
MHA's three projections at large scale (because the down-projections are
small). The inference-time advantage: the KV cache stores only the
d_c-dimensional latent per position, not full per-head K and V.

================ Why MLA matters for our flow analysis ================

At our pilot scale (146M, seq_len=1024), the inference-cache benefit is
irrelevant — we're not running long-context inference. What MATTERS for
the lines-of-thought analysis is the **structural constraint** MLA imposes
on attention.

In standard MHA, K and V are unconstrained full-rank H-dimensional matrices.
In MLA, K and V are constructed by up-projection from a rank-d_c latent.
This means *every layer's attention output is constrained to a particular
low-rank subspace structure*. We hypothesize this will show up in the
recovered linear flow as:

  - Lower effective rank in the residual stream (the rank constraint on
    K/V propagates through attention into the residual stream)
  - Different principal-direction trajectory R(t) (the attention output
    occupies a more structured subspace)
  - Possibly different variance scaling rate λ (rank-constrained additions
    may scale differently with depth)

If universality at the basis-invariant level holds across all four
variants despite this structural difference, that's a strong finding.
If it fails specifically on this variant, the failure points to the
attention rank-constraint as architecturally consequential.

================ Implementation choices ================

DeepSeek-V3's actual MLA includes a "decoupled RoPE" mechanism: RoPE is
incompatible with naive MLA because the rotation depends on position and
breaks the latent-cache abstraction. DeepSeek's fix is to split K (and Q)
into a non-positional part (derived from the latent) and a positional
part (separately projected, with RoPE applied), then concatenate.

For the present pilot we use **simplified MLA**: apply RoPE on the
up-projected Q and K directly, ignoring the inference-cache issue. This
is a known simplification — the decoupled-RoPE machinery is purely an
inference optimization and doesn't affect training dynamics or the
recovered flow structure. The CORE architectural feature of MLA — the
rank-constrained K/V derived from a low-dimensional latent — is fully
preserved. We disclose this simplification explicitly so a reviewer
checking against DeepSeek-V3's reference doesn't get confused.

The parameter count differs from Llama. With H=896, d_c=96, d'_c=192,
num_heads=14, head_dim=64:
  Standard MHA per layer: 4 × 896² = 3.21M
  MLA per layer:
    W_DKV:  H × d_c = 896 × 96 = 0.086M
    W_UK:   d_c × (num_heads × head_dim) = 96 × 896 = 0.086M
    W_UV:   d_c × (num_heads × head_dim) = 96 × 896 = 0.086M
    W_DQ:   H × d'_c = 896 × 192 = 0.172M
    W_UQ:   d'_c × (num_heads × head_dim) = 192 × 896 = 0.172M
    W_O:    H × H = 0.803M
    Total:  ~1.41M per layer (~56% fewer attention params than standard MHA)

At 12 layers that's ~22M fewer parameters in attention vs Llama, partly
offset elsewhere by the SwiGLU MLP being the same. Total parameter count
should still be around 130M — close to Llama's 146M but visibly smaller.
This is intrinsic to MLA's design; we cannot make MLA match standard MHA's
parameter count without distorting the architectural feature.

Reference: DeepSeek-V2 (May 2024), DeepSeek-V3 Technical Report (Dec 2024).
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from config import ModelConfig
from .shared import RMSNorm, RotaryEmbedding, apply_rope, SwiGLUMLP


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (simplified for training-only setting).

    Architecture:

        # Inputs
        x: (B, T, H)

        # Q path: down-project to latent, then up-project to per-head Q
        c_Q = x @ W_DQ            # (B, T, d'_c)
        Q   = c_Q @ W_UQ          # (B, T, num_heads * head_dim)

        # KV path: down-project to (smaller) latent, then up-project K and V
        c_KV = x @ W_DKV          # (B, T, d_c)
        K    = c_KV @ W_UK        # (B, T, num_heads * head_dim)
        V    = c_KV @ W_UV        # (B, T, num_heads * head_dim)

        # Reshape to heads, apply RoPE on Q and K, scaled dot-product attention
        Q, K, V = reshape_to_heads(Q, K, V)
        Q, K = apply_rope(Q, K)
        attn_out = SDPA(Q, K, V, is_causal=True)
        return out_proj(reshape_back(attn_out))

    The simplification is that RoPE is applied directly on the up-projected
    Q and K (after reshape to per-head). DeepSeek-V3's reference impl
    uses a decoupled scheme for inference-cache efficiency — see module
    docstring.
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.kv_latent_dim = config.mla_kv_latent_dim
        self.q_latent_dim = config.mla_q_latent_dim
        self.rotary_emb = rotary_emb

        # Q path.
        self.q_down_proj = nn.Linear(
            config.hidden_size, config.mla_q_latent_dim, bias=False,
        )
        self.q_up_proj = nn.Linear(
            config.mla_q_latent_dim, config.hidden_size, bias=False,
        )
        # The DeepSeek paper also normalizes the Q latent (q_a_layernorm in
        # their reference). We use RMSNorm to match the framework's overall
        # use of RMSNorm.
        self.q_latent_norm = RMSNorm(config.mla_q_latent_dim, eps=config.rms_norm_eps)

        # KV path.
        self.kv_down_proj = nn.Linear(
            config.hidden_size, config.mla_kv_latent_dim, bias=False,
        )
        self.kv_latent_norm = RMSNorm(config.mla_kv_latent_dim, eps=config.rms_norm_eps)
        # K and V each get their own up-projection from the shared KV latent.
        self.k_up_proj = nn.Linear(
            config.mla_kv_latent_dim, config.hidden_size, bias=False,
        )
        self.v_up_proj = nn.Linear(
            config.mla_kv_latent_dim, config.hidden_size, bias=False,
        )

        # Output projection (same as standard MHA).
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape

        # Q path: down → norm → up.
        c_q = self.q_down_proj(x)                              # (B, T, q_latent)
        c_q = self.q_latent_norm(c_q)
        q = self.q_up_proj(c_q)                                # (B, T, H)

        # KV path: down → norm → (up-K, up-V).
        c_kv = self.kv_down_proj(x)                            # (B, T, kv_latent)
        c_kv = self.kv_latent_norm(c_kv)
        k = self.k_up_proj(c_kv)                               # (B, T, H)
        v = self.v_up_proj(c_kv)                               # (B, T, H)

        # Reshape to (B, num_heads, T, head_dim).
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (simplified — no decoupled scheme).
        cos, sin = self.rotary_emb(T, x.device, x.dtype)
        q, k = apply_rope(q, k, cos, sin)

        # Standard scaled dot-product attention.
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True,
        )

        # Reshape back to (B, T, H) and apply output projection.
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out_proj(attn_out)


class DeepSeekBlock(nn.Module):
    """
    DeepSeek-style transformer block: pre-RMSNorm + MLA + residual,
    then pre-RMSNorm + SwiGLU MLP + residual.

    Same block structure as Llama; only the attention module differs.

        h = x + MLA(RMSNorm(x))
        out = h + SwiGLU_MLP(RMSNorm(h))
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config, rotary_emb)
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class DeepSeekStyleTransformer(nn.Module):
    """
    150M-class DeepSeek-style decoder-only transformer using MLA.

    Architecturally identical to LlamaStyleTransformer except that attention
    is replaced with MultiHeadLatentAttention. See module docstring for the
    architectural rationale and our simplification of MLA.

    The forward signature, hidden-state layout, and public API are
    intentionally the same as LlamaStyleTransformer — drop-in replacement
    for the training loop and analysis pipeline.

    Parameter count is somewhat smaller than Llama's (~130M vs ~146M at
    matched H, L, I). This is intrinsic to MLA's rank-compressed attention
    and cannot be eliminated without distorting the architectural feature.
    Phase 2's matched-compute training comparison will use the same token
    budget across all four variants; the loss-conditioned comparison will
    factor out any quality differences from this parameter-count gap.
    """

    architecture_name = "deepseek"

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
            DeepSeekBlock(config, self.rotary_emb)
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
