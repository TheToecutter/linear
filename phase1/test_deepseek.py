"""
Tests for DeepSeekStyleTransformer (simplified MLA).

DeepSeek differs from Llama in the attention module: Multi-head Latent
Attention (MLA) replaces standard MHA. The tests verify:

  - MLA module has the right component structure (down/up projections, norms)
  - K and V are rank-constrained (rank ≤ kv_latent_dim)
  - Q is rank-constrained (rank ≤ q_latent_dim)
  - Parameter count is smaller than Llama's (as expected for MLA)
  - Forward shape, hidden-state layout, determinism, causal non-leakage
  - Factory dispatch + analyzer integration

Run:  python3 test_deepseek.py
"""

import sys
import math
import torch
import torch.nn as nn

from config import ModelConfig
from models import DeepSeekStyleTransformer, RMSNorm, count_parameters
from models.deepseek import MultiHeadLatentAttention, DeepSeekBlock


def make_tiny_config(architecture="deepseek") -> ModelConfig:
    """Tiny config for fast tests.

    KV latent (16) and Q latent (32) are smaller than head_dim × num_heads (64)
    so MLA's rank constraints are clearly testable.
    """
    return ModelConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=128, gradient_checkpointing=False,
        architecture=architecture,
        mla_kv_latent_dim=16,
        mla_q_latent_dim=32,
    )


# ----------------------------------------------------------------------
# Structural tests.
# ----------------------------------------------------------------------
def test_deepseek_block_has_mla():
    """Each block's attention should be MultiHeadLatentAttention."""
    print("test_deepseek_block_has_mla ... ", end="")
    cfg = make_tiny_config()
    model = DeepSeekStyleTransformer(cfg)
    for block in model.blocks:
        assert isinstance(block.attn, MultiHeadLatentAttention), (
            f"Block attention is {type(block.attn).__name__}, expected MLA"
        )
    print("OK")


def test_mla_has_down_up_projections():
    """MLA should have the four projections: q_down, q_up, kv_down, k_up, v_up."""
    print("test_mla_has_down_up_projections ... ", end="")
    cfg = make_tiny_config()
    model = DeepSeekStyleTransformer(cfg)
    attn = model.blocks[0].attn
    # Q path
    assert attn.q_down_proj.in_features == cfg.hidden_size
    assert attn.q_down_proj.out_features == cfg.mla_q_latent_dim
    assert attn.q_up_proj.in_features == cfg.mla_q_latent_dim
    assert attn.q_up_proj.out_features == cfg.hidden_size
    # KV path
    assert attn.kv_down_proj.in_features == cfg.hidden_size
    assert attn.kv_down_proj.out_features == cfg.mla_kv_latent_dim
    assert attn.k_up_proj.in_features == cfg.mla_kv_latent_dim
    assert attn.k_up_proj.out_features == cfg.hidden_size
    assert attn.v_up_proj.in_features == cfg.mla_kv_latent_dim
    assert attn.v_up_proj.out_features == cfg.hidden_size
    # Latent norms
    assert isinstance(attn.q_latent_norm, RMSNorm)
    assert isinstance(attn.kv_latent_norm, RMSNorm)
    assert attn.q_latent_norm.weight.shape[0] == cfg.mla_q_latent_dim
    assert attn.kv_latent_norm.weight.shape[0] == cfg.mla_kv_latent_dim
    # Output projection
    assert attn.out_proj.in_features == cfg.hidden_size
    assert attn.out_proj.out_features == cfg.hidden_size
    print("OK")


def test_mla_kv_rank_constrained():
    """The K and V tensors that MLA produces must lie in a kv_latent_dim
    subspace of the hidden space, since they're up-projected from a latent
    of that dimension.

    We probe this by capturing the (B*T, H)-shaped K and V tensors after
    up-projection and checking their numerical rank.
    """
    print("test_mla_kv_rank_constrained ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    attn = model.blocks[0].attn

    # Forward through just the first block's MLA-relevant part (manually).
    B, T = 4, 32
    H = cfg.hidden_size
    x = torch.randn(B, T, H)

    with torch.no_grad():
        # Replicate MLA's K and V computation up to RoPE.
        c_kv = attn.kv_down_proj(x)        # (B, T, kv_latent_dim)
        c_kv = attn.kv_latent_norm(c_kv)
        k_full = attn.k_up_proj(c_kv)       # (B, T, H)
        v_full = attn.v_up_proj(c_kv)       # (B, T, H)

    # Flatten across (B, T) to get a (B*T, H) matrix; check its rank.
    k_flat = k_full.reshape(-1, H)          # (N, H)
    v_flat = v_full.reshape(-1, H)
    k_rank = torch.linalg.matrix_rank(k_flat).item()
    v_rank = torch.linalg.matrix_rank(v_flat).item()

    # Rank must be ≤ kv_latent_dim. (May be exactly kv_latent_dim if the
    # up-projection matrix is full-rank, which it should be at random init.)
    assert k_rank <= cfg.mla_kv_latent_dim, (
        f"K rank {k_rank} exceeds kv_latent_dim {cfg.mla_kv_latent_dim}"
    )
    assert v_rank <= cfg.mla_kv_latent_dim, (
        f"V rank {v_rank} exceeds kv_latent_dim {cfg.mla_kv_latent_dim}"
    )
    assert k_rank == cfg.mla_kv_latent_dim, (
        f"K rank {k_rank} below maximum {cfg.mla_kv_latent_dim} — up-projection"
        f" may be rank-deficient at init"
    )
    print(f"OK (K rank = V rank = {k_rank}, kv_latent_dim = {cfg.mla_kv_latent_dim})")


def test_mla_q_rank_constrained():
    """Similar to K/V: Q is rank-constrained by q_latent_dim."""
    print("test_mla_q_rank_constrained ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    attn = model.blocks[0].attn

    B, T = 4, 32
    H = cfg.hidden_size
    x = torch.randn(B, T, H)

    with torch.no_grad():
        c_q = attn.q_down_proj(x)
        c_q = attn.q_latent_norm(c_q)
        q_full = attn.q_up_proj(c_q)        # (B, T, H)

    q_flat = q_full.reshape(-1, H)
    q_rank = torch.linalg.matrix_rank(q_flat).item()
    assert q_rank <= cfg.mla_q_latent_dim, (
        f"Q rank {q_rank} exceeds q_latent_dim {cfg.mla_q_latent_dim}"
    )
    assert q_rank == cfg.mla_q_latent_dim, (
        f"Q rank {q_rank} below maximum {cfg.mla_q_latent_dim}"
    )
    print(f"OK (Q rank = {q_rank}, q_latent_dim = {cfg.mla_q_latent_dim})")


def test_deepseek_parameter_count_smaller_than_llama():
    """MLA has fewer attention parameters than standard MHA at our scales.

    Per-layer attention parameter count:
      Llama (standard MHA): 4 × H² = 4 × hidden_size²
      DeepSeek (MLA):
        q_down:  H × q_latent          = hidden_size × q_latent_dim
        q_up:    q_latent × H          = q_latent_dim × hidden_size
        kv_down: H × kv_latent         = hidden_size × kv_latent_dim
        k_up:    kv_latent × H         = kv_latent_dim × hidden_size
        v_up:    kv_latent × H         = kv_latent_dim × hidden_size
        out:     H × H                  = hidden_size × hidden_size
        + RMSNorm for q_latent (q_latent_dim params)
        + RMSNorm for kv_latent (kv_latent_dim params)
    """
    print("test_deepseek_parameter_count_smaller_than_llama ... ", end="")
    cfg_ds = make_tiny_config(architecture="deepseek")
    cfg_llama = make_tiny_config(architecture="llama")

    ds = DeepSeekStyleTransformer(cfg_ds)
    from models import LlamaStyleTransformer
    llama = LlamaStyleTransformer(cfg_llama)

    ds_total, _ = count_parameters(ds)
    llama_total, _ = count_parameters(llama)
    diff = llama_total - ds_total

    # Compute expected diff per layer.
    H, qL, kvL = cfg_ds.hidden_size, cfg_ds.mla_q_latent_dim, cfg_ds.mla_kv_latent_dim
    llama_attn_per_layer = 4 * H * H
    mla_attn_per_layer = (
        H * qL + qL * H +              # q_down, q_up
        H * kvL + kvL * H + kvL * H +  # kv_down, k_up, v_up
        H * H +                         # out_proj
        qL + kvL                        # RMSNorms (per-element gains)
    )
    expected_diff_per_layer = llama_attn_per_layer - mla_attn_per_layer
    expected_total_diff = expected_diff_per_layer * cfg_ds.num_hidden_layers

    assert diff == expected_total_diff, (
        f"Llama has {diff} more params than DeepSeek; expected {expected_total_diff}. "
        f"(per-layer diff: actual = {diff // cfg_ds.num_hidden_layers}, "
        f"expected = {expected_diff_per_layer})"
    )
    # Confirm MLA is indeed smaller (the test would also catch the wrong sign).
    assert ds_total < llama_total, "DeepSeek should have fewer params than Llama"
    print(f"OK (DeepSeek: {ds_total:,}, Llama: {llama_total:,}, "
          f"Llama-DeepSeek = +{diff})")


# ----------------------------------------------------------------------
# Behavioral tests.
# ----------------------------------------------------------------------
def test_deepseek_forward_shape():
    """Forward produces logits of the right shape."""
    print("test_deepseek_forward_shape ... ", end="")
    cfg = make_tiny_config()
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, hidden = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None and hidden is None
    print("OK")


def test_deepseek_hidden_states_layout():
    """Hidden states layout: L+2 tensors of shape (B, T, H)."""
    print("test_deepseek_hidden_states_layout ... ", end="")
    cfg = make_tiny_config()
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    _, _, hidden = model(input_ids, return_hidden_states=True)
    expected_len = cfg.num_hidden_layers + 2
    assert len(hidden) == expected_len
    for h in hidden:
        assert h.shape == (B, T, cfg.hidden_size)
    print(f"OK ({expected_len} hidden states)")


def test_deepseek_loss_at_init_reasonable():
    """At init, loss should be near log(vocab_size) (uniform baseline)."""
    print("test_deepseek_loss_at_init_reasonable ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(42)
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    B, T = 4, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        _, loss, _ = model(input_ids, labels=input_ids)
    baseline = math.log(cfg.vocab_size)
    assert abs(loss.item() - baseline) < 1.0, (
        f"Loss at init {loss.item():.3f} far from baseline {baseline:.3f}"
    )
    print(f"OK (loss={loss.item():.3f}, baseline={baseline:.3f})")


def test_deepseek_forward_deterministic():
    """Forward is deterministic given a fixed seed."""
    print("test_deepseek_forward_deterministic ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0); m1 = DeepSeekStyleTransformer(cfg); m1.eval()
    torch.manual_seed(0); m2 = DeepSeekStyleTransformer(cfg); m2.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    out1, _, _ = m1(input_ids)
    out2, _, _ = m2(input_ids)
    assert torch.allclose(out1, out2)
    print("OK")


def test_deepseek_causal_attention_no_leak():
    """Causal: changing token at position t shouldn't affect earlier positions."""
    print("test_deepseek_causal_attention_no_leak ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = DeepSeekStyleTransformer(cfg)
    model.eval()
    B, T = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out1, _, _ = model(input_ids)
        input_ids2 = input_ids.clone()
        input_ids2[0, -1] = (input_ids2[0, -1] + 1) % cfg.vocab_size
        out2, _, _ = model(input_ids2)
    diff = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max()
    assert diff < 1e-5, f"Causal leak: diff {diff} at earlier positions"
    print("OK")


def test_deepseek_factory_dispatch():
    """build_model produces a DeepSeekStyleTransformer for arch='deepseek'."""
    print("test_deepseek_factory_dispatch ... ", end="")
    cfg = make_tiny_config()
    from models import build_model
    model = build_model(cfg)
    assert isinstance(model, DeepSeekStyleTransformer)
    print("OK")


def test_deepseek_distinct_from_llama_at_init():
    """Same seed, same config except architecture — outputs should differ
    substantially because MLA replaces attention entirely."""
    print("test_deepseek_distinct_from_llama_at_init ... ", end="")
    cfg_ds = make_tiny_config(architecture="deepseek")
    cfg_llama = make_tiny_config(architecture="llama")
    torch.manual_seed(0); ds = DeepSeekStyleTransformer(cfg_ds)
    torch.manual_seed(0)
    from models import LlamaStyleTransformer
    llama = LlamaStyleTransformer(cfg_llama)
    ds.eval(); llama.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg_ds.vocab_size, (B, T))
    with torch.no_grad():
        d_logits, _, _ = ds(input_ids)
        l_logits, _, _ = llama(input_ids)
    diff = (d_logits - l_logits).abs().max().item()
    assert diff > 1e-2, (
        f"DeepSeek and Llama outputs are too close: max diff {diff}"
    )
    print(f"OK (max output diff = {diff:.4f})")


def test_deepseek_analyzer_integration():
    """A DeepSeek model's hidden states should feed cleanly into the analyzer
    and produce finite, sensible values."""
    print("test_deepseek_analyzer_integration ... ", end="")
    import numpy as np
    from analyze import recover_linear_flow

    cfg = make_tiny_config()
    cfg = ModelConfig(**{
        **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()},
        "num_hidden_layers": 6,  # more depth for clearer trajectories
    })
    torch.manual_seed(0)
    model = DeepSeekStyleTransformer(cfg)
    model.eval()

    B, T = 16, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        _, _, hidden = model(input_ids, return_hidden_states=True)

    H = cfg.hidden_size
    activations = np.stack([h.numpy().reshape(-1, H) for h in hidden], axis=0)
    flow = recover_linear_flow(activations, center=True)

    assert np.isfinite(flow['lambda'])
    assert np.isfinite(flow['log_alpha'])
    assert np.all(np.isfinite(flow['effective_rank']))
    print(f"OK (λ={flow['lambda']:+.3f}, log α={flow['log_alpha']:+.3f}, "
          f"eff_rank[0]={flow['effective_rank'][0]:.1f})")


def main():
    tests = [
        test_deepseek_block_has_mla,
        test_mla_has_down_up_projections,
        test_mla_kv_rank_constrained,
        test_mla_q_rank_constrained,
        test_deepseek_parameter_count_smaller_than_llama,
        test_deepseek_forward_shape,
        test_deepseek_hidden_states_layout,
        test_deepseek_loss_at_init_reasonable,
        test_deepseek_forward_deterministic,
        test_deepseek_causal_attention_no_leak,
        test_deepseek_factory_dispatch,
        test_deepseek_distinct_from_llama_at_init,
        test_deepseek_analyzer_integration,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"FAIL\n   {e}")
            failures.append((fn.__name__, str(e)))
        except Exception as e:
            print(f"ERROR\n   {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append((fn.__name__, f"{type(e).__name__}: {e}"))
    print()
    if failures:
        print(f"❌ {len(failures)}/{len(tests)} tests failed:")
        for name, msg in failures:
            print(f"   - {name}: {msg}")
        sys.exit(1)
    print(f"✅ All {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
