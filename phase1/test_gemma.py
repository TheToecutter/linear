"""
Tests for GemmaStyleTransformer.

Gemma differs from Llama in four architectural respects (see models/gemma.py
docstring). The tests verify each:

  - Hybrid pre+post RMSNorm structure (4 norms per block, not 2)
  - GeGLU MLP (GELU gate, not SiLU)
  - Alternating sliding/full attention
  - Softcaps on attention scores and final logits

Plus the standard tests covering shape, hidden-state layout, determinism,
causal-attention non-leakage, and factory dispatch.

Run:  python3 test_gemma.py
"""

import sys
import math
import torch
import torch.nn as nn

from config import ModelConfig
from models import GemmaStyleTransformer, RMSNorm, count_parameters
from models.gemma import (
    GemmaCausalSelfAttention, GemmaBlock, GeGLUMLP,
)


def make_tiny_config(architecture="gemma") -> ModelConfig:
    """Tiny config for fast tests."""
    return ModelConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4,
        max_position_embeddings=128, gradient_checkpointing=False,
        architecture=architecture,
        # Test with smaller sliding window so it's not always inert.
        sliding_window=32,
    )


# ----------------------------------------------------------------------
# Structural tests.
# ----------------------------------------------------------------------
def test_gemma_block_has_four_norms():
    """A Gemma block should have pre+post RMSNorm for both attention and MLP."""
    print("test_gemma_block_has_four_norms ... ", end="")
    cfg = make_tiny_config()
    model = GemmaStyleTransformer(cfg)
    for block in model.blocks:
        assert hasattr(block, "pre_attn_norm")
        assert hasattr(block, "post_attn_norm")
        assert hasattr(block, "pre_mlp_norm")
        assert hasattr(block, "post_mlp_norm")
        for name in ["pre_attn_norm", "post_attn_norm", "pre_mlp_norm", "post_mlp_norm"]:
            norm = getattr(block, name)
            assert isinstance(norm, RMSNorm), f"{name} is not RMSNorm"
            assert norm.weight.shape[0] == cfg.hidden_size, (
                f"{name} has wrong dim"
            )
    print("OK")


def test_gemma_mlp_is_geglu():
    """The MLP should be GeGLUMLP, not SwiGLUMLP."""
    print("test_gemma_mlp_is_geglu ... ", end="")
    cfg = make_tiny_config()
    model = GemmaStyleTransformer(cfg)
    for block in model.blocks:
        assert isinstance(block.mlp, GeGLUMLP), (
            f"Block MLP is {type(block.mlp).__name__}, expected GeGLUMLP"
        )
    print("OK")


def test_gemma_alternating_attention_pattern():
    """Even layers should be sliding; odd layers should be full attention."""
    print("test_gemma_alternating_attention_pattern ... ", end="")
    cfg = make_tiny_config()
    model = GemmaStyleTransformer(cfg)
    for i, block in enumerate(model.blocks):
        expected_sliding = (i % 2 == 0)
        actual = block.attn.is_sliding
        assert actual == expected_sliding, (
            f"Layer {i}: is_sliding={actual}, expected {expected_sliding}"
        )
    print(f"OK ({cfg.num_hidden_layers} layers: alternating sliding/full)")


# ----------------------------------------------------------------------
# Behavioral tests.
# ----------------------------------------------------------------------
def test_gemma_forward_shape():
    """Forward produces logits of the right shape."""
    print("test_gemma_forward_shape ... ", end="")
    cfg = make_tiny_config()
    model = GemmaStyleTransformer(cfg)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, hidden = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None and hidden is None
    print("OK")


def test_gemma_hidden_states_layout():
    """Hidden states layout: L+2 tensors of shape (B, T, H)."""
    print("test_gemma_hidden_states_layout ... ", end="")
    cfg = make_tiny_config()
    model = GemmaStyleTransformer(cfg)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    _, _, hidden = model(input_ids, return_hidden_states=True)
    expected_len = cfg.num_hidden_layers + 2
    assert len(hidden) == expected_len
    for h in hidden:
        assert h.shape == (B, T, cfg.hidden_size)
    print(f"OK ({expected_len} hidden states)")


def test_gemma_loss_at_init_reasonable():
    """At init, loss should be near log(vocab_size).

    Note: Gemma's final logit softcap (default 30) bounds logits, which
    shifts the post-softmax distribution slightly even at init. The
    softcap doesn't make the uniform-baseline check inapplicable — at
    init the model produces near-zero logits which the softcap leaves
    nearly unchanged (tanh(small/30) * 30 ≈ small). So we expect the
    baseline check to pass with the same tolerance as Llama/Qwen."""
    print("test_gemma_loss_at_init_reasonable ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(42)
    model = GemmaStyleTransformer(cfg)
    model.eval()
    B, T = 4, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        _, loss, _ = model(input_ids, labels=input_ids)
    baseline = math.log(cfg.vocab_size)
    assert abs(loss.item() - baseline) < 1.0, (
        f"Loss at init {loss.item():.3f} far from uniform baseline {baseline:.3f}"
    )
    print(f"OK (loss={loss.item():.3f}, baseline={baseline:.3f})")


def test_gemma_forward_deterministic():
    """Forward is deterministic given a fixed seed."""
    print("test_gemma_forward_deterministic ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0); m1 = GemmaStyleTransformer(cfg); m1.eval()
    torch.manual_seed(0); m2 = GemmaStyleTransformer(cfg); m2.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    out1, _, _ = m1(input_ids)
    out2, _, _ = m2(input_ids)
    assert torch.allclose(out1, out2)
    print("OK")


def test_gemma_causal_attention_no_leak():
    """Causal attention: changing token at position t shouldn't affect t-1 or earlier."""
    print("test_gemma_causal_attention_no_leak ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = GemmaStyleTransformer(cfg)
    model.eval()
    B, T = 1, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out1, _, _ = model(input_ids)
        input_ids2 = input_ids.clone()
        input_ids2[0, -1] = (input_ids2[0, -1] + 1) % cfg.vocab_size
        out2, _, _ = model(input_ids2)
    diff = (out1[:, :-1, :] - out2[:, :-1, :]).abs().max()
    assert diff < 1e-5, f"Causal leak: max diff {diff} at earlier positions"
    print("OK")


def test_gemma_sliding_window_restricts_attention():
    """A sliding-window-only Gemma should give different output than a full-attn
    Gemma when seq_len > sliding_window.

    We use seq_len=64 > sliding_window=32, so the sliding-window layers
    cannot see tokens more than 32 positions in the past. Changing token 0
    should not affect the output at the last position (which is 63 positions
    later) — assuming all layers are sliding.

    To make this clean we use a config where all layers are sliding (force).
    """
    print("test_gemma_sliding_window_restricts_attention ... ", end="")
    cfg = make_tiny_config()
    # Build a single GemmaBlock with sliding-window=True, all-sliding model.
    # Easier: use the GemmaCausalSelfAttention directly on a seq longer than window.
    from models.shared import RotaryEmbedding
    torch.manual_seed(0)
    rotary = RotaryEmbedding(
        head_dim=cfg.head_dim, max_seq_len=cfg.max_position_embeddings,
        base=cfg.rope_theta,
    )
    attn = GemmaCausalSelfAttention(cfg, rotary, is_sliding=True)
    attn.eval()

    H = cfg.hidden_size
    T = 64  # > sliding_window=32
    x = torch.randn(1, T, H)
    with torch.no_grad():
        out1 = attn(x)
        # Modify token 0 (>= 32 positions before the end).
        x2 = x.clone()
        x2[0, 0, :] = torch.randn(H)
        out2 = attn(x2)

    # Output at position T-1 (= 63) is 63 positions away from position 0,
    # so > sliding_window=32. The sliding window cannot reach token 0 from
    # position 63 → outputs at positions >= sliding_window must be identical.
    diff_far = (out1[:, cfg.sliding_window:, :] - out2[:, cfg.sliding_window:, :]).abs().max()
    diff_near = (out1[:, :cfg.sliding_window, :] - out2[:, :cfg.sliding_window, :]).abs().max()
    assert diff_far < 1e-5, (
        f"Sliding window leaks: positions >= {cfg.sliding_window} should not see "
        f"token 0, but max diff = {diff_far}"
    )
    # Positions WITHIN the sliding window of token 0 should see the change.
    assert diff_near > 1e-3, (
        f"Sliding window too restrictive: positions < {cfg.sliding_window} should "
        f"see token 0, but max diff = {diff_near}"
    )
    print(f"OK (far positions diff={diff_far:.2e}, near positions diff={diff_near:.4f})")


def test_gemma_logit_softcap_bounds_output():
    """The final logit softcap should bound logits to ±softcap.

    We can test this by setting a small softcap and checking that no
    logit exceeds it (since the softcap is applied to ALL logits including
    those of "winning" tokens during inference).
    """
    print("test_gemma_logit_softcap_bounds_output ... ", end="")
    cfg = make_tiny_config()
    # Override softcap to a small value to make the test sharp.
    cfg.final_logit_softcap = 3.0
    torch.manual_seed(0)
    model = GemmaStyleTransformer(cfg)
    model.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _, _ = model(input_ids)
    max_abs_logit = logits.abs().max().item()
    assert max_abs_logit < cfg.final_logit_softcap + 1e-3, (
        f"Logits exceed softcap {cfg.final_logit_softcap}: max abs = {max_abs_logit}"
    )
    print(f"OK (max abs logit {max_abs_logit:.3f} < softcap {cfg.final_logit_softcap})")


def test_gemma_attention_softcap_bounds_scores():
    """Attention logit softcap should bound the pre-softmax attention scores.

    We can't easily probe the scores from the outside, but we can verify
    behaviorally: with extremely large input magnitudes, the model should
    NOT produce NaN/Inf because the softcap prevents extreme scores.
    """
    print("test_gemma_attention_softcap_bounds_scores ... ", end="")
    cfg = make_tiny_config()
    cfg.attn_logit_softcap = 5.0
    torch.manual_seed(0)
    model = GemmaStyleTransformer(cfg)
    model.eval()
    # Forward with a normal input — should produce finite output.
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        logits, _, _ = model(input_ids)
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
    # Manually inflate the attention input by overriding embeddings.
    # This stress-tests that the softcap kicks in to keep things finite.
    with torch.no_grad():
        model.token_embed.weight.mul_(100.0)  # huge embeddings
        try:
            logits2, _, _ = model(input_ids)
            assert torch.isfinite(logits2).all(), (
                "Logits NaN/Inf even with attention softcap"
            )
        finally:
            model.token_embed.weight.div_(100.0)  # restore
    print("OK")


def test_gemma_factory_dispatch():
    """build_model produces a GemmaStyleTransformer for architecture='gemma'."""
    print("test_gemma_factory_dispatch ... ", end="")
    cfg = make_tiny_config()
    from models import build_model
    model = build_model(cfg)
    assert isinstance(model, GemmaStyleTransformer)
    print("OK")


def test_gemma_distinct_from_llama_at_init():
    """Same seed, same config except architecture — outputs should differ
    because Gemma has hybrid norm + GeGLU + softcaps."""
    print("test_gemma_distinct_from_llama_at_init ... ", end="")
    cfg_gemma = make_tiny_config(architecture="gemma")
    cfg_llama = make_tiny_config(architecture="llama")

    torch.manual_seed(0); gemma = GemmaStyleTransformer(cfg_gemma)
    torch.manual_seed(0)
    from models import LlamaStyleTransformer
    llama = LlamaStyleTransformer(cfg_llama)

    gemma.eval(); llama.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg_gemma.vocab_size, (B, T))
    with torch.no_grad():
        g_logits, _, _ = gemma(input_ids)
        l_logits, _, _ = llama(input_ids)

    diff = (g_logits - l_logits).abs().max().item()
    # Gemma should differ substantially from Llama because of four
    # architectural changes that all affect the forward pass.
    assert diff > 1e-2, (
        f"Gemma and Llama outputs are too close: max diff {diff}. "
        f"Are any of the architectural changes silently no-ops?"
    )
    print(f"OK (max output diff = {diff:.4f})")


def test_gemma_parameter_count_close_to_llama():
    """Gemma has slightly more params than Llama because hybrid norm adds
    2 extra RMSNorm modules per block (post-norms)."""
    print("test_gemma_parameter_count_close_to_llama ... ", end="")
    cfg_gemma = make_tiny_config(architecture="gemma")
    cfg_llama = make_tiny_config(architecture="llama")
    gemma = GemmaStyleTransformer(cfg_gemma)
    from models import LlamaStyleTransformer
    llama = LlamaStyleTransformer(cfg_llama)

    g_total, _ = count_parameters(gemma)
    l_total, _ = count_parameters(llama)
    extra = g_total - l_total
    # Per block: 2 extra RMSNorm of dim H. Total: 2 × H × L.
    expected_extra = 2 * cfg_gemma.hidden_size * cfg_gemma.num_hidden_layers
    assert extra == expected_extra, (
        f"Gemma has {extra} more params than Llama; expected {expected_extra}"
    )
    print(f"OK (Gemma: {g_total:,}, Llama: {l_total:,}, diff: +{extra})")


def main():
    tests = [
        test_gemma_block_has_four_norms,
        test_gemma_mlp_is_geglu,
        test_gemma_alternating_attention_pattern,
        test_gemma_forward_shape,
        test_gemma_hidden_states_layout,
        test_gemma_loss_at_init_reasonable,
        test_gemma_forward_deterministic,
        test_gemma_causal_attention_no_leak,
        test_gemma_sliding_window_restricts_attention,
        test_gemma_logit_softcap_bounds_output,
        test_gemma_attention_softcap_bounds_scores,
        test_gemma_factory_dispatch,
        test_gemma_distinct_from_llama_at_init,
        test_gemma_parameter_count_close_to_llama,
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
