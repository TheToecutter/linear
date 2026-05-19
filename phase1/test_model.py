"""
Tests for the model module.

Run: python3 test_model.py

These tests are designed to be fast (run on CPU in a few seconds) and to
catch the most likely classes of bugs:
  - parameter counting wrong
  - hidden state shapes wrong
  - tied embedding not actually tied
  - forward pass not deterministic at fixed seed
  - RoPE math wrong (rotation should be norm-preserving)
  - gradient checkpointing path equivalent to non-checkpointing path
"""

import sys
import torch
import torch.nn.functional as F

from config import ModelConfig
from models import (
    LlamaStyleTransformer, RMSNorm, RotaryEmbedding,
    apply_rope, count_parameters,
)


def make_tiny_config():
    """Smaller config for fast tests."""
    return ModelConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        gradient_checkpointing=False,
    )


def test_param_count_matches_estimate():
    """The actual parameter count should match the analytic estimate exactly."""
    print("test_param_count_matches_estimate ... ", end="")
    cfg = ModelConfig()  # full 150M config
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg)
    total, trainable = count_parameters(model)
    estimate = cfg.estimate_param_count()
    diff = abs(trainable - estimate)
    assert diff < 1000, (
        f"Parameter count {trainable} differs from estimate {estimate} by {diff}"
    )
    print(f"OK ({trainable / 1e6:.3f}M)")


def test_hidden_states_shape():
    """Hidden states should be (num_layers + 2) tensors of shape (B, T, H)."""
    print("test_hidden_states_shape ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg).eval()
    B, T = 3, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        _, _, hidden = model(input_ids, return_hidden_states=True)
    assert hidden is not None
    expected_len = cfg.num_hidden_layers + 2  # input + L outputs + final-norm
    assert len(hidden) == expected_len, (
        f"Expected {expected_len} hidden states, got {len(hidden)}"
    )
    for i, h in enumerate(hidden):
        assert h.shape == (B, T, cfg.hidden_size), (
            f"hidden[{i}].shape = {h.shape}, expected ({B}, {T}, {cfg.hidden_size})"
        )
    print("OK")


def test_tied_embedding_actually_tied():
    """With tie_embeddings=True, LM head weight is the embedding weight."""
    print("test_tied_embedding_actually_tied ... ", end="")
    cfg = make_tiny_config()
    cfg.tie_embeddings = True
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg)
    lm_head_w = model.get_lm_head_weight()
    embed_w = model.token_embed.weight
    # Same tensor object — not just equal values.
    assert lm_head_w.data_ptr() == embed_w.data_ptr(), (
        "Tied embedding: LM head weight should share memory with token_embed.weight"
    )
    # Should not have a separate lm_head module.
    assert not hasattr(model, "lm_head") or model.lm_head is None
    print("OK")


def test_untied_embedding_separate():
    """With tie_embeddings=False, LM head is a separate parameter."""
    print("test_untied_embedding_separate ... ", end="")
    cfg = make_tiny_config()
    cfg.tie_embeddings = False
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg)
    lm_head_w = model.get_lm_head_weight()
    embed_w = model.token_embed.weight
    # Different tensors.
    assert lm_head_w.data_ptr() != embed_w.data_ptr(), (
        "Untied: LM head weight should NOT share memory with embedding"
    )
    print("OK")


def test_forward_deterministic_at_fixed_seed():
    """Two forward passes with the same input and seeded init give identical outputs."""
    print("test_forward_deterministic_at_fixed_seed ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model1 = LlamaStyleTransformer(cfg).eval()
    torch.manual_seed(0)
    model2 = LlamaStyleTransformer(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits1, _, _ = model1(input_ids)
        logits2, _, _ = model2(input_ids)
    assert torch.allclose(logits1, logits2), "Same seed → different output"
    print("OK")


def test_rope_norm_preserving():
    """RoPE rotates Q and K, so it should preserve their norms."""
    print("test_rope_norm_preserving ... ", end="")
    head_dim = 64
    seq_len = 32
    rope = RotaryEmbedding(head_dim=head_dim, max_seq_len=seq_len)
    cos, sin = rope(seq_len, device="cpu", dtype=torch.float32)
    # Random Q and K of shape (B, H, T, D).
    torch.manual_seed(0)
    q = torch.randn(2, 4, seq_len, head_dim)
    k = torch.randn(2, 4, seq_len, head_dim)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    # Norms per (batch, head, position) should be preserved within float
    # precision.
    q_norm = q.pow(2).sum(dim=-1).sqrt()
    q_rot_norm = q_rot.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(q_norm, q_rot_norm, atol=1e-5), (
        f"RoPE didn't preserve Q norms: "
        f"max diff = {(q_norm - q_rot_norm).abs().max().item()}"
    )
    k_norm = k.pow(2).sum(dim=-1).sqrt()
    k_rot_norm = k_rot.pow(2).sum(dim=-1).sqrt()
    assert torch.allclose(k_norm, k_rot_norm, atol=1e-5)
    print("OK")


def test_rmsnorm_unit_norm_output():
    """RMSNorm(x) should have RMS = 1 (before applying the weight scale)."""
    print("test_rmsnorm_unit_norm_output ... ", end="")
    dim = 128
    rn = RMSNorm(dim, eps=1e-6)
    # Force weight to 1 so the output is just normalized x.
    with torch.no_grad():
        rn.weight.fill_(1.0)
    torch.manual_seed(0)
    x = torch.randn(4, 16, dim) * 5.0  # Arbitrary scale.
    y = rn(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), (
        f"RMSNorm output should have RMS=1; got mean RMS = {rms.mean().item():.4f}"
    )
    print("OK")


def test_loss_is_finite_at_init():
    """At random init, loss should be finite and close to log(V)."""
    print("test_loss_is_finite_at_init ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        _, loss, _ = model(input_ids, labels=input_ids)
    assert loss is not None and torch.isfinite(loss)
    expected = torch.log(torch.tensor(float(cfg.vocab_size)))
    # Should be within 1 nat of uniform-prediction baseline.
    assert abs(loss.item() - expected.item()) < 1.0, (
        f"Loss {loss.item():.3f} far from uniform baseline {expected.item():.3f}"
    )
    print(f"OK (loss={loss.item():.3f}, uniform baseline={expected.item():.3f})")


def test_gradient_checkpointing_equivalence():
    """With and without gradient checkpointing, the forward pass should give
    identical outputs (the difference is only memory; the math is the same)."""
    print("test_gradient_checkpointing_equivalence ... ", end="")
    cfg = make_tiny_config()
    cfg.gradient_checkpointing = False
    torch.manual_seed(0)
    model_a = LlamaStyleTransformer(cfg)
    model_a.train()  # checkpoint() is a no-op when training=False, so we need train mode
    torch.manual_seed(0)
    cfg.gradient_checkpointing = True
    model_b = LlamaStyleTransformer(cfg)
    model_b.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    # Both in train mode, both seeded identically.
    # No gradients: checkpointing matters only for backward, so forward
    # values must match.
    with torch.no_grad():
        logits_a, _, _ = model_a(input_ids)
        logits_b, _, _ = model_b(input_ids)
    assert torch.allclose(logits_a, logits_b, atol=1e-5), (
        f"Forward outputs differ between checkpointed and non-checkpointed: "
        f"max diff = {(logits_a - logits_b).abs().max().item()}"
    )
    print("OK")


def test_causal_attention_does_not_leak():
    """Causal attention: changing token at position i should not affect
    outputs at positions j < i."""
    print("test_causal_attention_does_not_leak ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = LlamaStyleTransformer(cfg).eval()
    B, T = 1, 8
    input1 = torch.randint(0, cfg.vocab_size, (B, T))
    # Change only the last token.
    input2 = input1.clone()
    input2[0, -1] = (input2[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits1, _, _ = model(input1)
        logits2, _, _ = model(input2)
    # Outputs at positions 0 to T-2 should be identical.
    max_diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max().item()
    assert max_diff < 1e-5, (
        f"Causal attention leaked: changing last token affected earlier "
        f"positions by max {max_diff}"
    )
    print("OK")


def test_memory_estimate_makes_sense():
    """Memory estimate should be small for tiny config, larger for full config."""
    print("test_memory_estimate_makes_sense ... ", end="")
    from models import estimate_training_memory_gb
    tiny = make_tiny_config()
    full = ModelConfig()
    mem_tiny = estimate_training_memory_gb(tiny, micro_batch_size=4, seq_len=64)
    mem_full = estimate_training_memory_gb(full, micro_batch_size=8, seq_len=1024)
    assert mem_tiny["total_gb"] < 0.1
    assert 1.0 < mem_full["total_gb"] < 10.0, (
        f"Full-config memory estimate looks wrong: {mem_full['total_gb']:.2f} GB"
    )
    print("OK")


def main():
    tests = [
        test_param_count_matches_estimate,
        test_hidden_states_shape,
        test_tied_embedding_actually_tied,
        test_untied_embedding_separate,
        test_forward_deterministic_at_fixed_seed,
        test_rope_norm_preserving,
        test_rmsnorm_unit_norm_output,
        test_loss_is_finite_at_init,
        test_gradient_checkpointing_equivalence,
        test_causal_attention_does_not_leak,
        test_memory_estimate_makes_sense,
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
