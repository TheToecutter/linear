"""
Tests for QwenStyleTransformer.

The Qwen variant is architecturally identical to Llama except for QK-Norm
inside attention, so the test surface is essentially the same: parameter
counts, hidden-state shape and layout, RoPE behavior, RMSNorm behavior,
loss-at-init sanity, gradient checkpointing equivalence, causal attention
non-leakage.

Additional tests specific to Qwen:
  - QK-Norm modules exist on attention layers
  - QK-Norm modules are RMSNorm with the right dim (head_dim)
  - Loss at init is similar to Llama's (because the additional QK-norm
    parameters are initialized to identity-equivalent values)

Run:  python3 test_qwen.py
"""

import sys
import torch
import torch.nn as nn

from config import ModelConfig, TrainingConfig
from models import QwenStyleTransformer, RMSNorm, count_parameters
from models.qwen import QwenCausalSelfAttention


def make_tiny_config() -> ModelConfig:
    """Tiny config for fast tests."""
    cfg = ModelConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=128, gradient_checkpointing=False,
        architecture="qwen",
    )
    return cfg


# ----------------------------------------------------------------------
# Structural tests.
# ----------------------------------------------------------------------
def test_qwen_has_qk_norm():
    """Every attention layer in a Qwen model should have q_norm and k_norm."""
    print("test_qwen_has_qk_norm ... ", end="")
    cfg = make_tiny_config()
    model = QwenStyleTransformer(cfg)
    for block in model.blocks:
        assert hasattr(block.attn, "q_norm"), "Missing q_norm in attention"
        assert hasattr(block.attn, "k_norm"), "Missing k_norm in attention"
        assert isinstance(block.attn.q_norm, RMSNorm), (
            "q_norm should be RMSNorm"
        )
        assert isinstance(block.attn.k_norm, RMSNorm), (
            "k_norm should be RMSNorm"
        )
        # head_dim should be the right dimension.
        assert block.attn.q_norm.weight.shape[0] == cfg.head_dim, (
            f"q_norm dim mismatch: got {block.attn.q_norm.weight.shape[0]}, "
            f"expected head_dim={cfg.head_dim}"
        )
    print("OK")


def test_qwen_parameter_count_close_to_llama():
    """Qwen has the same params as Llama plus 2 * num_heads * head_dim
    per layer (one RMSNorm gain per Q and K per head)."""
    print("test_qwen_parameter_count_close_to_llama ... ", end="")
    cfg = make_tiny_config()
    qwen_model = QwenStyleTransformer(cfg)

    # Build a Llama with the same config (override architecture).
    from models import LlamaStyleTransformer
    llama_cfg = ModelConfig(**{
        **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()
           if f.name != "architecture"},
        "architecture": "llama",
    })
    llama_model = LlamaStyleTransformer(llama_cfg)

    qwen_total, _ = count_parameters(qwen_model)
    llama_total, _ = count_parameters(llama_model)
    extra = qwen_total - llama_total
    # Expected: 2 norms per attention layer × head_dim per norm × L layers.
    # In our implementation, q_norm and k_norm each have dim=head_dim
    # (the norm is applied per-head on the head_dim axis, with a SHARED
    # learnable gain across heads).
    expected_extra = 2 * cfg.head_dim * cfg.num_hidden_layers
    assert extra == expected_extra, (
        f"Qwen has {extra} more params than Llama; expected {expected_extra}"
    )
    print(f"OK (Qwen: {qwen_total:,}, Llama: {llama_total:,}, "
          f"diff: +{extra})")


def test_qwen_forward_shape():
    """Forward should produce logits of the right shape."""
    print("test_qwen_forward_shape ... ", end="")
    cfg = make_tiny_config()
    model = QwenStyleTransformer(cfg)
    model.eval()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss, hidden = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Bad logits shape: {logits.shape}"
    )
    assert loss is None
    assert hidden is None
    print("OK")


def test_qwen_hidden_states_layout():
    """Hidden states have the same layout as Llama: L+2 tensors."""
    print("test_qwen_hidden_states_layout ... ", end="")
    cfg = make_tiny_config()
    model = QwenStyleTransformer(cfg)
    model.eval()

    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    _, _, hidden = model(input_ids, return_hidden_states=True)
    expected_len = cfg.num_hidden_layers + 2  # input + L block outputs + final-norm
    assert len(hidden) == expected_len, (
        f"Expected {expected_len} hidden states, got {len(hidden)}"
    )
    for i, h in enumerate(hidden):
        assert h.shape == (B, T, cfg.hidden_size), (
            f"hidden[{i}] has shape {h.shape}, expected ({B}, {T}, {cfg.hidden_size})"
        )
    print(f"OK ({expected_len} hidden states of shape ({B}, {T}, {cfg.hidden_size}))")


def test_qwen_loss_at_init_reasonable():
    """At init, cross-entropy loss should be close to log(vocab_size)
    (uniform baseline)."""
    print("test_qwen_loss_at_init_reasonable ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(42)
    model = QwenStyleTransformer(cfg)
    model.eval()

    B, T = 4, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = input_ids.clone()

    with torch.no_grad():
        _, loss, _ = model(input_ids, labels=labels)
    import math
    uniform_baseline = math.log(cfg.vocab_size)
    assert abs(loss.item() - uniform_baseline) < 1.0, (
        f"Loss at init {loss.item():.3f} too far from uniform baseline "
        f"{uniform_baseline:.3f}"
    )
    print(f"OK (loss={loss.item():.3f}, baseline={uniform_baseline:.3f})")


def test_qwen_forward_deterministic():
    """Forward should be deterministic given a fixed seed."""
    print("test_qwen_forward_deterministic ... ", end="")
    cfg = make_tiny_config()

    torch.manual_seed(0)
    model1 = QwenStyleTransformer(cfg)
    torch.manual_seed(0)
    model2 = QwenStyleTransformer(cfg)
    model1.eval(); model2.eval()

    B, T = 2, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    logits1, _, _ = model1(input_ids)
    logits2, _, _ = model2(input_ids)
    assert torch.allclose(logits1, logits2), (
        "Outputs differ for the same seed!"
    )
    print("OK")


def test_qwen_causal_attention_no_leak():
    """Token at position t shouldn't depend on tokens at positions > t."""
    print("test_qwen_causal_attention_no_leak ... ", end="")
    cfg = make_tiny_config()
    torch.manual_seed(0)
    model = QwenStyleTransformer(cfg)
    model.eval()

    B, T = 1, 8
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        logits1, _, _ = model(input_ids)

        # Change input at the last position; logits at positions 0..T-2
        # should not change.
        input_ids2 = input_ids.clone()
        input_ids2[0, -1] = (input_ids2[0, -1] + 1) % cfg.vocab_size
        logits2, _, _ = model(input_ids2)

    diff = (logits1[:, :-1, :] - logits2[:, :-1, :]).abs().max()
    assert diff < 1e-5, (
        f"Causal attention leaked: max diff at earlier positions = {diff}"
    )
    print("OK")


def test_qwen_distinct_from_llama_at_init():
    """Same seed, same config but different architecture string — Qwen and
    Llama should produce different outputs because Qwen has the extra
    QK-Norm step that Llama doesn't."""
    print("test_qwen_distinct_from_llama_at_init ... ", end="")
    cfg = make_tiny_config()
    cfg_llama = ModelConfig(**{
        **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()
           if f.name != "architecture"},
        "architecture": "llama",
    })

    torch.manual_seed(0)
    qwen = QwenStyleTransformer(cfg)
    torch.manual_seed(0)
    from models import LlamaStyleTransformer
    llama = LlamaStyleTransformer(cfg_llama)

    qwen.eval(); llama.eval()
    B, T = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        q_logits, _, _ = qwen(input_ids)
        l_logits, _, _ = llama(input_ids)

    # Outputs should differ — QK-norm changes attention behavior.
    diff = (q_logits - l_logits).abs().max().item()
    assert diff > 1e-3, (
        f"Qwen and Llama outputs are suspiciously close: max diff {diff}. "
        f"QK-Norm might not be doing anything?"
    )
    print(f"OK (max output diff = {diff:.4f})")


def test_qwen_factory_dispatch():
    """build_model should produce a QwenStyleTransformer when
    config.architecture == 'qwen'."""
    print("test_qwen_factory_dispatch ... ", end="")
    cfg = make_tiny_config()
    assert cfg.architecture == "qwen"

    from models import build_model
    model = build_model(cfg)
    assert isinstance(model, QwenStyleTransformer), (
        f"build_model gave {type(model).__name__}, expected QwenStyleTransformer"
    )
    print("OK")


def main():
    tests = [
        test_qwen_has_qk_norm,
        test_qwen_parameter_count_close_to_llama,
        test_qwen_forward_shape,
        test_qwen_hidden_states_layout,
        test_qwen_loss_at_init_reasonable,
        test_qwen_forward_deterministic,
        test_qwen_causal_attention_no_leak,
        test_qwen_distinct_from_llama_at_init,
        test_qwen_factory_dispatch,
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
