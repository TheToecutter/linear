"""
The models/ package: all 150M architecture variants for the pilot.

Public API:
  - LlamaStyleTransformer (Variant A, reference): models/llama.py
  - QwenStyleTransformer (Variant C): models/qwen.py
  - GemmaStyleTransformer (Variant B): models/gemma.py [TODO]
  - DeepSeekStyleTransformer (Variant D): models/deepseek.py [TODO]
  - build_model(config): factory dispatching on config.architecture

  - count_parameters, estimate_training_memory_gb (utility functions)

Adding a new architecture variant:
  1. Create models/<new_name>.py with a `<NewName>StyleTransformer` class
     that has the same forward signature as LlamaStyleTransformer.
  2. Add the import and registration below.
  3. Add the architecture name to config.ModelConfig.architecture's
     allowed values (or relax that field's validation).
  4. Add tests in test_<new_name>.py.
"""

from .llama import LlamaStyleTransformer
from .qwen import QwenStyleTransformer
from .gemma import GemmaStyleTransformer
from .deepseek import DeepSeekStyleTransformer
from .shared import (
    RMSNorm, RotaryEmbedding, apply_rope, SwiGLUMLP,
    count_parameters, estimate_training_memory_gb,
)

# Registry of architecture name -> class.
ARCHITECTURE_REGISTRY = {
    "llama": LlamaStyleTransformer,
    "qwen": QwenStyleTransformer,
    "gemma": GemmaStyleTransformer,
    "deepseek": DeepSeekStyleTransformer,
}


def build_model(config):
    """
    Build a transformer model from a ModelConfig.

    Dispatches based on config.architecture:
      - "llama"     → LlamaStyleTransformer
      - "qwen"      → QwenStyleTransformer  (Llama + QK-Norm)
      - "gemma"     → GemmaStyleTransformer (hybrid norm + GeGLU + sliding/full + softcaps)
      - "deepseek"  → DeepSeekStyleTransformer (Llama + MLA, simplified)

    For backwards compatibility with the pre-multi-architecture pipeline,
    if config.architecture is None, "llama" is used.
    """
    arch = getattr(config, "architecture", None) or "llama"
    if arch not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {arch!r}. "
            f"Known: {sorted(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[arch](config)


__all__ = [
    "LlamaStyleTransformer",
    "QwenStyleTransformer",
    "GemmaStyleTransformer",
    "DeepSeekStyleTransformer",
    "build_model",
    "ARCHITECTURE_REGISTRY",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rope",
    "SwiGLUMLP",
    "count_parameters",
    "estimate_training_memory_gb",
]
