"""
Data loading and tokenization for Phase 1.

Loads FineWeb-Edu (the 10BT sample), tokenizes with the Mistral-7B-v0.1
tokenizer (32k vocab, exactly what the model expects), packs into fixed-
length chunks for training, and reserves a deterministic held-out tail
for evaluation.

Public functions:
  - prepare_dataset(model_cfg, train_cfg, num_proc=None, cache_dir=None)
      Returns (train_dataset, held_out_dataset) of packed chunks.
  - make_dataloaders(train_ds, held_out_ds, train_cfg, seed)
      Returns (train_loader, eval_loader) ready for the training loop.

The held-out split is deterministic — the last `held_out_chunks` chunks
of the tokenized corpus are reserved for eval. Taking the tail (rather
than a random sample) means the eval set is identical across all multi-
seed runs, which is what we want for comparing models trained with
different seeds.

The Mistral tokenizer is identical to Llama 2's SentencePiece BPE but
without the gated-access requirement, so this code can run on a clean
machine without authentication.
"""

import os
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from config import ModelConfig, TrainingConfig


# ----------------------------------------------------------------------
# Tokenization and packing.
# ----------------------------------------------------------------------
def prepare_dataset(
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    num_proc: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> Tuple["Dataset", "Dataset"]:
    """
    Load FineWeb-Edu, tokenize with the Mistral tokenizer, pack into
    fixed-length chunks of train_cfg.train_seq_len, and split into train
    + held-out tail.

    Returns:
        (train_dataset, held_out_dataset) — both HuggingFace Datasets in
        torch format with a single 'input_ids' column. Each example is a
        tensor of shape (train_seq_len,).

    Args:
        model_cfg: For vocab_size sanity check.
        train_cfg: For train_seq_len and held_out_chunks.
        num_proc: Multiprocessing count for tokenization. Defaults to
            min(cpu_count(), 32).
        cache_dir: Hugging Face datasets cache directory. None = default.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("   ↳ Loading Mistral-7B-v0.1 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    actual_vocab = len(tokenizer)
    if model_cfg.vocab_size < actual_vocab:
        raise ValueError(
            f"model_cfg.vocab_size={model_cfg.vocab_size} is smaller than the "
            f"tokenizer vocabulary ({actual_vocab}). Set "
            f"model_cfg.vocab_size = {actual_vocab} or larger."
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    seq_len = train_cfg.train_seq_len
    if seq_len > model_cfg.max_position_embeddings:
        raise ValueError(
            f"train_seq_len={seq_len} exceeds RoPE cache size "
            f"({model_cfg.max_position_embeddings})."
        )

    print("   ↳ Loading FineWeb-Edu (sample-10BT) ...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        cache_dir=cache_dir,
    )

    def tokenize_and_pack(examples):
        tokenized = tokenizer(examples["text"], add_special_tokens=True)
        all_tokens = [tok for doc in tokenized["input_ids"] for tok in doc]
        # Drop the partial-chunk tail.
        total = (len(all_tokens) // seq_len) * seq_len
        chunks = [all_tokens[i : i + seq_len] for i in range(0, total, seq_len)]
        return {"input_ids": chunks}

    if num_proc is None:
        num_proc = min(os.cpu_count() or 4, 32)

    print(f"   ↳ Tokenizing and packing into {seq_len}-token chunks "
          f"({num_proc} processes) ...")
    tokenized_dataset = dataset.map(
        tokenize_and_pack,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing+packing",
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])

    # Deterministic held-out split: the last `held_out_chunks` chunks.
    total_chunks = len(tokenized_dataset)
    held = train_cfg.held_out_chunks
    if held >= total_chunks:
        raise ValueError(
            f"held_out_chunks={held} >= total chunks ({total_chunks})."
        )
    train_dataset = tokenized_dataset.select(range(total_chunks - held))
    held_out_dataset = tokenized_dataset.select(range(total_chunks - held, total_chunks))

    train_tokens = len(train_dataset) * seq_len
    held_tokens = len(held_out_dataset) * seq_len

    # Sanity check: do we have enough data for the requested training duration?
    requested_tokens = (
        train_cfg.micro_batch_size * train_cfg.grad_accum_steps
        * seq_len * train_cfg.total_steps
    )
    print(f"   ↳ ✅ Train: {len(train_dataset):,} chunks "
          f"({train_tokens / 1e9:.2f}B tokens)")
    print(f"   ↳ ✅ Held-out: {len(held_out_dataset):,} chunks "
          f"({held_tokens / 1e6:.1f}M tokens)")
    epochs_needed = requested_tokens / max(train_tokens, 1)
    print(f"   ↳ Training will consume {requested_tokens / 1e9:.2f}B tokens "
          f"(~{epochs_needed:.2f} epochs over the train set)")
    if epochs_needed > 4.0:
        print(f"   ⚠️  Multi-epoch ({epochs_needed:.1f}×) — duplicate exposure "
              f"to the same chunks. Consider a larger corpus or fewer steps.")

    return train_dataset, held_out_dataset


# ----------------------------------------------------------------------
# DataLoader construction.
# ----------------------------------------------------------------------
def make_dataloaders(
    train_dataset,
    held_out_dataset,
    train_cfg: TrainingConfig,
    seed: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap the train and held-out datasets in DataLoaders configured for
    the training loop.

    The train loader is shuffled per-epoch with the given seed (so
    different runs with different seeds see different example orders).
    The eval loader is NOT shuffled — we want stable per-step eval
    metrics across runs.

    Args:
        train_dataset, held_out_dataset: Outputs of prepare_dataset().
        train_cfg: For batch sizes and seq_len.
        seed: Random seed for the train loader's shuffle.
        num_workers: DataLoader worker count. 4 is conservative; the
            5090 workstation has 64 CPU cores so more is fine.
        pin_memory: Use pinned memory for faster H2D transfer.
    """

    def collate(batch):
        # Each example is {"input_ids": tensor of shape (T,)}. Stack to (B, T).
        ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)
        return {"input_ids": ids}

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.micro_batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        generator=train_generator,
        persistent_workers=(num_workers > 0),
    )
    eval_loader = DataLoader(
        held_out_dataset,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, eval_loader
