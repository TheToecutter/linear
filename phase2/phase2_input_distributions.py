"""
Phase 2 Tier 1b: input-distribution decomposition (§5.4 of the proposal).

The proposal's Tier 1b experiment isolates attention's input-sensitive
contribution to the macro structure by running the *same trained model*
on three input distributions:

  1. real     : the standard FineWeb-Edu held-out chunks.
  2. shuffled : the same 500 chunks, but with token order permuted
                within each chunk. Preserves marginal token statistics
                and the embedding-level distribution but destroys
                inter-token correlations.
  3. random   : uniformly sampled tokens from the top-K most frequent
                vocabulary positions, matched to the language token
                distribution but with no semantic correlations.

The FFN-vs-attention prediction (§5.4):
  - FFN operates per-position and should give nearly identical
    macro statistics across (1)/(2)/(3).
  - Attention exploits inter-token structure. Differences between
    (1) and (2) attribute to attention's context-sensitive contribution.

This module supplies the loaders. The analysis side is in
phase2_analyze.py.

Determinism
-----------
The shuffled and random loaders are deterministic given a seed (they
produce the same input_ids tensor every call with the same seed). This
is essential so that flow_real, flow_shuffled, flow_random are
comparable across re-runs and across variants.
"""

from typing import Iterable, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig, TrainingConfig


# ----------------------------------------------------------------------
# Tag for distinguishing flow output directories per input distribution.
# ----------------------------------------------------------------------
INPUT_DIST_NAMES = ("real", "shuffled", "random")

FLOW_SUBDIR_FOR_INPUT = {
    "real":     "flow_analysis",            # the existing baseline analysis
    "shuffled": "flow_analysis_shuffled",
    "random":   "flow_analysis_random",
}


# ----------------------------------------------------------------------
# Shuffled-within-chunk dataset.
# ----------------------------------------------------------------------
class ShuffledChunksDataset(Dataset):
    """Wraps a held-out chunks dataset; on each __getitem__, returns the
    chunk with its token order permuted by a deterministic per-chunk RNG.

    The permutation is keyed by (seed, chunk_index), so the same chunk
    always returns the same shuffle. This is critical for cross-variant
    comparability: every model sees the exact same shuffled token
    sequences.
    """

    def __init__(self, base_dataset, seed: int = 0):
        # base_dataset: a HF-style dataset with .__len__() and __getitem__
        # returning a dict with "input_ids" tensor of shape (T,).
        self.base = base_dataset
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        example = self.base[idx]
        input_ids = example["input_ids"]
        # Use a deterministic RNG seeded by (master_seed, idx).
        # numpy makes this convenient and reproducible across torch versions.
        rng = np.random.default_rng((self.seed, idx))
        perm = rng.permutation(input_ids.shape[0])
        shuffled = input_ids[torch.from_numpy(perm)]
        return {"input_ids": shuffled}


# ----------------------------------------------------------------------
# Random-vocabulary dataset.
# ----------------------------------------------------------------------
class RandomVocabDataset(Dataset):
    """Returns chunks of uniformly random tokens drawn from a top-K
    vocabulary slice (the K most frequent tokens of the training corpus).

    Why top-K rather than full vocab: the full-vocab distribution
    includes many tokens that almost never appear in real text. A
    uniform draw from the full vocab would push the embedding-level
    distribution into a region the trained model never visited, which
    confounds "no inter-token correlations" with "out-of-distribution
    tokens". Sampling from the most-frequent K matches the embedding-
    level marginal distribution more closely (and exactly matches it in
    the limit where token_frequencies is the exact frequency vector).
    """

    def __init__(
        self,
        num_chunks: int,
        seq_len: int,
        token_ids: np.ndarray,
        token_probs: Optional[np.ndarray] = None,
        seed: int = 0,
    ):
        """
        Args:
            num_chunks: how many shuffled chunks to produce.
            seq_len: length of each chunk (matches the trained seq_len).
            token_ids: 1-D int array of vocabulary positions to sample
                from (typically the top-K most frequent).
            token_probs: same length as token_ids; if None, samples
                uniformly. If provided, samples in proportion (good
                approximation of the marginal token distribution).
            seed: master seed for the chunk generator.
        """
        self.num_chunks = int(num_chunks)
        self.seq_len = int(seq_len)
        self.token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_probs is not None:
            p = np.asarray(token_probs, dtype=np.float64)
            if p.shape != self.token_ids.shape:
                raise ValueError(
                    f"token_probs shape {p.shape} != token_ids shape "
                    f"{self.token_ids.shape}"
                )
            p = p / p.sum()  # normalize to PMF
            self.token_probs = p
        else:
            self.token_probs = None
        self.seed = seed

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        rng = np.random.default_rng((self.seed, idx))
        if self.token_probs is None:
            sample = rng.choice(self.token_ids, size=self.seq_len, replace=True)
        else:
            sample = rng.choice(
                self.token_ids, size=self.seq_len, replace=True,
                p=self.token_probs,
            )
        return {"input_ids": torch.from_numpy(sample.astype(np.int64))}


# ----------------------------------------------------------------------
# Frequency-table computation for the random loader.
# ----------------------------------------------------------------------
def compute_token_frequencies(
    held_out_dataset,
    vocab_size: int,
    max_chunks: Optional[int] = None,
) -> np.ndarray:
    """Count tokens across the held-out dataset.

    Returns an int64 array of length vocab_size where entry i is the
    number of occurrences of token i. The held-out set is small enough
    (~500 chunks × 1024 tokens ≈ 500k tokens) that a full pass takes
    well under a second.

    Note we compute on the held-out set rather than the train set so
    the random loader's marginal is matched to what the real loader
    sees.
    """
    counts = np.zeros(vocab_size, dtype=np.int64)
    n_chunks = len(held_out_dataset) if max_chunks is None else min(
        max_chunks, len(held_out_dataset)
    )
    for idx in range(n_chunks):
        ids = held_out_dataset[idx]["input_ids"].numpy()
        # bincount is fast and exact.
        counts += np.bincount(ids, minlength=vocab_size)
    return counts


def topk_vocabulary(
    token_counts: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick the top-K most frequent tokens.

    Returns (token_ids, token_probs) where token_probs are normalized
    over the top-K slice. Useful for parameterizing the random loader
    to match the language marginal closely.
    """
    if top_k > token_counts.shape[0]:
        top_k = token_counts.shape[0]
    # Sort descending; argpartition is faster but we want a clean
    # sorted slice and the vocab is small.
    order = np.argsort(-token_counts)
    top_ids = order[:top_k]
    top_counts = token_counts[top_ids].astype(np.float64)
    top_probs = top_counts / max(top_counts.sum(), 1.0)
    return top_ids, top_probs


# ----------------------------------------------------------------------
# Loader factory.
# ----------------------------------------------------------------------
def make_input_distribution_loaders(
    held_out_dataset,
    train_cfg: TrainingConfig,
    model_cfg: ModelConfig,
    seed: int = 0,
    num_workers: int = 2,
    pin_memory: bool = True,
    random_top_k: int = 4096,
    random_chunk_count: Optional[int] = None,
    random_use_marginal: bool = True,
) -> dict:
    """Build the three DataLoaders for Tier 1b.

    Args:
        held_out_dataset: the held-out tail of the prepared dataset
            (output of data.prepare_dataset).
        train_cfg: for batch sizes and seq_len.
        model_cfg: for vocab_size sanity check.
        seed: master seed for the shuffled/random generators.
        num_workers: DataLoader workers.
        pin_memory: pinned-memory transfer.
        random_top_k: number of top-frequency tokens the random loader
            samples from (4096 ≈ 12.5% of the 32k vocab, captures
            ~95% of held-out token mass for a typical web corpus).
        random_chunk_count: how many chunks the random loader produces
            (defaults to len(held_out_dataset) so the sample sizes
            match the real/shuffled loaders exactly).
        random_use_marginal: if True, samples in proportion to held-out
            top-K marginal frequencies; if False, samples uniformly.

    Returns:
        dict with keys "real", "shuffled", "random", mapping each to
        a torch.utils.data.DataLoader yielding {"input_ids": (B, T)}
        batches.
    """
    def collate(batch):
        ids = torch.stack([ex["input_ids"] for ex in batch], dim=0)
        return {"input_ids": ids}

    # --- real ---
    real_loader = DataLoader(
        held_out_dataset,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # --- shuffled ---
    shuffled_ds = ShuffledChunksDataset(held_out_dataset, seed=seed)
    shuffled_loader = DataLoader(
        shuffled_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    # --- random ---
    # Compute token frequencies on the held-out set, take top-K.
    counts = compute_token_frequencies(
        held_out_dataset, vocab_size=model_cfg.vocab_size,
    )
    top_ids, top_probs = topk_vocabulary(counts, top_k=random_top_k)
    random_probs = top_probs if random_use_marginal else None
    n_random = (
        random_chunk_count if random_chunk_count is not None
        else len(held_out_dataset)
    )
    random_ds = RandomVocabDataset(
        num_chunks=n_random,
        seq_len=train_cfg.train_seq_len,
        token_ids=top_ids,
        token_probs=random_probs,
        seed=seed,
    )
    random_loader = DataLoader(
        random_ds,
        batch_size=train_cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    return {
        "real": real_loader,
        "shuffled": shuffled_loader,
        "random": random_loader,
    }


# ----------------------------------------------------------------------
# Smoke-test helpers (no GPU required).
# ----------------------------------------------------------------------
def _make_synthetic_held_out(num_chunks: int = 20, seq_len: int = 64,
                              vocab_size: int = 1024, seed: int = 0):
    """Return a synthetic held-out dataset for unit tests.

    Mimics the HF-dataset interface that the real prepare_dataset()
    returns: a list-like object of {"input_ids": LongTensor(T,)} dicts.
    """
    rng = np.random.default_rng(seed)
    chunks = []
    for _ in range(num_chunks):
        ids = rng.integers(low=0, high=vocab_size, size=seq_len)
        chunks.append({"input_ids": torch.from_numpy(ids.astype(np.int64))})

    class _SyntheticDS:
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    return _SyntheticDS(chunks)
