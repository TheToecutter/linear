"""
Tests for phase2_input_distributions.py.

These tests use only synthetic data (no FineWeb-Edu, no GPU). They
verify:
  - The shuffled loader produces the same multiset of tokens as the
    real loader, only in different order.
  - The shuffling is deterministic per (master_seed, chunk_idx).
  - The random loader produces tokens only from the requested top-K
    vocabulary slice.
  - The marginal-matching mode of the random loader produces a
    distribution roughly proportional to token_probs.
  - All three loaders produce the same batch shape and dtype.

Run with:
    python3 -m pytest test_phase2_input_distributions.py -v
"""

import numpy as np
import pytest
import torch

from config import ModelConfig, TrainingConfig
from phase2_input_distributions import (
    ShuffledChunksDataset, RandomVocabDataset,
    compute_token_frequencies, topk_vocabulary,
    make_input_distribution_loaders, _make_synthetic_held_out,
    INPUT_DIST_NAMES, FLOW_SUBDIR_FOR_INPUT,
)


# ----------------------------------------------------------------------
# Fixtures.
# ----------------------------------------------------------------------
@pytest.fixture
def synthetic_held_out():
    return _make_synthetic_held_out(
        num_chunks=20, seq_len=64, vocab_size=128, seed=0,
    )


@pytest.fixture
def configs():
    model_cfg = ModelConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2,
        max_position_embeddings=128,
    )
    train_cfg = TrainingConfig(
        train_seq_len=64, eval_batch_size=4,
    )
    return model_cfg, train_cfg


# ----------------------------------------------------------------------
# Shuffled dataset.
# ----------------------------------------------------------------------
class TestShuffledDataset:
    def test_shuffled_preserves_multiset(self, synthetic_held_out):
        ds = ShuffledChunksDataset(synthetic_held_out, seed=42)
        for idx in range(len(ds)):
            orig = synthetic_held_out[idx]["input_ids"].numpy()
            shuf = ds[idx]["input_ids"].numpy()
            assert sorted(orig.tolist()) == sorted(shuf.tolist()), (
                f"Chunk {idx}: shuffled multiset != original multiset"
            )

    def test_shuffled_is_deterministic(self, synthetic_held_out):
        ds1 = ShuffledChunksDataset(synthetic_held_out, seed=42)
        ds2 = ShuffledChunksDataset(synthetic_held_out, seed=42)
        for idx in range(len(ds1)):
            a = ds1[idx]["input_ids"].numpy()
            b = ds2[idx]["input_ids"].numpy()
            assert np.array_equal(a, b)

    def test_different_seeds_give_different_orderings(self, synthetic_held_out):
        ds1 = ShuffledChunksDataset(synthetic_held_out, seed=1)
        ds2 = ShuffledChunksDataset(synthetic_held_out, seed=2)
        # At seq_len=64 the chance of two random permutations agreeing is
        # negligible. We just need to find ONE chunk that differs.
        any_different = False
        for idx in range(len(ds1)):
            if not np.array_equal(
                ds1[idx]["input_ids"].numpy(),
                ds2[idx]["input_ids"].numpy(),
            ):
                any_different = True
                break
        assert any_different

    def test_shuffled_changes_order(self, synthetic_held_out):
        ds = ShuffledChunksDataset(synthetic_held_out, seed=7)
        any_reordered = False
        for idx in range(len(ds)):
            orig = synthetic_held_out[idx]["input_ids"].numpy()
            shuf = ds[idx]["input_ids"].numpy()
            if not np.array_equal(orig, shuf):
                any_reordered = True
                break
        assert any_reordered, "Shuffling produced identical sequences"


# ----------------------------------------------------------------------
# Random vocabulary dataset.
# ----------------------------------------------------------------------
class TestRandomVocabDataset:
    def test_uniform_samples_in_vocab_slice(self):
        token_ids = np.array([3, 7, 11, 13, 17])
        ds = RandomVocabDataset(
            num_chunks=10, seq_len=50, token_ids=token_ids, seed=0,
        )
        for idx in range(len(ds)):
            sample = ds[idx]["input_ids"].numpy()
            for t in sample:
                assert t in token_ids, (
                    f"Random sample contained out-of-vocab-slice token {t}"
                )

    def test_uniform_covers_vocab_slice(self):
        """With seq_len=200 and 5 tokens, every token should appear."""
        token_ids = np.array([3, 7, 11, 13, 17])
        ds = RandomVocabDataset(
            num_chunks=1, seq_len=200, token_ids=token_ids, seed=0,
        )
        sample = ds[0]["input_ids"].numpy()
        unique = set(sample.tolist())
        assert unique == set(token_ids.tolist())

    def test_marginal_mode_respects_probs(self):
        """With heavily skewed probs, the high-prob token should dominate."""
        token_ids = np.array([10, 20, 30])
        probs = np.array([0.9, 0.05, 0.05])
        ds = RandomVocabDataset(
            num_chunks=1, seq_len=1000, token_ids=token_ids,
            token_probs=probs, seed=0,
        )
        sample = ds[0]["input_ids"].numpy()
        n_token_10 = int((sample == 10).sum())
        # Expect ~900 of 1000; allow generous slack.
        assert n_token_10 > 800

    def test_deterministic(self):
        token_ids = np.arange(50)
        ds1 = RandomVocabDataset(5, 32, token_ids, seed=42)
        ds2 = RandomVocabDataset(5, 32, token_ids, seed=42)
        for idx in range(5):
            assert np.array_equal(
                ds1[idx]["input_ids"].numpy(),
                ds2[idx]["input_ids"].numpy(),
            )


# ----------------------------------------------------------------------
# Frequency computation.
# ----------------------------------------------------------------------
class TestFrequencies:
    def test_counts_sum_to_total_tokens(self, synthetic_held_out):
        vocab_size = 128
        counts = compute_token_frequencies(synthetic_held_out, vocab_size)
        total = sum(int(synthetic_held_out[i]["input_ids"].numel())
                    for i in range(len(synthetic_held_out)))
        assert counts.sum() == total

    def test_topk_returns_correct_shape(self, synthetic_held_out):
        counts = compute_token_frequencies(synthetic_held_out, 128)
        ids, probs = topk_vocabulary(counts, top_k=10)
        assert ids.shape == (10,)
        assert probs.shape == (10,)
        # Probs sum to 1 over the slice.
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_topk_descending(self):
        counts = np.array([5, 1, 100, 50, 0, 75])
        ids, probs = topk_vocabulary(counts, top_k=3)
        # Top 3 should be indices 2, 5, 3 in that order.
        assert ids.tolist() == [2, 5, 3]


# ----------------------------------------------------------------------
# Combined loaders.
# ----------------------------------------------------------------------
class TestLoaders:
    def test_three_loaders_built(self, synthetic_held_out, configs):
        model_cfg, train_cfg = configs
        loaders = make_input_distribution_loaders(
            held_out_dataset=synthetic_held_out,
            train_cfg=train_cfg, model_cfg=model_cfg,
            seed=0, num_workers=0,
        )
        assert set(loaders.keys()) == {"real", "shuffled", "random"}

    def test_loader_batch_shape(self, synthetic_held_out, configs):
        model_cfg, train_cfg = configs
        loaders = make_input_distribution_loaders(
            held_out_dataset=synthetic_held_out,
            train_cfg=train_cfg, model_cfg=model_cfg,
            seed=0, num_workers=0,
        )
        for name, loader in loaders.items():
            batch = next(iter(loader))
            ids = batch["input_ids"]
            assert ids.dtype == torch.int64, name
            B, T = ids.shape
            assert B <= train_cfg.eval_batch_size, name
            assert T == train_cfg.train_seq_len, name

    def test_loader_input_dist_names_consistent(self):
        # Sanity check that the constants match.
        assert set(INPUT_DIST_NAMES) == set(FLOW_SUBDIR_FOR_INPUT.keys())

    def test_real_and_shuffled_same_tokens(self, synthetic_held_out, configs):
        """The shuffled loader's tokens should be a permutation of the
        real loader's tokens at each batch position."""
        model_cfg, train_cfg = configs
        loaders = make_input_distribution_loaders(
            held_out_dataset=synthetic_held_out,
            train_cfg=train_cfg, model_cfg=model_cfg,
            seed=0, num_workers=0,
        )
        real_batches = list(loaders["real"])
        shuf_batches = list(loaders["shuffled"])
        for rb, sb in zip(real_batches, shuf_batches):
            r = rb["input_ids"].numpy()
            s = sb["input_ids"].numpy()
            assert r.shape == s.shape
            for i in range(r.shape[0]):
                assert sorted(r[i].tolist()) == sorted(s[i].tolist())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
