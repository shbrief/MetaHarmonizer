"""Tests for EmbeddingStore — caching, deduplication, and disk persistence."""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.utils.embedding_store import EmbeddingRecord, EmbeddingStore


# ── Helpers ──────────────────────────────────────────────────────────


class _FakeModel:
    """Minimal SentenceTransformer stand-in for tests."""

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.encode_call_count = 0

    def encode(self, texts, batch_size=64, show_progress_bar=False):
        self.encode_call_count += 1
        rng = np.random.default_rng(hash(tuple(texts)) % (2**31))
        return rng.standard_normal((len(texts), self.dim)).astype(np.float32)


@pytest.fixture
def store_with_fake_model():
    """EmbeddingStore with a fake model injected (no HF download)."""
    s = EmbeddingStore(model_name="test-model", strategy="st")
    fake = _FakeModel()
    s._model = fake  # skip lazy loading
    return s, fake


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary directory for disk persistence tests."""
    d = tmp_path / "emb_cache"
    d.mkdir()
    return d


# ── Tests: Basic embed() ────────────────────────────────────────────


class TestEmbedBasic:
    """Core embed() functionality."""

    def test_returns_correct_shape(self, store_with_fake_model):
        store, _ = store_with_fake_model
        texts = ["age", "sex", "tumor_size"]
        vecs = store.embed(texts, source="column_name")
        assert vecs.shape == (3, 32)

    def test_empty_input(self, store_with_fake_model):
        store, _ = store_with_fake_model
        vecs = store.embed([], source="test")
        assert vecs.shape == (0, 0)

    def test_single_text(self, store_with_fake_model):
        store, _ = store_with_fake_model
        vecs = store.embed(["hello"], source="test")
        assert vecs.shape == (1, 32)

    def test_size_property(self, store_with_fake_model):
        store, _ = store_with_fake_model
        assert store.size == 0
        store.embed(["a", "b", "c"], source="test")
        assert store.size == 3


# ── Tests: Cache behaviour ──────────────────────────────────────────


class TestCacheBehaviour:
    """Caching, deduplication, and cache hits."""

    def test_second_call_is_cache_hit(self, store_with_fake_model):
        store, fake = store_with_fake_model
        texts = ["age", "sex"]
        v1 = store.embed(texts, source="col")
        count_after_first = fake.encode_call_count

        v2 = store.embed(texts, source="col")
        # No new encode call
        assert fake.encode_call_count == count_after_first
        np.testing.assert_array_equal(v1, v2)

    def test_different_source_same_text_is_cache_hit(self, store_with_fake_model):
        store, fake = store_with_fake_model
        store.embed(["tumor_size"], source="column_name")
        count = fake.encode_call_count

        store.embed(["tumor_size"], source="schema_field")
        assert fake.encode_call_count == count

    def test_partial_cache_hit(self, store_with_fake_model):
        store, fake = store_with_fake_model
        store.embed(["age", "sex"], source="col")
        first_count = fake.encode_call_count

        # "age" cached, "tumor_size" not
        store.embed(["age", "tumor_size"], source="col")
        assert fake.encode_call_count == first_count + 1
        assert store.size == 3

    def test_clear(self, store_with_fake_model):
        store, _ = store_with_fake_model
        store.embed(["a", "b"], source="test")
        assert store.size == 2
        store.clear()
        assert store.size == 0


# ── Tests: get_cached_* methods ──────────────────────────────────────


class TestCachedAccessors:
    """get_cached_texts and get_cached_vectors."""

    def test_get_cached_texts_all(self, store_with_fake_model):
        store, _ = store_with_fake_model
        store.embed(["a", "b"], source="col")
        store.embed(["c"], source="field")
        texts = store.get_cached_texts()
        assert len(texts) == 3

    def test_get_cached_texts_by_source(self, store_with_fake_model):
        store, _ = store_with_fake_model
        store.embed(["a", "b"], source="col")
        store.embed(["c"], source="field")
        assert len(store.get_cached_texts(source="col")) == 2
        assert len(store.get_cached_texts(source="field")) == 1

    def test_get_cached_vectors_shape(self, store_with_fake_model):
        store, _ = store_with_fake_model
        store.embed(["x", "y", "z"], source="test")
        vecs = store.get_cached_vectors()
        assert vecs.shape == (3, 32)

    def test_get_cached_vectors_empty(self, store_with_fake_model):
        store, _ = store_with_fake_model
        vecs = store.get_cached_vectors()
        assert vecs.shape == (0, 0)


# ── Tests: Disk persistence ─────────────────────────────────────────


class TestDiskPersistence:
    """save() and load() round-trip."""

    def test_save_creates_files(self, store_with_fake_model, tmp_cache_dir):
        store, _ = store_with_fake_model
        store.embed(["alpha", "beta"], source="test")
        store.save(tmp_cache_dir)

        assert (tmp_cache_dir / "vectors.npy").exists()
        assert (tmp_cache_dir / "metadata.csv").exists()

    def test_save_load_round_trip(self, store_with_fake_model, tmp_cache_dir):
        store, _ = store_with_fake_model
        store.embed(["alpha", "beta", "gamma"], source="test")
        original_vecs = store.get_cached_vectors()
        store.save(tmp_cache_dir)

        # Fresh store, load from disk
        store2 = EmbeddingStore(model_name="test-model", strategy="st")
        store2._model = _FakeModel()
        loaded = store2.load(tmp_cache_dir)

        assert loaded == 3
        assert store2.size == 3
        loaded_vecs = store2.get_cached_vectors()
        np.testing.assert_array_almost_equal(
            np.sort(original_vecs, axis=0),
            np.sort(loaded_vecs, axis=0),
        )

    def test_load_nonexistent_returns_zero(self, store_with_fake_model, tmp_path):
        store, _ = store_with_fake_model
        loaded = store.load(tmp_path / "does_not_exist")
        assert loaded == 0

    def test_load_skips_duplicates(self, store_with_fake_model, tmp_cache_dir):
        store, _ = store_with_fake_model
        store.embed(["alpha"], source="test")
        store.save(tmp_cache_dir)

        # Load into same store (already has "alpha")
        loaded = store.load(tmp_cache_dir)
        assert loaded == 0
        assert store.size == 1

    def test_save_empty_cache(self, store_with_fake_model, tmp_cache_dir):
        store, _ = store_with_fake_model
        result = store.save(tmp_cache_dir)
        assert result == tmp_cache_dir
        # No files created for empty cache
        assert not (tmp_cache_dir / "vectors.npy").exists()

    def test_save_no_path_raises(self, store_with_fake_model):
        store, _ = store_with_fake_model
        store.embed(["x"], source="test")
        with pytest.raises(ValueError, match="No cache_dir"):
            store.save()

    def test_load_no_path_raises(self, store_with_fake_model):
        store, _ = store_with_fake_model
        with pytest.raises(ValueError, match="No cache_dir"):
            store.load()


# ── Tests: Lazy model loading ────────────────────────────────────────


class TestLazyLoading:
    """Model is NOT loaded at construction time."""

    def test_model_is_none_at_init(self):
        store = EmbeddingStore(model_name="test-model", strategy="st")
        assert store._model is None

    def test_model_loaded_on_first_embed(self):
        store = EmbeddingStore(model_name="test-model", strategy="st")
        fake = _FakeModel()
        store._model = fake
        store.embed(["hello"], source="test")
        assert fake.encode_call_count >= 1


# ── Tests: Cache key ─────────────────────────────────────────────────


class TestCacheKey:
    """Cache key generation."""

    def test_same_text_same_key(self):
        store = EmbeddingStore(model_name="m1")
        k1 = store._cache_key("hello world")
        k2 = store._cache_key("hello world")
        assert k1 == k2

    def test_different_model_different_key(self):
        s1 = EmbeddingStore(model_name="m1")
        s2 = EmbeddingStore(model_name="m2")
        k1 = s1._cache_key("hello")
        k2 = s2._cache_key("hello")
        assert k1 != k2

    def test_key_includes_model_prefix(self):
        store = EmbeddingStore(model_name="my-model")
        key = store._cache_key("test")
        assert key.startswith("my-model::")
