"""Tests for SemanticClusteringEngine — both EMBEDDING and LLM modes."""

import numpy as np
import pytest

from src.field_suggester.semantic_clustering import (
    SCSMode,
    SemanticCluster,
    SemanticClusteringConfig,
    SemanticClusteringEngine,
)


# ── Mock LLM clients ────────────────────────────────────────────────


def _mock_llm_high_scs(prompt: str) -> str:
    return (
        '{"score": 0.95, "category": "demographics", '
        '"reasoning": "All items are demographic fields."}'
    )


def _mock_llm_low_scs(prompt: str) -> str:
    return (
        '{"score": 0.3, "category": null, '
        '"reasoning": "Items are heterogeneous."}'
    )


def _mock_llm_adaptive(prompt: str) -> str:
    """Returns high SCS for small item lists, low SCS for large ones."""
    count = prompt.count("- ")
    if count <= 5:
        return (
            '{"score": 0.95, "category": "test", '
            '"reasoning": "Small coherent group."}'
        )
    return (
        '{"score": 0.4, "category": null, '
        '"reasoning": "Too diverse."}'
    )


# ── Shared helpers ───────────────────────────────────────────────────


def _tight_embeddings(n: int, dim: int = 64) -> np.ndarray:
    """Embeddings that are very close together (high cosine similarity)."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal((1, dim))
    return base + rng.standard_normal((n, dim)) * 0.01


def _two_group_embeddings(n_per_group: int, dim: int = 64) -> np.ndarray:
    """Two clearly separated groups."""
    rng = np.random.default_rng(42)
    g1 = rng.standard_normal((n_per_group, dim)) + np.full(dim, 5.0)
    g2 = rng.standard_normal((n_per_group, dim)) + np.full(dim, -5.0)
    return np.vstack([g1, g2])


# ── Tests: Embedding mode (no LLM) ──────────────────────────────────


class TestEmbeddingSCSMode:
    """Tests using SCSMode.EMBEDDING — no API calls."""

    def test_tight_cluster_high_scs(self):
        """Tight embeddings should produce SCS >= threshold → single cluster."""
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.EMBEDDING, scs_threshold=0.9
        )
        engine = SemanticClusteringEngine(config=config)
        items = ["age", "sex", "gender", "race", "ethnicity"]
        embeddings = _tight_embeddings(5)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) == 1
        assert set(clusters[0].items) == set(items)
        assert clusters[0].scs >= 0.9

    def test_separated_groups_split(self):
        """Two distant groups should split into multiple clusters."""
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.EMBEDDING, scs_threshold=0.9, min_cluster_size=1
        )
        engine = SemanticClusteringEngine(config=config)
        items = ["age", "sex", "race", "tumor_size", "stage", "grade"]
        embeddings = _two_group_embeddings(3)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) >= 2

    def test_no_llm_calls(self):
        """Embedding mode must make zero LLM calls."""
        config = SemanticClusteringConfig(scs_mode=SCSMode.EMBEDDING)
        engine = SemanticClusteringEngine(config=config)
        rng = np.random.default_rng(42)
        items = [f"field_{i}" for i in range(20)]
        embeddings = rng.standard_normal((20, 64))

        engine.cluster(items, embeddings)
        assert engine._api_call_count == 0

    def test_centroid_cosine_metric(self):
        """centroid_cosine metric should work and produce valid SCS."""
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.EMBEDDING,
            embedding_scs_metric="centroid_cosine",
            scs_threshold=0.9,
        )
        engine = SemanticClusteringEngine(config=config)
        items = ["a", "b", "c"]
        embeddings = _tight_embeddings(3)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) >= 1
        for c in clusters:
            assert 0.0 <= c.scs <= 1.0

    def test_unknown_metric_falls_back(self):
        """Unknown metric string should fall back to cosine without error."""
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.EMBEDDING,
            embedding_scs_metric="unknown_metric",
            scs_threshold=0.9,
        )
        engine = SemanticClusteringEngine(config=config)
        items = ["a", "b", "c"]
        embeddings = _tight_embeddings(3)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) >= 1


# ── Tests: LLM mode ─────────────────────────────────────────────────


class TestLLMSCSMode:
    """Tests using SCSMode.LLM with mock clients."""

    def test_high_scs_single_cluster(self):
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.LLM, scs_threshold=0.9
        )
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_high_scs
        )
        items = ["age", "sex", "gender", "race", "ethnicity"]
        embeddings = _tight_embeddings(5)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) == 1
        assert set(clusters[0].items) == set(items)

    def test_low_scs_splits(self):
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.LLM, scs_threshold=0.9, min_cluster_size=1
        )
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_adaptive
        )
        items = ["age", "sex", "race", "tumor_size", "stage", "grade"]
        embeddings = _two_group_embeddings(3)

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) >= 2

    def test_api_call_count_incremented(self):
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.LLM, scs_threshold=0.9
        )
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_high_scs
        )
        items = ["a", "b", "c"]
        embeddings = _tight_embeddings(3)
        engine.cluster(items, embeddings)
        assert engine._api_call_count >= 1

    def test_stratified_sampling_large_input(self):
        config = SemanticClusteringConfig(
            scs_mode=SCSMode.LLM,
            scs_threshold=0.9,
            sample_size=10,
            kmeans_n_clusters=3,
        )
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_high_scs
        )
        rng = np.random.default_rng(42)
        items = [f"field_{i}" for i in range(50)]
        embeddings = rng.standard_normal((50, 64))

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) >= 1


# ── Tests: Shared behaviour (mode-agnostic) ──────────────────────────


class TestSharedBehavior:
    """Tests that should pass regardless of SCS mode."""

    @pytest.mark.parametrize("mode", [SCSMode.EMBEDDING, SCSMode.LLM])
    def test_single_item_returns_one_cluster(self, mode):
        config = SemanticClusteringConfig(scs_mode=mode, min_cluster_size=2)
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_high_scs
        )
        rng = np.random.default_rng(42)
        items = ["age_at_diagnosis"]
        embeddings = rng.standard_normal((1, 64))

        clusters = engine.cluster(items, embeddings)
        assert len(clusters) == 1
        assert clusters[0].items == ["age_at_diagnosis"]

    @pytest.mark.parametrize("mode", [SCSMode.EMBEDDING, SCSMode.LLM])
    def test_all_items_accounted_for(self, mode):
        config = SemanticClusteringConfig(
            scs_mode=mode, scs_threshold=0.9, min_cluster_size=1
        )
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_adaptive
        )
        rng = np.random.default_rng(42)
        items = [f"field_{i}" for i in range(20)]
        embeddings = rng.standard_normal((20, 64))

        clusters = engine.cluster(items, embeddings)
        all_items = []
        for c in clusters:
            all_items.extend(c.items)
        assert sorted(all_items) == sorted(items)

    @pytest.mark.parametrize("mode", [SCSMode.EMBEDDING, SCSMode.LLM])
    def test_mismatch_raises(self, mode):
        config = SemanticClusteringConfig(scs_mode=mode)
        engine = SemanticClusteringEngine(
            config=config, llm_client=_mock_llm_high_scs
        )
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Length mismatch"):
            engine.cluster(["a", "b"], rng.standard_normal((3, 64)))


# ── Tests: SCS response parsing ──────────────────────────────────────


class TestSCSParsing:
    """Tests for _parse_scs_response."""

    def test_valid_json(self):
        engine = SemanticClusteringEngine()
        score = engine._parse_scs_response(
            '{"score": 0.85, "category": "test", "reasoning": "ok"}'
        )
        assert score == 0.85

    def test_fallback_float_extraction(self):
        engine = SemanticClusteringEngine()
        score = engine._parse_scs_response(
            "The score is 0.72 based on my analysis."
        )
        assert score == pytest.approx(0.72)

    def test_unparseable_returns_zero(self):
        engine = SemanticClusteringEngine()
        score = engine._parse_scs_response("no numbers here")
        assert score == 0.0


# ── Tests: SemanticCluster dataclass ─────────────────────────────────


class TestSemanticCluster:
    """Smoke test for SemanticCluster dataclass."""

    def test_fields(self):
        sc = SemanticCluster(
            items=["a", "b"], indices=[0, 1], scs=0.95, depth=2
        )
        assert sc.items == ["a", "b"]
        assert sc.indices == [0, 1]
        assert sc.scs == 0.95
        assert sc.depth == 2
