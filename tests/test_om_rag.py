"""Tests for OntoMapRAG — FAISS + optional cross-encoder reranker strategy.

Covers: basic retrieval path, reranker path (extra score columns),
_rerank_results reordering, and test-mode guard.
"""
import numpy as np
import pytest
from types import SimpleNamespace

from src.models.ontology_mapper_rag import OntoMapRAG


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _LoggerStub:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass


def _make_doc(term, score):
    d = SimpleNamespace()
    d.page_content = term
    d.metadata = {"term": term, "score": score}
    return d


class _FakeVS:
    """Vector store stub with a non-None index sentinel and recorded calls."""
    def __init__(self, docs):
        self.index = object()  # satisfies base-class property guard
        self.calls = []
        self._docs = docs

    def similarity_search(self, query, k=5, as_documents=False):
        self.calls.append((query, k))
        return self._docs[:k]


_DOCS = [
    _make_doc("Lung Adenocarcinoma", 0.95),
    _make_doc("Lung Cancer",         0.80),
    _make_doc("Breast Cancer",       0.60),
]


def _make_rag(query=None, use_reranker=False, topk=2, reranker_topk=3):
    query = query or ["LUAD"]
    m = OntoMapRAG.__new__(OntoMapRAG)
    m.logger = _LoggerStub()
    m.query = query
    m.corpus = ["Lung Adenocarcinoma", "Lung Cancer", "Breast Cancer"]
    m.topk = topk
    m.use_reranker = use_reranker
    m.reranker_method = "minilm" if use_reranker else None
    m.reranker_topk = reranker_topk if use_reranker else None
    m._reranker = None
    m._vs = _FakeVS(_DOCS)
    return m


# ── get_match_results — basic (no reranker) ────────────────────────────────────

class TestOntoMapRAGBasic:
    def test_similarity_search_called_with_topk(self):
        m = _make_rag(use_reranker=False, topk=2)
        m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        assert len(m._vs.calls) == 1
        _, k = m._vs.calls[0]
        assert k == 2

    def test_output_has_expected_columns(self):
        m = _make_rag(topk=2)
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        for col in ("original_value", "curated_ontology", "match_level",
                    "match1", "match1_score", "match2", "match2_score"):
            assert col in df.columns

    def test_match_level_hit(self):
        m = _make_rag(topk=2)
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        assert df.loc[0, "match_level"] == 1

    def test_match_level_miss(self):
        m = _make_rag(topk=2)
        df = m.get_match_results(cura_map={"LUAD": "Not In Corpus"}, topk=2)
        assert df.loc[0, "match_level"] == 99

    def test_prod_mode_curated_not_available(self):
        m = _make_rag(topk=2)
        df = m.get_match_results(cura_map=None, topk=2, test_or_prod="prod")
        assert df.loc[0, "curated_ontology"] == "Not Available for Prod Environment"

    def test_raises_in_test_mode_without_cura_map(self):
        m = _make_rag(topk=2)
        with pytest.raises(ValueError, match="cura_map"):
            m.get_match_results(cura_map=None, topk=2, test_or_prod="test")


# ── get_match_results — reranker path ─────────────────────────────────────────

class TestOntoMapRAGReranker:
    def test_similarity_search_called_with_reranker_topk(self):
        """With reranker enabled, retrieval uses reranker_topk (wider net)."""
        m = _make_rag(use_reranker=True, topk=2, reranker_topk=3)
        m._reranker = SimpleNamespace(
            predict=lambda pairs: np.array([0.9, 0.5, 0.7])
        )
        m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        _, k = m._vs.calls[0]
        assert k == 3

    def test_reranker_adds_extra_score_columns(self):
        m = _make_rag(use_reranker=True, topk=2, reranker_topk=3)
        m._reranker = SimpleNamespace(
            predict=lambda pairs: np.array([0.9, 0.5, 0.7])
        )
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        assert "match1_reranker_score" in df.columns
        assert "match1_similarity_score" in df.columns


# ── _rerank_results ────────────────────────────────────────────────────────────

class TestReRankResults:
    def test_reorders_by_reranker_score(self):
        m = _make_rag(use_reranker=True)
        candidates = [_make_doc("A", 0.9), _make_doc("B", 0.5), _make_doc("C", 0.7)]
        # C gets highest reranker score → should be first
        m._reranker = SimpleNamespace(
            predict=lambda pairs: np.array([0.1, 0.2, 0.8])
        )
        result = m._rerank_results("q", candidates, topk=2)
        assert result[0].page_content == "C"
        assert len(result) == 2

    def test_adds_reranker_score_to_metadata(self):
        m = _make_rag(use_reranker=True)
        candidates = [_make_doc("A", 0.9)]
        m._reranker = SimpleNamespace(predict=lambda pairs: np.array([0.75]))
        result = m._rerank_results("q", candidates, topk=1)
        assert result[0].metadata["reranker_score"] == pytest.approx(0.75, abs=1e-4)
        assert "similarity_score" in result[0].metadata

    def test_no_reranker_returns_topk_as_is(self):
        m = _make_rag(use_reranker=False)
        candidates = [_make_doc(f"Doc{i}", 1.0 - i * 0.1) for i in range(5)]
        result = m._rerank_results("q", candidates, topk=2)
        assert len(result) == 2
        assert result[0].page_content == "Doc0"
