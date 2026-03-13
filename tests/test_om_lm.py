"""Tests for OntoMapLM — CLS-pooled language model strategy.

Covers get_match_results output contract and the key distinction from ST:
LM does NOT L2-normalise query embeddings before FAISS search.
"""
import numpy as np
import pytest

from src.models.ontology_mapper_lm import OntoMapLM


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _LoggerStub:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _CapturingIndex:
    """FAISS index stub that records what matrix was passed to search."""
    def __init__(self, n_corpus):
        self.last_mat = None
        self._n = n_corpus

    def search(self, mat, k):
        self.last_mat = mat.copy()
        n = mat.shape[0]
        I = np.tile(np.arange(min(k, self._n), dtype=np.int64), (n, 1))
        D = np.ones((n, k), dtype=np.float32) * 0.9
        return D, I


_CORPUS = ["Lung Adenocarcinoma", "Lung Cancer", "Breast Cancer"]
_DIM = 4


def _make_lm(query=None, corpus=None, topk=3):
    query = query or ["LUAD"]
    corpus = corpus or _CORPUS
    m = OntoMapLM.__new__(OntoMapLM)
    m.logger = _LoggerStub()
    m.query = query
    m.corpus = corpus
    m.topk = topk
    m._vs = type("VS", (), {"index": _CapturingIndex(len(corpus))})()
    rng = np.random.default_rng(0)
    # Stable fake embeddings: each call returns reproducible non-unit vectors
    m.create_embeddings = lambda lst, convert_to_tensor=False: (
        rng.standard_normal((len(lst), _DIM)).astype("float32").tolist()
    )
    return m


# ── get_match_results ──────────────────────────────────────────────────────────

class TestOntoMapLMGetMatchResults:
    def test_output_has_expected_columns(self):
        m = _make_lm(topk=3)
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=3)
        for col in ("original_value", "curated_ontology", "match_level",
                    "match1", "match1_score", "match3", "match3_score"):
            assert col in df.columns

    def test_one_row_per_query(self):
        m = _make_lm(query=["LUAD", "BRCA"], topk=2)
        df = m.get_match_results(cura_map={}, topk=2)
        assert len(df) == 2

    def test_match_level_hit(self):
        # _CapturingIndex always returns indices [0, 1, 2] → corpus[0] = "Lung Adenocarcinoma"
        m = _make_lm(query=["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=3)
        assert df.loc[0, "match_level"] == 1

    def test_match_level_miss(self):
        m = _make_lm(query=["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Not In Corpus"}, topk=3)
        assert df.loc[0, "match_level"] == 99

    def test_prod_mode_curated_not_available(self):
        m = _make_lm()
        df = m.get_match_results(cura_map=None, topk=3, test_or_prod="prod")
        assert df.loc[0, "curated_ontology"] == "Not Available for Prod Environment"

    def test_query_not_l2_normalised(self):
        """LM passes raw (non-unit) embeddings to FAISS — no normalisation."""
        m = _make_lm(query=["LUAD"])
        raw = np.array([[3.0, 4.0, 0.0, 0.0]], dtype="float32")  # norm = 5, not 1
        m.create_embeddings = lambda lst, convert_to_tensor=False: raw.tolist()
        m.get_match_results(cura_map={}, topk=2, test_or_prod="prod")
        norms = np.linalg.norm(m._vs.index.last_mat, axis=1)
        assert norms[0] > 1.1  # still the original scale, not normalised
