"""Tests for OntoMapST — SentenceTransformer (mean-pooling) strategy.

ST uses a full SentenceTransformer model (mean pooling), whereas LM uses a
raw AutoModel with CLS-token pooling via EmbeddingAdapter.

get_match_results also L2-normalises query embeddings before the FAISS search,
which LM skips. That implementation difference is verified here alongside the
shared output contract (columns, match_level, prod mode).
"""
import numpy as np
import pytest

from src.models.ontology_mapper_st import OntoMapST


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _LoggerStub:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _CapturingIndex:
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


def _make_st(query=None, corpus=None, topk=3):
    query = query or ["LUAD"]
    corpus = corpus or _CORPUS
    m = OntoMapST.__new__(OntoMapST)
    m.logger = _LoggerStub()
    m.query = query
    m.corpus = corpus
    m.topk = topk
    m._vs = type("VS", (), {"index": _CapturingIndex(len(corpus))})()
    rng = np.random.default_rng(1)
    m.create_embeddings = lambda lst, convert_to_tensor=False: (
        rng.standard_normal((len(lst), _DIM)).astype("float32").tolist()
    )
    return m


# ── get_match_results ──────────────────────────────────────────────────────────

class TestOntoMapSTGetMatchResults:
    def test_query_is_l2_normalised_before_search(self):
        """ST normalises query to unit vectors — the defining difference from LM."""
        m = _make_st(query=["LUAD"])
        raw = np.array([[3.0, 4.0, 0.0, 0.0]], dtype="float32")  # norm = 5
        m.create_embeddings = lambda lst, convert_to_tensor=False: raw.tolist()
        m.get_match_results(cura_map={}, topk=2, test_or_prod="prod")
        norms = np.linalg.norm(m._vs.index.last_mat, axis=1)
        np.testing.assert_allclose(norms, np.ones(1), atol=1e-5)

    def test_output_has_expected_columns(self):
        m = _make_st(topk=3)
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=3)
        for col in ("original_value", "curated_ontology", "match_level",
                    "match1", "match1_score", "match3", "match3_score"):
            assert col in df.columns

    def test_match_level_hit(self):
        m = _make_st(query=["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=3)
        assert df.loc[0, "match_level"] == 1

    def test_match_level_miss(self):
        m = _make_st(query=["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Not In Corpus"}, topk=3)
        assert df.loc[0, "match_level"] == 99

    def test_prod_mode_curated_not_available(self):
        m = _make_st()
        df = m.get_match_results(cura_map=None, topk=3, test_or_prod="prod")
        assert df.loc[0, "curated_ontology"] == "Not Available for Prod Environment"
