"""Tests for OntoMapLLM — LLM query rewriting + FAISS re-search (Stage 4).

Covers: prompt building, JSON response parsing, FAISS re-search for LM/ST,
result DataFrame schema, context extraction from query_df, and error handling.
"""
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.models.ontology_mapper_llm import OntoMapLLM


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _LoggerStub:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def debug(self, *a, **k): pass
    def error(self, *a, **k): pass


class _FakeIndex:
    """FAISS index stub that returns predetermined results."""
    def __init__(self, n_vectors=100):
        self.ntotal = n_vectors

    def search(self, query_mat, k):
        n = query_mat.shape[0]
        I = np.tile(np.arange(k), (n, 1))
        D = np.tile(np.linspace(0.9, 0.5, k), (n, 1)).astype("float32")
        return D, I


class _FakeCursor:
    """SQLite cursor stub."""
    def __init__(self, terms):
        self._terms = terms

    def execute(self, sql, params=None):
        if params:
            db_id = params[0]
            if db_id < len(self._terms):
                return SimpleNamespace(fetchone=lambda: (self._terms[db_id],))
            return SimpleNamespace(fetchone=lambda: None)
        # Bulk query: SELECT id, term FROM ...
        return SimpleNamespace(
            fetchall=lambda: [(i, t) for i, t in enumerate(self._terms)]
        )


class _FakeVS:
    """Vector store stub for S2 model."""
    def __init__(self, terms):
        self.index = _FakeIndex(len(terms))
        self._ids = list(range(len(terms)))
        self.table_name = "test_table"
        self.cursor = _FakeCursor(terms)


class _FakeEmbedder:
    """Embedding model stub."""
    def embed_documents(self, texts):
        return [np.random.randn(128).astype("float32").tolist() for _ in texts]

    def encode(self, texts, convert_to_tensor=False, device=None):
        return np.random.randn(len(texts), 128).astype("float32")


CORPUS_TERMS = [
    "Thymus Squamous Cell Carcinoma",
    "Thymus Neuroendocrine Carcinoma",
    "Lung Adenocarcinoma",
    "Breast Cancer",
    "Parietal Lobe",
]


def _make_llm(strategy='lm', query_df=None, term_col=None):
    """Create an OntoMapLLM instance bypassing __init__."""
    m = OntoMapLLM.__new__(OntoMapLLM)
    m.category = "disease"
    m.topk = 5
    m.max_retries = 3
    m.logger = _LoggerStub()
    m.query_df = query_df
    m._term_col = term_col

    # Fake S2 model
    s2 = SimpleNamespace()
    s2.om_strategy = strategy
    s2.model = _FakeEmbedder()
    s2.vector_store = _FakeVS(CORPUS_TERMS)
    m.s2_model = s2

    # Fake LLM model
    m._genai_model = MagicMock()
    m._llm_model_name = "test-model"

    return m


# ── Prompt building ───────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_query_and_category(self):
        m = _make_llm()
        prompt = m._build_prompt("TC:SCC", "")
        assert "TC:SCC" in prompt
        assert "disease" in prompt

    def test_contains_context_when_provided(self):
        m = _make_llm()
        prompt = m._build_prompt("TC:SCC", "CANCER_TYPE: THYMIC TUMOR")
        assert "CANCER_TYPE: THYMIC TUMOR" in prompt

    def test_no_context_block_when_empty(self):
        m = _make_llm()
        prompt = m._build_prompt("TC:SCC", "")
        assert "Clinical context:" not in prompt


# ── Response parsing ──────────────────────────────────────────────────────────

class TestParseResponse:
    def test_valid_json(self):
        m = _make_llm()
        text = '[{"term": "Thymus SCC", "reasoning": "abbreviation"}]'
        assert m._parse_response(text) == ["Thymus SCC"]

    def test_multiple_terms(self):
        m = _make_llm()
        text = '[{"term": "Term A", "reasoning": "x"}, {"term": "Term B", "reasoning": "y"}]'
        assert m._parse_response(text) == ["Term A", "Term B"]

    def test_markdown_fenced_json(self):
        m = _make_llm()
        text = '```json\n[{"term": "Parietal Lobe", "reasoning": "typo fix"}]\n```'
        assert m._parse_response(text) == ["Parietal Lobe"]

    def test_generic_code_fence(self):
        m = _make_llm()
        text = '```\n[{"term": "Diaphragm"}]\n```'
        assert m._parse_response(text) == ["Diaphragm"]

    def test_malformed_json_returns_empty(self):
        m = _make_llm()
        assert m._parse_response("not json at all") == []

    def test_non_list_returns_empty(self):
        m = _make_llm()
        assert m._parse_response('{"term": "single"}') == []

    def test_empty_term_filtered(self):
        m = _make_llm()
        text = '[{"term": "", "reasoning": "x"}, {"term": "Good Term"}]'
        assert m._parse_response(text) == ["Good Term"]

    def test_string_items_accepted(self):
        m = _make_llm()
        text = '["Term A", "Term B"]'
        assert m._parse_response(text) == ["Term A", "Term B"]


# ── FAISS re-search ──────────────────────────────────────────────────────────

class TestReSearch:
    def test_lm_strategy_no_normalization(self):
        m = _make_llm(strategy='lm')
        D, I = m._re_search(["test term"], topk=3)
        assert D.shape == (1, 3)
        assert I.shape == (1, 3)

    def test_st_strategy_with_normalization(self):
        m = _make_llm(strategy='st')
        D, I = m._re_search(["test term"], topk=3)
        assert D.shape == (1, 3)
        assert I.shape == (1, 3)

    def test_empty_terms_returns_empty(self):
        m = _make_llm()
        D, I = m._re_search([], topk=3)
        assert D.shape[0] == 0

    def test_multiple_terms(self):
        m = _make_llm()
        D, I = m._re_search(["term1", "term2", "term3"], topk=2)
        assert D.shape == (3, 2)

    def test_unsupported_strategy_raises(self):
        m = _make_llm(strategy='lm')
        m.s2_model.om_strategy = 'rag'
        with pytest.raises(ValueError, match="Unsupported"):
            m._re_search(["test"], topk=3)


# ── Index-to-term mapping ────────────────────────────────────────────────────

class TestFaissIndicesToTerms:
    def test_maps_indices_to_terms(self):
        m = _make_llm()
        results = m._faiss_indices_to_terms(
            np.array([0, 1, 2]),
            np.array([0.9, 0.8, 0.7])
        )
        assert len(results) == 3
        assert results[0][0] == "Thymus Squamous Cell Carcinoma"
        assert results[0][1] == pytest.approx(0.9)

    def test_skips_minus_one_indices(self):
        m = _make_llm()
        results = m._faiss_indices_to_terms(
            np.array([0, -1, 2]),
            np.array([0.9, 0.0, 0.7])
        )
        assert len(results) == 2


# ── get_match_results ─────────────────────────────────────────────────────────

class TestGetMatchResults:
    def test_output_has_expected_columns(self):
        m = _make_llm()
        m._genai_model.generate_content.return_value = SimpleNamespace(
            text='[{"term": "Thymus SCC", "reasoning": "abbrev"}]'
        )
        df = m.get_match_results(queries=["TC:SCC"], topk=3)
        for col in ("original_value",
                     "match1", "match1_score", "match2", "match2_score",
                     "match3", "match3_score"):
            assert col in df.columns
        # eval columns should NOT be present
        assert "curated_ontology" not in df.columns
        assert "match_level" not in df.columns

    def test_returns_faiss_matches(self):
        m = _make_llm()
        m._genai_model.generate_content.return_value = SimpleNamespace(
            text='[{"term": "Thymus Squamous Cell Carcinoma"}]'
        )
        df = m.get_match_results(queries=["TC:SCC"], topk=5)
        # match1 from FAISS should be a real corpus term (not N/A)
        assert df.loc[0, "match1"] != "N/A"

    def test_llm_failure_produces_empty_row(self):
        m = _make_llm()
        m._genai_model.generate_content.side_effect = Exception("API error")
        df = m.get_match_results(queries=["FAIL"], topk=3)
        assert len(df) == 1
        assert df.loc[0, "match1"] == "N/A"

    def test_multiple_queries(self):
        m = _make_llm()
        m._genai_model.generate_content.return_value = SimpleNamespace(
            text='[{"term": "Test Term"}]'
        )
        df = m.get_match_results(queries=["Q1", "Q2", "Q3"], topk=3)
        assert len(df) == 3


# ── Context extraction ────────────────────────────────────────────────────────

class TestContextExtraction:
    def test_extracts_context_from_query_df(self):
        query_df = pd.DataFrame({
            "original_value": ["TC:SCC", "PATIETAL"],
            "CANCER_TYPE": ["THYMIC TUMOR", "GLIOMA"],
            "BODY_SITE": ["THYMUS", "BRAIN"],
        })
        m = _make_llm(query_df=query_df, term_col="original_value")
        ctx = m._get_context_for_query("TC:SCC")
        assert "THYMIC TUMOR" in ctx
        assert "THYMUS" in ctx

    def test_no_query_df_returns_empty(self):
        m = _make_llm(query_df=None)
        ctx = m._get_context_for_query("TC:SCC")
        assert ctx == ""

    def test_query_not_in_df_returns_empty(self):
        query_df = pd.DataFrame({
            "original_value": ["OTHER"],
            "CANCER_TYPE": ["X"],
        })
        m = _make_llm(query_df=query_df, term_col="original_value")
        ctx = m._get_context_for_query("TC:SCC")
        assert ctx == ""

    def test_nan_values_excluded(self):
        query_df = pd.DataFrame({
            "original_value": ["TC:SCC"],
            "CANCER_TYPE": ["THYMIC TUMOR"],
            "NOTES": [float('nan')],
        })
        m = _make_llm(query_df=query_df, term_col="original_value")
        ctx = m._get_context_for_query("TC:SCC")
        assert "THYMIC TUMOR" in ctx
        assert "nan" not in ctx.lower()
