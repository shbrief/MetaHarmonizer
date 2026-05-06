"""Tests for stage2_matchers — ValueDictMatcher, OntologyMatcher."""
import math
import pytest
import torch
from unittest.mock import MagicMock

from metaharmonizer.models.schema_mapper.matchers.stage2_matchers import (
    ValueDictMatcher,
    OntologyMatcher,
)
from metaharmonizer.models.schema_mapper.config import VALUE_UNIQUE_CAP, VALUE_PERCENTAGE_THRESH


# ── ValueDictMatcher helpers ───────────────────────────────────────────────────

_DIM = 4


def _unit(values):
    """Return a (len(values), _DIM) tensor of random L2-normalised rows."""
    t = torch.randn(len(values), _DIM)
    return torch.nn.functional.normalize(t, p=2, dim=1)


def _make_vd_engine(
    unique_values,
    value_fields_list,
    corpus_embs,
    query_embs=None,
    frequencies=None,
    top_k=3,
):
    """
    Build a minimal mock engine for ValueDictMatcher.

    corpus_embs  : (N, _DIM) tensor — the pre-built value_embs in the engine
    query_embs   : (M, _DIM) tensor — what dict_model.encode() returns
                   defaults to the same vectors as corpus_embs[:M]
    """
    engine = MagicMock()
    engine.top_k = top_k
    engine.value_texts = ["dummy"]          # non-None / non-empty guard
    engine.value_embs = corpus_embs
    engine.value_fields_list = value_fields_list
    engine.unique_values.return_value = unique_values
    engine.value_frequencies.return_value = (
        frequencies or {v: 1.0 / len(unique_values) for v in unique_values}
    )
    q = query_embs if query_embs is not None else corpus_embs[: len(unique_values)]
    engine.dict_model.encode.return_value = q
    return engine


# ── ValueDictMatcher ───────────────────────────────────────────────────────────

class TestValueDictMatcher:
    def test_no_value_texts_returns_empty(self):
        engine = _make_vd_engine(["Male"], [["sex"]], _unit(["Male"]))
        engine.value_texts = None
        assert ValueDictMatcher(engine).match("sex") == []

    def test_no_value_embs_returns_empty(self):
        engine = _make_vd_engine(["Male"], [["sex"]], _unit(["Male"]))
        engine.value_embs = None
        assert ValueDictMatcher(engine).match("sex") == []

    def test_empty_unique_values_returns_empty(self):
        engine = _make_vd_engine([], [], _unit([]))
        engine.unique_values.return_value = []
        assert ValueDictMatcher(engine).match("sex") == []

    def test_too_many_unique_values_returns_empty(self):
        many = [str(i) for i in range(VALUE_UNIQUE_CAP + 1)]
        corpus = _unit(many)
        engine = _make_vd_engine(many, [["sex"]] * len(many), corpus)
        assert ValueDictMatcher(engine).match("sex") == []

    def test_high_similarity_returns_field(self):
        # query vector == corpus vector → cosine sim = 1.0 > VALUE_DICT_THRESH
        vec = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), p=2, dim=1)
        engine = _make_vd_engine(
            unique_values=["Male"],
            value_fields_list=[["sex"]],
            corpus_embs=vec,
            query_embs=vec,
        )
        result = ValueDictMatcher(engine).match("sex")
        assert any(r[0] == "sex" for r in result)

    def test_orthogonal_vectors_return_empty(self):
        # cosine sim = 0 < VALUE_DICT_THRESH → no matches
        q = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), p=2, dim=1)
        c = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), p=2, dim=1)
        engine = _make_vd_engine(
            unique_values=["Male"],
            value_fields_list=[["sex"]],
            corpus_embs=c,
            query_embs=q,
        )
        assert ValueDictMatcher(engine).match("sex") == []

    def test_results_sorted_by_proportion_descending(self):
        # Two corpus entries: field_a appears in both; field_b only in one
        # → field_a gets higher proportion
        vec_a = torch.nn.functional.normalize(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), p=2, dim=1)
        vec_b = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0, 0.0]]), p=2, dim=1)
        corpus = torch.cat([vec_a, vec_b])
        # Two query values, both close to vec_a
        q = torch.cat([vec_a, vec_a])
        engine = _make_vd_engine(
            unique_values=["v1", "v2"],
            value_fields_list=[["field_a"], ["field_b"]],
            corpus_embs=corpus,
            query_embs=q,
        )
        result = ValueDictMatcher(engine).match("col")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)


# ── OntologyMatcher helpers ────────────────────────────────────────────────────

def _make_onto_engine(unique_values, hits, frequencies=None, top_k=5):
    engine = MagicMock()
    engine.top_k = top_k
    engine.unique_values.return_value = unique_values
    n = max(len(unique_values), 1)
    engine.value_frequencies.return_value = (
        frequencies or {v: 1.0 / n for v in unique_values}
    )
    engine.nci_client.map_value_to_schema.return_value = hits
    return engine


# ── OntologyMatcher ────────────────────────────────────────────────────────────

class TestOntologyMatcher:
    def test_no_hits_returns_empty(self):
        engine = _make_onto_engine(["Male", "Female"], hits={})
        assert OntologyMatcher(engine).match("sex") == []

    def test_too_many_unique_values_returns_empty(self):
        many = [str(i) for i in range(VALUE_UNIQUE_CAP + 1)]
        engine = _make_onto_engine(many, hits={"sex": ["Male"]})
        assert OntologyMatcher(engine).match("sex") == []

    def test_field_above_threshold_included(self):
        # Both values matched → proportion = 1.0 > VALUE_PERCENTAGE_THRESH
        engine = _make_onto_engine(
            ["Male", "Female"],
            hits={"sex": ["Male", "Female"]},
        )
        result = OntologyMatcher(engine).match("sex")
        assert any(r[0] == "sex" for r in result)

    def test_field_below_threshold_filtered(self):
        # 10 unique values; only 1 matched → proportion ≈ 0.1 < VALUE_PERCENTAGE_THRESH
        many = [str(i) for i in range(10)]
        engine = _make_onto_engine(many, hits={"sex": [many[0]]})
        result = OntologyMatcher(engine).match("col")
        assert result == []

    def test_col_name_match_adds_bonus(self):
        # Proportion from data alone would be < threshold, but col-name bonus lifts it
        engine = _make_onto_engine(
            ["v1", "v2", "v3", "v4", "v5", "v6", "v7"],
            hits={"sex": ["sex"]},   # only the col name matched
        )
        result = OntologyMatcher(engine).match("sex")
        assert any(r[0] == "sex" for r in result)

    def test_results_sorted_by_proportion_descending(self):
        engine = _make_onto_engine(
            ["Male", "Female", "Lung", "Breast"],
            hits={
                "sex":         ["Male", "Female"],
                "cancer_site": ["Lung"],
            },
        )
        result = OntologyMatcher(engine).match("col")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_cancer_type_match_adds_disease(self):
        engine = _make_onto_engine(
            ["Breast Cancer", "Lung Cancer"],
            hits={"cancer_type": ["Breast Cancer", "Lung Cancer"]},
        )
        result = OntologyMatcher(engine).match("cancer")
        fields = [r[0] for r in result]
        assert "cancer_type" in fields
        assert "disease" in fields

    def test_cancer_type_match_adds_details(self):
        engine = _make_onto_engine(
            ["Breast Cancer", "Lung Cancer"],
            hits={"cancer_type": ["Breast Cancer", "Lung Cancer"]},
        )
        result = OntologyMatcher(engine).match("cancer")
        fields = [r[0] for r in result]
        assert "cancer_type_details" in fields
