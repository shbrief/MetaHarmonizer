"""Tests for OntoMapSynonym — NCI synonym dictionary strategy.

OntoMapSynonym does not extend OntoModelsBase and builds its result rows
independently, so match_level logic and min_score filtering are tested here.
"""
import pytest
import pandas as pd

from metaharmonizer.models.ontology_mapper_synonym import OntoMapSynonym


# ── Stubs ──────────────────────────────────────────────────────────────────────

class _LoggerStub:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


class _FakeSynDict:
    """Returns a fixed hit list per query, ignoring the actual term."""
    def __init__(self, hits_per_query):
        # hits_per_query: list of {"official_label": str, "score": float}
        self._hits = hits_per_query

    def search_many(self, queries, top_k):
        return [self._hits[:top_k] for _ in queries]


def _make_syn(query, corpus=None, topk=3, hits=None):
    corpus = corpus or ["Lung Adenocarcinoma", "Lung Cancer"]
    hits = hits if hits is not None else [
        {"official_label": "Lung Adenocarcinoma", "score": 0.95},
        {"official_label": "Lung Cancer",         "score": 0.80},
    ]
    m = OntoMapSynonym.__new__(OntoMapSynonym)
    m.logger = _LoggerStub()
    m.query = query
    m.corpus = corpus
    m.topk = topk
    m.syn_dict = _FakeSynDict(hits)
    return m


# ── get_match_results ──────────────────────────────────────────────────────────

class TestOntoMapSynGetMatchResults:
    def test_match_level_hit(self):
        m = _make_syn(["LUAD"])
        df = m.get_match_results(
            cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=2)
        assert df.loc[0, "match_level"] == 1

    def test_match_level_second_position(self):
        m = _make_syn(["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Lung Cancer"}, topk=2)
        assert df.loc[0, "match_level"] == 2

    def test_match_level_miss(self):
        m = _make_syn(["LUAD"])
        df = m.get_match_results(cura_map={"LUAD": "Not In Results"}, topk=2)
        assert df.loc[0, "match_level"] == 99

    def test_min_score_filters_low_score_matches(self):
        hits = [
            {"official_label": "Lung Adenocarcinoma", "score": 0.3},
        ]
        m = _make_syn(["LUAD"], hits=hits)
        df = m.get_match_results(
            cura_map={"LUAD": "Lung Adenocarcinoma"}, topk=1, min_score=0.5)
        # Score below min_score → term set to None → curated cannot be found
        assert df.loc[0, "match_level"] == 99

    def test_prod_mode_curated_not_available(self):
        m = _make_syn(["LUAD"])
        df = m.get_match_results(cura_map=None, topk=2, test_or_prod="prod")
        assert df.loc[0, "curated_ontology"] == "Not Available for Prod Environment"

    def test_output_has_topk_match_columns(self):
        m = _make_syn(["LUAD"], topk=2)
        df = m.get_match_results(cura_map={}, topk=2)
        for i in (1, 2):
            assert f"match{i}" in df.columns
            assert f"match{i}_score" in df.columns

    def test_one_row_per_query(self):
        m = _make_syn(["LUAD", "BRCA"])
        df = m.get_match_results(cura_map={}, topk=2)
        assert len(df) == 2
