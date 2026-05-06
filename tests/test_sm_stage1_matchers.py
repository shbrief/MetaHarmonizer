"""Tests for stage1_matchers — StandardExactMatcher, AliasExactMatcher, StandardFuzzyMatcher, AliasFuzzyMatcher."""
import pytest
from unittest.mock import MagicMock

from metaharmonizer.models.schema_mapper.matchers.stage1_matchers import (
    StandardExactMatcher,
    AliasExactMatcher,
    StandardFuzzyMatcher,
    AliasFuzzyMatcher,
)
from metaharmonizer.utils.schema_mapper_utils import normalize


def _make_engine(has_alias: bool = False, top_k: int = 3):
    """Build a minimal mock engine for stage1 matcher tests."""
    std_fields = ["patient_age", "sample_type", "tumor_size", "sex"]
    std_normed = [normalize(f) for f in std_fields]
    normed_to_std = dict(zip(std_normed, std_fields))

    engine = MagicMock()
    engine.standard_fields = std_fields
    engine.standard_fields_normed = std_normed
    engine.normed_std_to_std = normed_to_std
    engine.top_k = top_k
    engine.has_alias_dict = has_alias

    # Alias attributes (used when has_alias=True)
    alias_raw = ["age", "tumour size", "sample type"]
    alias_normed = [normalize(k) for k in alias_raw]
    engine.sources_keys = alias_normed
    engine.sources_to_fields = {
        normalize("age"): ["patient_age"],
        normalize("tumour size"): ["tumor_size"],
        normalize("sample type"): ["sample_type"],
    }
    engine.normed_source_to_source = {
        normalize("age"): "GEO",
        normalize("tumour size"): "TCGA",
        normalize("sample type"): "GDC",
    }
    return engine


# ── StandardExactMatcher ─────────────────────────────────────────────────────


class TestStandardExactMatcher:
    def test_exact_hit_score_1(self):
        engine = _make_engine()
        result = StandardExactMatcher(engine).match("patient_age")
        assert len(result) == 1
        assert result[0][0] == "patient_age"
        assert result[0][1] == 1.0

    def test_no_hit_returns_empty(self):
        engine = _make_engine()
        assert StandardExactMatcher(engine).match("nonexistent_xyz") == []

    def test_case_insensitive_via_normalize(self):
        engine = _make_engine()
        result = StandardExactMatcher(engine).match("PATIENT_AGE")
        assert len(result) == 1
        assert result[0][0] == "patient_age"

    def test_source_is_empty_string(self):
        engine = _make_engine()
        result = StandardExactMatcher(engine).match("sex")
        assert result[0][2] == ""

    def test_special_chars_normalized(self):
        # "tumor-size" normalizes to "tumor size" which matches "tumor_size" → "tumor size"
        engine = _make_engine()
        result = StandardExactMatcher(engine).match("tumor-size")
        assert len(result) == 1
        assert result[0][0] == "tumor_size"


# ── AliasExactMatcher ────────────────────────────────────────────────────────


class TestAliasExactMatcher:
    def test_no_alias_dict_returns_empty(self):
        engine = _make_engine(has_alias=False)
        assert AliasExactMatcher(engine).match("age") == []

    def test_alias_hit_score_1(self):
        engine = _make_engine(has_alias=True)
        result = AliasExactMatcher(engine).match("age")
        assert len(result) >= 1
        assert result[0][0] == "patient_age"
        assert result[0][1] == 1.0

    def test_alias_source_populated(self):
        engine = _make_engine(has_alias=True)
        result = AliasExactMatcher(engine).match("tumour size")
        assert len(result) >= 1
        assert result[0][2] != ""  # source should be set

    def test_no_match_returns_empty(self):
        engine = _make_engine(has_alias=True)
        assert AliasExactMatcher(engine).match("completely_unknown_field") == []

    def test_case_insensitive_alias(self):
        engine = _make_engine(has_alias=True)
        result = AliasExactMatcher(engine).match("SAMPLE TYPE")
        assert len(result) >= 1
        assert result[0][0] == "sample_type"


# ── StandardFuzzyMatcher ─────────────────────────────────────────────────────


class TestStandardFuzzyMatcher:
    def test_near_match_returns_result(self):
        engine = _make_engine(top_k=5)
        # "patient age" normalizes to exactly "patient age" == normalize("patient_age")
        result = StandardFuzzyMatcher(engine).match("patient age")
        assert len(result) > 0

    def test_result_scores_positive(self):
        engine = _make_engine(top_k=5)
        result = StandardFuzzyMatcher(engine).match("patient age")
        assert all(r[1] > 0 for r in result)

    def test_result_sorted_descending(self):
        engine = _make_engine(top_k=5)
        result = StandardFuzzyMatcher(engine).match("patient age")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_very_different_query_returns_empty(self):
        engine = _make_engine(top_k=3)
        # FUZZY_THRESH = 92; completely unrelated string should score below it
        result = StandardFuzzyMatcher(engine).match("zzz_xyz_qrst_uvw")
        assert result == []

    def test_max_results_bounded_by_top_k(self):
        engine = _make_engine(top_k=2)
        result = StandardFuzzyMatcher(engine).match("patient age")
        assert len(result) <= 2


# ── AliasFuzzyMatcher ────────────────────────────────────────────────────────


class TestAliasFuzzyMatcher:
    def test_no_alias_dict_returns_empty(self):
        engine = _make_engine(has_alias=False)
        assert AliasFuzzyMatcher(engine).match("age") == []

    def test_fuzzy_alias_hit(self):
        engine = _make_engine(has_alias=True, top_k=5)
        # "sample type" should fuzzy-match alias key normalize("sample type")
        result = AliasFuzzyMatcher(engine).match("sample type")
        assert len(result) >= 1

    def test_result_sorted_descending(self):
        engine = _make_engine(has_alias=True, top_k=5)
        result = AliasFuzzyMatcher(engine).match("age")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_no_dedup_duplicates(self):
        engine = _make_engine(has_alias=True, top_k=5)
        result = AliasFuzzyMatcher(engine).match("age")
        fields = [r[0] for r in result]
        assert len(fields) == len(set(fields))

    def test_empty_sources_returns_empty(self):
        engine = _make_engine(has_alias=True, top_k=3)
        engine.sources_keys = []
        assert AliasFuzzyMatcher(engine).match("age") == []
