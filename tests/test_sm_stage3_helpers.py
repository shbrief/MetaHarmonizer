"""Tests for stage3_matchers pure-Python helpers — _merge_top_k, _is_treatment_column."""
import pytest

from src.models.schema_mapper.matchers.stage3_matchers import (
    _merge_top_k,
    _is_treatment_column,
)


class TestMergeTopK:
    def test_non_overlapping_all_kept(self):
        std = [("field_a", 0.9, ""), ("field_b", 0.8, "")]
        alias = [("field_c", 0.7, "src1")]
        result = _merge_top_k(std, alias, top_k=5)
        fields = [r[0] for r in result]
        assert "field_a" in fields
        assert "field_b" in fields
        assert "field_c" in fields

    def test_overlapping_alias_higher_score_wins(self):
        std = [("field_a", 0.5, "")]
        alias = [("field_a", 0.9, "src1")]
        result = _merge_top_k(std, alias, top_k=5)
        assert len(result) == 1
        assert result[0] == ("field_a", 0.9, "src1")

    def test_overlapping_std_higher_score_wins(self):
        std = [("field_a", 0.95, "")]
        alias = [("field_a", 0.5, "src1")]
        result = _merge_top_k(std, alias, top_k=5)
        assert len(result) == 1
        assert result[0][1] == 0.95

    def test_top_k_truncation(self):
        std = [(f"field_{i}", float(i) / 10, "") for i in range(6)]
        result = _merge_top_k(std, [], top_k=3)
        assert len(result) == 3

    def test_sorted_by_score_descending(self):
        std = [("b", 0.3, ""), ("a", 0.9, ""), ("c", 0.6, "")]
        result = _merge_top_k(std, [], top_k=5)
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_both_empty(self):
        assert _merge_top_k([], [], top_k=5) == []

    def test_empty_alias(self):
        std = [("field_a", 0.9, "")]
        result = _merge_top_k(std, [], top_k=5)
        assert result == [("field_a", 0.9, "")]

    def test_empty_std(self):
        alias = [("field_a", 0.9, "src1")]
        result = _merge_top_k([], alias, top_k=5)
        assert result == [("field_a", 0.9, "src1")]

    def test_top_k_zero_returns_empty(self):
        std = [("field_a", 0.9, "")]
        assert _merge_top_k(std, [], top_k=0) == []

    def test_no_duplicates_in_output(self):
        std = [("field_a", 0.9, ""), ("field_b", 0.7, "")]
        alias = [("field_a", 0.8, "src"), ("field_b", 0.6, "src")]
        result = _merge_top_k(std, alias, top_k=10)
        fields = [r[0] for r in result]
        assert len(fields) == len(set(fields))


class TestIsTreatmentColumn:
    def test_tx_prefix(self):
        assert _is_treatment_column("tx_response") is True

    def test_tx_suffix(self):
        assert _is_treatment_column("response_tx") is True

    def test_tx_middle(self):
        assert _is_treatment_column("pre_tx_dose") is True

    def test_tx_standalone_not_matched(self):
        # "tx" embedded inside a word (no _tx_ boundary) must not trigger
        assert _is_treatment_column("context") is False

    def test_treatment_keywords_matched(self):
        # One representative per TREATMENT_KEYWORDS_SUBSTRING branch is enough
        for col in ("hormone_therapy", "treatment_type", "drug_dose",
                    "chemo_agent", "surgery_type", "radiation_field",
                    "regimen_name"):
            assert _is_treatment_column(col) is True, f"{col!r} should be treatment"

    def test_unrelated_columns_not_matched(self):
        for col in ("patient_age", "sex", "tumor_size"):
            assert _is_treatment_column(col) is False, f"{col!r} should not be treatment"
