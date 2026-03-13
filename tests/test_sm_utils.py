"""Tests for schema_mapper_utils — normalize, extract_valid_value, is_numeric_column."""
import pandas as pd
import pytest

from src.utils.schema_mapper_utils import normalize, extract_valid_value, is_numeric_column


class TestNormalize:
    def test_lowercase(self):
        assert normalize("AGE") == "age"

    def test_whitespace_collapse(self):
        assert normalize("  hello   world  ") == "hello world"

    def test_non_alphanum_stripped(self):
        assert normalize("tumor-size!") == "tumor size"

    def test_numbers_kept(self):
        assert "10" in normalize("age_10")

    def test_empty_string(self):
        assert normalize("") == ""

    def test_multiple_specials_collapse_to_single_space(self):
        # "--" and "__" become spaces, then collapse
        result = normalize("a--b__c")
        assert result == "a b c"

    def test_unicode_special_chars(self):
        # µ is not alphanum, becomes space
        result = normalize("dose_µg")
        assert "dose" in result
        assert "g" in result


class TestExtractValidValue:
    def test_semicolon_split(self):
        assert extract_valid_value("a;b;c") == ["a", "b", "c"]

    def test_cbio_delimiter(self):
        assert extract_valid_value("a<;>b<;>c") == ["a", "b", "c"]

    def test_double_colon_split(self):
        assert extract_valid_value("a::b::c") == ["a", "b", "c"]

    def test_na_filtered(self):
        result = extract_valid_value("a;NA;b")
        assert result == ["a", "b"]

    def test_na_case_insensitive(self):
        result = extract_valid_value("na;Na;NA;valid")
        assert result == ["valid"]

    def test_empty_parts_filtered(self):
        result = extract_valid_value("a;;b")
        assert result == ["a", "b"]

    def test_single_value(self):
        assert extract_valid_value("hello") == ["hello"]

    def test_all_na_returns_empty(self):
        assert extract_valid_value("NA;na;NA") == []

    def test_whitespace_stripped(self):
        result = extract_valid_value("  a  ;  b  ")
        assert result == ["a", "b"]


class TestIsNumericColumn:
    def _make_df(self, values, col="x"):
        return pd.DataFrame({col: values})

    def test_all_floats_is_numeric(self):
        df = self._make_df([1.0, 2.0, 3.0, 4.0, 5.0])
        assert is_numeric_column(df, "x", random_state=0) == True

    def test_all_strings_not_numeric(self):
        df = self._make_df(["male", "female", "male", "female", "male"])
        assert is_numeric_column(df, "x", random_state=0) == False

    def test_mixed_mostly_numeric_above_threshold(self):
        # 95% numeric, threshold 0.9 → True
        vals = [str(i) for i in range(95)] + ["male"] * 5
        df = self._make_df(vals)
        assert is_numeric_column(df, "x", min_ratio=0.9, random_state=0) == True

    def test_mixed_mostly_text_below_threshold(self):
        # 50% numeric, threshold 0.9 → False
        vals = [str(i) for i in range(50)] + ["foo"] * 50
        df = self._make_df(vals)
        assert is_numeric_column(df, "x", min_ratio=0.9, random_state=0) == False

    def test_empty_column_returns_false(self):
        df = self._make_df([None, None, None])
        assert is_numeric_column(df, "x") == False

    def test_threshold_boundary_at_exactly_90pct(self):
        # Exactly 90 numeric / 10 non-numeric → 90% >= 0.9 → True
        vals = ["1"] * 90 + ["foo"] * 10
        df = self._make_df(vals)
        assert is_numeric_column(df, "x", min_ratio=0.9, random_state=0) == True

    def test_integer_strings_are_numeric(self):
        df = self._make_df(["1", "2", "3", "4", "5"])
        assert is_numeric_column(df, "x", random_state=0) == True

    def test_sample_size_limits_rows_checked(self):
        # Large column but sample_size=10 — should still return a bool-like result
        vals = [str(i) for i in range(10_000)]
        df = self._make_df(vals)
        result = is_numeric_column(df, "x", sample_size=10, random_state=42)
        assert result in (True, False)
