"""Tests for FieldSuggester – hybrid NER + embedding clustering."""
import pytest
import pandas as pd
import numpy as np

from src.field_suggester import FieldSuggester


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def suggester():
    """FieldSuggester with default settings (NER may or may not be available)."""
    return FieldSuggester()


@pytest.fixture
def ajcc_columns():
    """Typical AJCC staging columns that should cluster together."""
    return [
        "AJCC Clinical Group Stage",
        "AJCC Clinical M-Stage",
        "AJCC Clinical N-Stage",
        "AJCC Clinical T-Stage",
        "AJCC Pathologic M-Stage",
        "AJCC Pathologic N-Stage",
        "AJCC Pathologic Stage",
        "AJCC Pathologic T-Stage",
    ]


@pytest.fixture
def mixed_columns():
    """Mixed columns from several domains."""
    return [
        "AJCC Clinical Group Stage",
        "AJCC Clinical T-Stage",
        "AJCC Pathologic T-Stage",
        "AJCC Pathologic Stage",
        "Patient Blood Pressure Systolic",
        "Patient Blood Pressure Diastolic",
        "Mean Arterial Pressure",
        "Tumor Location Primary",
        "Tumor Location Metastatic",
        "Site of Tumor",
    ]


@pytest.fixture
def sample_df(mixed_columns):
    """DataFrame with mixed columns and dummy values."""
    rng = np.random.default_rng(42)
    data = {}
    for col in mixed_columns:
        if "Stage" in col:
            data[col] = rng.choice(
                ["Stage I", "Stage II", "Stage III", "Stage IV", None],
                size=50,
            )
        elif "Pressure" in col or "Arterial" in col:
            data[col] = rng.uniform(60, 180, size=50).round(1)
        else:
            data[col] = rng.choice(
                ["Lung", "Liver", "Brain", "Bone", None], size=50
            )
    return pd.DataFrame(data)


# ======================================================================
# Tests: Basic Functionality
# ======================================================================


class TestSuggestBasic:
    """Basic suggest() behaviour."""

    def test_empty_input_returns_empty(self, suggester):
        result = suggester.suggest([])
        assert result == {}

    def test_single_column_below_min_cluster_size(self, suggester):
        result = suggester.suggest(["lonely_column"])
        # min_cluster_size=2 by default, so a single column should not form a group
        assert len(result) == 0

    def test_two_similar_columns_form_group(self, suggester):
        cols = ["AJCC Clinical T-Stage", "AJCC Pathologic T-Stage"]
        result = suggester.suggest(cols)
        assert len(result) >= 1
        # Both columns should be in the same suggested field
        all_sources = []
        for info in result.values():
            all_sources.extend(info["source_columns"])
        assert set(cols).issubset(set(all_sources))

    def test_duplicate_columns_deduplicated(self, suggester):
        cols = ["AJCC Clinical T-Stage", "AJCC Clinical T-Stage", "AJCC Pathologic T-Stage"]
        result = suggester.suggest(cols)
        all_sources = []
        for info in result.values():
            all_sources.extend(info["source_columns"])
        # No duplicates in output
        assert len(all_sources) == len(set(all_sources))


# ======================================================================
# Tests: AJCC Clustering
# ======================================================================


class TestAJCCClustering:
    """AJCC staging columns should cluster into one or very few groups."""

    def test_ajcc_columns_cluster_together(self, suggester, ajcc_columns):
        result = suggester.suggest(ajcc_columns)
        assert len(result) >= 1

        # All AJCC columns should appear somewhere in results
        all_sources = set()
        for info in result.values():
            all_sources.update(info["source_columns"])
        assert set(ajcc_columns) == all_sources

    def test_ajcc_suggested_name_contains_ajcc_or_stage(
        self, suggester, ajcc_columns
    ):
        result = suggester.suggest(ajcc_columns)
        names = list(result.keys())
        # At least one name should reference "ajcc" or "stage"
        combined = " ".join(names).lower()
        assert "ajcc" in combined or "stage" in combined

    def test_ajcc_confidence_above_threshold(self, suggester, ajcc_columns):
        result = suggester.suggest(ajcc_columns)
        for info in result.values():
            assert info["confidence"] >= 0.3


# ======================================================================
# Tests: Mixed Domain Separation
# ======================================================================


class TestMixedDomains:
    """Columns from different domains should separate into distinct groups."""

    def test_multiple_groups_formed(self, suggester, mixed_columns):
        result = suggester.suggest(mixed_columns)
        # We expect at least 2 distinct groups (staging vs pressure vs tumor location)
        assert len(result) >= 2

    def test_staging_and_pressure_not_mixed(self, suggester, mixed_columns):
        result = suggester.suggest(mixed_columns)
        staging_cols = {"AJCC Clinical Group Stage", "AJCC Clinical T-Stage",
                        "AJCC Pathologic T-Stage", "AJCC Pathologic Stage"}
        pressure_cols = {"Patient Blood Pressure Systolic",
                         "Patient Blood Pressure Diastolic",
                         "Mean Arterial Pressure"}

        for info in result.values():
            members = set(info["source_columns"])
            # No group should contain both staging and pressure columns
            has_staging = bool(members & staging_cols)
            has_pressure = bool(members & pressure_cols)
            assert not (has_staging and has_pressure), (
                f"Staging and pressure columns mixed: {members}"
            )


# ======================================================================
# Tests: DataFrame Integration
# ======================================================================


class TestWithDataFrame:
    """Tests that pass a DataFrame for value enrichment."""

    def test_sample_values_included(self, suggester, mixed_columns, sample_df):
        result = suggester.suggest(mixed_columns, df=sample_df)
        for info in result.values():
            assert "sample_values" in info
            for col, vals in info["sample_values"].items():
                assert isinstance(vals, list)

    def test_suggest_to_df_returns_dataframe(
        self, suggester, mixed_columns, sample_df
    ):
        df_result = suggester.suggest_to_df(mixed_columns, df=sample_df)
        assert isinstance(df_result, pd.DataFrame)
        expected_cols = {"suggested_field", "source_column", "ner_entities", "confidence"}
        assert expected_cols == set(df_result.columns)

    def test_suggest_to_df_no_empty_field_names(
        self, suggester, mixed_columns, sample_df
    ):
        df_result = suggester.suggest_to_df(mixed_columns, df=sample_df)
        if not df_result.empty:
            assert df_result["suggested_field"].str.len().min() > 0


# ======================================================================
# Tests: Output Structure
# ======================================================================


class TestOutputStructure:
    """Validate the structure of suggest() output."""

    def test_output_keys(self, suggester, ajcc_columns):
        result = suggester.suggest(ajcc_columns)
        for name, info in result.items():
            assert isinstance(name, str)
            assert "source_columns" in info
            assert "ner_entities" in info
            assert "confidence" in info
            assert isinstance(info["source_columns"], list)
            assert isinstance(info["ner_entities"], list)
            assert isinstance(info["confidence"], float)

    def test_confidence_in_range(self, suggester, ajcc_columns):
        result = suggester.suggest(ajcc_columns)
        for info in result.values():
            assert 0.0 <= info["confidence"] <= 1.0

    def test_results_sorted_by_confidence_desc(self, suggester, mixed_columns):
        result = suggester.suggest(mixed_columns)
        confidences = [info["confidence"] for info in result.values()]
        assert confidences == sorted(confidences, reverse=True)

    def test_no_column_appears_twice(self, suggester, mixed_columns):
        result = suggester.suggest(mixed_columns)
        all_sources = []
        for info in result.values():
            all_sources.extend(info["source_columns"])
        assert len(all_sources) == len(set(all_sources)), (
            "A column appears in multiple suggested fields"
        )


# ======================================================================
# Tests: Edge Cases
# ======================================================================


class TestEdgeCases:
    """Edge cases and robustness."""

    def test_numeric_column_names(self, suggester):
        result = suggester.suggest(["123", "456", "789"])
        # Should not crash; may or may not form groups
        assert isinstance(result, dict)

    def test_very_long_column_names(self, suggester):
        long_cols = [f"very_long_column_name_{'x' * 200}_{i}" for i in range(5)]
        result = suggester.suggest(long_cols)
        assert isinstance(result, dict)

    def test_special_characters_in_names(self, suggester):
        cols = [
            "stage (AJCC 7th ed.)",
            "stage [AJCC 8th ed.]",
            "stage/grade",
        ]
        result = suggester.suggest(cols)
        assert isinstance(result, dict)

    def test_min_cluster_size_respected(self):
        suggester = FieldSuggester(min_cluster_size=5)
        # Only 3 similar columns – should not meet threshold
        cols = ["AJCC T-Stage", "AJCC N-Stage", "AJCC M-Stage"]
        result = suggester.suggest(cols)
        assert len(result) == 0

    def test_custom_distance_threshold(self):
        # Very small threshold → many tiny clusters (likely filtered out)
        strict = FieldSuggester(distance_threshold=0.05)
        cols = ["AJCC Clinical T-Stage", "Total Protein Level", "Date of Birth"]
        result = strict.suggest(cols)
        assert isinstance(result, dict)


# ======================================================================
# Tests: suggest_from_identifier (integration placeholder)
# ======================================================================


class TestSuggestFromIdentifier:
    """Test the convenience wrapper that takes SchemaIdentifier results."""

    def test_all_mapped_returns_empty(self, suggester):
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        identifier_results = {
            "col_a": [("sex", 0.9)],
            "col_b": [("age", 0.8)],
        }
        result = suggester.suggest_from_mapping_results(df, identifier_results)
        assert result == {}

    def test_unmapped_columns_extracted(self, suggester):
        df = pd.DataFrame({
            "col_a": [1],
            "AJCC Clinical T-Stage": ["Stage I"],
            "AJCC Pathologic T-Stage": ["Stage II"],
        })
        identifier_results = {
            "col_a": [("sex", 0.9)],
            "AJCC Clinical T-Stage": [],
            "AJCC Pathologic T-Stage": [],
        }
        result = suggester.suggest_from_mapping_results(df, identifier_results)
        # The two AJCC columns should be suggested
        all_sources = set()
        for info in result.values():
            all_sources.update(info["source_columns"])
        assert "AJCC Clinical T-Stage" in all_sources
        assert "AJCC Pathologic T-Stage" in all_sources
        assert "col_a" not in all_sources


# ======================================================================
# Tests: value-enriched embeddings
# ======================================================================


@pytest.fixture
def vital_status_df():
    """DataFrame with two semantically equivalent categorical columns
    and one numeric column that should NOT be enriched."""
    return pd.DataFrame({
        "vital_status":    ["alive", "dead"] * 10,
        "os_status":       ["alive", "dead"] * 10,
        "age_at_diagnosis": [45, 67, 52, 71] * 5,
    })


class TestValueEnrichedEmbedding:
    """Tests for embed_top_k parameter and _build_embed_text."""

    def test_build_embed_text_no_df_returns_col_text(self):
        text = FieldSuggester._build_embed_text("vital_status", None, embed_top_k=5)
        assert text == "vital status"

    def test_build_embed_text_embed_top_k_zero_returns_col_text(self, vital_status_df):
        text = FieldSuggester._build_embed_text("vital_status", vital_status_df, embed_top_k=0)
        assert text == "vital status"

    def test_build_embed_text_categorical_includes_values(self, vital_status_df):
        text = FieldSuggester._build_embed_text("vital_status", vital_status_df, embed_top_k=5)
        assert text.startswith("vital status:")
        assert "alive" in text
        assert "dead" in text

    def test_build_embed_text_numeric_column_skipped(self, vital_status_df):
        text = FieldSuggester._build_embed_text("age_at_diagnosis", vital_status_df, embed_top_k=5)
        assert text == "age at diagnosis"
        assert ":" not in text

    def test_build_embed_text_top_k_respected(self, vital_status_df):
        # vital_status has 2 unique values; requesting top 1 should return only 1
        text = FieldSuggester._build_embed_text("vital_status", vital_status_df, embed_top_k=1)
        values_part = text.split(": ")[1]
        assert len(values_part.split(", ")) == 1

    def test_build_embed_text_values_are_sorted(self):
        df = pd.DataFrame({"status": ["z_val", "a_val", "m_val"]})
        text = FieldSuggester._build_embed_text("status", df, embed_top_k=3)
        values_part = text.split(": ")[1]
        vals = values_part.split(", ")
        assert vals == sorted(vals)

    def test_build_embed_text_value_truncated_at_30_chars(self):
        long_val = "x" * 100
        df = pd.DataFrame({"col": [long_val]})
        text = FieldSuggester._build_embed_text("col", df, embed_top_k=5)
        values_part = text.split(": ")[1]
        for val in values_part.split(", "):
            assert len(val) <= 30

    def test_embed_top_k_zero_no_df_same_result(self, vital_status_df):
        text_no_df = FieldSuggester._build_embed_text("vital_status", None, 0)
        text_zero_k = FieldSuggester._build_embed_text("vital_status", vital_status_df, 0)
        assert text_no_df == text_zero_k

    def test_custom_embed_top_k_stored_on_instance(self):
        s = FieldSuggester(embed_top_k=10)
        assert s.embed_top_k == 10

    def test_suggest_without_df_unchanged(self, suggester):
        """When df=None, value enrichment path is never taken."""
        cols = ["AJCC Clinical T-Stage", "AJCC Pathologic T-Stage"]
        result = suggester.suggest(cols)
        assert isinstance(result, dict)

    def test_vital_status_columns_present_in_output(self, vital_status_df):
        """Both semantically equivalent columns must appear somewhere in output."""
        cols = ["vital_status", "os_status"]
        s = FieldSuggester(embed_top_k=5)
        result = s.suggest(cols, df=vital_status_df)
        all_sources: set = set()
        for info in result.values():
            all_sources.update(info["source_columns"])
        assert "vital_status" in all_sources
        assert "os_status" in all_sources
