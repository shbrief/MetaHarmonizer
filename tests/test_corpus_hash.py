"""Tests for corpus content hashing utility."""

import pandas as pd
import pytest
from metaharmonizer.utils.corpus_hash import compute_corpus_hash


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "clean_code": ["C1234", "C5678", "C9012"],
        "official_label": ["Lung Cancer", "Breast Cancer", "Melanoma"],
    })


class TestComputeCorpusHash:

    def test_returns_8_char_hex(self, sample_df):
        h = compute_corpus_hash(sample_df)
        assert len(h) == 8
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self, sample_df):
        assert compute_corpus_hash(sample_df) == compute_corpus_hash(sample_df)

    def test_row_order_independent(self, sample_df):
        reversed_df = sample_df.iloc[::-1].reset_index(drop=True)
        assert compute_corpus_hash(sample_df) == compute_corpus_hash(reversed_df)

    def test_different_content_different_hash(self, sample_df):
        other_df = pd.DataFrame({
            "clean_code": ["C1234", "C5678", "C9999"],
            "official_label": ["Lung Cancer", "Breast Cancer", "Leukemia"],
        })
        assert compute_corpus_hash(sample_df) != compute_corpus_hash(other_df)

    def test_custom_length(self, sample_df):
        h = compute_corpus_hash(sample_df, n=16)
        assert len(h) == 16

    def test_label_change_changes_hash(self, sample_df):
        changed_df = sample_df.copy()
        changed_df.loc[0, "official_label"] = "Lung Adenocarcinoma"
        assert compute_corpus_hash(sample_df) != compute_corpus_hash(changed_df)
