"""Tests for stage3_matchers — NumericStandardMatcher, SemanticStandardMatcher,
NumericCombinedMatcher, SemanticCombinedMatcher, treatment_boost."""
import numpy as np
import pandas as pd
import pytest
import torch

from metaharmonizer.utils.schema_mapper_utils import normalize
from metaharmonizer.models.schema_mapper.matchers.stage3_matchers import (
    NumericCombinedMatcher,
    NumericStandardMatcher,
    SemanticCombinedMatcher,
    SemanticStandardMatcher,
    treatment_boost,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_DIM = 4


def _normalized_tensor(seed: int):
    """Return a (1, _DIM) L2-normalized float tensor."""
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.standard_normal((1, _DIM)), dtype=torch.float32)
    return t / t.norm().clamp(min=1e-8)


def _embedding_matrix(n: int, seed: int = 0):
    """Return an (n, _DIM) matrix of L2-normalized row vectors."""
    rng = np.random.default_rng(seed)
    t = torch.tensor(rng.standard_normal((n, _DIM)), dtype=torch.float32)
    norms = t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return t / norms


# ── Mock engine ───────────────────────────────────────────────────────────────


class _MockEngine3:
    """Minimal engine for stage3 matcher tests.

    Pre-sets all lazily-built caches so no real model is needed.
    """

    def __init__(self, has_alias: bool = False, numeric_cols=None, top_k: int = 3):
        self.top_k = top_k
        self.has_alias_dict = has_alias
        self._numeric_cols = set(numeric_cols or [])

        # Standard fields
        self.standard_fields = ["patient_age", "treatment_dose", "tumor_size", "sex"]
        self.standard_fields_normed = [normalize(f) for f in self.standard_fields]

        # Pre-built std field embeddings (skips dict_model.encode in SemanticStdMatcher)
        self._std_field_embs = _embedding_matrix(len(self.standard_fields), seed=1)

        # Curated DataFrame with treatment fields included
        self.curated_df = pd.DataFrame({
            "field_name": self.standard_fields + ["treatment_type"],
            "is_numeric_field": ["yes", "yes", "yes", "no", "no"],
        })

        # Pre-built numeric field embeddings (skips _ensure_std_numeric_index)
        self._std_numeric_fields = ["patient_age", "treatment_dose", "tumor_size"]
        self._std_numeric_embs = _embedding_matrix(len(self._std_numeric_fields), seed=2)

        # Alias setup
        if has_alias:
            self.sources_keys = [normalize("age"), normalize("tumor size")]
            self.sources_to_fields = {
                normalize("age"): ["patient_age"],
                normalize("tumor size"): ["tumor_size"],
            }
            self.alias_embs = _embedding_matrix(len(self.sources_keys), seed=3)
        else:
            self.alias_embs = None
            self.sources_keys = []
            self.sources_to_fields = {}

    def is_col_numeric(self, col: str) -> bool:
        return col in self._numeric_cols

    # no-ops: caches are pre-set in __init__
    def _ensure_std_numeric_index(self):
        pass

    def _ensure_std_field_embs(self):
        pass

    def _ensure_numeric_index(self):
        pass

    def _enc(self, text: str):
        seed = abs(hash(text)) % (2 ** 31)
        return _normalized_tensor(seed)


# ── NumericStandardMatcher ────────────────────────────────────────────────────


class TestNumericStandardMatcher:
    def test_non_numeric_col_returns_empty(self):
        engine = _MockEngine3(numeric_cols=[])
        assert NumericStandardMatcher(engine).match("sex") == []

    def test_numeric_col_returns_results(self):
        engine = _MockEngine3(numeric_cols=["dose_mg"])
        result = NumericStandardMatcher(engine).match("dose_mg")
        assert len(result) > 0

    def test_result_count_bounded_by_top_k(self):
        engine = _MockEngine3(numeric_cols=["age"], top_k=2)
        result = NumericStandardMatcher(engine).match("age")
        assert len(result) <= 2

    def test_result_fields_from_known_set(self):
        engine = _MockEngine3(numeric_cols=["dose"])
        result = NumericStandardMatcher(engine).match("dose")
        for field, _, _ in result:
            assert field in engine._std_numeric_fields

    def test_results_sorted_descending(self):
        engine = _MockEngine3(numeric_cols=["age"], top_k=3)
        result = NumericStandardMatcher(engine).match("age")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)


# ── SemanticStandardMatcher ───────────────────────────────────────────────────


class TestSemanticStandardMatcher:
    def test_returns_results(self):
        engine = _MockEngine3()
        result = SemanticStandardMatcher(engine).match("age")
        assert len(result) > 0

    def test_result_count_bounded_by_top_k(self):
        engine = _MockEngine3(top_k=2)
        result = SemanticStandardMatcher(engine).match("tumor")
        assert len(result) <= 2

    def test_results_sorted_descending(self):
        engine = _MockEngine3(top_k=4)
        result = SemanticStandardMatcher(engine).match("patient age")
        if len(result) > 1:
            scores = [r[1] for r in result]
            assert scores == sorted(scores, reverse=True)

    def test_field_names_from_standard_fields(self):
        engine = _MockEngine3(top_k=3)
        result = SemanticStandardMatcher(engine).match("age")
        for field, _, _ in result:
            assert field in engine.standard_fields

    def test_no_duplicates(self):
        engine = _MockEngine3(top_k=4)
        result = SemanticStandardMatcher(engine).match("dose")
        fields = [r[0] for r in result]
        assert len(fields) == len(set(fields))


# ── NumericCombinedMatcher ────────────────────────────────────────────────────


class TestNumericCombinedMatcher:
    def test_non_numeric_returns_empty(self):
        engine = _MockEngine3(has_alias=True, numeric_cols=[])
        assert NumericCombinedMatcher(engine).match("sex") == []

    def test_numeric_col_returns_results(self):
        engine = _MockEngine3(has_alias=False, numeric_cols=["dose"])
        result = NumericCombinedMatcher(engine).match("dose")
        assert len(result) > 0

    def test_no_duplicate_fields(self):
        engine = _MockEngine3(has_alias=False, numeric_cols=["age"])
        result = NumericCombinedMatcher(engine).match("age")
        fields = [r[0] for r in result]
        assert len(fields) == len(set(fields))

    def test_bounded_by_top_k(self):
        engine = _MockEngine3(has_alias=False, numeric_cols=["dose"], top_k=2)
        result = NumericCombinedMatcher(engine).match("dose")
        assert len(result) <= 2


# ── SemanticCombinedMatcher ───────────────────────────────────────────────────


class TestSemanticCombinedMatcher:
    def test_no_alias_uses_std_only(self):
        engine = _MockEngine3(has_alias=False)
        result = SemanticCombinedMatcher(engine).match("age")
        assert len(result) > 0

    def test_with_alias_returns_results(self):
        engine = _MockEngine3(has_alias=True)
        result = SemanticCombinedMatcher(engine).match("age")
        assert len(result) > 0

    def test_no_duplicate_fields(self):
        engine = _MockEngine3(has_alias=True, top_k=5)
        result = SemanticCombinedMatcher(engine).match("tumor")
        fields = [r[0] for r in result]
        assert len(fields) == len(set(fields))

    def test_bounded_by_top_k(self):
        engine = _MockEngine3(has_alias=True, top_k=2)
        result = SemanticCombinedMatcher(engine).match("age")
        assert len(result) <= 2


# ── treatment_boost ───────────────────────────────────────────────────────────


class TestTreatmentBoost:
    def test_tx_col_and_treatment_field_returns_boost(self):
        engine = _MockEngine3()
        # "tx_response" triggers tx_ pattern; "treatment_dose" is in curated_df
        assert treatment_boost("treatment_dose", "tx_response", engine) == 0.2

    def test_unrelated_col_returns_zero(self):
        engine = _MockEngine3()
        assert treatment_boost("patient_age", "patient_sex", engine) == 0.0

    def test_treatment_col_non_treatment_field_returns_zero(self):
        engine = _MockEngine3()
        # col is tx_ but "patient_age" is not a treatment field
        assert treatment_boost("patient_age", "tx_response", engine) == 0.0

    def test_treatment_type_field_with_therapy_col(self):
        engine = _MockEngine3()
        # "treatment_type" is in curated_df fields; "hormone_therapy" is a treatment col
        assert treatment_boost("treatment_type", "hormone_therapy", engine) == 0.2

    def test_caches_treatment_fields(self):
        engine = _MockEngine3()
        treatment_boost("treatment_dose", "tx_response", engine)
        assert hasattr(engine, "_treatment_fields_cache")
