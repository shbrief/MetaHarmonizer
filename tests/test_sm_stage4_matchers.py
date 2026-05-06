"""Tests for stage4_matchers — LLMMatcher (Gemini fallback).

Bypasses __init__ (requires GEMINI_API_KEY) via __new__ + stub injection.
Covers: _build_prompt content, match() JSON parsing and validation logic.
"""
import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from metaharmonizer.models.schema_mapper.matchers.stage4_matchers import LLMMatcher


# ── Stub helpers ───────────────────────────────────────────────────────────────

def _make_engine(top_k=3, standard_fields=None):
    engine = MagicMock()
    engine.top_k = top_k
    engine.standard_fields = standard_fields or [
        "age_at_diagnosis", "sex", "tumor_size", "treatment_type", "sample_type"
    ]
    engine.unique_values = MagicMock(return_value=["val1", "val2"])
    return engine


def _make_llm(top_k=3, standard_fields=None):
    """Build LLMMatcher without calling __init__ (no API key needed)."""
    engine = _make_engine(top_k=top_k, standard_fields=standard_fields)
    m = LLMMatcher.__new__(LLMMatcher)
    m.engine = engine
    m.model = MagicMock()
    m.model_name = "models/gemini-stub"
    return m


def _resp(text):
    """Wrap a string as a fake generate_content response."""
    return SimpleNamespace(text=text)


# ── _build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_contains_col_name(self):
        m = _make_llm()
        prompt = m._build_prompt("patient_age")
        assert "patient_age" in prompt

    def test_contains_standard_fields(self):
        m = _make_llm()
        prompt = m._build_prompt("age")
        for field in m.engine.standard_fields:
            assert field in prompt

    def test_sample_values_included_when_provided(self):
        m = _make_llm()
        prompt = m._build_prompt("sex", sample_values=["Male", "Female"])
        assert "Male" in prompt
        assert "Female" in prompt

    def test_no_sample_values_section_when_omitted(self):
        m = _make_llm()
        prompt = m._build_prompt("sex", sample_values=None)
        assert "Sample Values" not in prompt


# ── match — happy path ─────────────────────────────────────────────────────────

class TestLLMMatcherMatch:
    def test_valid_response_returns_tuples(self):
        m = _make_llm()
        payload = json.dumps([
            {"field": "age_at_diagnosis", "confidence": 0.95},
            {"field": "sex",              "confidence": 0.80},
        ])
        m.model.generate_content.return_value = _resp(payload)
        result = m.match("patient_age")
        assert len(result) == 2
        assert result[0] == ("age_at_diagnosis", 0.95, "llm")

    def test_results_sorted_by_confidence_descending(self):
        m = _make_llm()
        payload = json.dumps([
            {"field": "sex",              "confidence": 0.60},
            {"field": "age_at_diagnosis", "confidence": 0.90},
        ])
        m.model.generate_content.return_value = _resp(payload)
        result = m.match("age")
        scores = [r[1] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_truncated_to_top_k(self):
        m = _make_llm(top_k=2)
        payload = json.dumps([
            {"field": "age_at_diagnosis", "confidence": 0.95},
            {"field": "sex",              "confidence": 0.80},
            {"field": "tumor_size",       "confidence": 0.70},
        ])
        m.model.generate_content.return_value = _resp(payload)
        result = m.match("age")
        assert len(result) <= 2

    def test_markdown_wrapped_json_extracted(self):
        m = _make_llm()
        payload = json.dumps([{"field": "sex", "confidence": 0.9}])
        wrapped = f"```json\n{payload}\n```"
        m.model.generate_content.return_value = _resp(wrapped)
        result = m.match("gender")
        assert len(result) == 1
        assert result[0][0] == "sex"

    def test_empty_array_response_returns_empty(self):
        m = _make_llm()
        m.model.generate_content.return_value = _resp("[]")
        assert m.match("xyz") == []


# ── match — validation / error paths ──────────────────────────────────────────

    def test_unknown_field_skipped(self):
        m = _make_llm()
        payload = json.dumps([
            {"field": "not_a_real_field", "confidence": 0.99},
            {"field": "sex",              "confidence": 0.80},
        ])
        m.model.generate_content.return_value = _resp(payload)
        result = m.match("gender")
        fields = [r[0] for r in result]
        assert "not_a_real_field" not in fields
        assert "sex" in fields

    def test_confidence_above_1_skipped(self):
        m = _make_llm()
        payload = json.dumps([{"field": "sex", "confidence": 1.5}])
        m.model.generate_content.return_value = _resp(payload)
        assert m.match("gender") == []

    def test_non_numeric_confidence_skipped(self):
        m = _make_llm()
        payload = json.dumps([{"field": "sex", "confidence": "high"}])
        m.model.generate_content.return_value = _resp(payload)
        assert m.match("gender") == []

    def test_json_decode_error_returns_empty(self):
        m = _make_llm()
        m.model.generate_content.return_value = _resp("not valid json {{")
        assert m.match("col") == []

    def test_api_exception_returns_empty(self):
        m = _make_llm()
        m.model.generate_content.side_effect = RuntimeError("API down")
        assert m.match("col") == []
