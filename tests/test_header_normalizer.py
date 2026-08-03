"""Unit tests for source-header abbreviation expansion + double-matching (Issue #87).

Covers the util layer in schema_mapper_utils (expand_abbreviations / header_variants /
_load_abbrev_map degradation) and the double-match invariant at the matcher level
(a raw match is never dropped; expansion only adds candidates / raises scores).
"""
from unittest.mock import MagicMock

import pytest

from metaharmonizer.utils import schema_mapper_utils as u
from metaharmonizer.utils.schema_mapper_utils import (
    normalize,
    expand_abbreviations,
    header_variants,
)
from metaharmonizer.models.schema_mapper.matchers.stage1_matchers import (
    StandardExactMatcher,
    AliasExactMatcher,
)

# Real loader captured before the autouse fixture patches it (for the
# degradation test, which must exercise the real _load_abbrev_map body).
_REAL_LOAD = u._load_abbrev_map

# Small deterministic abbreviation map, independent of the bundled CSV.
_ABBREV = {"dx": "diagnosis", "tx": "treatment", "os": "overall survival",
           "her2": "human epidermal growth factor receptor 2"}


@pytest.fixture(autouse=True)
def _fixed_abbrev(monkeypatch):
    """Pin the abbreviation map (replacing the loader sidesteps the lru_cache)."""
    monkeypatch.setattr(u, "_load_abbrev_map", lambda: dict(_ABBREV))


# ── expand_abbreviations ─────────────────────────────────────────────────────

def test_expand_single_and_multiword():
    assert expand_abbreviations("age at dx") == "age at diagnosis"
    assert expand_abbreviations("os months") == "overall survival months"

def test_expand_unknown_passthrough():
    assert expand_abbreviations("mutation count") == "mutation count"

def test_expand_empty():
    assert expand_abbreviations("") == ""


# ── header_variants (raw-first, dedup) ───────────────────────────────────────

def test_variants_expandable_header():
    # raw first, then expanded
    assert header_variants("AGE_AT_DX") == ["age at dx", "age at diagnosis"]

def test_variants_single_when_no_expansion():
    assert header_variants("status") == ["status"]        # no abbrev token
    assert header_variants("Sample_Type") == ["sample type"]

def test_variants_gene_symbol_keeps_raw_first():
    # F1: raw symbol must be present and first, even though it also expands
    v = header_variants("HER2")
    assert v[0] == "her2"
    assert v == ["her2", "human epidermal growth factor receptor 2"]


def test_variants_plural_not_normalized():
    # singular/plural is intentionally NOT normalized: trailing 's' is kept
    assert header_variants("SAMPLES") == ["samples"]


# ── _load_abbrev_map degradation (never silent) ──────────────────────────────

def test_load_abbrev_map_none_path_degrades(monkeypatch):
    import metaharmonizer.models.schema_mapper.config as cfg
    monkeypatch.setattr(cfg, "CLINICAL_ABBREV_PATH", None)
    monkeypatch.setattr(u, "_load_abbrev_map", _REAL_LOAD)  # real body, not the fixture stub
    warn = MagicMock()
    monkeypatch.setattr(u.logger, "warning", warn)
    _REAL_LOAD.cache_clear()
    try:
        # degrades to {} AND logs a warning (never silent)
        assert u._load_abbrev_map() == {}
        warn.assert_called_once()
        assert "DISABLED" in warn.call_args[0][0]
        # expansion degrades to a no-op
        assert expand_abbreviations("age at dx") == "age at dx"
    finally:
        _REAL_LOAD.cache_clear()  # don't leak the {} result to other tests


# ── double-match invariant (matcher level, no model needed) ──────────────────

def _exact_engine(std=None, sources=None):
    """Minimal mock engine for the exact matchers."""
    eng = MagicMock()
    eng.normed_std_to_std = std or {}
    eng.has_alias_dict = bool(sources)
    eng.sources_to_fields = {k: v[0] for k, v in (sources or {}).items()}
    eng.normed_source_to_source = {k: v[1] for k, v in (sources or {}).items()}
    return eng

def test_std_exact_raw_hit_never_dropped_by_expansion():
    # raw 'her2' hits a symbol field; expansion misses -> raw result preserved (F1)
    eng = _exact_engine(std={"her2": "gene_her2"})
    out = StandardExactMatcher(eng).match("HER2")
    assert out == [("gene_her2", 1.0, "")]

def test_std_exact_expansion_recovers_missed_hit():
    # raw 'dx' misses; expanded 'diagnosis' hits -> expansion adds the hit
    eng = _exact_engine(std={"diagnosis": "diagnosis_field"})
    out = StandardExactMatcher(eng).match("DX")
    assert out == [("diagnosis_field", 1.0, "")]

def test_std_exact_both_variants_hit_raw_first():
    eng = _exact_engine(std={"dx": "abbr_field", "diagnosis": "full_field"})
    out = StandardExactMatcher(eng).match("DX")
    # both surfaced at 1.0, raw variant's field ranked first
    assert out == [("abbr_field", 1.0, ""), ("full_field", 1.0, "")]

def test_alias_exact_raw_source_wins_on_tie():
    # both variants map to the same field; raw variant's source is kept
    eng = _exact_engine(sources={"dx": (["diagnosis_field"], "DX_RAW"),
                                 "diagnosis": (["diagnosis_field"], "DIAGNOSIS_SRC")})
    out = AliasExactMatcher(eng).match("DX")
    assert out == [("diagnosis_field", 1.0, "DX_RAW")]
