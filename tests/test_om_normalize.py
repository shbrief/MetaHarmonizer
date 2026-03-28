"""Unit tests for OntoMapEngine._normalize_query()."""
import pytest
from src.Engine.ontology_mapping_engine import OntoMapEngine


def _engine(category: str = "disease"):
    """Minimal OntoMapEngine stub with a small corpus."""
    e = OntoMapEngine.__new__(OntoMapEngine)
    e.corpus = ["Carcinoma", "Lung Cancer", "Leukemia", "Tumor"]
    e.category = category
    return e


_CORPUS_SET = {"carcinoma", "lung cancer", "leukemia", "tumor"}


# ── underscore → space ──────────────────────────────────────────────────────

def test_underscore_to_space():
    e = _engine()
    assert e._normalize_query("breast_cancer", _CORPUS_SET) == "breast cancer"


def test_multiple_underscores():
    e = _engine()
    assert e._normalize_query("non_small_cell", _CORPUS_SET) == "non small cell"


# ── British → American spelling ─────────────────────────────────────────────

def test_leukaemia():
    e = _engine()
    assert e._normalize_query("leukaemia", _CORPUS_SET) == "leukemia"


def test_leukaemia_titlecase():
    # re.sub with (?i) uses the literal replacement string, so result is lowercase
    e = _engine()
    assert e._normalize_query("Leukaemia", _CORPUS_SET) == "leukemia"


def test_tumour():
    e = _engine()
    assert e._normalize_query("tumour", _CORPUS_SET) == "tumor"


def test_haemato_prefix():
    e = _engine()
    assert e._normalize_query("haematology", _CORPUS_SET) == "hematology"


def test_oedema():
    e = _engine()
    assert e._normalize_query("oedema", _CORPUS_SET) == "edema"


def test_paediatric():
    e = _engine()
    assert e._normalize_query("paediatric", _CORPUS_SET) == "pediatric"


def test_foetal():
    e = _engine()
    assert e._normalize_query("foetal", _CORPUS_SET) == "fetal"


def test_anaemia():
    e = _engine()
    assert e._normalize_query("anaemia", _CORPUS_SET) == "anemia"


def test_gynaecology():
    e = _engine()
    assert e._normalize_query("gynaecology", _CORPUS_SET) == "gynecology"


def test_diarrhoea():
    e = _engine()
    assert e._normalize_query("diarrhoea", _CORPUS_SET) == "diarrhea"


# ── plural stripping ─────────────────────────────────────────────────────────

def test_plural_strip_when_singular_in_corpus():
    e = _engine()
    assert e._normalize_query("Carcinomas", _CORPUS_SET) == "Carcinoma"


def test_plural_not_stripped_when_singular_absent():
    e = _engine()
    assert e._normalize_query("Sarcomas", _CORPUS_SET) == "Sarcomas"


def test_plural_short_word_not_stripped():
    """Words ≤ 3 chars should not be touched."""
    e = _engine()
    assert e._normalize_query("bus", _CORPUS_SET) == "bus"


# ── passthrough ──────────────────────────────────────────────────────────────

def test_no_change_passthrough():
    e = _engine()
    assert e._normalize_query("Lung Cancer", _CORPUS_SET) == "Lung Cancer"


# ── combined ─────────────────────────────────────────────────────────────────

def test_underscore_and_british():
    e = _engine()
    assert e._normalize_query("acute_leukaemia", _CORPUS_SET) == "acute leukemia"
