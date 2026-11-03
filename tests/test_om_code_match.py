"""Stage 1 code matching (OntoMapEngine._code_matching).

A query that is a bare code (``8155/3``) or starts with ``<code> :``
(``C24.0 : Common bile duct``) is matched to the single corpus term carrying
that code, as a label prefix or a bare-code synonym. Codes carried by more
than one term, and code-like tokens not followed by ``:`` (``C32 cell``), are
left to the later stages.
"""
import sqlite3

import pytest

import metaharmonizer._paths as paths
from metaharmonizer.engine.ontology_mapping_engine import (
    DEFAULT_CODE_PATTERN, OntoMapEngine)


SITES = ["C24.0 : Extrahepatic bile duct", "C30.0 : Nasal cavity",
         "C80.9 : Unknown primary site", "C80.9 : Unknown primary site, NOS",
         "Not Reported"]


def _engine(corpus, code_pattern=DEFAULT_CODE_PATTERN, ont="c3dc"):
    """Minimal stub; __init__ is heavy and irrelevant to this logic."""
    e = OntoMapEngine.__new__(OntoMapEngine)
    e.corpus = corpus
    e.code_pattern = code_pattern
    e.category = "disease"
    e._table_suffix = "_test"
    e._ontology_source = ont
    e.ground_truth_map = {}
    e._test_or_prod = "prod"
    e.top_k = 5
    return e


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    db = tmp_path / "vector_db.sqlite"
    monkeypatch.setattr(paths, "VECTOR_DB_PATH", db)
    return db


def test_label_prefix_code(tmp_db):
    out = _engine(SITES)._code_matching(
        ["C24.0 : Common bile duct", "C30.0: Nasal septum, NOS", "c30.0 : x"])
    assert out == {"C24.0 : Common bile duct": "C24.0 : Extrahepatic bile duct",
                   "C30.0: Nasal septum, NOS": "C30.0 : Nasal cavity",
                   "c30.0 : x": "C30.0 : Nasal cavity"}


def test_ambiguous_code_falls_through(tmp_db):
    # C80.9 heads two corpus labels: no unique answer, leave it to Stage 2
    assert _engine(SITES)._code_matching(["C80.9 : Unknown"]) == {}


def test_code_must_be_followed_by_colon_or_end(tmp_db):
    # "C32 cell" is a cell line name, not "C32 : ..."
    corpus = ["C32", "C24.0 : Extrahepatic bile duct"]
    assert _engine(corpus)._code_matching(["C32 cell", "C24.0 bile"]) == {}
    assert _engine(corpus)._code_matching(["C32"]) == {"C32": "C32"}


def test_bare_code_synonym_from_concept_table(tmp_db):
    with sqlite3.connect(tmp_db) as conn:
        conn.execute("CREATE TABLE c3dc_synonym_disease_test "
                     "(id INTEGER PRIMARY KEY, synonym TEXT, official_label TEXT, code TEXT)")
        conn.executemany(
            "INSERT INTO c3dc_synonym_disease_test (synonym, official_label, code) "
            "VALUES (?, ?, ?)",
            [("8155/3", "Vipoma", "C3DC_dx_8155-3"),
             ("Vipoma, NOS", "Vipoma", "C3DC_dx_8155-3"),
             ("9835/3", "Precursor cell lymphoblastic leukemia, NOS", "C3DC_dx_9835-3"),
             ("9835/3", "Acute Lymphoblastic Leukemia, NOS", "C3DC_dx_x0001")])
    corpus = ["Vipoma", "Precursor cell lymphoblastic leukemia, NOS",
              "Acute Lymphoblastic Leukemia, NOS"]
    e = _engine(corpus)
    assert e._code_matching(["8155/3", "9835/3", "Vipoma, NOS"]) == {"8155/3": "Vipoma"}


def test_disabled_and_no_code_queries(tmp_db):
    assert _engine(SITES, code_pattern=None)._code_matching(
        ["C24.0 : Common bile duct"]) == {}
    assert _engine(SITES)._code_matching(["Pain", "Asthma"]) == {}


def test_build_exact_df_reports_code_match(tmp_db):
    e = _engine(SITES)
    df = e._build_exact_df(["Not Reported"],
                           {"C24.0 : Common bile duct": "C24.0 : Extrahepatic bile duct"})
    assert list(df["original_value"]) == ["Not Reported", "C24.0 : Common bile duct"]
    assert list(df["match1"]) == ["Not Reported", "C24.0 : Extrahepatic bile duct"]
    assert set(df["stage"]) == {1.0}
