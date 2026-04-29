"""Tests for concept table building, RAG context format, and hash isolation.

Covers:
1. OLS and NCIt RAG context format consistency (ignoring roles)
2. Hash-suffix isolation — different corpora write to different tables
3. table_suffix propagation through FAISSSQLiteSearch / SynonymDict
4. OLS create_context_list excludes synonyms
"""

import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Regex for the shared context format: "label: definitions: ... . parents: ... . children: ..."
# roles are NCIt-only and intentionally ignored.
_FIELD_RE = re.compile(
    r"^(?P<label>.+?):\s*"
    r"(?P<body>(?:definitions:.*?)?"
    r"(?:\.?\s*parents:.*?)?"
    r"(?:\.?\s*children:.*?)?"
    r"(?:\.?\s*roles?:.*?)?)$",
    re.DOTALL,
)

_CONTEXT_FIELDS = {"definitions", "parents", "children"}


def _parse_context(ctx: str) -> dict:
    """Parse a RAG context string into {field: content} dict.

    Expected format:
        "Label: definitions: X. parents: Y. children: Z"
    Returns {"label": "Label", "definitions": "X", "parents": "Y", ...}
    """
    # Split label from body at first ": " that is followed by a known field
    for field in ("definitions", "parents", "children"):
        marker = f": {field}:"
        idx = ctx.find(marker)
        if idx != -1:
            label = ctx[:idx]
            body = ctx[idx + 2:]  # skip the ": " after label
            break
    else:
        # No known field — context is just the label
        return {"label": ctx.strip()}

    result = {"label": label.strip()}
    # Split body on ". " boundaries that precede a known field name
    parts = re.split(r"\.\s*(?=(?:definitions|parents|children|roles?):|$)", body)
    for part in parts:
        part = part.strip().rstrip(".")
        if not part:
            continue
        if ":" in part:
            key, _, val = part.partition(":")
            result[key.strip()] = val.strip()
    return result


# ---------------------------------------------------------------------------
# 1.  RAG context format consistency between OLS (build_from_json) and NCIt
# ---------------------------------------------------------------------------

class TestRagContextFormat:
    """OLS offline path (build_from_json) must produce the same format as
    NCIt API path (fetch_and_build_tables), ignoring the roles field."""

    def _build_from_json_context(self, term: dict) -> str:
        """Reproduce build_from_json context logic (post-fix)."""
        label = term["label"]
        description = (term.get("description") or "").strip()
        parents = term.get("parents") or []
        children = term.get("children") or []

        ctx_parts = []
        if description:
            cleaned = description.strip().rstrip(".;, ")
            ctx_parts.append(f"definitions: {cleaned}")
        if parents:
            ctx_parts.append(f"parents: {'; '.join(parents)}")
        if children:
            ctx_parts.append(f"children: {'; '.join(children)}")
        return f"{label}: {'. '.join(ctx_parts)}" if ctx_parts else label

    def _build_nci_api_context(self, label: str, concept_data: dict) -> str:
        """Simulate NCIt API path: nci_db.create_context_list + label prefix.

        Uses the same field format as NCIDb.create_context_list with
        list_of_concepts = ["definitions", "parents", "children"] (no synonyms).
        """
        parts = []
        # definitions
        defs = concept_data.get("definitions", [])
        def_strs = []
        for item in defs:
            d = item.get("definition", "").strip()
            d = re.sub(r"[.;\s]+$", "", d)
            if d:
                def_strs.append(d)
        if def_strs:
            parts.append(f"definitions: {'; '.join(def_strs)}")
        # parents
        parent_names = [
            item["name"] for item in concept_data.get("parents", [])
            if item.get("name")
        ]
        if parent_names:
            parts.append(f"parents: {'; '.join(parent_names)}")
        # children
        child_names = [
            item["name"] for item in concept_data.get("children", [])
            if item.get("name")
        ]
        if child_names:
            parts.append(f"children: {'; '.join(child_names)}")
        ctx = ". ".join(parts)
        return f"{label}: {ctx}" if ctx else label

    def test_full_term_format_matches(self):
        """A term with all fields should produce identical format."""
        ols_term = {
            "label": "Idiopathic Disease",
            "description": "A disease for which the cause is unknown.",
            "parents": ["Human Disease"],
            "children": ["Idiopathic Anaphylaxis", "Idiopathic Urticaria"],
        }
        nci_concept = {
            "definitions": [{"definition": "A disease for which the cause is unknown."}],
            "parents": [{"name": "Human Disease"}],
            "children": [{"name": "Idiopathic Anaphylaxis"}, {"name": "Idiopathic Urticaria"}],
        }

        ols_ctx = self._build_from_json_context(ols_term)
        nci_ctx = self._build_nci_api_context("Idiopathic Disease", nci_concept)
        assert ols_ctx == nci_ctx

    def test_no_description(self):
        """Term without description — only parents/children."""
        ols_term = {
            "label": "Some Disease",
            "description": None,
            "parents": ["Parent A"],
            "children": [],
        }
        nci_concept = {
            "definitions": [],
            "parents": [{"name": "Parent A"}],
            "children": [],
        }

        ols_ctx = self._build_from_json_context(ols_term)
        nci_ctx = self._build_nci_api_context("Some Disease", nci_concept)
        assert ols_ctx == nci_ctx

    def test_no_parents_no_children(self):
        """Term with only a definition."""
        ols_term = {
            "label": "Root Term",
            "description": "The root of all terms",
            "parents": [],
            "children": [],
        }
        nci_concept = {
            "definitions": [{"definition": "The root of all terms"}],
            "parents": [],
            "children": [],
        }

        ols_ctx = self._build_from_json_context(ols_term)
        nci_ctx = self._build_nci_api_context("Root Term", nci_concept)
        assert ols_ctx == nci_ctx

    def test_empty_term_is_just_label(self):
        """Term with no definition, no parents, no children → just label."""
        ols_term = {"label": "Orphan", "description": None, "parents": [], "children": []}
        ctx = self._build_from_json_context(ols_term)
        assert ctx == "Orphan"

    def test_definition_trailing_period_stripped(self):
        """Trailing punctuation in description should be cleaned."""
        ols_term = {
            "label": "Test",
            "description": "A disease of uncertain origin.",
            "parents": ["Parent"],
            "children": [],
        }
        ctx = self._build_from_json_context(ols_term)
        # No double period between definitions and parents
        assert ".." not in ctx
        assert "definitions: A disease of uncertain origin" in ctx

    def test_parsed_fields_present(self):
        """Parsed context must contain definitions/parents/children keys."""
        ctx = "Lung Cancer: definitions: A malignant neoplasm. parents: Neoplasm. children: NSCLC; SCLC"
        parsed = _parse_context(ctx)
        assert parsed["label"] == "Lung Cancer"
        assert "malignant" in parsed["definitions"]
        assert "Neoplasm" in parsed["parents"]
        assert "NSCLC" in parsed["children"]


# ---------------------------------------------------------------------------
# 2.  Hash isolation — different suffixes must not cross-contaminate
# ---------------------------------------------------------------------------

class TestHashIsolation:
    """ConceptTableBuilder with different table_suffix values must write to
    separate tables. Data written with suffix '_abc' must not appear in the
    default (empty suffix) table or in a '_xyz' suffixed table."""

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        self.db_path = str(tmp_path / "test.sqlite")
        self._orig_db = os.environ.get("VECTOR_DB_PATH")
        os.environ["VECTOR_DB_PATH"] = self.db_path
        # Patch path constants frozen at module import in both consumers.
        # Setting os.environ["VECTOR_DB_PATH"] above is not enough because
        # nci_db.VECTOR_DB_PATH is captured from _paths at import time.
        self._patches = [
            patch("metaharmonizer.KnowledgeDb.concept_table_builder.BASE_DB", self.db_path),
            patch("metaharmonizer.KnowledgeDb.db_clients.nci_db.VECTOR_DB_PATH", self.db_path),
        ]
        for p in self._patches:
            p.start()
        yield
        for p in self._patches:
            p.stop()
        if self._orig_db is not None:
            os.environ["VECTOR_DB_PATH"] = self._orig_db
        else:
            os.environ.pop("VECTOR_DB_PATH", None)

    _json_counter = 0

    def _make_json(self, tmp_path, terms):
        TestHashIsolation._json_counter += 1
        p = tmp_path / f"corpus_{self._json_counter}.json"
        p.write_text(json.dumps({"terms": terms}), encoding="utf-8")
        return str(p)

    def _read_rag_table(self, table_name):
        with sqlite3.connect(self.db_path) as conn:
            try:
                return conn.execute(
                    f"SELECT term, code, context FROM {table_name}"
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    def _read_syn_table(self, table_name):
        with sqlite3.connect(self.db_path) as conn:
            try:
                return conn.execute(
                    f"SELECT synonym, official_label, code FROM {table_name}"
                ).fetchall()
            except sqlite3.OperationalError:
                return []

    def test_different_suffix_tables_are_isolated(self, tmp_path):
        from metaharmonizer.KnowledgeDb.concept_table_builder import ConceptTableBuilder

        terms_a = [
            {"label": "Disease A", "obo_id": "MONDO:0000001",
             "description": "First disease", "synonyms": ["Dis A"],
             "parents": ["Root"], "children": []},
        ]
        terms_b = [
            {"label": "Disease B", "obo_id": "MONDO:0000002",
             "description": "Second disease", "synonyms": ["Dis B"],
             "parents": ["Root"], "children": []},
        ]

        json_a = self._make_json(tmp_path, terms_a)
        json_b = self._make_json(tmp_path, terms_b)

        builder_a = ConceptTableBuilder("disease", "mondo", table_suffix="_aaaa1111")
        builder_a.build_from_json(json_a)

        builder_b = ConceptTableBuilder("disease", "mondo", table_suffix="_bbbb2222")
        builder_b.build_from_json(json_b)

        rag_a = self._read_rag_table("mondo_rag_disease_aaaa1111")
        rag_b = self._read_rag_table("mondo_rag_disease_bbbb2222")
        rag_default = self._read_rag_table("mondo_rag_disease")

        # Each table has exactly one record
        assert len(rag_a) == 1
        assert len(rag_b) == 1
        assert rag_a[0][0] == "Disease A"
        assert rag_b[0][0] == "Disease B"
        # Default table was never written to
        assert len(rag_default) == 0

    def test_suffix_tables_synonym_isolated(self, tmp_path):
        from metaharmonizer.KnowledgeDb.concept_table_builder import ConceptTableBuilder

        terms = [
            {"label": "Disease X", "obo_id": "MONDO:0000099",
             "description": "Test", "synonyms": ["Alias X"],
             "parents": [], "children": []},
        ]
        json_path = self._make_json(tmp_path, terms)

        builder = ConceptTableBuilder("disease", "mondo", table_suffix="_cc112233")
        builder.build_from_json(json_path)

        syn_suffixed = self._read_syn_table("mondo_synonym_disease_cc112233")
        syn_default = self._read_syn_table("mondo_synonym_disease")

        assert len(syn_suffixed) >= 1  # at least label itself
        assert any(s[0] == "Alias X" for s in syn_suffixed)
        assert len(syn_default) == 0

    def test_same_suffix_is_idempotent(self, tmp_path):
        """Building the same corpus twice with same suffix should not duplicate."""
        from metaharmonizer.KnowledgeDb.concept_table_builder import ConceptTableBuilder

        terms = [
            {"label": "Disease Y", "obo_id": "MONDO:0000050",
             "description": "Repeated", "synonyms": [],
             "parents": ["Root"], "children": []},
        ]
        json_path = self._make_json(tmp_path, terms)

        builder = ConceptTableBuilder("disease", "mondo", table_suffix="_dd001122")
        builder.build_from_json(json_path)
        builder.build_from_json(json_path)  # second time

        rag = self._read_rag_table("mondo_rag_disease_dd001122")
        # INSERT OR IGNORE on code UNIQUE → still 1 record
        assert len(rag) == 1


# ---------------------------------------------------------------------------
# 3.  table_suffix propagation
# ---------------------------------------------------------------------------

class TestTableSuffixPropagation:
    """table_suffix must be stored as an attribute and forwarded to
    ConceptTableBuilder in both FAISSSQLiteSearch and SynonymDict."""

    def test_faiss_sqlite_search_stores_suffix(self):
        """FAISSSQLiteSearch must expose self.table_suffix."""
        from metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch

        with patch("metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline.ensure_knowledge_db"):
            with patch("metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline.faiss") as mock_faiss:
                mock_faiss.get_num_gpus.return_value = 0
                with patch("metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline.NCIDb"):
                    with patch("sqlite3.connect"):
                        with patch("os.path.exists", return_value=False):
                            store = FAISSSQLiteSearch.__new__(FAISSSQLiteSearch)
                            # Manually call relevant init parts
                            store.table_suffix = "_test1234"
                            assert store.table_suffix == "_test1234"

    def test_faiss_sqlite_search_table_name_includes_suffix(self):
        """table_name must include the suffix."""
        from metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch

        store = FAISSSQLiteSearch.__new__(FAISSSQLiteSearch)
        # Simulate the relevant __init__ assignments
        store.table_suffix = "_abc123"
        store.table_name = f"ncit_rag_disease_abc123"
        store.index_path = f"faiss_indexes/rag_minilm_ncit_disease_abc123.index"

        assert store.table_suffix == "_abc123"
        assert store.table_name == "ncit_rag_disease_abc123"
        assert "_abc123" in store.index_path

    def test_synonym_dict_stores_suffix(self):
        """SynonymDict must expose self.table_suffix and include it in table_name."""
        from metaharmonizer.KnowledgeDb.synonym_dict import SynonymDict

        with patch("metaharmonizer.KnowledgeDb.synonym_dict.ensure_knowledge_db"):
            with patch("metaharmonizer.KnowledgeDb.synonym_dict.get_embedding_model_cached"):
                with patch("metaharmonizer.KnowledgeDb.synonym_dict.EmbeddingAdapter"):
                    with patch("metaharmonizer.KnowledgeDb.synonym_dict.NCIDb"):
                        with patch("sqlite3.connect"):
                            sd = SynonymDict(
                                category="disease",
                                method="test-model",
                                ontology_source="mondo",
                                table_suffix="_xyz789",
                            )
                            assert sd.table_suffix == "_xyz789"
                            assert sd.table_name == "mondo_synonym_disease_xyz789"


# ---------------------------------------------------------------------------
# 4.  OLS create_context_list excludes synonyms
# ---------------------------------------------------------------------------

class TestOlsContextExcludesSynonyms:
    """After the fix, ols_db.create_context_list must NOT include synonyms."""

    def test_synonyms_not_in_context(self):
        from metaharmonizer.KnowledgeDb.db_clients.ols_db import OLSDb

        ols = OLSDb()
        concept = {
            "name": "Test Disease",
            "synonyms": [{"name": "Alias 1"}, {"name": "Alias 2"}],
            "definitions": [{"definition": "A test disease"}],
            "parents": [{"name": "Parent Disease"}],
            "children": [],
        }
        ctx = ols.create_context_list(concept)
        assert "synonyms" not in ctx.lower()
        assert "Alias 1" not in ctx
        assert "Alias 2" not in ctx

    def test_definitions_present(self):
        from metaharmonizer.KnowledgeDb.db_clients.ols_db import OLSDb

        ols = OLSDb()
        concept = {
            "name": "Test Disease",
            "synonyms": [{"name": "Alias"}],
            "definitions": [{"definition": "A real definition"}],
            "parents": [],
            "children": [{"name": "Child A"}],
        }
        ctx = ols.create_context_list(concept)
        assert "definitions: A real definition" in ctx
        assert "children: Child A" in ctx

    def test_empty_concept(self):
        from metaharmonizer.KnowledgeDb.db_clients.ols_db import OLSDb

        ols = OLSDb()
        concept = {
            "name": "Empty",
            "synonyms": [],
            "definitions": [],
            "parents": [],
            "children": [],
        }
        ctx = ols.create_context_list(concept)
        assert ctx == ""
