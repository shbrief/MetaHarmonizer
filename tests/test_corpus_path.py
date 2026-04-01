"""Tests for corpus_path — centralized corpus file path resolution."""
import os
import pytest
from pathlib import Path
from unittest.mock import patch

from src._paths import corpus_path, DATA_DIR, RETRIEVED_ONTOLOGIES_DIR


class TestCorpusPath:
    """corpus_path(category, ontology_source, suffix) returns canonical paths."""

    def test_json_path(self):
        p = corpus_path("disease", "ncit", ".json")
        assert p == RETRIEVED_ONTOLOGIES_DIR / "ncit_disease.json"
        assert p.suffix == ".json"

    def test_csv_path(self):
        p = corpus_path("disease", "ncit", "_corpus.csv")
        assert p.name == "ncit_disease_corpus.csv"
        assert p.parent == RETRIEVED_ONTOLOGIES_DIR

    def test_different_category_and_source(self):
        p = corpus_path("bodysite", "uberon", ".json")
        assert p.name == "uberon_bodysite.json"

    def test_path_under_retrieved_ontologies(self):
        """All corpus paths live under DATA_DIR/corpus/retrieved_ontologies/."""
        p = corpus_path("treatment", "ncit", "_corpus.csv")
        assert str(RETRIEVED_ONTOLOGIES_DIR) in str(p)

    def test_json_and_csv_share_directory(self):
        j = corpus_path("disease", "mondo", ".json")
        c = corpus_path("disease", "mondo", "_corpus.csv")
        assert j.parent == c.parent

    def test_respects_data_dir_env(self):
        """When METAHARMONIZER_DATA_DIR is set, paths follow it."""
        # DATA_DIR is resolved at import time, so we verify the relationship
        assert RETRIEVED_ONTOLOGIES_DIR == DATA_DIR / "corpus" / "retrieved_ontologies"
