"""CURIE → IRI conversion for OLS term lookup.

SNOMED IRIs are ``http://snomed.info/id/{id}`` with no ``PREFIX_`` infix.
The OBO default (``http://purl.obolibrary.org/obo/SNOMED_{id}``) is not
what OLS4 serves.
"""
from urllib.parse import quote_plus

from metaharmonizer.knowledge_db.db_clients.ols_db import (
    OLSDb,
    _curie_to_iri,
)


def test_snomed_iri_has_no_prefix_infix():
    assert _curie_to_iri("SNOMED", "443172007") == "http://snomed.info/id/443172007"


def test_obo_and_special_case_iris_unchanged():
    assert _curie_to_iri("UBERON", "0001062") == (
        "http://purl.obolibrary.org/obo/UBERON_0001062"
    )
    assert _curie_to_iri("EFO", "0000400") == "http://www.ebi.ac.uk/efo/EFO_0000400"
    assert _curie_to_iri("Orphanet", "558") == (
        "http://www.orpha.net/ORDO/Orphanet_558"
    )


def test_snomed_obo_id_and_clean_code_build_ols_url(tmp_path, monkeypatch):
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "ols.log"))
    iri = "http://snomed.info/id/443172007"
    expected = (
        "https://www.ebi.ac.uk/ols4/api/ontologies/snomed/terms/"
        + quote_plus(quote_plus(iri))
    )
    db = OLSDb()
    assert OLSDb.obo_id_to_iri("SNOMED:443172007") == iri
    assert OLSDb.infer_ontology("SNOMED:443172007") == "snomed"
    assert db._build_url("SNOMED:443172007") == expected
    assert db._build_url("SNOMED_443172007") == expected
