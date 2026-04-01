"""Tests for OntoMapEngine._partition_codes — code prefix routing logic."""
import pytest

from src.Engine.ontology_mapping_engine import OntoMapEngine


class TestPartitionCodes:
    """_partition_codes splits clean_codes into ontology-source groups."""

    def test_bare_codes_go_to_ncit(self):
        result = OntoMapEngine._partition_codes(["C12345", "C99999"])
        assert result == {"ncit": ["C12345", "C99999"]}

    def test_uberon_prefix(self):
        result = OntoMapEngine._partition_codes(["UBERON_0001062"])
        assert result == {"uberon": ["UBERON_0001062"]}

    def test_mondo_prefix(self):
        result = OntoMapEngine._partition_codes(["MONDO_0005015", "MONDO_0000001"])
        assert result == {"mondo": ["MONDO_0005015", "MONDO_0000001"]}

    def test_mixed_prefixes(self):
        codes = ["C12345", "UBERON_0001062", "C99999", "MONDO_0005015"]
        result = OntoMapEngine._partition_codes(codes)
        assert set(result.keys()) == {"ncit", "uberon", "mondo"}
        assert result["ncit"] == ["C12345", "C99999"]
        assert result["uberon"] == ["UBERON_0001062"]
        assert result["mondo"] == ["MONDO_0005015"]

    def test_empty_list(self):
        result = OntoMapEngine._partition_codes([])
        assert result == {}

    def test_known_ols_prefixes(self):
        """All prefixes in PREFIX_TO_ONTOLOGY should map correctly."""
        codes = ["EFO_001", "HP_001", "DOID_001", "CL_001", "CHEBI_001"]
        result = OntoMapEngine._partition_codes(codes)
        assert "efo" in result
        assert "hp" in result
        assert "doid" in result
        assert "cl" in result
        assert "chebi" in result

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown ontology prefix"):
            OntoMapEngine._partition_codes(["XYZ_12345"])
