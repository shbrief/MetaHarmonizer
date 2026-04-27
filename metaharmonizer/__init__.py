"""
MetaHarmonizer — biomedical metadata harmonization platform.

Public entry-points (lazy-loaded to avoid importing optional heavy
dependencies like faiss until they are actually needed)
-------------------
OntoMapEngine         Ontology mapping (exact/fuzzy → embedding → RAG)
SchemaMapEngine       Schema mapping (dict/fuzzy → value → type → LLM)
FieldSuggester        New field discovery via NER + embedding clustering
suggest_from_schema_mapper
                      Convenience wrapper: schema mapper output → suggestions
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Resolved from installed distribution metadata. Requires `pip install -e .`
# (or a regular install) — if the package is imported from a bare source
# checkout without being installed, `__version__` falls back to the sentinel.
try:
    __version__ = _pkg_version("metaharmonizer")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "OntoMapEngine",
    "SchemaMapEngine",
    "FieldSuggester",
    "suggest_from_schema_mapper",
]


def __getattr__(name: str):
    if name == "OntoMapEngine":
        from metaharmonizer.Engine.ontology_mapping_engine import OntoMapEngine
        return OntoMapEngine
    if name == "SchemaMapEngine":
        from metaharmonizer.models.schema_mapper import SchemaMapEngine
        return SchemaMapEngine
    if name == "FieldSuggester":
        from metaharmonizer.field_suggester import FieldSuggester
        return FieldSuggester
    if name == "suggest_from_schema_mapper":
        from metaharmonizer.field_suggester import suggest_from_schema_mapper
        return suggest_from_schema_mapper
    raise AttributeError(f"module 'metaharmonizer' has no attribute {name!r}")
