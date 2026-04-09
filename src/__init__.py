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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("MetaHarmonizer")
except PackageNotFoundError:
    __version__ = "0.0.0-unknown"

__all__ = [
    "__version__",
    "OntoMapEngine",
    "SchemaMapEngine",
    "FieldSuggester",
    "suggest_from_schema_mapper",
]


def __getattr__(name: str):
    if name == "OntoMapEngine":
        from src.Engine.ontology_mapping_engine import OntoMapEngine
        return OntoMapEngine
    if name == "SchemaMapEngine":
        from src.models.schema_mapper import SchemaMapEngine
        return SchemaMapEngine
    if name == "FieldSuggester":
        from src.field_suggester import FieldSuggester
        return FieldSuggester
    if name == "suggest_from_schema_mapper":
        from src.field_suggester import suggest_from_schema_mapper
        return suggest_from_schema_mapper
    raise AttributeError(f"module 'src' has no attribute {name!r}")
