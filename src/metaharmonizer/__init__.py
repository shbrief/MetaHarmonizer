"""
MetaHarmonizer — biomedical metadata harmonization platform.

Public entry-points (lazy-loaded to avoid importing optional heavy
dependencies like faiss until they are actually needed)
-------------------
OntoMapEngine         Ontology mapping (exact/fuzzy → embedding → RAG)
SchemaMapEngine       Schema mapping (dict/fuzzy → value → type → LLM)

New-field discovery (FieldSuggester / suggest_from_schema_mapper) is kept in
the package but intentionally not re-exported here; import it directly from
``metaharmonizer.field_suggester`` when needed.
"""

from importlib.metadata import version as _pkg_version, PackageNotFoundError

# Resolved from installed distribution metadata. Requires `pip install -e .`
# (or a regular install) — if the package is imported from a bare source
# checkout without being installed, `__version__` falls back to the sentinel.
try:
    __version__ = _pkg_version("metaharmonizer")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "OntoMapEngine",
    "SchemaMapEngine",
]


def __getattr__(name: str):
    if name == "OntoMapEngine":
        from metaharmonizer.engine.ontology_mapping_engine import OntoMapEngine
        return OntoMapEngine
    if name == "SchemaMapEngine":
        from metaharmonizer.models.schema_mapper import SchemaMapEngine
        return SchemaMapEngine
    raise AttributeError(f"module 'metaharmonizer' has no attribute {name!r}")
