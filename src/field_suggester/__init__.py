"""FieldSuggester – standalone new-field discovery via hybrid NER + embedding clustering."""

__all__ = [
    "FieldSuggester",
    "suggest_from_schema_mapper",
    "SCSMode",
    "SemanticCluster",
    "SemanticClusteringConfig",
    "SemanticClusteringEngine",
]


def __getattr__(name: str):
    if name == "FieldSuggester":
        from .field_suggester import FieldSuggester
        return FieldSuggester
    if name == "suggest_from_schema_mapper":
        from .integration import suggest_from_schema_mapper
        return suggest_from_schema_mapper
    if name in ("SCSMode", "SemanticCluster", "SemanticClusteringConfig", "SemanticClusteringEngine"):
        from .semantic_clustering import (
            SCSMode,
            SemanticCluster,
            SemanticClusteringConfig,
            SemanticClusteringEngine,
        )
        _map = {
            "SCSMode": SCSMode,
            "SemanticCluster": SemanticCluster,
            "SemanticClusteringConfig": SemanticClusteringConfig,
            "SemanticClusteringEngine": SemanticClusteringEngine,
        }
        return _map[name]
    raise AttributeError(f"module 'src.field_suggester' has no attribute {name!r}")
