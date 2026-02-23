"""
Shared utilities for MetaHarmonizer engines.

Exports are lazy-loaded to avoid pulling in optional heavy dependencies
(faiss, torch model-loading chain) when only lightweight utilities are needed.
"""

__all__ = ["normalize", "load_model", "EmbeddingStore"]


def __getattr__(name: str):
    if name == "normalize":
        from src.utils.schema_mapper_utils import normalize
        return normalize
    if name == "load_model":
        from src.utils.model_loader import load_model
        return load_model
    if name == "EmbeddingStore":
        from src.utils.embedding_store import EmbeddingStore
        return EmbeddingStore
    raise AttributeError(f"module 'src.utils' has no attribute {name!r}")
