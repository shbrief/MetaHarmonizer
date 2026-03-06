"""
Ontology mapping model strategies.

Exports are lazy-loaded to avoid importing faiss and other optional
heavy dependencies until they are explicitly requested.
"""

__all__ = ["OntoMapLM", "OntoMapST", "OntoMapRAG"]


def __getattr__(name: str):
    if name == "OntoMapLM":
        from .ontology_mapper_lm import OntoMapLM
        return OntoMapLM
    if name == "OntoMapST":
        from .ontology_mapper_st import OntoMapST
        return OntoMapST
    if name == "OntoMapRAG":
        from .ontology_mapper_rag import OntoMapRAG
        return OntoMapRAG
    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
