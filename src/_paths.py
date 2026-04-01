"""Resolve project data paths used by the ontology pipeline."""
import os
import re
from pathlib import Path

DATA_DIR: Path = Path(
    os.getenv("METAHARMONIZER_DATA_DIR",
              str(Path(__file__).parent.parent / "data"))
)
CORPUS_DIR: Path = DATA_DIR / "corpus"
RETRIEVED_ONTOLOGIES_DIR: Path = CORPUS_DIR / "retrieved_ontologies"


_SAFE_ID = re.compile(r"^[a-z0-9_]+$")


def corpus_path(category: str, ontology_source: str, suffix: str) -> Path:
    """Canonical cache path for a (category, ontology_source) corpus file.

    Parameters
    ----------
    suffix : str
        File extension including the dot, e.g. ``".json"`` or ``"_corpus.csv"``.
    """
    if not _SAFE_ID.match(category):
        raise ValueError(f"Unsafe category: {category!r}. Must match [a-z0-9_]+.")
    if not _SAFE_ID.match(ontology_source):
        raise ValueError(f"Unsafe ontology_source: {ontology_source!r}. Must match [a-z0-9_]+.")
    return RETRIEVED_ONTOLOGIES_DIR / f"{ontology_source}_{category}{suffix}"
