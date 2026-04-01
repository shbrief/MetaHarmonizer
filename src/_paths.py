"""Resolve project data paths used by the ontology pipeline."""
import os
from pathlib import Path

DATA_DIR: Path = Path(
    os.getenv("METAHARMONIZER_DATA_DIR",
              str(Path(__file__).parent.parent / "data"))
)
CORPUS_DIR: Path = DATA_DIR / "corpus"
RETRIEVED_ONTOLOGIES_DIR: Path = CORPUS_DIR / "retrieved_ontologies"


def corpus_path(category: str, ontology_source: str, suffix: str) -> Path:
    """Canonical cache path for a (category, ontology_source) corpus file.

    Parameters
    ----------
    suffix : str
        File extension including the dot, e.g. ``".json"`` or ``"_corpus.csv"``.
    """
    return RETRIEVED_ONTOLOGIES_DIR / f"{ontology_source}_{category}{suffix}"
