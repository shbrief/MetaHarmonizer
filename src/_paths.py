"""Resolve project data paths used by the ontology pipeline."""
import os
from pathlib import Path

DATA_DIR: Path = Path(
    os.getenv("METAHARMONIZER_DATA_DIR",
              str(Path(__file__).parent.parent / "data"))
)
CORPUS_DIR: Path = DATA_DIR / "corpus"
RETRIEVED_ONTOLOGIES_DIR: Path = CORPUS_DIR / "retrieved_ontologies"


def retrieved_ontology_json_path(category: str, ontology_source: str) -> Path:
    """Canonical JSON cache path for fetched ontology corpora."""
    return RETRIEVED_ONTOLOGIES_DIR / f"{ontology_source}_{category}.json"
