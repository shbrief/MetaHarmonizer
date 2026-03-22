"""Resolve the project data directory.

In a cloned repo the data/ folder sits at the repo root (one level above src/).
After ``pip install`` that location no longer exists; users must point to their
own copy via the env var METAHARMONIZER_DATA_DIR.

Import DATA_DIR wherever a hardcoded ``"data/..."`` string was used.
"""
import os
from pathlib import Path

DATA_DIR: Path = Path(
    os.getenv("METAHARMONIZER_DATA_DIR",
              str(Path(__file__).parent.parent / "data"))
)
