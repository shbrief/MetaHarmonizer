"""Package-relative path to the bundled data directory.

Import DATA_DIR wherever a hardcoded ``"data/..."`` string was used so that
paths resolve correctly both from the repo root and after ``pip install``.
"""
from pathlib import Path

DATA_DIR: Path = Path(__file__).parent.parent / "data"
