"""Resolve project data paths used by the ontology pipeline.

Also loads a project-local ``.env`` file (if present in CWD) as a developer
convenience — this is the single place that does that, so downstream modules
can rely on ``os.getenv`` returning populated values. For installed-package
users who don't have ``.env``, ``load_dotenv()`` is a silent no-op.
"""
import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# User-cache root. All per-user runtime artifacts hang off this (data dir,
# knowledge-DB, model cache). Kept as a single constant so tests / CLIs can
# patch it in one place.
USER_CACHE_ROOT: Path = Path.home() / ".metaharmonizer"


def _abs(p: str | Path) -> Path:
    """Resolve a path string: expand ``~``, make absolute relative to CWD at
    import time. Keeps behavior predictable when downstream code ``chdir``s."""
    return Path(p).expanduser().resolve()


DEFAULT_DATA_DIR: Path = USER_CACHE_ROOT / "data"
DATA_DIR: Path = _abs(os.getenv("METAHARMONIZER_DATA_DIR", str(DEFAULT_DATA_DIR)))
CORPUS_DIR: Path = DATA_DIR / "corpus"
RETRIEVED_ONTOLOGIES_DIR: Path = CORPUS_DIR / "retrieved_ontologies"

# Knowledge-DB runtime artifacts (SQLite + FAISS indexes) do NOT belong inside
# the installed package directory. Default is a per-user cache; dev setups
# override via env vars (KNOWLEDGE_DB_DIR, VECTOR_DB_PATH, FAISS_INDEX_DIR).
DEFAULT_KNOWLEDGE_DB_DIR: Path = USER_CACHE_ROOT / "KnowledgeDb"
KNOWLEDGE_DB_DIR: Path = _abs(
    os.getenv("KNOWLEDGE_DB_DIR", str(DEFAULT_KNOWLEDGE_DB_DIR))
)
VECTOR_DB_PATH: Path = _abs(
    os.getenv("VECTOR_DB_PATH", str(KNOWLEDGE_DB_DIR / "vector_db.sqlite"))
)
FAISS_INDEX_DIR: Path = _abs(
    os.getenv("FAISS_INDEX_DIR", str(KNOWLEDGE_DB_DIR / "faiss_indexes"))
)

# HuggingFace model snapshots downloaded by `metaharmonizer.utils.model_loader`.
DEFAULT_MODEL_CACHE_DIR: Path = USER_CACHE_ROOT / "model_cache"
MODEL_CACHE_DIR: Path = _abs(
    os.getenv("MODEL_CACHE_ROOT", os.getenv("MODEL_CACHE_DIR", str(DEFAULT_MODEL_CACHE_DIR)))
)

# Bundled small data files shipped inside the wheel. These are a read-only
# fallback when METAHARMONIZER_DATA_DIR is unset or the file is missing in
# the user's data dir. Keep bundle contents small — large corpora belong in
# METAHARMONIZER_DATA_DIR, not the package.
_BUNDLED_DATA_DIR: Path = Path(__file__).parent / "_bundled_data"


def bundled_file(rel_path: str) -> Path:
    """Return the absolute path to a file shipped inside the package.

    Parameters
    ----------
    rel_path : str
        Path relative to ``metaharmonizer/_bundled_data/`` (e.g.
        ``"corpus/oncotree_code_to_name.csv"``).
    """
    return _BUNDLED_DATA_DIR / rel_path


def resolve_data_file(rel_path: str) -> Path:
    """Locate a data file by checking the user data dir, then the bundle.

    Returns the path in ``DATA_DIR`` if it exists, otherwise the bundled
    copy. Raises ``FileNotFoundError`` if neither exists.

    Parameters
    ----------
    rel_path : str
        Path relative to ``DATA_DIR`` / ``_bundled_data/``
        (e.g. ``"corpus/oncotree_code_to_name.csv"``).
    """
    user_path = DATA_DIR / rel_path
    if user_path.exists():
        return user_path
    bundled = bundled_file(rel_path)
    if bundled.exists():
        return bundled
    raise FileNotFoundError(
        f"Data file {rel_path!r} not found in METAHARMONIZER_DATA_DIR "
        f"({DATA_DIR}) or bundled resources ({_BUNDLED_DATA_DIR})."
    )


_SAFE_ID = re.compile(r"^[a-z0-9_]+$")
_SAFE_SUFFIX = re.compile(r"^[a-z0-9_.]+$")


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
    if os.sep in suffix or (os.altsep and os.altsep in suffix):
        raise ValueError(f"Unsafe suffix: {suffix!r}. Must not contain path separators.")
    return RETRIEVED_ONTOLOGIES_DIR / f"{ontology_source}_{category}{suffix}"
