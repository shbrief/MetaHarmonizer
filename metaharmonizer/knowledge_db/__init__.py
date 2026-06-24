"""
KnowledgeDb initialization module.
Ensures FAISS indexes and SQLite database are available before use.
If missing, first run falls through to local-build mode (pipelines rebuild
the corpus + index on demand from upstream ontology APIs).
"""

import threading
from metaharmonizer.CustomLogger.custom_logger import CustomLogger
# `.env` auto-load happens in `metaharmonizer._paths`; this import also pulls
# the resolved path constants we need below.
from metaharmonizer._paths import (
    KNOWLEDGE_DB_DIR, VECTOR_DB_PATH, FAISS_INDEX_DIR,
)

_initialized = False
_init_lock = threading.Lock()
_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = CustomLogger().custlogger(loglevel="INFO")
    return _logger


def _check_files_exist() -> bool:
    """
    Check if required KnowledgeDb files exist.

    Returns:
        True if both SQLite and at least one FAISS index exist.
    """
    has_sqlite = VECTOR_DB_PATH.exists()
    has_faiss = FAISS_INDEX_DIR.exists() and any(
        FAISS_INDEX_DIR.glob("*.index"))
    return has_sqlite and has_faiss


def ensure_knowledge_db() -> None:
    """
    Ensure KnowledgeDb files are available.

    Checks for existing files; if missing, falls through to local-build mode
    (downstream pipelines rebuild on demand from ontology APIs).

    Should be called before initializing FAISSSQLiteSearch or SynonymDict.
    """
    global _initialized

    # Fast path: already initialized
    if _initialized:
        return

    with _init_lock:
        # Double-check after acquiring lock
        if _initialized:
            return

        logger = _get_logger()

        if _check_files_exist():
            logger.info(f"KnowledgeDb files found at {KNOWLEDGE_DB_DIR}, using existing data")
        else:
            logger.info(
                f"KnowledgeDb files not found at {KNOWLEDGE_DB_DIR}; "
                "will use local build mode (first run may be slow)"
            )

        _initialized = True


def reset_initialization() -> None:
    """
    Reset initialization state. Useful for testing.
    """
    global _initialized
    with _init_lock:
        _initialized = False
