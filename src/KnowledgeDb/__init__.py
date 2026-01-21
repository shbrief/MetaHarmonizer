"""
KnowledgeDb initialization module.
Ensures FAISS indexes and SQLite database are available before use.
Downloads from Google Drive if missing, falls back to local build if download fails.
"""

import os
import threading
from pathlib import Path
from src.CustomLogger.custom_logger import CustomLogger
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")
KNOWLEDGE_DB_DIR = Path(os.getenv("KNOWLEDGE_DB_DIR", "src/KnowledgeDb"))
VECTOR_DB_PATH = Path(
    os.getenv("VECTOR_DB_PATH", "src/KnowledgeDb/vector_db.sqlite"))
FAISS_INDEX_DIR = Path(
    os.getenv("FAISS_INDEX_DIR", "src/KnowledgeDb/faiss_indexes"))

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


def _try_download() -> bool:
    """
    Attempt to download KnowledgeDb files from Google Drive.
    
    Returns:
        True if download succeeded, False otherwise.
    """
    logger = _get_logger()

    # try:
    #     from src.utils.drive_utils import download_folder
    #     logger.info(
    #         "KnowledgeDb files missing, attempting download from Google Drive..."
    #     )
    #     download_folder(FOLDER_ID, str(KNOWLEDGE_DB_DIR))
    #     logger.info("Download completed successfully")
    #     return True
    # except ImportError:
    #     logger.warning("drive_utils not available, skipping download")
    #     return False
    # except Exception as e:
    #     logger.warning(f"Download failed: {e}")
    #     return False


def ensure_knowledge_db() -> None:
    """
    Ensure KnowledgeDb files are available.
    
    This function checks for existing files, attempts download if missing,
    and falls back to local build mode if download fails.
    
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

        # Check if files already exist
        if _check_files_exist():
            logger.info("KnowledgeDb files found, using existing data")
            _initialized = True
            return

        # Try to download
        if _try_download():
            if _check_files_exist():
                _initialized = True
                return

        # Download failed or incomplete, will use local build
        logger.info("Will use local build mode (first run may be slow)")
        _initialized = True


def reset_initialization() -> None:
    """
    Reset initialization state. Useful for testing.
    """
    global _initialized
    with _init_lock:
        _initialized = False
