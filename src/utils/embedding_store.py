"""
Shared embedding store for MetaHarmonizer engines.

Centralizes embedding computation and caching so that SchemaMapEngine,
FieldSuggester, SemanticClustering, and OntoMapEngine can reuse
embeddings without redundant model loading or inference.

Usage
-----
::

    store = EmbeddingStore(model_name="all-MiniLM-L6-v2", strategy="st")

    # First call computes; second call for the same texts is a cache hit.
    vecs = store.embed(["age_at_diagnosis", "tumor_size"], source="column_name")
    vecs2 = store.embed(["age_at_diagnosis"], source="schema_field")  # cache hit

    # Optional disk persistence
    store.save(Path("data/outputs/embedding_cache"))
    store.load(Path("data/outputs/embedding_cache"))
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.CustomLogger.custom_logger import CustomLogger
from src.utils.schema_mapper_utils import normalize

# Default model — matches FieldSuggester and SchemaMapEngine
_DEFAULT_MODEL = os.environ.get("FIELD_MODEL", "all-MiniLM-L6-v2")


# ======================================================================
# Data Structures
# ======================================================================


@dataclass
class EmbeddingRecord:
    """A cached embedding with its provenance metadata."""

    text: str
    normalized_text: str
    vector: np.ndarray
    model_name: str
    source: str  # e.g. "column_name", "ontology_term", "schema_field"


# ======================================================================
# EmbeddingStore
# ======================================================================


class EmbeddingStore:
    """Centralized embedding store that serves all MetaHarmonizer engines.

    Features
    --------
    - **Deduplication** : same text + same model → computed once.
    - **Single model load** : model stays in memory (lazy-loaded on first
      ``embed()`` call).
    - **Cross-engine reuse** : FieldSuggester can consume embeddings
      already computed by SchemaMapEngine.
    - **Optional disk persistence** : save / load to ``.npy`` + CSV.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier or SentenceTransformer name.
    strategy : {"st", "lm"}
        ``"st"`` for SentenceTransformer ``.encode()`` (default).
        ``"lm"`` for CLS-token embeddings via HuggingFace ``AutoModel``.
    cache_dir : Path, optional
        If set, embeddings can be persisted to / loaded from this directory.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        strategy: Literal["st", "lm"] = "st",
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.strategy = strategy
        self.cache_dir = cache_dir

        # In-memory cache: cache_key → EmbeddingRecord
        self._cache: Dict[str, EmbeddingRecord] = {}

        # Lazy-loaded model
        self._model = None
        self._tokenizer = None  # only used for "lm" strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: List[str],
        source: str = "unknown",
        batch_size: int = 64,
    ) -> np.ndarray:
        """Compute or retrieve embeddings for a list of texts.

        Parameters
        ----------
        texts : list of str
            Raw text strings to embed.
        source : str
            Provenance tag (``"column_name"``, ``"ontology_term"``, etc.).
            This is metadata only — the same text always maps to the same
            cache entry regardless of source.
        batch_size : int
            Batch size for model inference.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0, 0))

        normalized = [normalize(t) for t in texts]

        # Partition into cached vs. uncached
        to_compute_indices: List[int] = []
        to_compute_texts: List[str] = []

        for i, norm_text in enumerate(normalized):
            key = self._cache_key(norm_text)
            if key not in self._cache:
                to_compute_indices.append(i)
                to_compute_texts.append(texts[i])

        # Compute missing embeddings
        if to_compute_texts:
            log = CustomLogger().custlogger("INFO")
            log.info(
                "Computing %d new embeddings (%d cached). Source: %s",
                len(to_compute_texts),
                len(texts) - len(to_compute_texts),
                source,
            )
            new_vectors = self._compute_embeddings(to_compute_texts, batch_size)

            for idx, vec in zip(to_compute_indices, new_vectors):
                norm_text = normalized[idx]
                record = EmbeddingRecord(
                    text=texts[idx],
                    normalized_text=norm_text,
                    vector=vec,
                    model_name=self.model_name,
                    source=source,
                )
                self._cache[self._cache_key(norm_text)] = record

        # Assemble result matrix in original order
        result = np.stack(
            [self._cache[self._cache_key(n)].vector for n in normalized]
        )
        return result

    def get_cached_texts(self, source: Optional[str] = None) -> List[str]:
        """Return all cached original texts, optionally filtered by source."""
        records = self._cache.values()
        if source is not None:
            records = [r for r in records if r.source == source]
        return [r.text for r in records]

    def get_cached_vectors(self, source: Optional[str] = None) -> np.ndarray:
        """Return all cached vectors, optionally filtered by source."""
        records = list(self._cache.values())
        if source is not None:
            records = [r for r in records if r.source == source]
        if not records:
            return np.empty((0, 0))
        return np.stack([r.vector for r in records])

    @property
    def size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)

    def clear(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Disk Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save cache to disk as ``.npy`` + ``metadata.csv``.

        Returns the directory path.
        """
        save_dir = Path(path) if path else self.cache_dir
        if save_dir is None:
            raise ValueError("No cache_dir configured and no path provided.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._cache.values())
        if not records:
            return save_dir

        vectors = np.stack([r.vector for r in records])
        np.save(save_dir / "vectors.npy", vectors)

        metadata = pd.DataFrame(
            [
                {
                    "text": r.text,
                    "normalized_text": r.normalized_text,
                    "model_name": r.model_name,
                    "source": r.source,
                }
                for r in records
            ]
        )
        metadata.to_csv(save_dir / "metadata.csv", index=False)

        log = CustomLogger().custlogger("INFO")
        log.info("Saved %d embeddings to %s", len(records), save_dir)
        return save_dir

    def load(self, path: Optional[Path] = None) -> int:
        """Load cache from disk.  Returns the number of *new* records loaded."""
        load_dir = Path(path) if path else self.cache_dir
        if load_dir is None:
            raise ValueError("No cache_dir configured and no path provided.")

        load_dir = Path(load_dir)
        vectors_path = load_dir / "vectors.npy"
        metadata_path = load_dir / "metadata.csv"

        if not vectors_path.exists() or not metadata_path.exists():
            return 0

        vectors = np.load(vectors_path)
        metadata = pd.read_csv(metadata_path)

        loaded = 0
        for i, row in metadata.iterrows():
            record = EmbeddingRecord(
                text=row["text"],
                normalized_text=row["normalized_text"],
                vector=vectors[i],
                model_name=row["model_name"],
                source=row["source"],
            )
            key = self._cache_key(record.normalized_text)
            if key not in self._cache:
                self._cache[key] = record
                loaded += 1

        log = CustomLogger().custlogger("INFO")
        log.info(
            "Loaded %d new embeddings from %s (total cache: %d)",
            loaded,
            load_dir,
            self.size,
        )
        return loaded

    # ------------------------------------------------------------------
    # Internal — cache key
    # ------------------------------------------------------------------

    def _cache_key(self, normalized_text: str) -> str:
        """Cache key = model_name + truncated SHA-256 of normalized text."""
        h = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()[:16]
        return f"{self.model_name}::{h}"

    # ------------------------------------------------------------------
    # Internal — embedding computation
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self, texts: List[str], batch_size: int
    ) -> np.ndarray:
        """Compute embeddings using the configured model and strategy."""
        if self.strategy == "st":
            return self._compute_st(texts, batch_size)
        elif self.strategy == "lm":
            return self._compute_lm(texts, batch_size)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy!r}")

    def _compute_st(self, texts: List[str], batch_size: int) -> np.ndarray:
        """SentenceTransformer ``.encode()`` embeddings."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            log = CustomLogger().custlogger("INFO")
            log.info("Loaded SentenceTransformer: %s", self.model_name)
        return self._model.encode(
            texts, batch_size=batch_size, show_progress_bar=False
        )

    def _compute_lm(self, texts: List[str], batch_size: int) -> np.ndarray:
        """CLS-token embeddings from a HuggingFace Transformers model."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device).eval()
            log = CustomLogger().custlogger("INFO")
            log.info("Loaded LM model: %s (device: %s)", self.model_name, device)

        device = next(self._model.parameters()).device
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = self._model(**encoded)

            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_emb)

        return np.concatenate(all_embeddings, axis=0)
