import gc
import torch
import sqlite3
import faiss
import os
import asyncio
import httpx
import numpy as np
import pandas as pd
from typing import List
from functools import lru_cache
from tqdm.auto import tqdm
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import get_embedding_model_cached
from src.CustomLogger.custom_logger import CustomLogger
from src.KnowledgeDb import ensure_knowledge_db
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.KnowledgeDb.concept_table_builder import ConceptTableBuilder

# Load environment variables for paths and API key
BASE_DB = os.getenv("VECTOR_DB_PATH")
BASE_IDX_DIR = os.getenv("FAISS_INDEX_DIR")
UMLS_API_KEY = os.getenv("UMLS_API_KEY")

TERM_BATCH_SIZE = 60  # Default batch size for term processing


class Document:
    """A simple Document class to hold page content and metadata."""

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FAISSSQLiteSearch:
    """
    This class provides a hybrid FAISS + SQLite backend for storing and retrieving
    embedded vectors and associated metadata for ontology mapping.
    It supports both RAG (Retrieval-Augmented Generation) and LM/ST strategies.
    """

    def __init__(
        self,
        method: str,
        category: str,
        om_strategy: str = "rag",
    ):
        ensure_knowledge_db()
        self.db_path = BASE_DB
        self.db = NCIDb(UMLS_API_KEY)
        self.method = method
        self.om_strategy = om_strategy
        self.category = category
        if self.om_strategy in ("st", "lm"):
            self.table_name = f"corpus_{category}"
        elif self.om_strategy in ("rag", "rag_bie"):
            self.table_name = f"rag_{category}"
        else:
            raise ValueError(
                f"Unsupported om_strategy: {om_strategy}. Choose from 'rag', 'rag_bie', 'st', 'lm'."
            )
        method_clean = method.replace('-', '_')
        idx_name = f"{om_strategy}_{method_clean}_{category}.index"
        self.index_path = os.path.join(BASE_IDX_DIR, idx_name)

        raw_model = get_embedding_model_cached(method)
        self.embedder = EmbeddingAdapter(raw_model)
        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.term_batch_size = TERM_BATCH_SIZE

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if self.om_strategy in ("st", "lm"):
            # Corpus table
            create_sql = f"""
              CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL
              )
            """
        elif self.om_strategy in ("rag", "rag_bie"):
            # RAG table (created by ConceptTableBuilder, ensure existence here)
            create_sql = f"""
              CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                context TEXT NOT NULL
              )
            """

        self.cursor.execute(create_sql)
        self.conn.commit()

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            rows = self.cursor.execute(
                f"SELECT id FROM {self.table_name} ORDER BY id").fetchall()
            self._ids = [r[0] for r in rows]
        else:
            self.index = None
            self._ids = []

    def ensure_corpus_integrity(self,
                                corpus: List[str],
                                corpus_df: pd.DataFrame = None):
        """
        Ensure all terms in the provided corpus are embedded and stored in FAISS and SQLite.
        Strategy-dependent.
        """
        if self.om_strategy in ("rag", "rag_bie"):
            if corpus_df is None:
                raise ValueError("corpus_df is required for RAG strategy")

            codes = corpus_df['clean_code'].dropna().unique().tolist()

            # 1. Use ConceptTableBuilder to build synonym and rag tables
            builder = ConceptTableBuilder(self.category)
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    builder.fetch_and_build_tables(codes, force_rebuild=False))
            finally:
                loop.close()

            # 2. Check and build FAISS index
            self._ensure_faiss_index()

        else:
            # ST/LM Strategy
            existing_terms = self._get_existing_terms()
            missing_terms = [
                term for term in corpus if term not in existing_terms
            ]

            if missing_terms:
                self.logger.info(
                    f"Adding {len(missing_terms)} new terms to corpus table")
                self._populate_corpus_table(missing_terms)
            else:
                self.logger.info("All corpus terms already in table")

            self._ensure_faiss_index()

        self._get_existing_terms.cache_clear()

    def _populate_corpus_table(self, corpus: List[str]):
        """
        Fill corpus table (for ST/LM)
        
        Only store terms, do not call NCI API
        """
        records = [(t, ) for t in corpus]
        self.cursor.executemany(
            f"INSERT INTO {self.table_name}(term) VALUES(?)", records)
        self.conn.commit()
        self.logger.info(
            f"✅ Inserted {len(records)} terms into {self.table_name}")

    def _ensure_faiss_index(self):
        """
        Ensure FAISS index exists and is synchronized with the table
        """
        # Check the number of records in the table
        table_count = self.cursor.execute(
            f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]

        if table_count == 0:
            self.logger.warning(f"Table {self.table_name} is empty")
            return

        # Check index status
        need_rebuild = False

        if not os.path.exists(self.index_path):
            self.logger.info(f"FAISS index not found, will build")
            need_rebuild = True
        elif self.index is None:
            self.logger.info(f"FAISS index not loaded, will build")
            need_rebuild = True
        elif self.index.ntotal != table_count:
            self.logger.info(
                f"⚠️ Index mismatch (table: {table_count}, index: {self.index.ntotal}), rebuilding..."
            )
            need_rebuild = True
        else:
            self.logger.info(
                f"✅ FAISS index up-to-date ({self.index.ntotal} vectors)")
            return

        # Rebuild index
        if need_rebuild:
            self._build_faiss_from_table()

    def _build_faiss_from_table(self):
        """
        Read data from table and build FAISS index
        
        General method: supports all strategies
        """
        # Read different columns based on strategy
        if self.om_strategy in ("rag", "rag_bie"):
            # RAG: Read context
            rows = self.cursor.execute(
                f"SELECT id, context FROM {self.table_name} ORDER BY id"
            ).fetchall()
            texts = [row[1] for row in rows]
        else:
            # ST/LM: Read term
            rows = self.cursor.execute(
                f"SELECT id, term FROM {self.table_name} ORDER BY id"
            ).fetchall()
            texts = [row[1] for row in rows]

        if not rows:
            self.logger.warning(f"No data in table {self.table_name}")
            return

        db_ids = [row[0] for row in rows]

        self.logger.info(f"Embedding {len(texts)} texts with {self.method}")

        # Batch embedding
        embed_batch_size = 512
        if faiss.get_num_gpus() > 0:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (
                1024**3)
            used_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - used_mem

            if free_mem < 2:
                embed_batch_size = max(64, embed_batch_size // 2)
            elif free_mem > 8:
                embed_batch_size = min(2048, embed_batch_size * 2)

        all_vectors = []
        for i in tqdm(range(0, len(texts), embed_batch_size),
                      desc="Embedding batches",
                      leave=False):
            batch_texts = texts[i:i + embed_batch_size]
            batch_vecs = self.embedder.embed_documents(batch_texts)
            all_vectors.extend(batch_vecs)

            del batch_texts, batch_vecs
            if faiss.get_num_gpus() > 0:
                torch.cuda.empty_cache()
            gc.collect()

        # Build FAISS index
        vectors_array = np.array(all_vectors, dtype=np.float32)
        vectors_array = self.normalize(vectors_array)

        dim = vectors_array.shape[1]
        cpu_index = faiss.IndexFlatIP(dim)

        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        # Add vectors in order
        self.index.add(vectors_array)
        self._ids = db_ids

        # Save
        os.makedirs(BASE_IDX_DIR, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        self.logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors")

    @lru_cache(maxsize=1)
    def _get_existing_terms(self):
        """Return a set of terms already in the database.
        Used to avoid duplicate API calls.
        """
        self.cursor.execute(f"SELECT term FROM {self.table_name}")
        return set(row[0] for row in self.cursor.fetchall())

    def normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize rows of v to unit length for cosine-similarity via IndexFlatIP."""
        if getattr(self, "embedder", None) and getattr(
                self.embedder.model, "_is_normalized", False):
            return v
        v = v.astype('float32', copy=False)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-12)

    def similarity_search(self,
                          query: str,
                          k: int = 5,
                          as_documents: bool = False):
        """Perform similarity search over FAISS index and return top-k matches.
        Only works with RAG strategy.
        Args:
            query (str): The query string to search for.
            k (int): Number of top matches to return.
            as_documents (bool): If True, return results as Document objects.
        Returns:
            List[dict] or List[Document]: A list of dictionaries or Document objects containing the
            term, context, code, and score for each match.
        Raises:
            NotImplementedError: If the strategy is not RAG.
        """
        if self.om_strategy not in ("rag", "rag_bie"):
            raise NotImplementedError(
                "LM/ST strategy should use get_match_results method.")

        vector = self.embedder.embed_query(query)
        vector = np.array(vector, dtype="float32").reshape(1, -1)
        vector = self.normalize(vector)
        distances, indices = self.index.search(vector, k)
        results = []
        for db_idx, score in zip(indices[0], distances[0]):
            if db_idx == -1:
                continue
            record_id = self._ids[db_idx]
            row = self.cursor.execute(
                f"SELECT term, context, code FROM {self.table_name} WHERE id = ?",
                (record_id, )).fetchone()
            if row:
                term, context, code = row
                result = {
                    "term": term,
                    "context": context,
                    "code": code,
                    "score": float(score)
                }
                if as_documents:
                    results.append(
                        Document(page_content=context, metadata=result))
                else:
                    results.append(result)
        return results

    def close(self):
        self.conn.close()
