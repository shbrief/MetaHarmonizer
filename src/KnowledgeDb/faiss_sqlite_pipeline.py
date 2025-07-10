import gc
from typing import List
import asyncio
import httpx
import numpy as np
import pandas as pd
import sqlite3
import faiss
import os
from functools import lru_cache
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import get_embedding_model_cached
from tqdm import tqdm
from src.CustomLogger.custom_logger import CustomLogger
import torch
from src.KnowledgeDb.db_clients.nci_db import NCIDb

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
        self.db_path = BASE_DB
        self.db = NCIDb(UMLS_API_KEY)
        idx_name = f"{om_strategy}_{method}_{category}.index"
        self.index_path = os.path.join(BASE_IDX_DIR, idx_name)
        self.table_name = f"{om_strategy}_{method.replace('-', '_')}_{category}"
        self.method = method
        self.om_strategy = om_strategy
        raw_model = get_embedding_model_cached(method)
        self.embedder = EmbeddingAdapter(raw_model)
        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.term_batch_size = TERM_BATCH_SIZE

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if self.om_strategy in ("rag", "rag_bie"):
            create_sql = f"""
              CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                context TEXT NOT NULL
              )
            """
        else:
            create_sql = f"""
              CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL
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
        existing_terms = self._get_existing_terms()
        missing_terms = [term for term in corpus if term not in existing_terms]
        df_missing = pd.DataFrame()
        if corpus_df is not None:
            df_missing = corpus_df[~corpus_df["official_label"].
                                   isin(existing_terms)].copy()

        if not missing_terms and (corpus_df is None or df_missing.empty):
            self.logger.info("All corpus terms already processed.")
            return

        total_missing = len(df_missing) if corpus_df is not None else len(
            missing_terms)
        self.logger.info(f"{total_missing} new terms to add to the index.")

        if self.om_strategy in ("rag", "rag_bie"):
            if corpus_df is not None:
                self.logger.info(
                    "Using provided DataFrame to update term-code pairs.")
                self.fetch_and_store_terms(terms=missing_terms,
                                           corpus_df=df_missing)
            else:
                self.fetch_and_store_terms(missing_terms)

        else:
            self.build_corpus_vector_db(missing_terms)

        self._get_existing_terms.cache_clear()

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

    def _insert_to_sqlite(self, records: List[tuple]) -> List[int]:
        """Insert records to SQLite and return their corresponding DB ids."""
        self.cursor.executemany(
            f"INSERT OR IGNORE INTO {self.table_name} (term, code, context) VALUES (?, ?, ?)",
            records)
        self.conn.commit()
        codes = [rec[1] for rec in records]
        placeholders = ','.join('?' * len(codes))
        self.cursor.execute(
            f"SELECT code, id FROM {self.table_name} WHERE code IN ({placeholders})",
            codes)
        code_to_id = {row[0]: row[1] for row in self.cursor.fetchall()}
        inserted_ids = []
        for _, code, _ in records:
            if code in code_to_id:
                inserted_ids.append(code_to_id[code])
        return inserted_ids

    def _insert_to_faiss(self, vectors):
        """Add vector embeddings to FAISS index."""
        if not vectors:
            return
        vec_array = np.array(vectors, dtype=np.float32)
        vec_array = self.normalize(vec_array)
        if self.index is None:
            dim = vec_array.shape[1]
            cpu_index = faiss.IndexFlatIP(dim)

            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            else:
                self.index = cpu_index

        self.index.add(vec_array)
        faiss.write_index(self.index, self.index_path)

    async def _get_term_code_pair(self, term: str, client: httpx.AsyncClient):
        """Fetch NCI codes for a single term."""
        try:
            codes = await self.db._umls_db.get_nci_code_by_term(term, client)
            return (term, codes) if codes else None
        except Exception as e:
            self.logger.warning(f"Failed to get code for '{term}': {e}")
            return None

    async def fetch_term_code_pairs_async(self, terms, api_key):
        """Fetch NCI codes for a list of terms asynchronously.
        Args:
            terms (List[str]): List of terms to fetch NCI codes for.
            api_key (str): API key for UMLS API.
        Returns:
            List[tuple]: List of tuples containing term and its associated NCI codes.
        """

        existing_terms = self._get_existing_terms()
        new_terms = [term for term in terms if term not in existing_terms]
        if not new_terms:
            self.logger.info("No new terms to process.")
            return []

        self.logger.info(f"Fetching NCI codes for {len(new_terms)} new terms")

        all_term_code_pairs = []
        async with httpx.AsyncClient(limits=httpx.Limits(
                max_connections=15)) as client:
            if len(new_terms) <= self.term_batch_size:
                for term in new_terms:
                    result = await self._get_term_code_pair(term, client)
                    if result:
                        all_term_code_pairs.append(result)
            else:
                for i in range(0, len(new_terms), self.term_batch_size):
                    batch_terms = new_terms[i:i + self.term_batch_size]
                    tasks = [
                        self._get_term_code_pair(term, client)
                        for term in batch_terms
                    ]
                    results = await asyncio.gather(*tasks,
                                                   return_exceptions=True)
                    for result in results:
                        if result and not isinstance(result, Exception):
                            all_term_code_pairs.append(result)
                    self.logger.info(
                        f"Processed UMLS batch {i//self.term_batch_size + 1} of {(len(new_terms) + self.term_batch_size - 1)//self.term_batch_size}"
                    )

        return all_term_code_pairs

    def fetch_term_code_pairs_from_df(
            self, df: pd.DataFrame) -> list[tuple[str, list[str]]]:
        """
        Return list of (term, [code]) pairs from a pre-defined DataFrame.
        The format matches fetch_term_code_pairs_async output.
        This function is an alternative to fetch_term_code_pairs_async
        for cases where terms and codes are already available in a DataFrame.

        Expected df columns:
            - official_label: term
            - clean_code: standard code (e.g. NCIT)
        """
        if not {"official_label", "clean_code"}.issubset(df.columns):
            raise ValueError(
                "corpus_df must contain 'official_label' and 'clean_code' columns."
            )

        df = df.dropna(subset=["official_label", "clean_code"])
        df = df.drop_duplicates(subset=["official_label", "clean_code"])

        term_code_map = {}
        for _, row in df.iterrows():
            term = row["official_label"]
            code = row["clean_code"]
            term_code_map.setdefault(term, set()).add(code)

        return [(term, list(codes)) for term, codes in term_code_map.items()]

    async def fetch_and_store_terms_async(self,
                                          terms,
                                          api_key,
                                          corpus_df=None):
        """Fetch, embed, and store term-code pairs and their vector representations.
        Args:
            terms (List[str]): List of terms to fetch NCI codes for.
            api_key (str): API key for UMLS API.
        """

        if corpus_df is not None:
            self.logger.info(
                "Using provided DataFrame to fetch term-code pairs.")
            all_term_code_pairs = self.fetch_term_code_pairs_from_df(corpus_df)
        else:
            all_term_code_pairs = await self.fetch_term_code_pairs_async(
                terms, api_key)

        if not all_term_code_pairs:
            self.logger.info("No new terms found with NCI codes.")
            return

        self.logger.info(
            f"Retrieved codes for {len(all_term_code_pairs)} terms")

        all_codes = list(
            set(code for _, codes in all_term_code_pairs for code in codes))
        self.logger.info(
            f"Fetching concept data for {len(all_codes)} unique codes")

        async with httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.db.concurrency *
                                    self.db.batch_size)) as client:
            concept_data_map = await self.db.get_custom_concepts_by_codes(
                all_codes, client)

        self.logger.info(
            f"Retrieved {len(concept_data_map)} concept data entries")

        contexts = []
        records = []
        seen_codes = set()

        for term, codes in tqdm(all_term_code_pairs,
                                desc="Building context and records"):
            for code in codes:
                if code not in concept_data_map or code in seen_codes:
                    continue
                seen_codes.add(code)
                try:
                    context = self.db.create_context_list(
                        concept_data_map[code])
                    combined = f"{term}: {context}"
                    contexts.append(combined)
                    records.append((term, code, combined))
                except Exception as e:
                    self.logger.warning(f"Error processing {term}-{code}: {e}")

        if not records:
            self.logger.info("No valid records to insert into SQLite.")
            return

        self.logger.info(f"Inserting {len(records)} records into SQLite")
        inserted_ids = self._insert_to_sqlite(records)
        self._ids.extend(inserted_ids)

        self.logger.info("Starting vector embedding and FAISS index insertion")
        embed_batch_size = 512
        if faiss.get_num_gpus() > 0:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (
                1024**3)
            used_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - used_mem
            if free_mem < 2:
                embed_batch_size = max(64, embed_batch_size // 2)
                self.logger.warning(
                    f"Low GPU memory ({free_mem:.2f}GB), reducing embed_batch_size to {embed_batch_size}"
                )
            elif free_mem > 8:
                embed_batch_size = min(2048, embed_batch_size * 2)
                self.logger.info(
                    f"High GPU memory ({free_mem:.2f}GB), increasing embed_batch_size to {embed_batch_size}"
                )
        for i in tqdm(range(0, len(contexts), embed_batch_size),
                      desc="Embedding batches"):
            batch_ctx = contexts[i:i + embed_batch_size]
            batch_vecs = self.embedder.embed_documents(batch_ctx)
            self._insert_to_faiss(batch_vecs)
            del batch_ctx, batch_vecs
            if faiss.get_num_gpus() > 0:
                torch.cuda.empty_cache()
            gc.collect()

        self.logger.info("Finished fetching and storing all terms.")

    def fetch_and_store_terms(self,
                              terms,
                              api_key=UMLS_API_KEY,
                              corpus_df=None):
        """Sync wrapper to run the async fetch and store method."""
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                self.fetch_and_store_terms_async(terms,
                                                 api_key,
                                                 corpus_df=corpus_df))
        finally:
            loop.close()
            self._get_existing_terms.cache_clear()
        return result

    def build_corpus_vector_db(self, corpus: List[str]):
        """Build a vector database from a corpus (curated ontology) list.
        This method is designed to work with the ST/LM strategy.
        Args:
            corpus (List[str]): List of documents to embed and store.
        Raises:
            ValueError: If the corpus is empty.
        """
        if not corpus:
            raise ValueError("Corpus cannot be empty.")

        if self.om_strategy in ("rag", "rag_bie"):
            raise NotImplementedError(
                "RAG/RAG_BIE strategy should use fetch_and_store_terms method."
            )

        # Initialize the embedder and dimensions with the first batch
        first_batch = corpus[:256]
        first_vecs = self.embedder.embed_documents(first_batch)
        dim = len(first_vecs[0])
        cpu_index = faiss.IndexFlatIP(dim)

        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

        first_array = self.normalize(np.array(first_vecs, dtype="float32"))
        self.index.add(first_array)

        # Batch add vectors to the index
        for i in range(256, len(corpus), 256):
            batch = corpus[i:i + 256]
            batch_vecs = self.embedder.embed_documents(batch)
            batch_array = np.array(batch_vecs, dtype="float32")
            batch_array = self.normalize(batch_array)
            self.index.add(batch_array)

            # Delete the batch and vectors to free memory
            del batch, batch_vecs, batch_array
            if faiss.get_num_gpus() > 0:
                torch.cuda.empty_cache()
            gc.collect()

        os.makedirs(BASE_IDX_DIR, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        del self.index
        gc.collect()
        self.index = faiss.read_index(self.index_path, faiss.IO_FLAG_MMAP)

        records = [(t, ) for t in corpus]
        self.cursor.executemany(
            f"INSERT INTO {self.table_name}(term) VALUES(?)", records)
        self.conn.commit()
        self._ids.extend(r[0] for r in self.cursor.execute(
            f"SELECT id FROM {self.table_name} ORDER BY id"))

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
