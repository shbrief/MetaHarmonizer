import os
import sqlite3
import faiss
import asyncio
import httpx
import gc
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import OrderedDict
from tqdm.auto import tqdm
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import get_embedding_model_cached
from src.KnowledgeDb import ensure_knowledge_db
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.CustomLogger.custom_logger import CustomLogger

# -------- ENV / CONSTS --------
BASE_DB = os.getenv("VECTOR_DB_PATH") or "src/KnowledgeDb/vector_db.sqlite"
BASE_IDX_DIR = os.getenv("FAISS_INDEX_DIR") or "src/KnowledgeDb/faiss_indexes"
UMLS_API_KEY = os.getenv("UMLS_API_KEY")


class SynonymDict:
    """
    SQLite + FAISS based synonym dictionary with proper ID mapping.
    """

    def __init__(self, category: str, method: str):
        ensure_knowledge_db()
        self.category = category
        self.method = method
        method_clean = method.replace('-', '_')
        self.table_name = f"synonym_{method_clean}_{category}"
        self.index_name = f"synonym_{method_clean}_{category}.index"

        self.db_path = BASE_DB
        self.index_path = os.path.join(BASE_IDX_DIR, self.index_name)
        self.logger = CustomLogger().custlogger(loglevel="INFO")

        # Initialize embedding model
        raw_model = get_embedding_model_cached(self.method)
        self.embedder = EmbeddingAdapter(raw_model)

        # Initialize NCI client
        self.nci_db = NCIDb(UMLS_API_KEY)

        # Ensure directories exist
        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.index_path)).mkdir(parents=True,
                                                     exist_ok=True)

        # Initialize database
        self._ensure_db()

        # Keep a persistent connection for faster queries
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")

        # Initialize FAISS index (using IndexIDMap2)
        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None

        if os.path.exists(self.index_path):
            self.load_index()
            self.logger.info(
                f"📊 Loaded index: FAISS vectors={self.index.ntotal}")

        # Query cache for performance
        self._qcache = OrderedDict()
        self._qcache_cap = 200000

    def close(self):
        """Explicitly close the database connection."""
        if hasattr(self, '_conn') and self._conn:
            try:
                self._conn.close()
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
            self._conn = None

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close connection."""
        self.close()

    def __del__(self):
        """Close connection on cleanup."""
        self.close()

    def _cache_get(self, key):
        v = self._qcache.get(key)
        if v is not None:
            self._qcache.move_to_end(key)
        return v

    def _cache_put(self, key, value):
        self._qcache[key] = value
        self._qcache.move_to_end(key)
        if len(self._qcache) > self._qcache_cap:
            self._qcache.popitem(last=False)

    # ---------------- SQLite Operations ----------------
    def _ensure_db(self):
        """Create table if not exists."""
        with sqlite3.connect(self.db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name}(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    synonym TEXT NOT NULL,
                    official_label TEXT NOT NULL,
                    nci_code TEXT NOT NULL
                );
            """)
            con.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_code 
                ON {self.table_name}(nci_code);
            """)

    def get_indexed_codes(self) -> Set[str]:
        """Get all NCI codes already indexed."""
        cursor = self._conn.cursor()
        rows = cursor.execute(
            f"SELECT DISTINCT nci_code FROM {self.table_name}").fetchall()
        return set(row[0] for row in rows)

    def has_ontology(self) -> bool:
        """Check if this category has any synonyms indexed."""
        cursor = self._conn.cursor()
        count = cursor.execute(
            f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        return count > 0

    def _insert_synonyms_batch(
            self, records: List[Tuple[str, str, str]]) -> List[int]:
        """
        Atomic batch insert with RETURNING clause (SQLite 3.35+).
        Uses chunking to avoid SQLITE_MAX_VARIABLE_NUMBER limits.
        
        Args:
            records: List of (synonym, official_label, nci_code) tuples
            
        Returns:
            List of inserted row IDs
            
        Raises:
            RuntimeError: If SQLite version < 3.35.0
            sqlite3.Error: If insertion fails
        """
        if not records:
            return []

        # Check SQLite version
        if sqlite3.sqlite_version_info < (3, 35, 0):
            raise RuntimeError(
                f"SQLite {sqlite3.sqlite_version} detected. "
                f"Minimum required version is 3.35.0 for atomic batch insert with RETURNING clause. "
                f"Please upgrade your SQLite installation.")

        cursor = self._conn.cursor()
        all_inserted_ids = []

        CHUNK_SIZE = 3000

        try:
            for i in range(0, len(records), CHUNK_SIZE):
                chunk = records[i:i + CHUNK_SIZE]

                placeholders = ', '.join(['(?, ?, ?)'] * len(chunk))
                sql = (
                    f"INSERT INTO {self.table_name}(synonym, official_label, nci_code) "
                    f"VALUES {placeholders} RETURNING id")

                flat_values = [item for record in chunk for item in record]

                cursor.execute(sql, flat_values)
                chunk_ids = [row[0] for row in cursor.fetchall()]
                all_inserted_ids.extend(chunk_ids)

            self._conn.commit()

        except sqlite3.Error as e:
            self._conn.rollback()
            self.logger.error(f"Batch insert failed: {e}")
            raise

        return all_inserted_ids

    def get_meta_by_ids(self, ids: List[int]) -> List[Dict]:
        """Get full metadata for given IDs using persistent connection."""
        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        cursor = self._conn.cursor()
        rows = cursor.execute(
            f"SELECT id, synonym, official_label, nci_code "
            f"FROM {self.table_name} WHERE id IN ({placeholders})",
            ids).fetchall()

        out = [{
            "id": int(rid),
            "synonym": syn,
            "official_label": label,
            "nci_code": code
        } for rid, syn, label, code in rows]

        # Preserve order
        order = {rid: i for i, rid in enumerate(ids)}
        out.sort(key=lambda x: order.get(x["id"], 1e9))
        return out

    # ---------------- FAISS Operations ----------------
    def _new_index(self, dim: int):
        self.dim = dim

        # Create base flat index
        base_index = faiss.IndexFlatIP(dim)

        # Wrap with IndexIDMap2 for custom ID mapping
        cpu_index = faiss.IndexIDMap2(base_index)

        # Use GPU if available
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            self.index = cpu_index

    def _l2_normalize(self, v: np.ndarray) -> np.ndarray:
        if getattr(self.embedder, "model", None) and getattr(
                self.embedder.model, "_is_normalized", False):
            return v.astype("float32", copy=False)

        v = v.astype("float32", copy=False)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-12)

    def save_index(self):
        if self.index is None:
            raise RuntimeError("Index is None.")

        tmp = self.index_path + ".tmp"

        # If using GPU, transfer to CPU first
        if faiss.get_num_gpus() > 0:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, tmp)
        else:
            faiss.write_index(self.index, tmp)

        os.replace(tmp, self.index_path)

    def load_index(self):
        """Load FAISS index from disk."""
        self.index = faiss.read_index(self.index_path)
        self.dim = self.index.d
        out_dim = getattr(self.embedder, "output_dim", None)
        if out_dim and out_dim != self.dim:
            raise RuntimeError(f"Index dim {self.dim} != model dim {out_dim}")

    def _add_to_faiss(self, vectors: np.ndarray, ids: np.ndarray):
        """Add vectors with their SQLite IDs to FAISS."""
        if len(vectors) == 0:
            return

        vec_array = np.array(vectors, dtype=np.float32)
        vec_array = self._l2_normalize(vec_array)
        ids_array = np.array(ids, dtype=np.int64)

        if self.index is None:
            self._new_index(vec_array.shape[1])

        self.index.add_with_ids(vec_array, ids_array)

    # ---------------- Encoding ----------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        return self._l2_normalize(vecs)

    # ---------------- Build Index from NCI API ----------------
    async def build_index_from_codes_async(self,
                                           codes: List[str],
                                           force_rebuild: bool = False):
        """Build index from NCI codes using NCI API (async version)."""
        codes_set = set(codes)

        if force_rebuild:
            self.logger.info(
                f"🔄 Force rebuild: fetching all {len(codes_set)} codes")
            cursor = self._conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            self._conn.commit()
            codes_to_fetch = list(codes_set)
        else:
            indexed_codes = self.get_indexed_codes()
            codes_to_fetch = list(codes_set - indexed_codes)

            if not codes_to_fetch:
                self.logger.info(
                    f"✅ All {len(codes_set)} codes already indexed. Skipping API calls."
                )
                return

            self.logger.info(
                f"📥 Fetching synonyms for {len(codes_to_fetch)} new codes "
                f"(skipping {len(indexed_codes & codes_set)} already indexed)")

        # Fetch concept data from NCI API
        self.logger.info(
            f"Fetching concept data for {len(codes_to_fetch)} codes")

        async with httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.nci_db.concurrency *
                                    self.nci_db.batch_size)) as client:
            concept_data = await self.nci_db.get_custom_concepts_by_codes(
                codes_to_fetch, client)

        if not concept_data:
            self.logger.warning("No concept data fetched from NCI")
            return

        self.logger.info(f"Retrieved {len(concept_data)} concept data entries")

        # Extract synonyms and prepare records
        records = []
        synonyms_list = []

        for code, data in tqdm(concept_data.items(),
                               desc="Extracting synonyms",
                               leave=False):
            if not data:
                continue

            official_label = data.get("name", "").strip()
            if not official_label:
                continue

            synonym_set = set()
            synonym_set.add(official_label)

            if "synonyms" in data:
                for syn_item in data["synonyms"]:
                    if isinstance(syn_item, dict):
                        syn_name = syn_item.get("name", "").strip()
                        if syn_name:
                            synonym_set.add(syn_name)

            for synonym in synonym_set:
                records.append((synonym, official_label, code))
                synonyms_list.append(synonym)

        if not records:
            self.logger.warning("No valid records to insert")
            return

        self.logger.info(
            f"Extracted {len(records)} synonyms from {len(concept_data)} codes"
        )

        # ✅ Insert into SQLite atomically
        self.logger.info("Inserting records into SQLite...")
        inserted_ids = self._insert_synonyms_batch(records)

        # Build FAISS index in batches
        self.logger.info("Building FAISS index...")

        embed_batch_size = 512
        if faiss.get_num_gpus() > 0:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (
                1024**3)
            used_mem = torch.cuda.memory_allocated(0) / (1024**3)
            free_mem = total_mem - used_mem

            if free_mem < 2:
                embed_batch_size = max(64, embed_batch_size // 2)
                self.logger.warning(
                    f"Low GPU memory ({free_mem:.2f}GB), reducing batch size to {embed_batch_size}"
                )
            elif free_mem > 8:
                embed_batch_size = min(2048, embed_batch_size * 2)
                self.logger.info(
                    f"High GPU memory ({free_mem:.2f}GB), increasing batch size to {embed_batch_size}"
                )

        # ✅ Process in batches with proper ID mapping
        for i in tqdm(range(0, len(synonyms_list), embed_batch_size),
                      desc="Embedding batches",
                      leave=False):
            batch_syns = synonyms_list[i:i + embed_batch_size]
            batch_ids = inserted_ids[i:i + embed_batch_size]

            batch_vecs = self.embedder.embed_documents(batch_syns)
            self._add_to_faiss(batch_vecs, np.array(batch_ids))

            del batch_syns, batch_vecs
            if faiss.get_num_gpus() > 0:
                torch.cuda.empty_cache()
            gc.collect()

        # Save index
        self.save_index()
        self.logger.info(
            f"✅ FAISS index built with {self.index.ntotal} vectors")

    def build_index_from_codes(self,
                               codes: List[str],
                               force_rebuild: bool = False):
        """Synchronous wrapper for build_index_from_codes_async."""
        asyncio.run(self.build_index_from_codes_async(codes, force_rebuild))

    # ---------------- Warm Run ----------------
    def warm_run(self, codes: List[str], force_rebuild: bool = False):
        """Warm run: Check if index exists and build if needed."""
        if not force_rebuild and self.has_ontology():
            stats = self.get_stats()
            self.logger.info(
                f"✅ Synonyms already exist for '{self.category}': "
                f"{stats['total_synonyms']} synonyms, {stats['unique_labels']} unique labels"
            )
            return

        self.build_index_from_codes(codes, force_rebuild)

    # ---------------- Search Operations ----------------
    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Semantic search using FAISS with direct ID mapping."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q = (query_text or "").strip()
        if not q:
            return []

        k = min(top_k, self.index.ntotal)

        cache_key = (q, k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [{
                "id": c[0],
                "synonym": c[1],
                "official_label": c[2],
                "nci_code": c[3],
                "score": c[4]
            } for c in cached]

        q_emb = self._encode([q])
        D, I = self.index.search(q_emb, k)

        # ✅ I[0] now contains actual SQLite IDs (not indices)
        db_ids = [int(i) for i in I[0] if i != -1]
        scores = [float(d) for d in D[0][:len(db_ids)]]

        metas = self.get_meta_by_ids(db_ids)
        for m, s in zip(metas, scores):
            m["score"] = s

        frozen = tuple(
            (m["id"], m["synonym"], m["official_label"], m["nci_code"],
             float(m.get("score", 0.0))) for m in metas)
        self._cache_put(cache_key, frozen)

        return metas

    def search_many(self,
                    queries: List[str],
                    top_k: int = 10) -> List[List[Dict]]:
        """Batch search with direct ID mapping."""
        results: List[List[Dict]] = []

        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in queries]

        Q = []
        needed_idx = []
        k = min(top_k, self.index.ntotal)
        cache_keys = []

        for i, q in enumerate(queries):
            q = (q or "").strip()
            if not q:
                results.append([])
                cache_keys.append(None)
                continue

            ck = (q, k)
            cache_keys.append(ck)
            cached = self._cache_get(ck)

            if cached is not None:
                metas = [{
                    "id": c[0],
                    "synonym": c[1],
                    "official_label": c[2],
                    "nci_code": c[3],
                    "score": c[4]
                } for c in cached]
                results.append(metas)
            else:
                results.append(None)  # type: ignore
                needed_idx.append(i)
                Q.append(q)

        if not Q:
            return results

        embs = self._encode(Q)
        D, I = self.index.search(embs, k)

        for j, i_query in enumerate(needed_idx):
            row_I = I[j]
            row_D = D[j]

            # row_I contains actual SQLite IDs
            db_ids = [int(i) for i in row_I if i != -1]
            scores = [float(d) for d in row_D[:len(db_ids)]]

            metas = self.get_meta_by_ids(db_ids)
            for m, s in zip(metas, scores):
                m["score"] = s

            results[i_query] = metas  # type: ignore

            ck = cache_keys[i_query]
            frozen = tuple(
                (m["id"], m["synonym"], m["official_label"], m["nci_code"],
                 float(m.get("score", 0.0))) for m in metas)
            self._cache_put(ck, frozen)

        return results

    def get_stats(self) -> Dict:
        """Get statistics."""
        cursor = self._conn.cursor()
        total = cursor.execute(
            f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        unique_labels = cursor.execute(
            f"SELECT COUNT(DISTINCT official_label) FROM {self.table_name}"
        ).fetchone()[0]
        unique_codes = cursor.execute(
            f"SELECT COUNT(DISTINCT nci_code) FROM {self.table_name}"
        ).fetchone()[0]

        return {
            "total_synonyms": total,
            "unique_labels": unique_labels,
            "unique_codes": unique_codes,
            "index_size": self.index.ntotal if self.index else 0
        }
