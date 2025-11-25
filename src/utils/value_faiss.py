from collections import OrderedDict
import os, json, re, sqlite3, numpy as np, pandas as pd
from typing import List, Dict, Iterable, Tuple, Optional
from pathlib import Path
import faiss
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import get_embedding_model_cached
from src.utils.schema_mapper_utils import normalize
from src.CustomLogger.custom_logger import CustomLogger

# -------- ENV / CONSTS --------
BASE_DB = os.getenv("VECTOR_DB_PATH") or "src/KnowledgeDb/vector_db.sqlite"
BASE_IDX_DIR = os.getenv("FAISS_INDEX_DIR") or "src/KnowledgeDb/faiss_indexes"
MODEL_NAME = "mt-sap-bert"
JSON_VOCAB = os.getenv("FIELD_VALUE_JSON") or "field_value_dict.json"
IDX_NAME = "dict_value.index"


class ValueFAISSStore:
    """
    SQLite: items(id INTEGER PRIMARY KEY, value TEXT, value_norm TEXT UNIQUE, fields TEXT(JSON))
    FAISS:  IndexIDMap2(FlatIP) — Vector embeddings with IDs from SQLite.
    """

    def __init__(self):
        self.db_path = BASE_DB
        self.index_path = os.path.join(BASE_IDX_DIR, IDX_NAME)
        self.logger = CustomLogger().custlogger(loglevel="INFO")

        raw_model = get_embedding_model_cached(MODEL_NAME)
        self.embedder = EmbeddingAdapter(raw_model)

        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.index_path)).mkdir(parents=True,
                                                     exist_ok=True)

        self._ensure_db()

        self.index: Optional[faiss.Index] = None
        self.dim: Optional[int] = None
        if os.path.exists(self.index_path):
            self.load_index()

        self._qcache = OrderedDict()
        self._qcache_cap = 200000

        self._sync_with_vocab(JSON_VOCAB)

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

    # ---------------- SQLite ----------------
    def _ensure_db(self):
        with sqlite3.connect(self.db_path) as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("""
                CREATE TABLE IF NOT EXISTS items(
                id INTEGER PRIMARY KEY,
                value TEXT NOT NULL UNIQUE,
                fields TEXT NOT NULL
            );
            """)

    def _insert_or_merge(self, value: str, fields: Iterable[str]) -> int:
        """
        Insert or merge a record (deduplicated by value; fields are merged as a set). 
        Returns the ID of the row.
        """
        v = (value or "").strip()

        fset = sorted({x for x in (fields or []) if x})
        fjson = json.dumps(fset, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT id, fields FROM items WHERE value = ?", (v, ))
            row = cur.fetchone()
            if row:
                rid, f_old = row
                old_set = set(json.loads(f_old))
                new_set = sorted(old_set.union(fset))
                if new_set != old_set:
                    cur.execute("UPDATE items SET fields=? WHERE id=?",
                                (json.dumps(new_set, ensure_ascii=False), rid))
                return rid
            cur.execute("INSERT INTO items(value, fields) VALUES(?,?)",
                        (v, fjson))
            return cur.lastrowid

    def _get_all_id_value(self) -> List[Tuple[int, str]]:
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                "SELECT id, value FROM items ORDER BY id").fetchall()
        return [(int(rid), val) for rid, val in rows]

    def get_meta_by_ids(self, ids: List[int]) -> List[Dict]:
        if not ids: return []
        q = ",".join(["?"] * len(ids))
        with sqlite3.connect(self.db_path) as con:
            rows = con.execute(
                f"SELECT id, value, fields FROM items WHERE id IN ({q})",
                ids).fetchall()
        out = [{
            "id": int(rid),
            "value": val,
            "fields": json.loads(fjson)
        } for rid, val, fjson in rows]
        order = {rid: i for i, rid in enumerate(ids)}
        out.sort(key=lambda x: order.get(x["id"], 1e9))
        return out

    # ---------------- FAISS ----------------
    def _new_index(self, dim: int):
        self.dim = dim
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)

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
        faiss.write_index(self.index, tmp)
        os.replace(tmp, self.index_path)

    def load_index(self):
        self.index = faiss.read_index(self.index_path)
        self.dim = self.index.d
        out_dim = getattr(self.embedder, "output_dim", None)
        if out_dim and out_dim != self.dim:
            raise RuntimeError(f"Index dim {self.dim} != model dim {out_dim}")

    def _remove_ids_if_exist(self, ids: List[int]):
        if not ids or self.index is None:
            return
        selector = faiss.IDSelectorBatch(np.asarray(ids, dtype=np.int64))
        self.index.remove_ids(selector)

    def _index_ids(self) -> set:
        if self.index is None:
            return set()
        try:
            return set(map(int, faiss.vector_to_array(self.index.id_map)))
        except Exception:
            return set()

    # ---------------- Encode ----------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        vecs = self.embedder.embed_documents(texts)
        vecs = np.asarray(vecs, dtype=np.float32)
        return self._l2_normalize(vecs)

    # ---------------- Build / Update ----------------
    def add_values(self, pairs: List[Tuple[str, Iterable[str]]]):
        """
        Add or update (value, fields) pairs in the database and FAISS index.
        """
        ids, texts = [], []
        for val, fields in pairs:
            rid = self._insert_or_merge(val, fields)
            ids.append(rid)
            texts.append(val)
        if not texts:
            return

        embs = self._encode(texts)
        if self.index is None:
            self._new_index(embs.shape[1])

        self._remove_ids_if_exist(ids)
        self.index.add_with_ids(embs, np.asarray(ids, dtype=np.int64))
        self.save_index()

    # ---------------- Query ----------------
    def _pack_hits(self, ids: List[int], sims: List[float]) -> List[Dict]:
        metas = self.get_meta_by_ids(ids)
        for m, s in zip(metas, sims):
            m["score"] = s
        return metas

    def search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = (query_text or "").strip()
        if not q:
            return []
        k = min(top_k, self.index.ntotal)

        cache_key = (normalize(q), k)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return [{
                "id": c[0],
                "value": c[1],
                "fields": list(c[3]),
                "score": c[2]
            } for c in cached]

        q_emb = self._encode([q])
        D, I = self.index.search(q_emb, k)
        ids = [int(i) for i in I[0] if i != -1]
        sims = [float(d) for d in D[0][:len(ids)]]
        metas = self._pack_hits(ids, sims)

        frozen = tuple((m["id"], m["value"], float(m.get("score", 0.0)),
                        tuple(m["fields"])) for m in metas)
        self._cache_put(cache_key, frozen)
        return metas

    def search_many(self,
                    queries: List[str],
                    top_k: int = 10) -> List[List[Dict]]:
        """
        Batch search for multiple queries.
        """
        results: List[List[Dict]] = []
        if self.index is None or self.index.ntotal == 0:
            return [[] for _ in range(len(queries))]

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
            ck = (normalize(q), k)
            cache_keys.append(ck)
            cached = self._cache_get(ck)
            if cached is not None:
                metas = [{
                    "id": c[0],
                    "value": c[1],
                    "fields": list(c[3]),
                    "score": c[2]
                } for c in cached]
                results.append(metas)
            else:
                results.append(None)  # type: ignore
                needed_idx.append(i)
                Q.append(q)

        if not Q:
            return results

        embs = self._encode(Q)  # shape = [len(Q), dim]
        D, I = self.index.search(embs, k)  # batched search

        for j, i_query in enumerate(needed_idx):
            row_I = I[j]
            row_D = D[j]
            ids = [int(i) for i in row_I if i != -1]
            sims = [float(d) for d in row_D[:len(ids)]]
            metas = self._pack_hits(ids, sims)
            results[i_query] = metas  # type: ignore

            ck = cache_keys[i_query]
            frozen = tuple((m["id"], m["value"], float(m.get("score", 0.0)),
                            tuple(m["fields"])) for m in metas)
            self._cache_put(ck, frozen)

        return results

    # ---------------- Bootstrap / Sync ----------------
    def _load_vocab_pairs(self,
                          json_path: str) -> List[Tuple[str, Iterable[str]]]:
        """
        Load vocabulary pairs from a JSON file.
        """
        if not os.path.exists(json_path):
            self.logger.warning(
                f"JSON vocab not found at {json_path}. Skip syncing.")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rev: Dict[str, set] = {}
        total_pairs = 0
        for field, vals in data.items():
            if not isinstance(vals, (list, tuple)):
                continue
            for v in vals:
                if not isinstance(v, str):
                    continue
                rev.setdefault(v, set()).add(field)
                total_pairs += 1

        if total_pairs > 0:
            return [(v, sorted(list(fs))) for v, fs in rev.items()]

        pairs = []
        for v, fields in data.items():
            if isinstance(fields, (list, tuple)) and all(
                    isinstance(x, str) for x in fields):
                pairs.append((v, sorted(set(fields))))
        return pairs

    def _sync_with_vocab(self, json_path: str):
        """
        Sync vocabulary with the database and FAISS index.
        """
        pairs = self._load_vocab_pairs(json_path)
        if not pairs:
            return

        ids, texts = [], []
        for val, fields in pairs:
            rid = self._insert_or_merge(val, fields)
            ids.append(rid)
            texts.append(val)

        if self.index is None:
            all_rows = self._get_all_id_value()
            if not all_rows:
                return

            probe = self._encode([all_rows[0][1]])
            self._new_index(probe.shape[1])

            B = 2048
            for i in range(0, len(all_rows), B):
                batch = all_rows[i:i + B]
                batch_ids = np.asarray([rid for rid, _ in batch],
                                       dtype=np.int64)
                batch_txt = [val for _, val in batch]
                embs = self._encode(batch_txt)
                self.index.add_with_ids(embs, batch_ids)
            self.save_index()
            return

        index_ids = self._index_ids()
        missing = [(rid, txt) for rid, txt in zip(ids, texts)
                   if rid not in index_ids]
        if not missing:
            return
        B = 4096
        for i in range(0, len(missing), B):
            chunk = missing[i:i + B]
            chunk_ids = np.asarray([rid for rid, _ in chunk], dtype=np.int64)
            chunk_txt = [txt for _, txt in chunk]
            embs = self._encode(chunk_txt)

            self.index.add_with_ids(embs, chunk_ids)
        self.save_index()
