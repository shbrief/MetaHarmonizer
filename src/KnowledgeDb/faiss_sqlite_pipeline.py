import re
import gc
from typing import Dict, List
import grequests
import numpy as np
import requests
import json
import sqlite3
import faiss
import os
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import get_embedding_model_cached
from concurrent.futures import ThreadPoolExecutor

BASE_DB = os.getenv("VECTOR_DB_PATH")
BASE_IDX_DIR = os.getenv("FAISS_INDEX_DIR")
UMLS_API_KEY = os.getenv("UMLS_API_KEY")


class UMLSDb:

    def __init__(self, api_key):
        self.api_key = api_key
        self._base_url = "https://uts-ws.nlm.nih.gov/rest"

    def get_nci_code_by_term(self, term: str):
        uri = f"{self._base_url}/search/current"
        query = {
            "string": term,
            "apiKey": self.api_key,
            "pageNumber": 1,
            "searchType": "exact",
            "sabs": "NCI",
            "returnIdType": "code",
        }
        try:
            r = requests.get(uri, params=query)
            r.raise_for_status()
            results = r.json()["result"]["results"]
            codes = [
                res["ui"] for res in results if res.get("rootSource") == "NCI"
            ]
            if not codes:
                raise LookupError(f"No NCI code found for term '{term}'")
            return codes
        except Exception as e:
            print(f"Error fetching NCI code for '{term}': {e}")
            return []


class NCIDb:

    def __init__(self, umls_api_key):
        self._base_url = "https://api-evsrest.nci.nih.gov/api/v1"
        self._umls_db = UMLSDb(umls_api_key)

    def get_custom_concepts_by_codes(self, codes, list_of_concepts):
        """
        Fetches custom concepts from the NCI EVS API for a list of codes.
        Args:
            codes (list): List of NCI codes to fetch concepts for.
            list_of_concepts (list): List of concepts to include in the response.
        Returns:
            dict: A dictionary where keys are NCI codes and values are the corresponding concept data.
        """
        urls = [
            f"{self._base_url}/concept/ncit/{code}?include={','.join(list_of_concepts)}"
            for code in codes
        ]
        rs = (grequests.get(u) for u in urls)
        responses = grequests.map(rs)
        result = {}
        for code, response in zip(codes, responses):
            if response and response.status_code == 200:
                result[code] = json.loads(response.text)
        return result

    def create_context_list(
        self,
        concept_data: dict,
        list_of_concepts: List[str] = [
            "synonyms",
            "definitions",
            "parents",
            "children",
            "roles",
        ],
    ) -> str:
        """
        Convert a concept dictionary into a structured context string.
        Args:
            concept_data (dict): The concept data fetched from the NCI API.
            list_of_concepts (list): List of concepts to include in the context. Defaults to ["synonyms", "children", "roles", "definitions", "parents"].
        Returns:
            str: A string representation of the context, formatted as "concept: item1, item2, ...".
        """
        context = []
        excluded_prefixes = set([
            "Gene_", "Allele_", "Molecular_Abnormality_Involves_Gene",
            "Cytogenetic_Abnormality_Involves_Chromosome", "EO_Disease_",
            "Conceptual_Part_Of", "Chemotherapy_Regimen_Has_Component",
            "Biological_Process_"
        ])

        for concept in list_of_concepts:
            if concept not in concept_data:
                continue
            parts = []
            if concept == "roles":
                role_map = {}
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        role_type = item.get("type")
                        role_target = item.get("relatedName")
                        if any(
                                role_type.startswith(prefix)
                                for prefix in excluded_prefixes):
                            continue
                        if role_type and role_target:
                            role_map.setdefault(role_type,
                                                []).append(role_target)
                simplified_role_map = {}
                for role_type, targets in role_map.items():
                    key = re.sub(
                        r"^(?:Disease_|Procedure_)?(?:Has|Is|May_Have|Mapped_To|Excludes|Has_Accepted_Use_For)?_?",
                        "", role_type)
                    simplified_role_map.setdefault(key, set()).update(targets)
                for role_type, target_set in simplified_role_map.items():
                    target_list = list(target_set)[:10]  # Limit to 10 targets
                    parts.append(f"{role_type}: {'; '.join(target_list)}")
            elif concept == "definitions":
                seen_defs = set()
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        definition = item.get("definition", "")
                        if definition:
                            cleaned_def = definition.strip()
                            cleaned_def = re.sub(r'[.;\s]+$', '', cleaned_def)
                            norm_def = ' '.join(cleaned_def.split())
                            norm_def = re.sub(r'[\s\.;,:-]*\(?NCI\)?$',
                                              '',
                                              norm_def,
                                              flags=re.IGNORECASE)
                            norm_def_lower = norm_def.lower()
                            if norm_def_lower not in seen_defs:
                                seen_defs.add(norm_def_lower)
                                parts.append(norm_def)
            else:
                names = []
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        if name:
                            names.append(name)
                parts = list(set(names)) if concept == "synonyms" else list(
                    set(names))[:10]
            if parts:
                context.append(f"{concept}: {'; '.join(parts)}")
        return ". ".join(context)

    # def create_context_list(
    #     self,
    #     concept_data: dict,
    #     list_of_concepts: List[str] = [
    #         "synonyms",
    #         "children",
    #         "roles",
    #         "definitions",
    #         "parents",
    #     ],
    # ) -> str:
    #     """
    #     Convert a concept dictionary into a structured context string.
    #     Args:
    #         concept_data (dict): The concept data fetched from the NCI API.
    #         list_of_concepts (list): List of concepts to include in the context. Defaults to ["synonyms", "children", "roles", "definitions", "parents"].
    #     Returns:
    #         str: A string representation of the context, formatted as "concept: item1, item2, ...".
    #     """
    #     context = []

    #     for concept in list_of_concepts:
    #         if concept not in concept_data:
    #             continue

    #         parts = []

    #         if concept == "roles":
    #             for item in concept_data[concept]:
    #                 if isinstance(item, dict):
    #                     role_type = item.get("type")
    #                     role_target = item.get("relatedName")
    #                     if role_type and role_target:
    #                         parts.append(f"{role_type}: {role_target}")
    #         elif concept == "definitions":
    #             seen_defs = set()
    #             for item in concept_data[concept]:
    #                 if isinstance(item, dict):
    #                     definition = item.get("definition", "")
    #                     if definition:
    #                         cleaned_def = definition.strip()
    #                         cleaned_def = re.sub(r'[.;\s]+$', '', cleaned_def)
    #                         norm_def = ' '.join(cleaned_def.split())
    #                         norm_def = re.sub(r'[\s\.;,:-]*\(?NCI\)?$',
    #                                           '',
    #                                           norm_def,
    #                                           flags=re.IGNORECASE)
    #                         norm_def_lower = norm_def.lower()
    #                         if norm_def_lower not in seen_defs:
    #                             seen_defs.add(norm_def_lower)
    #                             parts.append(norm_def)
    #         else:  # synonyms, children, parents
    #             names = []
    #             for item in concept_data[concept]:
    #                 if isinstance(item, dict):
    #                     name = item.get("name", "")
    #                     if name:
    #                         names.append(name)
    #             parts = list(set(names)) if concept == "synonyms" else names

    #         if parts:
    #             context.append(f"{concept}: {'; '.join(parts)}")

    #     return ". ".join(context)

    def get_context_map_by_codes(self, codes: List[str],
                                 fields: List[str]) -> Dict[str, str]:
        """ 
        Fetches concepts for a list of NCI codes and creates a context map.
        Args:
            codes (list): List of NCI codes to fetch concepts for.
            fields (list): List of fields to include in the context.
        Returns:
            dict: A dictionary where keys are NCI codes and values are the corresponding context strings.
        """
        concept_map = self.get_custom_concepts_by_codes(codes, fields)
        return {
            code: self.create_context_list(data, fields)
            for code, data in concept_map.items()
        }


class Document:

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class FAISSSQLiteSearch:

    def __init__(
        self,
        method: str,
        category: str,
        om_strategy: str = "rag",
    ):
        self.db_path = BASE_DB
        idx_name = f"{om_strategy}_{method}_{category}.index"
        self.index_path = os.path.join(BASE_IDX_DIR, idx_name)
        self.table_name = f"{om_strategy}_{method.replace('-', '_')}_{category}"
        self.method = method
        self.om_strategy = om_strategy
        raw_model = get_embedding_model_cached(method)
        self.embedder = EmbeddingAdapter(raw_model)

        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        if self.om_strategy == "rag":
            create_sql = f"""
              CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                context TEXT NOT NULL,
                token_length INTEGER
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

    def _get_existing_terms(self):
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
        self.cursor.executemany(
            f"INSERT OR IGNORE INTO {self.table_name} (term, code, context, token_length) VALUES (?, ?, ?, ?)",
            records)
        self.conn.commit()
        codes = [rec[1] for rec in records]
        placeholders = ','.join('?' * len(codes))
        self.cursor.execute(
            f"SELECT code, id FROM {self.table_name} WHERE code IN ({placeholders})",
            codes)
        code_to_id = {row[0]: row[1] for row in self.cursor.fetchall()}
        inserted_ids = []
        for _, code, _, _ in records:
            if code in code_to_id:
                inserted_ids.append(code_to_id[code])
        return inserted_ids

    def _insert_to_faiss(self, vectors):
        if not vectors:
            return
        vec_array = np.array(vectors, dtype=np.float32)
        vec_array = self.normalize(vec_array)
        if self.index is None:
            dim = vec_array.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(vec_array)
        faiss.write_index(self.index, self.index_path)

    def fetch_and_store_terms(self, terms, api_key=UMLS_API_KEY):
        db = NCIDb(api_key)
        tokenizer = self.embedder.tokenizer
        existing_terms = self._get_existing_terms()
        new_terms = [term for term in terms if term not in existing_terms]
        contexts = []
        records = []
        seen_codes = set()

        for term in new_terms:
            codes = db._umls_db.get_nci_code_by_term(term)
            if not codes:
                continue
            concept_data = db.get_custom_concepts_by_codes(
                codes,
                ["synonyms", "children", "roles", "definitions", "parents"])
            for code in codes:
                if code not in concept_data or code in seen_codes:
                    continue
                seen_codes.add(code)
                context = db.create_context_list(concept_data[code])
                combined = f"{term}: {context}"
                token_length = len(tokenizer.tokenize(combined))
                contexts.append(combined)
                records.append((term, code, combined, token_length))

        if not records:
            print("No new records to insert.")
            return

        inserted_ids = self._insert_to_sqlite(records)
        self._ids.extend(inserted_ids)

        for i in range(0, len(contexts), 512):
            batch_ctx = contexts[i:i + 512]
            batch_vecs = self.embedder.embed_documents(batch_ctx)
            self._insert_to_faiss(batch_vecs)

            del batch_ctx, batch_vecs
            gc.collect()

    def build_corpus_vector_db(self, corpus: List[str]):
        if not corpus:
            raise ValueError("Corpus cannot be empty.")

        if self.om_strategy == "rag":
            raise NotImplementedError(
                "RAG strategy should use fetch_and_store_terms method.")

        # Initialize the embedder and dimensions with the first batch
        first_batch = corpus[:256]
        first_vecs = self.embedder.embed_documents(first_batch)
        dim = len(first_vecs[0])
        self.index = faiss.IndexFlatIP(dim)
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
                f"SELECT term, context, code, token_length FROM {self.table_name} WHERE id = ?",
                (record_id, )).fetchone()
            if row:
                term, context, code, token_length = row
                result = {
                    "term": term,
                    "context": context,
                    "code": code,
                    "token_length": token_length,
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
