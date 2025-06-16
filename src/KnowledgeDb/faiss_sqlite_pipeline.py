import re
from typing import Dict, List
import grequests
import numpy as np
import requests
import json
import sqlite3
import faiss
import os
from src.utils.embeddings import EmbeddingAdapter
from src.utils.model_loader import DEFAULT_YAML_PATH, get_embedding_model_cached


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
            return [res["ui"] for res in results if res["rootSource"] == "NCI"]
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
            "children",
            "roles",
            "definitions",
            "parents",
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

        for concept in list_of_concepts:
            if concept not in concept_data:
                continue

            parts = []

            if concept == "roles":
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        role_type = item.get("type")
                        role_target = item.get("relatedName")
                        if role_type and role_target:
                            parts.append(f"{role_type}: {role_target}")
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
            else:  # synonyms, children, parents
                names = []
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        if name:
                            names.append(name)
                parts = list(set(names)) if concept == "synonyms" else names

            if parts:
                context.append(f"{concept}: {'; '.join(parts)}")

        return ". ".join(context)

    # def create_context_list(
    #     self,
    #     concepts_for_curated_term2: dict[str],
    #     list_of_concepts: List[str] = [
    #         "synonyms", "children", "roles", "definitions", "parents"
    #     ]
    # ) -> str:
    #     """
    #     Create a context list from the given concepts.

    #     :param concepts_for_curated_term2: A dictionary of concepts for a curated term.
    #     :return: A string representing the context list.
    #     """
    #     context_list = []
    #     concepts = ["synonyms", "children", "roles", "definitions", "parents"]
    #     for concept in concepts:
    #         if concept in concepts_for_curated_term2:
    #             if concept == "synonyms":
    #                 str_tmp = "The synonyms for the term are:"
    #                 names = {
    #                     syn["name"]
    #                     for syn in concepts_for_curated_term2[concept]
    #                     if "name" in syn
    #                 }
    #                 for name in names:
    #                     str_tmp += name + ", "
    #                 context_list.append(str_tmp)
    #             elif concept == "children":
    #                 str_tmp = "The children of the term are:"
    #                 for child in concepts_for_curated_term2[concept]:
    #                     str_tmp += child["name"] + ", "
    #                 context_list.append(str_tmp)
    #             elif concept == "parents":
    #                 str_tmp = "The parents of the term are:"
    #                 for parent in concepts_for_curated_term2[concept]:
    #                     str_tmp += parent["name"] + ", "
    #                 context_list.append(str_tmp)
    #             elif concept == "roles":
    #                 str_tmp = "The roles of the term are:"
    #                 for role in concepts_for_curated_term2[concept]:
    #                     str_tmp += role["type"] + " " + role[
    #                         "relatedName"] + ", "
    #                 context_list.append(str_tmp)
    #             elif concept == "definitions":
    #                 str_tmp = "The definitions for the term are:"
    #                 seen_defs = set()
    #                 for definition in concepts_for_curated_term2[concept]:
    #                     raw_def = definition["definition"]
    #                     cleaned_def = raw_def.strip().rstrip(".,;")

    #                     norm_def = ' '.join(cleaned_def.split())
    #                     norm_def = norm_def.replace(" (NCI)",
    #                                                 "").replace("(NCI)", "")
    #                     norm_def_lower = norm_def.lower()

    #                     if norm_def_lower not in seen_defs:
    #                         seen_defs.add(norm_def_lower)
    #                         str_tmp += " " + norm_def + ","
    #                 context_list.append(str_tmp)

    #     return ".".join(context_list)

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

    def __init__(self,
                 db_path="src/KnowledgeDb/vector_db.sqlite",
                 index_path="src/KnowledgeDb/faiss.index",
                 table_name: str = "term_info"):
        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}. ")
        self.db_path = db_path
        self.index_path = index_path
        self.table_name = table_name

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                term TEXT,
                code TEXT NOT NULL UNIQUE,
                context TEXT NOT NULL
            )
        """)
        self.conn.commit()

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
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
        v = v.astype('float32', copy=False)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-12)

    def _insert_to_sqlite(self, records: List[tuple]) -> List[int]:
        self.cursor.executemany(
            f"INSERT OR IGNORE INTO {self.table_name} (term, code, context) VALUES (?, ?, ?)",
            records)
        self.conn.commit()

        codes = [code for _, code, _ in records]
        placeholders = ','.join('?' * len(codes))
        self.cursor.execute(
            f"SELECT code, id FROM {self.table_name} WHERE code IN ({placeholders})",
            codes)
        code_to_id = {row[0]: row[1] for row in self.cursor.fetchall()}

        inserted_ids = []
        for _, code, _ in records:
            if code not in code_to_id:
                print(
                    f"Warning: Code {code} not found in database after insertion"
                )
                continue
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

    def fetch_and_store_terms(self,
                              terms,
                              api_key,
                              method,
                              yaml_path: str = DEFAULT_YAML_PATH):
        db = NCIDb(api_key)
        raw_model = get_embedding_model_cached(method, yaml_path)
        embedder = EmbeddingAdapter(raw_model)

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
                contexts.append(combined)
                records.append((term, code, combined))

        if not records:
            print("No new records to insert.")
            return

        inserted_ids = self._insert_to_sqlite(records)
        self._ids.extend(inserted_ids)

        vectors = embedder.embed_documents(contexts)
        self._insert_to_faiss(vectors)

    def similarity_search(self,
                          query: str,
                          method: str,
                          yaml_path: str = DEFAULT_YAML_PATH,
                          k: int = 5,
                          as_documents: bool = False):
        raw_model = get_embedding_model_cached(method, yaml_path)
        embedder = EmbeddingAdapter(raw_model)
        vector = embedder.embed_query(query)
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
