import torch
import numpy as np
from sentence_transformers import util
import pandas as pd
from metaharmonizer.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch
from metaharmonizer.utils.model_loader import load_method_model_dict, get_reranker_cached
from metaharmonizer.models.reranker import RERANKER_TYPE_MAP
from metaharmonizer.CustomLogger.custom_logger import CustomLogger


class OntoModelsBase:

    def __init__(
        self,
        method: str,
        category: str,
        om_strategy: str,
        topk: int,
        query: list,
        corpus: list,
        query_df: pd.DataFrame = None,
        corpus_df: pd.DataFrame = None,
        ontology_source: str = 'ncit',
        table_suffix: str = "",
    ) -> None:
        self.method = method
        self.category = category
        self.ontology_source = ontology_source
        self.om_strategy = om_strategy
        self.query = query
        self.corpus = corpus
        self.query_df = query_df
        self.corpus_df = corpus_df
        self.table_suffix = table_suffix

        if self.method is None:
            raise ValueError("Method name cannot be None")

        if om_strategy in ["rag_bie", "rag", "syn"] and corpus_df is None:
            raise ValueError(
                "corpus_df must be provided when om_strategy is 'rag_bie', 'rag', or 'syn'"
            )
        if om_strategy == "rag_bie" and query_df is None:
            raise ValueError(
                "query_df must be provided when om_strategy is 'rag_bie'")

        if len(self.query) == 0:
            raise ValueError("Query list cannot be empty")
        if len(self.corpus) == 0:
            raise ValueError("Corpus list cannot be empty")

        if om_strategy != 'syn':
            # Load method_model_dict from a YAML file
            self.method_model_dict = load_method_model_dict()
            self.list_of_methods = list(self.method_model_dict.keys())
            if self.method not in self.list_of_methods:
                raise ValueError(
                    f"Method name should be one of {self.list_of_methods}")
        else:
            self.method_model_dict = {}
            self.list_of_methods = []

        self.topk = topk
        self.logger = CustomLogger().custlogger(loglevel='INFO')

    @property
    def vector_store(self):
        if not hasattr(self,
                       "_vs") or self._vs is None or self._vs.index is None:
            store = FAISSSQLiteSearch(method=self.method,
                                      category=self.category,
                                      om_strategy=self.om_strategy,
                                      ontology_source=self.ontology_source,
                                      table_suffix=self.table_suffix)

            # For rag & rag bie, use ConceptTableBuilder
            if self.om_strategy in ("rag", "rag_bie"):
                if self.corpus_df is None:
                    raise ValueError("corpus_df is required for RAG strategy")

                store.ensure_corpus_integrity(self.corpus, self.corpus_df)

            # For ST/LM: use corpus list
            elif self.corpus:
                store.ensure_corpus_integrity(self.corpus, None)

            self._vs = store

            self.logger.info(
                f"{self._vs.index is not None} - Vector store initialized for "
                f"method={self.method}, category={self.category}, om_strategy={self.om_strategy}"
            )

        return self._vs

    def calc_similarity(self, query_emb, corpus_emb):
        """
        Function to create embeddings for sentence transformer model

        ARGS:
            query_list: list of str items
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings

        RETURNS:
            numpy or tensor datatype for output embeddings
        """
        cosine_scores = util.cos_sim(query_emb, corpus_emb)
        cosine_sim_df = pd.DataFrame(cosine_scores)
        return cosine_sim_df

    # ---------------- Reranker (shared by RAG / BIE subclasses) ----------------
    def _init_reranker(self, use_reranker: bool, reranker_method: str, reranker_topk: int):
        self.use_reranker = use_reranker
        if use_reranker:
            if reranker_method not in RERANKER_TYPE_MAP:
                raise ValueError(
                    f"Unknown reranker_method '{reranker_method}'. "
                    f"Must be one of {list(RERANKER_TYPE_MAP.keys())}"
                )
            self.reranker_method = reranker_method
            self.reranker_topk = reranker_topk
        else:
            self.reranker_method = None
            self.reranker_topk = None
        self._reranker = None

    @property
    def reranker(self):
        """Lazy-load reranker on first use."""
        if self._reranker is None and self.use_reranker:
            reranker_type = RERANKER_TYPE_MAP.get(self.reranker_method)
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
                use_8bit = (vram < 20e9) and (reranker_type in ["t5", "generative"])
            else:
                use_8bit = False
            self.logger.info(f"Loading {reranker_type} reranker: {self.reranker_method}")
            self._reranker = get_reranker_cached(self.reranker_method, use_8bit=use_8bit)
        return self._reranker

    def _rerank_results(self, query: str, candidates: list, topk: int) -> list:
        """Rerank FAISS candidates using cross-encoder; returns top-k."""
        if not candidates or not self.use_reranker:
            return candidates[:topk]
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked_indices = np.argsort(-scores)[:topk]
        reranked = []
        for idx in ranked_indices:
            doc = candidates[idx]
            doc.metadata['similarity_score'] = doc.metadata.get('score', 0)
            doc.metadata['reranker_score'] = float(scores[idx])
            doc.metadata['score'] = float(scores[idx])
            reranked.append(doc)
        return reranked

    ##### To be implemented in the child class #####
    def create_embeddings(self):
        """
        This method is responsible for creating embeddings using sentence transformers, LM's or LLM's for the cBioPortal's queries or corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError(
            "create_embeddings will be implemented in the child class")

    def _build_result_rows(self, I, D, cura_map, test_or_prod):
        """
        Builds result rows from FAISS search output (I = index matrix, D = score matrix).

        Args:
            I (np.ndarray): Top-k corpus indices per query, shape (n_queries, topk).
            D (np.ndarray): Top-k scores per query, shape (n_queries, topk).
            cura_map (dict[str, str] or None): Curated label map for test evaluation.
            test_or_prod (str): 'test' to look up ground truth; any other value skips it.

        Returns:
            pd.DataFrame: One row per query with match columns and scores.
        """
        # Map FAISS positions → DB IDs using the persisted _ids list,
        # then look up term strings from the DB.
        vs = self.vector_store
        db_rows = vs.cursor.execute(
            f"SELECT id, term FROM {vs.table_name}"
        ).fetchall()
        _id_to_term = {r[0]: r[1] for r in db_rows}

        rows = []
        for qi, q in enumerate(self.query):
            top_ids = I[qi]
            top_scores = D[qi].tolist()
            top_terms = [
                _id_to_term.get(vs._ids[i], "") if i < len(vs._ids) else ""
                for i in top_ids
            ]

            curated = (cura_map.get(q, "Not Found") if test_or_prod == 'test'
                       else "Not Available for Prod Environment")
            lvl = next(
                (i + 1 for i, t in enumerate(top_terms)
                 if (t or "").strip().lower() == (curated or "").strip().lower()), 99)

            row = {
                "original_value": q,
                "curated_ontology": curated,
                "match_level": lvl
            }
            for i, (t, s) in enumerate(zip(top_terms, top_scores), start=1):
                row[f"match{i}"] = t
                row[f"match{i}_score"] = f"{s:.4f}"
            rows.append(row)

        return pd.DataFrame(rows)

    def get_match_results(self):
        """
        This method is responsible for generating match results for the given queries and corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError(
            "get_match_results will be implemented in the child class")
