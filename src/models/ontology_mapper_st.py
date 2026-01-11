import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
import src.models.ontology_models as otm
from src.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch


class OntoMapST(otm.OntoModelsBase):
    """
    A class to map ontologies using sentence transformers.

    Attributes:
        method (str): The method to use for the model.
        query (list[str]): The list of query strings.
        corpus (list[str]): The list of corpus strings.
        topk (int): The number of top results to consider.
        from_tokenizer (bool): Whether to use a tokenizer.
        _query_embeddings (torch.Tensor or None): Embeddings for the queries.
        _corpus_embeddings (torch.Tensor or None): Embeddings for the corpus.
        _model (AutoModel or SentenceTransformer or None): The model instance.
        _tokenizer (AutoTokenizer or None): The tokenizer instance.
        logger (CustomLogger): Logger instance.
    """

    def __init__(self,
                 method: str,
                 category: str,
                 query: list[str],
                 corpus: list[str],
                 om_strategy: str = 'st',
                 topk: int = 5,
                 from_tokenizer: bool = False) -> None:
        """
        Initializes the OntoMapST class.

        Args:
            method (str): The method to use for the model.
            topk (int): The number of top results to consider.
            query (list[str]): The list of query strings.
            corpus (list[str]): The list of corpus strings.
            from_tokenizer (bool, optional): Whether to use a tokenizer. Defaults to False.
        """
        super().__init__(method, category, om_strategy, topk, query, corpus)

        self.from_tokenizer = from_tokenizer
        self._query_embeddings = None
        self._corpus_embeddings = None
        self._model = None
        self._tokenizer = None
        self.logger.info("Initialized OntoMap Sentence Transformer module")

    @property
    def tokenizer(self):
        """
        Gets the tokenizer instance.

        Returns:
            AutoTokenizer or None: The tokenizer instance if from_tokenizer is True, otherwise None.
        """
        if self.from_tokenizer:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.method_model_dict[self.method])
            return self._tokenizer
        else:
            return None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(
                self.method_model_dict[self.method],
                device='cuda' if torch.cuda.is_available() else 'cpu')
        return self._model

    @property
    def query_embeddings(self):
        """
        Gets the embeddings for the queries.

        Returns:
            torch.Tensor: The query embeddings.
        """
        if self._query_embeddings is None:
            embd = self.create_embeddings(self.query)
        return embd

    @property
    def corpus_embeddings(self):
        """
        Gets the embeddings for the corpus.

        Returns:
            torch.Tensor: The corpus embeddings.
        """
        if self._corpus_embeddings is None:
            idx = faiss.read_index(self.vector_store.index_path)
            vecs = np.vstack([idx.reconstruct(i) for i in range(idx.ntotal)])
            self._corpus_embeddings = vecs
        return self._corpus_embeddings

    def create_embeddings(self,
                          query_list: list[str],
                          convert_to_tensor: bool = True):
        """
        Creates embeddings for the sentence transformer model.

        Args:
            query_list (list[str]): List of query strings.
            convert_to_tensor (bool, optional): Whether to convert the output embeddings to tensor. Defaults to True.

        Returns:
            numpy.ndarray or torch.Tensor: The output embeddings.
        """
        return self.model.encode(
            query_list,
            convert_to_tensor=convert_to_tensor,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    def get_match_result_single_query(self, query: str,
                                      cosine_sim_df: pd.DataFrame):
        """
        Generates match results for a single query using cosine similarity DataFrame.

        Args:
            query (str): The query string.
            cosine_sim_df (pd.DataFrame): DataFrame containing cosine similarity scores.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError(
            "get_match_result_single_query will be implemented later")

    def get_match_results_mp(self):
        """
        Generates match results for the given queries and corpus using multiprocessing.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError(
            "get_match_results_mp will be implemented later")

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = 5,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        idx = self.vector_store.index

        q_embs = self.create_embeddings(self.query, convert_to_tensor=False)
        q_mat = np.array(q_embs, dtype="float32")

        norms = np.linalg.norm(q_mat, axis=1, keepdims=True)
        q_norm = q_mat / (norms + 1e-12)

        D, I = idx.search(q_norm, topk)

        rows = []
        for qi, q in enumerate(self.query):
            top_ids = I[qi]
            top_scores = D[qi].tolist()
            top_terms = [self.corpus[i] for i in top_ids]

            curated = (cura_map.get(q, "Not Found") if test_or_prod == 'test'
                       else "Not Available for Prod Environment")
            lvl = next(
                (i + 1 for i, t in enumerate(top_terms) if t == curated), 99)

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
