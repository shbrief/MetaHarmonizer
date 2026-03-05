import faiss
import numpy as np
import pandas as pd
import torch
import src.models.ontology_models as otm


class OntoMapLM(otm.OntoModelsBase):
    """
    A class to map ontologies using language models.

    Uses the same cached SentenceTransformer instance as FAISSSQLiteSearch
    to avoid loading duplicate model weights into memory.

    Attributes:
        method (str): The method to use for the model.
        query (list[str]): The list of query strings.
        corpus (list[str]): The list of corpus strings.
        topk (int): The number of top results to consider.
        from_tokenizer (bool): Whether to use a tokenizer.
        _query_embeddings (torch.Tensor or None): Embeddings for the queries.
        _corpus_embeddings (torch.Tensor or None): Embeddings for the corpus.
        _model (SentenceTransformer or None): The model instance.
        logger (CustomLogger): Logger instance.
    """

    def __init__(
        self,
        method: str,
        category: str,
        query: list[str],
        corpus: list[str],
        om_strategy: str = 'lm',
        topk: int = 5,
        from_tokenizer: bool = True,
    ) -> None:
        """
        Initializes the OntoMapLM class.

        Args:
            method (str): The method to use for the model.
            query (list[str]): The list of query strings.
            corpus (list[str]): The list of corpus strings.
            topk (int, optional): The number of top results to consider. Defaults to 5.
            from_tokenizer (bool, optional): Whether to use a tokenizer. Defaults to True.
        """
        super().__init__(method, category, om_strategy, topk, query, corpus)

        self.from_tokenizer = from_tokenizer
        self._query_embeddings = None
        self._corpus_embeddings = None
        self._model = None
        self.logger.info("Initialized OntoMap Language Model module")

    @property
    def model(self):
        """
        Gets the cached SentenceTransformer model instance.

        Reuses the same model loaded by FAISSSQLiteSearch / SynonymDict
        via get_embedding_model_cached(), preventing duplicate memory usage.

        Returns:
            SentenceTransformer: The model instance.
        """
        if self._model is None:
            from src.utils.model_loader import get_embedding_model_cached
            self._model = get_embedding_model_cached(self.method)
        return self._model

    @property
    def query_embeddings(self):
        """
        Gets the embeddings for the queries.

        Returns:
            numpy.ndarray: The query embeddings.
        """
        if self._query_embeddings is None:
            self._query_embeddings = self.create_embeddings(self.query)
        return self._query_embeddings

    @property
    def corpus_embeddings(self):
        """
        Gets the embeddings for the corpus.

        Returns:
            numpy.ndarray: The corpus embeddings.
        """
        if self._corpus_embeddings is None:
            idx = faiss.read_index(self.vector_store.index_path)
            vecs = np.vstack([idx.reconstruct(i) for i in range(idx.ntotal)])
            self._corpus_embeddings = vecs
        return self._corpus_embeddings

    def create_embeddings(self,
                          query_list: list[str],
                          convert_to_tensor: bool = False):
        """
        Creates normalized embeddings using the cached SentenceTransformer.

        Args:
            query_list (list[str]): List of query strings.
            convert_to_tensor (bool, optional): Whether to return a torch.Tensor
                instead of a list of lists. Defaults to False.

        Returns:
            list[list[float]] or torch.Tensor: The output embeddings.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embs = self.model.encode(
            query_list,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=True,
            device=device)

        if convert_to_tensor:
            return embs

        arr = np.array(embs, dtype="float32")
        return arr.tolist()

    def create_cura_map(self, query_list: list[str], corpus_list: list[str]):
        """
        Creates embeddings for the sentence transformer model.

        Args:
            query_list (list[str]): List of query strings.
            corpus_list (list[str]): List of corpus strings.

        Returns:
            tuple: A tuple containing query embeddings and corpus embeddings.
        """
        query_embeddings = self.create_embeddings(query_list)
        corpus_embeddings = self.create_embeddings(corpus_list)
        return query_embeddings, corpus_embeddings

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

        D, I = idx.search(q_mat, topk)

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
