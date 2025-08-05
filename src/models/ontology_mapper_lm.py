import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import torch
import src.models.ontology_models as otm


class OntoMapLM(otm.OntoModelsBase):
    """
    A class to map ontologies using language models.

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
        self._tokenizer = None
        self.logger.info("Initialized OntoMap Language Model module")

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
        """
        Gets the model instance.

        Returns:
            AutoModel or SentenceTransformer: The model instance.
        """
        if self._model is None:
            if self.from_tokenizer:
                from transformers import AutoModel
                self._model = AutoModel.from_pretrained(
                    self.method_model_dict[self.method]).eval().to(
                        'cuda' if torch.cuda.is_available() else 'cpu')
            else:
                from sentence_transformers import SentenceTransformer
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
                          convert_to_tensor: bool = False):
        """
        Creates embeddings using SAP-BERT like LM models using first token embeddings (CLS token).

        Args:
            query_list (list[str]): List of query strings.
            convert_to_tensor (bool, optional): Whether to convert the output embeddings to tensor. Defaults to False.

        Returns:
            numpy.ndarray or torch.Tensor: The output embeddings.
        """
        device = next(self.model.parameters()).device

        # Tokenize the texts and prepare input tensors
        # Note: Adjust max_length to 64, as some terms are longer than 25 tokens
        # encoded_input = tokenizer(query_list, padding="max_length", max_length=25, truncation=True, return_tensors='pt')
        encoded = self.tokenizer(query_list,
                                 padding="max_length",
                                 max_length=64,
                                 truncation=True,
                                 return_tensors='pt').to(device)

        with torch.no_grad():
            out = self.model(**encoded)

        # CLS
        hidden = getattr(out, "last_hidden_state", out[0])
        cls = hidden[:, 0, :]

        if convert_to_tensor:
            return cls

        arr = cls.cpu().numpy().astype("float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / (norms + 1e-12)

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
                row[f"top{i}_match"] = t
                row[f"top{i}_score"] = f"{s:.4f}"
            rows.append(row)

        return pd.DataFrame(rows)
