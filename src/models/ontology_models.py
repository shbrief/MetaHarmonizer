from sentence_transformers import util
import pandas as pd
from src.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch
from src.utils.model_loader import load_method_model_dict
from src.CustomLogger.custom_logger import CustomLogger


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
        use_reranker: bool = None,
        reranker_method: str = None,
    ) -> None:
        self.method = method
        self.category = category
        self.om_strategy = om_strategy
        self.query = query
        self.corpus = corpus
        self.query_df = query_df
        self.corpus_df = corpus_df
        self.use_reranker = use_reranker
        self.reranker_method = reranker_method

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
                                      om_strategy=self.om_strategy)

            if self.corpus or self.corpus_df:
                store.ensure_corpus_integrity(self.corpus, self.corpus_df)
            self._vs = store

            self.logger.info(
                f"{self._vs.index is not None} - Vector store initialized for method={self.method}, category={self.category}, om_strategy={self.om_strategy}"
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

    ##### To be implemented in the child class #####
    def create_embeddings(self):
        """
        This method is responsible for creating embeddings using sentence transformers, LM's or LLM's for the cBioPortal's queries or corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError(
            "create_embeddings will be implemented in the child class")

    def get_match_results(self):
        """
        This method is responsible for generating match results for the given queries and corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError(
            "get_match_results will be implemented in the child class")
