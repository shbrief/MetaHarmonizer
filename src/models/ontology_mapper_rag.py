import sys
import pandas as pd
from tqdm.auto import tqdm
from src.models.ontology_models import OntoModelsBase
from src.utils.model_loader import get_reranker_model_cached


class OntoMapRAG(OntoModelsBase):
    """
    RAG-based ontology mapping using FAISS + SQLite local vector store.
    """

    def __init__(
        self,
        method: str,
        category: str,
        query: list[str],
        corpus: list[str],
        corpus_df: pd.DataFrame,
        topk: int = 5,
        om_strategy: str = 'rag',
        use_reranker: bool = True,
        reranker_model: str = 'minilm',
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df)
        self.use_reranker = use_reranker
        self.rerank_topk = 50  # Number of candidates to rerank
        self._reranker = None
        self.reranker_model_name = reranker_model

        self.logger.info(
            f"Initialized OntoMapRAG module (reranker={'enabled' if use_reranker else 'disabled'})"
        )

    @property
    def reranker(self):
        """Lazy loading of reranker model"""
        if self._reranker is None and self.use_reranker:
            self.logger.info(f"Loading reranker: {self.reranker_method}")
            self._reranker = get_reranker_model_cached(self.reranker_method)
        return self._reranker

    def _rerank_results(self, query: str, candidates: list, topk: int) -> list:
        """
        Rerank candidates using cross-encoder.
        
        Parameters:
        - query: The search query
        - candidates: List of document objects from FAISS search
        - topk: Number of top results to return after reranking
        
        Returns:
        - Reranked list of candidates
        """
        if not candidates or not self.use_reranker:
            return candidates[:topk]

        pairs = [[query, doc.page_content] for doc in candidates]

        # Get reranking scores
        scores = self.reranker.predict(pairs)

        # Sort by reranker scores (descending)
        ranked_indices = scores.argsort()[::-1]

        # Update metadata with reranker scores
        reranked_candidates = []
        for idx in ranked_indices[:topk]:
            candidate = candidates[idx]
            candidate.metadata['similarity_score'] = candidate.metadata.get(
                'score', 0)
            candidate.metadata['rerank_score'] = float(scores[idx])
            candidate.metadata['score'] = float(scores[idx])
            reranked_candidates.append(candidate)

        return reranked_candidates

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = 5,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        self.logger.info("Generating results table")

        retrieval_k = self.rerank_topk if self.use_reranker else topk

        results = []
        for q in tqdm(self.query, desc="Processing queries", leave=False):
            # Step 1: FAISS retrieval
            search_results = self.vector_store.similarity_search(
                query=q, k=retrieval_k, as_documents=True)

            # Step 2: Optional reranking
            if self.use_reranker:
                search_results = self._rerank_results(q, search_results, topk)
            else:
                search_results = search_results[:topk]

            results.append(search_results)

        # Build results DataFrame
        df = pd.DataFrame({
            'original_value':
            self.query,
            'curated_ontology': [
                cura_map.get(q, "Not Found") if test_or_prod == 'test' else
                "Not Available for Prod Environment" for q in self.query
            ]
        })

        for i in range(topk):
            df[f'top{i+1}_match'] = [
                r[i].metadata['term'] if i < len(r) else "N/A" for r in results
            ]
            df[f'top{i+1}_score'] = [
                f"{r[i].metadata['score']:.4f}" if i < len(r) else "N/A"
                for r in results
            ]

            if self.use_reranker:
                df[f'top{i+1}_similarity_score'] = [
                    f"{r[i].metadata.get('similarity_score', 0):.4f}"
                    if i < len(r) else "N/A" for r in results
                ]

        df['match_level'] = df.apply(lambda row: next(
            (i + 1 for i in range(topk)
             if row[f'top{i+1}_match'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Results Generated")
        return df
