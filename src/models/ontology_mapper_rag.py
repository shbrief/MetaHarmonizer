import torch
import pandas as pd
from tqdm.auto import tqdm
from src.models.ontology_models import OntoModelsBase
from src.utils.model_loader import get_reranker_cached
from src.models.reranker import RERANKER_TYPE_MAP


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
        use_reranker: bool,
        reranker_method: str,
        reranker_topk: int,
        topk: int = 5,
        om_strategy: str = 'rag',
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df)
        self.use_reranker = use_reranker

        if self.use_reranker:
            if reranker_method not in RERANKER_TYPE_MAP:
                raise ValueError(
                    f"Unknown reranker_method '{reranker_method}'. Must be one of {list(RERANKER_TYPE_MAP.keys())}"
                )
            self.reranker_method = reranker_method
            self.reranker_topk = reranker_topk
        else:
            self.reranker_method = None
            self.reranker_topk = None

        self._reranker = None

        self.logger.info(
            f"Initialized OntoMapRAG (reranker={'enabled:'+reranker_method if use_reranker else 'disabled'})"
        )

    @property
    def reranker(self):
        """Lazy loading"""
        if self._reranker is None and self.use_reranker:
            reranker_type = RERANKER_TYPE_MAP.get(self.reranker_method)

            # Auto 8-bit for large models
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory
                use_8bit = (vram < 20e9) and (reranker_type
                                              in ["t5", "generative"])
            else:
                use_8bit = False

            self.logger.info(
                f"Loading {reranker_type} reranker: {self.reranker_method}")

            self._reranker = get_reranker_cached(
                self.reranker_method,
                use_8bit=use_8bit,
            )
        return self._reranker

    def _rerank_results(self, query: str, candidates: list, topk: int) -> list:
        """
        Reranking.
        
        Parameters:
        - query: The search query
        - candidates: List of document objects from FAISS search
        - topk: Number of top results to return after reranking
        
        Returns:
        - Reranked list of candidates
        """
        if not candidates or not self.use_reranker:
            return candidates[:topk]

        self.logger.debug(
            f"Reranking {len(candidates)} candidates for query: {query}")

        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked_indices = scores.argsort()[::-1]
        reranked_candidates = []
        for idx in ranked_indices[:topk]:
            candidate = candidates[idx]
            candidate.metadata['similarity_score'] = candidate.metadata.get(
                'score', 0)
            candidate.metadata['reranker_score'] = float(scores[idx])
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

        retrieval_k = self.reranker_topk if self.use_reranker else topk

        results = []
        for q in tqdm(self.query, desc="Processing queries", leave=False):
            # Step 1: FAISS retrieval (PubMedBERT, label + definition)
            search_results = self.vector_store.similarity_search(
                query=q, k=retrieval_k, as_documents=True)

            # Step 2: Reranking (top-50 → top-5)
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
            df[f'match{i+1}'] = [
                r[i].metadata['term'] if i < len(r) else "N/A" for r in results
            ]
            df[f'match{i+1}_score'] = [
                f"{r[i].metadata['score']:.4f}" if i < len(r) else "N/A"
                for r in results
            ]

            if self.use_reranker:
                df[f'match{i+1}_similarity_score'] = [
                    f"{r[i].metadata.get('similarity_score', 0):.4f}"
                    if i < len(r) else "N/A" for r in results
                ]
                df[f'match{i+1}_reranker_score'] = [
                    f"{r[i].metadata.get('reranker_score', 0):.4f}"
                    if i < len(r) else "N/A" for r in results
                ]

        df['match_level'] = df.apply(lambda row: next(
            (i + 1 for i in range(topk)
             if row[f'match{i+1}'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Results Generated")
        return df
