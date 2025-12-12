import torch
import pandas as pd
from tqdm.auto import tqdm
from src.models.ontology_models import OntoModelsBase
from src.utils.model_loader import get_llm_reranker_cached


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
        use_llm_reranker: bool = True,
        llm_reranker_method: str = 'rankllama-7b',
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df)
        self.use_llm_reranker = use_llm_reranker
        self.llm_reranker_method = llm_reranker_method
        self.llm_reranker_topk = 50
        self._llm_reranker = None

        rerank_status = f"LLM:{llm_reranker_method}" if use_llm_reranker else "disabled"
        self.logger.info(
            f"Initialized OntoMapRAG module (reranker={rerank_status})")

    @property
    def llm_reranker(self):
        """Lazy loading of LLM Reranker"""
        if self._llm_reranker is None and self.use_llm_reranker:
            self.logger.info(
                f"Loading LLM Reranker: {self.llm_reranker_method}")

            # Auto-detect if 8-bit quantization is needed based on VRAM
            if torch.cuda.is_available():
                total_vram = torch.cuda.get_device_properties(0).total_memory
                use_8bit = total_vram < 20e9
                self.logger.info(
                    f"GPU VRAM: {total_vram/1e9:.1f}GB, using 8-bit: {use_8bit}"
                )
            else:
                use_8bit = False

            self._llm_reranker = get_llm_reranker_cached(
                self.llm_reranker_method, use_8bit=use_8bit, batch_size=4)
        return self._llm_reranker

    def _rerank_results(self, query: str, candidates: list, topk: int) -> list:
        """
        LLM reranking.
        
        Parameters:
        - query: The search query
        - candidates: List of document objects from FAISS search
        - topk: Number of top results to return after reranking
        
        Returns:
        - Reranked list of candidates
        """
        if not candidates or not self.use_llm_reranker:
            return candidates[:topk]

        self.logger.debug(
            f"LLM reranking {len(candidates)} candidates for query: {query}")

        pairs = [[query, doc.page_content] for doc in candidates]
        llm_scores = self.llm_reranker.predict(pairs)
        ranked_indices = llm_scores.argsort()[::-1]
        reranked_candidates = []
        for idx in ranked_indices[:topk]:
            candidate = candidates[idx]
            candidate.metadata['similarity_score'] = candidate.metadata.get(
                'score', 0)
            candidate.metadata['llm_score'] = float(llm_scores[idx])
            candidate.metadata['score'] = float(llm_scores[idx])
            reranked_candidates.append(candidate)

        return reranked_candidates

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = 5,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        self.logger.info("Generating results table")

        retrieval_k = self.llm_reranker_topk if self.use_llm_reranker else topk

        results = []
        for q in tqdm(self.query, desc="Processing queries", leave=False):
            # Step 1: FAISS retrieval (PubMedBERT, label + definition)
            search_results = self.vector_store.similarity_search(
                query=q, k=retrieval_k, as_documents=True)

            # Step 2: LLM Reranking (top-50 → top-5)
            if self.use_llm_reranker:
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

            # If use LLM reranker, add both similarity_score and llm_score
            if self.use_llm_reranker:
                df[f'top{i+1}_similarity_score'] = [
                    f"{r[i].metadata.get('similarity_score', 0):.4f}"
                    if i < len(r) else "N/A" for r in results
                ]
                df[f'top{i+1}_llm_score'] = [
                    f"{r[i].metadata.get('llm_score', 0):.4f}"
                    if i < len(r) else "N/A" for r in results
                ]

        df['match_level'] = df.apply(lambda row: next(
            (i + 1 for i in range(topk)
             if row[f'top{i+1}_match'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Results Generated")
        return df
