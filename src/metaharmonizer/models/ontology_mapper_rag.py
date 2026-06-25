import pandas as pd
from tqdm.auto import tqdm
from metaharmonizer.models.ontology_models import OntoModelsBase


class OntoMapRAG(OntoModelsBase):
    """
    A class to map ontologies using Retrieval-Augmented Generation (RAG) with optional reranking.
    Corpus-side context is retrieved for giving the model more information for semantic mapping.
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
        ontology_source: str = 'ncit',
        table_suffix: str = "",
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df,
                         ontology_source=ontology_source,
                         table_suffix=table_suffix)
        self._init_reranker(use_reranker, reranker_method, reranker_topk)
        self.logger.info(
            f"Initialized OntoMapRAG (reranker={'enabled:'+reranker_method if use_reranker else 'disabled'})"
        )

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
             if str(row[f'match{i+1}']).strip().lower() == str(row['curated_ontology']).strip().lower()), 99),
                                     axis=1)

        self.logger.info("Results Generated")
        return df
