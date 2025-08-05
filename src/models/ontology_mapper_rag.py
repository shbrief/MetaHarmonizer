import pandas as pd
from tqdm import tqdm
from src.models.ontology_models import OntoModelsBase


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
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df)
        self.logger.info("Initialized OntoMapRAG module")

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = 5,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        self.logger.info("Generating results table")
        results = []
        for q in tqdm(self.query, desc="Processing queries"):
            search_results = self.vector_store.similarity_search(
                query=q, k=topk, as_documents=True)
            results.append(search_results)

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

        df['match_level'] = df.apply(lambda row: next(
            (i + 1 for i in range(topk)
             if row[f'top{i+1}_match'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Results Generated")
        return df
