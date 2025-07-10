import pandas as pd
import requests
from tqdm import tqdm
from src.models.ontology_models import OntoModelsBase


class OntoMapBIE(OntoModelsBase):
    """
    A class to map ontologies using bi-encoder models.
    """

    def __init__(
        self,
        method: str,
        category: str,
        query: list[str],  # Can be removed. Need confirmation
        corpus: list[str],
        query_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        topk: int = 5,
        om_strategy: str = 'rag_bie',
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         query_df=query_df,
                         corpus_df=corpus_df)
        self.logger.info("Initialized Bi-Encoder (query with context) module")
        self.code2name = self.load_oncotree_mapping(
            "data/corpus/oncotree_code_to_name.csv")

    def load_oncotree_mapping(self, path: str) -> dict:
        df = pd.read_csv(path)
        return dict(zip(df['code'].astype(str), df['name'].astype(str)))

    def get_cbioportal_study_info(self, study_id: str) -> tuple[str, str]:
        url = f"https://www.cbioportal.org/api/studies/{study_id}"
        try:
            res = requests.get(url, headers={"Accept": "application/json"})
            if res.status_code == 200:
                js = res.json()
                name = js.get("name", "")
                cancer_type_name = js.get("cancerType", {}).get("name", "")
                return name, cancer_type_name
        except Exception as e:
            self.logger.warning(
                f"Failed to fetch study info for {study_id}: {e}")
        return "", ""

    def add_context_to_query(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with one extra column: enriched_query
        TODO: Simplify context
        """
        enriched_queries = []

        for _, row in tqdm(query_df.iterrows(),
                           total=len(query_df),
                           desc="Adding context to query_df"):
            orig = str(row['original_cancer_type_value']).strip()
            oncotree_code = str(row.get('ONCOTREE_CODE', '')).strip()
            study_id = str(row.get('studyId', '')).strip()
            primary_site = str(row.get('PRIMARY_SITE', '')).strip()

            oncotree_name = self.code2name.get(oncotree_code, "")
            study_name, cancer_type_name = self.get_cbioportal_study_info(
                study_id)

            enriched_query = f"{orig}"
            if oncotree_name:
                enriched_query += f" [{oncotree_name}]"
            if study_name:
                enriched_query += f"; from study: {study_name}"
            if cancer_type_name:
                enriched_query += f"; cancer type: {cancer_type_name}"
            if primary_site:
                enriched_query += f"; site: {primary_site}"

            enriched_queries.append(enriched_query)

        query_df = query_df.copy()
        query_df['enriched_query'] = enriched_queries
        return query_df

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = None,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        k = topk or self.topk
        all_results = []

        if 'enriched_query' not in self.query_df.columns:
            self.logger.warning(
                "No enriched_query column found. Adding context now.")
            self.query_df = self.add_context_to_query(self.query_df)

        orig_queries = self.query_df['original_cancer_type_value'].tolist()
        ctx_queries = self.query_df['enriched_query'].tolist()

        for ctx_q in tqdm(ctx_queries, desc="Processing queries (Bi-Encoder)"):
            hits = self.vector_store.similarity_search(query=ctx_q,
                                                       k=k,
                                                       as_documents=True)
            all_results.append(hits)

        df = pd.DataFrame({
            'original_value':
            orig_queries,
            'ctx_query':
            ctx_queries,
            'curated_ontology': [
                cura_map.get(q, "Not Found")
                if test_or_prod == 'test' else "N/A" for q in orig_queries
            ]
        })

        for i in range(k):
            df[f'top{i+1}_match'] = [
                hits[i].metadata['term'] if i < len(hits) else "N/A"
                for hits in all_results
            ]
            df[f'top{i+1}_score'] = [
                f"{hits[i].metadata['score']:.4f}" if i < len(hits) else "N/A"
                for hits in all_results
            ]

        df['match_level'] = df.apply(lambda row: next(
            (j + 1 for j in range(k)
             if row[f'top{j+1}_match'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Bi-Encoder Results Generated")
        return df
