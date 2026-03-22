import os
import json
import pandas as pd
from tqdm.auto import tqdm
from src.models.ontology_models import OntoModelsBase
from src.utils.model_loader import load_method_model_dict

_BIE_LLM_MODEL = load_method_model_dict()["gemma-12b"]


class OntoMapBIE(OntoModelsBase):
    """
    A class to map ontologies using Retrieval-Augmented Generation (RAG) with optional reranking.
    Both query-side and corpus-side context is retrieved for giving the model more information for semantic mapping.
    """

    def __init__(
        self,
        method: str,
        category: str,
        query: list[str],  # TODO: confirm input format
        corpus: list[str],
        query_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        topk: int = 5,
        om_strategy: str = 'rag_bie',
        use_reranker: bool = False,
        reranker_method: str = 'minilm',
        reranker_topk: int = 50,
    ):
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         query_df=query_df,
                         corpus_df=corpus_df)
        self._init_reranker(use_reranker, reranker_method, reranker_topk)
        self.logger.info(
            f"Initialized Bi-Encoder (reranker="
            f"{'enabled:' + reranker_method if use_reranker else 'disabled'})"
        )

    def _llm_select_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Use LLM to pick which columns of df contain clinically useful context
        for cancer/disease name disambiguation.
        Returns a list of column names (subset of df.columns).
        """
        col_preview = df.head(3).to_dict(orient='list')
        prompt = (
            "You are a clinical informatics expert. Given a clinical metadata dataframe, "
            "identify columns that provide useful context for disambiguating a cancer/disease name.\n"
            "Useful columns: cancer type, histology, primary site, oncotree code, tissue type, etc.\n"
            "Exclude: patient IDs, numerical measurements, dates, study IDs, URLs.\n\n"
            f"Columns and sample values:\n{json.dumps(col_preview, indent=2, default=str)}\n\n"
            "Return ONLY a JSON array of selected column names, e.g.: [\"CANCER_TYPE_DETAILED\", \"PRIMARY_SITE\"]\n"
            "No explanation, no markdown, pure JSON only."
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.logger.warning("GEMINI_API_KEY not set, falling back to all columns")
            return list(df.columns)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(_BIE_LLM_MODEL)
        max_attempts = 3
        last_exc = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = model.generate_content(prompt)
                raw = resp.text.strip()
                # extract JSON from possible ```json ... ``` fences
                if "```json" in raw:
                    raw = raw.split("```json")[1].split("```")[0]
                elif "```" in raw:
                    raw = raw.split("```")[1].split("```")[0]
                cols = json.loads(raw.strip())
                if not isinstance(cols, list):
                    raise ValueError(f"LLM returned non-list: {cols!r}")
                self.logger.info(f"LLM chose columns (attempt {attempt}): {cols}")
                hallucinated = [c for c in cols if isinstance(c, str) and c not in df.columns]
                if hallucinated:
                    self.logger.warning(
                        f"LLM hallucinated columns not in df (attempt {attempt}): {hallucinated}"
                    )
                    raise ValueError(f"Hallucinated columns: {hallucinated}")
                valid = [c for c in cols if isinstance(c, str) and c in df.columns]
                self.logger.info(f"LLM selected context columns: {valid}")
                return valid
            except Exception as e:
                last_exc = e
                if attempt < max_attempts:
                    self.logger.warning(f"LLM column selection attempt {attempt} failed ({e}), retrying...")
        self.logger.warning(f"LLM column selection failed after {max_attempts} attempts ({last_exc}), falling back to all columns")
        return list(df.columns)

    def _detect_term_column(self, df: pd.DataFrame) -> str:
        """
        Find the df column that best overlaps with self.query (the join key).
        """
        query_set = set(self.query)
        best_col, best_overlap = None, 0
        for col in df.columns:
            overlap = df[col].astype(str).isin(query_set).sum()
            if overlap > best_overlap:
                best_overlap, best_col = overlap, col
        if best_col is None:
            raise ValueError("Could not detect query term column in query_df")
        self.logger.info(f"Detected term column: '{best_col}' ({best_overlap}/{len(self.query)} matches)")
        return best_col

    def add_context_to_query(self, query_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with one extra column: enriched_query.
        Uses LLM to select relevant context columns from whatever schema query_df has.
        """
        if not hasattr(self, '_ctx_cols'):
            self._ctx_cols = self._llm_select_columns(query_df)
        if not hasattr(self, '_term_col'):
            self._term_col = self._detect_term_column(query_df)

        enriched = []
        for _, row in query_df.iterrows():
            term = str(row[self._term_col]).strip()
            parts = [term]
            for col in self._ctx_cols:
                if col == self._term_col:
                    continue
                val = str(row.get(col, "")).strip()
                if val and val.lower() not in ("nan", "none", ""):
                    label = col.replace("_", " ").title()
                    parts.append(f"{label}: {val}")
            enriched.append("; ".join(parts))

        out = query_df.copy()
        out['enriched_query'] = enriched
        return out

    def get_match_results(self,
                          cura_map: dict[str, str] = None,
                          topk: int = None,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        if test_or_prod == 'test' and cura_map is None:
            raise ValueError("cura_map should be provided for test mode")

        k = topk or self.topk
        retrieval_k = self.reranker_topk if self.use_reranker else k
        all_results = []

        if 'enriched_query' not in self.query_df.columns:
            self.logger.warning("No enriched_query column found. Adding context now.")
            self.query_df = self.add_context_to_query(self.query_df)

        # _term_col may not be set if query_df was pre-enriched externally
        if not hasattr(self, '_term_col'):
            self._term_col = self._detect_term_column(self.query_df)

        orig_queries = self.query_df[self._term_col].tolist()
        ctx_queries = self.query_df['enriched_query'].tolist()

        for ctx_q in tqdm(ctx_queries, desc="Processing queries (Bi-Encoder)", leave=False):
            hits = self.vector_store.similarity_search(
                query=ctx_q, k=retrieval_k, as_documents=True)
            if self.use_reranker:
                hits = self._rerank_results(ctx_q, hits, k)
            else:
                hits = hits[:k]
            all_results.append(hits)

        df = pd.DataFrame({
            'original_value': orig_queries,
            'ctx_query': ctx_queries,
            'curated_ontology': [
                cura_map.get(q, "Not Found")
                if test_or_prod == 'test' else "N/A" for q in orig_queries
            ]
        })

        for i in range(k):
            df[f'match{i+1}'] = [
                hits[i].metadata['term'] if i < len(hits) else "N/A"
                for hits in all_results
            ]
            df[f'match{i+1}_score'] = [
                f"{hits[i].metadata['score']:.4f}" if i < len(hits) else "N/A"
                for hits in all_results
            ]
            if self.use_reranker:
                df[f'match{i+1}_similarity_score'] = [
                    f"{hits[i].metadata.get('similarity_score', 0):.4f}"
                    if i < len(hits) else "N/A" for hits in all_results
                ]
                df[f'match{i+1}_reranker_score'] = [
                    f"{hits[i].metadata.get('reranker_score', 0):.4f}"
                    if i < len(hits) else "N/A" for hits in all_results
                ]

        df['match_level'] = df.apply(lambda row: next(
            (j + 1 for j in range(k)
             if row[f'match{j+1}'] == row['curated_ontology']), 99),
                                     axis=1)

        self.logger.info("Bi-Encoder Results Generated")
        return df
