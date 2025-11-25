import pandas as pd
from typing import List, Dict
from src.KnowledgeDb.synonym_dict import SynonymDict
from src.CustomLogger.custom_logger import CustomLogger


class OntoMapSynonym:
    """
    Stage 2.5: Synonym dictionary matching using NCI API.
    """

    def __init__(self,
                 method: str,
                 category: str,
                 query: List[str],
                 corpus: List[str],
                 om_strategy: str = 'syn',
                 topk: int = 5,
                 corpus_df: pd.DataFrame = None):
        self.method = method
        self.category = category
        self.query = query
        self.corpus = corpus
        self.topk = topk
        self.corpus_df = corpus_df

        self.logger = CustomLogger().custlogger(loglevel='INFO')

        # Initialize SynonymDict with the same method as Stage 2
        self.syn_dict = SynonymDict(category=category, method=method)

        # Warm run: build index if needed
        if corpus_df is not None:
            self._warm_run()
        else:
            self.logger.warning(
                "No corpus_df provided. Using existing synonym index only.")

    def _warm_run(self):
        """
        Warm run: Extract NCI codes from corpus_df and build index.
        """
        self.logger.info(
            f"Warm run: Checking synonym index for '{self.category}'")

        # Extract NCI codes from corpus_df
        if "clean_code" not in self.corpus_df.columns:
            self.logger.error("corpus_df must contain 'clean_code' column")
            return

        codes = self.corpus_df["clean_code"].dropna().unique().tolist()
        self.logger.info(f"Found {len(codes)} unique NCI codes in corpus_df")

        # Build index (will skip if already exists)
        self.syn_dict.warm_run(codes=codes, force_rebuild=False)

    def get_match_results(self,
                          cura_map: Dict[str, str],
                          topk: int = None,
                          test_or_prod: str = 'test',
                          min_score: float = 0.0) -> pd.DataFrame:
        """
        Get matching results for all queries.
        
        Args:
            cura_map: Mapping of query terms to curated ontology values
            topk: Number of top matches (overrides self.topk if provided)
            test_or_prod: 'test' or 'prod' mode
            min_score: Minimum similarity score threshold
            
        Returns:
            DataFrame with columns: original_value, curated_ontology, match_level,
                                   top1_match, top1_score, ..., topN_match, topN_score
        """
        k = topk or self.topk
        self.logger.info(f"Searching synonyms for {len(self.query)} queries")

        # Batch search
        batch_results = self.syn_dict.search_many(self.query, top_k=k)

        results = []
        for q, matches in zip(self.query, batch_results):
            # Get curated ontology for this query
            curated = (cura_map.get(q, "Not Found") if test_or_prod == 'test'
                       else "Not Available for Prod Environment")

            # Collect top-k matched terms
            top_terms = []
            top_scores = []

            for i in range(k):
                if i < len(matches):
                    match = matches[i]
                    score = match.get('score', 0.0)

                    if score >= min_score:
                        top_terms.append(match['official_label'])
                        top_scores.append(score)
                    else:
                        top_terms.append(None)
                        top_scores.append(0.0)
                else:
                    top_terms.append(None)
                    top_scores.append(0.0)

            # Calculate match_level: position of curated term in top-k results
            match_level = next(
                (i + 1 for i, term in enumerate(top_terms) if term == curated),
                99  # 99 if curated term not in top-k
            )

            # Build result row
            row = {
                'original_value': q,
                'curated_ontology': curated,
                'match_level': match_level
            }

            # Fill top-k results
            for i in range(1, k + 1):
                row[f'top{i}_match'] = top_terms[i - 1]
                row[f'top{i}_score'] = f"{top_scores[i - 1]:.4f}"

            results.append(row)

        df = pd.DataFrame(results)
        self.logger.info(f"Synonym matching complete for {len(df)} queries")

        return df
