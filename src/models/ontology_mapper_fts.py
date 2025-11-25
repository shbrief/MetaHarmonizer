import pandas as pd
from typing import Dict, List
import src.models.ontology_models as otm
from src.KnowledgeDb.fts_synonym_db import FTSSynonymDb
import asyncio


class OntoMapFTS(otm.OntoModelsBase):
    """
    FTS-based ontology mapper using SQLite FTS5 with trigram fuzzy matching.
    
    This strategy is designed for Stage 1.5 synonym matching:
    - Fast fuzzy string matching using BM25 ranking
    - Handles spelling variations (e.g., breast -> breasst)
    - Lighter weight than BERT-based approaches
    
    Attributes:
        fts_db (FTSSynonymDb): FTS database for synonym matching
    """

    def __init__(self,
                 method: str,
                 category: str,
                 query: List[str],
                 corpus: List[str],
                 om_strategy: str = 'fts',
                 topk: int = 5,
                 corpus_df: pd.DataFrame = None) -> None:
        """
        Initialize FTS ontology mapper.
        
        Args:
            method: Model name (not used for FTS, but required by base class)
            category: Category name
            query: List of query strings to match
            corpus: List of corpus terms (standard terms)
            om_strategy: Strategy name (should be 'fts')
            topk: Number of top results to return
            corpus_df: DataFrame with 'official_label' and 'clean_code' columns
        """
        super().__init__(method,
                         category,
                         om_strategy,
                         topk,
                         query,
                         corpus,
                         corpus_df=corpus_df)

        self.fts_db = FTSSynonymDb(category=category)

        # Build index if not exists
        if not self.fts_db.index_exists():
            self._build_index()

    def _build_index(self):
        """Build FTS index from corpus_df"""
        if self.corpus_df is None:
            raise ValueError("corpus_df is required to build FTS index")

        if "clean_code" not in self.corpus_df.columns:
            raise ValueError("corpus_df must contain 'clean_code' column")

        codes = self.corpus_df["clean_code"].dropna().unique().tolist()

        self.logger.info(f"Building FTS index from {len(codes)} NCI codes...")

        # Run async build in sync context
        asyncio.run(self.fts_db.build_index_from_codes(codes))

    def create_embeddings(self, query, **kwargs):
        """
        FTS doesn't use embeddings, but this method is required by base class.
        Returns None to indicate embeddings are not used.
        """
        return None

    def get_match_results(self,
                          cura_map: Dict[str, str] = None,
                          topk: int = 5,
                          test_or_prod: str = 'test') -> pd.DataFrame:
        """
        Generate match results for queries using FTS search.
        
        Args:
            cura_map: Mapping of original values to curated ontologies
            topk: Number of top matches to return per query
            test_or_prod: 'test' to include curated_ontology, 'prod' otherwise
        
        Returns:
            DataFrame with columns:
                - original_value
                - curated_ontology
                - match_level
                - top1_match, top1_score, ..., topK_match, topK_score
        """
        rows = []

        for query in self.query:
            # Search FTS index
            results = self.fts_db.search(query, limit=topk)

            # Get curated ontology
            if test_or_prod == 'test' and cura_map:
                curated = cura_map.get(query, "Not Found")
            else:
                curated = "Not Available for Prod Environment"

            # Build row
            row = {
                "original_value": query,
                "curated_ontology": curated,
            }

            if results:
                # Check if curated term is in top matches
                top_terms = [r[0] for r in results]
                match_level = next(
                    (i + 1 for i, t in enumerate(top_terms) if t == curated),
                    99)
                row["match_level"] = match_level

                # Add top-k matches
                for i, (standard_term, nci_code,
                        confidence) in enumerate(results, start=1):
                    row[f"top{i}_match"] = standard_term
                    row[f"top{i}_score"] = f"{confidence:.4f}"
                    row[f"top{i}_code"] = nci_code

                # Fill remaining slots if fewer than topk results
                for i in range(len(results) + 1, topk + 1):
                    row[f"top{i}_match"] = ""
                    row[f"top{i}_score"] = "0.0000"
                    row[f"top{i}_code"] = ""

                self.logger.info(f"FTS: '{query}' → '{results[0][0]}' "
                                 f"[confidence={results[0][2]:.3f}]")
            else:
                # No matches found
                row["match_level"] = 99
                for i in range(1, topk + 1):
                    row[f"top{i}_match"] = ""
                    row[f"top{i}_score"] = "0.0000"
                    row[f"top{i}_code"] = ""

                self.logger.info(f"FTS: No match for '{query}'")

            rows.append(row)

        return pd.DataFrame(rows)

    def __del__(self):
        """Cleanup: close FTS database connection"""
        if hasattr(self, 'fts_db'):
            self.fts_db.close()
