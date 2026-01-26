"""Main schema mapping engine."""
import os
import time
import pandas as pd
from typing import Dict, Any, List
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import torch

from .config import (
    FUZZY_THRESH, OUTPUT_DIR, ALIAS_DICT_PATH, FIELD_MODEL,
    NUMERIC_THRESH, FIELD_ALIAS_THRESH, VALUE_PERCENTAGE_THRESH
)
from .loaders.dict_loader import DictLoader
from .loaders.value_loader import ValueLoader
from .matchers.base import MatchStrategy
from .matchers.stage1_matchers import (
    StandardExactMatcher, AliasExactMatcher,
    StandardFuzzyMatcher, AliasFuzzyMatcher
)
from .matchers.stage2_matchers import (
    NumericCombinedMatcher, SemanticCombinedMatcher
)
from .matchers.stage3_matchers import (
    ValueStandardMatcher, OntologyMatcher
)
from src.utils.schema_mapper_utils import normalize, is_numeric_column, extract_valid_value
from src.utils.invalid_column_utils import check_invalid
from src.utils.ncit_match_utils import NCIClientSync
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class SchemaMapEngine:
    """
    Main schema mapping engine with multi-stage cascade matching.
    
    Stages:
    - Stage 1: Dictionary matching (exact + fuzzy, standard + alias)
    - Stage 2: Field matching (numeric + semantic, standard + alias)
    - Stage 3: Value matching (value dict + ontology)
    """
    
    def __init__(self, clinical_data_path: str, mode: str = "auto", top_k: int = 5):
        """
        Initialize the schema mapping engine.
        
        Args:
            clinical_data_path: Path to clinical data CSV/TSV file
            mode: Execution mode ('auto' or 'manual')
            top_k: Number of top matches to return
        """
        # Load data
        if clinical_data_path.endswith(".tsv"):
            self.df = pd.read_csv(clinical_data_path, sep="\t", dtype=str)
        else:
            self.df = pd.read_csv(clinical_data_path, sep=",", dtype=str)
        
        logger.info(f"[Load] df_shape={self.df.shape} first_cols={list(self.df.columns[:5])}")
        
        self.top_k = top_k
        self.mode = mode
        
        # Setup output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.basename(clinical_data_path)
        root, _ = os.path.splitext(base)
        self.output_file = os.path.join(OUTPUT_DIR, f"{root}.csv")
        
        # Load dictionaries
        self._load_dictionaries()
        
        # Initialize models
        self.dict_model = SentenceTransformer(FIELD_MODEL)
        
        # Initialize NCI client
        self.nci_client = NCIClientSync()
        
        # Initialize matchers
        self._init_matchers()
        
        # Lazy-loaded components
        self._numeric_embs = None
        self._col_values_cache = {}
        
        logger.info("SchemaMapEngine initialized")
    
    def _load_dictionaries(self):
        """Load all dictionaries."""
        # Standard dict
        (self.standard_fields,
        self.standard_fields_normed,
        self.normed_std_to_std,
        self.curated_df) = DictLoader.load_standard_dict()
        
        # Alias dict (may not exist)
        (self.sources_to_fields,
        self.sources_keys,
        self.normed_source_to_source,
        self.has_alias_dict) = DictLoader.load_alias_dict()
        
        # Load full alias DataFrame for numeric matching (only if alias dict exists)
        if self.has_alias_dict:
            self.df_dict = pd.read_csv(ALIAS_DICT_PATH)
            self.df_num, self.numeric_sources = DictLoader.load_numeric_dict(self.df_dict)
            
            # Encode alias keys
            self.alias_embs = DictLoader.encode_fields(self.sources_keys)
        else:
            self.df_dict = None
            self.df_num = None
            self.numeric_sources = []
            self.alias_embs = None
            logger.warning("[Engine] Alias dictionary not available, alias-based matching will be skipped")
        
        # Load value dict
        ValueLoader.load_value_dict(self)
    
    def _init_matchers(self):
        """Initialize all matcher instances."""
        # Stage 1
        self.std_exact = StandardExactMatcher(self)
        self.std_fuzzy = StandardFuzzyMatcher(self)
        
        if self.has_alias_dict:
            self.alias_exact = AliasExactMatcher(self)
            self.alias_fuzzy = AliasFuzzyMatcher(self)
        else:
            self.alias_exact = None
            self.alias_fuzzy = None
        
        # Stage 2 - Use combined matchers
        self.numeric_combined = NumericCombinedMatcher(self)
        self.semantic_combined = SemanticCombinedMatcher(self)
        
        # Stage 3
        self.value_std = ValueStandardMatcher(self)
        self.ontology = OntologyMatcher(self)
    
    def _ensure_numeric_index(self):
        """Lazy-build numeric embedding index on first use."""
        if self._numeric_embs is not None:
            return
        self.norm_numeric = [normalize(s) for s in self.numeric_sources]
        if not self.norm_numeric:
            self._numeric_embs = torch.empty(0)
            return
        self._numeric_embs = self.dict_model.encode(
            self.norm_numeric,
            convert_to_tensor=True
        )
    
    @lru_cache(maxsize=None)
    def is_col_numeric(self, col: str) -> bool:
        """Cached wrapper around is_numeric_column."""
        return is_numeric_column(self.df, col)
    
    @lru_cache(maxsize=None)
    def _enc(self, text: str):
        """Cached text encoding."""
        return self.dict_model.encode(text, convert_to_tensor=True)
    
    def unique_values(self, col: str, cap: int | None = None) -> list[str]:
        """Get unique values from a column with caching."""
        if col not in self._col_values_cache:
            series = self.df[col].dropna().astype(str).apply(extract_valid_value)
            uniq, seen = [], set()
            for lst in series:
                for v in lst:
                    from .config import NOISE_VALUES
                    nv = normalize(v)
                    if nv and (nv not in NOISE_VALUES) and (nv not in seen):
                        seen.add(nv)
                        uniq.append(v)
            self._col_values_cache[col] = uniq
        
        vals = self._col_values_cache[col]
        return vals if (cap is None or cap >= len(vals)) else vals[:cap]
    
    def format_matches_to_row(
        self,
        col: str,
        stage: str,
        detail: str,
        matches: List,
    ) -> Dict[str, Any]:
        """
        Format matches into a dictionary row for output.
        
        Args:
            col: Original column name
            stage: Stage identifier (e.g., "stage1")
            detail: Method name (e.g., "std_exact")
            matches: List of (field, score, source) tuples
            
        Returns:
            Dictionary row for DataFrame
        """
        row = {"query": col, "stage": stage, "method": detail}
        
        for i, (field, score, source) in enumerate(matches[:self.top_k], start=1):
            row[f"match{i}"] = field
            row[f"match{i}_score"] = round(score, 4)
            row[f"match{i}_source"] = source
        
        return row
    
    def _run_cascade(
        self,
        col: str,
        stage: str,
        strategies: List[MatchStrategy]
    ) -> Dict[str, Any]:
        """
        Run a cascade of matching strategies.
        
        Cascade logic:
        - Try each strategy in order
        - If a strategy returns results with top1_score >= threshold, stop and return
        - If top1_score < threshold, continue to next strategy
        - Always keep the best result seen so far
        
        Args:
            col: Column name to match
            stage: Stage identifier
            strategies: List of matching strategies to try in order
            
        Returns:
            Formatted result dict with the best match found 
        """
        best_result = None
        best_score = -1.0
        
        for strategy in strategies:
            matches = strategy.match_func(col)
            
            if not matches:
                continue
            
            # Format the result
            result = self.format_matches_to_row(
                col=col,
                stage=stage,
                detail=strategy.name,
                matches=matches[:self.top_k]
            )
            
            # Get top1 score
            top1_score = matches[0][1] if matches else 0.0
            
            # Update best result if this is better
            if top1_score > best_score:
                best_result = result
                best_score = top1_score
            
            # Early stop if threshold is met
            if top1_score >= strategy.threshold:
                logger.info(
                    f"[{stage}] '{col}' matched by {strategy.name} "
                    f"(score={top1_score:.3f} >= thresh={strategy.threshold}), stopping cascade"
                )
                return best_result
            else:
                logger.info(
                    f"[{stage}] '{col}' matched by {strategy.name} "
                    f"(score={top1_score:.3f} < thresh={strategy.threshold}), continuing cascade"
                )
        
        # Return best result found (even if below threshold)
        if best_result:
            logger.info(
                f"[{stage}] '{col}' cascade complete, returning best result "
                f"(score={best_score:.3f})"
            )
            return best_result
        
        return {}
    
    def stage1_match(self, col: str) -> Dict[str, Any]:
        """Stage 1: Dictionary matching cascade."""
        strategies = [
            MatchStrategy("std_exact", self.std_exact.match, threshold=1.0),
        ]
        
        # Add alias strategies only if alias dict exists
        if self.has_alias_dict and self.alias_exact:
            strategies.append(
                MatchStrategy("alias_exact", self.alias_exact.match, threshold=1.0)
            )
        
        strategies.append(
            MatchStrategy("std_fuzzy", self.std_fuzzy.match, threshold=FUZZY_THRESH / 100)
        )
        
        if self.has_alias_dict and self.alias_fuzzy:
            strategies.append(
                MatchStrategy("alias_fuzzy", self.alias_fuzzy.match, threshold=FUZZY_THRESH / 100)
            )
        
        return self._run_cascade(col, "stage1", strategies)

    def stage2_match(self, col: str) -> Dict[str, Any]:
        """Stage 2: Field matching with combined results."""
        strategies = [
            MatchStrategy("numeric", self.numeric_combined.match, threshold=NUMERIC_THRESH),
            MatchStrategy("semantic", self.semantic_combined.match, threshold=FIELD_ALIAS_THRESH),
        ]
        
        return self._run_cascade(col, "stage2", strategies)
    
    def stage3_match(self, col: str) -> Dict[str, Any]:
        """Stage 3: Value matching cascade."""
        if self.is_col_numeric(col):
            return {}
        
        strategies = [
            MatchStrategy("value_std", self.value_std.match, threshold=VALUE_PERCENTAGE_THRESH),
            MatchStrategy("ontology", self.ontology.match, threshold=VALUE_PERCENTAGE_THRESH),
        ]
        return self._run_cascade(col, "stage3", strategies)
    
    def run_schema_mapping(self) -> pd.DataFrame:
        """
        Run schema mapping pipeline.
        
        Returns:
            DataFrame with mapping results
        """
        results = []
        
        for col in self.df.columns:
            # Check invalid
            is_invalid = check_invalid(self.df, col)
            if is_invalid:
                results.append({
                    "query": col,
                    "stage": "invalid",
                    "method": is_invalid
                })
                continue
            
            # Stage 1: Dictionary matching
            t0 = time.perf_counter()
            row = self.stage1_match(col)
            if row:
                results.append(row)
                logger.info(f"[Stage1] '{col}' matched in {time.perf_counter() - t0:.2f}s")
                continue

            # Stage 3: Value matching
            t0 = time.perf_counter()
            row = self.stage3_match(col)
            if row:
                results.append(row)
                logger.info(f"[Stage3] '{col}' matched in {time.perf_counter() - t0:.2f}s")
                continue
            
            # Stage 2: Field matching
            t0 = time.perf_counter()
            row = self.stage2_match(col)
            if row:
                results.append(row)
                logger.info(f"[Stage2] '{col}' matched in {time.perf_counter() - t0:.2f}s")
                continue
            
            # No match found - skip this column (original behavior)
            logger.warning(f"[NoMatch] '{col}' did not match any stage")
        
        df_out = pd.DataFrame(results)
        out_file = self.output_file.replace(".csv", f"_{self.mode}.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved results to {out_file}")
        
        return df_out
    
    def run_stage4_from_manual(self, manual_csv: str) -> pd.DataFrame:
        """
        Load manual results, run Stage4 (Gemini LLM) on low-confidence Stage2 matches.
        
        Args:
            manual_csv: Path to manual review CSV
            
        Returns:
            DataFrame with Stage4 results
        """
        df_manual = pd.read_csv(manual_csv)
        
        # Identify columns that need Stage 4
        # Criteria: stage2 with match1_score < FIELD_ALIAS_THRESH
        if "stage" in df_manual.columns and "match1_score" in df_manual.columns:
            mask = (
                (df_manual["stage"] == "stage2") &
                (df_manual["match1_score"] < FIELD_ALIAS_THRESH)
            )
            pending_cols = df_manual.loc[mask, "query"].tolist()
            
            logger.info(
                f"[Stage4] Found {len(pending_cols)} Stage2 columns with "
                f"score < {FIELD_ALIAS_THRESH}"
            )
        elif "query" in df_manual.columns:
            logger.warning("[Stage4] 'stage' or 'match1_score' not found; processing all queries.")
            pending_cols = df_manual["query"].dropna().astype(str).unique().tolist()
        else:
            raise ValueError("Expected column 'query' in manual CSV.")
        
        if not pending_cols:
            logger.info("[Stage4] No columns need Stage4 processing")
            return pd.DataFrame()
        
        results = []
        for col in pending_cols:
            try:
                # Use Gemini matcher
                gemini_matches = self.gemini.match(col)
                
                if gemini_matches:
                    row = self.format_matches_to_row(
                        col=col,
                        stage="stage4",
                        detail="gemini",
                        matches=gemini_matches
                    )
                    results.append(row)
                    logger.info(f"[Stage4] '{col}' matched by Gemini")
                else:
                    logger.warning(f"[Stage4] '{col}' no matches from Gemini")
                    # Record empty result
                    results.append({
                        "query": col,
                        "stage": "stage4",
                        "method": "gemini_no_match"
                    })
            except Exception as e:
                logger.error(f"[Stage4] Error processing '{col}': {e}")
                results.append({
                    "query": col,
                    "stage": "stage4",
                    "method": "gemini_error",
                    "error": str(e)
                })
        
        df_out = pd.DataFrame(results)
        out_file = manual_csv.replace(".csv", "_stage4.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"[Stage4] Saved {len(results)} results to {out_file}")
        
        return df_out