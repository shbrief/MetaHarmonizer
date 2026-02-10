"""Main schema mapping engine."""
import os
import time
import pandas as pd
from typing import Dict, Any, List, Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util
import torch

from .config import (
    FUZZY_THRESH, OUTPUT_DIR, ALIAS_DICT_PATH, FIELD_MODEL,
    NUMERIC_THRESH, FIELD_ALIAS_THRESH, VALUE_PERCENTAGE_THRESH,
    LLM_THRESHOLD 
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
from .matchers.stage4_matchers import LLMMatcher 
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
    - Stage 4: LLM fallback (auto mode only)
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
        
        # Stage 2
        self.numeric_combined = NumericCombinedMatcher(self)
        self.semantic_combined = SemanticCombinedMatcher(self)
        
        # Stage 3
        self.value_std = ValueStandardMatcher(self)
        self.ontology = OntologyMatcher(self)
        
        # Stage 4 - LLM (only in auto mode)
        if self.mode == "auto":
            try:
                self.llm = LLMMatcher(self)
                logger.info("[Engine] LLM matcher initialized for auto mode")
            except Exception as e:
                logger.warning(f"[Engine] Failed to initialize LLM matcher: {e}")
                self.llm = None
        else:
            self.llm = None
    
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
    
    def stage4_match(self, col: str) -> Dict[str, Any]:
        """Stage 4: LLM matching."""
        if self.llm is None:
            return {}
        
        matches = self.llm.match(col)
        
        if not matches:
            return {}
        
        return self.format_matches_to_row(
            col=col,
            stage="stage4",
            detail="llm",
            matches=matches
        )
    
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
                # Auto mode: check if LLM fallback should be triggered
                if self.mode == "auto" and self.llm and row.get('match1_score', 0) < LLM_THRESHOLD:
                    logger.info(
                        f"[Stage2] '{col}' low confidence "
                        f"({row.get('match1_score', 0):.3f} < {LLM_THRESHOLD}), "
                        f"trying LLM fallback"
                    )
                    
                    # Try LLM fallback
                    t0_llm = time.perf_counter()
                    llm_row = self.stage4_match(col)
                    
                    if llm_row and llm_row.get('match1_score', 0) > row.get('match1_score', 0):
                        logger.info(
                            f"[Stage4] '{col}' LLM improved: "
                            f"{row.get('match1_score', 0):.3f} → {llm_row.get('match1_score', 0):.3f}"
                        )
                        results.append(llm_row)
                    else:
                        logger.info(f"[Stage4] '{col}' keeping Stage2 result")
                        results.append(row)
                    
                    logger.info(f"[Stage4] '{col}' in {time.perf_counter() - t0_llm:.2f}s")
                else:
                    results.append(row)
                
                logger.info(f"[Stage2] '{col}' matched in {time.perf_counter() - t0:.2f}s")
                continue
            
            # No match found - try LLM as last resort (auto mode only)
            if self.mode == "auto" and self.llm:
                logger.info(f"[NoMatch] '{col}' trying LLM as last resort")
                t0 = time.perf_counter()
                row = self.stage4_match(col)
                if row:
                    results.append(row)
                    logger.info(f"[Stage4] '{col}' matched in {time.perf_counter() - t0:.2f}s")
                    continue
            
            # Really no match found
            logger.warning(f"[NoMatch] '{col}' did not match any stage")
        
        df_out = pd.DataFrame(results)
        out_file = self.output_file.replace(".csv", f"_{self.mode}.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved results to {out_file}")
        
        return df_out
    
    def run_llm_on_file(
        self,
        input_csv: str,
        output_csv: str,
        stage_filter: Optional[List[str]] = None,
        merge_results: bool = False
    ) -> pd.DataFrame:
        """
        Run LLM on an existing results file (manual mode).
        
        Args:
            input_csv: Path to existing results CSV
            output_csv: Path to save LLM results
            stage_filter: Only re-match specific stages (e.g., ['stage2'])
            merge_results: If True, merge with original results
            
        Returns:
            DataFrame with LLM results (merged or standalone)
        """
        # Initialize LLM matcher if not already done
        if self.llm is None:
            try:
                self.llm = LLMMatcher(self)
                logger.info("[Engine] LLM matcher initialized for manual mode")
            except Exception as e:
                logger.error(f"[Engine] Failed to initialize LLM matcher: {e}")
                return pd.DataFrame()
        
        # Load input file
        try:
            df_input = pd.read_csv(input_csv)
            logger.info(f"[Engine] Loaded {len(df_input)} rows from {input_csv}")
        except Exception as e:
            logger.error(f"[Engine] Failed to load input CSV: {e}")
            raise
        
        # Identify queries to re-match
        needs_rematching = pd.Series([True] * len(df_input))
        
        # Filter by confidence threshold
        if 'match1_score' in df_input.columns:
            needs_rematching &= (
                df_input['match1_score'].isna() | 
                (df_input['match1_score'] < LLM_THRESHOLD)
            )
        
        # Filter by stage
        if stage_filter and 'stage' in df_input.columns:
            needs_rematching &= df_input['stage'].isin(stage_filter)
        
        queries_to_rematch = df_input[needs_rematching]['query'].tolist()
        
        logger.info(f"[Engine] Found {len(queries_to_rematch)} queries for LLM")
        logger.info(f"[Engine] Criteria: match1_score < {LLM_THRESHOLD}")
        if stage_filter:
            logger.info(f"[Engine] Stage filter: {stage_filter}")
        
        if len(queries_to_rematch) == 0:
            logger.info("[Engine] No queries need LLM")
            return pd.DataFrame()
        
        # Re-match queries
        results = []
        for idx, query in enumerate(queries_to_rematch, 1):
            logger.info(f"[Engine] LLM {idx}/{len(queries_to_rematch)}: '{query}'")
            
            row = self.stage4_match(query)
            if row:
                results.append(row)
            else:
                results.append({
                    'query': query,
                    'stage': 'stage4',
                    'method': 'llm_no_match'
                })
            
            # Rate limiting
            time.sleep(2.0)
        
        df_llm = pd.DataFrame(results)
        
        # Save or merge
        if merge_results:
            # Merge: replace original results with LLM results
            rematched_queries = set(df_llm['query'].tolist())
            df_keep = df_input[~df_input['query'].isin(rematched_queries)]
            df_merged = pd.concat([df_keep, df_llm], ignore_index=True)
            
            df_merged.to_csv(output_csv, index=False)
            logger.info(f"[Engine] Saved merged results ({len(df_merged)} rows) to {output_csv}")
            return df_merged
        else:
            # Save only LLM results
            df_llm.to_csv(output_csv, index=False)
            logger.info(f"[Engine] Saved LLM results ({len(df_llm)} rows) to {output_csv}")
            return df_llm