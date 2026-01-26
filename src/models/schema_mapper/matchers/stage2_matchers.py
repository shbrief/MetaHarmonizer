"""Stage 2: Numeric and semantic field matching."""
from typing import List, Tuple, Dict
import torch
from sentence_transformers import util
from .base import BaseMatcher
from src.utils.schema_mapper_utils import normalize
from src.utils.numeric_match_utils import (
    strip_units_and_tags, 
    detect_numeric_semantic, 
    family_boost
)
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


# ============= Individual Matchers (kept for potential separate use) =============

class NumericStandardMatcher(BaseMatcher):
    """Numeric field matching against standard fields."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """Match numeric columns against standard numeric fields."""
        if not self.engine.is_col_numeric(col):
            return []
        
        self._ensure_std_numeric_index()
        
        if not hasattr(self.engine, '_std_numeric_fields') or \
           not self.engine._std_numeric_fields:
            return []
        
        key_raw = normalize(col)
        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)
        
        emb = self.engine._enc(key_clean or key_raw)
        
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self.engine._std_numeric_embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k, len(sims)))
        
        numeric_scores = []
        for score, idx in zip(top[0], top[1]):
            std_field = self.engine._std_numeric_fields[int(idx)]
            base = float(score)
            bonus = family_boost(std_field, family)
            final = base + bonus
            numeric_scores.append((std_field, final, ""))
        
        return sorted(numeric_scores, key=lambda x: x[1], reverse=True)
    
    def _ensure_std_numeric_index(self):
        """Lazy-build standard numeric field index."""
        if hasattr(self.engine, '_std_numeric_embs'):
            return
        
        curated_df = self.engine.curated_df
        std_numeric = curated_df[
            curated_df['is_numeric_field'] == 'yes'
        ]['field_name'].unique().tolist()
        
        if not std_numeric:
            self.engine._std_numeric_fields = []
            self.engine._std_numeric_embs = torch.empty(0)
            logger.warning("[NumericStd] No standard numeric fields found")
            return
        
        self.engine._std_numeric_fields = std_numeric
        self.engine._std_numeric_fields_normed = [normalize(f) for f in std_numeric]
        self.engine._std_numeric_embs = self.engine.dict_model.encode(
            self.engine._std_numeric_fields_normed,
            convert_to_tensor=True
        )
        
        logger.info(f"[NumericStd] Indexed {len(std_numeric)} standard numeric fields")


class NumericAliasMatcher(BaseMatcher):
    """Numeric field matching against alias sources."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """Match numeric columns against alias numeric sources."""
        if not self.engine.has_alias_dict:
            return []
        
        if not self.engine.is_col_numeric(col):
            return []
        
        self.engine._ensure_numeric_index()
        if self.engine._numeric_embs is None or len(self.engine.numeric_sources) == 0:
            return []
        
        key_raw = normalize(col)
        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)
        
        emb = self.engine._enc(key_clean or key_raw)
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self.engine._numeric_embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k, len(sims)))
        
        field_best: Dict[str, Tuple[float, str]] = {}
        
        for score, idx in zip(top[0], top[1]):
            src_name = self.engine.numeric_sources[int(idx)]
            matching_rows = self.engine.df_num[
                self.engine.df_num['source'] == src_name
            ]
            
            for f in matching_rows['field_name'].unique():
                base = float(score)
                bonus = family_boost(f, family)
                final = base + bonus
                
                if f not in field_best or final > field_best[f][0]:
                    field_best[f] = (final, src_name)
        
        numeric_scores = [
            (field, score, src) 
            for field, (score, src) in field_best.items()
        ]
        
        numeric_scores.sort(key=lambda x: x[1], reverse=True)
        
        return numeric_scores


# ============= Combined Matchers (for Stage 2) =============

class NumericCombinedMatcher(BaseMatcher):
    """Combined numeric matching: merges standard + alias results."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """
        Match numeric columns against both standard and alias sources.
        Merges results and returns top-k by score.
        """
        if not self.engine.is_col_numeric(col):
            return []
        
        # Get results from both matchers
        std_matcher = NumericStandardMatcher(self.engine)
        alias_matcher = NumericAliasMatcher(self.engine)
        
        std_results = std_matcher.match(col)
        alias_results = alias_matcher.match(col)
        
        # Merge results: deduplicate by field, keep highest score
        field_best: Dict[str, Tuple[float, str]] = {}
        
        for field, score, source in std_results:
            field_best[field] = (score, source)
        
        for field, score, source in alias_results:
            if field not in field_best or score > field_best[field][0]:
                field_best[field] = (score, source)
        
        # Convert to list and sort
        combined = [
            (field, score, src)
            for field, (score, src) in field_best.items()
        ]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        top_k = combined[:self.engine.top_k]
        
        logger.info(
            f"[NumericCombined] Column='{col}' "
            f"std={len(std_results)} alias={len(alias_results)} "
            f"merged={len(combined)} top_k={len(top_k)}"
        )
        
        return top_k


class SemanticStandardMatcher(BaseMatcher):
    """Semantic field matching against standard fields."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """Match column names against standard fields using semantic similarity."""
        key = normalize(col)
        emb = self.engine._enc(key)
        
        if not hasattr(self.engine, '_std_field_embs'):
            self.engine._std_field_embs = self.engine.dict_model.encode(
                self.engine.standard_fields_normed,
                convert_to_tensor=True
            )
            logger.info(f"[SemanticStd] Encoded {len(self.engine.standard_fields)} standard fields")
        
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self.engine._std_field_embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k, len(sims)))
        
        matches = []
        for score, idx in zip(top[0], top[1]):
            std_field = self.engine.standard_fields[int(idx)]
            matches.append((std_field, float(score), ""))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)


class SemanticAliasMatcher(BaseMatcher):
    """Semantic field matching against alias sources."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """Match column names against alias sources using semantic similarity."""
        if not self.engine.has_alias_dict or self.engine.alias_embs is None:
            return []
        
        key = normalize(col)
        emb = self.engine._enc(key)
        
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self.engine.alias_embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k * 2, len(sims)))
        
        field_best: Dict[str, Tuple[float, str]] = {}
        
        for score, idx in zip(top[0], top[1]):
            alias = self.engine.sources_keys[int(idx)]
            score_val = float(score)
            
            for f in self.engine.sources_to_fields[alias]:
                if f not in field_best or score_val > field_best[f][0]:
                    field_best[f] = (score_val, alias)
        
        alias_scores = [
            (field, score, src) 
            for field, (score, src) in field_best.items()
        ]
        alias_scores.sort(key=lambda x: x[1], reverse=True)
        
        return alias_scores[:self.engine.top_k]


class SemanticCombinedMatcher(BaseMatcher):
    """Combined semantic matching: merges standard + alias results."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        """
        Match column names against both standard and alias sources.
        Merges results and returns top-k by score.
        """
        # Get results from both matchers
        std_matcher = SemanticStandardMatcher(self.engine)
        alias_matcher = SemanticAliasMatcher(self.engine)
        
        std_results = std_matcher.match(col)
        alias_results = alias_matcher.match(col)
        
        # Merge results: deduplicate by field, keep highest score
        field_best: Dict[str, Tuple[float, str]] = {}
        
        for field, score, source in std_results:
            field_best[field] = (score, source)
        
        for field, score, source in alias_results:
            if field not in field_best or score > field_best[field][0]:
                field_best[field] = (score, source)
        
        # Convert to list and sort
        combined = [
            (field, score, src)
            for field, (score, src) in field_best.items()
        ]
        combined.sort(key=lambda x: x[1], reverse=True)
        
        top_k = combined[:self.engine.top_k]
        
        logger.info(
            f"[SemanticCombined] Column='{col}' "
            f"std={len(std_results)} alias={len(alias_results)} "
            f"merged={len(combined)} top_k={len(top_k)}"
        )
        
        return top_k