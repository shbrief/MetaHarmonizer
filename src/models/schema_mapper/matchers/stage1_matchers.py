"""Stage 1: Dictionary and fuzzy matching."""
from typing import List, Tuple
from rapidfuzz import process, fuzz
from .base import BaseMatcher
from ..config import FUZZY_THRESH
from src.utils.schema_mapper_utils import normalize

class StandardExactMatcher(BaseMatcher):
    """Standard field exact matching."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        norm = normalize(col)
        if norm in self.engine.normed_std_to_std:
            std_field = self.engine.normed_std_to_std[norm]
            return [(std_field, 1.0, "")]
        return []

class AliasExactMatcher(BaseMatcher):
    """Alias exact matching after normalization."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        # Safety check
        if not self.engine.has_alias_dict:
            return []
        
        norm = normalize(col)
        alias_fields = self.engine.sources_to_fields.get(norm, [])
        if not alias_fields:
            return []
        
        matches = []
        for f in alias_fields:
            src = self.engine.normed_source_to_source.get(norm, f)
            matches.append((f, 1.0, src))
        return sorted(matches, key=lambda x: x[1], reverse=True)

class StandardFuzzyMatcher(BaseMatcher):
    """Standard field fuzzy matching."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        norm = normalize(col)
        candidates = process.extract(
            norm,
            self.engine.standard_fields_normed,
            scorer=fuzz.token_sort_ratio,
            limit=self.engine.top_k
        )
        
        matches = []
        for cand_norm, score, _ in candidates:
            if score >= FUZZY_THRESH:
                std_field = self.engine.normed_std_to_std[cand_norm]
                matches.append((std_field, score / 100.0, ""))
        return sorted(matches, key=lambda x: x[1], reverse=True)

class AliasFuzzyMatcher(BaseMatcher):
    """Alias fuzzy matching using token_sort_ratio."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        # Safety check
        if not self.engine.has_alias_dict or not self.engine.sources_keys:
            return []
        
        norm = normalize(col)
        candidates = process.extract(
            norm,
            self.engine.sources_keys,
            scorer=fuzz.token_sort_ratio,
            limit=self.engine.top_k
        )
        
        best = {}
        for cand, score, _ in candidates:
            if score >= FUZZY_THRESH:
                for std_field in self.engine.sources_to_fields[cand]:
                    src = self.engine.normed_source_to_source.get(cand, std_field)
                    s = score / 100.0
                    if (std_field not in best) or (best[std_field][0] < s):
                        best[std_field] = (s, src)
        
        matches = [(f, sc, src) for f, (sc, src) in best.items()]
        return sorted(matches, key=lambda x: x[1], reverse=True)