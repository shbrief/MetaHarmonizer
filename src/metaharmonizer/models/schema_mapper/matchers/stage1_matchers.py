"""Stage 1: Dictionary and fuzzy matching."""
from typing import List, Tuple
from rapidfuzz import process, fuzz
from .base import BaseMatcher
from metaharmonizer.utils.schema_mapper_utils import header_variants

# Double-matching (Issue #87): every name-based matcher tries BOTH the plain
# normalized header and its abbreviation-expanded form (header_variants), keeping
# the best score per field. Variants are raw-first and merges use strict '>', so
# ties keep the un-expanded form — expansion only wins when it scores higher.

class StandardExactMatcher(BaseMatcher):
    """Standard field exact matching."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        matches, seen = [], set()
        for norm in header_variants(col):
            std_field = self.engine.normed_std_to_std.get(norm)
            if std_field is not None and std_field not in seen:
                seen.add(std_field)
                matches.append((std_field, 1.0, ""))
        return matches

class AliasExactMatcher(BaseMatcher):
    """Alias exact matching after normalization."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        # Safety check
        if not self.engine.has_alias_dict:
            return []

        best = {}  # field -> source (first variant wins; all exact => score 1.0)
        for norm in header_variants(col):
            for f in self.engine.sources_to_fields.get(norm, []):
                if f not in best:
                    best[f] = self.engine.normed_source_to_source.get(norm, f)
        return [(f, 1.0, src) for f, src in best.items()]

class StandardFuzzyMatcher(BaseMatcher):
    """Standard field fuzzy matching."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        best = {}  # field -> score
        for norm in header_variants(col):
            candidates = process.extract(
                norm,
                self.engine.standard_fields_normed,
                scorer=fuzz.token_sort_ratio,
                limit=self.engine.top_k
            )
            for cand_norm, score, _ in candidates:
                if score >= self.engine.settings.fuzzy_thresh:
                    std_field = self.engine.normed_std_to_std[cand_norm]
                    s = score / 100.0
                    if std_field not in best or s > best[std_field]:
                        best[std_field] = s
        matches = [(f, s, "") for f, s in best.items()]
        return sorted(matches, key=lambda x: x[1], reverse=True)

class AliasFuzzyMatcher(BaseMatcher):
    """Alias fuzzy matching using token_sort_ratio."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        # Safety check
        if not self.engine.has_alias_dict or not self.engine.sources_keys:
            return []

        best = {}
        for norm in header_variants(col):
            candidates = process.extract(
                norm,
                self.engine.sources_keys,
                scorer=fuzz.token_sort_ratio,
                limit=self.engine.top_k
            )
            for cand, score, _ in candidates:
                if score >= self.engine.settings.fuzzy_thresh:
                    for std_field in self.engine.sources_to_fields[cand]:
                        src = self.engine.normed_source_to_source.get(cand, std_field)
                        s = score / 100.0
                        if (std_field not in best) or (best[std_field][0] < s):
                            best[std_field] = (s, src)

        matches = [(f, sc, src) for f, (sc, src) in best.items()]
        return sorted(matches, key=lambda x: x[1], reverse=True)
