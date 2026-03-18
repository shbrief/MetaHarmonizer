"""Stage 3: Numeric and semantic field matching."""
import re
from typing import List, Tuple, Dict, Set
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


def _merge_top_k(
    std_results: List[Tuple[str, float, str]],
    alias_results: List[Tuple[str, float, str]],
    top_k: int,
) -> List[Tuple[str, float, str]]:
    """Merge two result lists, keeping highest score per field, returning top-k."""
    field_best: Dict[str, Tuple[float, str]] = {}
    for field, score, source in std_results:
        field_best[field] = (score, source)
    for field, score, source in alias_results:
        if field not in field_best or score > field_best[field][0]:
            field_best[field] = (score, source)
    combined = [(field, score, src) for field, (score, src) in field_best.items()]
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]


# ============= Treatment Boost Logic =============

TREATMENT_KEYWORDS_SUBSTRING = ("therapy", "chemo", "surgery", "radiation", "treatment", "drug", "regimen")
TREATMENT_BOOST = 0.2

_TX_PATTERN = re.compile(r'(?:^tx_|_tx$|_tx_)')


def _is_treatment_column(col_normed: str) -> bool:
    """Check if a normalized column name suggests treatment-related content."""
    if _TX_PATTERN.search(col_normed):
        return True
    return any(kw in col_normed for kw in TREATMENT_KEYWORDS_SUBSTRING)


def _get_treatment_fields(engine) -> Set[str]:
    """Get treatment-related field names from curated dict (cached on engine)."""
    if not hasattr(engine, '_treatment_fields_cache'):
        treatment_fields = set()
        if hasattr(engine, 'curated_df') and engine.curated_df is not None:
            for f in engine.curated_df['field_name'].unique():
                if 'treatment' in f.lower():
                    treatment_fields.add(f)
        engine._treatment_fields_cache = treatment_fields
        if treatment_fields:
            logger.info(f"[TreatmentBoost] Cached {len(treatment_fields)} treatment fields")
    return engine._treatment_fields_cache


def treatment_boost(field: str, col_normed: str, engine) -> float:
    """Return boost score if column is treatment-related and field is a treatment field."""
    if not _is_treatment_column(col_normed):
        return 0.0
    treatment_fields = _get_treatment_fields(engine)
    if field in treatment_fields:
        return TREATMENT_BOOST
    return 0.0


# ============= Abstract Bases (template-method pattern) =============

class _StandardEmbMatcher(BaseMatcher):
    """Template base for standard-field embedding matchers.

    Subclasses override:
        _guard()            — pre-condition check (default: True)
        _prepare_key()      — returns (key_raw, query_key, family)
        _extra_bonus()      — per-field bonus beyond treatment_boost (default: 0)
        _get_fields_and_embs() — lazy-build and return (fields, emb_tensor)
    """

    def _guard(self, col: str) -> bool:
        return True

    def _prepare_key(self, col: str):
        key_raw = normalize(col)
        return key_raw, key_raw, None

    def _extra_bonus(self, field: str, key_raw: str, family) -> float:
        return 0.0

    def _get_fields_and_embs(self):
        raise NotImplementedError

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        if not self._guard(col):
            return []
        fields, embs = self._get_fields_and_embs()
        if not fields:
            return []
        key_raw, query_key, family = self._prepare_key(col)
        emb = self.engine._enc(query_key)
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k * 3, len(sims)))
        results = []
        for score, idx in zip(top[0], top[1]):
            field = fields[int(idx)]
            bonus = self._extra_bonus(field, key_raw, family)
            bonus += treatment_boost(field, key_raw, self.engine)
            results.append((field, float(score) + bonus, ""))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.engine.top_k]


class _AliasEmbMatcher(BaseMatcher):
    """Template base for alias-source embedding matchers.

    Subclasses override:
        _guard()                — pre-condition check (default: True)
        _prepare_key()          — returns (key_raw, query_key, family)
        _extra_bonus()          — per-field bonus beyond treatment_boost (default: 0)
        _get_sources_and_embs() — return (sources_list, emb_tensor)
        _fields_for_source()    — map source name → field names
    """

    def _guard(self, col: str) -> bool:
        return True

    def _prepare_key(self, col: str):
        key_raw = normalize(col)
        return key_raw, key_raw, None

    def _extra_bonus(self, field: str, key_raw: str, family) -> float:
        return 0.0

    def _get_sources_and_embs(self):
        raise NotImplementedError

    def _fields_for_source(self, src_name: str) -> List[str]:
        raise NotImplementedError

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        if not self.engine.has_alias_dict or not self._guard(col):
            return []
        sources, embs = self._get_sources_and_embs()
        if embs is None or not sources:
            return []
        key_raw, query_key, family = self._prepare_key(col)
        emb = self.engine._enc(query_key)
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, embs)[0]
        top = torch.topk(sims, k=min(self.engine.top_k * 3, len(sims)))
        field_best: Dict[str, Tuple[float, str]] = {}
        for score, idx in zip(top[0], top[1]):
            src_name = sources[int(idx)]
            base = float(score)
            for f in self._fields_for_source(src_name):
                bonus = self._extra_bonus(f, key_raw, family)
                bonus += treatment_boost(f, key_raw, self.engine)
                final = base + bonus
                if f not in field_best or final > field_best[f][0]:
                    field_best[f] = (final, src_name)
        results = [(f, s, src) for f, (s, src) in field_best.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.engine.top_k]


class _CombinedMatcher(BaseMatcher):
    """Template base for combined (std + alias) matchers.

    Subclasses set class attributes _STD_CLASS, _ALIAS_CLASS, _TAG,
    and optionally override _guard().
    """
    _STD_CLASS = None
    _ALIAS_CLASS = None
    _TAG = ""

    def _guard(self, col: str) -> bool:
        return True

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        if not self._guard(col):
            return []
        std_results = self._STD_CLASS(self.engine).match(col)
        alias_results = self._ALIAS_CLASS(self.engine).match(col)
        merged = _merge_top_k(std_results, alias_results, self.engine.top_k)
        logger.info(
            f"[{self._TAG}] Column='{col}' "
            f"std={len(std_results)} alias={len(alias_results)} merged={len(merged)}"
        )
        return merged


# ============= Concrete Matchers =============

class NumericStandardMatcher(_StandardEmbMatcher):
    """Numeric field matching against standard fields."""

    def _guard(self, col):
        return self.engine.is_col_numeric(col)

    def _prepare_key(self, col):
        key_raw = normalize(col)
        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)
        return key_raw, key_clean or key_raw, family

    def _extra_bonus(self, field, key_raw, family):
        return family_boost(field, family)

    def _get_fields_and_embs(self):
        self.engine._ensure_std_numeric_index()
        return self.engine._std_numeric_fields, self.engine._std_numeric_embs


class SemanticStandardMatcher(_StandardEmbMatcher):
    """Semantic field matching against standard fields."""

    def _get_fields_and_embs(self):
        self.engine._ensure_std_field_embs()
        return self.engine.standard_fields, self.engine._std_field_embs


class NumericAliasMatcher(_AliasEmbMatcher):
    """Numeric field matching against alias sources."""

    def _guard(self, col):
        return self.engine.is_col_numeric(col)

    def _prepare_key(self, col):
        key_raw = normalize(col)
        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)
        return key_raw, key_clean or key_raw, family

    def _extra_bonus(self, field, key_raw, family):
        return family_boost(field, family)

    def _get_sources_and_embs(self):
        self.engine._ensure_numeric_index()
        return self.engine.numeric_sources, self.engine._numeric_embs

    def _fields_for_source(self, src_name):
        return self.engine.df_num[
            self.engine.df_num['source'] == src_name
        ]['field_name'].unique()


class SemanticAliasMatcher(_AliasEmbMatcher):
    """Semantic field matching against alias sources."""

    def _get_sources_and_embs(self):
        return self.engine.sources_keys, self.engine.alias_embs

    def _fields_for_source(self, src_name):
        return self.engine.sources_to_fields.get(src_name, [])


class NumericCombinedMatcher(_CombinedMatcher):
    """Combined numeric matching: merges standard + alias results."""
    _STD_CLASS = NumericStandardMatcher
    _ALIAS_CLASS = NumericAliasMatcher
    _TAG = "NumericCombined"

    def _guard(self, col):
        return self.engine.is_col_numeric(col)


class SemanticCombinedMatcher(_CombinedMatcher):
    """Combined semantic matching: merges standard + alias results."""
    _STD_CLASS = SemanticStandardMatcher
    _ALIAS_CLASS = SemanticAliasMatcher
    _TAG = "SemanticCombined"
