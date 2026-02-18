"""Stage 2: Value-based matching."""
from typing import List, Tuple, Dict
import math
import torch
from .base import BaseMatcher
from ..config import VALUE_DICT_THRESH, VALUE_PERCENTAGE_THRESH, VALUE_UNIQUE_CAP
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class ValueDictMatcher(BaseMatcher):
    """Value-to-field aggregation matching using standard value dictionary."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        if not getattr(self.engine, "value_texts", None) or \
           getattr(self.engine, "value_embs", None) is None:
            return []

        unique_values = self.engine.unique_values(col)
        if not unique_values:
            return []

        # Notes/ID columns: too many unique values → skip
        if len(unique_values) > VALUE_UNIQUE_CAP:
            return []

        total_unique = len(unique_values)
        freq = self.engine.value_frequencies(col)  # Dict[str, float], sums to 1.0
        default_freq = 1.0 / total_unique
        log_weights = {
            v: math.log1p(freq.get(v, default_freq) * 10)
            for v in unique_values
        }
        total_log_sum = sum(log_weights.values())

        with torch.no_grad():
            v_embs = self.engine.dict_model.encode(unique_values, convert_to_tensor=True)
            v_embs = torch.nn.functional.normalize(v_embs, p=2, dim=1)
            v_embs = v_embs.to(self.engine.value_embs.device)
            sims = v_embs @ self.engine.value_embs.T  # [M, N]

        field_count: Dict[str, int] = {}
        field_log_sum: Dict[str, float] = {}   # log-freq weighted accumulator (for avg_score denominator)
        field_score_sum: Dict[str, float] = {} # log-freq weighted similarity score
        field_example: Dict[str, Tuple[str, float]] = {}

        for i, v in enumerate(unique_values):
            row = sims[i]
            k = min(10, len(row))
            top_scores, top_idx = torch.topk(row, k=k)

            filtered = [(float(s), int(j)) for s, j in zip(top_scores, top_idx)
                        if float(s) >= VALUE_DICT_THRESH]
            if not filtered:
                continue

            # log-compressed frequency weight: log(1 + freq*10)
            w = log_weights[v]

            per_value_best: Dict[str, float] = {}
            for s, j in filtered:
                fields = self.engine.value_fields_list[j] or []
                for f in fields:
                    if s > per_value_best.get(f, 0.0):
                        per_value_best[f] = s
                        if (f not in field_example) or (s > field_example[f][1]):
                            field_example[f] = (v, s)

            for f, s in per_value_best.items():
                field_count[f] = field_count.get(f, 0) + 1
                field_log_sum[f] = field_log_sum.get(f, 0.0) + w
                field_score_sum[f] = field_score_sum.get(f, 0.0) + s * w

        if not field_count:
            return []

        results: List[Tuple[str, float, str, float, int]] = []
        for f, cnt in field_count.items():
            # proportion: log-freq weighted (field_log_sum / total_log_sum)
            proportion = field_log_sum[f] / total_log_sum
            if proportion < VALUE_PERCENTAGE_THRESH:
                continue
            # avg_score: log-freq weighted similarity (tie-break only)
            avg_score = field_score_sum[f] / max(field_log_sum[f], 1e-9)
            example_val = field_example.get(f, ("", 0.0))[0]
            results.append((f, proportion, example_val, avg_score, cnt))

        if not results:
            return []

        # Sort: proportion > avg_score > unique_count
        results.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
        return [(f, sc, ex) for (f, sc, ex, _avg, _cnt) in results]


class OntologyMatcher(BaseMatcher):
    """Value-based matching using NCI ontology."""

    def match(self, col: str) -> List[Tuple[str, float, str]]:
        unique_values_list = self.engine.unique_values(col)
        ONTOLOGY_COL_NAME_BONUS = 0.3

        # Notes/ID columns: too many unique values → skip
        if len(unique_values_list) > VALUE_UNIQUE_CAP:
            return []

        unique_values = set(unique_values_list)
        unique_values.add(col)

        # hits: {field: [matched_value, ...]}
        hits = self.engine.nci_client.map_value_to_schema(unique_values)
        if not hits:
            return []

        # Freq weights for data values only (col name handled separately)
        freq = self.engine.value_frequencies(col)
        default_freq = 1.0 / max(len(unique_values_list), 1)
        total_log_sum = sum(
            math.log1p(freq.get(v, default_freq) * 10)
            for v in unique_values_list
        )

        candidates = []
        for field, matched_values in hits.items():
            w_sum = sum(
                math.log1p(freq.get(v, default_freq) * 10)
                for v in matched_values
                if v != col
            )
            proportion = w_sum / max(total_log_sum, 1e-9)
            # Col name match: add fixed bonus (col name is a strong semantic signal)
            if col in matched_values:
                proportion = min(proportion + ONTOLOGY_COL_NAME_BONUS, 1.0)
            candidates.append((field, proportion, col))

        matches = [m for m in candidates if m[1] > VALUE_PERCENTAGE_THRESH]
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:self.engine.top_k]

        # Special handling for cancer_type
        has_cancer = any(m[0] == "cancer_type" for m in matches)
        has_disease = any(m[0] == "disease" for m in matches)

        if has_cancer and not has_disease and len(matches) < self.engine.top_k:
            cancer_prop = next(m[1] for m in matches if m[0] == "cancer_type")
            matches.append(("disease", cancer_prop, col))
        if has_cancer and len(matches) < self.engine.top_k:
            cancer_prop = next(m[1] for m in matches if m[0] == "cancer_type")
            matches.append(("cancer_type_details", cancer_prop, col))

        return matches