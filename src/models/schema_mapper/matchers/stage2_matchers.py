"""Stage 2: Value-based matching."""
from typing import List, Tuple, Dict
import torch
from .base import BaseMatcher
from ..config import VALUE_DICT_THRESH, VALUE_PERCENTAGE_THRESH
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class ValueDictMatcher(BaseMatcher):
    """Value-to-field aggregation matching using standard value dictionary."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        if not getattr(self.engine, "value_texts", None) or \
           getattr(self.engine, "value_embs", None) is None:
            return []
        
        # Configuration
        max_values = 100
        early_stop_min = 40
        early_stop_margin = 0.10
        
        # 1) Sample unique values
        unique_values = self.engine.unique_values(col, cap=max_values)
        if not unique_values:
            return []
        
        with torch.no_grad():
            v_embs = self.engine.dict_model.encode(unique_values, convert_to_tensor=True)
            v_embs = torch.nn.functional.normalize(v_embs, p=2, dim=1)
            
            device = self.engine.value_embs.device
            v_embs = v_embs.to(device)
            
            sims = v_embs @ self.engine.value_embs.T  # [M, N]
        
        # 2) Aggregate value→field matches
        field_count: Dict[str, int] = {}
        field_sum_score: Dict[str, float] = {}
        field_example: Dict[str, Tuple[str, float]] = {}
        hit_unique_count = 0
        total_unique = len(unique_values)
        
        for i, v in enumerate(unique_values):
            row = sims[i]
            k = min(10, len(row))
            top_scores, top_idx = torch.topk(row, k=k)
            
            filtered = [(float(s), int(j)) for s, j in zip(top_scores, top_idx)
                       if float(s) >= VALUE_DICT_THRESH]
            if not filtered:
                continue
            
            hit_unique_count += 1
            
            # Per-value best for each field
            per_value_best: Dict[str, float] = {}
            for s, j in filtered:
                fields = self.engine.value_fields_list[j] or []
                for f in fields:
                    if s > per_value_best.get(f, 0.0):
                        per_value_best[f] = s
                        if (f not in field_example) or (s > field_example[f][1]):
                            field_example[f] = (v, s)
            
            # Accumulate counts and scores
            for f, s in per_value_best.items():
                field_count[f] = field_count.get(f, 0) + 1
                field_sum_score[f] = field_sum_score.get(f, 0.0) + s
            
            # Early stopping
            if hit_unique_count >= early_stop_min:
                best_prop = max(cnt / total_unique for cnt in field_count.values()) \
                        if field_count else 0.0
                if best_prop >= (VALUE_PERCENTAGE_THRESH + early_stop_margin):
                    break
        
        if hit_unique_count == 0 or not field_count:
            return []
        
        # 3) Calculate scores: proportion + tie-break
        results: List[Tuple[str, float, str, float, int]] = []
        for f, cnt in field_count.items():
            proportion = cnt / total_unique
            if proportion < VALUE_PERCENTAGE_THRESH:
                continue
            example_val = field_example.get(f, ("", 0.0))[0]
            avg_score = field_sum_score.get(f, 0.0) / max(1, cnt)
            results.append((f, proportion, example_val, avg_score, cnt))
        
        if not results:
            return []
        
        # Sort: proportion > avg_score > count
        results.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
        
        # Return (field, score, source)
        return [(f, sc, ex) for (f, sc, ex, _avg, _cnt) in results]


class OntologyMatcher(BaseMatcher):
    """Value-based matching using NCI ontology."""
    
    def match(self, col: str) -> List[Tuple[str, float, str]]:
        unique_values = set(self.engine.unique_values(col, cap=20))
        unique_values.add(col)
        
        if not unique_values:
            return []
        
        cats = self.engine.nci_client.map_value_to_schema(unique_values)
        if not cats:
            return []
        
        den = len(unique_values)
        if den == 0:
            return []
        
        candidates = [(k, v / den, col) for k, v in cats.items()]
        matches = [m for m in candidates if m[1] > VALUE_PERCENTAGE_THRESH]
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:self.engine.top_k]
        
        # Special handling for cancer_type
        has_cancer = any(m[0] == "cancer_type" for m in matches)
        has_disease = any(m[0] == "disease" for m in matches)
        
        if has_cancer and not has_disease and len(matches) < self.engine.top_k:
            matches.append(("disease", cats["cancer_type"] / den, col))
        if has_cancer and len(matches) < self.engine.top_k:
            matches.append(("cancer_type_details", cats["cancer_type"] / den, col))
        
        return matches