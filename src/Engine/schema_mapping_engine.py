import json
import os
import time
from typing import Dict, List, Tuple, Any
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache
from src.CustomLogger.custom_logger import CustomLogger
from src.utils.schema_mapper_utils import normalize, is_numeric_column, extract_valid_value
from src.utils.invalid_column_utils import check_invalid
from src.utils.numeric_match_utils import strip_units_and_tags, detect_numeric_semantic, family_boost
from src.utils.ncit_match_utils import NCIClientSync

# === Configuration ===
logger = CustomLogger().custlogger(loglevel='WARNING')
OUTPUT_DIR = "data/schema_mapping_eval"
DICT_PATH = "data/schema/curated_fields_source_latest_with_flags.csv"
VALUE_DICT_PATH = os.getenv(
    "FIELD_VALUE_JSON") or "data/schema/field_value_dict.json"
FIELD_MODEL = "all-MiniLM-L6-v2"
FUZZY_THRESH = 90
NUMERIC_THRESH = 0.6
FIELD_ALIAS_THRESH = 0.5
VALUE_DICT_THRESH = 0.85
VALUE_PERCENTAGE_THRESH = 0.5
NOISE_VALUES = {
    "yes", "no", "true", "false", "unknown", "not reported", "not available",
    "na", "n/a", "none", "other", "missing", "not evaluated", "uninformative",
    "pending", "undetermined", "positive", "negative", "not applicable"
}


class SchemaMapEngine:
    """
    Performs schema mapping in three stages and outputs a CSV with columns:
      original_column | matched_stage | matched_stage_method |
      match1_field | match1_score | match1_source |
      match2_field | match2_score | match2_source |
      match3_field | match3_score | match3_source | ...
    - Stage1: Dict/fuzzy match against dictionary using alias names.
    - Stage2: Numeric/alias field matching based on header names and value types using SentenceTransformer embeddings.
    - Stage3: Value match using standard value dictionary and ncit.
    - Stage4: LLM matching (under development).
    - Mode: 'auto' runs Stage4 automatically if previous stages' scores are low,
            'manual' outputs Stage4 results for manual review, does not run Stage4.
    - top_k: Number of top matches to return for each column.
    - Output: Results saved to OUTPUT_DIR with filename based on input clinical data.
    - Requires SentenceTransformer and rapidfuzz for matching.
    - Uses a dictionary of field names and their aliases to perform matching.
    """

    def __init__(self,
                 clinical_data_path: str,
                 mode: str = "auto",
                 top_k: int = 5):
        if clinical_data_path.endswith(".tsv"):
            self.df = pd.read_csv(clinical_data_path, sep="\t", dtype=str)
        else:
            self.df = pd.read_csv(clinical_data_path, sep=",", dtype=str)

        logger.info(
            f"[Load] df_shape={self.df.shape} first_cols={list(self.df.columns[:5])}"
        )

        self.top_k = top_k
        self.mode = mode

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.basename(clinical_data_path)
        root, _ = os.path.splitext(base)
        self.output_file = os.path.join(OUTPUT_DIR, f"{root}.csv")

        # Load alias dictionary
        self.df_dict = pd.read_csv(DICT_PATH)

        # Normalize alias to fields mapping
        self.sources_to_fields = (self.df_dict.groupby(
            self.df_dict['source'].map(normalize))['field_name'].apply(
                lambda s: sorted(set(s))).to_dict())

        self.sources_keys = list(self.sources_to_fields.keys())

        self.normed_source_to_source = {}
        for _, row in self.df_dict.dropna(subset=['source']).iterrows():
            norm_src = normalize(row['source'])
            if norm_src not in self.normed_source_to_source:
                self.normed_source_to_source[norm_src] = row['source']

        # Load SentenceTransformer models
        self.dict_model = SentenceTransformer(FIELD_MODEL)
        self.alias_embs = self.dict_model.encode(self.sources_keys,
                                                 convert_to_tensor=True)

        # Prepare numeric sources
        self.df_num = self.df_dict[self.df_dict['is_numeric_field'] == 'yes']
        self.numeric_sources = self.df_num['source'].dropna().unique().tolist()
        self._numeric_embs = None  # lazy: set on first use

        # Load value dictionary
        self._load_value_dict()

        # Initialize NCI client
        self.nci_client = NCIClientSync()

    def _ensure_numeric_index(self):
        """Lazy-build numeric embedding index on first use."""
        if self._numeric_embs is not None:
            return
        self.norm_numeric = [normalize(s) for s in self.numeric_sources]
        if not self.norm_numeric:
            self._numeric_embs = torch.empty(0)
            return
        self._numeric_embs = self.dict_model.encode(self.norm_numeric,
                                                    convert_to_tensor=True)

    def _load_value_dict(self, json_path: str = VALUE_DICT_PATH):
        """
        Support { field: [value1, value2, ...], ... } format.
        Generate in-memory lists:
        - self.value_texts: List[str]               raw values (filtered)
        - self.value_fields_list: List[List[str]]   aligned field list per value
        """
        self.value_texts = []
        self.value_fields_list = []

        if not os.path.exists(json_path):
            logger.warning(f"[Init] Missing value dictionary: {json_path}")
            return

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for field, values in data.items():
            for v in values:
                v = str(v).strip()
                if not v or v.lower() in NOISE_VALUES:
                    continue
                self.value_texts.append(v)
                self.value_fields_list.append([field])

        logger.info(
            f"[Init] Loaded {len(self.value_texts)} value entries from {json_path}"
        )

    @lru_cache(maxsize=None)
    def is_col_numeric(self, col: str) -> bool:
        """Cached wrapper around is_numeric_column(df, col)."""
        return is_numeric_column(self.df, col)

    @lru_cache(maxsize=None)
    def _enc(self, text: str):
        return self.dict_model.encode(text, convert_to_tensor=True)

    def unique_values(self, col: str, cap: int | None = None) -> list[str]:
        if not hasattr(self, "_col_values_cache"):
            self._col_values_cache = {}

        if col not in self._col_values_cache:
            series = self.df[col].dropna().astype(str).apply(
                extract_valid_value)
            uniq, seen = [], set()
            for lst in series:
                for v in lst:
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
        matches: List[Tuple[str, float, str]],
    ) -> Dict[str, Any]:
        """
        Formats the matches into a dictionary row for output.
        
        Args:
            col (str): The original column name.
            stage (str): The stage of matching (e.g., "stage1", "stage2", "stage3").
            detail (str): Method about the matching stage.
            matches (List[Tuple[str, float, str]]): List of matches as tuples of (field, score, source).

        Returns:
            Dict[str, Any]: A row for output to the result table.
        """
        row: Dict[str, Any] = {
            "original_column": col,
            "matched_stage": stage,
            "matched_stage_method": detail
        }

        for i, (field, score, source) in enumerate(matches[:self.top_k],
                                                   start=1):
            row[f"match{i}_field"] = field
            row[f"match{i}_score"] = round(score, 4)
            row[f"match{i}_source"] = source

        return row

    def dict_fuzzy_match(self, col: str, fuzzy_thresh=FUZZY_THRESH):
        """
        Stage 1: Dictionary & Fuzzy Matching
        """
        norm = normalize(col)
        matches = []
        detail = None

        dict_match_fields = self.sources_to_fields.get(norm, [])
        if dict_match_fields:
            # Exact match (after normalization)
            detail = "dict"
            for f in dict_match_fields:
                if normalize(f) == norm:
                    matches = [(f, 1.0, f)]
                    break
                else:
                    src = self.normed_source_to_source.get(norm, f)
                    matches.append((f, 1.0, src))
        else:
            # Fuzzy match (also on normalized terms)
            detail = "fuzzy"
            fuzzy_candidates = process.extract(norm,
                                               self.sources_keys,
                                               scorer=fuzz.token_sort_ratio,
                                               limit=self.top_k)

            best = {}
            for cand, score, _ in fuzzy_candidates:
                if score >= fuzzy_thresh:
                    for std_field in self.sources_to_fields[cand]:
                        src = self.normed_source_to_source.get(cand, std_field)
                        s = score / 100.0
                        if (std_field not in best) or (best[std_field][0] < s):
                            best[std_field] = (s, src)

            matches = [(f, sc, src) for f, (sc, src) in best.items()]

        matches = sorted(matches, key=lambda x: x[1],
                         reverse=True)[:self.top_k]

        return self.format_matches_to_row(col=col,
                                          stage="stage1",
                                          detail=detail,
                                          matches=matches)

    def numeric_field_match(self, col: str) -> Dict[str, Any]:
        """
        Stage2a: Match column based on numeric patterns.
        Aim to map: time and dose
        """
        logger.info(
            f"[Stage2a] Running numeric field match for column '{col}'")
        key_raw = normalize(col)
        candidates = []
        detail = None

        if not self.is_col_numeric(col):
            return {}

        self._ensure_numeric_index()
        if self._numeric_embs is None or len(self.numeric_sources) == 0:
            return {}

        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)

        emb = self._enc(key_clean or key_raw)
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self._numeric_embs)[0]
        top = torch.topk(sims, k=min(self.top_k, len(sims)))

        numeric_scores = []
        for score, idx in zip(top[0], top[1]):
            src_name = self.numeric_sources[int(idx)]
            for f in self.df_num[self.df_num['source'] ==
                                 src_name]['field_name']:
                base = float(score)
                bonus = family_boost(f, family)  # 0 ~ 0.15
                final = base + bonus
                numeric_scores.append((f, final, src_name))

        numeric_scores_sorted = sorted(numeric_scores,
                                       key=lambda x: x[1],
                                       reverse=True)[:self.top_k]
        match1_score = numeric_scores_sorted[0][
            1] if numeric_scores_sorted else 0.0

        logger.info(
            f"[Stage2a Numeric Attempt] Column='{col}' family={family} tags={list(unit_tags)} "
            f"best={match1_score:.3f} top={numeric_scores_sorted[:5]}")

        if match1_score >= NUMERIC_THRESH:
            candidates = numeric_scores_sorted
            detail = "numeric"

        # Deduplicate and take top_k
        best: Dict[str, Tuple[str, float, str]] = {}
        for f, score, src in candidates:
            if f not in best or best[f][1] < score:
                best[f] = (f, score, src)
        formatted_res = list(best.values())
        formatted_res = sorted(formatted_res, key=lambda x: x[1],
                               reverse=True)[:self.top_k]

        return self.format_matches_to_row(col=col,
                                          stage="stage2",
                                          detail=detail,
                                          matches=formatted_res)

    def alias_field_match(self, col: str) -> Dict[str, Any]:
        """
        Stage2b: Match column based on alias patterns.
        Aim to map: alias fields
        """
        logger.info(f"[Stage2b] Running field match for column '{col}'")
        key = normalize(col)
        candidates = []
        detail = None

        emb2 = self._enc(key)
        with torch.no_grad():
            sims2 = util.pytorch_cos_sim(emb2, self.alias_embs)[0]
        top2 = torch.topk(sims2, k=min(self.top_k, len(sims2)))

        alias_scores = []
        for score, idx in zip(top2[0], top2[1]):
            alias = self.sources_keys[int(idx)]
            for f in self.sources_to_fields[alias]:
                alias_scores.append((f, float(score), alias))

        if not alias_scores:
            return {}

        alias_scores.sort(key=lambda x: x[1], reverse=True)
        if alias_scores[0][1] >= FIELD_ALIAS_THRESH:
            detail = "alias"
            candidates = sorted(alias_scores, key=lambda x: x[1], reverse=True)

        # Deduplicate and take top_k
        best: Dict[str, Tuple[str, float, str]] = {}
        for f, score, src in candidates:
            if f not in best or best[f][1] < score:
                best[f] = (f, score, src)
        formatted_res = list(best.values())
        formatted_res = sorted(formatted_res, key=lambda x: x[1],
                               reverse=True)[:self.top_k]

        return self.format_matches_to_row(col=col,
                                          stage="stage2",
                                          detail=detail,
                                          matches=formatted_res)

    def field_value_match(self, col: str) -> Dict[str, Any]:
        """
        Stage3: Value→Field aggregation matching (batched + robust scoring)
        """
        logger.info(f"[Stage3] Running value-field match for column '{col}'")

        detail = "value"

        if not self.value_texts:
            return {}

        # Configure sampling limits
        max_values = 100

        # 1) Unique value sampling
        unique_values = self.unique_values(col, cap=max_values)
        if not unique_values:
            return {}
        value_embs = torch.stack([self._enc(v)
                                  for v in unique_values])  # [M, D]
        dict_embs = torch.stack([self._enc(v)
                                 for v in self.value_texts])  # [N, D]
        device = value_embs.device
        dict_embs = dict_embs.to(device)

        with torch.no_grad():
            sims = util.pytorch_cos_sim(value_embs, dict_embs)  # [M, N]

        # 2) Aggregation of value→field matches
        field_count: Dict[str, int] = {
        }  # Count of "hit unique values" for each field
        field_sum_score: Dict[str, float] = {
        }  # Cumulative best score (for tie-break)
        field_example: Dict[str, Tuple[str, float]] = {
        }  # Field representative value (value, score)
        hit_unique_count = 0  # Denominator: Count of hit unique values

        # Early stopping thresholds
        EARLY_STOP_MIN = 40  # At least this many samples must be processed before considering early stopping
        EARLY_STOP_MARGIN = 0.10  # Leading safety margin

        for i, v in enumerate(unique_values):
            row = sims[i]
            k = min(10, len(row))
            top_scores, top_idx = torch.topk(row, k=k)

            filtered = [(float(s), int(j))
                        for s, j in zip(top_scores, top_idx)
                        if float(s) >= VALUE_DICT_THRESH]
            if not filtered:
                continue

            hit_unique_count += 1

            # Avoid duplicate counting of the same value for the same field: keep the highest score for each field
            per_value_best_for_field: Dict[str, float] = {}
            for s, j in filtered:
                fields = self.value_fields_list[j] or []
                for f in fields:
                    if s > per_value_best_for_field.get(f, 0.0):
                        per_value_best_for_field[f] = s
                        if (f not in field_example) or (s
                                                        > field_example[f][1]):
                            field_example[f] = (v, s)

            # Count and score accumulation
            for f, s in per_value_best_for_field.items():
                field_count[f] = field_count.get(f, 0) + 1
                field_sum_score[f] = field_sum_score.get(f, 0.0) + s

            # Early stopping: sufficient samples and leader proportion significantly exceeds threshold + margin
            if hit_unique_count >= EARLY_STOP_MIN:
                best_prop = 0.0
                if hit_unique_count > 0:
                    best_prop = max(cnt / hit_unique_count
                                    for cnt in field_count.values())
                if best_prop >= (VALUE_PERCENTAGE_THRESH + EARLY_STOP_MARGIN):
                    # Early stopping
                    break

        if hit_unique_count == 0 or not field_count:
            return {}

        # 3) Calculate scores: proportion + tie-break (average score)
        results: List[Tuple[str, float, str, float, int]] = [
        ]  # (field, score, example_val, avg_score, count)
        for f, cnt in field_count.items():
            proportion = cnt / hit_unique_count
            if proportion < VALUE_PERCENTAGE_THRESH:
                continue
            example_val = field_example.get(f, ("", 0.0))[0]
            avg_score = field_sum_score.get(f, 0.0) / max(1, cnt)
            # Main sorting by proportion, then average score, then count
            results.append((f, proportion, example_val, avg_score, cnt))

        if not results:
            return {}

        # Sort: first by proportion, then by average score, then by count
        results.sort(key=lambda x: (x[1], x[3], x[4]), reverse=True)
        results = results[:self.top_k]

        # 4) Output
        # Only return (field, main score, source/example)
        trimmed = [(f, sc, ex) for (f, sc, ex, _avg, _cnt) in results]

        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail=detail,
                                          matches=trimmed)

    def ncit_match(self, col: str) -> Dict[str, Any]:
        unique_values = set(self.unique_values(col, cap=20))

        if not unique_values:
            return {}

        cats = self.nci_client.map_value_to_schema(unique_values)
        if not cats:
            return {}

        den = sum(cats.values())
        if den == 0:
            return {}
        candidates = [(k, v / den, col) for k, v in cats.items()]
        matches = [m for m in candidates if m[1] > VALUE_PERCENTAGE_THRESH]
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:self.top_k]

        has_cancer = any(m[0] == "cancer_type" for m in matches)
        has_disease = any(m[0] == "disease" for m in matches)
        if has_cancer and not has_disease and len(matches) < self.top_k:
            matches.append(("disease", cats["cancer_type"] / den, col))
        if has_cancer and len(matches) < self.top_k:
            matches.append(
                ("cancer_type_details", cats["cancer_type"] / den, col))

        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail="ontology",
                                          matches=matches)

    def run_schema_mapping(self) -> pd.DataFrame:
        """
        Run Stage1, 2 and 3.
        - auto mode: automatically decide whether to run Stage4 based on similarity cutoff
        - manual mode: output Stage3 results for manual review, do not directly enter Stage4
        """

        results = []

        for col in self.df.columns:
            # Check invalid columns first
            is_invalid = check_invalid(self.df, col)
            if is_invalid:
                results.append({
                    "original_column": col,
                    "matched_stage": "invalid",
                    "matched_stage_method": is_invalid
                })
                continue

            # Stage1
            row = self.dict_fuzzy_match(col)
            if row.get("match1_score"):
                results.append(row)
                continue

            # Stage2a
            row = self.numeric_field_match(col)
            if row.get("match1_score"):
                results.append(row)
                continue

            t0 = time.perf_counter()
            if not self.is_col_numeric(col):
                # Stage3a
                row = self.field_value_match(col)
                if row.get("match1_score"):
                    results.append(row)
                    continue

                # Stage3b
                row = self.ncit_match(col)
                if row.get("match1_score"):
                    results.append(row)
                    continue
            t1 = time.perf_counter()
            logger.info(
                f"[Timing] Stage3 for column '{col}' took {t1 - t0:.2f} seconds."
            )

            # Stage2b
            row = self.alias_field_match(col)
            if row.get("match1_score"):
                results.append(row)
                continue

            # stage4 llm, under development

        df_out = pd.DataFrame(results)
        out_file = self.output_file.replace(".csv", f"_{self.mode}.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved Stage1&2&3 ({self.mode}) results to {out_file}")
        return df_out

    def run_stage4_from_manual(self, manual_csv: str) -> pd.DataFrame:
        """
        Load manual Stage3 results, run Stage4 on unmatched/pending columns.
        """
        df_manual = pd.read_csv(manual_csv)
        if "matched_stage" in df_manual.columns and "original_column" in df_manual.columns:
            mask = ((df_manual["matched_stage"] == "stage3_pending")
                    | df_manual["matched_stage"].isna()
                    |
                    (df_manual["matched_stage"].astype(str).str.strip() == ""))
            pending_cols = df_manual.loc[mask, "original_column"]
        elif "original_column" in df_manual.columns:
            logger.warning(
                "[Stage3] 'matched_stage' not found; assuming all 'original_column' need Stage4."
            )
            pending_cols = (df_manual["original_column"].dropna().astype(
                str).unique().tolist())
        else:
            raise ValueError(
                "Expected columns 'matched_stage' and/or 'original_column'. ")

        results = []
        for col in pending_cols:
            row = self.field_value_match(col)
            results.append(row)

        df_out = pd.DataFrame(results)
        out_file = manual_csv.replace(".csv", "_stage4.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved Stage4 results to {out_file}")
        try:
            self.nci_client.save_cache()
        except Exception:
            pass
        return df_out
