import os
import time
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache
from src.CustomLogger.custom_logger import CustomLogger
from src.utils.schema_mapper_utils import normalize, is_numeric_column, is_stage_column, extract_valid_value
from src.utils.numeric_match_utils import strip_units_and_tags, detect_numeric_semantic, family_boost
from src.utils.ncit_match_utils import NCIClientSync
from src.utils.value_faiss import ValueFAISSStore
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# === Configuration ===
logger = CustomLogger().custlogger(loglevel='WARNING')
OUTPUT_DIR = "data/schema_mapping_eval"
DICT_PATH = "data/schema/curated_fields_source_latest_with_flags.csv"
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
      original_column | matched_stage | matched_stage_detail |
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
        self.df = pd.read_csv(clinical_data_path, sep=None, engine='python')
        logger.info(
            f"[Load] df_shape={self.df.shape} first_cols={list(self.df.columns[:5])}"
        )
        if self.df.shape[1] == 1 and isinstance(
                self.df.columns[0], str) and (',' in self.df.columns[0]):
            self.df = pd.read_csv(clinical_data_path)
            logger.info(f"[Reload as CSV] df_shape={self.df.shape}")

        self.top_k = top_k
        self.mode = mode

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.basename(clinical_data_path)
        self.output_file = os.path.join(
            OUTPUT_DIR, base.replace('_clinical_data.tsv', '_schema_map.csv'))

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
        df_num = self.df_dict[self.df_dict['is_numeric_field'] == 'yes']
        self.df_num = df_num
        self.numeric_sources = df_num['source'].dropna().unique().tolist()
        self.norm_numeric = [normalize(s) for s in self.numeric_sources]
        self.numeric_embs = self.dict_model.encode(self.norm_numeric,
                                                   convert_to_tensor=True)
        self.col_is_numeric = {
            col: is_numeric_column(self.df, col)
            for col in self.df.columns
        }

        # Load FS VDB for value embeddings
        self.value_store = ValueFAISSStore()

        # Initialize NCI client
        self.nci_client = NCIClientSync()
        self.cat_cols = [
            c for c in self.df.columns
            if not self.col_is_numeric.get(c, False)
        ]

        # ---- Preheat Budget (overridable by environment variables) ----
        PREWARM_ON = False  # Whether to preheat common values
        PREWARM_MAX = 300  # Global max number of values to preheat
        PER_COL_CAP = 80  # Max number of unique values per column
        PREWARM_WORKERS = 4  # Conservative number of threads

        if PREWARM_ON and PREWARM_MAX > 0:
            freq = Counter()
            norm2orig = {}
            # 1) Skip "disease stage" columns to avoid preheating values like M0/Stage II
            cols = [
                c for c in self.cat_cols if is_stage_column(self.df, c) is None
            ]
            # 2) Count the frequency of each value (deduplicated by normalize)
            for c in cols:
                for v in self.unique_values(c, cap=PER_COL_CAP):
                    nv = normalize(v)
                    if nv not in norm2orig:
                        norm2orig[nv] = v
                    freq[nv] += 1
            # 3) Only preheat global Top-K high-frequency values
            selected = [
                norm2orig[nv] for nv, _ in freq.most_common(PREWARM_MAX)
            ]
            if selected:
                with ThreadPoolExecutor(max_workers=PREWARM_WORKERS) as ex:
                    list(ex.map(self.nci_client.map_value_to_schema, selected))

    def _to_01(self, x: float) -> float:
        return max(0.0, min(1.0, (x + 1.0) / 2.0))

    @lru_cache(maxsize=None)
    def _enc(self, text: str):
        return self.dict_model.encode(text, convert_to_tensor=True)

    def unique_values(self, col: str, cap: Optional[int] = None) -> list[str]:
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
            detail (str): Details about the matching stage.
            matches (List[Tuple[str, float, str]]): List of matches as tuples of (field, score, source).

        Returns:
            Dict[str, Any]: A row for output to the result table.
        """
        row: Dict[str, Any] = {
            "original_column": col,
            "matched_stage": stage,
            "matched_stage_detail": detail
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

        if not self.col_is_numeric.get(col, False):
            return {}

        key_clean, unit_tags = strip_units_and_tags(key_raw)
        family = detect_numeric_semantic(key_clean, unit_tags)

        emb = self._enc(key_clean or key_raw)
        with torch.no_grad():
            sims = util.pytorch_cos_sim(emb, self.numeric_embs)[0]
        top = torch.topk(sims, k=min(self.top_k, len(sims)))

        numeric_scores = []
        for score, idx in zip(top[0], top[1]):
            src_name = self.numeric_sources[int(idx)]
            for f in self.df_num[self.df_num['source'] ==
                                 src_name]['field_name']:
                base = float(score)
                base01 = self._to_01(base)  # [-1,1] -> [0,1]
                bonus = family_boost(f, family)  # 0 ~ 0.15
                final = base01 + bonus
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
        if self._to_01(alias_scores[0][1]) >= FIELD_ALIAS_THRESH:
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
        t0 = time.perf_counter()
        detail = "value"

        if (self.value_store.index is None) or (self.value_store.index.ntotal
                                                == 0):
            return {}

        # Configure sampling limits
        max_values = 100
        per_value_fetch_k = min(10, self.value_store.index.ntotal)

        # 1) Unique value sampling
        unique_values = self.unique_values(col, cap=max_values)
        if not unique_values:
            return {}

        # 2) Batch retrieval (one-time encoding + FAISS batched)
        batch_hits = self.value_store.search_many(unique_values,
                                                  top_k=per_value_fetch_k)
        if len(batch_hits) != len(unique_values):
            # Defense: If lengths are inconsistent (due to upstream changes or padding), fall back to zip the shortest
            n = min(len(batch_hits), len(unique_values))
            unique_values = unique_values[:n]
            batch_hits = batch_hits[:n]

        # Aggregation container
        field_count: Dict[str, int] = {
        }  # Count of "hit unique values" for each field
        field_sum_score: Dict[str, float] = {
        }  # Cumulative best score (for tie-break)
        field_example: Dict[str, Tuple[str, float]] = {
        }  # Field representative value (value, score)
        hit_unique_count = 0  # Denominator: Count of hit unique values

        # Early stopping thresholds
        TARGET_PROP = VALUE_PERCENTAGE_THRESH
        EARLY_STOP_MIN = 40  # At least this many samples must be processed before considering early stopping
        EARLY_STOP_MARGIN = 0.10  # Leading safety margin

        for idx, (v, hits) in enumerate(zip(unique_values, batch_hits),
                                        start=1):
            # Filter out low-confidence hits
            hits = [
                h for h in (hits or [])
                if float(h.get("score", 0.0)) >= VALUE_DICT_THRESH
            ]
            if not hits:
                # This unique value has no hits, do not count towards denominator
                continue

            hit_unique_count += 1

            # Avoid duplicate counting of the same value for the same field: keep the highest score for each field
            per_value_best_for_field: Dict[str, float] = {}
            for h in hits:
                score = float(h.get("score", 0.0))
                src_val = h.get("value", v)
                f_list = h.get("fields", []) or []
                for f in f_list:
                    if score > per_value_best_for_field.get(f, 0.0):
                        per_value_best_for_field[f] = score
                        # Update field representative example
                        cur = field_example.get(f)
                        if (cur is None) or (score > cur[1]):
                            field_example[f] = (src_val, score)

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
                if best_prop >= (TARGET_PROP + EARLY_STOP_MARGIN):
                    # Early stopping
                    break

        if hit_unique_count == 0 or not field_count:
            return {}

        # 3) Calculate scores: proportion + tie-break (average score)
        results: List[Tuple[str, float, str, float, int]] = [
        ]  # (field, score, example_val, avg_score, count)
        for f, cnt in field_count.items():
            proportion = cnt / hit_unique_count
            if proportion < TARGET_PROP:
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

        logger.info(
            f"[Stage3 timing] col='{col}' hit_unique={hit_unique_count} elapsed={time.perf_counter()-t0:.3f}s"
        )
        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail=detail,
                                          matches=trimmed)

    def ncit_match(self, col: str) -> Dict[str, Any]:
        unique_values = set(self.unique_values(col, cap=100))

        if not unique_values:
            return {}

        cats = [self.nci_client.map_value_to_schema(v) for v in unique_values]
        valid = [c for c in cats if c != "Unclassified"]
        if not valid:
            return {}

        vote = Counter(valid)
        den = len(valid)
        candidates = [(k, v / den, col) for k, v in vote.items()]
        matches = [m for m in candidates if m[1] > VALUE_PERCENTAGE_THRESH]
        matches.sort(key=lambda x: x[1], reverse=True)
        matches = matches[:self.top_k]
        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail=("ncit" if matches else None),
                                          matches=matches)

    def run_schema_mapping(self) -> pd.DataFrame:
        """
        Run Stage1, 2 and 3.
        - auto mode: automatically decide whether to run Stage4 based on similarity cutoff
        - manual mode: output Stage3 results for manual review, do not directly enter Stage4
        """

        results = []

        for col in self.df.columns:
            # Add a check to skip id columns
            if " ID" in col or " id" in col or " Id" in col:
                results.append({
                    "original_column": col,
                    "matched_stage": "invalid",
                    "matched_stage_detail": "id_column",
                })
                continue

            is_stage_detail = is_stage_column(self.df, col)
            if is_stage_detail is not None:
                results.append({
                    "original_column": col,
                    "matched_stage": "invalid",
                    "matched_stage_detail": is_stage_detail
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

            if not self.col_is_numeric.get(col, False):
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
