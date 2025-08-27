import os
import re
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
from src.CustomLogger.custom_logger import CustomLogger
from src.utils.schema_mapper_utils import normalize, is_numeric_column, extract_valid_value, map_value_to_schema
from src.utils.numeric_match_utils import strip_units_and_tags, detect_numeric_semantic, family_boost
from src.utils.value_faiss import ValueFAISSStore

# === Configuration ===
logger = CustomLogger().custlogger(loglevel='INFO')
OUTPUT_DIR = "data/schema_mapping_eval"
DICT_PATH = "data/curated_fields_source_latest_with_flags.csv"
FIELD_MODEL = "all-MiniLM-L6-v2"
FUZZY_THRESH = 90
NUMERIC_THRESH = 0.6
FIELD_ALIAS_THRESH = 0.5
VALUE_DICT_THRESH = 0.8
VALUE_PERCENTAGE_THRESH = 0.5
SCHEMA_MAP_PATH = 'data/schema_map_generated.pkl'
DATA_DICT_PATH = 'data/cBioPortalData_data_dictionary.xlsx'
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
    - Stage1: Fuzzy match against dictionary using alias names.
    - Stage2: Numeric field matching based on header names and value types.
    - Stage3: Alias match using SentenceTransformer embeddings.
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
        self.df = pd.read_csv(clinical_data_path, sep='\t')
        self.schema_map_path = SCHEMA_MAP_PATH
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

        # Load schema map for non-numeric frequency-based matching
        with open(SCHEMA_MAP_PATH, 'rb') as f:
            schema_map = pickle.load(f)
        new_map = {}
        for topic, content in schema_map.items():
            if isinstance(content, dict):
                if "unique_col_values" in content and "__NUMERIC__" in content[
                        "unique_col_values"]:
                    continue
                vals = content.get("unique_col_values", [])
            else:
                vals = content or []
            values_norm = {
                normalize(str(v))
                for v in vals if v is not None and str(v).strip() != ""
            }
            new_map[topic] = {"values_norm": values_norm}
        self.schema_map = new_map

        # Load FS VDB for value embeddings
        self.value_store = ValueFAISSStore()

    def _to_01(self, x: float) -> float:
        return max(0.0, min(1.0, (x + 1.0) / 2.0))

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

    def value_match(self, col: str) -> Dict[str, Any]:
        # previous frequency match, need adjustment
        if col not in self.df.columns:
            return {}

        key = normalize(col)
        candidates = []
        detail = None

        s = (self.df[col].dropna().astype(str).apply(extract_valid_value))
        col_vals_norm = {normalize(v) for parts in s for v in parts if v}
        col_vals_norm = {v
                         for v in col_vals_norm
                         if v not in NOISE_VALUES}  # Filter out noisy values
        if not col_vals_norm:
            return {}

        # Continue with frequency matching logic
        matches = []
        for topic, info in self.schema_map.items():
            topic_vals = info["values_norm"]
            if not topic_vals:
                continue

            intersection = 0
            unmatched = list(col_vals_norm)

            for sv in topic_vals:
                best = process.extractOne(sv,
                                          unmatched,
                                          scorer=fuzz.token_sort_ratio,
                                          score_cutoff=FUZZY_THRESH)
                if best and 0 <= best[2] < len(unmatched):
                    logger.debug(
                        f"[Freq Match] Found match for '{sv}' in column '{col}': {best}"
                    )
                    intersection += 1
                    del unmatched[best[2]]

            union = len(col_vals_norm) + len(topic_vals) - intersection
            score = intersection / union if union else 0.0

            if score > 0.0:
                matches.append((topic, score, "freq"))

        matches.sort(key=lambda x: x[1], reverse=True)
        best = {}
        for field, score, src in matches:
            if field not in best or score > best[field][1]:
                best[field] = (field, score, src)

        candidates = list(best.values())
        detail = "freq"

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

        emb = self.dict_model.encode(key_clean or key_raw,
                                     convert_to_tensor=True)
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

        emb2 = self.dict_model.encode(key, convert_to_tensor=True)
        sims2 = util.pytorch_cos_sim(emb2, self.alias_embs)[0]
        top2 = torch.topk(sims2, k=min(self.top_k, len(sims2)))

        alias_scores = []
        for score, idx in zip(top2[0], top2[1]):
            alias = self.sources_keys[int(idx)]
            for f in self.sources_to_fields[alias]:
                alias_scores.append((f, float(score), alias))

        if not alias_scores:
            return {}

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
        Stage3: Value→Field aggregation matching
        - Extract unique value of the column
        - Retrieve each value using value_store with threshold filtering
        - Aggregate by field: count how many unique values hit that field
        - Sort by "hit proportion" as score; source provides representative hit value
        """
        logger.info(f"[Stage3] Running value-field match for column '{col}'")
        detail = "value"

        if (self.value_store.index is None) or (self.value_store.index.ntotal
                                                == 0):
            return {}

        # 1) Extract unique values from the column
        series_of_lists = self.df[col].dropna().astype(str).apply(
            extract_valid_value)
        all_values = [item for sublist in series_of_lists for item in sublist]
        # Filter out empty strings and noise values
        unique_values = []
        seen_norm = set()
        for v in all_values:
            nv = normalize(v)
            if (not nv) or (nv in NOISE_VALUES):
                continue
            if nv not in seen_norm:
                unique_values.append(v)
                seen_norm.add(nv)

        total_unique = len(unique_values)
        if total_unique == 0:
            return {}

        # 2) Query value_store for each unique value and aggregate by field
        #    - count: How many unique values hit that field
        #    - sum_score: The best score of that field across all values
        #    - example: The best example value for that field
        field_count: Dict[str,
                          int] = {}  # how many unique values hit this field
        field_sum_score: Dict[str,
                              float] = {}  # sum of best scores across values
        field_example: Dict[str, Tuple[str, float]] = {
        }  # (best_source_value, best_score) per field

        per_value_fetch_k = min(
            10, self.value_store.index.ntotal)  # limit per-value candidates

        for v in unique_values:
            hits = self.value_store.search(v, top_k=per_value_fetch_k)
            # Keep only confident hits
            hits = [
                h for h in hits
                if float(h.get("score", 0.0)) >= VALUE_DICT_THRESH
            ]
            if not hits:
                continue

            # For THIS value: allow multi-field hits,
            # but keep only the best score per (value, field) to avoid overcounting.
            per_value_best_for_field: Dict[str, float] = {}

            for h in hits:
                score = float(h.get("score", 0.0))
                src_val = h.get("value", v)
                for f in h.get("fields", []):
                    # keep the best score for this (value, field)
                    if score > per_value_best_for_field.get(f, 0.0):
                        per_value_best_for_field[f] = score
                        # maintain global representative example per field
                        cur = field_example.get(f)
                        if (cur is None) or (score > cur[1]):
                            field_example[f] = (src_val, score)

            # Aggregate this value's contribution to each hit field (count + score)
            for f, s in per_value_best_for_field.items():
                field_count[f] = field_count.get(f, 0) + 1
                field_sum_score[f] = field_sum_score.get(f, 0.0) + s

        if not field_count:
            return {}

        # 3) Build results: score = proportion of unique values that hit this field.
        results: List[Tuple[str, float, str]] = []
        for f, cnt in field_count.items():
            proportion = cnt / total_unique  # primary score
            example_val = field_example.get(
                f, ("", 0.0))[0]  # representative source value
            results.append((f, proportion, example_val))

        # Require a minimal consensus (avoid single-value false positives)
        results = [r for r in results if r[1] >= VALUE_PERCENTAGE_THRESH]
        if not results:
            return {}

        # Sort by proportion (optionally, tie-break by avg score or count if needed)
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:self.top_k]

        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail=detail,
                                          matches=results)

    def ncit_match(self, col: str) -> Dict[str, Any]:
        logger.info(f"[Stage3] Running NCIT match for column '{col}'")

        # 1. Extract unique values
        series_of_lists = self.df[col].dropna().apply(extract_valid_value)
        all_values = [item for sublist in series_of_lists for item in sublist]
        unique_values = set(all_values)

        if not unique_values:
            logger.info(f"INFO: No valid values found in column '{col}'")
            return {}

        logger.info(
            f"INFO: Found {len(unique_values)} unique values for analysis, e.g.: {list(unique_values)[:5]}"
        )

        # 2. Run all unique values through our NCIt classification logic
        classifications = [map_value_to_schema(v) for v in unique_values]

        # 3. Vote on classifications
        valid_classifications = [
            c for c in classifications if c != "Unclassified"
        ]
        if not valid_classifications:
            logger.info(
                f"INFO: All unique values are unclassifiable in the Schema.")
            return {}

        den = len(valid_classifications)
        vote_count = Counter(valid_classifications)
        logger.info(f"INFO: Content analysis voting results: {vote_count}")

        candidates = []
        for category, count in vote_count.items():
            score = count / den if den else 0.0  # Score = value count / total unique values
            candidates.append((category, score, col))

        # 4. Filter by threshold and sort
        matches = [m for m in candidates if m[1] > VALUE_PERCENTAGE_THRESH]
        matches = sorted(matches, key=lambda x: x[1],
                         reverse=True)[:self.top_k]

        detail = "ncit" if matches else None

        # 5. Return results in the desired format
        return self.format_matches_to_row(col=col,
                                          stage="stage3",
                                          detail=detail,
                                          matches=matches)

    def run_schema_mapping(self) -> pd.DataFrame:
        """
        Run Stage1, 2 and 3.
        - auto mode: automatically decide whether to run Stage4 based on similarity cutoff
        - manual mode: output Stage3 results for manual review, do not directly enter Stage4
        """

        results = []

        for col in self.df.columns:
            # Add a check to skip id columns, such as if they were all unique and not null. Need to be careful with this check.
            if " ID" in col or " id" in col or " Id" in col:
                results.append({
                    "original_column": col,
                    "matched_stage": "",
                    "matched_stage_detail": "<ID Column>",
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

            # Stage2b
            row = self.alias_field_match(col)
            if row.get("match1_score"):
                results.append(row)
                continue

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
        return df_out
