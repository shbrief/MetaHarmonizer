import os
from typing import List, Dict, Tuple, Any
import pandas as pd
import re
from rapidfuzz import process, fuzz
from sentence_transformers import SentenceTransformer, util
import torch
from collections import defaultdict
import src.models.schema_mapper_bert as sm_bert
import src.models.schema_mapper_freq as sm_freq
from src.CustomLogger.custom_logger import CustomLogger
from src.utils.schema_mapper_utils import normalize, is_numeric_column

# === Configuration ===
logger = CustomLogger().custlogger(loglevel='INFO')
OUTPUT_DIR = "data/schema_mapping_eval"
DICT_PATH = "data/curated_fields_source_latest_with_flags.csv"
SCHEMA_MAP_PATH = "data/schema_map_less_nested_dict.pkl"
DICT_MODEL = "all-MiniLM-L6-v2"
STAGE1_FUZZY_THRESH = 90
STAGE2_NUMERIC_THRESH = 0.6
STAGE2_ALIAS_THRESH = 0.5


class SchemaMapEngine:
    """
    Performs schema mapping in three stages and outputs a CSV with columns:
      original_column | matched_stage | matched_stage_detail |
      match1_field | match1_score | match1_source |
      match2_field | match2_score | match2_source |
      match3_field | match3_score | match3_source | ...
    - Stage1: Fuzzy match against dictionary using alias names.
    - Stage2: Match against field names, prioritizing numeric fields.
    - Stage3: If no good matches, fallback to a more complex matcher (BERT or frequency-based).
    - Mode: 'auto' runs Stage3 automatically if Stage2 scores are low,
            'manual' outputs Stage2 results for manual review, does not run Stage3.
    - top_k: Number of top matches to return for each column.
    - Output: Results saved to OUTPUT_DIR with filename based on input clinical data.
    - Requires SentenceTransformer and rapidfuzz for matching.
    - Uses a dictionary of field names and their aliases to perform matching.
    """

    def __init__(self,
                 clinical_data_path: str,
                 matcher_type: str = "Freq",
                 mode: str = "auto",
                 top_k: int = 5):
        self.df = pd.read_csv(clinical_data_path, sep='\t')
        self.schema_map_path = SCHEMA_MAP_PATH
        self.matcher_type = matcher_type
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
        self.dict_model = SentenceTransformer(DICT_MODEL)
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

    def dict_fuzzy_match(self, col: str, fuzzy_thresh=STAGE1_FUZZY_THRESH):

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

        row = {
            "original_column": col,
            "matched_stage": "stage1",
            "matched_stage_detail": detail,
        }
        for i, (f, score, src) in enumerate(matches, start=1):
            row[f"match{i}_field"] = f
            row[f"match{i}_score"] = round(score, 2)
            row[f"match{i}_source"] = src

        return row

    def field_match(self, col: str) -> Dict[str, Any]:
        """ 
        Stage2: Match column using field names.
        2a. If numeric, try numeric match first; if low score, fallback to alias.
        2b. If not numeric, directly use alias.
        """
        key = normalize(col)
        candidates = []
        detail = None

        if self.col_is_numeric.get(col, False):
            # Stage2a: numeric attempt
            emb = self.dict_model.encode(key, convert_to_tensor=True)
            sims = util.pytorch_cos_sim(emb, self.numeric_embs)[0]
            top = torch.topk(sims, k=min(self.top_k, len(sims)))
            numeric_scores = []
            for score, idx in zip(top[0], top[1]):
                src_name = self.numeric_sources[int(idx)]
                for f in self.df_num[self.df_num['source'] ==
                                     src_name]['field_name']:
                    numeric_scores.append((f, float(score), src_name))

            numeric_scores_sorted = sorted(numeric_scores,
                                           key=lambda x: x[1],
                                           reverse=True)[:self.top_k]
            match1_score = numeric_scores_sorted[0][
                1] if numeric_scores_sorted else 0.0

            logger.info(
                f"[Stage2 Numeric Attempt] Column '{col}': match1_score={match1_score:.3f}, "
                f"top_candidates={numeric_scores_sorted}")

            if match1_score >= STAGE2_NUMERIC_THRESH:
                candidates = numeric_scores_sorted
                detail = "numeric"

            else:
                # fallback to alias if numeric score too low
                logger.info(
                    f"[Stage2 Numeric Fallback] Column '{col}' fallback to alias "
                    f"(match1_score={match1_score:.3f} too low)")
                emb2 = self.dict_model.encode(key, convert_to_tensor=True)
                sims2 = util.pytorch_cos_sim(emb2, self.alias_embs)[0]
                top2 = torch.topk(sims2, k=min(self.top_k, len(sims2)))

                alias_scores = []
                for score, idx in zip(top2[0], top2[1]):
                    alias = self.sources_keys[int(idx)]
                    for f in self.sources_to_fields[alias]:
                        alias_scores.append((f, float(score), alias))

                candidates = sorted(alias_scores,
                                    key=lambda x: x[1],
                                    reverse=True)[:self.top_k]
                detail = "alias"
        else:
            # Stage2b: non-numeric → alias only
            emb2 = self.dict_model.encode(key, convert_to_tensor=True)
            sims2 = util.pytorch_cos_sim(emb2, self.alias_embs)[0]
            top2 = torch.topk(sims2, k=min(self.top_k, len(sims2)))
            for score, idx in zip(top2[0], top2[1]):
                alias = self.sources_keys[int(idx)]
                for f in self.sources_to_fields[alias]:
                    candidates.append((f, float(score), alias))
            detail = "alias"

        # Deduplicate and take top_k
        best: Dict[str, Tuple[str, float, str]] = {}
        for f, score, src in candidates:
            if f not in best or best[f][1] < score:
                best[f] = (f, score, src)
        formatted_res = list(best.values())
        formatted_res = sorted(formatted_res, key=lambda x: x[1],
                               reverse=True)[:self.top_k]

        # Build output row
        row: Dict[str, Any] = {
            "original_column": col,
            "matched_stage": "stage2",
            "matched_stage_detail": detail
        }
        for i, (f, score, src) in enumerate(formatted_res, start=1):
            row[f"match{i}_field"] = f
            row[f"match{i}_score"] = round(score, 2)
            row[f"match{i}_source"] = src

        return row

    def field_value_match(self, col: str) -> Dict[str, Any]:
        # Postponed fallback matcher initialization
        if not hasattr(self, "fallback"):
            if self.matcher_type == 'Bert':
                self.fallback = sm_bert.ClinicalDataMatcherBert(
                    self.df, self.top_k)
                detail = 'bert'
            else:
                self.fallback = sm_freq.ClinicalDataMatcherFreq(
                    self.df, self.top_k)
                detail = 'freq'
            self._fallback_detail = detail
        else:
            detail = self._fallback_detail

        if detail == 'bert':
            res = self.fallback.bert_match(col)
        else:
            res = self.fallback.freq_match(col)

        row = {
            "original_column": col,
            "matched_stage": "stage3",
            "matched_stage_detail": detail,
        }

        if not res:
            logger.error(
                f"[Stage3] No results for column '{col}' using {detail}")
            res = []
            row["matched_stage"] = ""
            row["matched_stage_detail"] = ""

        formatted_res = []
        for item in res[:self.top_k]:
            formatted_res.append(item)

        for i, (f, score, src) in enumerate(formatted_res, start=1):
            row[f"match{i}_field"] = f
            row[f"match{i}_score"] = round(score, 2)
            row[f"match{i}_source"] = src

        return row

    def run_schema_mapping(self) -> pd.DataFrame:
        """
        Run Stage1 and Stage2.
        - auto mode: automatically decide whether to run Stage3 based on similarity cutoff
        - manual mode: output Stage2 results for manual review, do not directly enter Stage3
        """

        results = []

        for col in self.df.columns:
            # TODO: Add a check to skip id columns, such as if they were all unique and not null. Need to be careful with this check.

            # Stage1
            row = self.dict_fuzzy_match(col)
            if row.get("match1_score"):
                results.append(row)
                continue

            # Stage2
            row = self.field_match(col)
            if self.mode == "auto":
                if self.col_is_numeric.get(col, False):
                    results.append(row)
                    continue

                top_score = row.get("match1_score")
                if top_score is None:
                    logger.error(
                        f"[SchemaMapEngine] No match1_score produced for column '{col}'. "
                    )
                    # Stage3 fallback if no score from stage 2
                    row = self.field_value_match(col)
                    results.append(row)
                    continue

                is_numeric_match = row["matched_stage_detail"] == "numeric"
                is_alias_match = row["matched_stage_detail"] == "alias"

                if (is_numeric_match and top_score >= STAGE2_NUMERIC_THRESH) or \
                   (is_alias_match and top_score >= STAGE2_ALIAS_THRESH):
                    results.append(row)
                else:
                    # Stage3 fallback
                    row = self.field_value_match(col)
                    results.append(row)

            elif self.mode == "manual":
                # Output Stage2 results for manual review
                row["matched_stage"] = "stage2_pending"
                results.append(row)

        df_out = pd.DataFrame(results)
        out_file = self.output_file.replace(".csv", f"_{self.mode}.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved Stage1&2 ({self.mode}) results to {out_file}")
        return df_out

    def run_stage3_from_manual(self, manual_csv: str) -> pd.DataFrame:
        """
        Load manual Stage2 results, run Stage3 on unmatched/pending columns.
        """
        df_manual = pd.read_csv(manual_csv)
        if "matched_stage" in df_manual.columns and "original_column" in df_manual.columns:
            mask = ((df_manual["matched_stage"] == "stage2_pending")
                    | df_manual["matched_stage"].isna()
                    |
                    (df_manual["matched_stage"].astype(str).str.strip() == ""))
            pending_cols = df_manual.loc[mask, "original_column"]
        elif "original_column" in df_manual.columns:
            logger.warning(
                "[Stage3] 'matched_stage' not found; assuming all 'original_column' need Stage3."
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
        out_file = manual_csv.replace(".csv", "_stage3.csv")
        df_out.to_csv(out_file, index=False)
        logger.info(f"Saved Stage3 results to {out_file}")
        return df_out
