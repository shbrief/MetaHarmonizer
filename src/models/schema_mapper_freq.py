import pandas as pd
from rapidfuzz import process, fuzz
from src.CustomLogger.custom_logger import CustomLogger
from src.models.schema_mapper import ClinicalDataMatcher
from src.utils.schema_mapper_utils import normalize, extract_valid_value

logger = CustomLogger().custlogger(loglevel='INFO')


class ClinicalDataMatcherFreq(ClinicalDataMatcher):
    """
    A class to match clinical data columns to a schema map based on value frequency.
    """

    def __init__(self, clinical_data: pd.DataFrame, top_k: int):
        """
        Initializes the ClinicalDataMatcherFreq class.

        Args:
            clinical_data (pd.DataFrame): The clinical data DataFrame.
            schema_map_path (str): Path to the schema map file.
        """
        super().__init__(clinical_data, top_k)

        new_map = {}
        for topic, content in self.schema_map.items():
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

    def freq_match(self,
                   column_name: str,
                   fuzzy_threshold: int = 90) -> list[tuple[str, float, str]]:
        """
        Gets the top k value-based matches for a given column using a fuzzy Jaccard similarity.

        Args:
            column_name (str): The name of the column to match.
            fuzzy_threshold (int, optional): The minimum similarity score for fuzzy matching. Defaults to 90.

        Returns:
            list: A list of tuples containing (match_field, match_score, match_source).
        """
        if column_name not in self.clinical_data.columns:
            return []

        s = (self.clinical_data[column_name].dropna().astype(str).apply(
            extract_valid_value))
        col_vals_norm = {normalize(v) for parts in s for v in parts if v}
        if not col_vals_norm:
            return []

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
                                          score_cutoff=fuzzy_threshold)
                if best:
                    logger.debug(
                        f"[Freq Match] Found match for '{sv}' in column '{column_name}': {best}"
                    )
                    intersection += 1
                    del unmatched[best[2]]

            union = len(col_vals_norm) + len(topic_vals) - intersection
            score = intersection / union if union else 0.0
            # score = intersection / len(col_vals_norm)

            if score > 0.0:
                matches.append((topic, score, "freq"))

        matches.sort(key=lambda x: x[1], reverse=True)
        best = {}
        for field, score, src in matches:
            if field not in best or score > best[field][1]:
                best[field] = (field, score, src)

        return list(best.values())[:self.top_k]
