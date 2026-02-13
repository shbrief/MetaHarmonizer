"""Integration helpers for connecting FieldSuggester with SchemaMapEngine output."""

from typing import Dict, List, Optional, Union

import pandas as pd

from .field_suggester import FieldSuggester


def suggest_from_schema_mapper(
    schema_mapper_results: Union[pd.DataFrame, str],
    df: pd.DataFrame,
    score_threshold: float = 0.5,
    suggester: Optional[FieldSuggester] = None,
) -> Dict[str, dict]:
    """Extract unmapped columns from SchemaMapEngine output and suggest new fields.

    Parameters
    ----------
    schema_mapper_results : pd.DataFrame or str
        Either a DataFrame or a CSV file path containing SchemaMapEngine
        output.  Expected columns: ``query``, ``match1_score`` (at minimum).
        Columns marked ``"invalid"`` in the ``stage`` column are excluded.
    df : pd.DataFrame
        Original clinical data table (used for value-based NER enrichment).
    score_threshold : float
        Columns whose ``match1_score`` is below this threshold (or missing)
        are considered unmapped and forwarded to FieldSuggester.
    suggester : FieldSuggester or None
        Pre-configured ``FieldSuggester`` instance.  If ``None``, one is
        created with default parameters.

    Returns
    -------
    dict
        Same format as ``FieldSuggester.suggest()``::

            {suggested_field_name: {
                "source_columns": [str, ...],
                "ner_entities": [str, ...],
                "confidence": float,
                "sample_values": {col: [val, ...], ...}
            }}
    """
    # Load results if a path string is given
    if isinstance(schema_mapper_results, str):
        schema_mapper_results = pd.read_csv(schema_mapper_results)

    results_df = schema_mapper_results.copy()

    # Ensure expected columns exist
    if "query" not in results_df.columns:
        raise ValueError(
            "schema_mapper_results must contain a 'query' column."
        )

    # Identify unmapped columns:
    # 1. Columns with no match1_score or score below threshold
    # 2. Exclude columns flagged as invalid
    if "stage" in results_df.columns:
        results_df = results_df[results_df["stage"] != "invalid"]

    if "match1_score" in results_df.columns:
        unmapped_mask = (
            results_df["match1_score"].isna()
            | (results_df["match1_score"] < score_threshold)
        )
    else:
        # No score column → treat all as unmapped
        unmapped_mask = pd.Series(True, index=results_df.index)

    unmapped_columns: List[str] = results_df.loc[unmapped_mask, "query"].tolist()

    # Also include columns from df that are absent from mapper results entirely
    mapped_queries = set(schema_mapper_results["query"].tolist())
    missing_from_results = [
        col for col in df.columns if col not in mapped_queries
    ]
    unmapped_columns.extend(missing_from_results)

    if not unmapped_columns:
        return {}

    if suggester is None:
        suggester = FieldSuggester()

    return suggester.suggest(unmapped_columns, df=df)
