import os
from typing import Dict, List, Optional, Sequence, Set, Iterable, Tuple
import pandas as pd

CANCER_TOKENS: Iterable[str] = ("cancer_type", "cancer_subtype",
                                "cancer_type_details")
DISEASE_LABEL: str = "disease"

# --------------------------- helpers ---------------------------


def _norm(text: Optional[str], do_normalize: bool) -> str:
    if text is None:
        return ""
    s = str(text)
    return s.strip().lower() if do_normalize else s


def _safe_split_pipe(s) -> List[str]:
    if s is None:
        return []
    if pd.isna(s):
        return []
    s_str = str(s).strip()
    if not s_str or s_str.lower() == "nan":
        return []
    return [
        t for t in s_str.split("|") if t and t.strip() and t.lower() != "nan"
    ]


def _pred_cols(df: pd.DataFrame, top_k: int) -> List[str]:
    return [
        f"match{i}" for i in range(1, top_k + 1)
        if f"match{i}" in df.columns
    ]


def _earliest_rank_of_label(row: pd.Series,
                            label: str,
                            pred_cols: List[str],
                            normalize: bool = True) -> Optional[int]:
    """Find the earliest rank (1-based) of label in pred_cols for the given row."""
    tgt = str(label).strip().lower() if normalize else str(label)
    for i, c in enumerate(pred_cols, start=1):
        val = str(row.get(c, "")).strip()
        if normalize:
            val = val.lower()
        if val == tgt:
            return i
    return None


def _apply_cancer_disease_override(
    df: pd.DataFrame,
    pred_cols: List[str],
) -> pd.DataFrame:
    """
    Apply override rules to match_level based on ref_match content:
    
    Cancer/Disease cross-matching rules:
    1. If ref_match is any cancer token (cancer_type, cancer_subtype, cancer_type_details):
       - Accept disease as valid match
       - Accept any other cancer token as valid match
    2. If ref_match is disease:
       - Accept any cancer token as valid match
    
    Returns: updated DataFrame with modified match_level column
    """
    cancer_tokens_set = {t.strip().lower() for t in CANCER_TOKENS}
    disease_norm = DISEASE_LABEL.strip().lower()
    
    # All cancer-related tokens (including disease)
    cancer_related_set = cancer_tokens_set | {disease_norm}

    def is_cancer_row(s: str) -> bool:
        """Check if ref_match contains any cancer token."""
        toks = [t.strip().lower() for t in str(s or "").split("|") if t]
        return any(t in cancer_tokens_set for t in toks)

    def is_disease_row(s: str) -> bool:
        """Check if ref_match contains disease."""
        toks = [t.strip().lower() for t in str(s or "").split("|") if t]
        return disease_norm in toks
    
    def is_cancer_or_disease_row(s: str) -> bool:
        """Check if ref_match contains any cancer-related token (cancer or disease)."""
        toks = [t.strip().lower() for t in str(s or "").split("|") if t]
        return any(t in cancer_related_set for t in toks)

    def find_earliest_cancer_related_rank(row: pd.Series) -> Optional[int]:
        """Find earliest rank of ANY cancer-related token in predictions."""
        for i, col in enumerate(pred_cols, start=1):
            val = str(row.get(col, "")).strip().lower()
            if val in cancer_related_set:
                return i
        return None

    # Only apply override to unmatched rows
    unmatched = df["match_level"].fillna(99).astype(int) == 99

    # Case 1: ref_match is cancer-related AND unmatched
    # → Check if ANY cancer-related token appears in predictions
    is_cancer_related = df["ref_match"].apply(is_cancer_or_disease_row)
    cancer_related_ranks = df.apply(find_earliest_cancer_related_rank, axis=1)
    
    override_mask = unmatched & is_cancer_related & cancer_related_ranks.notna()
    df.loc[override_mask, "match_level"] = cancer_related_ranks[override_mask].astype(int)

    return df


# --------------------- 1) build eval DataFrame ----------------------


def build_eval_df(
    pred_file: str,
    truth_file: str,
    top_k: int = 5,
    normalize_text: bool = False,
    save_eval: bool = True,
    out_dir: Optional[str] = "data/schema_mapping_eval",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Create an augmented DataFrame with:
      - ref_match: all valid truths (pipe-joined) for the source
      - match_level: 1..top_k if any truth appears in top-k predictions; 99 otherwise

    Args:
        pred_file: Path to predictions CSV (must have 'query' column and match1..matchN columns)
        truth_file: Path to ground truth CSV (must have 'source' and 'ref_match' columns)
        top_k: Number of top predictions to consider
        normalize_text: Whether to normalize text for matching (lowercase, strip)
        save_eval: Whether to save the evaluation DataFrame
        out_dir: Directory to save evaluation file

    Returns:
        Tuple of (evaluation DataFrame, path to saved file or None)

    NOTE: This function NEVER filters rows by method.
    The saved *_eval.csv contains ALL prediction rows.
    """
    pred = pd.read_csv(pred_file)
    truth_raw = pd.read_csv(truth_file)

    # Validate required columns
    if "query" not in pred.columns:
        raise ValueError("pred_file missing column 'query'")
    for col in ("source", "ref_match"):
        if col not in truth_raw.columns:
            raise ValueError(f"truth_file missing column '{col}'")

    # Prepare truth data
    truth_clean = truth_raw.dropna(subset=["ref_match"]).copy()
    truth_clean["source"] = truth_clean["source"].astype(str)
    truth_clean["ref_match"] = truth_clean["ref_match"].astype(str)
    
    # Group multiple truths per source
    truth_grouped = (truth_clean.groupby("source")["ref_match"].apply(
        lambda s: sorted(set(map(str, s)))).reset_index())
    truth_for_merge = truth_grouped.copy()
    truth_for_merge["ref_match"] = truth_for_merge["ref_match"].apply(
        lambda lst: "|".join(lst))

    # Merge predictions with truth
    merged = pred.merge(
        truth_for_merge,
        how="left",
        left_on="query",
        right_on="source",
        suffixes=("", "_truth"),
    )
    if merged.empty:
        raise ValueError("No rows after merge (query vs source).")

    if "source" in merged.columns:
        merged = merged.drop(columns=["source"])

    # Get prediction columns
    pred_cols: List[str] = _pred_cols(merged, top_k)
    if not pred_cols:
        raise ValueError(
            f"No prediction columns like 'match1'..'match{top_k}'."
        )

    # Build normalized truth map
    truth_map_norm: Dict[str, Set[str]] = {}
    for _, r in truth_for_merge.iterrows():
        src = str(r["source"])
        truths = _safe_split_pipe(r["ref_match"])
        truth_map_norm[src] = {_norm(v, normalize_text) for v in truths}

    # Calculate match levels
    match_levels: List[int] = []
    ref_match_lists: List[str] = []

    for _, row in merged.iterrows():
        src = str(row.get("query"))
        truths_norm = truth_map_norm.get(src, set())
        raw = row.get("ref_match")
        parts = _safe_split_pipe(raw)
        ref_match_lists.append("|".join(sorted(parts)) if parts else "")

        rank = 99
        if truths_norm:
            for r, col in enumerate(pred_cols, start=1):
                if _norm(row.get(col), normalize_text) in truths_norm:
                    rank = r
                    break
        match_levels.append(rank)

    # Update ref_match column
    merged["ref_match"] = ref_match_lists
    
    # Reorder columns: query, ref_match, match_level, ...
    cols = list(merged.columns)
    query_idx = cols.index("query") if "query" in cols else 0
    if "ref_match" in cols:
        cols.remove("ref_match")
    cols.insert(query_idx + 1, "ref_match")
    merged = merged[cols]
    
    # Insert match_level after ref_match
    merged.insert(
        loc=merged.columns.get_loc("ref_match") + 1,
        column="match_level",
        value=match_levels,
    )

    # Save if requested
    saved_path: Optional[str] = None
    if save_eval:
        base_name = os.path.basename(pred_file).replace(".csv", "_eval.csv")
        out_dir = out_dir or os.path.dirname(os.path.abspath(pred_file))
        os.makedirs(out_dir, exist_ok=True)
        saved_path = os.path.join(out_dir, base_name)
        merged.to_csv(saved_path, index=False)

    return merged, saved_path


# ------------------ 2) compute accuracy FROM an eval CSV (filter only for metrics) ------------------


def compute_accuracy_from_eval(
    eval_csv: str,
    top_k: int = 5,
    include_methods: Optional[Iterable[str]] = None,
    exclude_methods: Optional[Iterable[str]] = None,
    apply_cancer_disease_override: bool = True,
) -> Dict[str, float]:
    """
    Compute Top-k accuracy FROM an existing *_eval.csv.
    
    Args:
        eval_csv: Path to evaluation CSV file
        top_k: Number of top predictions to evaluate
        include_methods: Only compute metrics for these methods (mutually exclusive with exclude_methods)
        exclude_methods: Exclude these methods from metrics (mutually exclusive with include_methods)
        apply_cancer_disease_override: Whether to apply cancer/disease override rules
        
    Returns:
        Dictionary with accuracy metrics: acc@1, acc@3, acc@5, n_rows
        
    NOTE: include/exclude (by method) are applied ONLY for metric computation.
    """
    df = pd.read_csv(eval_csv)
    
    # Only rows with truth
    df = df[df["ref_match"].notna() & (df["ref_match"] != "")]

    if "match_level" not in df.columns:
        raise ValueError("Eval table is missing 'match_level'.")
    if include_methods and exclude_methods:
        raise ValueError(
            "include_methods and exclude_methods are mutually exclusive.")

    # Apply cancer-disease override if requested
    if apply_cancer_disease_override:
        pred_cols = _pred_cols(df, top_k)
        if not pred_cols:
            raise ValueError(f"No prediction columns found in {eval_csv}")
        df = _apply_cancer_disease_override(df, pred_cols)

    # Apply include/exclude ONLY for metrics (does not touch the CSV on disk)
    if include_methods:
        if "method" not in df.columns:
            raise ValueError(
                "include_methods was specified but 'method' is missing in eval CSV."
            )
        keep = {str(x) for x in include_methods}
        df = df[df["method"].astype(str).isin(keep)].copy()
    if exclude_methods:
        if "method" not in df.columns:
            raise ValueError(
                "exclude_methods was specified but 'method' is missing in eval CSV."
            )
        drop = {str(x) for x in exclude_methods}
        df = df[~df["method"].astype(str).isin(drop)].copy()

    # Calculate accuracy for each k
    k_list: Sequence[int] = (1, 3, 5) if top_k == 5 else list(range(1, top_k + 1))
    n = len(df)
    results: Dict[str, float] = {}
    
    if n == 0:
        for k in k_list:
            results[f"acc@{k}"] = 0.0
        results["n_rows"] = 0.0
        return results

    for k in k_list:
        hits = int((df["match_level"].fillna(99).astype(int) <= k).sum())
        results[f"acc@{k}"] = hits / n
    results["n_rows"] = float(n)
    
    return results


# ------------------ 3) wrapper ------------------


def compute_accuracy(
    pred_file: Optional[str] = None,
    truth_file: Optional[str] = None,
    *,
    eval_csv: Optional[str] = None,
    top_k: int = 5,
    normalize_text: bool = False,
    apply_cancer_disease_override: bool = True,
    include_methods: Optional[Iterable[str]] = None,
    exclude_methods: Optional[Iterable[str]] = None,
    save_eval: bool = False,
    out_dir: Optional[str] = "data/schema_mapping_eval",
) -> Dict[str, float]:
    """
    Compute Top-k accuracy for schema mapping predictions.

    You can EITHER:
      - pass (pred_file, truth_file) and it will build eval (optionally save) then compute metrics, OR
      - pass eval_csv directly to compute metrics.

    Args:
        pred_file: Path to predictions CSV
        truth_file: Path to ground truth CSV
        eval_csv: Path to existing evaluation CSV (alternative to pred_file + truth_file)
        top_k: Number of top predictions to evaluate
        normalize_text: Whether to normalize text for matching
        apply_cancer_disease_override: Whether to apply cancer/disease override rules
        include_methods: Only compute metrics for these methods
        exclude_methods: Exclude these methods from metrics
        save_eval: Whether to save evaluation DataFrame when building from pred/truth files
        out_dir: Directory to save evaluation file

    Returns:
        Dictionary with accuracy metrics and optional eval_path

    NOTE: Filtering (include_methods/exclude_methods) applies to metrics only, not to saved eval CSV.
    """
    if eval_csv is None:
        # Build eval from pred_file and truth_file
        if not (pred_file and truth_file):
            raise ValueError(
                "Provide either eval_csv OR both pred_file and truth_file.")
        
        eval_df, saved_path = build_eval_df(
            pred_file=pred_file,
            truth_file=truth_file,
            top_k=top_k,
            normalize_text=normalize_text,
            save_eval=save_eval,
            out_dir=out_dir,
        )
        
        # Compute on the just-built df in-memory
        df = eval_df.copy()
        mask = df["ref_match"].notna() & (df["ref_match"] != "")
        df = df[mask]  # only rows with truth

        if apply_cancer_disease_override:
            pred_cols = _pred_cols(df, top_k)
            if pred_cols:
                df = _apply_cancer_disease_override(df, pred_cols)

        if include_methods and exclude_methods:
            raise ValueError(
                "include_methods and exclude_methods are mutually exclusive.")
        
        if include_methods:
            if "method" not in df.columns:
                raise ValueError(
                    "include_methods was specified but 'method' is missing in eval DF."
                )
            keep = {str(x) for x in include_methods}
            df = df[df["method"].astype(str).isin(keep)].copy()
        
        if exclude_methods:
            if "method" not in df.columns:
                raise ValueError(
                    "exclude_methods was specified but 'method' is missing in eval DF."
                )
            drop = {str(x) for x in exclude_methods}
            df = df[~df["method"].astype(str).isin(drop)].copy()

        # Calculate accuracy
        k_list: Sequence[int] = (1, 3, 5) if top_k == 5 else list(range(1, top_k + 1))
        n = len(df)
        results: Dict[str, float] = {}
        
        if n == 0:
            for k in k_list:
                results[f"acc@{k}"] = 0.0
            results["n_rows"] = 0.0
            if save_eval and saved_path:
                results["eval_path"] = saved_path
            return results

        for k in k_list:
            hits = int((df["match_level"].fillna(99).astype(int) <= k).sum())
            results[f"acc@{k}"] = hits / n
        results["n_rows"] = float(n)
        
        if save_eval and saved_path:
            results["eval_path"] = saved_path
        
        return results

    else:
        # Compute from provided eval CSV
        return compute_accuracy_from_eval(
            eval_csv=eval_csv,
            top_k=top_k,
            apply_cancer_disease_override=apply_cancer_disease_override,
            include_methods=include_methods,
            exclude_methods=exclude_methods,
        )