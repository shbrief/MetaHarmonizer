import os
from typing import Dict, List, Optional, Sequence, Set, Iterable, Tuple
import pandas as pd

CANCER_TOKENS: Tuple[str, ...] = ("cancer_type", "cancer_subtype", "cancer_type_details")
DISEASE_LABEL: str = "disease"
TREATMENT_PREFIX: str = "treatment_"

_CANCER_RELATED_SET: Set[str] = {*CANCER_TOKENS, DISEASE_LABEL}

# --------------------------- helpers ---------------------------


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


def _build_output_filename(base_name: str,
                           include_methods: Optional[Iterable[str]],
                           exclude_methods: Optional[Iterable[str]],
                           treatment_override: bool) -> str:
    """
    Build output filename with suffixes reflecting active options.

    Examples:
      exclude=["alias_exact"], treatment_override=True
      → "test_manual_ex_alias_exact_treatment_eval.csv"

      no options
      → "test_manual_eval.csv"
    """
    stem = base_name.replace(".csv", "")
    parts = [stem]

    if exclude_methods:
        methods_str = "_".join(sorted(str(m) for m in exclude_methods))
        parts.append(f"ex_{methods_str}")

    if include_methods:
        methods_str = "_".join(sorted(str(m) for m in include_methods))
        parts.append(f"inc_{methods_str}")

    if treatment_override:
        parts.append("treatment")

    parts.append("eval")
    return "_".join(parts) + ".csv"


def _apply_cancer_disease_override(
    df: pd.DataFrame,
    pred_cols: List[str],
) -> pd.DataFrame:
    """
    Cancer/Disease cross-matching rules:
    1. If ref_match is any cancer token (cancer_type, cancer_subtype, cancer_type_details):
       - Accept disease as valid match
       - Accept any other cancer token as valid match
    2. If ref_match is disease:
       - Accept any cancer token as valid match
    """
    def is_cancer_or_disease_row(s: str) -> bool:
        toks = [t.strip().lower() for t in str(s or "").split("|") if t]
        return any(t in _CANCER_RELATED_SET for t in toks)

    def find_earliest_cancer_related_rank(row: pd.Series) -> Optional[int]:
        for i, col in enumerate(pred_cols, start=1):
            val = str(row.get(col, "")).strip().lower()
            if val in _CANCER_RELATED_SET:
                return i
        return None

    unmatched = df["match_level"].fillna(99).astype(int) == 99
    is_cancer_related = df["ref_match"].apply(is_cancer_or_disease_row)
    cancer_related_ranks = df.apply(find_earliest_cancer_related_rank, axis=1)

    override_mask = unmatched & is_cancer_related & cancer_related_ranks.notna()
    df.loc[override_mask, "match_level"] = cancer_related_ranks[override_mask].astype(int)

    return df


def _apply_treatment_override(
    df: pd.DataFrame,
    pred_cols: List[str],
) -> pd.DataFrame:
    """
    Treatment cross-matching rule:
    If ref_match contains any treatment_* token AND any prediction is also
    treatment_* (regardless of subtype), count as match at that rank.
    Only applied to currently unmatched rows (match_level == 99).
    """
    def ref_has_treatment(s: str) -> bool:
        toks = [t.strip().lower() for t in str(s or "").split("|") if t]
        return any(t.startswith(TREATMENT_PREFIX) for t in toks)

    def find_earliest_treatment_rank(row: pd.Series) -> Optional[int]:
        for i, col in enumerate(pred_cols, start=1):
            val = str(row.get(col, "")).strip().lower()
            if val.startswith(TREATMENT_PREFIX):
                return i
        return None

    unmatched = df["match_level"].fillna(99).astype(int) == 99
    ref_is_treatment = df["ref_match"].apply(ref_has_treatment)
    treatment_ranks = df.apply(find_earliest_treatment_rank, axis=1)

    override_mask = unmatched & ref_is_treatment & treatment_ranks.notna()
    df.loc[override_mask, "match_level"] = treatment_ranks[override_mask].astype(int)

    return df


# --------------------- 1) build eval DataFrame ----------------------


def build_eval_df(
    pred_file: str,
    truth_file: str,
    top_k: int = 5,
    apply_cancer_disease_override: bool = True,
    apply_treatment_override: bool = False,
    save_eval: bool = True,
    out_dir: Optional[str] = "data/schema_mapping_eval",
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Create an augmented DataFrame with:
      - ref_match: all valid truths (pipe-joined) for the source
      - match_level: 1..top_k if any truth appears in top-k predictions; 99 otherwise

    Args:
        pred_file: Path to predictions CSV
        truth_file: Path to ground truth CSV
        top_k: Number of top predictions to consider
        apply_cancer_disease_override: Whether to apply cancer/disease override rules
        apply_treatment_override: Whether to treat any treatment_* match as valid
        save_eval: Whether to save the evaluation DataFrame
        out_dir: Directory to save evaluation file

    Returns:
        Tuple of (evaluation DataFrame, path to saved file or None)

    NOTE: This function NEVER filters rows by method.
    """
    pred = pd.read_csv(pred_file)
    truth_raw = pd.read_csv(truth_file)

    if "query" not in pred.columns:
        raise ValueError("pred_file missing column 'query'")
    for col in ("source", "ref_match"):
        if col not in truth_raw.columns:
            raise ValueError(f"truth_file missing column '{col}'")

    truth_clean = truth_raw.dropna(subset=["ref_match"]).copy()
    truth_clean["source"] = truth_clean["source"].astype(str)
    truth_clean["ref_match"] = truth_clean["ref_match"].astype(str)

    truth_grouped = (truth_clean.groupby("source")["ref_match"].apply(
        lambda s: sorted(set(map(str, s)))).reset_index())
    truth_for_merge = truth_grouped.copy()
    truth_for_merge["ref_match"] = truth_for_merge["ref_match"].apply(
        lambda lst: "|".join(lst))

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

    pred_cols: List[str] = _pred_cols(merged, top_k)
    if not pred_cols:
        raise ValueError(f"No prediction columns like 'match1'..'match{top_k}'.")

    truth_map: Dict[str, Set[str]] = {}
    for _, r in truth_for_merge.iterrows():
        src = str(r["source"])
        truth_map[src] = set(_safe_split_pipe(r["ref_match"]))

    match_levels: List[int] = []
    ref_match_lists: List[str] = []

    for _, row in merged.iterrows():
        src = str(row.get("query"))
        truths = truth_map.get(src, set())
        raw = row.get("ref_match")
        parts = _safe_split_pipe(raw)
        ref_match_lists.append("|".join(sorted(parts)) if parts else "")

        rank = 99
        if truths:
            for r, col in enumerate(pred_cols, start=1):
                if str(row.get(col, "")).strip() in truths:
                    rank = r
                    break
        match_levels.append(rank)

    merged["ref_match"] = ref_match_lists

    cols = list(merged.columns)
    query_idx = cols.index("query") if "query" in cols else 0
    if "ref_match" in cols:
        cols.remove("ref_match")
    cols.insert(query_idx + 1, "ref_match")
    merged = merged[cols]

    merged.insert(
        loc=merged.columns.get_loc("ref_match") + 1,
        column="match_level",
        value=match_levels,
    )

    if apply_cancer_disease_override:
        merged = _apply_cancer_disease_override(merged, pred_cols)

    if apply_treatment_override:
        merged = _apply_treatment_override(merged, pred_cols)

    saved_path: Optional[str] = None
    if save_eval:
        out_filename = _build_output_filename(
            os.path.basename(pred_file),
            include_methods=None,
            exclude_methods=None,
            treatment_override=apply_treatment_override,
        )
        out_dir = out_dir or os.path.dirname(os.path.abspath(pred_file))
        os.makedirs(out_dir, exist_ok=True)
        saved_path = os.path.join(out_dir, out_filename)
        merged.to_csv(saved_path, index=False)

    return merged, saved_path


# ------------------ 2) compute accuracy FROM an eval CSV ------------------


def compute_accuracy_from_eval(
    eval_csv: str,
    top_k: int = 5,
    include_methods: Optional[Iterable[str]] = None,
    exclude_methods: Optional[Iterable[str]] = None,
    apply_cancer_disease_override: bool = True,
    apply_treatment_override: bool = False,
) -> Dict[str, float]:
    """
    Compute Top-k accuracy FROM an existing *_eval.csv.

    Args:
        eval_csv: Path to evaluation CSV file
        top_k: Number of top predictions to evaluate
        include_methods: Only compute metrics for these methods
        exclude_methods: Exclude these methods from metrics
        apply_cancer_disease_override: Whether to apply cancer/disease override rules
        apply_treatment_override: Whether to treat any treatment_* match as valid

    Returns:
        Dictionary with accuracy metrics: acc@1, acc@3, acc@5, n_rows
    """
    df = pd.read_csv(eval_csv)
    df = df[df["ref_match"].notna() & (df["ref_match"] != "")]

    if "match_level" not in df.columns:
        raise ValueError("Eval table is missing 'match_level'.")
    if include_methods and exclude_methods:
        raise ValueError("include_methods and exclude_methods are mutually exclusive.")

    pred_cols = _pred_cols(df, top_k)
    if not pred_cols:
        raise ValueError(f"No prediction columns found in {eval_csv}")

    if apply_cancer_disease_override:
        df = _apply_cancer_disease_override(df, pred_cols)

    if apply_treatment_override:
        df = _apply_treatment_override(df, pred_cols)

    if include_methods:
        if "method" not in df.columns:
            raise ValueError("include_methods specified but 'method' column missing.")
        keep = {str(x) for x in include_methods}
        df = df[df["method"].astype(str).isin(keep)].copy()

    if exclude_methods:
        if "method" not in df.columns:
            raise ValueError("exclude_methods specified but 'method' column missing.")
        drop = {str(x) for x in exclude_methods}
        df = df[~df["method"].astype(str).isin(drop)].copy()

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
    apply_cancer_disease_override: bool = True,
    apply_treatment_override: bool = False,
    include_methods: Optional[Iterable[str]] = None,
    exclude_methods: Optional[Iterable[str]] = None,
    save_eval: bool = False,
    out_dir: Optional[str] = "data/schema_mapping_eval",
) -> Dict[str, float]:
    """
    Compute Top-k accuracy for schema mapping predictions.

    Pass either (pred_file + truth_file) or eval_csv.

    Args:
        pred_file: Path to predictions CSV
        truth_file: Path to ground truth CSV
        eval_csv: Path to existing evaluation CSV
        top_k: Number of top predictions to evaluate
        apply_cancer_disease_override: Whether to apply cancer/disease override rules
        apply_treatment_override: Whether to treat any treatment_* match as valid
        include_methods: Only compute metrics for these methods
        exclude_methods: Exclude these methods from metrics
        save_eval: Whether to save evaluation DataFrame
        out_dir: Directory to save evaluation file

    Output filename examples (when save_eval=True):
        No options:                      test_manual_eval.csv
        exclude_methods=["alias_exact"]: test_manual_ex_alias_exact_eval.csv
        include_methods=["value_std"]:   test_manual_inc_value_std_eval.csv
        apply_treatment_override=True:   test_manual_treatment_eval.csv
        exclude + treatment:             test_manual_ex_alias_exact_treatment_eval.csv

    Returns:
        Dictionary with acc@1, acc@3, acc@5, n_rows (and eval_path if saved)
    """
    if include_methods and exclude_methods:
        raise ValueError("include_methods and exclude_methods are mutually exclusive.")

    if eval_csv is None:
        if not (pred_file and truth_file):
            raise ValueError("Provide either eval_csv OR both pred_file and truth_file.")

        # Build full eval df (no filtering, no saving)
        eval_df, _ = build_eval_df(
            pred_file=pred_file,
            truth_file=truth_file,
            top_k=top_k,
            apply_cancer_disease_override=apply_cancer_disease_override,
            apply_treatment_override=apply_treatment_override,
            save_eval=False,
        )

        # Apply method filtering
        df = eval_df[eval_df["ref_match"].notna() & (eval_df["ref_match"] != "")].copy()

        if include_methods:
            if "method" not in df.columns:
                raise ValueError("include_methods specified but 'method' column missing.")
            keep = {str(x) for x in include_methods}
            df = df[df["method"].astype(str).isin(keep)].copy()

        if exclude_methods:
            if "method" not in df.columns:
                raise ValueError("exclude_methods specified but 'method' column missing.")
            drop = {str(x) for x in exclude_methods}
            df = df[~df["method"].astype(str).isin(drop)].copy()

        # Save filtered df with appropriate filename
        saved_path: Optional[str] = None
        if save_eval:
            out_filename = _build_output_filename(
                os.path.basename(pred_file),
                include_methods=include_methods,
                exclude_methods=exclude_methods,
                treatment_override=apply_treatment_override,
            )
            out_dir = out_dir or os.path.dirname(os.path.abspath(pred_file))
            os.makedirs(out_dir, exist_ok=True)
            saved_path = os.path.join(out_dir, out_filename)
            df.to_csv(saved_path, index=False)

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
        return compute_accuracy_from_eval(
            eval_csv=eval_csv,
            top_k=top_k,
            apply_cancer_disease_override=apply_cancer_disease_override,
            apply_treatment_override=apply_treatment_override,
            include_methods=include_methods,
            exclude_methods=exclude_methods,
        )