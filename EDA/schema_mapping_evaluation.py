import os
from typing import Dict, List, Optional, Sequence, Set
import pandas as pd


def sm_evaluate(
    pred_file: str,
    truth_file: str,
    top_k: int = 5,
    normalize_text: bool = False,
    out_path: Optional[str] = "data/schema_mapping_eval",
) -> Dict[str, float]:
    """
    Evaluate Top-k accuracy allowing multiple valid truths per source.
    Appends two columns to predictions:
      - 'curated_field'  : pipe-joined list of all valid truths for this source
      - 'matched_rank'   : 1..top_k if any truth is hit; 99 otherwise
    """

    # If top_k==5, report Acc@1/3/5; else report Acc@1..Acc@top_k
    k_list: Sequence[int] = (1, 3,
                             5) if top_k == 5 else list(range(1, top_k + 1))

    def _norm(x: Optional[str]) -> str:
        if x is None:
            return ""
        s = str(x)
        return s.strip().lower() if normalize_text else s

    # ---- Load predictions & truth ----
    pred = pd.read_csv(pred_file)
    truth_raw = pd.read_csv(truth_file)

    # ---- Build multi-truth mapping: source -> set(curated_fields) ----
    # Drop NA curated_field, group by source, collect unique curated_field values
    truth_grouped = (truth_raw.dropna(
        subset=["curated_field"]).groupby("source")["curated_field"].apply(
            lambda s: sorted(set(map(str, s)))).reset_index())
    # For merging (one row per source)
    truth_for_merge = truth_grouped.copy()
    truth_for_merge["curated_field"] = truth_for_merge["curated_field"].apply(
        lambda lst: "|".join(lst))

    # ---- Left-join truth onto predictions (one row per pred) ----
    merged = pred.merge(truth_for_merge,
                        how="left",
                        left_on="original_column",
                        right_on="source",
                        suffixes=("", "_truth"))
    if merged.empty:
        raise ValueError(
            "No rows after merge. Check keys: original_column vs source.")

    # Determine available prediction columns up to top_k
    pred_cols: List[str] = [
        f"match{i}_field" for i in range(1, top_k + 1)
        if f"match{i}_field" in merged.columns
    ]
    if not pred_cols:
        raise ValueError(
            f"No prediction columns like 'match1_field'..'match{top_k}_field' found."
        )

    # ---- Compute matched_rank with multi-truth ----
    matched_ranks: List[int] = []
    curated_lists: List[str] = []  # pipe-joined truths for output

    # Build a dict for quick lookup: source -> set of normalized truths
    truth_map_norm: Dict[str, Set[str]] = {
        str(row["source"]):
        {_norm(v)
         for v in row["curated_field"].split("|")}
        for _, row in truth_for_merge.iterrows()
    }

    for _, row in merged.iterrows():
        src = str(row.get("original_column"))
        truths_norm = truth_map_norm.get(src, set())
        # Keep a pretty string column for output (may be empty if no truth)
        curated_lists.append("|".join(
            sorted(
                {t
                 for t in (row.get("curated_field") or "").split("|") if t})))

        rank = 99
        if truths_norm:
            for r, col in enumerate(pred_cols, start=1):
                pv = row.get(col)
                if _norm(pv) in truths_norm:
                    rank = r
                    break
        matched_ranks.append(rank)

    # Insert curated_field and matched_rank right after original_column
    # Ensure curated_field column reflects the pipe-joined list we created
    merged["curated_field"] = curated_lists
    if "curated_field" in merged.columns:
        cols = list(merged.columns)
        try:
            oc_idx = cols.index("original_column")
        except ValueError:
            oc_idx = 0
        cols.remove("curated_field")
        cols.insert(oc_idx + 1, "curated_field")
        merged = merged[cols]
    # Insert matched_rank after both original_column and curated_field
    MATCHED_RANK_INSERT_OFFSET = 2  # Insert after original_column and curated_field
    merged.insert(loc=merged.columns.get_loc("original_column") + MATCHED_RANK_INSERT_OFFSET,
                  column="matched_rank",
                  value=matched_ranks)

    # ---- Compute Acc@k (denominator = all rows, as per your criteria) ----
    n_total = len(merged)
    results: Dict[str, float] = {}
    for k in k_list:
        hits = int((merged["matched_rank"] <= k).sum())
        results[f"acc@{k}"] = hits / n_total if n_total else 0.0

    results.update({
        "n_total_pred_rows": float(n_total),
    })

    # ---- Save one augmented CSV ----
    base_name = os.path.basename(pred_file).replace(".csv", "_eval.csv")
    out_csv = os.path.join(out_path,
                           base_name) if out_path else pred_file.replace(
                               ".csv", "_eval.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    merged.to_csv(out_csv, index=False)

    # Optional print
    print(f"Saved augmented predictions -> {out_csv}")
    for k in k_list:
        print(f"Acc@{k}: {results[f'acc@{k}']:.2%}")

    return results
