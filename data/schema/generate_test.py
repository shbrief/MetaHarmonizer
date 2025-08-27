import csv, re
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict

INPUT_CSV = "data/cBioPortal_curated_metadata.csv"
OUT_TEST = "test.csv"  # Columns = source; each column stacks original_xxx values vertically
OUT_TRUTH = "truth.csv"  # Deduplicated (source, curated_field) pairs
OUT_MISMATCH = "mismatch_columns.csv"  # AGGREGATED: per base(column), reasons and counts
REQUIRE_CURATED = True  # Keep only rows where curated_xxx is non-empty / non-NA

NA_TOKENS = {"", "NA", "N/A", "NULL"}


def split_tokens_keep_na(cell: str) -> List[str]:
    """
    Split by delimiters ; | <;> | ::, keep NA tokens (case-insensitive), strip whitespace.
    (We keep NA here only for position alignment; NA pairs are filtered when pairing.)
    """
    parts = re.split(r";|<;>|::", str(cell))
    out = []
    for p in parts:
        t = p.strip()
        if not t:  # skip empty
            continue
        out.append(t)
    return out


def uniqueize(names):
    """Make names unique by appending _2, _3, ... when duplicates appear."""
    seen = {}
    out = []
    for n in names:
        base = n
        k = seen.get(base, 0)
        if k == 0 and base not in out:
            out.append(base)
            seen[base] = 1
        else:
            k = max(1, k)
            while True:
                k += 1
                cand = f"{base}_{k}"
                if cand not in out:
                    out.append(cand)
                    seen[base] = k
                    break
    return out


def generate_test_and_truth(input_csv=INPUT_CSV,
                            out_test=OUT_TEST,
                            out_truth=OUT_TRUTH,
                            out_mismatch=OUT_MISMATCH):
    Path(out_test).parent.mkdir(parents=True, exist_ok=True)
    Path(out_truth).parent.mkdir(parents=True, exist_ok=True)
    Path(out_mismatch).parent.mkdir(parents=True, exist_ok=True)

    # ---------- Read header and find paired columns ----------
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []

    pairs = []
    for sc in fields:
        if sc.startswith("curated_") and sc.endswith("_source"):
            base = sc[len("curated_"):-len("_source")]  # xxx
            oc = f"original_{base}"  # original_xxx
            cc = f"curated_{base}"  # curated_xxx (optional)
            if oc in fields:
                pairs.append((sc, oc, cc, base))

    if not pairs:
        # empty fallback
        with open(out_test, "w", encoding="utf-8", newline="") as _:
            pass
        with open(out_truth, "w", encoding="utf-8", newline="") as ft:
            csv.writer(ft).writerow(["source", "curated_field"])
        with open(out_mismatch, "w", encoding="utf-8", newline="") as fm:
            csv.writer(fm).writerow(["base", "reasons", "reason_counts"])
        print("No curated_*_source and original_* pairs found.")
        return

    # ---------- Single pass: collect data ----------
    values_by_source: Dict[str, List[str]] = defaultdict(
        list)  # {raw_source -> [original values]}
    truth_pairs = set()  # {(raw_source, base)}
    # NEW: aggregate problems per base(column)
    problems_by_base: Dict[str, Counter] = defaultdict(Counter)

    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row_idx, row in enumerate(r, start=2):  # start=2 includes header
            for sc, oc, cc, base in pairs:
                cs_val = (row.get(sc) or "").strip()
                ov_val = (row.get(oc) or "").strip()

                # Filter: curated_xxx is empty/NA (optional gate)
                if REQUIRE_CURATED and (cc in fields):
                    cur = (row.get(cc) or "").strip()
                    if cur.upper() in NA_TOKENS:
                        problems_by_base[base]["curated_na"] += 1
                        continue

                # Filter: original missing or NA
                if not ov_val or ov_val.upper() in NA_TOKENS:
                    problems_by_base[base]["original_missing_or_na"] += 1
                    continue

                src_tokens = split_tokens_keep_na(cs_val)
                if not src_tokens:
                    problems_by_base[base]["no_source_tokens"] += 1
                    continue

                # Single source: direct map (original is non-NA here)
                if len(src_tokens) == 1:
                    src = src_tokens[0]
                    if src.upper() in NA_TOKENS:
                        problems_by_base[base]["single_source_is_na"] += 1
                        continue
                    values_by_source[src].append(ov_val)
                    truth_pairs.add((src, base))
                    continue

                # Multi-source: pairwise with original; skip NA pairs
                ov_tokens = split_tokens_keep_na(ov_val)

                if len(ov_tokens) != len(src_tokens):
                    problems_by_base[base]["token_count_mismatch"] += 1
                    continue

                kept_any = False
                for s_tok, o_tok in zip(src_tokens, ov_tokens):
                    if s_tok.upper() in NA_TOKENS or o_tok.upper(
                    ) in NA_TOKENS:
                        # skip pairs with NA on either side
                        continue
                    values_by_source[s_tok].append(o_tok)
                    truth_pairs.add((s_tok, base))
                    kept_any = True

                if not kept_any:
                    problems_by_base[base]["all_pairs_filtered_by_na"] += 1

    # ---------- Write truth.csv (deduplicated mappings) ----------
    with open(out_truth, "w", encoding="utf-8", newline="") as ft:
        w = csv.writer(ft)
        w.writerow(["source", "curated_field"])
        for src, base in sorted(truth_pairs,
                                key=lambda x: (x[0].lower(), x[1].lower())):
            w.writerow([src, base])

    # ---------- Write test.csv (columns = sources; stacked values) ----------
    raw_sources = sorted(values_by_source.keys(), key=lambda x: x.lower())
    col_names = uniqueize(raw_sources)

    max_len = max((len(v) for v in values_by_source.values()), default=0)
    with open(out_test, "w", encoding="utf-8", newline="") as fw:
        w = csv.writer(fw)
        w.writerow(col_names)
        for i in range(max_len):
            row_out = []
            for raw in raw_sources:
                vals = values_by_source[raw]
                row_out.append(vals[i] if i < len(vals) else "")
            w.writerow(row_out)

    # ---------- Write aggregated mismatch (per column/base) ----------
    with open(out_mismatch, "w", encoding="utf-8", newline="") as fm:
        w = csv.writer(fm)
        w.writerow(["base", "reasons", "reason_counts"])
        for base in sorted(problems_by_base.keys(), key=str.lower):
            cnts = problems_by_base[base]
            # join reasons and counts for readability
            reasons = "|".join(sorted(cnts.keys()))
            reason_counts = "|".join(f"{k}:{v}"
                                     for k, v in sorted(cnts.items()))
            w.writerow([base, reasons, reason_counts])

    print(f"✅ test.csv written: cols={len(col_names)} rows={max_len}")
    print(f"✅ truth.csv written: mapping pairs={len(truth_pairs)}")
    print(
        f"📝 {OUT_MISMATCH} written: {len(problems_by_base)} problematic columns summarized."
    )
    if problems_by_base:
        # optional quick glance: top 5 bases by total issues
        summary = sorted(
            ((b, sum(c.values())) for b, c in problems_by_base.items()),
            key=lambda x: x[1],
            reverse=True)[:5]
        print("Top problematic bases (by total issue count):", summary)


if __name__ == "__main__":
    generate_test_and_truth()
