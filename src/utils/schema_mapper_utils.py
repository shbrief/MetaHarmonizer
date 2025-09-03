import re
from typing import List
import pandas as pd


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]", " ",
                                      str(text))).lower().strip()


def extract_valid_value(cell: str) -> List[str]:
    """
    Split a cell on ; <;> :: and keep non-empty, non-'NA' parts.
    """
    parts = re.split(r";|<;>|::", str(cell))
    return [
        p.strip() for p in parts if p.strip().upper() != 'NA' and p.strip()
    ]


def is_numeric_column(df: pd.DataFrame,
                      col: str,
                      min_ratio: float = 0.9,
                      sample_size: int = 1000,
                      random_state: int = None) -> bool:
    """
    Sample up to sample_size non-null cells, extract sub-values,
    convert to numeric, and require at least min_ratio valid numbers.

    Args:
        df: The DataFrame containing the column.
        col: The column name to check.
        min_ratio: Minimum ratio of valid numbers required.
        sample_size: Maximum number of cells to sample.
        random_state: Seed for random sampling (default None for true randomness).
    """
    vals = df[col].dropna().astype(str)
    if vals.empty:
        return False
    sample = vals.sample(min(len(vals), sample_size),
                         random_state=random_state)
    all_vals = [v for cell in sample for v in extract_valid_value(cell)]
    if not all_vals:
        return False
    converted = pd.to_numeric(pd.Series(all_vals), errors='coerce')
    return converted.notna().sum() / len(converted) >= min_ratio


def is_stage_column(df: pd.DataFrame, col: str) -> bool:
    """
    Determine if a column is likely to contain cancer staging (AJCC/TNM) information.
    Compatible with: 'Stage IIIC', 'IIIA', 'III', '0', 'pT2b', 'cN1', 'M0', 'TX/NX/MX', etc.
    """
    # Exclude any other grading possibilities: grade, who, nyha, asa, clavien, trial, phase, class, type, factor, complex, hla, mhc, collagen, version, section
    sample_size = 200

    # 1) Name contains 'disease stage'
    name = str(col)
    name_has_stage = re.search(
        r'\b(ajcc|pathologic|clinical|cstage|pstage)?\s*stage\b',
        name,
        flags=re.I) is not None

    # 2) 'Stage ...' + Roman/Numeric + optional a/b/c
    stage_group_re = re.compile(
        r'\bstage\s*(?:group\s*)?[:\-]?\s*(?:0|[1-4]|i{1,3}v?)\s*[abc]?\b',
        re.I)

    # 3) TNM fragments (supporting c/p prefixes; includes Tis, TX/NX/MX, T0-4[a-e], N0-3[a-c], M0/1[a-c])
    tnm_re = re.compile(
        r'^(?:[cp])?(?:t(?:is|x|[0-4][a-e]?)|n(?:x|[0-3][abc]?)|m(?:x|[01][abc]?))$',
        re.I)

    vals = df[col].dropna().astype(str)
    if vals.empty:
        return None
    if len(vals) > sample_size:
        vals = vals.sample(n=sample_size, random_state=0)

    stage_group_hits = 0
    tnm_hits = 0
    n = len(vals)
    for v in vals:
        s = v.strip()
        s_compact = re.sub(r'\s+', '', s)
        if stage_group_re.search(s):
            stage_group_hits += 1
        if name_has_stage and tnm_re.match(s_compact):
            tnm_hits += 1

    stage_group_ratio = stage_group_hits / n
    tnm_ratio = tnm_hits / n

    if name_has_stage:
        return 'header_stage'
    if name_has_stage and tnm_ratio >= 0.8:
        return 'stage+tnm'
    if stage_group_ratio >= 0.8:
        return 'stage_group'

    return None
