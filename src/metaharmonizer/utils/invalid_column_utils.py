import re
import pandas as pd
from rapidfuzz import fuzz


def is_stage_column(df: pd.DataFrame, col: str) -> bool:
    """
    Determine if a column is likely to contain cancer staging (AJCC/TNM) information.
    Compatible with: 'Stage IIIC', 'IIIA', 'III', '0', 'pT2b', 'cN1', 'M0', 'TX/NX/MX', etc.
    """

    # From column name
    name = str(col).lower().strip()
    stage_keywords = [
        "stage", "ajcc", "tnm", "cstage", "pstage", "figo", "ann arbor"
    ]
    if any(x in name for x in stage_keywords):
        return "stage_column"

    # From cell values (sample up to 50 non-null cells)
    sample_size = 50
    # 1) 'Stage ...' + Roman/Numeric + optional a/b/c
    stage_group_re = re.compile(
        r'\bstage\s*(?:group\s*)?[:\-]?\s*(?:0|[1-4]|i{1,3}v?)\s*[abc]?\b',
        re.I)

    # 2) TNM fragments (supporting c/p prefixes; includes Tis, TX/NX/MX, T0-4[a-e], N0-3[a-c], M0/1[a-c])
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
        if tnm_re.match(s_compact):
            tnm_hits += 1

    stage_group_ratio = stage_group_hits / n
    tnm_ratio = tnm_hits / n

    if stage_group_ratio >= 0.8 or tnm_ratio >= 0.8:
        return True

    return False


def is_id_column(col: str) -> bool:
    """
    Determine if a column is likely to contain identifiers (IDs).
    Combines explicit keyword list with flexible pattern matching.
    """

    def norm(s: str) -> str:
        """Removes separators and lowercases."""
        return re.sub(r"[._\-\s]", "", s.lower())

    exact_keywords = {
        'id', 'identifier', 'code', 'mrn', 'uid', 'uuid', 'guid', 'accession'
    }

    id_entities = [
        'patient', 'study', 'sample', 'subject', 'case', 'participant',
        'enrollment', 'specimen', 'trial', 'record', 'medical',
        'medicalrecord', 'accession'
    ]

    col_lower = col.lower().strip()

    # exact match
    if col_lower in exact_keywords:
        return True

    # entity + id / number
    norm_col = norm(col)

    contains_entity = any(norm_col.startswith(e) for e in id_entities)
    contains_id_suffix = any(norm_col.endswith(s) for s in exact_keywords)

    if contains_entity and contains_id_suffix:
        return True

    # general suffix: ends with separator + id
    if re.search(r'(^|[\s_\-\.])id$', col_lower):
        return True

    return False


def is_count_column(col: str) -> bool:
    """
    Determine if a column is likely to contain count data, e.g. sample count, mutation count.
    """
    name = str(col).lower()
    candidates = ["sample count", "mutation count"]

    for cand in candidates:
        if fuzz.partial_ratio(name, cand) >= 95:
            return True
    return False


def check_invalid(df: pd.DataFrame, col: str) -> bool:
    """
    Check if any invalid columns are present in the DataFrame.

    Args:
        df: The DataFrame to check.
        col: The column name to check.
    """
    if is_id_column(col):
        return "id_column"

    if is_count_column(col):
        return "count_column"

    if is_stage_column(df, col):
        return "stage_column"

    return None
