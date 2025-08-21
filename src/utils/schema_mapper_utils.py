import re
from typing import Dict, List, Set, Iterable, Tuple, Any
import pandas as pd
import json
import requests
from collections import Counter
import time

BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"
ncit_dict = {
    "C12219": "body_site",
    "C1909": "treatment_name",
    "C3262": "disease"
}
api_cache = {}


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
    sample = vals.sample(min(len(vals), sample_size), random_state=random_state)
    all_vals = [v for cell in sample for v in extract_valid_value(cell)]
    if not all_vals:
        return False
    converted = pd.to_numeric(pd.Series(all_vals), errors='coerce')
    return converted.notna().sum() / len(converted) >= min_ratio


# --- NCIt API Functions ---
def _search_ncit_code(term: str) -> Tuple[str | None, str | None]:
    if term in api_cache: return api_cache[term]
    params = {'term': term, 'type': 'match', 'limit': 1, 'terminology': 'ncit'}
    url = f"{BASE_URL}/concept/search"
    try:
        time.sleep(0.05)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('concepts'):
            res = data['concepts'][0].get('code'), data['concepts'][0].get(
                'name')
            api_cache[term] = res
            return res
        api_cache[term] = (None, None)
        return None, None
    except requests.exceptions.RequestException:
        api_cache[term] = (None, None)
        return None, None


def _get_parents(ncit_code: str) -> List[Dict[str, str]]:
    if ncit_code in api_cache: return api_cache[ncit_code]
    if not ncit_code: return []
    params = {'include': 'parents'}
    url = f"{BASE_URL}/concept/ncit/{ncit_code}"
    try:
        time.sleep(0.05)
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        parents = data.get('parents', [])
        api_cache[ncit_code] = parents
        return parents
    except requests.exceptions.RequestException:
        api_cache[ncit_code] = []
        return []


def _check_lineage(start_code: str,
                   target_codes: Set[str]) -> Tuple[bool, str | None]:
    visited_codes = set()

    def recursive_check(code: str) -> Tuple[bool, str | None]:
        if code in visited_codes: return False, None
        visited_codes.add(code)
        parents = _get_parents(code)
        if not parents: return False, None
        for parent in parents:
            parent_code = parent.get('code')
            if parent_code in target_codes: return True, parent_code
            found, matched_code = recursive_check(parent_code)
            if found: return True, matched_code
        return False, None

    return recursive_check(start_code)


def map_value_to_schema(value: str) -> str:
    if not value or not isinstance(value, str): return "Unclassified"
    target_codes = set(ncit_dict.keys())
    start_code, _ = _search_ncit_code(value)
    if not start_code: return "Unclassified"
    if start_code in target_codes: return ncit_dict[start_code]
    found, matched_ancestor_code = _check_lineage(start_code, target_codes)
    if found: return ncit_dict[matched_ancestor_code]
    else: return "Unclassified"
