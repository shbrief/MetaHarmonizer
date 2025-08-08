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
    sample = vals.sample(min(len(vals), sample_size), random_state=random_state)
    all_vals = [v for cell in sample for v in extract_valid_value(cell)]
    if not all_vals:
        return False
    converted = pd.to_numeric(pd.Series(all_vals), errors='coerce')
    return converted.notna().sum() / len(converted) >= min_ratio
