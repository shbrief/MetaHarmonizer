import re
import functools
from typing import Dict, List
import pandas as pd

from metaharmonizer.custom_logger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]", " ",
                                      str(text))).lower().strip()


@functools.lru_cache(maxsize=1)
def _load_abbrev_map() -> Dict[str, str]:
    """Load the closed clinical abbreviation dict once (abbrev -> expansion).

    Loaded lazily and cached. On a missing / unreadable / empty file the feature
    degrades to a no-op (identity map) but ALWAYS logs a WARNING first — it never
    fails silently.
    """
    from metaharmonizer.models.schema_mapper.config import CLINICAL_ABBREV_PATH
    if not CLINICAL_ABBREV_PATH:
        # None (file missing at config resolve time) or "" (explicitly disabled).
        logger.warning(
            "[clinical_abbrev] Abbreviation dict unavailable (missing file or "
            "disabled); header abbreviation expansion is DISABLED."
        )
        return {}

    try:
        df = pd.read_csv(CLINICAL_ABBREV_PATH, dtype=str).fillna("")
    except FileNotFoundError:
        logger.warning(
            f"[clinical_abbrev] Clinical abbreviation dict not found at "
            f"{CLINICAL_ABBREV_PATH}; header abbreviation expansion is DISABLED. "
            f"Check the bundled file / packaging (pyproject package-data)."
        )
        return {}
    except (OSError, UnicodeDecodeError) as e:
        logger.warning(
            f"[clinical_abbrev] Could not open clinical abbreviation dict "
            f"{CLINICAL_ABBREV_PATH}: {e!r}; expansion DISABLED."
        )
        return {}
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        logger.warning(
            f"[clinical_abbrev] Could not read clinical abbreviation dict "
            f"{CLINICAL_ABBREV_PATH}: {e!r}; expansion DISABLED."
        )
        return {}

    if "abbrev" not in df.columns or "expansion" not in df.columns:
        logger.warning(
            f"[clinical_abbrev] Clinical abbreviation dict {CLINICAL_ABBREV_PATH} "
            f"is missing 'abbrev'/'expansion' columns; expansion DISABLED."
        )
        return {}

    amap = {
        normalize(a): normalize(e)
        for a, e in zip(df["abbrev"], df["expansion"])
        if normalize(a) and normalize(e)
    }
    if not amap:
        logger.warning(
            f"[clinical_abbrev] Clinical abbreviation dict {CLINICAL_ABBREV_PATH} "
            f"loaded 0 usable rows; expansion DISABLED."
        )
    return amap


def expand_abbreviations(text: str) -> str:
    """Token-wise closed-set clinical abbreviation expansion (Issue #87).

    Expects already-normalized text; unknown tokens pass through unchanged.
    """
    amap = _load_abbrev_map()
    if not amap:
        return text
    out: List[str] = []
    for tok in text.split():
        out.extend(amap.get(tok, tok).split())
    return " ".join(out)


def header_variants(text: str) -> List[str]:
    """Query variants a source header is matched on (Issue #87): the plain
    normalized form and, when different, the abbreviation-expanded form.

    Ordered RAW-FIRST so downstream "keep max score, strict >" merges keep the
    un-expanded form on ties — the expansion only wins when it scores strictly
    higher.

    Safety scope: a field's score can only *rise* vs raw-only, so an exact raw
    match (e.g. gene symbols like `her2` at 1.0) is never lost, and an over-eager
    expansion is discarded unless it genuinely scores higher. It does NOT
    guarantee ranking is never worse: for fuzzy / embedding matches an expansion
    may push a *different* (possibly wrong) field above the correct one, or trip
    an earlier cascade stage's early-stop. So it strongly mitigates — but does not
    fully eliminate — mis-expansion of ambiguous tokens; a clean dictionary is the
    real defense.

    Note: singular/plural is intentionally NOT normalized. Many header tokens
    legitimately end in 's', so stripping is error-prone; and a true plural that
    misses exact/fuzzy is still caught by the stage-3 embedding matcher, which is
    largely number-insensitive."""
    raw = normalize(text)
    expanded = expand_abbreviations(raw)
    return [raw] if raw == expanded else [raw, expanded]


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
