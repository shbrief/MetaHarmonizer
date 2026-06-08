"""LLM-based alias dictionary generation for SchemaMapper.

Expands a set of standardized field names into a larger set of plausible
real-world column-name aliases by issuing multi-pass prompts to an LLM,
and returns a ``pandas.DataFrame`` matching the alias-dictionary schema
consumed by :class:`SchemaMapEngine` via
:meth:`DictLoader.load_alias_dict`.

The provider is auto-detected from the model identifier prefix
(``claude-*`` -> Anthropic, ``gemini-*`` -> Gemini). The prompt set is
selected by ``schema_domain``; extend
:data:`alias_dict_prompts.ALIAS_DICT_PROMPTS` to add new domains.
"""
from __future__ import annotations

import io
import re
import sys
from typing import Iterable, Optional, Union

import pandas as pd

from metaharmonizer.utils.llm_client import (
    call_llm,
    detect_provider,
    resolve_api_key,
)
from .alias_dict_prompts import (
    ALIAS_DICT_PROMPTS,
    OUTPUT_FORMAT_INSTRUCTIONS,
)


ALIAS_DICT_COLUMNS = ["field_name", "source", "is_numeric_field"]


def generate_alias_dict(
    target_fields: Union[Iterable[str], pd.DataFrame],
    model: str,
    api_key: Optional[str] = None,
    schema_domain: str = "cancer_genomics",
    is_numeric_field: Optional[Iterable[bool]] = None,
    max_tokens: int = 16000,
    batch_size: int = 80,
    dedupe: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate an alias dictionary for SchemaMapper using an LLM.

    Args:
        target_fields: Standardized field names. Either an iterable of strings,
            or a DataFrame with a ``field_name`` column (and optionally an
            ``is_numeric_field`` column).
        model: LLM model identifier. Provider auto-detected from the prefix:
            ``claude-*`` for Anthropic, ``gemini-*`` for Google Gemini.
        api_key: API key for the resolved provider. If ``None``, the
            environment variable for the provider is consulted
            (``ANTHROPIC_API_KEY`` or ``GEMINI_API_KEY``).
        schema_domain: Key into :data:`ALIAS_DICT_PROMPTS`. Default
            ``"cancer_genomics"`` runs the five-pass pipeline (synonym,
            abbreviation, value-encoded, composite, institutional).
        is_numeric_field: Optional iterable of bools aligned with
            ``target_fields`` marking numeric-valued fields. Ignored when
            ``target_fields`` is a DataFrame carrying this column.
        max_tokens: Maximum tokens for the LLM response (per call).
        batch_size: Max number of fields sent per API call. ``0`` disables
            batching (all fields in a single call).
        dedupe: If True, deduplicate rows on normalized ``(field_name,
            source)`` keys.
        verbose: If True, print per-pass / per-batch progress.

    Returns:
        DataFrame with columns ``field_name``, ``source``, ``is_numeric_field``.
        ``is_numeric_field`` is ``"yes"`` or ``""`` (the format
        :meth:`DictLoader.load_numeric_dict` expects). Persist with
        ``df.to_csv(path, index=False)`` and pass via ``ALIAS_DICT_PATH`` to
        :class:`SchemaMapEngine`.
    """
    if not model or not isinstance(model, str):
        raise ValueError("'model' must be a non-empty string.")
    if max_tokens <= 0:
        raise ValueError("'max_tokens' must be positive.")
    if batch_size < 0:
        raise ValueError("'batch_size' must be non-negative.")

    if schema_domain not in ALIAS_DICT_PROMPTS:
        available = ", ".join(repr(k) for k in ALIAS_DICT_PROMPTS)
        raise ValueError(
            f"Unknown schema_domain {schema_domain!r}. Available: {available}"
        )

    provider = detect_provider(model)
    api_key = resolve_api_key(api_key, provider)
    fields_df = _normalize_target_fields(target_fields, is_numeric_field)
    valid_fields = set(fields_df["field_name"])

    batches = _batch_fields(fields_df, batch_size)
    prompt_set = ALIAS_DICT_PROMPTS[schema_domain]

    if verbose and len(batches) > 1:
        print(
            f"[generate_alias_dict] {len(fields_df)} fields -> "
            f"{len(batches)} batches of <= {batch_size}",
            file=sys.stderr,
        )

    rows_chunks: list[pd.DataFrame] = []
    for pass_name, prompt_text in prompt_set.items():
        if verbose:
            print(f"[generate_alias_dict] pass: {pass_name}", file=sys.stderr)
        for i, batch_csv in enumerate(batches, start=1):
            user_message = _build_user_message(prompt_text, batch_csv)
            try:
                text = call_llm(
                    provider, api_key, user_message, model, max_tokens,
                )
            except Exception as e:  # network, API, auth, parsing
                print(
                    f"[generate_alias_dict] WARNING: pass '{pass_name}' batch "
                    f"{i}/{len(batches)} call failed: "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                continue
            csv_text = _extract_csv_block(text)
            try:
                rows = _parse_alias_csv(csv_text)
            except ValueError as e:
                print(
                    f"[generate_alias_dict] WARNING: pass '{pass_name}' batch "
                    f"{i}/{len(batches)} parse error: {e}",
                    file=sys.stderr,
                )
                continue
            rows_chunks.append(rows)
            if verbose:
                print(
                    f"[generate_alias_dict]   batch {i}/{len(batches)} -> "
                    f"{len(rows)} rows",
                    file=sys.stderr,
                )

    if rows_chunks:
        result = pd.concat(rows_chunks, ignore_index=True)
    else:
        result = _empty_alias_df()

    if dedupe:
        result = _dedupe_alias_rows(result)

    result = result[result["field_name"].isin(valid_fields)].reset_index(
        drop=True
    )

    if verbose:
        n_covered = result["field_name"].nunique()
        print(
            f"[generate_alias_dict] done: {len(result)} alias rows, "
            f"{n_covered}/{len(valid_fields)} fields covered",
            file=sys.stderr,
        )
    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_target_fields(
    target_fields: Union[Iterable[str], pd.DataFrame],
    is_numeric_field: Optional[Iterable[bool]],
) -> pd.DataFrame:
    if isinstance(target_fields, pd.DataFrame):
        if "field_name" not in target_fields.columns:
            raise ValueError(
                "DataFrame 'target_fields' must have a 'field_name' column."
            )
        df = pd.DataFrame({"field_name": target_fields["field_name"].astype(str)})
        if "is_numeric_field" in target_fields.columns:
            df["is_numeric_field"] = _coerce_numeric_flag(
                target_fields["is_numeric_field"]
            )
        else:
            df["is_numeric_field"] = ""
    else:
        names = [str(x) for x in target_fields]
        if is_numeric_field is None:
            flags = [""] * len(names)
        else:
            flags = list(_coerce_numeric_flag(list(is_numeric_field)))
            if len(flags) != len(names):
                raise ValueError(
                    "'is_numeric_field' must match 'target_fields' length."
                )
        df = pd.DataFrame({"field_name": names, "is_numeric_field": flags})

    df = df[df["field_name"].str.len() > 0]
    df = df.drop_duplicates(subset=["field_name"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("'target_fields' is empty after coercion.")
    return df


def _coerce_numeric_flag(values) -> list[str]:
    out: list[str] = []
    truthy_strings = {"yes", "true", "1"}
    for v in values:
        if v is True:
            out.append("yes")
        elif isinstance(v, str) and v.strip().lower() in truthy_strings:
            out.append("yes")
        elif isinstance(v, (int, float)) and not isinstance(v, bool) and v == 1:
            out.append("yes")
        else:
            out.append("")
    return out


def _batch_fields(fields_df: pd.DataFrame, batch_size: int) -> list[str]:
    n = len(fields_df)
    if batch_size <= 0 or n <= batch_size:
        return [_fields_to_csv(fields_df)]
    chunks: list[str] = []
    for start in range(0, n, batch_size):
        chunks.append(_fields_to_csv(fields_df.iloc[start:start + batch_size]))
    return chunks


def _fields_to_csv(fields_df: pd.DataFrame) -> str:
    buf = io.StringIO()
    fields_df[["field_name", "is_numeric_field"]].to_csv(buf, index=False)
    return buf.getvalue().rstrip("\n")


def _build_user_message(prompt_text: str, fields_csv: str) -> str:
    return (
        prompt_text
        + OUTPUT_FORMAT_INSTRUCTIONS
        + "\n\nTarget/harmonized attributes (CSV):\n```csv\n"
        + fields_csv
        + "\n```"
    )


_CSV_FENCED_RE = re.compile(r"```csv\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PLAIN_FENCED_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_CSV_UNCLOSED_RE = re.compile(r"```csv\s*\n(.*)", re.DOTALL | re.IGNORECASE)


def _extract_csv_block(text: str) -> str:
    if not text:
        return ""
    m = _CSV_FENCED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _PLAIN_FENCED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _CSV_UNCLOSED_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _parse_alias_csv(csv_text: str) -> pd.DataFrame:
    if not csv_text:
        return _empty_alias_df()
    try:
        df = pd.read_csv(io.StringIO(csv_text), dtype=str, keep_default_na=False)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM CSV: {e}") from e
    missing = {"field_name", "source"} - set(df.columns)
    if missing:
        raise ValueError(
            f"LLM CSV missing required columns: {sorted(missing)}"
        )
    if "is_numeric_field" not in df.columns:
        df["is_numeric_field"] = ""
    df["field_name"] = df["field_name"].str.strip()
    df["source"] = df["source"].str.strip()
    df["is_numeric_field"] = df["is_numeric_field"].str.strip().str.lower().where(
        lambda s: s == "yes", ""
    )
    df = df[(df["field_name"] != "") & (df["source"] != "")]
    return df[ALIAS_DICT_COLUMNS].reset_index(drop=True)


_NORM_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_alias_key(x: pd.Series) -> pd.Series:
    return x.str.lower().str.replace(_NORM_NON_ALNUM, " ", regex=True).str.strip()


def _dedupe_alias_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = (
        _normalize_alias_key(df["field_name"]) + "\x1f"
        + _normalize_alias_key(df["source"])
    )
    return df[~key.duplicated()].reset_index(drop=True)


def _empty_alias_df() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ALIAS_DICT_COLUMNS})
