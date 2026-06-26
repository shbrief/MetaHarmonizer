#!/usr/bin/env python3
"""Generate an LLM-based alias dictionary for SchemaMapper.

The alias dictionary expands each *standardized* field name into the many
real-world column-name aliases it might appear as in raw clinical data
(e.g. ``age_at_diagnosis`` -> ``AGE_D``, ``AGE_AT_DX``,
``INVASIVE_CARCINOMA_DX_AGE``). Stage-1 of ``SchemaMapEngine`` matches
incoming column names against this dictionary first, so a good alias
dictionary is the single biggest lever on SchemaMapper accuracy.

This is a one-time/occasional *setup* step: you generate the CSV once,
persist it, then point the engine at it. It is a script rather than a
notebook on purpose — every run costs LLM tokens, so you don't want a
cell you re-run by accident.

Lifecycle
---------
    1. read target fields  (default: the bundled cbio_target_attrs.csv)
    2. generate_alias_dict(...) -> DataFrame   [LLM calls happen here]
    3. df.to_csv(out, index=False)             [the persisted artifact]
    4. SchemaMapEngine(alias_dict_path=out)    [the engine consumes it]

The output schema is exactly ``field_name,source,is_numeric_field`` —
the same columns the bundled default alias dictionary uses, and the only
columns ``DictLoader.load_alias_dict`` reads.

Examples
--------
Cheap smoke test (3 fields, one provider call set; ~pennies)::

    export ANTHROPIC_API_KEY=sk-...
    python generate_alias_dict.py --model claude-sonnet-4-6 --limit 3

Full bundled schema with Gemini, custom output path::

    export GEMINI_API_KEY=...
    python generate_alias_dict.py --model gemini-2.5-pro \
        --out data/outputs/alias_dict.csv

Generate for your own field set, then verify the engine loads it::

    python generate_alias_dict.py --model claude-sonnet-4-6 \
        --fields my_fields.csv --out my_alias_dict.csv --demo-match
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from metaharmonizer.models.schema_mapper.config import CURATED_DICT_PATH
from metaharmonizer.models.schema_mapper.generate_alias_dict import (
    generate_alias_dict,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate an LLM-based alias dictionary for SchemaMapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="LLM model id. Provider is auto-detected from the prefix: "
        "'claude-*' -> Anthropic (ANTHROPIC_API_KEY), "
        "'gemini-*' -> Gemini (GEMINI_API_KEY).",
    )
    p.add_argument(
        "--fields",
        default=str(CURATED_DICT_PATH),
        help="CSV of target fields. Needs a 'field_name' column; an optional "
        "'is_numeric_field' column is forwarded. Defaults to the bundled "
        "cbio_target_attrs.csv (the engine's default target schema).",
    )
    p.add_argument(
        "--out",
        default="data/outputs/alias_dict.csv",
        help="Where to write the generated alias dictionary CSV.",
    )
    p.add_argument(
        "--schema-domain",
        default="cancer_genomics",
        help="Prompt set to use (key into ALIAS_DICT_PROMPTS).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only use the first N target fields. Use a small value "
        "(e.g. 3) for a cheap smoke test before a full run.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Override the provider env var (ANTHROPIC_API_KEY / "
        "GEMINI_API_KEY). Usually leave unset and use the env var.",
    )
    p.add_argument(
        "--demo-match",
        action="store_true",
        help="After writing, load the generated dict into a SchemaMapEngine "
        "and run one mapping to prove the artifact is consumable. Needs the "
        "ML extras installed (sentence-transformers, etc.).",
    )
    return p.parse_args(argv)


def load_target_fields(path: str, limit: int) -> pd.DataFrame:
    """Read the target-field CSV, keeping is_numeric_field if present."""
    df = pd.read_csv(path)
    if "field_name" not in df.columns:
        raise SystemExit(
            f"--fields file {path!r} has no 'field_name' column "
            f"(found: {list(df.columns)})."
        )
    keep = ["field_name"]
    if "is_numeric_field" in df.columns:
        keep.append("is_numeric_field")
    df = df[keep]
    if limit and limit > 0:
        df = df.head(limit)
    return df.reset_index(drop=True)


def demo_match(out_path: str) -> None:
    """Load the generated dict into the engine and map one alias-y column.

    Kept import-local so the script's generation path doesn't require the
    heavier ML stack the engine pulls in.
    """
    from metaharmonizer.models.schema_mapper.engine import SchemaMapEngine

    alias_df = pd.read_csv(out_path)
    # Pick a real alias from the generated file as a synthetic input column,
    # so the demo reflects what was actually produced.
    sample_alias = str(alias_df["source"].iloc[0])
    expected_field = str(alias_df["field_name"].iloc[0])
    print(
        f"\n[demo-match] feeding column {sample_alias!r} "
        f"(should map to {expected_field!r})",
        file=sys.stderr,
    )

    demo_input = "data/outputs/_alias_demo_input.csv"
    pd.DataFrame({sample_alias: ["x"]}).to_csv(demo_input, index=False)

    engine = SchemaMapEngine(
        clinical_data_path=demo_input,
        mode="manual",
        top_k=3,
        alias_dict_path=out_path,
    )
    result = engine.run_schema_mapping()
    cols = [c for c in ("query", "stage", "method", "match1") if c in result.columns]
    print(result[cols].to_string(index=False), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    fields_df = load_target_fields(args.fields, args.limit)
    print(
        f"[generate_alias_dict.py] {len(fields_df)} target fields from "
        f"{args.fields}"
        + (f" (limited to {args.limit})" if args.limit else ""),
        file=sys.stderr,
    )

    alias_df = generate_alias_dict(
        target_fields=fields_df,
        model=args.model,
        api_key=args.api_key,
        schema_domain=args.schema_domain,
    )

    if alias_df.empty:
        print(
            "[generate_alias_dict.py] ERROR: no alias rows produced "
            "(every LLM call failed or returned nothing). Nothing written.",
            file=sys.stderr,
        )
        return 1

    # Ensure the output directory exists, then persist in the canonical schema.
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    alias_df.to_csv(args.out, index=False)
    print(
        f"[generate_alias_dict.py] wrote {len(alias_df)} alias rows "
        f"({alias_df['field_name'].nunique()} fields) -> {args.out}",
        file=sys.stderr,
    )

    if args.demo_match:
        demo_match(args.out)

    print(
        "\nNext: point the engine at it ->\n"
        f"    SchemaMapEngine(clinical_data_path=..., "
        f"alias_dict_path={args.out!r})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
