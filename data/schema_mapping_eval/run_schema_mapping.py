# Minimal CLI for SchemaMapEngine
# Usage:
#   export PYTHONPATH="$PWD"
#   python data/schema_mapping_eval/run_schema_mapping.py --input /home/lcc/projects/MetaHarmonizer/data/schema/test.csv --mode manual --top-k 5

import argparse
import os
import sys
from datetime import datetime

# (Optional) load .env if present; silently ignore if python-dotenv not installed
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

try:
    from src.Engine import SchemaMapEngine
except Exception as e:
    print("[ERROR] Cannot import SchemaMapEngine from src.Engine:",
          e,
          file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run schema mapping with SchemaMapEngine")
    p.add_argument(
        "-i",
        "--input",
        default=os.getenv("CLINICAL_DATA_PATH", "test.csv"),
        help=
        "Path to clinical data CSV (default: $CLINICAL_DATA_PATH or test.csv)",
    )
    p.add_argument(
        "--mode",
        default=os.getenv("SCHEMA_MODE", "manual"),
        choices=["manual", "auto", "hybrid"],
        help='Mapping mode (default: $SCHEMA_MODE or "manual")',
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "5")),
        help="Top-K candidates to keep (default: $TOP_K or 5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start = datetime.now()
    print(f"[INFO] Start schema mapping at {start:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] input={args.input} mode={args.mode} top_k={args.top_k}")

    engine = SchemaMapEngine(
        clinical_data_path=args.input,
        mode=args.mode,
        top_k=args.top_k,
    )
    engine.run_schema_mapping()

    end = datetime.now()
    print(f"[INFO] Done at {end:%Y-%m-%d %H:%M:%S} (elapsed: {end - start})")


if __name__ == "__main__":
    main()
