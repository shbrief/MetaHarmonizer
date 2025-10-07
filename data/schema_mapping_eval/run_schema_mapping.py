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
    from src.Engine import get_schema_engine
    SchemaMapEngine = get_schema_engine()
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
        default=os.getenv("CLINICAL_DATA_PATH",
                          "data/schema/cBioPortal_all_clinicalData_2024.tsv"),
        help=
        "Path to clinical data CSV (default: $CLINICAL_DATA_PATH or data/schema/cBioPortal_all_clinicalData_2024.tsv)",
    )
    p.add_argument(
        "--mode",
        default=os.getenv("SCHEMA_MODE", "manual"),
        choices=["manual", "auto"],
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

    if not os.path.exists(args.input):
        print(f"[ERROR] input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    start = datetime.now()
    print(f"[INFO] Start schema mapping at {start:%Y-%m-%d %H:%M:%S}")
    print(f"[INFO] input={args.input} mode={args.mode} top_k={args.top_k}")

    try:
        engine = SchemaMapEngine(
            clinical_data_path=args.input,
            mode=args.mode,
            top_k=args.top_k,
        )
        df = engine.run_schema_mapping()
        out_file = engine.output_file.replace(".csv", f"_{engine.mode}.csv")
        print(f"[INFO] Results saved to: {out_file}")
        print(f"[INFO] Mapped columns: {len(df)}")
    except Exception as e:
        print(f"[ERROR] schema mapping failed: {e}", file=sys.stderr)
        sys.exit(3)

    end = datetime.now()
    print(f"[INFO] Done at {end:%Y-%m-%d %H:%M:%S} (elapsed: {end - start})")


if __name__ == "__main__":
    main()
