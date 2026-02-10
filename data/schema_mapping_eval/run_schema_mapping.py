"""
Minimal CLI for SchemaMapEngine

Usage:
    export PYTHONPATH="$PWD"
    python data/schema_mapping_eval/run_schema_mapping.py \
        --input data/schema/test.csv \
        --mode manual \
        --top-k 5
    
    # With LLM fallback (auto mode only)
    python data/schema_mapping_eval/run_schema_mapping.py \
        --input data/schema/test.csv \
        --mode auto \
        --top-k 5
    
    # Manual LLM fallback on existing results
    python data/schema_mapping_eval/run_schema_mapping.py \
        --input data/schema/test.csv \
        --mode manual \
        --llm-fallback-file data/schema_mapping_eval/test_manual.csv \
        --llm-output data/schema_mapping_eval/test_llm.csv
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import SchemaMapEngine
try:
    from src.models.schema_mapper import SchemaMapEngine
except ImportError as e:
    print(f"[ERROR] Cannot import SchemaMapEngine: {e}", file=sys.stderr)
    print("[ERROR] Make sure PYTHONPATH is set correctly", file=sys.stderr)
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run schema mapping with SchemaMapEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python run_schema_mapping.py --input data/test.csv --mode manual
  
  # Auto mode with LLM fallback
  python run_schema_mapping.py --input data/test.csv --mode auto
  
  # Manual LLM fallback on existing results
  python run_schema_mapping.py --input data/test.csv --mode manual \\
      --llm-fallback-file results.csv --llm-output results_llm.csv
        """
    )
    
    # Input/output arguments
    p.add_argument(
        "-i", "--input",
        required=True,
        help="Path to clinical data CSV/TSV file"
    )
    p.add_argument(
        "--mode",
        default="manual",
        choices=["manual", "auto"],
        help="Mapping mode: 'manual' (no LLM) or 'auto' (with LLM fallback)"
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)"
    )
    
    # LLM fallback arguments (manual mode)
    p.add_argument(
        "--llm-fallback-file",
        default=None,
        help="(Manual mode) Existing results CSV to run LLM fallback on"
    )
    p.add_argument(
        "--llm-output",
        default=None,
        help="(Manual mode) Output file for LLM fallback results"
    )
    p.add_argument(
        "--llm-stage-filter",
        nargs="+",
        default=None,
        help="(Manual mode) Only re-match specific stages (e.g., stage2 stage3)"
    )
    p.add_argument(
        "--llm-merge",
        action="store_true",
        help="(Manual mode) Merge LLM results with original results"
    )
    
    return p.parse_args()


def main():
    """Main execution."""
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)
    
    start = datetime.now()
    print("=" * 80)
    print(f"Schema Mapping Pipeline")
    print("=" * 80)
    print(f"Start time: {start:%Y-%m-%d %H:%M:%S}")
    print(f"Input file: {args.input}")
    print(f"Mode: {args.mode}")
    print(f"Top-K: {args.top_k}")
    print("=" * 80)
    
    try:
        # Initialize engine
        print(f"\n[INFO] Initializing SchemaMapEngine...")
        engine = SchemaMapEngine(
            clinical_data_path=args.input,
            mode=args.mode,
            top_k=args.top_k
        )
        
        # Run schema mapping
        print(f"[INFO] Running schema mapping...")
        df_results = engine.run_schema_mapping()
        
        # Get output file path
        out_file = engine.output_file.replace(".csv", f"_{engine.mode}.csv")
        
        print(f"\n[INFO] ✓ Schema mapping complete!")
        print(f"[INFO] Results saved to: {out_file}")
        print(f"[INFO] Total columns processed: {len(df_results)}")
        
        # Print stage distribution
        if 'stage' in df_results.columns:
            print(f"\n[INFO] Stage distribution:")
            for stage, count in df_results['stage'].value_counts().items():
                print(f"        {stage}: {count}")
        
        # Print average score
        if 'match1_score' in df_results.columns:
            avg_score = df_results['match1_score'].mean()
            print(f"\n[INFO] Average match1_score: {avg_score:.4f}")
        
        # Manual mode: LLM fallback on existing results
        if args.mode == "manual" and args.llm_fallback_file:
            print(f"\n{'=' * 80}")
            print(f"Running LLM Fallback on Existing Results")
            print(f"{'=' * 80}")
            
            if not os.path.exists(args.llm_fallback_file):
                print(f"[ERROR] LLM fallback file not found: {args.llm_fallback_file}", 
                      file=sys.stderr)
                sys.exit(3)
            
            # Determine output file
            if args.llm_output is None:
                base = Path(args.llm_fallback_file).stem
                args.llm_output = str(Path(args.llm_fallback_file).parent / f"{base}_llm.csv")
            
            print(f"[INFO] Input: {args.llm_fallback_file}")
            print(f"[INFO] Output: {args.llm_output}")
            if args.llm_stage_filter:
                print(f"[INFO] Stage filter: {args.llm_stage_filter}")
            print(f"[INFO] Merge results: {args.llm_merge}")
            
            # Run LLM fallback
            df_llm = engine.run_llm_on_file(
                input_csv=args.llm_fallback_file,
                output_csv=args.llm_output,
                stage_filter=args.llm_stage_filter,
                merge_results=args.llm_merge
            )
            
            print(f"\n[INFO] ✓ LLM fallback complete!")
            print(f"[INFO] Processed queries: {len(df_llm)}")
            
            if 'match1_score' in df_llm.columns:
                avg_llm_score = df_llm['match1_score'].mean()
                print(f"[INFO] Average LLM score: {avg_llm_score:.4f}")
        
    except Exception as e:
        print(f"\n[ERROR] Schema mapping failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(3)
    
    # Print summary
    end = datetime.now()
    elapsed = end - start
    
    print(f"\n{'=' * 80}")
    print(f"Summary")
    print(f"{'=' * 80}")
    print(f"End time: {end:%Y-%m-%d %H:%M:%S}")
    print(f"Elapsed time: {elapsed}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()