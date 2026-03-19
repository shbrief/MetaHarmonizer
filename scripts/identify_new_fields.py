"""Identify new harmonized fields from unmapped columns using FieldSuggester."""

import argparse
import pandas as pd
from src.models.schema_mapper.engine import SchemaMapEngine
from src.field_suggester import suggest_from_schema_mapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Identify new harmonized fields from unmapped columns."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the clinical data TSV or CSV file.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold below which a column is considered "
                             "unmapped (default: 0.5).")
    args = parser.parse_args()

    sep = "\t" if args.input.endswith(".tsv") else ","
    df = pd.read_csv(args.input, sep=sep)

    engine = SchemaMapEngine(clinical_data_path=args.input, mode="auto")
    results_df = engine.run_schema_mapping()

    suggestions = suggest_from_schema_mapper(
        schema_mapper_results=results_df,
        df=df,
        score_threshold=args.threshold,
    )

    print("\nSuggested New Harmonized Fields:")
    print("=" * 80)

    if suggestions:
        for field_name, info in suggestions.items():
            print(f"\n{field_name} (confidence: {info['confidence']:.3f}):")
            print(f"  Source columns: {info['source_columns']}")
            if info['ner_entities']:
                print(f"  NER entities: {info['ner_entities']}")
    else:
        print("No field suggestions generated.")
