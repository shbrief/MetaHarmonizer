"""Identify new harmonized fields from unmapped columns using FieldSuggester."""

import pandas as pd
from src.field_suggester import FieldSuggester
from src.models.schema_mapper.engine import SchemaMapEngine
from src.utils.embedding_store import EmbeddingStore


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/schema/hnsc_tcga_gdc_clinical_data.tsv", sep="\t")

    # Run schema mapping
    engine = SchemaMapEngine(
        clinical_data_path="data/schema/hnsc_tcga_gdc_clinical_data.tsv",
        mode="auto"
    )
    results_df = engine.run_schema_mapping()

    # Find columns that weren't confidently mapped
    unmapped = results_df[
        (results_df['match1_score'].isna()) | (results_df['match1_score'] < 0.5)
    ]['query'].tolist()

    print(f"Found {len(unmapped)} unmapped columns")
    print(f"Unmapped: {unmapped}\n")

    # Suggest new harmonized fields using FieldSuggester
    # Optionally share an EmbeddingStore for cross-engine reuse:
    #   store = EmbeddingStore()
    #   suggester = FieldSuggester(embedding_store=store)
    suggester = FieldSuggester()
    suggestions = suggester.suggest(unmapped, df=df)

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
