# MetaHarmonizer Demo Data
    
This directory contains demo datasets for testing MetaHarmonizer functionality.

## Files Created:

### For Schema Mapping:
- `clinical_metadata_demo.tsv` - Sample clinical metadata with 100 patients and messy column names

### For Ontology Mapping:
- `disease_query_terms.csv` - Sample disease terms to be standardized
- `treatment_query_terms.csv` - Sample treatment/drug terms to be standardized  
- `body_site_query_terms.csv` - Sample anatomical location terms to be standardized
- `disease_corpus_terms.csv` - Mock standardized disease ontology terms
- `treatment_corpus_terms.csv` - Mock standardized treatment ontology terms
- `body_site_corpus_terms.csv` - Mock standardized body site ontology terms

### Usage Examples:
- `usage_examples.py` - Complete code examples showing how to use both components

## Getting Started:

1. Run the schema mapping on the clinical data:
```python
from src.Engine import SchemaMapEngine
engine = SchemaMapEngine(path_to_input_table="demo_data/clinical_metadata_demo.tsv")
engine.run_schema_mapping()
```

2. Run ontology mapping on the query terms:
```python
# See usage_examples.py for complete examples
```

## Data Details:

The clinical metadata includes realistic but synthetic data with:
- 100 patients
- Demographics (age, sex, race/ethnicity)
- Disease information (tumor location, stage, grade)
- Treatment details (drugs, therapy types, responses)
- Outcomes (survival, vital status)
- Biomarkers (EGFR, HER2, ER status)

Column names are intentionally non-standardized to demonstrate schema mapping capabilities.
