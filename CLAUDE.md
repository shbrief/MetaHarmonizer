---
editor_options: 
  markdown: 
    wrap: 72
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project Overview

MetaHarmonizer is a biomedical metadata harmonization platform for
standardizing clinical data. Three core capabilities:

1.  **Ontology Mapping** — Maps free-text biomedical terms to
    standardized ontology terms (NCIT, UMLS, UBERON, SNOMED) via
    multi-stage cascade: exact/fuzzy match → embedding (LM/ST) → synonym
    boost → RAG (Cross-encoder Re-ranking)
2.  **Schema Mapping** — Maps clinical data column names to harmonized
    schema fields via 4-stage pipeline: dict/fuzzy (actual column names
    and their alias) → value matching (ontology tree conversion or value
    dictionary) → field-type based matching (e.g., numeric, pattern,
    etc.) → LLM (optional)
3.  **(on-going, experimental) Semantic Clustering & Field Suggestion**
    — Discovers new relevant, harmonizable fields & Suggest new
    harmonized field names for unmapped columns. Using NER + embedding
    clustering

## Common Commands

``` bash
# Environment setup
conda create -n metaharmonizer python=3.10 -y && conda activate metaharmonizer
pip install --upgrade pip && pip install -r requirements.txt

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_field_suggester.py

# Run a single test function
pytest tests/test_field_suggester.py::test_function_name -v

# Run scripts
python scripts/run_schema_identifier.py      # Schema mapping
python scripts/identify_new_fields.py         # Field suggestion
```

No linter or formatter is configured. No pyproject.toml or setup.cfg
exists.

## Architecture

### Engine Pattern (Multi-Stage Cascade)

Each engine processes inputs through sequential stages where each stage
handles progressively harder cases. Items matched in earlier stages skip
later ones.

-   **Ontology engine** (`src/Engine/ontology_mapping_engine.py`):
    `OntoMapEngine` — stages: exact → LM/ST embedding → synonym boost →
    RAG
-   **Schema engine** (`src/models/schema_mapper/engine.py`):
    `SchemaMapEngine`

### Strategy Pattern (Ontology Mappers)

All ontology mapping strategies inherit from `OntoModelsBase`
(`src/models/ontology_models.py`). Strategies: - `ontology_mapper_lm.py`
— CLS token embeddings - `ontology_mapper_st.py` — Sentence Transformer
embeddings - `ontology_mapper_rag.py` — FAISS + SQLite
retrieval-augmented generation - `ontology_mapper_bi_encoder.py` —
Bi-encoder with query enrichment - `ontology_mapper_synonym.py` —
Synonym-based matching

### Schema Mapper Pipeline (`src/models/schema_mapper/`)

Modular 4-stage matching with separate matcher classes: -
`matchers/stage1_matchers.py` — Exact/fuzzy matching -
`matchers/stage2_matchers.py` — Dictionary/alias matching -
`matchers/stage3_matchers.py` — Value/NCIT-based matching -
`matchers/stage4_matchers.py` — LLM-based matching - `config.py` —
Thresholds (FUZZY_THRESH=90, NUMERIC_THRESH=0.6, etc.) and paths -
`loaders/` — Data loaders for dictionaries and value mappings

### Field Suggester (`src/field_suggester/`)

-   `field_suggester.py` — `FieldSuggester`: NER + embedding clustering
    for new field discovery
-   `semantic_clustering.py` — Clustering logic for grouping similar
    fields

### Knowledge Database (`src/KnowledgeDb/`)

-   `faiss_sqlite_pipeline.py` — `FAISSSQLiteSearch`: vector similarity
    with SQLite backend
-   `db_clients/` — API clients for NCI Thesaurus, UMLS, EBI OLS4

## Coding Conventions

-   **Python 3.10+** with type hints
-   **Logging**: Use `CustomLogger` from
    `src/CustomLogger/custom_logger.py` — call `custlogger()` which
    auto-detects the calling class name. Never use `print()`.
-   **Data format**: pandas DataFrames as primary interchange format
    between engines
-   **Model loading**: Use `src/utils/model_loader.py` and
    `model_cache.py` — never load HuggingFace models directly
-   **String normalization**: Use `normalize()` from
    `src/utils/schema_mapper_utils.py` for all string comparison
-   **Tests**: pytest framework, test files in `tests/`. No conftest.py;
    tests are self-contained with inline fixtures.

## Required Environment Variables (.env)

```         
METHOD_MODEL_YAML=src/models/method_model.yaml
UMLS_API_KEY=YOUR_UMLS_API_KEY_HERE
VECTOR_DB_PATH=src/KnowledgeDb/vector_db.sqlite
FAISS_INDEX_DIR=src/KnowledgeDb/faiss_indexes
FIELD_VALUE_JSON=data/schema/field_value_dict.json
CLINICAL_DATA_PATH=data/test.csv
SCHEMA_MODE=manual
TOP_K=5
```

## Data Directories

| Directory | Contents |
|--------------------------------------|----------------------------------|
| `data/schema/` | Schema dictionaries (`curated_fields.csv`, `field_value_dict.json`, aliases) |
| `data/corpus/` | Ontology corpora (disease, body site, treatment) |
| `data/schema_mapping_eval/` | Schema mapping output files |
| `data/outputs/` | General output data |

Encrypted datasets require git-crypt authorization.

## Key Entry Points

``` python
# Ontology Mapping
from src.Engine.ontology_mapping_engine import OntoMapEngine
engine = OntoMapEngine(category='disease', query=terms, corpus=corpus,
                       cura_map={}, topk=5, s2_method='sap-bert',
                       s2_strategy='lm', test_or_prod='prod')
results_df = engine.run()

# Schema Mapping
from src.models.schema_mapper import SchemaMapEngine
engine = SchemaMapEngine(clinical_data_path='data/clinical.csv', mode='auto', top_k=5)
results_df = engine.run_schema_mapping()

# Field Suggestion
from src.field_suggester import FieldSuggester
suggester = FieldSuggester()
suggestions = suggester.suggest(unmapped_columns, df=clinical_df)
```

## Team Workflow

-   **Branches**: `<username>-dev` (e.g., `lcc-dev`, `abhi_dev`) or
    descriptive feature branches
-   **PRs**: Target `main` branch, require review before merge
-   **Commits**: Short descriptive phrases (e.g., "Refactor schema
    mapping", "Add stage4 to schema mapper")
-   **Testing**: Run `pytest tests/` before merging PRs

## Sibling Projects
- `~/OmicsMLRepo/MetaHarmonizerR`: An R wrapper package for MetaHarmonizer
- `~/OmicsMLRepo/MetaHarmonizerEval`: Evaluation on MetaHarmonizer outputs
