# Demos and Utilities

This directory contains Jupyter notebooks demonstrating the **Ontology Mapper**
and **Schema Mapper** workflows, along with notebooks for evaluation and data
alignment checks. Sample inputs live under [`data/`](data/) and the evaluation
helpers under [`functions/`](functions/); most notebooks run out of the box (the
evaluation notebook is the exception — it reads results from your own prior runs).

> **Note:** Run the notebooks from this `examples/` directory (the default when
> you open a notebook here in Jupyter). Data paths are written relative to
> `examples/`, e.g. `data/...`, so no `%cd` is required.
>
> Sample inputs are **small fabricated samples** so the notebooks run
> end-to-end out of the box. They are illustrative, not the full research
> corpora. Files produced by a run (engine outputs, fetched/cached corpora,
> evaluation tables, `missing_terms.csv`, etc.) are written back under `data/`
> in subdirectories (`data/corpus/`, `data/outputs/`, `data/schema_mapping_eval/`)
> that are created on first run.

## 📂 Layout

- `data/` – flat directory of sample inputs the notebooks read
- `functions/` – evaluation helpers (`calc_stats`, `schema_mapping_evaluation`),
  imported as `from functions... import ...`

## 📂 Workflows

- **ontology_mapper_workflow.ipynb** – Demonstration of the Ontology Mapper
  workflow via `OntoMapEngine`. Covers preprocessing, mapping, and accuracy
  evaluation.
  - Sample inputs: `data/disease_query_updated.csv`,
    `data/query_with_selected_fields_for_bie.csv`
  - Note: the engine builds/fetches the ontology corpus (NCIt/MONDO) from
    upstream APIs on first run and downloads embedding models — network access
    is required for the mapping cells.

- **schema_mapper_workflow.ipynb** – Demonstrates the Schema Mapper workflow via
  the `SchemaMapEngine`, which maps clinical-data columns to standard field
  names.
  - Sample input: `data/sm_test.csv` (small clinical sample)
  - Standard field names, NCIt descendants, and the field-value dictionary are
    resolved from the installed package's bundled data by default
    (`metaharmonizer/_bundled_data/schema/...`); settings live in
    `metaharmonizer/models/schema_mapper/config.py`
  - Evaluation reads `data/sm_truth.csv` and the prediction file produced by
    the run

## 📂 Evaluation & Utilities

- **evaluation_and_visualization.ipynb** – Compares thresholds and strategies
  across saved result files and computes Top-1/Top-3/Top-5 accuracy.
  - Inputs are result CSVs under `data/outputs/...` produced by your own
    ontology-mapper runs — none are bundled, and missing combinations are
    skipped gracefully, so run the ontology-mapper notebook first to populate
    them.

- **query_term_normalization.ipynb** – Checks for alignment issues between
  **query data** and **corpus data**. Misalignments can distort accuracy
  evaluation, so running this before large-scale evaluation is recommended.
  - Sample inputs: `data/disease_query_updated.csv`,
    `data/disease_corpus_updated.csv`, and the `data/disease_*_for_NCIT:C3262.csv`
    query/corpus pair
  - The NCI label-mismatch cell calls the NCI API and requires network access
