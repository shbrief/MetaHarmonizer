# Demos and Utilities

This directory contains Jupyter notebooks demonstrating the **Ontology Mapper**
and **Schema Mapper** workflows, along with notebooks for evaluation and data
alignment checks. It is self-contained: sample inputs live under
[`data/`](data/) and the evaluation helpers under [`functions/`](functions/).

> **Note:** Run the notebooks from this `examples/` directory (the default when
> you open a notebook here in Jupyter). Data paths are written relative to
> `examples/`, e.g. `data/...`, so no `%cd` is required.
>
> Some inputs and all result/output files are **small fabricated samples** so the
> notebooks run end-to-end out of the box. They are illustrative, not the full
> research corpora. Files produced by a run (engine outputs, evaluation tables,
> `missing_terms.csv`, etc.) are written back under `data/`.

## 📂 Layout

- `data/` – sample inputs mirroring the paths the notebooks expect
- `functions/` – evaluation helpers (`calc_stats`, `schema_mapping_evaluation`),
  imported as `from functions... import ...`

## 📂 Workflows

- **ontology_mapper_workflow.ipynb** – Demonstration of the Ontology Mapper
  workflow via `OntoMapEngine`. Covers preprocessing, mapping, and accuracy
  evaluation.
  - Sample inputs: `data/corpus/cbio_disease/disease_query_updated.csv`,
    `data/corpus/cbio_disease/query_with_selected_fields_for_bie.csv`
  - Note: the engine builds/fetches the ontology corpus (NCIt/MONDO) from
    upstream APIs on first run and downloads embedding models — network access
    is required for the mapping cells.

- **schema_mapper_workflow.ipynb** – Demonstrates the Schema Mapper workflow via
  the `SchemaMapEngine`, which maps clinical-data columns to standard field
  names.
  - Sample input: `data/schema/test.csv` (small clinical sample)
  - Standard field names, NCIt descendants, and the field-value dictionary are
    resolved from the installed package's bundled data by default
    (`metaharmonizer/_bundled_data/schema/...`); settings live in
    `metaharmonizer/models/schema_mapper/config.py`
  - Evaluation reads `data/schema_mapping_eval/truth.csv` and the prediction
    file produced by the run

## 📂 Evaluation & Utilities

- **evaluation_and_visualization.ipynb** – Compares thresholds and strategies
  across saved result files and computes Top-1/Top-3/Top-5 accuracy.
  - Sample inputs: `data/outputs/2025/large_corpus/1128/*_result_*.csv`
    (fabricated result samples; missing combinations are skipped gracefully)

- **query_term_normalization.ipynb** – Checks for alignment issues between
  **query data** and **corpus data**. Misalignments can distort accuracy
  evaluation, so running this before large-scale evaluation is recommended.
  - Sample inputs under `data/corpus/cbio_disease/` (query/corpus pairs)
  - The NCI label-mismatch cell calls the NCI API and requires network access
