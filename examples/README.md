# Demos and Utilities

This directory contains Jupyter notebooks demonstrating the **OntologyMapper**
and **SchemaMapper** workflows, along with notebooks for evaluation and data
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

## 🚀 Start here

- **quickstart.ipynb** – The shortest path from your own data to mapped results.
  Runs both engines in **`prod` mode** (you supply only the input — no ground
  truth, no `cura_map`) and explains how to read each output. Start here if you
  are new; the workflow notebooks below go deeper into evaluation and tuning.
  - Sample inputs: `data/disease_corpus_updated.csv`, `data/sm_test.tsv`
  - Companion reference: [`../docs/input_formats.md`](../docs/input_formats.md)

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
  - Sample input: `data/sm_test.tsv` (small clinical sample)
  - Standard field names, NCIt descendants, and the field-value dictionary are
    resolved from the installed package's bundled data by default
    (`src/metaharmonizer/_bundled_data/schema/...`); settings live in
    `src/metaharmonizer/models/schema_mapper/config.py`
  - Evaluation reads `data/sm_truth.tsv` and the prediction file produced by
    the run

- **generate_alias_dict.ipynb** – Generates the Stage-1 **alias dictionary**
  SchemaMapper matches against, the biggest lever on its accuracy. Defines a
  small, diverse set of target fields, calls an LLM to expand each into
  real-world column-name aliases, writes a `field_name,source,is_numeric_field`
  CSV, then loads it back into a `SchemaMapEngine` to prove it is consumable.
  - **Spends LLM tokens** on the generate cell; needs `ANTHROPIC_API_KEY` or
    `GEMINI_API_KEY` (provider auto-detected from the model id). The demo uses a
    5-field target set so a run costs pennies — re-run the generate cell only to
    regenerate.
  - A non-interactive CLI equivalent (argparse, `--limit`, `--demo-match`) is in
    [`generate_alias_dict.py`](generate_alias_dict.py).
  - Format reference: [`../docs/input_formats.md`](../docs/input_formats.md) §2.5

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
