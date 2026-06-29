# MetaHarmonizer: a robust, fully local biomedical metadata harmonization system

The pre-print is now available:

> ***MetaHarmonizer: robust biomedical metadata harmonization and a
> contamination control for inflated LLM performance on public
> benchmarks***\
> Changchang Li, Abhilash Dhal, Kai Gravel-Pucillo, Kaelyn Long, Michele
> Waters, Ino de Bruijn, Sean Davis, Sehyun Oh\
> doi: <https://doi.org/10.64898/2026.06.13.732088>

MetaHarmonizer currently provides two key modules:

| Module | Engine | Purpose |
|------------------------|------------------------|------------------------|
| SchemaMapper (SM) | `SchemaMapEngine` | Map clinical-data columns to standardized field names (dict → fuzzy → value → name). |
| OntologyMapper (OM) | `OntoMapEngine` | Map free-text values to ontology terms (exact → semantic → synonym). |

![](metaharmonizer_diagram.png)

## Table of contents

1.  [Installation](#1-installation)
2.  [Environment variables](#2-environment-variables)
    -   [Project config file](#project-config-file)
3.  [Quickstart](#3-quickstart)
    -   [Minimal setup](#minimal-setup)
    -   [Schema mapping](#schema-mapping)
    -   [Ontology mapping](#ontology-mapping)
4.  [Datasets](#4-datasets)
5.  [SchemaMapper](#5-schemamapper)
6.  [OntologyMapper](#6-ontologymapper)
7.  [Example notebooks](#7-example-notebooks)
8.  [Frequently asked questions](#8-frequently-asked-questions)
    -   [Prepare inputs](#prepare-inputs)
    -   [Interpret outputs](#interpret-outputs)
    -   [Choosing an engine and mode](#choosing-an-engine-and-mode)
    -   [Performance and caching](#performance-and-caching)

## 1. Installation

``` bash
# Clone
git clone https://github.com/shbrief/MetaHarmonizer
cd MetaHarmonizer

# Create a Python 3.10 environment
conda create -n mh python=3.10 -y
conda activate mh

# Install FAISS (intentionally NOT a pip dependency of this package).
#   - macOS: use conda-forge. The PyPI `faiss-cpu` wheel bundles its own libomp,
#     which collides with torch's and segfaults during search; the conda-forge
#     build links the env's shared llvm-openmp instead.
#   - Linux / Windows / any conda-free env (incl. CI): `pip install faiss-cpu`
#     works — the libomp clash is macOS-specific.
conda install -c conda-forge faiss-cpu -y   # macOS
# pip install faiss-cpu                      # Linux / Windows / CI

# Install the package. Pick one of:
pip install -e .                       # core only (no LLM backends)
pip install -e ".[llm-gemini]"         # + Gemini (Stage-4 LLM, OntoMapLLM, etc.)
pip install -e ".[llm-openai]"         # + OpenAI (FieldSuggester semantic clustering)
pip install -e ".[llm-anthropic]"      # + Anthropic (Claude LLM backend)
pip install -e ".[notebook]"           # + nest-asyncio for Jupyter workflows
pip install -e ".[dev]"                # + pytest & coverage
pip install -e ".[eval]"               # + scipy for evaluation scripts
pip install -e ".[all]"                # notebook + eval + all three LLM backends
```

Install directly from GitHub (non-editable). FAISS is not a pip
dependency, so install it first — conda-forge on macOS, pip elsewhere
(see the note above):

``` bash
conda install -c conda-forge faiss-cpu -y   # macOS
# pip install faiss-cpu                      # Linux / Windows / CI
pip install "git+https://github.com/shbrief/MetaHarmonizer#egg=metaharmonizer[llm-gemini]"
```

> The full ontology corpus is **not bundled** in the wheel. Set
> `METAHARMONIZER_DATA_DIR` (see [Environment
> variables](#2-environment-variables)) to point at a local copy, or let
> the engine fetch/build it on first run (set `UMLS_API_KEY` and/or pass
> `corpus_df=`). A small reference dataset
> (`schema/cbio_target_attrs.csv`, `corpus/oncotree_code_to_name.csv`,
> etc.) ships inside the wheel for `SchemaMapEngine` and OncoTree
> lookups. Small, runnable sample inputs for the demo notebooks live
> under [`examples/data/`](examples/data/).

## 2. Environment variables

Configuration is resolved through a single precedence chain (highest
wins):

```         
engine/CLI argument  >  environment variable  >  project config file  >  built-in default
```

-   **Arguments** — what changes per run, passed to `OntoMapEngine` /
    `SchemaMapEngine` (e.g. `s2_method`, `top_k`, `target_schema_path`,
    `value_dict_path`, `alias_dict_path`, `corpus_hash`).
-   **Environment variables** — secrets and deployment/ops (table
    below).
-   **Project config file** — project-level defaults; see [Project
    config file](#project-config-file).
-   **Built-in defaults** — ship with the package; everything works
    unset.

Copy `.env.example` → `.env` (or export in your shell) before running
the mappers. `python-dotenv` auto-loads `.env` on import.

```         
# Set up environment variables (see the table below)
cp .env.example .env
```

`UMLS_API_KEY` and `GEMINI_API_KEY` are **secrets** — env-only, never
put them in a config file. Here are the environment variables set in
`.env`.

| Variable | Required for | Default | Notes |
|------------------|------------------|------------------|--------------------|
| `UMLS_API_KEY` | OM Stage 2.5 NCI Thesaurus lookups, concept-table builder, `update_term_via_code` | — | Required for `ontology_source="ncit"` pipeline stages that hit the live API. |
| `METAHARMONIZER_DATA_DIR` | Locating corpus + schema reference files (`oncotree_code_to_name.csv`, `cbio_target_attrs.csv`, etc.) | `~/.metaharmonizer/data` | Small reference files ship inside the wheel as a fallback when this dir is empty; set this to a local corpus copy to override. |
| `SM_OUTPUT_DIR` | `SM` output path | `$METAHARMONIZER_DATA_DIR/schema_mapping_eval` | Overrides where CSV results are written. |
| `FIELD_VALUE_JSON` | SM value dictionary | `$METAHARMONIZER_DATA_DIR/schema/field_value_dict.json` | Point at an alternative value dict. |
| `VECTOR_DB_PATH` | OM Knowledge-DB SQLite file | `$KNOWLEDGE_DB_DIR/vector_db.sqlite` | — |
| `FAISS_INDEX_DIR` | OM STage 2/3 FAISS index cache | `$KNOWLEDGE_DB_DIR/faiss_indexes` | — |
| `KNOWLEDGE_DB_DIR` | Root dir for KnowledgeDb assets | `~/.metaharmonizer/KnowledgeDb` | — |
| `METHOD_MODEL_YAML` | Method→model registry | bundled `src/metaharmonizer/models/method_model.yaml` | — |
| `MODEL_CACHE_ROOT` / `MODEL_CACHE_DIR` | Hugging Face model cache | `~/.metaharmonizer/model_cache` | `MODEL_CACHE_ROOT` takes precedence; `MODEL_CACHE_DIR` is a fallback. |
| `FIELD_MODEL` | SM embedding stages encoder | `all-MiniLM-L6-v2` | — |
| `NCIT_POOL_SIZE` | NCI async client connection pool | `8` | Raise for bulk corpus builds. |
| `LOG_FILE` / `LOG_ENV` | Logger config | `out.log` / `development` | — |

### Project config file

Project-level defaults (thresholds, model keys, the noise-value set) can
be set without code changes in `metaharmonizer.toml` (the whole file) or
a `[tool.metaharmonizer]` table in `pyproject.toml`, discovered from the
current working directory. These sit *below* environment variables and
arguments in the precedence chain, so a per-run argument always wins.

``` toml
# metaharmonizer.toml  (or [tool.metaharmonizer] in pyproject.toml)
field_model = "minilm-l6"      # method key from method_model.yaml
llm_model   = "gemma-27b"
top_k        = 5

# schema-mapper thresholds
fuzzy_thresh            = 92
numeric_thresh          = 0.6
field_alias_thresh      = 0.5
value_dict_thresh       = 0.85
value_unique_cap        = 50
value_percentage_thresh = 0.2
llm_threshold           = 0.5

# replaces the built-in noise-value set when present
noise_values = ["yes", "no", "unknown", "not reported", "n/a"]
```

> Parsing uses the stdlib `tomllib` (Python ≥ 3.11) or the optional
> `tomli` backport on 3.10. If neither is available the file layer is
> silently skipped and built-in defaults apply.

## 3. Quickstart

The snippets below read the small sample inputs under
[`examples/data/`](examples/data/) (run them from the repo root, or
adjust the paths to your own files).

### Minimal setup

``` bash
# Clone
git clone https://github.com/shbrief/MetaHarmonizer
cd MetaHarmonizer

# Create a Python 3.10 environment
conda create -n mh python=3.10 -y
conda activate mh

# Install FAISS from conda-forge (NOT pip) — see note in §1 Installation
conda install -c conda-forge faiss-cpu -y   # macOS
# pip install faiss-cpu                      # Linux / Windows / CI

# Install the package. Pick one of:
pip install -e .                       # core only (no LLM backends)

# Set up environment variables (see the table below)
cp .env.example .env
```

### Schema mapping

``` python
from metaharmonizer import SchemaMapEngine

engine = SchemaMapEngine(
    input_path="examples/data/Gillette_source.csv",
    schema="gdc",
)
results = engine.run_schema_mapping()
print(results.head())
```

### Ontology mapping

Pass `corpus_df=` to `OntoMapEngine` or set `UMLS_API_KEY` so the engine
can fetch the ontology corpus on first use.

> The initial `OntoMapEngine` run for the example below does one-time
> expensive work — fetching the NCIt corpus from the API, building
> concept tables, downloading the sap-bert encoder (\~440 MB), and
> building the FAISS index (\~4 min for the full corpus). All of it is
> cached, so subsequent runs take \~7 sec.

``` python
import pandas as pd
from metaharmonizer import OntoMapEngine

df = pd.read_csv("examples/data/disease_query_updated.csv")
engine = OntoMapEngine(
    corpus_category="disease",
    query_ls=df["original_value"].tolist(),
    ground_truth_map=dict(zip(df["original_value"], df["curated_ontology"])),
    output_dir="examples/data/outputs",
)
results = engine.run()
print(results.head())
```

Richer examples (custom corpus, MONDO/UBERON sources, Stage-4 LLM
review) are in the reference sections below and the notebooks under
[`examples/`](examples/).

## 4. Datasets

-   For **schema mapping**, provide a biomedical metadata file.
    Currently, it is verified for cancer-related metadata, but other
    domains should work with an alias dictionary. The schema-mapping
    reference dictionary ships bundled inside the wheel for key schemas.
    (`src/metaharmonizer/_bundled_data/schema/`).
-   For **ontology mapping**, you must provide:
    -   A list of query terms via the `query` parameter (or a
        `query_df` + `query_col` pair).
    -   A `corpus` list and/or `corpus_df` are **optional** — the engine
        auto-resolves them from cached CSV or the API when not provided;
        currently, it supports corpus for several key attributes.
-   Small, runnable sample inputs for the demo notebooks live under
    [`examples/data/`](examples/data/); they are illustrative, not the
    full research corpora.

## 5. SchemaMapper

> **The alias dictionary is highly recommended.** The Stage-1 `dict`
> step matches your column names against curated aliases (known
> synonyms, abbreviations, and naming variants for each target field),
> and most high-confidence hits come from this step rather than the
> fuzzy/value/semantic fallbacks. The bundled schema for GDC ships with
> an LLM-generated (Haiku 4.5) alias dictionary, so the default path is
> already covered. If you map to your own field set via
> `target_schema_path=`, the bundled aliases are disabled — generate a
> matching one with `generate_alias_dict()` (needs `ANTHROPIC_API_KEY`
> or `GEMINI_API_KEY`) and pass it as `alias_dict_path=`. Skipping this
> for a custom schema measurably degrades accuracy. See [input_formats
> §2.5](docs/input_formats.md#25-the-alias-dictionary-bundled-llm-generated).

``` python
from metaharmonizer import SchemaMapEngine

engine = SchemaMapEngine(
    input_path=YOUR_QUERY_FILE,
)

# Run Stage 1, 2 & 3 (and 4 if mode="auto")
engine.run_schema_mapping()

# (Optional) Run Stage 4 after manual review
engine.run_llm_on_file(
    input_csv="path_to_stage3_results.csv",
    output_csv="path_to_stage3_results_with_stage4.csv",
    stage_filter=["stage3"],
    merge_results=True,
)
```

**Parameters:**

| Parameter | Type | Description |
|-------------|-------------|----------------------------------------------|
| `input_path` | str | Path to clinical dataset (TSV or CSV). |
| `mode` | str | `"auto"` → automatically proceed to Stage 4 if Stage 3 confidence is low. `"manual"` (Default) → output Stage 3 results for review; Stage 4 must be triggered manually. |
| `top_k` | int | Number of top matches returned for each column. |

**Output:**

| Aspect | Detail |
|---------------------|---------------------------------------------------|
| Location | CSV file saved to `SM_OUTPUT_DIR` (see [Environment variables](#2-environment-variables)). |
| Filename (manual mode) | `<input_root>_s3_<field_model_short>_<mode>_<YYYYMMDD_HHMMSS>.csv` |
| Filename (auto mode) | `<input_root>_s3_<field_model_short>_s4_<llm_model_short>_<mode>_<YYYYMMDD_HHMMSS>.csv` |
| Filename (manual Stage 4) | When Stage 4 is run manually via `run_llm_on_file(...)`, `output_csv` controls the filename and location. |
| Columns | `query`, `stage` (stage1/stage2/stage3), `method` (dict, fuzzy, numeric, alias, bert, freq), and `match{i}`, `match{i}_score`, `match{i}_source` for the top-k matches. |

## 6. OntologyMapper

**Using a non-NCIt ontology (e.g. MONDO):**

``` python
from metaharmonizer import OntoMapEngine

engine = OntoMapEngine(
    corpus_category="disease",
    query_ls=query_list,
    ontology_source="mondo",  # uses EBI OLS4 API
)
results = engine.run()
```

**Custom corpus (advanced):**

``` python
import pandas as pd
from metaharmonizer import OntoMapEngine

# Provide your own corpus_df — ontology_source is inferred from code prefixes.
# A content hash isolates user tables from the official ones, so different
# corpora never cross-contaminate each other or the built-in tables.
my_corpus = pd.read_csv("my_custom_corpus.csv")  # must have 'label' and 'obo_id' columns
engine = OntoMapEngine(
    corpus_category="disease",
    query_ls=query_list,
    corpus_df=my_corpus,
    output_dir="data/outputs/my_run",  # optional: auto-save results here
)
results = engine.run()
```

**Parameters:**

| Parameter | Type (default) | Description |
|-----------------|-----------------|---------------------------------------|
| `category` | str | Ontology category — `disease`, `bodysite`, `treatment`, or `phenotype`. |
| `query` | list | List of query terms to map. |
| `query_df` | DataFrame (optional) | DataFrame query mode (alternative to `query`); requires `query_col`. |
| `query_col` | str (optional) | Column in `query_df` holding the query terms. |
| `ground_truth_map` | dict | Mapping of query terms to curated ontology values (for evaluation in `test` mode). |
| `corpus` | list (optional) | Explicit list of corpus terms for **Stage 2 matching only** (Stage 3 always uses `corpus_df`). Auto-derived from `corpus_df` when omitted. |
| `corpus_df` | DataFrame (optional) | DataFrame with `label` and `obo_id` columns. Auto-loaded from cached CSV or built from API when omitted. |
| `ontology_source` | str (`"ncit"`) | Ontology backend. Supported `(category, ontology_source)` pairs: `disease`/`bodysite`/`treatment` → `ncit` (NCI Thesaurus via EVSREST); `disease` → `mondo`, `bodysite` → `uberon` (via EBI OLS4 API); `phenotype` → `efo` (pre-built static corpus, no API key needed). When `corpus_df` is provided, this is inferred from code prefixes. |
| `s2_strategy` | str | Stage 2 strategy — `lm` (CLS-token pooling) or `st` (SentenceTransformer mean pooling). |
| `s2_method` | str | Transformer model key from `method_model.yaml` (e.g. `sap-bert`, `pubmed-bert`). |
| `s3_strategy` | str (optional) | Stage 3 strategy — `rag`, `rag_bie`, or `None` to disable. |
| `top_k` | int (5) | Number of top matches per query. |
| `output_dir` | str (optional) | Directory to auto-save result CSV. Filename pattern: `om_{ontology_source}_{category}_s2_{strategy}_{method}_{timestamp}.csv`. |
| `persist_corpus` | bool (`False`) | When `True` and `corpus_df` is caller-provided, persist it to the cache CSV. |

**Pipeline stages:**

| Stage | Description |
|-----------------|-------------------------------------------------------|
| Stage 1 | Exact matching against corpus. |
| Stage 2 | Embedding-based similarity (LM or ST strategy). |
| Stage 2.5 | Synonym verification — boosts low-confidence Stage 2 matches using synonym data from concept tables. |
| Stage 3 (optional) | RAG-based re-matching with retrieved context from the knowledge database. |

**Output:** DataFrame with top-k matches, scores, and match levels for
each query term.

## 7. Example notebooks

Demonstration notebooks for the ontology and schema mappers live under
[`examples/`](examples/). See [examples/README.md](examples/README.md)
for an overview of each notebook and its required inputs.

## 8. Frequently asked questions

### Prepare inputs

A deeper, example-driven walk-through of every input and output lives in
[docs/input_formats.md](docs/input_formats.md); the answers below are
the short version.

#### 🤔 What are the minimum inputs for SchemaMapper?

Just one: a path to your clinical data file (CSV or TSV) whose **column
names** are what get mapped. Everything else — the target schema and the
alias dictionary — ships bundled inside the wheel, so `mode="manual"`
runs with no keys and no extra files.

``` python
SchemaMapEngine(input_path="examples/data/ucec_cptac_2020_before_harmonization.csv").run_schema_mapping()
```

#### 🤔 How to prepare my inputs for SchemaMapper?

The column *names* are the input; values are sampled to help value-based
and numeric matching, so a few real rows is plenty. To map to your own
field set instead of the bundled one, pass `target_schema_path=` — and
usually a matching `alias_dict_path=` too, since a custom schema
disables the bundled alias dictionary. See [input_formats
§2.1–2.3](docs/input_formats.md#21-the-clinical-data-file-required) and
[§2.5](docs/input_formats.md#25-the-alias-dictionary-bundled-llm-generated).

#### 🤔 What are the minimum inputs for OntologyMapper?

Just a list of query terms. That alone runs:

``` python
OntoMapEngine(corpus_category="disease", query_ls=["TNBC"]).run()
```

Pass a `ground_truth_map` (term → known-correct label) and the engine
switches to `test` mode automatically, scoring accuracy via
`match_level`. You can still force the mode with `test_or_prod=` if
needed. The corpus is **optional** — the engine resolves it from cache
or the source API. NCIt corpus/concept-table builds need `UMLS_API_KEY`;
`mondo`/`uberon` (EBI OLS4) need no key.

#### 🤔 How to prepare my inputs for OntologyMapper?

Supply queries either as a plain list (`query=`) or as a DataFrame +
column (`query_df=`, `query_col=` — required for the `rag_bie` Stage-3
strategy). The query column is de-duplicated, stripped, and blank/`nan`
values dropped automatically. To bring your own corpus, pass
`corpus_df=` with a label column (`official_label` or `label`) and a
code column (`clean_code` or `obo_id`); `ontology_source` is inferred
from the code prefixes. See [input_formats
§1.1–1.3](docs/input_formats.md#11-the-query-required).

### Interpret outputs

#### 🤔 How to interpret my outputs from SchemaMapper?

`run_schema_mapping()` returns a DataFrame with one row per **input
column** (and writes a CSV under `SM_OUTPUT_DIR`). Key columns: `query`
(original column name), `stage` (`stage1`–`stage4`), `method` (how it
matched — `std_exact`, `std_fuzzy`, `value`, `numeric`, `semantic`,
`llm`, …), and `match{i}` / `match{i}_score` / `match{i}_source` for the
top-k mapped field names. See [§5](#5-schemamapper) and [input_formats
§2.4](docs/input_formats.md#24-output).

#### 🤔 How to interpret my outputs from OntologyMapper?

`run()` returns a DataFrame, one row per query term: `query`, the top-k
`match{i}` / `match{i}_score` candidates (best first), `stage` (1 exact,
2 embedding, 2.5 synonym, 3 RAG, 4 LLM), plus `match_level` and
`ref_match` — the latter two are **meaningful in `test` mode only**
(`ref_match` is `"Not Found"` in `prod`; ignore it there). Set
`output_dir=` to also write a timestamped CSV. See [input_formats
§1.5](docs/input_formats.md#15-output).

#### 🤔 Do I need LLM? If so, where and how?

No, for the default paths. Both engines run their core stages with **no
LLM key**: SchemaMapper `mode="manual"` (Stages 1–3) and OntologyMapper
with `s4_strategy=None` (the default). An LLM is only needed for the
**Stage 4** refinement:

-   **SchemaMapper `mode="auto"`** and **OntologyMapper
    `s4_strategy="llm"`** call the Stage-4 LLM (set `GEMINI_API_KEY`).
-   **Generating an alias dictionary** for a custom schema via
    `generate_alias_dict()` needs `ANTHROPIC_API_KEY` or
    `GEMINI_API_KEY`.

Note that `UMLS_API_KEY` (for NCIt) is *not* an LLM key — it's a
vocabulary API. See [input_formats
§3](docs/input_formats.md#3-which-environment-variable-gates-which-stage)
for which variable gates which stage.

### Choosing an engine and mode

#### 🤔 Which engine do I use — OntologyMapper or SchemaMapper?

Pick by *what* you are aligning. If you have free-text **values** (e.g.
`"SQUAMOUS CELL CARCINOMA, PHARYNX"`) and want the matching ontology
term and code, use **`OntoMapEngine`**. If you have a data file whose
**column names** are non-standard and want them mapped to your standard
field names, use **`SchemaMapEngine`**. See the [engine-picker
table](docs/input_formats.md#preparing-inputs-for-metaharmonizer) in
input_formats.

#### 🤔 What is the difference between `test` and `prod` mode?

`prod` is for mapping real, **unlabelled** data — you supply only the
terms and the engine fills in the answers; no ground truth needed.
`test` is for **measuring accuracy** when you already know the correct
answer for each term: you pass a `ground_truth_map` (term →
known-correct label), and the output adds `match_level` / `ref_match` so
you can score how often the right label was recovered. (SchemaMapper has
the analogous `manual`/`auto` mode split — see the LLM question above.)

### Performance and caching

#### 🤔 Why is the first OntologyMapper run so slow, and later runs fast?

The first run for a given NCIt corpus does the one-time expensive work:
it builds concept tables from the NCI API (minutes for the full
\~14k-term disease corpus, seconds for a small slice), downloads the
sap-bert encoder (\~440 MB), then builds a FAISS index (\~4 min for the
full corpus). All of it is cached, so subsequent runs reuse the index
and take \~7 sec. `SchemaMapEngine` similarly downloads
`all-MiniLM-L6-v2` once on first use.

#### 🤔 Where does MetaHarmonizer store its caches and models?

Under `~/.metaharmonizer/` by default: Hugging Face encoders in
`model_cache/`, FAISS indexes and the vector DB in `KnowledgeDb/`, and
corpus/schema reference files in `data/`. Each location is overridable
via an environment variable (`MODEL_CACHE_ROOT`, `KNOWLEDGE_DB_DIR`,
`FAISS_INDEX_DIR`, `METAHARMONIZER_DATA_DIR`); see
[§2](#2-environment-variables).

#### 🤔 Where are my output CSVs written?

SchemaMapper writes to `SM_OUTPUT_DIR` (default
`$METAHARMONIZER_DATA_DIR/schema_mapping_eval`). OntologyMapper only
writes a CSV when you pass `output_dir=`. Both use timestamped filename
patterns documented in [§5](#5-schemamapper) and
[§6](#6-ontologymapper).
