### 1. Codebase folder structure

```md

├── data
├── demo_nb
├── scripts
├── EDA
├── evaluation
├── src
│   ├── models
│   │   ├── init.py
│   │   ├── ontology_models.py
│   │   ├── ontology_mapper_rag.py
│   │   ├── ontology_mapper_lm.py
│   │   ├── ontology_mapper_st.py
│   │   ├── ontology_mapper_bi_encoder.py
│   │   ├── ontology_mapper_fts.py
│   │   ├── ontology_mapper_synonym.py
│   │   ├── reranker.py
│   │   ├── method_model.yaml
│   │   ├── schema_mapper
│   │   │   ├── engine.py
│   │   │   ├── config.py
│   │   │   ├── loaders
│   │   │   └── matchers
│   ├── Engine
│   │   ├── ontology_mapping_engine.py
│   ├── CustomLogger
│   ├── KnowledgeDb
│   │   ├── faiss_sqlite_pipeline.py
│   │   ├── corpus_builder.py
│   │   ├── concept_table_builder.py
│   │   ├── synonym_dict.py
│   │   ├── fts_synonym_db.py
│   │   └── db_clients
│   │       ├── nci_db.py
│   │       ├── ols_db.py
│   │       └── umls_db.py
│   ├── _paths.py
│   ├── _async_utils.py
│   ├── utils
│   ├── Plotter
├── pyproject.toml
└── readme.md
```

### 2. Usage

In order to use schema and/or ontology mapping functionality in metaharmonizer, please follow the steps below.

#### 2.1. Installation

```bash
# 1. Clone
git clone https://github.com/shbrief/MetaHarmonizer
cd MetaHarmonizer

# 2. Create a Python 3.10 environment
conda create -n mh python=3.10 -y
conda activate mh
pip install --upgrade pip

# 3. Install the package. Pick one of:
pip install -e .                       # core only (no LLM backends)
pip install -e ".[llm-gemini]"         # + Gemini (Stage-4 LLM, OntoMapLLM, etc.)
pip install -e ".[llm-openai]"         # + OpenAI (FieldSuggester semantic clustering)
pip install -e ".[notebook]"           # + nest-asyncio for Jupyter workflows
pip install -e ".[dev]"                # + pytest & coverage
pip install -e ".[eval]"               # + scipy for evaluation scripts
pip install -e ".[all]"                # notebook + eval + both LLM backends
```

Install directly from GitHub (non-editable):
```bash
pip install "git+https://github.com/shbrief/MetaHarmonizer#egg=metaharmonizer[llm-gemini]"
```

> The `data/` corpus directory is **not bundled** in the wheel. Installed users
> should either clone the repo alongside the install, or set
> `METAHARMONIZER_DATA_DIR` (see [2.2](#22-environment-variables)) to a local
> copy.

#### 2.2. Environment variables

Copy `.env.example` → `.env` (or export in your shell) before running the
mappers. `python-dotenv` auto-loads `.env` on import.

| Variable | Required for | Default | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | Stage-4 LLM matcher, `OntoMapLLM`, bi-encoder column inference, `expand_terms_with_gemini` | — | Needed whenever the `llm-gemini` extra is exercised. |
| `UMLS_API_KEY` | NCI Thesaurus lookups, concept-table builder, `update_term_via_code` | — | Required for `ontology_source="ncit"` pipeline stages that hit the live API. |
| `METAHARMONIZER_DATA_DIR` | Locating corpus + schema reference files (`oncotree_code_to_name.csv`, `curated_fields.csv`, etc.) | `~/.metaharmonizer/data` | Dev tip: point at `<repo>/data` in your `.env` to reuse the repo's bundled examples. Small reference files ship inside the wheel as a fallback when this dir is empty. |
| `SM_OUTPUT_DIR` | `SchemaMapEngine` output path | `$METAHARMONIZER_DATA_DIR/schema_mapping_eval` | Overrides where CSV results are written. |
| `FIELD_VALUE_JSON` | Schema-mapper value dictionary | `$METAHARMONIZER_DATA_DIR/schema/field_value_dict.json` | Point at an alternative value dict. |
| `VECTOR_DB_PATH` | Knowledge-DB SQLite file | `$KNOWLEDGE_DB_DIR/vector_db.sqlite` | — |
| `FAISS_INDEX_DIR` | FAISS index cache | `$KNOWLEDGE_DB_DIR/faiss_indexes` | — |
| `KNOWLEDGE_DB_DIR` | Root dir override for KnowledgeDb assets | `~/.metaharmonizer/KnowledgeDb` | — |
| `METHOD_MODEL_YAML` | Method→model registry | bundled `metaharmonizer/models/method_model.yaml` | — |
| `MODEL_CACHE_ROOT` / `MODEL_CACHE_DIR` | Hugging Face model cache | `~/.metaharmonizer/model_cache` | `MODEL_CACHE_ROOT` takes precedence; `MODEL_CACHE_DIR` is a fallback. |
| `FIELD_MODEL` | Sentence-transformer for `FieldSuggester` | `all-MiniLM-L6-v2` | — |
| `NCIT_POOL_SIZE` | NCI async client connection pool | `8` | Raise for bulk corpus builds. |
| `LOG_FILE` / `LOG_ENV` | Logger config | `out.log` / `development` | — |

#### 2.3. Quickstart

The snippets below assume you cloned the repo (so `data/` is on disk and the
cached ontology corpus under `data/corpus/retrieved_ontologies/` is available).
For wheel-only installs:
- replace the input paths with your own files (or point
  `METAHARMONIZER_DATA_DIR` at a local copy of `data/`),
- and pass `corpus_df=` to `OntoMapEngine` or set `UMLS_API_KEY` so the engine
  can fetch the ontology on first use. A small reference dataset
  (`schema/curated_fields.csv`, `corpus/oncotree_code_to_name.csv`) is bundled
  inside the wheel for `SchemaMapEngine` and OncoTree lookups, but **the full
  ontology corpus is not**.

Minimal `OntoMapEngine` example (Stage 1–2; no API key needed when the cached
NCIT corpus is present locally):

```python
import pandas as pd
from metaharmonizer import OntoMapEngine

df = pd.read_csv("data/corpus/cbio_disease/disease_query_updated.csv")
engine = OntoMapEngine(
    category="disease",
    query=df["original_value"].tolist(),
    cura_map=dict(zip(df["original_value"], df["curated_ontology"])),
    s2_method="sap-bert",
    s2_strategy="st",
    test_or_prod="test",
)
results = engine.run()
print(results.head())
```

Schema mapper in one call:

```python
from metaharmonizer import SchemaMapEngine

engine = SchemaMapEngine(
    clinical_data_path="data/schema/test.csv",
    mode="manual",   # "auto" to auto-run Stage-4 LLM on low-confidence rows
    top_k=5,
)
engine.run_schema_mapping()
```

Richer examples (custom corpus, MONDO/UBERON sources, Stage-4 LLM review) live
in [2.5 Setting up the mappers](#25-setting-up-the-mappers) and the notebooks
under `demo_nb/`.

#### 2.4 Datasets
- The datasets in this repository are encrypted to prevent contamination of the gold standard.  
- For **ontology mapping**, you must provide:
  - A list of `query_terms`  
  - A `corpus` list and/or `corpus_df` are **optional** — the engine auto-resolves them from cached CSV or API when not provided  
- For **schema mapping**, provide a clinical metadata file.  
  - The schema mapping dictionary is available in the `/data` folder.  
- ⚠️ You will not be able to use the encrypted demo datasets without authorization, but you can supply your own query and corpus lists.
  
#### 2.5 Setting up the mappers 

1. Ontology Mapping

**Minimal example (auto-resolved corpus):**

```python
%cd <user_path>/MetaHarmonizer/

import pandas as pd
from metaharmonizer.Engine import get_ontology_engine

OntoMapEngine = get_ontology_engine()

df = pd.read_csv("data/corpus/cbio_disease/disease_query_updated.csv")
query_list = df["original_value"].tolist()
cura_map = dict(zip(df["original_value"], df["curated_ontology"]))

# corpus is auto-loaded from cached CSV or built from NCI/OLS API
engine = OntoMapEngine(
    category="disease",
    query=query_list,
    cura_map=cura_map,
    s2_method="sap-bert",
    s2_strategy="st",
    test_or_prod="test",
)
results = engine.run()
```

**Using a non-NCIt ontology (e.g. MONDO):**

```python
engine = OntoMapEngine(
    category="disease",
    query=query_list,
    cura_map=cura_map,
    s2_method="sap-bert",
    s2_strategy="st",
    s3_strategy="rag",
    test_or_prod="test",
    ontology_source="mondo",  # uses EBI OLS4 API
)
results = engine.run()
```

**Custom corpus (advanced):**

```python
# Provide your own corpus_df — ontology_source is inferred from code prefixes.
# A content hash isolates user tables from the official ones, so different
# corpora never cross-contaminate each other or the built-in tables.
my_corpus = pd.read_csv("my_custom_corpus.csv")  # must have 'label' and 'obo_id' columns
engine = OntoMapEngine(
    category="disease",
    query=query_list,
    cura_map=cura_map,
    s2_strategy="lm",
    test_or_prod="test",
    corpus_df=my_corpus,
    output_dir="data/outputs/my_run",  # optional: auto-save results to this directory
)
results = engine.run()
```

For more examples, see `demo_nb/ontology_mapper_workflow.ipynb`.

**Offline corpus building (OLS ontologies):**

```bash
python scripts/build_ols_corpus.py \
    --term UBERON:0001062 --category bodysite --ontology uberon \
    --include-hierarchy
```

This fetches the full ontology tree once and caches it locally so subsequent pipeline runs require no API calls.

- **Parameters:**
  - **category** (str): Ontology category — `disease`, `bodysite`, or `treatment`.
  - **query** (list): List of query terms to map.
  - **cura_map** (dict): Mapping of query terms to curated ontology values (for evaluation in `test` mode).
  - **corpus** (list, optional): Explicit list of corpus terms for Stage 2 matching. Auto-derived from `corpus_df` when omitted.
  - **corpus_df** (DataFrame, optional): DataFrame with `label` and `obo_id` columns. Auto-loaded from cached CSV or built from API when omitted.
  - **ontology_source** (str, default `"ncit"`): Ontology backend. Supported: `ncit` (NCI Thesaurus via EVSREST), `mondo`, `uberon` (via EBI OLS4 API). When `corpus_df` is provided, this is inferred from code prefixes.
  - **s2_strategy** (str): Stage 2 strategy — `lm` (CLS-token pooling) or `st` (SentenceTransformer mean pooling).
  - **s2_method** (str): Transformer model key from `method_model.yaml` (e.g. `sap-bert`, `pubmed-bert`).
  - **s3_strategy** (str, optional): Stage 3 strategy — `rag`, `rag_bie`, or `None` to disable.
  - **topk** (int, default 5): Number of top matches per query.
  - **test_or_prod** (str): `test` includes curated_ontology in output for evaluation; `prod` omits it.
  - **output_dir** (str, optional): Directory to auto-save result CSV. Filename pattern: `om_{ontology_source}_{category}_s2_{strategy}_{method}_{timestamp}.csv`.
  - **persist_corpus** (bool, default `False`): When `True` and `corpus_df` is caller-provided, persist it to the cache CSV.

- **Pipeline stages:**
  - **Stage 1:** Exact matching against corpus.
  - **Stage 2:** Embedding-based similarity (LM or ST strategy).
  - **Stage 2.5:** Synonym verification — boosts low-confidence Stage 2 matches using synonym data from concept tables.
  - **Stage 3** (optional): RAG-based re-matching with retrieved context from knowledge database.

- **Output:** DataFrame with top-k matches, scores, and match levels for each query term.

2. Schema mapping
```python
from metaharmonizer.models.schema_mapper import SchemaMapEngine

# Initialize the engine
engine = SchemaMapEngine(
    clinical_data_path=YOUR_QUERY_FILE,
    mode="manual",   # Options: "auto" or "manual"
    top_k=5,
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

- Parameters that can be changed in the model:
  - clinical_data_path (str): Path to clinical dataset (TSV or CSV).
  - mode (str):
    - "auto" → automatically proceed to Stage 4 if Stage 3 confidence is low
    - "manual" → output Stage 3 results for review; Stage 4 must be triggered manually
  - top_k (int): Number of top matches returned for each column.

- Output  
  - CSV File: Results saved to OUTPUT_DIR (configured in metaharmonizer/models/schema_mapper/config.py). Filename pattern from `run_schema_mapping()` is:  
`<input_root>_s3_<field_model_short>_<mode>_<YYYYMMDD_HHMMSS>.csv` (manual mode)  
`<input_root>_s3_<field_model_short>_s4_<llm_model_short>_<mode>_<YYYYMMDD_HHMMSS>.csv` (auto mode)  
  - If Stage 4 is run manually via `run_llm_on_file(...)`, use `output_csv` to control the output filename and location.  
  - Columns:  
query  
stage (stage1, stage2, stage3  )  
method (dict, fuzzy, numeric, alias, bert, freq)  
match{i}, match{i}_score, match{i}_source (for top-k matches)

#### 2.6. Demo Notebooks For Schema and Ontology Mapping

The demo notebooks are located across `/demo_nb` folder

### 3. Resources
| Topic | Links | Resource Type |
|----------|----------|----------|
| Review paper on all pretrained biomedical BERT models | [Link](https://www.sciencedirect.com/science/article/pii/S1532046421003117) | paper |
| Review of deep learning approaches for biomedical entity recognition | [Link](https://academic.oup.com/bib/article/22/6/bbab282/6326536?login=false) | paper |
| Comprehensive Review of pre-trained foundation models | [Link](https://arxiv.org/pdf/2302.09419) | paper |
| KERMIT Knowledge graphs | [Link](https://arxiv.org/pdf/2204.13931) | paper |
| LLMs4OM (Uses RAG Framework for matching concepts)| [Link](https://arxiv.org/pdf/2404.10317v1) | paper |
| DeepOnto | [Link](https://arxiv.org/html/2307.03067v2) | computational_tool |
| Text2Onto | [Link](https://github.com/krishnanlab/txt2onto) | computational_tool |
| SapBert | [Link](https://aclanthology.org/2021.naacl-main.334/) | computational_tool |
| Ontology mapping with LLM’s | [Link](https://dl.acm.org/doi/fullHtml/10.1145/3587259.3627571) | computational_tool |
| Exploring LLM’s for ontology alignment | [Link](https://arxiv.org/pdf/2309.07172) | computational_tool |
| Ontology alignment evaluation initiative | [Link](https://ceur-ws.org/Vol-3324/oaei22_paper0.pdf) | dataset |
| Commonly used dataset for benchmarking of new methods | [Link](https://github.com/chanzuckerberg/MedMentions) | dataset |
| NCIT Ontologies | [Link](https://www.ebi.ac.uk/ols4/ontologies/ncit) | dataset |
| ML Friendly datasets for equivalence and subsumption mapping | [Link](https://arxiv.org/pdf/2205.03447) | dataset |
| Positive and Negative Sampling Strategies for Representation Learning in Semantic Search | [Link](https://blog.reachsumit.com/posts/2023/03/pairing-for-representation/) | blog |
| How to train sentence transformers | [Link](https://huggingface.co/blog/how-to-train-sentence-transformers) | blog |


---

### R Interface

R users can access MetaHarmonizer via the
[**MetaHarmonizerR**](https://github.com/shbrief/MetaHarmonizerR) package, which
wraps the Python engines using `{reticulate}`.

```bash
# Install the Python backend first
pip install git+https://github.com/shbrief/MetaHarmonizer
```

```r
# Install the R wrapper
remotes::install_github("shbrief/MetaHarmonizerR")
library(MetaHarmonizerR)

init_field_suggester()
suggestions <- suggest_fields(unmapped_columns = c("age_at_diagnosis", "tumor_size"))
```
