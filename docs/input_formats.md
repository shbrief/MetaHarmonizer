---
editor_options: 
  markdown: 
    wrap: 72
---

# Preparing inputs for MetaHarmonizer

This is a practical reference for the **inputs each engine expects** and
the **shape of the output** it returns. If you just want to run
something end to end, start with
[`examples/quickstart.ipynb`](../examples/quickstart.ipynb); come back
here when you need to wire in your own data.

MetaHarmonizer has two engines. Pick by what you are aligning:

| You have… | You want… | Engine |
|----|----|----|
| Free-text *values* (e.g. `"SQUAMOUS CELL CARCINOMA, PHARYNX"`) | The matching ontology term + code | `OntoMapEngine` |
| A data file whose *column names* are non-standard | Those columns mapped to your standard field names | `SchemaMapEngine` |

Throughout, **`test` vs `prod` mode** is the distinction that trips up
most first-time users:

-   **`prod`** — you are mapping real, unlabelled data. You provide
    *only* the terms; the engine fills in the answers. No ground truth
    needed.
-   **`test`** — you already know the correct answer for each term and
    want to *measure accuracy*. You must pass a `ground_truth_map` (term
    → known-correct label).

------------------------------------------------------------------------

## 1. OntologyMapper (`OntoMapEngine`)

### 1.1 The query (required)

The terms you want to map. Two equivalent ways to supply them:

**A. A plain Python list** — simplest:

``` python
query = [
    "SQUAMOUS CELL CARCINOMA, PHARYNX",
    "astrocytoma grade III-IV",
    "TNBC",
]
```

**B. A DataFrame + column name** — when your terms already live in a
CSV. This is also required for the `rag_bie` Stage-3 strategy, which
reads extra context columns from the frame:

``` python
import pandas as pd
df = pd.read_csv("my_terms.csv")
# pass query_df=df, query_col="original_value"   (no `query=` needed)
```

The query column is de-duplicated and stripped automatically; blank /
`"nan"` / `"None"` strings are dropped.

### 1.2 `ground_truth_map` — **test mode only**

A `dict` mapping each query term to its known-correct ontology label,
used to score accuracy. **Required in `test` mode, ignored in `prod`
mode** (where it is auto-filled with `"Not Found"`).

``` python
ground_truth_map = {
    "SQUAMOUS CELL CARCINOMA, PHARYNX": "Pharyngeal Squamous Cell Carcinoma",
    "astrocytoma grade III-IV":         "High Grade Astrocytic Tumor",
}
```

A convenient way to build it from a labelled CSV:

``` python
ground_truth_map = dict(zip(df["original_value"], df["curated_ontology"]))
```

### 1.3 The corpus (optional)

The reference vocabulary the queries are matched against. **You usually
do not provide this** — omit it and the engine resolves the corpus for
the `(category, ontology_source)` pair, either from a cached CSV or by
fetching it from the source API on first run (NCIt needs `UMLS_API_KEY`;
see §3).

> For *how the engine decides and resolves* the corpus per attribute —
> the registry, the two corpus kinds, validation, known limits — see the
> companion explanation [`corpus_selection.md`](corpus_selection.md).
> This section covers the task side: what to supply, and how to create
> one.

Supported built-in combinations:

| `category`  | `ontology_source` | Backend / source       |
|-------------|-------------------|------------------------|
| `disease`   | `ncit` (default)  | NCI EVSREST            |
| `bodysite`  | `ncit`            | NCI EVSREST            |
| `treatment` | `ncit`            | NCI EVSREST            |
| `disease`   | `mondo`           | EBI OLS4               |
| `bodysite`  | `uberon`          | EBI OLS4               |
| `phenotype` | `efo`             | static CSV — see below |

The `phenotype`/`efo` corpus is a **static, shipped** corpus (a merged
EFO corpus, 12 ontologies flattened under the EFO namespace). It is
*not* rebuildable from an API: it lives in the repo's `data/` tree and
must be copied into your data cache once before first use —

``` bash
python -m metaharmonizer.scripts.bootstrap_data        # copy missing files
```

(or set `METAHARMONIZER_DATA_DIR` to the repo's `data/` dir; see §3). A
cache miss for a static corpus raises a `FileNotFoundError` pointing you
at exactly this step.

#### Creating / supplying your own corpus (`corpus_df=`)

Pass `corpus_df=` to override resolution entirely. The frame needs, at
minimum, a **label column** (`official_label` *or* `label`) and a **code
column** (`clean_code` *or* `obo_id`). `ontology_source` is then
*inferred* from the code prefixes, so **all codes must share one
ontology**.

``` text
label,obo_id
"Neoplasm by Site",NCIT:C3263
"Genitourinary System Neoplasm",NCIT:C156482
"Pharyngeal Squamous Cell Carcinoma",NCIT:C102872
```

Three ways to produce that frame:

1.  **Hand-authored CSV** — write the two columns yourself (as above)
    and `pd.read_csv(...)` it. Best for a small, curated candidate set.
2.  **Built from an ontology root** — use `CorpusBuilder` to collect all
    descendants of a root term from OLS4, then load the result. This is
    the same machinery the engine uses internally for OLS-backed
    categories; see
    [`knowledge_db/readme.md`](../src/metaharmonizer/knowledge_db/readme.md)
    ("CorpusBuilder Quickstart") for the Python API and the
    `scripts/build_corpus.py` CLI.
3.  **Reuse a shipped/static corpus** — run
    `python -m metaharmonizer.scripts.bootstrap_data` (above) and read
    the resulting CSV from the cache, or point at it directly.

Notes:

-   `obo_id` codes are normalized automatically:
    `NCIT:C156482 → C156482`, `UBERON:0001062 → UBERON_0001062`.
-   A content hash isolates your corpus's cache tables from the official
    ones, so custom and built-in corpora never cross-contaminate.
-   Pass `persist_corpus=True` to save your corpus to the canonical
    cache path for reuse.

A ready-to-use sample lives at
[`examples/data/disease_corpus_updated.csv`](../examples/data/disease_corpus_updated.csv).

### 1.4 Minimal calls

``` python
from metaharmonizer import OntoMapEngine

# PROD: map your own terms, no ground truth (prod mode is inferred)
engine = OntoMapEngine(
    category="disease",
    query=["TNBC", "astrocytoma grade III-IV"],
)
results = engine.run()

# TEST: measure accuracy against known labels
# (passing ground_truth_map switches on test mode automatically)
engine = OntoMapEngine(
    category="disease",
    query=list(ground_truth_map),
    ground_truth_map=ground_truth_map,
)
results = engine.run()
```

### 1.5 Output

`run()` returns a `DataFrame`, one row per query term:

| Column | Meaning |
|----|----|
| `query` | The original input term |
| `match1` … `match{k}` | Top-k candidate ontology labels, best first |
| `match1_score` … | Similarity score for each candidate |
| `stage` | Which stage produced the row (1 exact, 2 embedding, 2.5 synonym, 3 RAG, 4 LLM) |
| `match_level` | Rank at which the *correct* label appeared — **meaningful in `test` mode only** |
| `ref_match` | The `ground_truth_map` label — **`"Not Found"` in `prod` mode; ignore it there** |

Set `output_dir=` to also write a timestamped CSV
(`om_{ontology_source}_{category}_s2_{strategy}_{method}_{timestamp}.csv`).

> Exact string matches are short-circuited at Stage 1. If your terms are
> already canonical you will see `stage == 1`; messy real-world strings
> flow into the embedding stages, which is the interesting path.

------------------------------------------------------------------------

## 2. SchemaMapper (`SchemaMapEngine`)

### 2.1 The clinical data file (required)

A path to a **CSV or TSV** (`.tsv` is read tab-separated; anything else
is read comma-separated). **Column names are the input** — they are what
gets mapped to your standard fields. Values are sampled to support
value-based and numeric-field matching, so real data helps but a few
rows is enough.

``` text
patientId,CANCER_TYPE,AGE,SEX,RACE,ETHNICITY,TUMOR_SITE
C3L-00006,Endometrial Carcinoma,64,Female,White,Not-Hispanic or Latino,Anterior endometrium
C3L-00008,Endometrial Carcinoma,58,Female,White,Not-Hispanic or Latino,Posterior endometrium
```

A ready-to-use sample lives at
[`examples/data/ucec_cptac_2020_before_harmonization.csv`](../examples/data/ucec_cptac_2020_before_harmonization.csv).

### 2.2 Mode

-   **`manual`** — runs Stages 1–3 and stops, emitting candidates for
    review. **Needs no LLM key** and no network for the LLM. Good
    default for a first run.
-   **`auto`** (alpha stage) — additionally calls the Stage-4 LLM when
    Stage-3 confidence is low. **Requires a Gemini key**
    (`GEMINI_API_KEY`); see §3.

### 2.3 The standard schema (bundled; override optional)

By default the engine maps to the **bundled** curated field list
(`_bundled_data/schema/cbio_target_attrs.csv` — columns `field_name`,
`is_numeric_field`). To map to *your own* field set, pass
`curated_dict_path=`. When you do, the bundled alias dictionary is
disabled (it is keyed to the default schema) unless you also pass
`alias_dict_path=`, and the value dictionary is filtered to your fields.

``` python
engine = SchemaMapEngine(
    clinical_data_path="my_clinical.csv",
    mode="manual",
    top_k=5,
    curated_dict_path="my_fields.csv",   # optional: your own target schema
)
```

### 2.4 Output

`run_schema_mapping()` returns a `DataFrame`, one row per **input
column**, and also writes a CSV under `SM_OUTPUT_DIR`:

| Column | Meaning |
|----|----|
| `query` | The original column name |
| `stage` | `stage1`–`stage4` that produced the match |
| `method` | `std_exact`, `std_fuzzy`, `value`, `ontology`, `numeric`, `semantic`, `llm`, … |
| `match{i}` / `match{i}_score` / `match{i}_source` | Top-k mapped field names, scores, and where each came from |

### 2.5 The alias dictionary (bundled; LLM-generated)

Stage 1 matches incoming column names against an **alias dictionary** —
a many-to-one mapping from real-world column names to standard fields —
before any embedding work. It is the single biggest lever on
SchemaMapper accuracy, so when you map to your own schema
(`curated_dict_path=`) you will usually want to supply a matching alias
dictionary too.

A **CSV with exactly three columns**; these are the only columns the
loader reads:

| Column | Meaning |
|----|----|
| `field_name` | the standard field this alias maps to (must exist in the curated schema) |
| `source` | the alias — a raw column name that should map to `field_name` |
| `is_numeric_field` | `"yes"` for numeric-valued fields, empty otherwise |

``` text
field_name,source,is_numeric_field
age_at_diagnosis,AGE_D,yes
age_at_diagnosis,AGE_AT_DX,yes
sex,GENDER,
```

The bundled default lives at
`_bundled_data/schema/cbio_target_attrs_alias_manual.csv` and is keyed
to the bundled `cbio_target_attrs.csv`. Pass your own with
`alias_dict_path=`, or disable alias matching entirely with
`alias_dict_path=""`. Note the interaction in §2.3: a custom
`curated_dict_path=` disables the bundled alias dictionary unless you
also pass `alias_dict_path=`.

**Generating one.** `generate_alias_dict()` expands a set of standard
field names into this CSV using an LLM (Anthropic or Gemini,
auto-detected from the model id):

``` python
import pandas as pd
from metaharmonizer.models.schema_mapper.generate_alias_dict import generate_alias_dict

fields = pd.read_csv("my_fields.csv")          # needs a 'field_name' column
alias_df = generate_alias_dict(fields, model="claude-sonnet-4-6")
alias_df.to_csv("my_alias_dict.csv", index=False)

engine = SchemaMapEngine(
    clinical_data_path="my_clinical.csv",
    curated_dict_path="my_fields.csv",
    alias_dict_path="my_alias_dict.csv",
)
```

A runnable demo of this whole lifecycle — define target fields,
generate, persist, then load the result into a `SchemaMapEngine` — is at
[`examples/generate_alias_dict.ipynb`](../examples/generate_alias_dict.ipynb)
(with a CLI equivalent in
[`generate_alias_dict.py`](../examples/generate_alias_dict.py)).
Generation costs LLM tokens and needs `ANTHROPIC_API_KEY` or
`GEMINI_API_KEY` set; see §3.

------------------------------------------------------------------------

## 3. Which environment variable gates which stage

Everything works with **no** keys set *except* the stages noted below.
Copy `.env.example` → `.env` and fill in what you need.

| Variable | Needed for | Without it |
|----|----|----|
| `UMLS_API_KEY` | **Any** `ontology_source="ncit"` run: the corpus build *and* the per-term concept-table build that runs at `OntoMapEngine(...)` construction (also Stage 2.5 synonyms) | NCIt mapping cannot complete on a cold cache. Passing `corpus_df=` avoids only the *corpus* fetch, not the concept-table build. Workarounds: reuse a warm cache from a prior keyed run, or use a non-NCIt `ontology_source` (`mondo`/`uberon`, via OLS — no key). |
| `GEMINI_API_KEY` | OM Stage 4 (`s4_strategy="llm"`) and SM `mode="auto"` Stage 4 | Use SM `mode="manual"`; leave OM `s4_strategy=None` (the default) |
| `METAHARMONIZER_DATA_DIR` | Pointing at a local corpus/schema copy | Falls back to bundled reference files and `~/.metaharmonizer/data` |

> **First-run cost.** The first `OntoMapEngine` run for a given NCIt
> corpus builds concept tables from the NCI API at construction time
> (proportional to the number of corpus terms — minutes for the full
> \~14 k-term disease corpus, seconds for a small slice), downloads the
> sap-bert encoder (\~440 MB), then builds a FAISS index (\~4 min for
> the full corpus). Everything is cached, so subsequent runs take \~7
> sec. `SchemaMapEngine` downloads the `all-MiniLM-L6-v2` encoder once
> on first use.

See the main [README](../README.md) §2 for the full environment-variable
table and the project-config-file layer.
