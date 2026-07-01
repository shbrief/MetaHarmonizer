---
editor_options:
  markdown:
    wrap: 72
---

# How the OntologyMapper picks a target corpus

This explains **how `OntoMapEngine` decides which reference vocabulary
(the "corpus") to match your query terms against**, and how it resolves
that corpus to an actual table of terms at runtime.

It is the *why/how* companion to two task-oriented docs:

-   **Supplying or creating a corpus** —
    [`input_formats.md`](input_formats.md) §1.3 ("The corpus"). Start
    there if you just want to bring your own.
-   **Building a corpus from an ontology root** —
    [`knowledge_db/readme.md`](../src/metaharmonizer/knowledge_db/readme.md)
    ("CorpusBuilder Quickstart"). The mechanics of fetching descendants
    from OLS.

The authoritative source for the mapping below is the code, not this
page: see `_CORPUS_REGISTRY` and `_STATIC_CORPUS_SOURCES` in
[`engine/ontology_mapping_engine.py`](../src/metaharmonizer/engine/ontology_mapping_engine.py).
The tables here are illustrative and will drift; the dict is canonical.

------------------------------------------------------------------------

## 1. The principle: one attribute → one root → one corpus

You map the *values* of a metadata attribute (a **`category`** such as
`disease`, `bodysite`, `treatment`, `phenotype`) against ontology terms.
For each category you also pick an **`ontology_source`** (`ncit`,
`mondo`, `uberon`, `efo`, …). The pair **`(category, ontology_source)`**
is the key that selects the corpus.

Each known pair maps to a single **OBO root term**. The corpus is then
"that root and everything under it" — the descendant subtree. Picking
`disease` + `ncit` means *"match against the Neoplasm subtree of NCIt"*;
`bodysite` + `uberon` means *"match against the UBERON anatomy subtree"*.

This is analogous to a LinkML dynamic enum (`reachable_from`): the root
term defines membership.

------------------------------------------------------------------------

## 2. The registry: two kinds of corpus

`_CORPUS_REGISTRY` maps each supported `(category, ontology_source)` to
its root term. The entries fall into **two kinds**, which differ only in
how they are *materialized* on a cache miss:

| `category`  | `ontology_source` | Root term      | Kind            | Backend on miss        |
|-------------|-------------------|----------------|-----------------|------------------------|
| `disease`   | `ncit` (default)  | `NCIT:C3262`   | API-buildable   | NCI EVSREST            |
| `bodysite`  | `ncit`            | `NCIT:C32221`  | API-buildable   | NCI EVSREST            |
| `treatment` | `ncit`            | `NCIT:C1909`   | API-buildable   | NCI EVSREST            |
| `disease`   | `mondo`           | `MONDO:0000001`| API-buildable   | EBI OLS4               |
| `bodysite`  | `uberon`          | `UBERON:0001062`| API-buildable  | EBI OLS4               |
| `phenotype` | `efo`             | `EFO:0000001`  | **static**      | *(not buildable)*      |

-   **API-buildable** — if the corpus isn't cached, the engine fetches
    the root's descendants live (NCIt via EVSREST, everything else via
    OLS4) and writes the CSV.
-   **Static / shipped** — the `efo` phenotype corpus is a *merged*
    corpus (12 ontologies flattened under the EFO namespace). It is
    curated, **shipped as a CSV in the repo's `data/` tree, and cannot
    be rebuilt from an API**. Pairs of this kind are listed in
    `_STATIC_CORPUS_SOURCES`. A cache miss is an error, not a build —
    see §3.

The static corpus is deliberately *not* bundled inside the wheel (it is
~2 MB); it is materialized into the user cache by the bootstrap step.

------------------------------------------------------------------------

## 3. Resolution order

At construction, `OntoMapEngine` resolves the corpus in this order
(`_resolve_corpus_df`):

1.  **Caller-provided `corpus_df`** — if you pass one, it wins outright;
    the registry is not consulted, and `ontology_source` is *inferred*
    from the code prefixes instead. See
    [`input_formats.md`](input_formats.md) §1.3 for the contract and
    creation recipes.
2.  **Cached CSV** — otherwise the engine looks for
    `{ontology_source}_{category}_corpus.csv` at the path from
    `corpus_path()` (under `~/.metaharmonizer/data/corpus/retrieved_ontologies/`,
    relocatable via `METAHARMONIZER_DATA_DIR`). If present, it's loaded
    as-is.
3.  **On a cache miss, branch by kind:**
    -   **static** → raise `FileNotFoundError` pointing at the bootstrap
        step:
        `python -m metaharmonizer.scripts.bootstrap_data`
        (copies the shipped CSV from the repo `data/` tree into the
        cache; idempotent, `--force` to overwrite). Setting
        `METAHARMONIZER_DATA_DIR` to the repo's `data/` dir also works.
    -   **API-buildable** → fetch the root's descendants, write the CSV
        + a `.json` metadata envelope, and return the frame.

Before a guard at step 2/3, an **unknown** `(category, ontology_source)`
pair that isn't in the registry (and where no `corpus_df` was supplied)
raises `ValueError` listing the supported combinations — you cannot
silently map an attribute against a corpus that was never defined.

------------------------------------------------------------------------

## 4. What "correct" means here

The engine enforces *structural* correctness, not semantic adequacy:

-   **Registry membership** — an undefined `(category, ontology_source)`
    is rejected up front (§3).
-   **Schema validation** — a caller-provided `corpus_df` must carry a
    label column (`official_label` or `label`) and a code column
    (`clean_code` or `obo_id`), else `ValueError`
    (`_validate_user_corpus`). `_normalize_df` then fills
    `official_label` from `label` and `clean_code` from `obo_id`, and
    drops null/duplicate rows.
-   **Single-ontology partitioning** — for a caller-provided corpus,
    codes are partitioned by prefix (`_partition_codes`); a corpus
    mixing prefixes from more than one ontology is rejected. The single
    detected source overrides the declared `ontology_source` (with a
    warning).
-   **Cache isolation by content hash** — a caller-provided corpus is
    MD5-hashed; if it differs from the official corpus, a `_{hash}`
    suffix is appended to the SQLite table / FAISS index names so a
    custom corpus never collides with or pollutes the canonical cache.
    Identical hash → reuse the standard tables.
-   **Provenance** — API-buildable corpora carry source/version metadata
    in their `.json` envelope; static corpora are versioned by whatever
    the repo ships (bootstrap provenance, not API provenance).

------------------------------------------------------------------------

## 5. Known limits

-   **Disease root is Neoplasm-only.** `disease` + `ncit` resolves to
    `NCIT:C3262` (Neoplasm), so non-neoplastic diseases fall outside the
    corpus. Use `disease` + `mondo` for broader disease coverage.
-   **No staleness check.** A cached CSV is trusted if it exists; there
    is no comparison against the upstream ontology version (see the
    `TODO` in `_get_retrieved_ontology_metadata`). Delete the CSV to
    force a rebuild.
-   **The registry is hardcoded.** Adding an attribute means adding a
    `_CORPUS_REGISTRY` entry (and, for shipped corpora,
    `_STATIC_CORPUS_SOURCES` + the `bootstrap_data` manifest). There is
    no config-file or schema-driven discovery.
-   **Static corpora can't self-heal.** A `phenotype`/`efo` cache miss
    cannot be repaired by the engine — it requires the bootstrap step.

------------------------------------------------------------------------

## See also

-   [`input_formats.md`](input_formats.md) §1.3 — supplying / creating a
    `corpus_df` (the input contract and creation recipes).
-   [`knowledge_db/readme.md`](../src/metaharmonizer/knowledge_db/readme.md)
    — `CorpusBuilder` and `scripts/build_corpus.py` (building a corpus
    from an ontology root).
-   `engine/ontology_mapping_engine.py` — `_CORPUS_REGISTRY`,
    `_STATIC_CORPUS_SOURCES`, `_resolve_corpus_df` (canonical source).
