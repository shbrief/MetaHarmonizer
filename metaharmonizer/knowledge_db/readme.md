# KnowledgeDb

Current version already supports vector-based retrieval and database queries.

## 📂 Components

### 1. **Vector Database**
- **`vector_db.sqlite`** – SQLite-based metadata storage for vector search.
- **`faiss_indexes/`** – FAISS indexes for fast semantic similarity search

### 2. **Pipeline**
- **`faiss_sqlite_pipeline.py`** – End-to-end pipeline combining FAISS vector 
search with SQLite metadata storage. Handles vector insertion, retrieval, and 
search operations.

### 3. **Database Clients**
- **`db_clients/nci_db.py`** – Client for querying the NCI Thesaurus API.
- **`db_clients/umls_db.py`** – Client for querying the UMLS API.
- **`db_clients/ols_client.py`** – Async client for the EBI OLS4 REST API 
(ontology term lookup and descendant traversal).

### 4. **Corpus Builder**
- **`corpus_builder.py`** – Build an ontology term corpus from a root term by 
collecting all its descendants via the EBI OLS API. Analogous to LinkML dynamic 
enums (`reachable_from`).

---

## 📌 Notes
- **Development Status:**  
  - Vector database integration is functional.  
  - FAISS indexes and SQLite database are provided for immediate use.  
  - Database clients are ready for NCI and UMLS queries.  
  - Knowledge DB of bio-clinical metadata is planned.

- This folder acts as the **backend vector store** for mapping workflows and
retrieval-augmented generation.

---

## CorpusBuilder Quickstart

CorpusBuilder collects all descendants of a given ontology term from the [EBI OLS4 API](https://www.ebi.ac.uk/ols4) and saves them as JSON. It supports any ontology indexed by OLS (NCIT, MONDO, HP, EFO, etc.).

### Python API

```python
from metaharmonizer.KnowledgeDb.corpus_builder import CorpusBuilder

builder = CorpusBuilder()

# Build corpus from all descendants of a root term
corpus = builder.build_sync("NCIT:C3262")          # Neoplasm
builder.save(corpus, "data/corpus/neoplasm.json", root_term_id="NCIT:C3262")

# Works with any ontology in OLS (prefix is auto-detected)
corpus = builder.build_sync("MONDO:0005070")        # Disease
corpus = builder.build_sync("HP:0000118")           # Phenotypic abnormality

# Exclude the root term itself from the output
corpus = builder.build_sync("NCIT:C3262", include_root=False)

# Async usage
import asyncio
records = asyncio.run(builder.build("NCIT:C2924"))
```

### CLI

```bash
python scripts/build_corpus.py NCIT:C3262 -o data/corpus/neoplasm.json
python scripts/build_corpus.py MONDO:0005070 -o data/corpus/mondo_disease.json
python scripts/build_corpus.py NCIT:C3262 --no-root -o output.json
```

### Output format

```json
{
  "metadata": {
    "root_term_id": "NCIT:C3262",
    "root_term_label": "Neoplasm",
    "ontology": "ncit",
    "total_terms": 15432,
    "generated_at": "2026-02-10T..."
  },
  "terms": [
    {
      "iri": "http://purl.obolibrary.org/obo/NCIT_C3262",
      "ontology_name": "ncit",
      "ontology_prefix": "NCIT",
      "short_form": "NCIT_C3262",
      "label": "Neoplasm",
      "obo_id": "NCIT:C3262",
      "description": "A benign or malignant tissue growth...",
      "type": "class"
    }
  ]
}
```

---

## Descriptions

#### Overview
The KnowledgeDb subsystem in MetaHarmonizer provides a hybrid vector–metadata 
store for ontology mapping and synonym lookup. It combines a lightweight SQLite 
metadata store with FAISS vector indexes to support semantic retrieval 
(for RAG-style mapping) and a synonym-oriented FAISS+SQLite module for direct 
semantic lookup of synonyms and their canonical terms. The implementation also 
includes an FTS5-based synonym index for fast string matching with BM25-like 
ranking.

#### Initialization and availability
On construction, the system ensures on-disk artifacts are available (SQLite
database and FAISS index files). Paths are configurable via environment
variables (`VECTOR_DB_PATH`, `FAISS_INDEX_DIR`, `KNOWLEDGE_DB_DIR`); defaults
resolve to `~/.metaharmonizer/KnowledgeDb/` for installed users. If artifacts
are missing the system falls back to local-build mode — downstream pipelines
rebuild the corpus + index on demand from upstream ontology APIs (first run
may be slow). Initialization is thread-safe and idempotent to support use in
multi-threaded workflows.

Embedding models and adapter
KnowledgeDb loads a cached embedding model per configured method using a model loader utility and wraps the raw model with an EmbeddingAdapter. The adapter exposes two main operations used by the pipeline:
- embed_documents: create batch embeddings for arbitrary text (contexts, synonyms, corpus entries);
- embed_query: produce a single-query embedding for similarity search.
The code checks and enforces dimensional consistency between model outputs and persisted FAISS indexes when loading an existing index.

Core components and schemas
1. FAISSSQLiteSearch (RAG / ST usage)
- Purpose: maintain term → code → context triples and their vector representations for retrieval-augmented mapping.
- SQLite schema (for RAG strategies): table columns id (PRIMARY KEY), term (TEXT), code (TEXT UNIQUE), context (TEXT).
- Non-RAG strategies store only term text in the table.
- Index file naming follows a convention combining strategy, method, and category; the SQLite table name is similarly derived.

2. SynonymDict (FAISS + SQLite with ID mapping)
- Purpose: store synonym ↔ official_label ↔ nci_code triples with exact SQLite ID ↔ FAISS-ID mapping so FAISS search returns persistent database identifiers.
- SQLite schema: id (PRIMARY KEY), synonym (TEXT), official_label (TEXT), nci_code (TEXT) with an index on nci_code for efficient filtering.
- FAISS index is an IndexIDMap2 wrapper around an IndexFlatIP base so vectors can be added with explicit SQLite IDs.

3. FTSSynonymDb (FTS5 trigram)
- Purpose: FTS5 virtual table tuned for lexical synonym matching. Table columns: synonym, standard_term, nci_code; trigram tokenizer used to improve fuzzy matches and partial hits.
- The FTS module offers BM25-style ranking; returned ranks are normalized via an exponential mapping to a 0–1 confidence score (scale parameter = 10.0).

Data ingestion and enrichment pipeline
For RAG-style mapping (FAISSSQLiteSearch.fetch_and_store_terms):
- Term list ingestion: the pipeline identifies missing terms by comparing an in-memory set of existing terms (cached via an lru_cache) to the incoming corpus or DataFrame.
- Term → code resolution: NCI/UMLS clients are queried asynchronously to obtain candidate concept codes for each term. The system batches network calls and uses an asynchronous HTTP client with a connection limit; default intra-batch size for term-to-code fetching is TERM_BATCH_SIZE = 60.
- Concept enrichment: for resolved codes, concept metadata is fetched in bulk (get_custom_concepts_by_codes) and used to build contextual strings (e.g., "term: context") that combine the surface term with curated concept description fields.
- Storage: contextual records (term, code, context) are inserted into SQLite using parameterized executemany operations. The code then computes embeddings on the assembled contexts and inserts their vectors into FAISS.

For corpus (LM/ST strategies):
- build_corpus_vector_db accepts an ordered list of curated documents or labels, computes embeddings in batches, and stores the vectors in a FAISS flat index. A corresponding SQLite table stores the textual entries. This method is intended for strategies that do not require code/context enrichment via external APIs.

Synonym index building (SynonymDict.build_index_from_codes)
- Code selection: input is a list of NCI codes (optionally force-rebuilt). Already-indexed codes are skipped unless force_rebuild is set.
- Concept extraction: concept entries are converted to a list of unique synonyms per code (including the canonical label and synonyms from the metadata).
- Atomic batch insert: records are inserted into SQLite atomically using a batched INSERT ... RETURNING approach (SQLite >= 3.35.0 required). To avoid parameter limits, insertions are chunked (CHUNK_SIZE = 3000).
- FAISS construction: synonyms are embedded in batches (default embedding batch 512, adaptively adjusted for GPU memory) and added to an IndexIDMap2 along with their inserted SQLite IDs. The index is saved atomically to disk by writing to a temporary file and performing an atomic replace.

Search and retrieval
- Semantic similarity (FAISS): For RAG flows, similarity_search embeds the query, normalizes embeddings to unit length (unless adapters/models are already normalized), and performs an inner-product search (IndexFlatIP) to obtain top-k matches. For each FAISS hit the code maps the returned FAISS position to the corresponding SQLite id and retrieves metadata (term, context, code) and score. Results can be returned as raw dicts or Document objects.
- Synonym semantic search (SynonymDict.search): Queries are semantically embedded and matched against the FAISS index that directly returns SQLite IDs (IndexIDMap2). Metadata for returned IDs is retrieved via a persistent SQLite connection and returned with cosine-like scores. A query result cache (OrderedDict) with a large capacity (default cap 200,000 entries) speeds repeat queries. Batch search is supported via search_many which encodes multiple queries in a single call and maps results back to the original query order.
- Lexical search (FTS): The FTS5 index supports fast string matching; when a literal MATCH query fails due to tokenization or quoting edge-cases a fallback quoted variant is attempted. BM25-style ranks returned by FTS are normalized to confidence scores using an exponential mapping.

Numerical and operational details
- Normalization: vectors are L2-normalized for inner-product indexes so the index effectively performs cosine-similarity ranking with IndexFlatIP. If the embedding model flags pre-normalized outputs, normalization is skipped to avoid double normalization.
- Batch sizing and GPU awareness: default embedding batch sizes are tuned (512), and the system probes available GPU memory (if any) to dynamically reduce/increase batch size to avoid out-of-memory conditions. After GPU operations, CUDA caches are cleared and Python garbage collection invoked to release memory promptly.
- FAISS storage and IO: indexes are persisted under FAISS_INDEX_DIR; when building on GPU the index is transferred to CPU prior to writing. Index writes are guarded by temporary file writes and atomic replacement to avoid corrupted index files.
- Threading and async behavior: network-bound code paths use asyncio and an asynchronous HTTP client to maximize throughput; CPU-bound embedding is done synchronously in batches to control memory usage.
- Robustness: lru_cache and explicit in-process caches avoid repeated API calls for already-known terms. The ingestion code detects missing or malformed concept entries and skips them gracefully, emitting diagnostic logs.

Logging and observability
All components instantiate a custom logger configured at INFO level (configurable). Progress updates for large operations (synonym extraction, embedding batches) are emitted via progress bars and log messages to help monitor long-running builds.

Configuration and reproducibility
All file-location and API key settings are controlled by environment variables (`VECTOR_DB_PATH`, `FAISS_INDEX_DIR`, `KNOWLEDGE_DB_DIR`, `UMLS_API_KEY`). The `ensure_knowledge_db` routine checks for required artifacts and falls back to local-build mode when missing, enabling reproducible setup for new environments.

Summary
KnowledgeDb implements a pragmatic, reproducible hybrid retrieval backend combining FAISS vector search with SQLite metadata and FTS5 lexical indices. It supports both retrieval-augmented mapping (term→NCI-code→context) and synonym-centric search with persistent ID mapping, asynchronous API enrichment, GPU-aware batching, and careful transactional semantics for bulk inserts — enabling efficient, production-ready ontology mapping and synonym lookup in MetaHarmonizer.