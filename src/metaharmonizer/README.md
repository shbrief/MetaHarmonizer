## MetaHarmonizer Mapping Engines Architecture

### Key Components:

#### 1. **Main Engines**
- **OntoMapEngine**: Orchestrates the multi-stage ontology mapping pipeline
  (Stage 2 base match → Stage 2.5 synonym → Stage 3 RAG re-match → Stage 4 LLM rewrite)
  - 📁 [`engine/ontology_mapping_engine.py`](engine/ontology_mapping_engine.py)
- **SchemaMapEngine**: Multi-stage schema mapping engine
  - 📁 [`models/schema_mapper/engine.py`](models/schema_mapper/engine.py)

#### 2. **Ontology Mappers**
- **OntoMapLM**: Uses language models with CLS token embeddings (Stage 2, `s2_strategy='lm'`)
  - 📁 [`models/ontology_mapper_lm.py`](models/ontology_mapper_lm.py)
- **OntoMapST**: Uses sentence transformers for semantic similarity (Stage 2, `s2_strategy='st'`)
  - 📁 [`models/ontology_mapper_st.py`](models/ontology_mapper_st.py)
- **OntoMapSynonym**: Synonym-dictionary matching via NCI/OLS APIs (Stage 2.5)
  - 📁 [`models/ontology_mapper_synonym.py`](models/ontology_mapper_synonym.py)
- **OntoMapRAG**: Retrieval-augmented generation with FAISS vector search (Stage 3, `s3_strategy='rag'`)
  - 📁 [`models/ontology_mapper_rag.py`](models/ontology_mapper_rag.py)
- **OntoMapBIE**: Bi-encoder RAG with query- and corpus-side context retrieval (Stage 3, `s3_strategy='rag_bie'`)
  - 📁 [`models/ontology_mapper_bi_encoder.py`](models/ontology_mapper_bi_encoder.py)
- **OntoMapLLM**: LLM query rewriting + FAISS re-search for low-confidence matches (Stage 4, `s4_strategy='llm'`)
  - 📁 [`models/ontology_mapper_llm.py`](models/ontology_mapper_llm.py)
- **Reranker**: Unified reranker (cross-encoder / T5 / generative LLM) used to re-score Stage 3 candidates
  - 📁 [`models/reranker.py`](models/reranker.py)

#### 3. **Schema Mappers**
- Schema mapping is handled by the multi-stage **SchemaMapEngine** (see Main Engines),
  built from per-stage matchers under [`models/schema_mapper/matchers/`](models/schema_mapper/matchers/):
  - Stage 1 — dictionary / fuzzy column-name matching ([`stage1_matchers.py`](models/schema_mapper/matchers/stage1_matchers.py))
  - Stage 2 — value-based matching ([`stage2_matchers.py`](models/schema_mapper/matchers/stage2_matchers.py))
  - Stage 3 — type / numeric / semantic field matching ([`stage3_matchers.py`](models/schema_mapper/matchers/stage3_matchers.py))
  - Stage 4 — LLM-based fallback matching for low-confidence columns (`mode='auto'` only)
    ([`stage4_matchers.py`](models/schema_mapper/matchers/stage4_matchers.py))

#### 4. **Field Suggester**
- **FieldSuggester**: Hybrid NER + embedding-clustering tool that proposes new harmonized
  fields for columns that could not be mapped to any existing target field.
  - 📁 [`field_suggester/field_suggester.py`](field_suggester/field_suggester.py)
  - Semantic clustering (hierarchical + Semantic Consistency Score):
    📁 [`field_suggester/semantic_clustering.py`](field_suggester/semantic_clustering.py)
  - Post-clustering refinement (split over-merged clusters):
    📁 [`field_suggester/cluster_refiner.py`](field_suggester/cluster_refiner.py)
  - SchemaMapEngine integration helper:
    📁 [`field_suggester/integration.py`](field_suggester/integration.py)

#### 5. **Base Classes**
- **OntoModelsBase**: Common functionality for ontology mappers
  - 📁 [`models/ontology_models.py`](models/ontology_models.py)

#### 6. **Database/Storage**
- **FAISSSQLiteSearch**: Vector similarity search with SQLite backend
  - 📁 [`knowledge_db/faiss_sqlite_pipeline.py`](knowledge_db/faiss_sqlite_pipeline.py)
- **SynonymDict**: Synonym index backing Stage 2.5 matching
  - 📁 [`knowledge_db/synonym_dict.py`](knowledge_db/synonym_dict.py)
- **Corpus / Concept builders**: Build corpus and concept tables from ontology backends
  - 📁 [`knowledge_db/corpus_builder.py`](knowledge_db/corpus_builder.py)
  - 📁 [`knowledge_db/concept_table_builder.py`](knowledge_db/concept_table_builder.py)
- **External Databases**: Integration with NCI, UMLS, OLS ontologies
  - 📁 [`knowledge_db/db_clients/nci_db.py`](knowledge_db/db_clients/nci_db.py)
  - 📁 [`knowledge_db/db_clients/umls_db.py`](knowledge_db/db_clients/umls_db.py)
  - 📁 [`knowledge_db/db_clients/ols_db.py`](knowledge_db/db_clients/ols_db.py)
- **Model Cache/Loader & Embedding Store**: Efficient model management and caching
  - 📁 [`utils/model_loader.py`](utils/model_loader.py)
  - 📁 [`utils/model_cache.py`](utils/model_cache.py)
  - 📁 [`utils/embedding_store.py`](utils/embedding_store.py)

### Features:
- **Multi-Stage Ontology Pipeline**: Cascading stages with per-stage confidence thresholds
- **Exact & Fuzzy Matching**: Multiple matching strategies
- **Synonym Resolution**: NCI/OLS synonym dictionary lookups
- **Vector Similarity Search**: Semantic understanding via FAISS
- **Optional Reranking**: Cross-encoder / T5 / generative rerankers for Stage 3
- **LLM Fallback**: Query rewriting (ontology) and field matching (schema) for hard cases
- **New-Field Suggestion**: NER + embedding clustering for unmapped columns
- **Configurable Top-K Results**: Flexible result ranking
- **JSON / CSV Output Format**: Standardized output
- **Batch Processing**: Efficient bulk operations
- **Custom Logger Integration**: Comprehensive logging

### Brief Description of Each Function

**Engines**

- `OntoMapEngine.run()`
  - Main function to execute the multi-stage ontology mapping pipeline.
  - Takes queries and a corpus, runs the selected Stage 2 base matcher (`lm`/`st`),
    optionally cascades low-confidence queries through Stage 2.5 synonym matching,
    Stage 3 RAG re-matching (`rag`/`rag_bie`, optionally reranked), and Stage 4 LLM
    query rewriting, returning the top matches for each query term.

- `SchemaMapEngine.run_schema_mapping()`
  - Executes the multi-stage schema mapping pipeline.
  - Aligns clinical data columns to harmonized schema fields using exact/fuzzy
    matching, value-based matching, type/numeric/semantic checks, and (in `auto`
    mode) an LLM fallback for low-confidence columns.

**Ontology Mapper Models**

- `get_match_results()`
  - Computes the most similar/correct ontology term(s) for a given query term using embedding models or scoring.
  - Implemented by each strategy (`OntoMapLM`, `OntoMapST`, `OntoMapRAG`, `OntoMapBIE`, `OntoMapSynonym`).

- `create_embeddings()`
  - Converts terms into vector embeddings using the selected NLP model, to enable similarity comparisons.

**Reranker**

- `Reranker.rerank()`
  - Re-scores a candidate list against a query using the configured reranker type
    (cross-encoder, T5, or generative LLM), auto-detected from the method key.

**Field Suggester**

- `FieldSuggester.suggest()`
  - Groups unmapped columns into candidate new fields via NER + embedding clustering,
    generates readable field names, and scores each suggestion by NER/embedding agreement.
- `suggest_from_schema_mapper()`
  - Convenience wrapper that extracts low-confidence (unmapped) columns from
    SchemaMapEngine output and forwards them to `FieldSuggester.suggest()`.

**KnowledgeDb**

- `get_nci_code_by_term()`
  - Retrieves the NCIt code for a given term, using local or remote knowledge base (e.g., UMLS, NCIt, FAISS index).
  - Supports synonym and hierarchical lookups to resolve biomedical terms.

<img src="https://github.com/shbrief/MetaHarmonizer/blob/main/Figures/MetaHarmonizer%20-%20Architecture.png" />

---

### Usage Patterns:
```python
from metaharmonizer import OntoMapEngine, SchemaMapEngine

# Ontology Mapping
onto_engine = OntoMapEngine(
    category='disease',
    query=query_list,
    ground_truth_map=ground_truth_map,        # query term -> curated ontology (for test mode)
    ontology_source='ncit',   # 'ncit' (NCI EVSREST), 'mondo', 'uberon' (EBI OLS4)
    s2_method='sap-bert',     # transformer model key from method_model.yaml
    s2_strategy='lm',         # Stage 2: 'lm' (CLS pooling) or 'st' (SentenceTransformer)
    s3_strategy='rag',        # Stage 3 re-match: 'rag', 'rag_bie', or None
    s3_threshold=0.9,         # route queries below this Stage-2 score to Stage 3
    s4_strategy='llm',        # Stage 4 LLM query rewrite: 'llm' or None
    s4_threshold=0.6,         # route queries below this score to Stage 4
    top_k=5,
    test_or_prod='test',      # optional: inferred from ground_truth_map if omitted ('test' if provided, else 'prod')
)
results = onto_engine.run()

# Schema Mapping
schm_engine = SchemaMapEngine(
    input_path=file,
    mode='auto',              # 'auto' enables the Stage-4 LLM fallback; or 'manual'
    top_k=5,
)
schm_engine.run_schema_mapping()
```

> See the top-level [`README.md`](../../README.md) for the full parameter reference
> and runnable quickstart.
