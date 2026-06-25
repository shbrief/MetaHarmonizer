## MetaHarmonizer Mapping Engines Architecture

### Key Components:

#### 1. **Main Engines** (Blue)
- **OntoMapEngine**: Orchestrates ontology mapping with strategies (lm, st, rag)
  - 📁 [`engine/ontology_mapping_engine.py`](engine/ontology_mapping_engine.py)
- **SchemaMapEngine**: Multi-stage schema mapping engine
  - 📁 [`models/schema_mapper/engine.py`](models/schema_mapper/engine.py)

#### 2. **Ontology Mappers** (Purple)
- **OntoMapLM**: Uses language models with CLS token embeddings
  - 📁 [`models/ontology_mapper_lm.py`](models/ontology_mapper_lm.py)
- **OntoMapST**: Uses sentence transformers for semantic similarity
  - 📁 [`models/ontology_mapper_st.py`](models/ontology_mapper_st.py)
- **OntoMapRAG**: Retrieval-augmented generation with FAISS vector search
  - 📁 [`models/ontology_mapper_rag.py`](models/ontology_mapper_rag.py)

#### 3. **Schema Mappers** (Orange)
- Schema mapping is handled by the multi-stage **SchemaMapEngine** (see Main Engines),
  built from per-stage matchers under [`models/schema_mapper/matchers/`](models/schema_mapper/matchers/):
  - Stage 1 — dictionary / fuzzy column-name matching ([`stage1_matchers.py`](models/schema_mapper/matchers/stage1_matchers.py))
  - Stage 2 — value-based matching ([`stage2_matchers.py`](models/schema_mapper/matchers/stage2_matchers.py))
  - Stage 3 — type / numeric / semantic field matching ([`stage3_matchers.py`](models/schema_mapper/matchers/stage3_matchers.py))

#### 4. **Base Classes** (Red)
- **OntoModelsBase**: Common functionality for ontology mappers
  - 📁 [`models/ontology_models.py`](models/ontology_models.py)

#### 5. **Database/Storage** (Green)
- **FAISSSQLiteSearch**: Vector similarity search with SQLite backend
  - 📁 [`knowledge_db/faiss_sqlite_pipeline.py`](knowledge_db/faiss_sqlite_pipeline.py)
- **External Databases**: Integration with NCI, UMLS, OLS ontologies
  - 📁 [`knowledge_db/db_clients/nci_db.py`](knowledge_db/db_clients/nci_db.py)
  - 📁 [`knowledge_db/db_clients/umls_db.py`](knowledge_db/db_clients/umls_db.py)
  - 📁 [`knowledge_db/db_clients/ols_db.py`](knowledge_db/db_clients/ols_db.py)
- **Model Cache/Loader**: Efficient model management and caching
  - 📁 [`utils/model_loader.py`](utils/model_loader.py)
  - 📁 [`utils/model_cache.py`](utils/model_cache.py)

### Features:
- **Exact & Fuzzy Matching**: Multiple matching strategies
- **Vector Similarity Search**: Semantic understanding
- **Configurable Top-K Results**: Flexible result ranking
- **JSON Output Format**: Standardized output
- **Batch Processing**: Efficient bulk operations
- **Custom Logger Integration**: Comprehensive logging

### Brief Description of Each Function

**Engines**

- `OntoMapEngine.run()`
  - Main function to execute ontology mapping.
  - Takes queries and a corpus, applies selected mapping strategy (language model, sentence transformer, etc.), and returns top matches for each query term.

- `SchemaMapEngine.run_schema_mapping()`
  - Executes the multi-stage schema mapping pipeline.
  - Aligns clinical data columns to harmonized schema fields using a combination of exact/fuzzy matching, value-based matching, type/numeric checks, and semantic similarity.

**Ontology Mapper Models**

- `get_match_results()`
  - Computes the most similar/correct ontology term(s) for a given query term using embedding models or scoring.
  - Implemented by each strategy (`OntoMapLM`, `OntoMapST`, `OntoMapRAG`).

- `create_embeddings()`
  - Converts terms into vector embeddings using the selected NLP model, to enable similarity comparisons.

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
    cura_map=cura_map,        # query term -> curated ontology (for test mode)
    s2_method='sap-bert',     # transformer model key from method_model.yaml
    s2_strategy='lm',         # 'lm' (CLS pooling) or 'st' (SentenceTransformer)
    s3_strategy='rag',        # optional Stage-3 re-matching; None to disable
    topk=5,
)
results = onto_engine.run()

# Schema Mapping
schm_engine = SchemaMapEngine(
    clinical_data_path=file,
    mode='auto',              # or 'manual'
    top_k=5,
)
schm_engine.run_schema_mapping()
```

> See the top-level [`README.md`](../../README.md) for the full parameter reference
> and runnable quickstart.
