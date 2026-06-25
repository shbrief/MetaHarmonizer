## MetaHarmonizer Mapping Engines Architecture

### Key Components:

#### 1. **Main Engines** (Blue)
- **OntoMapEngine**: Orchestrates ontology mapping with strategies (lm, st, rag)
  - 📁 [`src/metaharmonizer/engine/ontology_mapping_engine.py`](src/metaharmonizer/engine/ontology_mapping_engine.py)
- **SchemaMapEngine**: Multi-stage schema mapping engine
  - 📁 [`src/metaharmonizer/engine/schema_mapping_engine.py`](src/metaharmonizer/engine/schema_mapping_engine.py)

#### 2. **Ontology Mappers** (Purple)
- **OntoMapLM**: Uses language models with CLS token embeddings
  - 📁 [`src/metaharmonizer/models/ontology_mapper_lm.py`](src/metaharmonizer/models/ontology_mapper_lm.py)
- **OntoMapST**: Uses sentence transformers for semantic similarity
  - 📁 [`src/metaharmonizer/models/ontology_mapper_st.py`](src/metaharmonizer/models/ontology_mapper_st.py)
- **OntoMapRAG**: Retrieval-augmented generation with FAISS vector search
  - 📁 [`src/metaharmonizer/models/ontology_mapper_rag.py`](src/metaharmonizer/models/ontology_mapper_rag.py)

#### 3. **Schema Mappers** (Orange)
- Schema mapping is handled entirely by the multi-stage SchemaMapEngine (see Main Engines).

#### 4. **Base Classes** (Red)
- **OntoModelsBase**: Common functionality for ontology mappers
  - 📁 [`src/metaharmonizer/models/ontology_models.py`](src/metaharmonizer/models/ontology_models.py)

#### 5. **Database/Storage** (Green)
- **FAISSSQLiteSearch**: Vector similarity search with SQLite backend
  - 📁 [`src/metaharmonizer/knowledge_db/faiss_sqlite_pipeline.py`](src/metaharmonizer/knowledge_db/faiss_sqlite_pipeline.py)
- **External Databases**: Integration with NCI, UMLS ontologies
  - 📁 [`src/metaharmonizer/knowledge_db/db_clients/nci_db.py`](src/metaharmonizer/knowledge_db/db_clients/nci_db.py)
  - 📁 [`src/metaharmonizer/knowledge_db/db_clients/umls_db.py`](src/metaharmonizer/knowledge_db/db_clients/umls_db.py)
- **Model Cache/Loader**: Efficient model management and caching
  - 📁 [`src/metaharmonizer/utils/model_loader.py`](src/metaharmonizer/utils/model_loader.py)
  - 📁 [`src/metaharmonizer/utils/model_cache.py`](src/metaharmonizer/utils/model_cache.py)

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
  - Aligns clinical data columns to harmonized schema fields using a combination of exact/fuzzy matching, numeric checks, semantic similarity, and value-based matching.

**Ontology Mapper Models**

- `get_matches()`
  - Computes the most similar/correct ontology term(s) for a given query term using embedding models or scoring.
  - Used for both language model and sentence transformer strategies.

- `encode_terms()`
  - Converts terms into vector embeddings using the selected NLP model, to enable similarity comparisons.

**KnowledgeDb**

- `get_nci_code_by_term()`
  - Retrieves the NCIt code for a given term, using local or remote knowledge base (e.g., UMLS, NCIt, FAISS index).
  - Supports synonym and hierarchical lookups to resolve biomedical terms.

<img src="https://github.com/shbrief/MetaHarmonizer/blob/main/Figures/MetaHarmonizer%20-%20Architecture.png" />

---

### Usage Patterns:
```python
# Ontology Mapping
onto_engine = OntoMapEngine(
    category='disease',
    query=query_list,
    cura_map=cura_map,        # query term -> curated ontology (for test mode)
    s2_method='sap-bert',     # transformer model key from method_model.yaml
    s2_strategy='lm',         # 'lm' (CLS pooling) or 'st' (SentenceTransformer)
    s3_strategy='rag',        # optional Stage-3 re-matching; None to disable
    test_or_prod='test',
    topk=5,
)
results = onto_engine.run()

# Schema Mapping
schm_engine = SchemaMapEngine(
    clinical_data_path=file,
    mode='manual',  # or 'auto'
    top_k=5,
)
schm_engine.run_schema_mapping()
```

> See the top-level [`readme.md`](../readme.md) for the full parameter reference
> and runnable quickstart.

