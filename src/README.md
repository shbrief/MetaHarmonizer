## MetaHarmonizer Mapping Engines Architecture

<img src="https://github.com/shbrief/MetaHarmonizer/blob/diagrams/Figures/MetaHarmonizer_All%20_%20Mermaid%20Chart-2025-07-03-210436.png" width="100%" height="100%"/>

### Legend:
- 🔵 **Blue** - Main Engine Classes (Top-level orchestrators)
- 🟣 **Purple** - Ontology Mapper Classes (Specific mapping strategies)
- 🟠 **Orange** - Schema Mapper Classes (Different matching approaches)
- 🔴 **Red** - Base Classes (Common functionality/inheritance)
- 🟢 **Green** - Database/Storage Components (Data persistence & retrieval)

### Key Components:

#### 1. **Main Engines** (Blue)
- **OntoMapEngine**: Orchestrates ontology mapping with strategies (lm, st, rag)
  - 📁 [`src/Engine/ontology_mapping_engine.py`](src/Engine/ontology_mapping_engine.py)
- **SchemaMapEngine**: Manages schema mapping with different matcher types
  - 📁 [`src/Engine/schema_mapping_engine.py`](src/Engine/schema_mapping_engine.py)

#### 2. **Ontology Mappers** (Purple)
- **OntoMapLM**: Uses language models with CLS token embeddings
  - 📁 [`src/models/ontology_mapper_lm.py`](src/models/ontology_mapper_lm.py)
- **OntoMapST**: Uses sentence transformers for semantic similarity
  - 📁 [`src/models/ontology_mapper_st.py`](src/models/ontology_mapper_st.py)
- **OntoMapRAG**: Retrieval-augmented generation with FAISS vector search
  - 📁 [`src/models/ontology_mapper_rag_faiss.py`](src/models/ontology_mapper_rag_faiss.py)

#### 3. **Schema Mappers** (Orange)
- **ClinicalDataMatcherBert**: BERT-based embeddings for clinical data
  - 📁 [`src/models/schema_mapper_bert.py`](src/models/schema_mapper_bert.py)
- **ClinicalDataMatcherWithTopicModeling**: LDA topic modeling approach
  - 📁 [`src/models/schema_mapper_lda.py`](src/models/schema_mapper_lda.py)
- **ClinicalDataMatcher**: Frequency-based matching (base implementation)
  - 📁 [`src/models/schema_mapper_models.py`](src/models/schema_mapper_models.py)

#### 4. **Base Classes** (Red)
- **OntoModelsBase**: Common functionality for ontology mappers
  - 📁 [`src/models/ontology_models.py`](src/models/ontology_models.py)
- **ClinicalDataMatcher**: Base implementation for schema mapping
  - 📁 [`src/models/schema_mapper_models.py`](src/models/schema_mapper_models.py)

#### 5. **Database/Storage** (Green)
- **FAISSSQLiteSearch**: Vector similarity search with SQLite backend
  - 📁 [`src/KnowledgeDb/faiss_sqlite_pipeline.py`](src/KnowledgeDb/faiss_sqlite_pipeline.py)
- **External Databases**: Integration with NCI, UMLS ontologies
  - 📁 [`src/KnowledgeDb/db_clients/nci_db.py`](src/KnowledgeDb/db_clients/nci_db.py)
  - 📁 [`src/KnowledgeDb/db_clients/umls_db.py`](src/KnowledgeDb/db_clients/umls_db.py)
- **Model Cache/Loader**: Efficient model management and caching
  - 📁 [`src/utils/model_loader.py`](src/utils/model_loader.py)
  - 📁 [`src/utils/model_cache.py`](src/utils/model_cache.py)

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
  - Executes schema mapping between clinical tabular data and a harmonized schema dictionary.
  - Uses ML/embedding-based matching or frequency/statistics-based approaches to align schema columns.

**Ontology Mapper Models**

- `get_matches()`
  - Computes the most similar/correct ontology term(s) for a given query term using embedding models or scoring.
  - Used for both language model and sentence transformer strategies.

- `encode_terms()`
  - Converts terms into vector embeddings using the selected NLP model, to enable similarity comparisons.

**Schema Mapper Models**

- `map_schema()`
  - Matches columns from a source schema to a target/harmonized schema using machine learning or statistical heuristics.

- `fit_transform()`
  - Fits the model on provided schema data and applies transformation/mapping in a single step.

**KnowledgeDb**

- `get_nci_code_by_term()`
  - Retrieves the NCIt code for a given term, using local or remote knowledge base (e.g., UMLS, NCIt, FAISS index).
  - Supports synonym and hierarchical lookups to resolve biomedical terms.

<img src="https://github.com/shbrief/MetaHarmonizer/blob/main/Figures/MetaHarmonizer%20-%20Architecture.png" width="100%" height="100%"/>

---

### Usage Patterns:
```python
# Ontology Mapping
onto_engine = OntoMapEngine(
    method='sap-bert',
    category='disease',
    om_strategy='lm',  # or 'st', 'rag'
    query=query_list,
    corpus=corpus_list,
    topk=5
)

# Schema Mapping  
schema_engine = SchemaMapEngine(
    clinical_data_path='data.tsv',
    schema_map_path='schema.pkl',
    matcher_type='Bert',  # or 'LDA', None
    k=5
)
```

