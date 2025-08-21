# KnowledgeDb

Current version already supports vector-based retrieval and database queries.

## 📂 Components

### 1. **Vector Database**
- **`vector_db.sqlite`** – SQLite-based metadata storage for vector search.
- **`faiss_indexes/`** – FAISS indexes for fast semantic similarity search
### 2. **Pipeline**
- **`faiss_sqlite_pipeline.py`** – End-to-end pipeline combining FAISS vector search with SQLite metadata storage.  
  Handles vector insertion, retrieval, and search operations.
### 3. **Database Clients**
- **`db_clients/nci_db.py`** – Client for querying the NCI Thesaurus API.
- **`db_clients/umls_db.py`** – Client for querying the UMLS API.

---

## 📌 Notes
- **Development Status:**  
  - Vector database integration is functional.  
  - FAISS indexes and SQLite database are provided for immediate use.  
  - Database clients are ready for NCI and UMLS queries.  
  - Knowledge DB of bio-clinical metadata is planned.

- This folder acts as the **backend vector store** for mapping workflows and retrieval-augmented generation.
---
