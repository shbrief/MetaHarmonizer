
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
│   │   ├── schema_mapping_engine.py
│   ├── CustomLogger
│   ├── KnowledgeDb
│   │   ├── faiss_sqlite_pipeline.py
│   │   ├── corpus_builder.py
│   │   └── db_clients
│   │       ├── nci_db.py
│   │       ├── umls_db.py
│   │       └── ols_client.py
│   ├── utils
│   ├── Plotter
├── setup.py
└── readme.md
```
### 2. Usage

In order to use schema and/Or ontology mapping functionality in metaharmonizer, please follow the steps below. 

#### 2.1. Environment setup

- First create a `conda create -n demo_env python=3.10 -y` 
- Activate the environment as `conda activate demo_env`
- Install the dependencies `pip install -r requirements.txt` after `pip install --upgrade pip`

#### 2.2. Cloning the repository

```git clone https://github.com/shbrief/MetaHarmonizer```


#### 2.3 Datasets
- The datasets in this repository are encrypted to prevent contamination of the gold standard.  
- For **ontology mapping**, you must provide:
  - A list of `query_terms`  
  - A list of `corpus_terms`  
- For **schema mapping**, provide a clinical metadata file.  
  - The schema mapping dictionary is available in the `/data` folder.  
- ⚠️ You will not be able to use the encrypted demo datasets without authorization, but you can supply your own query and corpus lists.
  
#### 2.4 Setting up the mappers 

1. Ontology Mapping
```python
## Go into the correct directory by specifying the user_path where MetaHarmonizer was cloned
%cd <user_path>/MetaHarmonizer/

## Ontology Mapping
## Import required packages 
import nest_asyncio
import pandas as pd
from importlib import reload

## Allow nested usage
nest_asyncio.apply()

## Import the models/engine for ontology mapping
from src.Engine import get_ontology_engine
from src.models import ontology_mapper_st as om_st
from src.models import ontology_mapper_lm as om_lm
from src.models import ontology_mapper_rag as om_rag
from src.models import ontology_mapper_bi_encoder as om_bi

## The reload() calls are optional, useful only if you are editing the code live in a notebook.
reload(om_st)
reload(om_lm)
reload(om_rag)
reload(om_bi)

OntoMapEngine = get_ontology_engine()

## Import useful utilities 
from evaluation.calc_stats import CalcStats # for calculating accuracy (testing)
from src.utils.cleanup_vector_store import cleanup_vector_store # for cleaning up the vector store

## Now you must initialize the engine
other_params = {"test_or_prod": "test"}
onto_engine_large = OntoMapEngine(method='sap-bert',
                                      category='disease',
                                      topk=5,
                                      query=query_list,
                                      corpus=large_corpus_list,
                                      cura_map=cura_map,
                                      om_strategy='lm',
                                      **other_params)
lm_sapbert_disease_top5_result = onto_engine_large.run()
# for more examples, you can refer to demo_nb/ontology_mapper_workflow.ipynb

## Run the ontology mapping
results_engine_testing = onto_engine_large.run()
```
- Parameters that can be changed in the model:
  - **query(list):** list of query terms (can be 1 or Many)
  - **corpus(list):** list of corpus terms to match against
  - **query(df):** df of query, for query enrichment in rag_bie strategy
  - **corpus(df):** df of corpus, for concept retrieval in rag/rag_bie strategy
  - **om_strategy(str):** 4 types of strategy are available 
    - strategy **lm**: Use [CLS] tokens for capturing the embedding representation. CLS is calculated in a much more intricate way, taking into account both its own embeddings (token/pos) as well as the context.
    - strategy **st**: Sentence transformer based strategy use default embedding method.
    - strategy **rag**: Combines retrieval from a knowledge database (e.g., FAISS + SQLite) with embedding-based similarity. Useful when the query requires additional context from a large corpus.
    - strategy **rag_bie**: RAG with Bi-Encoder query enrichment. Still under development; may be merged into `rag` in future releases.
  - **method(str):** These are string keys that fetch the different transformer models found in the mapping method_model.yaml file.
  - **topk(int):** Number of top matches to return for each query term in the query list
  - **other_params(dict):** This is like a kwargs dictionary that currently only takes a value for the key **test_or_prod**. In the future if more parameters are added to the model, then it will be updated in this dictionary.
  - **cura_map(dict):** Is a dictionary of paired query and ontology terms for evaluating or testing in the 'test' environment. 

- Output: Dataframe containing top 5 matches for each query term and their scores.

2. Schema mapping
```python
from src.models.schema_mapper import SchemaMapEngine

# Initialize the engine
engine = SchemaMapEngine(
    clinical_data_path=YOUR_QUERY_FILE,
    mode="manual",   # Options: "auto" or "manual"
    top_k=5,
)

# Run Stage 1, 2 & 3 (and 4 if mode="auto")
engine.run_schema_mapping()

# (Optional) Run Stage 4 after manual review
engine.run_stage4_from_manual("path_to_stage3_results.csv")
```
- Parameters that can be changed in the model:
  - clinical_data_path (str): Path to clinical dataset (TSV or CSV).
  - mode (str):
    - "auto" → automatically proceed to Stage 4 if Stage 3 confidence is low
    - "manual" → output Stage 3 results for review; Stage 4 must be triggered manually
  - top_k (int): Number of top matches returned for each column.

- Output  
  - CSV File: Results saved to data/schema_mapping_eval/ with suffix:  
_schema_map_auto.csv for auto mode  
_schema_map_manual.csv for manual mode  
_schema_map_stage3.csv for Stage 3 results  
  - Columns:  
query  
stage (stage1, stage2, stage3  )  
method (dict, fuzzy, numeric, alias, bert, freq)  
match{i}, match{i}_score, match{i}_source (for top-k matches)

#### 2.5. Demo Notebooks For Schema and Ontology Mapping

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


