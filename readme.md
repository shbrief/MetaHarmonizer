
<<<<<<< HEAD
### 1. Codebase folder structure
=======
### Codebase Directory Subtree
>>>>>>> 3167b0c (Updated the Codebase Directory Subtree in Readme.md)

```md

├── data
├── demo_nb
├── scripts
├── EDA
├── src
│   ├── models
│   │   ├── init.py
│   │   └── calc_stats.py
│   │   ├── ontology_models.py
│   │   ├── ontology_mapper_rag.py
│   │   ├── ontology_mapper_lm.py
│   │   ├── ontology_mapper_st.py
│   │   ├── schema_mapper_models.py
│   │   ├── schema_mapper_bert.py
│   │   ├── schema_mapper_lda.py
│   │   ├── method_model.yaml
│   ├── Engine
│   │   ├── ontology_mapping_engine.py
│   │   ├── schema_mapping_engine.py
│   ├── CustomLogger   
│   ├── KnowledgeDb
│   ├── Plotter
<<<<<<< HEAD
├── setup.py   
└── readme.md
└── LICENSE
```
### 2. Usage

In order to use schema and/Or ontology mapping functionality in metaharmonizer, please follow the steps below. 

#### 2.1. Environment setup

- First create a `conda env create -f environment.yml` 
- Activate the environment as `conda activate metaharmonizer`

#### 2.2. Cloning the repository

The dev version is usually updated before the main version. Hence to avail any new functionality and upto date code clone the dev branch as
```git clone -b abhi_dev https://github.com/shbrief/MetaHarmonizer```


#### 2.3 Datasets
- The datasets in this repository are encrypted in order to prevent contamination of gold standard.
- For Ontology mapping you will need a list of query_terms and a list of corpus_terms
- For Schema mapping you will provide a clinical metadata file and the schema_mapping dictionary is in the `/data` folder.
  
#### 2.4 Setting up the mappers 

1. Ontology Mapping
```
## Go into the correct directory by specifying the user_path where MetaHarmonizer was cloned
%cd <user_path>/MetaHarmonizer/

## Ontology Mapping
## Import useful packages 
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
import os
import pandas as pd
from importlib import reload

## Import the models/engine for ontology mapping
import src.models.ontology_mapper_st as om_st
import src.Engine.ontology_mapping_engine as ome
import src.models.ontology_mapper_lm as om_lm
reload(om_st)
reload(ome)
reload(om_lm)

## Now you must initialize the engine
other_params = {"test_or_prod": "test"}
onto_engine_large = ome.OntoMapEngine(method='sap-bert', topk=5,
query=query_list,corpus=small_corpus_list,
cura_map=cura_map_lcs, 
yaml_path='./src/models/method_model.yaml', 
om_strategy='lm', **other_params)

## Run the ontology mapping
results_engine_testing = onto_engine_large.run()
```
- Parameters that can be changed in the model:
  - **query(list):** list of query terms (can be 1 or Many)
  - **corpus(list):** list of corpus terms
  - **om_strategy(str):** 2 types of strategy are available 
    - strategy **lm**: This is the default strategy that uses [CLS] tokens for capturing the embedding representation. CLS is calculated in a much more intricate way, taking into account both its own embeddings (token/pos) as well as the context.
    - strategy **st**: sentence transformer based strategy calculates non- [CLS] based embeddings. 
  - **method(str):** All available models are `bert-base, pubmed-bert, bio-bert, longformer
big-bird, clinical-bert, sap-bert`. These are string keys that fetch the different transformer models found in the mapping method_model.yaml file.
  - **topk(int):** Number of top matches to return for each query term in the query list
  - **other_params(dict):** This is like a kwargs dictionary that currently only takes a value for the key **test_or_prod**. In the future if more parameters are added to the model, then it will be updated in this dictionary.
  - **cura_map(dict):** Is a dictionary of paired query and ontology terms for evaluating or testing in the 'test' environment. 

- Output: CSV file containing top 5 matches for each query term and their scores.

#### 2.5. Demo Notebooks For Schema and Ontology Mapping

The demo notebooks are located across `/demo_nb` folder
### 3. Resources for Ontology Mapping
=======
│   ├── QA
│   ├── Tests 
└── README.md
└── <jupyter_nb1.ipynb> 
└── <jupyter_nb2.ipynb>
```


### Resources for Ontology Mapping
>>>>>>> a1c1b02 (v0.2.0 updates:)
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


<<<<<<< HEAD
=======

>>>>>>> a1c1b02 (v0.2.0 updates:)
