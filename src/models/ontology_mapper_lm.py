



from sentence_transformers import util
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch
import src.Models.ontology_models as otm
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
import os
from importlib import reload

logger = src.CustomLogger.custom_logger.CustomLogger()

## type annotations for all the arguments and return values
## build a different class for each type of models - sentence transformer class, LM class and LLM class
## 3 base classes for 3 different types of models  
## Avoid building 3 different mappers for treatment, bodysite and disease and build one curaMap 

class OntoMapLM(otm.OntoModelsBase):
    def __init__(self, method:str, query:list, corpus:list, topk:int=5, from_tokenizer:bool=True, yaml_path:str='method_model.yaml') -> None:
        super().__init__(method, query, corpus, yaml_path)

        self.from_tokenizer = from_tokenizer
        self.topk = topk
        self._query_embeddings = None 
        self._corpus_embeddings = None 
        self._model = None
        self._tokenizer = None 
        self.logger = logger.custlogger(loglevel='INFO')
        self.logger.info("Initialized OntoMap Language Model module")

    @property
    def tokenizer(self):
        if self.from_tokenizer is True:
            self._tokenizer = AutoTokenizer.from_pretrained(self.method_model_dict[self.method])
            return self._tokenizer
        else:
            return None

    @property
    def model(self):
        if self.from_tokenizer is True:
            self._model = AutoModel.from_pretrained(self.method_model_dict[self.method])             
        else:
            self._model = SentenceTransformer(self.method_model_dict[self.method])
        return self._model 

    
    @property    
    def query_embeddings(self):
        if self._query_embeddings is None:
            embd = self.create_embeddings(self.query)
        return embd
        
    @property
    def corpus_embeddings(self):
        if self._corpus_embeddings is None:
            embd = self.create_embeddings(self.corpus)
        return embd 

    def create_embeddings(self, query_list:list, convert_to_tensor=False):
        """
        Function to create embeddings using SAP-BERT like LM models using first token embeddings (CLS token)

        ARGS:
            query_list: list of str items
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings

        RETURNS:
            numpy or tensor datatype for output embeddings
        """
        # Ensure tokenizer and model are loaded (assuming they are set as self.tokenizer and self.model)
        tokenizer = self.tokenizer
        model = self.model

        # Tokenize the texts and prepare input tensors
        encoded_input = tokenizer(query_list, padding="max_length", max_length=25, truncation=True, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Extract CLS token embeddings (common practice with BERT-like models)
        embeddings = model_output[0][:, 0, :]  # taking the first token (CLS token) embeddings

        if convert_to_tensor:
            return embeddings
        else:
            return embeddings.numpy()

    def create_cura_map(self, query_list:list, corpus_list:list):
        """
        Function to create embeddings for sentence transformer model 

        ARGS:
            query_list: list of str items 
            corpus_list: list of str items 

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        query_embeddings = self.create_embeddings(query_list)
        corpus_embeddings = self.create_embeddings(corpus_list)
        return query_embeddings, corpus_embeddings
    
    def get_match_results(self, cura_map:dict, topk:int=5):
        """
        Generates match results for the given queries and corpus.

        Parameters:
            top_k (int): The number of top most similar vectors to retrieve from the corpus. Default is 5.

        Returns:
            pandas.DataFrame: A DataFrame containing the match results, including the original value, curated ontology,
                              top matches, match scores, and match levels.
        """
        queries = self.query
        corpus = self.corpus
        logger_child = self.logger.getChild("get_match_results")
        logger_child.info("Creating embeddings for query_list and corpus")

        query_emb = self.create_embeddings(queries, convert_to_tensor=False)
        corpus_emb = self.create_embeddings(corpus, convert_to_tensor=False)


        logger_child.info("Calculating cosine similarity matrix")
        cosine_sim_df = self.calc_similarity(query_emb, corpus_emb)
        cosine_sim_df.columns = corpus
        cosine_sim_df.index = queries

        logger_child.info("Generating results table")
        for row in cosine_sim_df.iterrows():
            query = row[0]
            x = row[1].nlargest(topk)
            self.matches_tmp['original_value'].append(query)

            curated_value = cura_map[query]
            self.matches_tmp['curated_ontology'].append(curated_value)

            result_labels = list(row[1].nlargest(topk).index.values)
            results_vals = list(row[1].nlargest(topk).values)

            for i in range(topk):
                self.matches_tmp[f'top{i+1}_match'].append(result_labels[i])
                self.matches_tmp[f'top{i+1}_score'].append("{:.4f}".format(results_vals[i]))

            match_level = 99
            if curated_value in result_labels:
                match_level = result_labels.index(curated_value) + 1
            self.matches_tmp['match_level'].append(match_level)

        results_df = pd.DataFrame.from_dict(self.matches_tmp)
        logger_child.info("Results Generated")
        return results_df
