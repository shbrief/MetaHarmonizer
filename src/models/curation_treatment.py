
from sentence_transformers import util
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import src.models.curation_models as curmodels
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
import seaborn as sns
from io import BytesIO
import os
from importlib import reload

logger = src.CustomLogger.custom_logger.CustomLogger()


class CuraTreatment(curmodels.CuraModelsBase):
    def __init__(self, method, cura_map, from_tokenizer) -> None:
        super().__init__(method, cura_map, from_tokenizer)

        self.from_tokenizer = from_tokenizer
        self.cura_map = cura_map
        self._query_embeddings = None 
        self._cura_embeddings = None 
        self.logger = logger.custlogger(loglevel='DEBUG')
        self.logger.debug("Initialized CuraTreatment module")

    def get_cura(self):
        cura = []
        curated_onto = list(np.unique(list(self.cura_map.values())))
        for x in curated_onto:
            cura.extend(x.split(';'))
        return cura 
    
    def get_queries(self):
        orig = []
        orig_onto = list(np.unique(list(self.cura_map.keys())))
        for x in orig_onto:
            orig.extend(x.split('<;>')) # separate rows containing multiple values
        return orig 

    @property    
    def query_embeddings(self):
        if self._query_embeddings is None:
            if self.from_tokenizer is False:
                embd = self.create_embeddings_st(self.get_queries())
            else:
                embd = self.create_embeddings_lm(self.get_queries()) 
            return embd
        
    @property
    def cura_embeddings(self):
        if self.cura_embeddings is None:
            if self.from_tokenizer is False:
                embd = self.create_embeddings_st(self.get_cura())
            else:
                embd = self.create_embeddings_lm(self.get_cura()) 
            return embd 
        
    def get_match_results(self, top_k = 5):
          # the number of top most similar vectors from the corpus
        queries = self.get_queries()
        corpus = self.get_cura()
        logger_child = self.logger.getChild("get_match_results")
        logger_child.info("Creating embeddings for query_list and corpus")
        if self.from_tokenizer is False:
            query_emb = self.create_embeddings_st(queries, convert_to_tensor=False)
            corpus_emb = self.create_embeddings_st(corpus, convert_to_tensor=False)
        else:
            query_emb = self.create_embeddings_lm(queries, convert_to_tensor=False)
            corpus_emb = self.create_embeddings_lm(corpus, convert_to_tensor=False)           

        logger_child.info("Calculating cosine similarity matrix")
        cosine_sim_df = self.calc_similarity(query_emb, corpus_emb)
        cosine_sim_df.columns = corpus
        cosine_sim_df.index = queries

        logger_child.info("Generating results table")
        for row in cosine_sim_df.iterrows():
            query = row[0]
            x = row[1].nlargest(5)
            self.matches_tmp['original_value'].append(query)

            curated_value = self.cura_map[query]
            self.matches_tmp['curated_ontology'].append(curated_value)

            result_labels = list(row[1].nlargest(5).index.values)
            results_vals = list(row[1].nlargest(5).values)

            for i in range(top_k):
                self.matches_tmp[f'top{i+1}_match'].append(result_labels[i])
                self.matches_tmp[f'top{i+1}_score'].append("{:.4f}".format(results_vals[i]) )
        
            match_level = 99
            if curated_value in result_labels:
                match_level = result_labels.index(curated_value) + 1
                # print(f"Top {match_level} match!") 
            self.matches_tmp['match_level'].append(match_level)
        results_df = pd.DataFrame.from_dict(self.matches_tmp)
        logger_child.info("Results Generated")
        return results_df
