
from sentence_transformers import util
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import torch

class CuraModelsBase:
    def __init__(self, method, cura_map, from_tokenizer) -> None:

        self.method = method
        ## should make a dictionary of methods/model names 
        self.list_of_methods = ['bert-base', 'pubmed-bert', 'bio-bert', 'longformer', 'big-bird', 'clinical-bert']
        self.list_of_models = ['bert-base-nli-mean-tokens', 'pritamdeka/S-PubMedBert-MS-MARCO', "dmis-lab/biobert-v1.1", "yikuan8/Clinical-Longformer", "yikuan8/Clinical-BigBird", "emilyalsentzer/Bio_ClinicalBERT"]
        self.method_model_dict = dict(zip(self.list_of_methods, self.list_of_models))
        self.from_tokenizer = from_tokenizer

    ### Need to add a better check value here 
        # if self.from_tokenizer is not bool:
        #     raise ValueError("from_tokenizer should be True or False")
        
        if self.method not in self.list_of_methods:
            raise ValueError(f"Method name should be one of {self.list_of_methods}")
        
        self.matches_tmp = {
            "original_value": [],
            "curated_ontology": [],
            "match_level": [],
            "top1_match": [],
            "top1_score": [],
            "top2_match": [],
            "top2_score": [],
            "top3_match": [],
            "top3_score": [],
            "top4_match": [],
            "top4_score": [],
            "top5_match": [],
            "top5_score": [],
        }
        
        self.cura_map = cura_map
        if self.cura_map is None:
            raise ValueError("cura_map must be a dictionary of {original: curated_ontologies}")
        self.topk = 5 
        
        self._tokenizer = None
        self._model = None
        self._corpus_embeddings = None 
    
    @property
    def tokenizer(self):
        if self.from_tokenizer is True:
            self._tokenizer = AutoTokenizer.from_pretrained(self.method_model_dict[self.method])
            return self._tokenizer

    @property
    def model(self):

        if self.from_tokenizer is True:
            # self._model = AutoModelForMaskedLM.from_pretrained(self.method_model_dict[self.method])
            # self._tokenizer = AutoTokenizer.from_pretrained(self.method_model_dict[self.method])
            tokenizer = self.tokenizer
            self._model = AutoModel.from_pretrained(self.method_model_dict[self.method])
             
        if self.from_tokenizer is False:
            self._model = SentenceTransformer(self.method_model_dict[self.method])
        
        return self._model 

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def get_tokenized(self, query_list, padding=True, truncation=True, return_tensors='pt'):
        """
        Function to create tokenized input 

        ARGS:
            query_list: list of str items 
            padding: Boolean 
            truncation: Boolean
            return_tensor: 

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        tokenizer = self.tokenizer        
        # Tokenize sentences
        encoded_input = tokenizer(query_list, padding=True, truncation=True, return_tensors='pt')
        return encoded_input
    
    
    def create_embeddings_st(self, query_list, convert_to_tensor=True):
        """
        Function to create embeddings for sentence transformer model 

        ARGS:
            query_list: list of str items 
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings 

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        return self.model.encode(query_list, convert_to_tensor = convert_to_tensor)





    def create_embeddings_lm(self, query_list, convert_to_tensor=False):
        """
        Function to create embeddings for language models 

        ARGS:
            query_list: list of str items 
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings 

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        # Tokenize the texts and prepare input tensors
        tokenizer = self.tokenizer
        encoded_input = tokenizer(query_list, padding=True, truncation=True, return_tensors='pt')
        
        model = self.model 
        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Perform mean pooling
        embeddings = model_output[0]
        attention_mask = encoded_input['attention_mask']
        mask_expansion = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expansion, 1)
        sum_mask = torch.clamp(mask_expansion.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if convert_to_tensor:
            return mean_pooled_embeddings
        else:
            return mean_pooled_embeddings.numpy()

    def calc_similarity(self, query_emb, corpus_emb):
        """
        Function to create embeddings for sentence transformer model 

        ARGS:
            query_list: list of str items 
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings 

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        cosine_scores = util.cos_sim(query_emb, corpus_emb) 
        cosine_sim_df = pd.DataFrame(cosine_scores)
        return cosine_sim_df 

    def get_embedding(texts, convert_to_tensor=False):
        # Tokenize the texts and prepare input tensors
        encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        
        # Compute token embeddings with no gradient calculation
        with torch.no_grad():
            model_output = model(**encoded_input)
        
        # Mean pooling - sum embeddings across the tokens and divide by number of tokens
        embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        mask_expansion = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expansion, 1)
        sum_mask = torch.clamp(mask_expansion.sum(1), min=1e-9)
        mean_pooled_embeddings = sum_embeddings / sum_mask

        if convert_to_tensor:
            return mean_pooled_embeddings
        else:
            return mean_pooled_embeddings.numpy()
        
    
    def calc_stats(self):
        raise NotImplementedError()
    
    # def create_embeddings_lm(self, query_list, convert_to_tensor=True):
    #     encoded_input = self.get_tokenized(query_list)
    #     model = self.model
    #     # Compute token embeddings
    #     with torch.no_grad():
    #         model_output = model(**encoded_input)

    #     # Perform pooling. In this case, max pooling.
    #     sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    #     return sentence_embeddings


