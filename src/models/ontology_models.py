
from sentence_transformers import util
import pandas as pd
import torch
import yaml


class OntoModelsBase:
    def __init__(self, method:str, topk:int, query:list, corpus:list, yaml_path:str='method_model.yaml') -> None:    
        self.method = method
        self.query = query
        self.corpus = corpus
        if self.method is None:
            raise ValueError("Method name cannot be None")
        
        if len(self.query) == 0:
            raise ValueError("Query list cannot be empty")
        if len(self.corpus) == 0:
            raise ValueError("Corpus list cannot be empty")
        
        # Load method_model_dict from a YAML file
        self.method_model_dict = self.load_method_model_dict(yaml_path)
        self.list_of_methods = list(self.method_model_dict.keys())
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
        

        self.topk = topk 
    

#### Helper Functions Common to all OntoMap Methods####    
    def load_method_model_dict(self, file_path):
        """
        This method is responsible for loading the method_model_dict from a YAML file.
        
        Args:
            file_path (str): The path to the YAML file.
        
        Returns:
            dict: The dictionary containing the method names and their corresponding model names.
        """
        with open(file_path, 'r') as file:
            # Load the dictionary from YAML file
            method_model_dict = yaml.safe_load(file)
        return method_model_dict

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the token embeddings.

        Args:
            model_output (tuple): The output of the model, where the first element contains all token embeddings.
            attention_mask (torch.Tensor): The attention mask for the input tokens.

        Returns:
            torch.Tensor: The mean-pooled token embeddings.

        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

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

##### To be implemented in the child class #####
    def create_params(self):
        """
        This method is responsible for creating parameters for the OntoMap Method Used.
        It will be implemented in the child class.
        """
        raise NotImplementedError("create_params will be implemented in the child class")
    
    def create_embeddings(self):
        """
        This method is responsible for creating embeddings using sentence transformers, LM's or LLM's for the cBioPortal's queries or corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError("create_embeddings will be implemented in the child class")

    def get_match_results(self):
        """
        This method is responsible for generating match results for the given queries and corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError("get_match_results will be implemented in the child class")    

    def calc_consolidated_stats(self):
        """
        This method is responsible for calculating consolidated statistics for the given queries and corpus.
        It will be implemented in the child class.
        """
        raise NotImplementedError("calc_consolidated_stats will be implemented in the child class")
    
