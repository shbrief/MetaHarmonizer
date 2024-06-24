from src.models import ontology_mapper_st as oms
from src.models import ontology_mapper_lm as oml
from src.models import ontology_mapper_llm as ollm
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
from thefuzz import fuzz

logger = src.CustomLogger.custom_logger.CustomLogger()



class OntoMapEngine():
    def __init__(self, 
                 method:str, 
                 query:list[str], 
                 corpus:list[str],
                 cura_map:dict, 
                 topk:int=5, 
                 from_tokenizer:bool=True, 
                 yaml_path:str='method_model.yaml', 
                 om_strategy='st') -> None:
        """
        This class is responsible for initializing the OntoMapEngine object.

        Args:
            method (str): The name of the method.
            query (list): The list of queries.
            corpus (list): The list of corpus.
            topk (int): The number of top matches to return.
            from_tokenizer (bool): A flag to indicate if the model is loaded from a tokenizer.
            yaml_path (str): The path to the YAML file.
        """
        self.method = method
        self.query = query
        self.corpus = corpus
        self.topk = topk
        self.om_strategy = om_strategy
        self.from_tokenizer = from_tokenizer
        self.yaml_path = yaml_path
        self.cura_map = cura_map

        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized OntoMap Engine module")


    def _exact_matching(self):
        """
        This function is responsible for performing exact matching.

        Here we want to separate exact matches of query to large corpus 
        Returns:
            list: The list of exact matches from query 
        """
        return [q for q in self.query if q in self.corpus] 
    
    def _fuzzy_matching(self, fuzz_ratio:int=80):
        """
        This function is responsible for performing fuzzy matching.

        Here we want to separate fuzzy matches of query to large corpus 
        Returns:
           list(str): The list of exact matches from query  
        """
        return [q for q in self.query if fuzz.partial_ratio(q, self.corpus) > fuzz_ratio]

    def _om_model_from_strategy(self):
        """
        This function is responsible for returning the OntoMap model based on the strategy
        """
        if self.om_strategy == 'st':
            return oms.OntoMapST(self.method, self.query, self.corpus, self.topk, self.from_tokenizer, self.yaml_path)
        elif self.om_strategy == 'llm':
            raise NotImplementedError("OntoMap LLM is not implemented yet")
        else:
            raise ValueError("om_strategy should be either 'st' or 'llm'")
            
    def separate_matches(self, matching_type:str='exact'):
        """
        This function is responsible for separating exact and non-exact matches

        So that we can then run LM or sentence transformer models on the non-exact matches

        Returns:
            list: The list of non-exact matches

        """
        if matching_type not in ['exact', 'fuzzy']:
            raise ValueError("Matching type should be either 'exact' or 'fuzzy'")
        
        if matching_type == 'exact':
            to_separate_matches = self._exact_matching()
        elif matching_type == 'fuzzy':
            to_separate_matches = self._fuzzy_matching()

        non_exact_matches = np.setdiff1d(self.query, to_separate_matches)
        return non_exact_matches

    def get_match_results(self, cura_map:dict, topk:int=5):
        """
        This function is responsible for retrieving the match results for the given cura_map and top_k value.

        Parameters:
        cura_map (dict): A dictionary containing the mapping of queries to curated values.
        top_k (int, optional): The number of top matches to retrieve. Defaults to 5.
        """
        onto_map = self._om_model_from_strategy()
        return onto_map.get_match_results(cura_map, topk)
    

    def run(self):
        """
        This function is responsible for running the OntoMap Engine module.
        """

        self._logger.custlogger.info("Running Ontology Mapping")
        self._logger.custlogger.info("Separating exact and non-exact matches")
        exact_matches = self._exact_matching()
        non_exact_matches = self.separate_matches(matching_type='exact')
        
        self._logger.custlogger.info("Running OntoMap model for non-exact matches")
        onto_map_res = self.get_match_results(self.cura_map, topk=self.topk)
        return onto_map_res
