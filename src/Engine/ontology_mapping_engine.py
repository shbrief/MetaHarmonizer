from src.models import ontology_mapper_st as oms
from src.models import ontology_mapper_lm as oml
from src.models import ontology_mapper_rag as ollm
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
from thefuzz import fuzz
logger = src.CustomLogger.custom_logger.CustomLogger()


<<<<<<< HEAD
class OntoMapEngine():
    def __init__(self, 
                 method:str, 
                 query:list[str], 
                 corpus:list[str],
                 cura_map:dict, 
                 topk:int=5, 
                 yaml_path:str='method_model.yaml', 
                 om_strategy:str='lm',
                 **other_params:dict[str,]) -> None:
        """
        This class is responsible for initializing the OntoMapEngine object.
        
        om_strategy options:
        st: sentence transformer models with .encode() method from pre-trained models 
        lm: language models with no .encode() method, you will have to create embeddings from the first token (CLS token) embeddings
        rag: LLM based embeddings and RAG framework for scoring the top matches 

        Args:
            method (str): The name of the method.
            query (list): The list of queries.
            corpus (list): The list of corpus.
            topk (int): The number of top matches to return.
            from_tokenizer (bool): A flag to indicate if the model is loaded from a tokenizer.
            yaml_path (str): The path to the YAML file.
            om_strategy (str): The strategy to use for OntoMap.
            other_params (dict): Other parameters to pass to the engine - particularly for RAG model or for specifying test/prod environment.
=======
class OntoMapEngine:
    """
    A class to initialize and run the OntoMapEngine for ontology mapping.

    Attributes:
        method (str): The name of the method.
        query (list[str]): The list of queries.
        corpus (list[str]): The list of corpus.
        cura_map (dict): The dictionary containing the mapping of queries to curated values.
        topk (int): The number of top matches to return.
        yaml_path (str): The path to the YAML file.
        om_strategy (str): The strategy to use for OntoMap.
        other_params (dict): Other parameters to pass to the engine.
        _test_or_prod (str): Indicates whether the environment is test or production.
        _logger (CustomLogger): Logger instance.
    """

    def __init__(self, 
                 method: str, 
                 query: list[str], 
                 corpus: list[str],
                 cura_map: dict, 
                 topk: int = 5, 
                 yaml_path: str = 'method_model.yaml', 
                 om_strategy: str = 'lm',
                 **other_params: dict[str,]) -> None:
        """
        Initializes the OntoMapEngine class.

        Args:
            method (str): The name of the method.
            query (list[str]): The list of queries.
            corpus (list[str]): The list of corpus.
            cura_map (dict): The dictionary containing the mapping of queries to curated values.
            topk (int, optional): The number of top matches to return. Defaults to 5.
            yaml_path (str, optional): The path to the YAML file. Defaults to 'method_model.yaml'.
            om_strategy (str, optional): The strategy to use for OntoMap. Defaults to 'lm'.
            other_params (dict, optional): Other parameters to pass to the engine.
>>>>>>> fd1285a (comitting unstaged changes for merging)
        """
        self.method = method
        self.query = query
        self.corpus = corpus
        self.topk = topk
        self.om_strategy = om_strategy
<<<<<<< HEAD
        # self.from_tokenizer = from_tokenizer
=======
>>>>>>> fd1285a (comitting unstaged changes for merging)
        self.yaml_path = yaml_path
        self.cura_map = cura_map
        self.other_params = other_params
        if 'test_or_prod' not in self.other_params.keys():
            raise ValueError("test_or_prod value must be defined in other_params dictionary")
        
        self._test_or_prod = self.other_params['test_or_prod'] 
        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized OntoMap Engine module")

<<<<<<< HEAD

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
=======
    def _exact_matching(self):
        """
        Performs exact matching of queries to the corpus.

        Returns:
            list: The list of exact matches from the query.
        """
        return [q for q in self.query if q in self.corpus] 
    
    def _fuzzy_matching(self, fuzz_ratio: int = 80):
        """
        Performs fuzzy matching of queries to the corpus.

        Args:
            fuzz_ratio (int, optional): The fuzz ratio threshold for matching. Defaults to 80.

        Returns:
            list[str]: The list of fuzzy matches from the query.
>>>>>>> fd1285a (comitting unstaged changes for merging)
        """
        return [q for q in self.query if fuzz.partial_ratio(q, self.corpus) > fuzz_ratio]

    def _om_model_from_strategy(self, non_exact_query_list: list[str]):
        """
<<<<<<< HEAD
        This function is responsible for returning the OntoMap model based on the strategy
=======
        Returns the OntoMap model based on the strategy.

        Args:
            non_exact_query_list (list[str]): The list of non-exact queries.

        Returns:
            object: The OntoMap model instance.

        Raises:
            NotImplementedError: If the 'rag' strategy is selected.
            ValueError: If an invalid strategy is provided.
>>>>>>> fd1285a (comitting unstaged changes for merging)
        """
        if self.om_strategy == 'lm':
            return oml.OntoMapLM(method=self.method, query=non_exact_query_list, corpus=self.corpus, topk=self.topk, from_tokenizer=True, 
                                 yaml_path=self.yaml_path)
<<<<<<< HEAD
            
        elif self.om_strategy == 'st':
            return oms.OntoMapST(method=self.method, query=non_exact_query_list, corpus=self.corpus,topk=self.topk, from_tokenizer=False, 
=======
        elif self.om_strategy == 'st':
            return oms.OntoMapST(method=self.method, query=non_exact_query_list, corpus=self.corpus, topk=self.topk, from_tokenizer=False, 
>>>>>>> fd1285a (comitting unstaged changes for merging)
                                 yaml_path=self.yaml_path)
        elif self.om_strategy == 'rag':
            raise NotImplementedError("OntoMap LLM is not implemented yet")
        else:
            raise ValueError("om_strategy should be either 'st', 'lm' or 'rag'")
            
<<<<<<< HEAD
    def _separate_matches(self, matching_type:str='exact'):
        """
        This function is responsible for separating exact and non-exact matches

        So that we can then run LM or sentence transformer models on the non-exact matches

        Returns:
            list: The list of non-exact matches

=======
    def _separate_matches(self, matching_type: str = 'exact'):
        """
        Separates exact and non-exact matches.

        Args:
            matching_type (str, optional): The type of matching to perform ('exact' or 'fuzzy'). Defaults to 'exact'.

        Returns:
            list: The list of non-exact matches.

        Raises:
            ValueError: If an invalid matching type is provided.
>>>>>>> fd1285a (comitting unstaged changes for merging)
        """
        if matching_type not in ['exact', 'fuzzy']:
            raise ValueError("Matching type should be either 'exact' or 'fuzzy'")
        
        if matching_type == 'exact':
            to_separate_matches = self._exact_matching()
        elif matching_type == 'fuzzy':
            to_separate_matches = self._fuzzy_matching()

        non_exact_matches = list(np.setdiff1d(self.query, to_separate_matches))
        return non_exact_matches

<<<<<<< HEAD
    def get_results_for_non_exact(self, non_exact_query_list: list[str], topk:int=5):
        """
        This function is responsible for retrieving the match results for the given cura_map and top_k value.

        Parameters:
        cura_map (dict): A dictionary containing the mapping of queries to curated values.
        top_k (int, optional): The number of top matches to retrieve. Defaults to 5.
        """
        
=======
    def get_results_for_non_exact(self, non_exact_query_list: list[str], topk: int = 5):
        """
        Retrieves the match results for the given non-exact queries.

        Args:
            non_exact_query_list (list[str]): The list of non-exact queries.
            topk (int, optional): The number of top matches to retrieve. Defaults to 5.

        Returns:
            pd.DataFrame: The match results.
        """
>>>>>>> fd1285a (comitting unstaged changes for merging)
        onto_map = self._om_model_from_strategy(non_exact_query_list=non_exact_query_list)
        return onto_map.get_match_results(cura_map=self.cura_map, topk=topk, test_or_prod=self._test_or_prod)    

    def run(self):
        """
<<<<<<< HEAD
        This function is responsible for running the OntoMap Engine module.
        """

=======
        Runs the OntoMap Engine module.

        Returns:
            tuple: A tuple containing the exact matches and the OntoMap results for non-exact matches.
        """
>>>>>>> fd1285a (comitting unstaged changes for merging)
        self._logger.info("Running Ontology Mapping")
        self._logger.info("Separating exact and non-exact matches")
        exact_matches = self._exact_matching()
        non_exact_matches_ls = self._separate_matches(matching_type='exact')
        
        self._logger.info("Running OntoMap model for non-exact matches")
        onto_map_res = self.get_results_for_non_exact(non_exact_query_list=non_exact_matches_ls, topk=self.topk)
<<<<<<< HEAD
        return exact_matches, onto_map_res
=======
        return exact_matches, onto_map_res
>>>>>>> fd1285a (comitting unstaged changes for merging)
