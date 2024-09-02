
from sentence_transformers import util
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import src.models.ontology_models as otm
import src.CustomLogger.custom_logger



logger = src.CustomLogger.custom_logger.CustomLogger()

## type annotations for all the arguments and return values
## build a different class for each type of models - sentence transformer class, LM class and LLM class
## 3 base classes for 3 different types of models  
## Avoid building 3 different mappers for treatment, bodysite and disease and build one curaMap 
class OntoMapST(otm.OntoModelsBase):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 11a0b3b (Final updates to schema mapper with schema_map  demo)
=======
>>>>>>> 1c20785 (comitting unstaged changes for merging)
    
    def __init__(self, method: str, topk: int, query: list, corpus: list, from_tokenizer:bool=False, yaml_path: str = 'method_model.yaml') -> None:
=======
    """
    A class to map ontologies using sentence transformers.

    Attributes:
        method (str): The method to use for the model.
        query (list[str]): The list of query strings.
        corpus (list[str]): The list of corpus strings.
        topk (int): The number of top results to consider.
        from_tokenizer (bool): Whether to use a tokenizer.
        yaml_path (str): Path to the YAML configuration file.
        _query_embeddings (torch.Tensor or None): Embeddings for the queries.
        _corpus_embeddings (torch.Tensor or None): Embeddings for the corpus.
        _model (AutoModel or SentenceTransformer or None): The model instance.
        _tokenizer (AutoTokenizer or None): The tokenizer instance.
        logger (CustomLogger): Logger instance.
    """

    def __init__(self, method: str, topk: int, query: list, corpus: list, from_tokenizer: bool = False, yaml_path: str = 'method_model.yaml') -> None:
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        """
        Initializes the OntoMapST class.

        Args:
<<<<<<< HEAD
            self (undefined):
            method (str):
            topk (int):
            query (list):
            corpus (list):
            from_tokenizer (bool=False):
            yaml_path (str='method_model.yaml'):

        Returns:
            None

=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
    """
    A class to map ontologies using sentence transformers.

    Attributes:
        method (str): The method to use for the model.
        query (list[str]): The list of query strings.
        corpus (list[str]): The list of corpus strings.
        topk (int): The number of top results to consider.
        from_tokenizer (bool): Whether to use a tokenizer.
        yaml_path (str): Path to the YAML configuration file.
        _query_embeddings (torch.Tensor or None): Embeddings for the queries.
        _corpus_embeddings (torch.Tensor or None): Embeddings for the corpus.
        _model (AutoModel or SentenceTransformer or None): The model instance.
        _tokenizer (AutoTokenizer or None): The tokenizer instance.
        logger (CustomLogger): Logger instance.
    """

    def __init__(self, method: str, topk: int, query: list, corpus: list, from_tokenizer: bool = False, yaml_path: str = 'method_model.yaml') -> None:
=======
            method (str): The method to use for the model.
            topk (int): The number of top results to consider.
            query (list): The list of query strings.
            corpus (list): The list of corpus strings.
            from_tokenizer (bool, optional): Whether to use a tokenizer. Defaults to False.
            yaml_path (str, optional): Path to the YAML configuration file. Defaults to 'method_model.yaml'.
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
    
    def __init__(self, method: str, topk: int, query: list, corpus: list, from_tokenizer:bool=False, yaml_path: str = 'method_model.yaml') -> None:
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
        """
        Description of __init__

        Args:
<<<<<<< HEAD
            method (str): The method to use for the model.
            topk (int): The number of top results to consider.
            query (list): The list of query strings.
            corpus (list): The list of corpus strings.
            from_tokenizer (bool, optional): Whether to use a tokenizer. Defaults to False.
            yaml_path (str, optional): Path to the YAML configuration file. Defaults to 'method_model.yaml'.
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
            self (undefined):
            method (str):
            topk (int):
            query (list):
            corpus (list):
            from_tokenizer (bool=False):
            yaml_path (str='method_model.yaml'):

        Returns:
            None

>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
<<<<<<< HEAD
=======
    def __init__(self, method:str, topk, query:list, corpus:list, cura_map:dict, from_tokenizer:bool=False, yaml_path:str='method_model.yaml') -> None:
>>>>>>> 5c7ccb3 (v0.1.6 updates: 5 patches made to ontology mapping)
=======
    def __init__(self, method: str, topk: int, query: list, corpus: list, from_tokenizer:bool=False, yaml_path: str = 'method_model.yaml') -> None:
>>>>>>> b224039 (v0.2.4 updates:)
=======
>>>>>>> 11a0b3b (Final updates to schema mapper with schema_map  demo)
        super().__init__(method, topk, query, corpus, yaml_path)

        self.from_tokenizer = from_tokenizer
        self._query_embeddings = None 
        self._corpus_embeddings = None 
        self._model = None
        self._tokenizer = None 
        self.logger = logger.custlogger(loglevel='INFO')
        self.logger.info("Initialized OntoMap Sentence Transformer module")

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

    def create_embeddings(self, query_list:list, convert_to_tensor:bool=True):
        """
        Function to create embeddings for sentence transformer model 

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        ARGS:
            query_list: list of str items 
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings 
=======
        Args:
            query_list (list[str]): List of query strings.
            convert_to_tensor (bool, optional): Whether to convert the output embeddings to tensor. Defaults to False.
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
        ARGS:
            query_list: list of str items 
            convert_to_tensor: boolean value for numpy or tensor datatype for output embeddings 
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)

        RETURNS:
            numpy or tensor datatype for output embeddings
         """
        return self.model.encode(query_list, convert_to_tensor = convert_to_tensor)

    def get_match_result_single_query(self, query:str, cosine_sim_df:pd.DataFrame):
        """
        Generates match results for a single query using cosine_sim_df
        """
        raise NotImplementedError("get_match_result_single_query will be implemented later")
    
    def get_match_results_mp(self):
        """
        Generates match results for the given queries and corpus using multiprocessing
        """
        raise NotImplementedError("get_match_results_mp will be implemented later")
       
    def get_match_results(self, cura_map:dict[str, str]=None, topk:int=5, test_or_prod:str='test'):
        """
        Generates match results for the given queries and corpus.

        Parameters:
            top_k (int): The number of top most similar vectors to retrieve from the corpus. Default is 5.

        Returns:
            pandas.DataFrame: A DataFrame containing the match results, including the original value, curated ontology,
                              top matches, match scores, and match levels.
        """
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        Args:
            query_list (list[str]): List of query strings.
            convert_to_tensor (bool, optional): Whether to convert the output embeddings to tensor. Defaults to False.

        Returns:
            numpy.ndarray or torch.Tensor: The output embeddings.
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

    def create_cura_map(self, query_list: list[str], corpus_list: list[str]):
        """
        Creates embeddings for sentence transformer model.

        Args:
            query_list (list[str]): List of query strings.
            corpus_list (list[str]): List of corpus strings.

        Returns:
            tuple: A tuple containing query embeddings and corpus embeddings.
        """
        query_embeddings = self.create_embeddings(query_list)
        corpus_embeddings = self.create_embeddings(corpus_list)
        return query_embeddings, corpus_embeddings
    
    def get_match_result_single_query(self, query: str, cosine_sim_df: pd.DataFrame):
        """
        Generates match results for a single query using cosine similarity DataFrame.

        Args:
            query (str): The query string.
            cosine_sim_df (pd.DataFrame): DataFrame containing cosine similarity scores.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("get_match_result_single_query will be implemented later")
    
    def get_match_results_mp(self):
        """
        Generates match results for the given queries and corpus using multiprocessing.

        Raises:
            NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError("get_match_results_mp will be implemented later")
        
    def get_match_results(self, cura_map: dict[str, str] = None, topk: int = 5, test_or_prod: str = 'test') -> pd.DataFrame:
        """
        Generates match results for the given queries and corpus.

        Args:
            cura_map (dict[str, str], optional): A dictionary mapping queries to curated values. Required for 'test' mode. Defaults to None.
            topk (int, optional): The number of top most similar vectors to retrieve from the corpus. Defaults to 5.
            test_or_prod (str, optional): Mode of operation, either 'test' or 'prod'. Defaults to 'test'.

        Returns:
            pd.DataFrame: A DataFrame containing the match results, including the original value, curated ontology,
                          top matches, match scores, and match levels.

        Raises:
            ValueError: If cura_map is not provided in 'test' mode.
        """
        if test_or_prod == 'test':
            if cura_map is None:
                raise ValueError("cura_map should be provided for test mode")
            
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
            topk_vals = row[1].nlargest(topk)
            self.matches_tmp['original_value'].append(query)
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> b224039 (v0.2.4 updates:)

            if test_or_prod == 'test':
                if query in cura_map.keys():
                    curated_value = cura_map[query]
                else:
                    curated_value = "Not Found"
<<<<<<< HEAD
            else:
                curated_value = "Not Available for Prod Environment"
<<<<<<< HEAD
<<<<<<< HEAD
=======
            
            if query in cura_map.keys():
                curated_value = cura_map[query]
            else:
                curated_value = "Not Found"
>>>>>>> 5c7ccb3 (v0.1.6 updates: 5 patches made to ontology mapping)
=======
            else:
                curated_value = "Not Available for Prod Environment"
>>>>>>> b224039 (v0.2.4 updates:)
=======
            if test_or_prod == 'test':
                if query in cura_map.keys():
                    curated_value = cura_map[query]
                else:
                    curated_value = "Not Found"
            else:
                curated_value = "Not Available for Prod Environment"
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
            self.matches_tmp['curated_ontology'].append(curated_value)

            result_labels = list(topk_vals.nlargest(topk).index.values)
            results_vals = list(topk_vals.nlargest(topk).values)

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
