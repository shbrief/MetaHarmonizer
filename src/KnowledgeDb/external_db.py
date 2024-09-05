import requests
import json
import logging 
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pathos.multiprocessing import ProcessingPool as Pool

class UMLSDb:
    """
    A class to interact with the UMLS database.

    Attributes:
        api_key (str): The API key for accessing UMLS.
        _base_url (str): The base URL for UMLS API.
        _version (str): The version of the UMLS API.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initializes the UMLSDb class with the provided API key.

        Args:
            api_key (str): The API key for accessing UMLS.
        """
        self.api_key = api_key
        self._base_url = 'https://uts-ws.nlm.nih.gov/rest'
        self._version = 'current'
    
    def search_single_term(self, version: str = None, string_query: str = None, searchType: str = 'normalizedWords'):
        """
        Searches for a single term in the UMLS database.

        Args:
            version (str, optional): The version of the UMLS API. Defaults to None.
            string_query (str, optional): The query string to search for. Defaults to None.
            searchType (str, optional): The type of search to perform. Defaults to 'normalizedWords'.

        Returns:
            dict: The search results from the UMLS API.
        """
        version = self._version
        string = string_query
        content_endpoint = "/rest/search/" + version
        full_url = self._base_url + content_endpoint
        page = 0
        
        try:
            while True:
                page += 1
                query = {'string': string, 'apiKey': self.api_key, 'pageNumber': page, 'searchType': searchType}
                r = requests.get(full_url, params=query)
                r.raise_for_status()
                print(r.url)
                r.encoding = 'utf-8'
                outputs = r.json()
                items = (([outputs['result']])[0])['results']
                
                if len(items) == 0:
                    if page == 1:
                        print('No results found.\n')
                        break
                    else:
                        break
                
                print("Results for page " + str(page) + "\n")
                
                for result in items:
                    print('UI: ' + result['ui'])
                    print('URI: ' + result['uri'])
                    print('Name: ' + result['name'])
                    print('Source Vocabulary: ' + result['rootSource'])
                    print('\n')
                return outputs
            print('*********')

        except Exception as except_error:
            print(except_error)
            
    def get_nci_code_by_term(self, term: str):
        """
        Retrieves the NCI code for a given term.

        Args:
            term (str): The term to search for.

        Returns:
            tuple: A tuple containing the search results and a list of NCI codes.
        """
        version = "current"
        uri = "https://uts-ws.nlm.nih.gov"
        content_endpoint = "/rest/search/" + version
        full_url = uri + content_endpoint
        page = 0
        
        try:
            while True:
                page += 1
                query = {'string': term, 'apiKey': self.api_key, 'pageNumber': page, 'searchType': 'exact', 'sabs': 'NCI'}
                query['returnIdType'] = ['code']
   
                r = requests.get(full_url, params=query)
                r.raise_for_status()
                print(r.url)
                r.encoding = 'utf-8'
                outputs = r.json()
                items = (([outputs['result']])[0])['results']
                
                if len(items) == 0:
                    if page == 1:
                        print('No results found.\n')
                        break
                    else:
                        break
                
                print("Results for page " + str(page) + "\n")
                
                for result in items:
                    print('UI: ' + result['ui'])
                    print('URI: ' + result['uri'])
                    print('Name: ' + result['name'])
                    print('Source Vocabulary: ' + result['rootSource'])
                    print('\n')
                ui_list = []
                for res in items:
                    if res['rootSource'] == 'NCI':
                        ui_list.append(res['ui'])
                return items, ui_list

            print('*********')

        except Exception as except_error:
            print(except_error)
            
class NCIDb:
    """
    A class to interact with the NCI database.

    Attributes:
        _base_url (str): The base URL for NCI API.
        _umls_api_key (str): The API key for accessing UMLS.
        _umls_db (UMLSDb): An instance of the UMLSDb class.
        _terminology (str): The terminology to use for NCI.
    """

    def __init__(self) -> None:
        """
        Initializes the NCIDb class.
        """
        self._base_url = "https://api-evsrest.nci.nih.gov/api/v1"
        self._umls_api_key = "acf85d30-60c7-4b65-97c1-07e6df7c624a"
        self._umls_db = UMLSDb(self._umls_api_key)
        self._terminology = "ncit"
        
    def get_synonyms(self, ncit_code: str, limit: int, n_children: int):
        """
        Retrieves synonyms for a given NCIT code.

        Args:
            ncit_code (str): The NCIT code to search for.
            limit (int): The limit on the number of results.
            n_children (int): The number of children to include.

        Returns:
            dict: The synonyms for the given NCIT code.
        """
        url = f"{self._base_url}/concept/{self._terminology}/{ncit_code}?limit={limit}&include=definitions%2C%20synonyms%2C%{n_children}children"
        response = requests.get(url)
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print

    def get_synonyms_by_term(self, term: str):
        """
        Retrieves synonyms for a given term.

        Args:
            term (str): The term to search for.

        Returns:
            dict: The synonyms for the given term.
        """
        url = self._base_url + f"/concept/ncit/search?terminology=ncit&term={term}&synonymType=FULL_SYN"
        response = requests.get(url)
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print

    def get_minimal_concept_list_by_code(self, list_of_codes: list[str]):
        """
        Retrieves minimal concept information for a list of codes.

        Args:
            list_of_codes (list[str]): The list of codes to search for.

        Returns:
            dict: The minimal concept information for the given codes.
        """
        ls_str = ','.join(list_of_codes)
        logging.info(f"Get minimal concepts by list - {ls_str}")
        logging.info("url = " + self._base_url + f"/concept/ncit?include=minimal&list={ls_str}")
        response = requests.get(self._base_url + f"/concept/ncit?include=minimal&list={ls_str}")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print

    def get_concept_by_search_term_match(self, term: str):
        """
        Retrieves concepts matching a search term with a search type of "match".

        Args:
            term (str): The term to search for.

        Returns:
            dict: The concepts matching the search term.
        """
        logging.info("Get concepts matching a search term within a specified terminology and a search type of 'match'.")
        logging.info("url = " + self._base_url + f"/concept/ncit/search?terminology=ncit&term={term}&term=match")
        response = requests.get(self._base_url + f"/concept/ncit/search?terminology=ncit&term={term}&term=match")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print

    def get_concept_by_search_term_highlights(self, term: str):
        """
        Retrieves concepts matching a search term and includes synonyms and highlighted text in the response.

        Args:
            term (str): The term to search for.

        Returns:
            dict: The concepts matching the search term with highlights.
        """
        logging.info("Get concepts matching a search term within a specified terminology and include synonyms and highlighted text in the response.")
        logging.info("url = " + self._base_url + f"/concept/ncit/search?terminology=ncit&term={term}&include=synonyms,highlights")
        response = requests.get(self._base_url + f"/concept/ncit/search?terminology=ncit&term={term}&include=synonyms,highlights")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print   

    def get_property_by_code_or_label(self, term_list: list[str]):
        """
        Retrieves properties for the specified codes or labels.

        Args:
            term_list (list[str]): The list of terms to search for.

        Returns:
            dict: The properties for the given terms.
        """
        term_ls_str = ','.join(term_list)
        logging.info(f"Get Property By Code or Label - {term_ls_str}")
        logging.info("url = " + self._base_url + f"/metadata/ncit/properties?include=summary&list={term_ls_str}")
        response = requests.get(self._base_url + f"/metadata/ncit/properties?include=summary&list={term_ls_str}")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print
    
    def get_summary_concept_by_code(self, term_code: str):
        """
        Retrieves summary information for a specified code.

        Args:
            term_code (str): The code to search for.

        Returns:
            dict: The summary information for the given code.
        """
        logging.info(f"Get summary concepts by code - {term_code}")
        logging.info("url = " + self._base_url + f"/concept/ncit/{term_code}?include=summary")
        response = requests.get(self._base_url + f"/concept/ncit/{term_code}?include=summary&list=C3224")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print
    
    def get_custom_concept_by_code(self, term_code: str, list_of_concepts: list[str]):
        """
        Retrieves custom concept information for a given terminology and code.

        Args:
            term_code (str): The code to search for.
            list_of_concepts (list[str]): The list of concepts to include.

        Returns:
            tuple: A tuple containing the custom concept information and the response.
        """
        ls_concept_str = ','.join(list_of_concepts)
        logging.info(f"Get custom concepts by code - {term_code}")
        logging.info("url = " + self._base_url + f"/concept/ncit/{term_code}?include={ls_concept_str}")
        response = requests.get(self._base_url + f"/concept/ncit/{term_code}?include={ls_concept_str}")
        assert response.status_code == requests.codes.ok
        pretty_print = json.loads(response.text)
        logging.info(json.dumps(pretty_print, indent=2))
        return pretty_print, response

    def get_context_list(self, string: str):
        """
        Retrieves the context list for a given string.

        Args:
            string (str): The string to search for.

        Returns:
            list: The context list for the given string.
        """
        code = self._umls_db.get_nci_code_by_term(string)[1][0]
        concepts = self.get_custom_concept_by_code(code, list_of_concepts=['synonyms', 'children', 'roles', 'definitions', 'parents'])
        return self.create_context_list(concepts[0])

    def create_context_list_mp(self, nci_codes: list[str]):
        """
        Creates a context list using multiprocessing.

        Args:
            nci_codes (list[str]): The list of NCI codes.

        Returns:
            list: The context list.
        """
        pool = Pool(processes=12)
        context_list = pool.map(self.get_context_list, nci_codes)
        return context_list
    
    def create_context_list(self, concepts_for_curated_term2: dict[str]):
        """
        Creates a context list from the given concepts.

        Args:
            concepts_for_curated_term2 (dict[str]): The concepts for the curated term.

        Returns:
            str: The context list as a string.
        """
        context_list = []
        concepts = ['synonyms', 'children', 'roles', 'definitions', 'parents']
        for concept in concepts:
            if concept in concepts_for_curated_term2:
                if concept == 'synonyms':
                    str_tmp = 'The synonyms for the term are:'
                    for syn in concepts_for_curated_term2[concept]:
                        str_tmp += syn['name'] + ', '
                    context_list.append(str_tmp)
                elif concept == 'children':
                    str_tmp = 'The children of the term are:'
                    for child in concepts_for_curated_term2[concept]:
                        str_tmp += child['name'] + ', '    
                    context_list.append(str_tmp)
                elif concept == 'parents':
                    str_tmp = 'The parents of the term are:'
                    for parent in concepts_for_curated_term2[concept]:
                        str_tmp += parent['name'] + ', '
                    context_list.append(str_tmp)
                elif concept == 'roles':
                    str_tmp = 'The roles of the term are:'
                    for role in concepts_for_curated_term2[concept]:
                        str_tmp += role['type'] + ' ' + role['relatedName'] + ', '
                    context_list.append(str_tmp)
                elif concept == 'definitions':
                    str_tmp = 'The definitions for the term are:'
                    for definition in concepts_for_curated_term2[concept]:
                        str_tmp += definition['definition'] + ', '    
                    context_list.append(str_tmp)
        
        return '.'.join(context_list)
    
class MongoDBUtils:
    """
    A class to interact with MongoDB.

    Attributes:
        conn_str (str): The connection string for MongoDB.
        _client (MongoClient): The MongoDB client.
        _db (Database): The MongoDB database.
        _atlas_search_index_name (str): The name of the Atlas search index.
        collection (Collection): The MongoDB collection.
    """

    def __init__(self, connection_str: str, db: str, collection: str = None) -> None:
        """
        Initializes the MongoDBUtils class.

        Args:
            connection_str (str): The connection string for MongoDB.
            db (str): The name of the database.
            collection (str, optional): The name of the collection. Defaults to None.
        """
        self.conn_str = connection_str
        self._client = MongoClient(self.conn_str, server_api=ServerApi('1'))
        self._db = self._client[db]
        self._atlas_search_index_name = "RAGDataIndex"
        if collection is not None:
            self.collection = self._db[collection]
        else:
            self.collection = None
            
    def ping(self):
        """
        Sends a ping to confirm a successful connection to MongoDB.
        """
        try:
            self._client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def create_collection(self, collection_name: str, validator: dict = None):
        """
        Creates a collection in MongoDB.

        Args:
            collection_name (str): The name of the collection.
            validator (dict, optional): The validator for the collection. Defaults to None.
        """
        if collection_name not in self._db.list_collection_names():
            if validator:
                self._db.create_collection(collection_name, validator=validator)
            else:
                self._db.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists.") 
            
    def insert_data_to_mongo(self, records: list[dict]):
        """
        Inserts data into the MongoDB collection.

        Args:
            records (list[dict]): The list of records to insert.
        """
        collection = self.collection
        collection.insert_many(records)
        return 
    
    def create_docs(self, context_list: list[str], corpus_term: str):
        """
        Creates documents from the context list and corpus term.

        Args:
            context_list (list[str]): The list of contexts.
            corpus_term (str): The corpus term.

        Returns:
            list[dict]: The list of documents.
        """
        corpus_docs = []
        for context in context_list:
            corpus_docs.append({"term": corpus_term, "context": context})
        return corpus_docs
    
    def create_vector_search_from_texts(self, context_list: list[str], curated_term_list: list[str]):
        """
        Creates a MongoDBAtlasVectorSearch object using the connection string, database, and collection names, along with the OpenAI embeddings and index configuration.

        Args:
            context_list (list[str]): The list of contexts.
            curated_term_list (list[str]): The list of curated terms.

        Returns:
            MongoDBAtlasVectorSearch: The MongoDBAtlasVectorSearch object.
        """
        vector_search = MongoDBAtlasVectorSearch.from_texts(
            texts=[context_list],
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=self.collection,
            metadatas=[{'curated_term': curated_term_list}],
            index_name=self._atlas_search_index_name,
        )
        return vector_search