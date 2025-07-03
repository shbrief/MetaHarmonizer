import grequests
import requests
import json
import logging
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI
from pathos.multiprocessing import ProcessingPool as Pool
import grequests


class UMLSDb:
    """
    A class to interact with the UMLS (Unified Medical Language System) API.
    """

    def __init__(self, api_key):
        """
        Initialize the UMLSDb class with the provided API key.

        :param api_key: The API key for accessing the UMLS API.
        """
        self.api_key = api_key
        self._base_url = "https://uts-ws.nlm.nih.gov/rest"
        self._version = "current"

    def search_single_term(
        self,
        version: str = None,
        string_query: str = None,
        searchType: str = "normalizedWords",
    ):
        """
        Search for a single term in the UMLS database.

        :param version: The version of the UMLS database to use.
        :param string_query: The term to search for.
        :param searchType: The type of search to perform (default is 'normalizedWords').
        :return: The search results in JSON format.
        """
        version = self._version
        string = string_query
        content_endpoint = "/rest/search/" + version
        full_url = self._base_url + content_endpoint
        page = 0

        try:
            while True:
                page += 1
                query = {
                    "string": string,
                    "apiKey": self.api_key,
                    "pageNumber": page,
                    "searchType": searchType,
                }
                r = requests.get(full_url, params=query)
                r.raise_for_status()
                print(r.url)
                r.encoding = "utf-8"
                outputs = r.json()
                items = (([outputs["result"]])[0])["results"]

                if len(items) == 0:
                    if page == 1:
                        print("No results found." + "\n")
                        break
                    else:
                        break

                print("Results for page " + str(page) + "\n")

                for result in items:
                    print("UI: " + result["ui"])
                    print("URI: " + result["uri"])
                    print("Name: " + result["name"])
                    print("Source Vocabulary: " + result["rootSource"])
                    print("\n")
                return outputs
            print("*********")

        except Exception as except_error:
            print(except_error)

    def get_nci_code_by_term(self, term: str):
        """
        Get the NCI code for a given term.

        :param term: The term to search for.
        :return: A tuple containing the search results and a list of NCI codes.
        """
        version = "current"
        term_str = term
        uri = "https://uts-ws.nlm.nih.gov"
        content_endpoint = "/rest/search/" + version
        full_url = uri + content_endpoint
        page = 0

        try:
            while True:
                page += 1
                query = {
                    "string": term,
                    "apiKey": self.api_key,
                    "pageNumber": page,
                    "searchType": "exact",
                    "sabs": "NCI",
                }
                query["returnIdType"] = ["code"]
                r = requests.get(full_url, params=query)
                r.raise_for_status()
                print(r.url)
                r.encoding = "utf-8"
                outputs = r.json()
                items = (([outputs["result"]])[0])["results"]

                if len(items) == 0:
                    if page == 1:
                        print("No results found." + "\n")
                        break
                    else:
                        break

                print("Results for page " + str(page) + "\n")

                for result in items:
                    print("UI: " + result["ui"])
                    print("URI: " + result["uri"])
                    print("Name: " + result["name"])
                    print("Source Vocabulary: " + result["rootSource"])
                    print("\n")
                ui_list = []
                for res in items:
                    if res["rootSource"] == "NCI":
                        ui_list.append(res["ui"])
                return items, ui_list

            print("*********")

        except Exception as except_error:
            print(except_error)


class NCIDb:
    """
    A class to interact with the NCI (National Cancer Institute) API.
    """

    def __init__(self):
        """
        Initialize the NCIDb class.
        """
        self._base_url = "https://api-evsrest.nci.nih.gov/api/v1"
        self._umls_api_key = "acf85d30-60c7-4b65-97c1-07e6df7c624a"
        self._umls_db = UMLSDb(self._umls_api_key)
        self._terminology = "ncit"

    def get_term_concept_urls(self, terms: list[str], list_of_concepts):
        """
        Generate URLs for term concepts.

        :param terms: A list of terms to generate URLs for.
        :param list_of_concepts: A list of concepts to include in the URLs.
        :return: A list of URLs.
        """
        ls_concept_str = ",".join(list_of_concepts)
        urls = [
            self._base_url + f"/concept/ncit/{term_code}?include={ls_concept_str}"
            for term_code in terms
        ]
        return urls

    def get_custom_concepts_by_codes(self, terms, list_of_concepts):
        """
        Return custom concept information for a given terminology and code.

        :param terms: A list of terms to search for.
        :param list_of_concepts: A list of concepts to include in the search.
        :return: A dictionary mapping terms to their corresponding responses.
        """
        urls = self.get_term_concept_urls(terms, list_of_concepts)
        rs = (grequests.get(u) for u in urls)
        responses = grequests.map(rs)
        term_response_map = {}
        pretty_print = []

        for term, response in zip(terms, responses):
            if response is None:
                continue
            if response.status_code == requests.codes.ok:
                response_json = json.loads(response.text)
                pretty_print.append(response_json)
                term_response_map[term] = response_json

        logging.info(json.dumps(pretty_print, indent=2))
        return term_response_map

    def create_context_list(self, concepts_for_curated_term2: dict[str]):
        """
        Create a context list from the given concepts.

        :param concepts_for_curated_term2: A dictionary of concepts for a curated term.
        :return: A string representing the context list.
        """
        context_list = []
        concepts = ["synonyms", "children", "roles", "definitions", "parents"]
        for concept in concepts:
            if concept in concepts_for_curated_term2:
                if concept == "synonyms":
                    str_tmp = "The synonyms for the term are:"
                    for syn in concepts_for_curated_term2[concept]:
                        str_tmp += syn["name"] + ", "
                    context_list.append(str_tmp)
                elif concept == "children":
                    str_tmp = "The children of the term are:"
                    for child in concepts_for_curated_term2[concept]:
                        str_tmp += child["name"] + ", "
                    context_list.append(str_tmp)
                elif concept == "parents":
                    str_tmp = "The parents of the term are:"
                    for parent in concepts_for_curated_term2[concept]:
                        str_tmp += parent["name"] + ", "
                    context_list.append(str_tmp)
                elif concept == "roles":
                    str_tmp = "The roles of the term are:"
                    for role in concepts_for_curated_term2[concept]:
                        str_tmp += role["type"] + " " + role["relatedName"] + ", "
                    context_list.append(str_tmp)
                elif concept == "definitions":
                    str_tmp = "The definitions for the term are:"
                    for definition in concepts_for_curated_term2[concept]:
                        str_tmp += definition["definition"] + ", "
                    context_list.append(str_tmp)

        return ".".join(context_list)


class MongoDBUtils:
    """
    A utility class for interacting with MongoDB.
    """

    def __init__(self, connection_str, db, collection) -> None:
        """
        Initialize the MongoDBUtils class.

        :param connection_str: The connection string for MongoDB.
        :param db: The name of the database.
        :param collection: The name of the collection.
        """
        self.conn_str = connection_str
        self._client = MongoClient(self.conn_str, server_api=ServerApi("1"))
        self._default_model = OpenAIEmbeddings(disallowed_special=())
        self._db = self._client[db]
        self._atlas_search_index_name = "RAGDataIndex"
        if collection is not None:
            self.collection = self._db[collection]
        else:
            self.collection = None

    def ping(self):
        """
        Send a ping to confirm a successful connection to MongoDB.
        """
        try:
            self._client.admin.command("ping")
            print("Pinged your deployment. You successfully connected to MongoDB!")
        except Exception as e:
            print(e)

    def create_collection(self, collection_name, validator=None):
        """
        Create a new collection in the database.

        :param collection_name: The name of the collection to create.
        :param validator: Optional validator for the collection.
        """
        if collection_name not in self._db.list_collection_names():
            if validator:
                self._db.create_collection(collection_name, validator=validator)
            else:
                self._db.create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists.")

    def insert_data_to_mongo(self, records):
        """
        Insert multiple records into the collection.

        :param records: A list of records to insert.
        """
        collection = self.collection
        collection.insert_many(records)
        return

    def create_docs(self, context_dict):
        """
        Create documents from a context dictionary.

        :param context_dict: A dictionary of context data.
        :return: A list of documents.
        """
        corpus_docs = []
        for context_key, context_value in context_dict.items():
            corpus_docs.append({"term": context_key, "context": context_value})
        return corpus_docs

    def create_filter_for_embedding(self, term: str):
        """
        Create a filter for documents that need embeddings.

        :param term: The term to filter by.
        :return: A filter dictionary.
        """
        filter_field = {
            "$and": [{term: {"$exists": True}}, {"embeddings": {"$exists": False}}]
        }
        return filter_field

    def insert_sapbert_embeddings(self, sapbert_model):
        """
        Insert SapBert embeddings into documents without existing embeddings.

        :param sapbert_model: The SapBert model to use for creating embeddings.
        """
        filter = {
            "$and": [{"context": {"$exists": True}}, {"embeddings": {"$exists": False}}]
        }
        count = 0

        for document in self.collection.find(filter):
            text = document["context"]
            np_embeddings = list(sapbert_model.create_embeddings(text)[0])
            embedding = [float(i) for i in np_embeddings]
            self.collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"embeddings": embedding}},
                upsert=True,
            )
            count += 1
            print(f"Documents updated: {count}")

    def insert_openai_embeddings(self, model=None):
        """
        Insert OpenAI embeddings into documents without existing embeddings.

        :param model: The OpenAI model to use for creating embeddings. If None, uses the default model.
        """
        if model is None:
            model = self._default_model

        filter = {
            "$and": [{"context": {"$exists": True}}, {"embeddings": {"$exists": False}}]
        }
        count = 0

        for document in self.collection.find(filter):
            text = document["context"]
            embedding = model.embed_query(text)
            self.collection.update_one(
                {"_id": document["_id"]},
                {"$set": {"embeddings": embedding}},
                upsert=True,
            )
            count += 1
            print(f"Documents updated: {count}")

    def insert_embeddings(self, embedding_type="openai", model=None):
        """
        Insert embeddings into documents based on the specified type.

        :param embedding_type: Type of embedding to use ('openai' or 'sapbert').
        :param model: The model to use for creating embeddings. Required for 'sapbert', optional for 'openai'.
        """
        if embedding_type == "openai":
            self.insert_openai_embeddings(model)
        elif embedding_type == "sapbert":
            if model is None:
                raise ValueError(
                    "SapBert model must be provided for sapbert embeddings."
                )
            self.insert_sapbert_embeddings(model)
        else:
            raise ValueError("Invalid embedding type. Choose 'openai' or 'sapbert'.")

    def create_vector_search_from_texts(self, context_list, curated_term_list):
        """
        Create a MongoDBAtlasVectorSearch object from texts.

        :param context_list: A list of context texts.
        :param curated_term_list: A list of curated terms.
        :return: A MongoDBAtlasVectorSearch object.
        """
        vector_search = MongoDBAtlasVectorSearch.from_texts(
            texts=[context_list],
            embedding=OpenAIEmbeddings(disallowed_special=()),
            collection=self.collection,
            metadatas=[{"curated_term": curated_term_list}],
            index_name=self._atlas_search_index_name,
        )
        return vector_search
