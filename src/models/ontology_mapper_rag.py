from sentence_transformers import util
import pandas as pd
import src.models.ontology_models as otm
import src.CustomLogger.custom_logger
# from langchain.embeddings import Embeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain.document_loaders import DirectoryLoader 
# from langchain.llms import OpenAI
# from langchain.chains import RetrievalQA
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import MongoDBAtlasVectorSearch
# from langchain_openai import ChatOpenAI
# from pathos.multiprocessing import ProcessingPool as Pool
logger = src.CustomLogger.custom_logger.CustomLogger()

class OntoMapRAG(otm.OntoModelsBase):
    def __init__(self, method:str, topk:int, query:list, corpus:list, yaml_path:str='method_model.yaml') -> None:    
        super().__init__(method, topk, query, corpus, yaml_path)
        
        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized OntoMap RAG module")
        
    def create_context_list_mp(self, nci_codes):
        # pool = Pool(processes=12)
        # context_list = pool.map(get_context_list, nci_codes)
        # return context_list 
        raise NotImplementedError()
