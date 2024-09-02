from sentence_transformers import util
import pandas as pd
import src.models.ontology_models as otm
import src.CustomLogger.custom_logger
logger = src.CustomLogger.custom_logger.CustomLogger()

class OntoMapLLM(otm.OntoModelsBase):
    def __init__(self, method:str, topk:int, query:list, corpus:list, yaml_path:str='method_model.yaml') -> None:    
        super().__init__(method, topk, query, corpus, yaml_path)
        
        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized OntoMap Large Language model module")

