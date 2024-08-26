import src.models.schema_mapper_bert as sm_bert 
import src.models.schema_mapper_lda as sm_lda 
import src.models.schema_mapper_models as sm_models
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
from thefuzz import fuzz
logger = src.CustomLogger.custom_logger.CustomLogger()

# This class `SchemaMapEngine` extends `ClinicalDataMatcher` and initializes with specified
# parameters.
class SchemaMapEngine(sm_models.ClinicalDataMatcher):
    def __init__(self, clinical_data_path:str, schema_map_path:str, matcher_type:str, output_file_name:str, k:int) -> None:
        super().__init__(clinical_data_path, schema_map_path)
        """This class integrates the different models available for schema mapping 
        """
        self.top_k = k
        self.matcher_type = matcher_type
        self.output_file_name = output_file_name
        self._matcher = None 
        self.clinical_data_path = clinical_data_path
        self.schema_map_path = schema_map_path
        self._logger = logger.custlogger(loglevel='INFO')
        self._logger.info("Initialized SchemaMap Engine module")

    @property
    def matcher(self):
        """
        Description of matcher

        Args:
            self (undefined):

        """
        if self._matcher is None:
            if self.matcher_type is None:
                self._matcher = sm_models.ClinicalDataMatcher(
                    clinical_data_path=self.clinical_data_path,
                    schema_map_path=self.schema_map_path
                    )
            elif self.matcher_type == 'Bert':
                self._matcher = sm_bert.ClinicalDataMatcherBert(
                    clinical_data_path=self.clinical_data_path,
                    schema_map_path=self.schema_map_path
                    )
            elif self.matcher_type == 'LDA':
                self._matcher = sm_lda.ClinicalDataMatcherWithTopicModeling(
                    clinical_data_path=self.clinical_data_path,
                    schema_map_path=self.schema_map_path,
                    k=self.top_k
                )
            return self._matcher    
    
    def run_schema_mapping(self):
        self._logger.info("Running Schema Mapper")
        if self.matcher_type == 'LDA':
            self.matcher.save_topic_mapping_to_json(self.output_file_name, num_topics=self.top_k, num_words=5)
        elif self.matcher_type == 'Bert':
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k = 5)
        elif self.matcher_type is None:
            self.matcher.save_output_to_json(self.output_file_name)      
        self._logger.info("Json file saved")