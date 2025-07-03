import src.models.schema_mapper_bert as sm_bert 
import src.models.schema_mapper_lda as sm_lda 
import src.models.schema_mapper_models as sm_models
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
from thefuzz import fuzz
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger()

class SchemaMapEngine(sm_models.ClinicalDataMatcher):
    """
    This class integrates the different models available for schema mapping.
    It extends the ClinicalDataMatcher class and initializes with specified parameters.

    Attributes:
        clinical_data_path (str): Path to the clinical data.
        schema_map_path (str): Path to the schema map.
        matcher_type (str): Type of matcher to use ('Bert', 'LDA', or None).
        output_file_name (str): Name of the output file.
        top_k (int): Number of top results to consider.
        _matcher (object): Instance of the matcher.
        _logger (object): Logger instance.
    """

    def __init__(self, clinical_data_path: str, schema_map_path: str, matcher_type: str, output_file_name: str, k: int) -> None:
        """
        Initialize the SchemaMapEngine class with specified parameters.

        Args:
            clinical_data_path (str): Path to the clinical data.
            schema_map_path (str): Path to the schema map.
            matcher_type (str): Type of matcher to use ('Bert', 'LDA', or None).
            output_file_name (str): Name of the output file.
            k (int): Number of top results to consider.
        """
        super().__init__(clinical_data_path, schema_map_path)
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
        Get the matcher instance based on the matcher type.

        Returns:
            object: Matcher instance.
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
        """
        Run the schema mapping process and save the output to a JSON file.
        """
        self._logger.info("Running Schema Mapper")
        if self.matcher_type == 'LDA':
            self.matcher.save_topic_mapping_to_json(self.output_file_name, num_topics=self.top_k, num_words=5)
        elif self.matcher_type == 'Bert':
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k=5)
        elif self.matcher_type is None:
            self.matcher.save_output_to_json(self.output_file_name)      
        self._logger.info("Json file saved")