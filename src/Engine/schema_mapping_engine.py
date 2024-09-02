import src.models.schema_mapper_bert as sm_bert 
import src.models.schema_mapper_lda as sm_lda 
import src.models.schema_mapper_models as sm_models
import src.CustomLogger.custom_logger
import pandas as pd
import numpy as np
from thefuzz import fuzz
logger = src.CustomLogger.custom_logger.CustomLogger()

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# This class `SchemaMapEngine` extends `ClinicalDataMatcher` and initializes with specified
# parameters.
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
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
<<<<<<< HEAD
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
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
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        Initialize the SchemaMapEngine class with specified parameters.

        Args:
            clinical_data_path (str): Path to the clinical data.
            schema_map_path (str): Path to the schema map.
            matcher_type (str): Type of matcher to use ('Bert', 'LDA', or None).
            output_file_name (str): Name of the output file.
            k (int): Number of top results to consider.
        """
        super().__init__(clinical_data_path, schema_map_path)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
# This class `SchemaMapEngine` extends `ClinicalDataMatcher` and initializes with specified
# parameters.
class SchemaMapEngine(sm_models.ClinicalDataMatcher):
    def __init__(self, clinical_data_path:str, schema_map_path:str, matcher_type:str, output_file_name:str, k:int) -> None:
        super().__init__(clinical_data_path, schema_map_path)
        """This class integrates the different models available for schema mapping 
        """
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
        Description of matcher

        Args:
            self (undefined):
<<<<<<< HEAD

=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        Get the matcher instance based on the matcher type.

        Returns:
            object: Matcher instance.
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
        Get the matcher instance based on the matcher type.

        Returns:
            object: Matcher instance.
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======

>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            return self._matcher    
    
    def run_schema_mapping(self):
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        return self._matcher    
    
    def run_schema_mapping(self):
        """
        Run the schema mapping process and save the output to a JSON file.
        """
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
        return self._matcher    
    
    def run_schema_mapping(self):
        """
        Run the schema mapping process and save the output to a JSON file.
        """
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
            return self._matcher    
    
    def run_schema_mapping(self):
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        self._logger.info("Running Schema Mapper")
        if self.matcher_type == 'LDA':
            self.matcher.save_topic_mapping_to_json(self.output_file_name, num_topics=self.top_k, num_words=5)
        elif self.matcher_type == 'Bert':
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k = 5)
=======
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k=5)
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k=5)
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k = 5)
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
            self.matcher.save_topic_mapping_to_json(self.output_file_name, k=5)
>>>>>>> 0570565 (Local changes to adding documentation updated)
        elif self.matcher_type is None:
            self.matcher.save_output_to_json(self.output_file_name)      
        self._logger.info("Json file saved")