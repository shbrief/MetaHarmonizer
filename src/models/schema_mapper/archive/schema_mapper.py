import pickle
import pandas as pd

SCHEMA_MAP_PATH = 'data/schema_map_generated.pkl'


class ClinicalDataMatcher:
    """
    Base class for matching clinical data columns to a schema map.
    """

    def __init__(self, clinical_data: pd.DataFrame, top_k: int):
        """
        Initializes the ClinicalDataMatcher class.

        Args:
            clinical_data (pd.DataFrame): The clinical data DataFrame.
        """
        self.clinical_data = clinical_data
        self.top_k = top_k
        with open(SCHEMA_MAP_PATH, 'rb') as f:
            self.schema_map = pickle.load(f)
