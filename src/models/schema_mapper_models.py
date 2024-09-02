import pandas as pd 
import pickle 
import json 

class ClinicalDataMatcher:
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    def __init__(self, clinical_data_path, schema_map_path):
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
    """
    A class to match clinical data columns to a schema map.

    Attributes:
        clinical_data (pd.DataFrame): The clinical data.
        schema_map (dict): The schema map.
    """

    def __init__(self, clinical_data_path, schema_map_path):
=======
    """
    A class to match clinical data columns to a schema map.

    Attributes:
        clinical_data (pd.DataFrame): The clinical data.
        schema_map (dict): The schema map.
    """

    def __init__(self, clinical_data_path, schema_map_path):
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        """
        Initializes the ClinicalDataMatcher class.

        Args:
            clinical_data_path (str): Path to the clinical data file.
            schema_map_path (str): Path to the schema map file.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
    def __init__(self, clinical_data_path, schema_map_path):
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        self.clinical_data = pd.read_csv(clinical_data_path, sep='\t')
        with open(schema_map_path, 'rb') as file:
            self.schema_map = pickle.load(file)
    
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    
    def get_value_based_matches(self, column_values, top_n=5):
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
    def get_value_based_matches(self, column_values, top_n=5):
=======
    def get_value_based_matches(self, column_values, top_n=5):
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        """
        Gets the top N value-based matches for a given set of column values.

        Args:
            column_values (list): The list of column values.
            top_n (int, optional): The number of top matches to return. Defaults to 5.

        Returns:
            list: A list of tuples containing the curated name, sub-topic name, and match score.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
    
    def get_value_based_matches(self, column_values, top_n=5):
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        matches = []

        for curated_name, sub_topics in self.schema_map.items():
            for sub_topic_name, sub_topic_content in sub_topics.items():
                if isinstance(sub_topic_content, dict):
                    sub_topic_values = sub_topic_content.get("column_names", []) + sub_topic_content.get("unique_col_values", [])
                elif isinstance(sub_topic_content, list):
                    sub_topic_values = sub_topic_content
                else:
                    continue

                matching_values = set(column_values) & set(sub_topic_values)
                match_score = len(matching_values) / len(column_values) if column_values else 0
                matches.append((curated_name, sub_topic_name, match_score))
        
        matches = sorted(matches, key=lambda x: x[2], reverse=True)[:top_n]
        return matches
    
    def map_columns_to_curated(self):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
            suggestions = {}
            for col in self.clinical_data.columns:
                if 'ID' in col.upper():  # Skip columns containing 'ID'
                    continue
                column_values = self.clinical_data[col].dropna().unique().tolist()
                matches = self.get_value_based_matches(column_values)
                suggestions[col] = [
                    [match[0], match[2]]
                    for match in matches
                ]
            return suggestions
<<<<<<< HEAD
=======
        """
        Maps clinical data columns to curated schema map topics.

        Returns:
            dict: A dictionary mapping clinical data columns to their top matches.
        """
        suggestions = {}
        for col in self.clinical_data.columns:
            if 'ID' in col.upper():  # Skip columns containing 'ID'
                continue
            column_values = self.clinical_data[col].dropna().unique().tolist()
            matches = self.get_value_based_matches(column_values)
            suggestions[col] = [
                [match[0], match[2]]
                for match in matches
            ]
        return suggestions
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
    
    def save_output_to_json(self, output_path):
        """
        Saves the column-to-curated mappings to a JSON file.

        Args:
            output_path (str): The path to the output JSON file.
        """
        mappings = self.map_columns_to_curated()
        with open(output_path, 'w') as json_file:
            json.dump(mappings, json_file, indent=4)
<<<<<<< HEAD
            

=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Maps clinical data columns to curated schema map topics.

        Returns:
            dict: A dictionary mapping clinical data columns to their top matches.
        """
        suggestions = {}
        for col in self.clinical_data.columns:
            if 'ID' in col.upper():  # Skip columns containing 'ID'
                continue
            column_values = self.clinical_data[col].dropna().unique().tolist()
            matches = self.get_value_based_matches(column_values)
            suggestions[col] = [
                [match[0], match[2]]
                for match in matches
            ]
        return suggestions
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
    
    def save_output_to_json(self, output_path):
        mappings = self.map_columns_to_curated()
        with open(output_path, 'w') as json_file:
            json.dump(mappings, json_file, indent=4)
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
            

>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
