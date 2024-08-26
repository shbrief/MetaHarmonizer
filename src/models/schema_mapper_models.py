import pandas as pd 
import pickle 
import json 

class ClinicalDataMatcher:
    def __init__(self, clinical_data_path, schema_map_path):
        self.clinical_data = pd.read_csv(clinical_data_path, sep='\t')
        with open(schema_map_path, 'rb') as file:
            self.schema_map = pickle.load(file)
    
    
    def get_value_based_matches(self, column_values, top_n=5):
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
    
    def save_output_to_json(self, output_path):
        mappings = self.map_columns_to_curated()
        with open(output_path, 'w') as json_file:
            json.dump(mappings, json_file, indent=4)
            

