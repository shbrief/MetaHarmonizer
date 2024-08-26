import pandas as pd
import json
import src.models.schema_mapper_models as sm_models
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class ClinicalDataMatcherWithTopicModeling(sm_models.ClinicalDataMatcher):
    def __init__(self, clinical_data_path, schema_map_path, k):
        super().__init__(clinical_data_path, schema_map_path)
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.top_k = k
        self.schema_documents = self.create_schema_documents()
        self.dictionary = corpora.Dictionary(self.schema_documents)
        self.lda_model = self.run_topic_modeling(num_topics=len(self.schema_map))
    
    
    def preprocess_text(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) 
                  for token in tokens 
                  if token.isalpha() and token not in self.stop_words]
        return tokens

    def create_schema_documents(self):
        documents = []
        self.schema_keys = []  # To keep track of schema map level 1 keys corresponding to documents
        for schema_key, sub_topics in self.schema_map.items():
            terms = []
            for sub_topic_key, sub_topic_content in sub_topics.items():
                if isinstance(sub_topic_content, dict):
                    # For quantitative/boolean, only use 'column_names'
                    if self.is_quantitative_or_boolean_series(sub_topic_content):
                        terms.extend(map(str, sub_topic_content.get("column_names", [])))  # Convert to strings
                    else:
                        terms.extend(map(str, sub_topic_content.get("column_names", [])))  # Convert to strings
                        terms.extend(map(str, sub_topic_content.get("unique_col_values", [])))  # Convert to strings
                elif isinstance(sub_topic_content, list):
                    terms.extend(map(str, sub_topic_content))  # Convert to strings
                # Add more conditions if the schema map has different structures
            preprocessed_terms = self.preprocess_text(' '.join(terms))
            documents.append(preprocessed_terms)
            self.schema_keys.append(schema_key)
        return documents

    def is_quantitative_or_boolean_series(self, sub_topic_content):
        # Determine if the sub_topic_content corresponds to quantitative or boolean fields
        # This method assumes that if 'unique_col_values' are numeric or boolean, then it's quantitative/boolean
        # Modify this logic based on the actual structure of your schema_map
        # Here, we check if all unique_col_values are numeric or boolean
        unique_values = sub_topic_content.get("unique_col_values", [])
        if not unique_values:
            return False  # Default to False if no unique_col_values
        first_val = unique_values[0]
        return isinstance(first_val, (int, float, bool))

    def run_topic_modeling(self, num_topics=5):
        # Use the preprocessed schema documents and dictionary to create a corpus
        corpus = [self.dictionary.doc2bow(doc) for doc in self.schema_documents]
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=self.dictionary, passes=15)
        return lda_model

    def map_columns_to_topics(self, num_topics=5, num_words=5):
        column_topic_mapping = {}
        for col in self.clinical_data.columns:
            if 'ID' in col.upper():  # Skip columns containing 'ID'
                continue
            column_values = self.clinical_data[col].dropna().astype(str).tolist()
            column_text = ' '.join(column_values)
            preprocessed_column = self.preprocess_text(column_text)
            bow = self.dictionary.doc2bow(preprocessed_column)
            topic_distribution = self.lda_model.get_document_topics(bow, minimum_probability=0)
            
            # Sort the topic distribution by probability score in descending order
            sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
            top_topics = sorted_topics[:5]  # Select the top 5 topics
            
            # Prepare the top topics with their terms and scores
            top_topics_info = []
            for topic_id, score in top_topics:
                topic_terms = [term for term, _ in self.lda_model.show_topic(topic_id, topn=num_words)]
                top_topics_info.append({
                    "topic_id": topic_id,
                    "terms": topic_terms,
                    "score": score
                })

            # Get the corresponding schema map keys for the top topics
            mapped_topics = []
            for topic_info in top_topics_info:
                topic_id = topic_info["topic_id"]
                if topic_id < len(self.schema_keys):
                    mapping_key = self.schema_keys[topic_id]
                else:
                    mapping_key = f"Topic {topic_id}"
                
                mapped_topics.append([
                    mapping_key,
                    # "terms": topic_info["terms"],
                    topic_info["score"]
                ])
            
            column_topic_mapping[col] = mapped_topics

        return column_topic_mapping
    
    def convert_to_serializable(self, obj):
        if isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(element) for element in obj]
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        else:
            return obj
    def save_topic_mapping_to_json(self, output_path, num_topics=5, num_words=5):
        topic_mappings = self.map_columns_to_topics(num_topics=num_topics, num_words=num_words)
        # Convert the topic mappings to JSON serializable format
        serializable_mappings = self.convert_to_serializable(topic_mappings)
        # Ensure all values are JSON serializable
        with open(output_path, 'w') as json_file:
            json.dump(serializable_mappings, json_file, indent=4)