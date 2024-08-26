from fuzzywuzzy import process
import src.models.schema_mapper_models as sm_models
import pandas as pd
import pickle
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
class ClinicalDataMatcherBert:
    def __init__(self, clinical_data_path, schema_map_path):
        # Load data
        self.clinical_data = pd.read_csv(clinical_data_path, sep='\t')
        with open(schema_map_path, 'rb') as f:
            self.schema_map = pickle.load(f)
        
        self.lemmatizer = WordNetLemmatizer()

        # Load the PubMed BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    def preprocess_text(self, text):
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
            return " ".join(tokens)
        else:
            return ""

    def create_embeddings(self, text_list, convert_to_tensor=False):
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
        # Tokenize the texts and prepare input tensors

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output[0][:, 0, :]
        return embeddings if convert_to_tensor else embeddings.numpy()

    def create_topic_embeddings(self):
        topic_embeddings = {}
        for topic, content in self.schema_map.items():
            combined_terms = " ".join(self.preprocess_text(term) for term in content['column_names'])
            topic_embeddings[topic] = self.create_embeddings([combined_terms])
        return topic_embeddings

    def create_column_embedding(self, column):
        column_values = self.clinical_data[column].dropna().astype(str).tolist()
        processed_values = " ".join(self.preprocess_text(value) for value in column_values)
        return self.create_embeddings([processed_values])

    def is_quantitative_or_boolean(self, series):
        return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

    def compare_to_subtopics(self, column_name):
        # Create embedding for the clinical data column name
        column_name_embedding = self.create_embeddings([self.preprocess_text(column_name)])

        topic_scores = {}
        for curated_name, content in self.schema_map.items():
            # Combine curated column name with all associated column names
            combined_terms = [curated_name] + content['column_names']
            combined_text = " ".join(self.preprocess_text(term) for term in combined_terms)
            
            # Create embeddings for the combined terms
            topic_embedding = self.create_embeddings([combined_text])

            # Compute cosine similarity
            similarity = util.cos_sim(column_name_embedding, topic_embedding).item()
            topic_scores[curated_name] = similarity

        # Sort topics by similarity score in descending order
        return sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)

    def map_columns_to_topics(self, k=5):
        topic_embeddings = self.create_topic_embeddings()
        column_topic_mapping = {}
        for col in self.clinical_data.columns:
            if 'ID' in col.upper():  # Skip columns containing 'ID'
                continue
            if self.is_quantitative_or_boolean(self.clinical_data[col]):
                # For Quant/Boolean columns, compare based on column names
                top_k_topics = self.compare_to_subtopics(col)[:k]
            else:
                # For Text columns, use BERT embeddings
                col_embedding = self.create_column_embedding(col)
                topic_distances = {topic: util.cos_sim(col_embedding, topic_embedding).item() for topic, topic_embedding in topic_embeddings.items()}
                sorted_topics = sorted(topic_distances.items(), key=lambda x: x[1], reverse=True)[:k]
                top_k_topics = sorted_topics
            column_topic_mapping[col] = top_k_topics
        return column_topic_mapping

    def save_topic_mapping_to_json(self, output_path, k=5):
        topic_mappings = self.map_columns_to_topics(k)
        with open(output_path, 'w') as json_file:
            json.dump(topic_mappings, json_file, indent=4)   