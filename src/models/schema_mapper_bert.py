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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)

>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
class ClinicalDataMatcherBert:
    """
    A class to match clinical data columns to schema map topics using BERT embeddings.

    Attributes:
        clinical_data (pd.DataFrame): The clinical data.
        schema_map (dict): The schema map.
        lemmatizer (WordNetLemmatizer): The lemmatizer instance.
        tokenizer (AutoTokenizer): The tokenizer instance.
        model (AutoModel): The model instance.
    """

    def __init__(self, clinical_data_path, schema_map_path):
<<<<<<< HEAD
=======

=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
class ClinicalDataMatcherBert:
    def __init__(self, clinical_data_path, schema_map_path):
<<<<<<< HEAD
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        """
        Initializes the ClinicalDataMatcherBert class.

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
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        # Load data
        self.clinical_data = pd.read_csv(clinical_data_path, sep='\t')
        with open(schema_map_path, 'rb') as f:
            self.schema_map = pickle.load(f)
        
        self.lemmatizer = WordNetLemmatizer()

        # Load the PubMed BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')

    def preprocess_text(self, text):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Preprocesses the input text by tokenizing, lowercasing, and lemmatizing.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        if isinstance(text, str):
            tokens = word_tokenize(text.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
            return " ".join(tokens)
        else:
            return ""

    def create_embeddings(self, text_list, convert_to_tensor=False):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Creates embeddings for a list of texts using the BERT model.

        Args:
            text_list (list[str]): List of texts to embed.
            convert_to_tensor (bool, optional): Whether to return the embeddings as tensors. Defaults to False.

        Returns:
            numpy.ndarray or torch.Tensor: The embeddings.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
        # Tokenize the texts and prepare input tensors

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = model_output[0][:, 0, :]
        return embeddings if convert_to_tensor else embeddings.numpy()

    def create_topic_embeddings(self):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Creates embeddings for the topics in the schema map.

        Returns:
            dict: A dictionary mapping topics to their embeddings.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        topic_embeddings = {}
        for topic, content in self.schema_map.items():
            combined_terms = " ".join(self.preprocess_text(term) for term in content['column_names'])
            topic_embeddings[topic] = self.create_embeddings([combined_terms])
        return topic_embeddings

    def create_column_embedding(self, column):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Creates an embedding for a column in the clinical data.

        Args:
            column (str): The column name.

        Returns:
            numpy.ndarray or torch.Tensor: The column embedding.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        column_values = self.clinical_data[column].dropna().astype(str).tolist()
        processed_values = " ".join(self.preprocess_text(value) for value in column_values)
        return self.create_embeddings([processed_values])

    def is_quantitative_or_boolean(self, series):
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

    def compare_to_subtopics(self, column_name):
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Checks if a pandas Series is of numeric or boolean type.

        Args:
            series (pd.Series): The pandas Series.

        Returns:
            bool: True if the series is numeric or boolean, False otherwise.
        """
        return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

    def compare_to_subtopics(self, column_name):
=======
        """
        Checks if a pandas Series is of numeric or boolean type.

        Args:
            series (pd.Series): The pandas Series.

        Returns:
            bool: True if the series is numeric or boolean, False otherwise.
        """
        return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

    def compare_to_subtopics(self, column_name):
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
        """
        Compares a clinical data column name to subtopics in the schema map.

        Args:
            column_name (str): The column name.

        Returns:
            list: A list of tuples containing subtopics and their similarity scores, sorted in descending order.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
        return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)

    def compare_to_subtopics(self, column_name):
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Maps clinical data columns to schema map topics.

        Args:
            k (int, optional): The number of top topics to consider. Defaults to 5.

        Returns:
            dict: A dictionary mapping clinical data columns to their top k topics.
        """
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        topic_mappings = self.map_columns_to_topics(k)
        with open(output_path, 'w') as json_file:
            json.dump(topic_mappings, json_file, indent=4)   
=======
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)
        """
        Saves the topic mappings to a JSON file.

        Args:
            output_path (str): The path to the output JSON file.
            k (int, optional): The number of top topics to consider. Defaults to 5.
        """
        topic_mappings = self.map_columns_to_topics(k)
        with open(output_path, 'w') as json_file:
            json.dump(topic_mappings, json_file, indent=4)
<<<<<<< HEAD
>>>>>>> fd1285a (comitting unstaged changes for merging)
=======
        """
        Saves the topic mappings to a JSON file.

        Args:
            output_path (str): The path to the output JSON file.
            k (int, optional): The number of top topics to consider. Defaults to 5.
        """
        topic_mappings = self.map_columns_to_topics(k)
        with open(output_path, 'w') as json_file:
            json.dump(topic_mappings, json_file, indent=4)
>>>>>>> 742544ac2f6635755f8ac6401976785606a7eb27
=======
        topic_mappings = self.map_columns_to_topics(k)
        with open(output_path, 'w') as json_file:
            json.dump(topic_mappings, json_file, indent=4)   
>>>>>>> parent of 742544a (Merge commit 'e73751844beddc38c705c558f40cc490b0f9f107' into abhi_devv)
=======
>>>>>>> 0570565 (Local changes to adding documentation updated)