import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import util
import torch
from typing import Union

from src.models.schema_mapper import ClinicalDataMatcher
from src.utils.schema_mapper_utils import normalize


class ClinicalDataMatcherBert(ClinicalDataMatcher):
    """
    A class to match clinical data columns to schema map topics using BERT embeddings.
    """

    def __init__(self, clinical_data: pd.DataFrame, schema_map_path: str):
        """
        Initializes the ClinicalDataMatcherBert class.

        Args:
            clinical_data (pd.DataFrame): The clinical data DataFrame.
            schema_map_path (str): Path to the schema map file.
        """
        super().__init__(clinical_data, schema_map_path)

        # Load the PubMed BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self.model = AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        self._topic_embeddings = self._create_topic_embeddings()

    def _normalize_text(self, text: str) -> str:
        """Normalizes text for BERT processing."""
        return normalize(text)

    def _create_embedding(self, text: str) -> torch.Tensor:
        """Creates a BERT embedding for a single text string."""
        encoded_input = self.tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output[0][:, 0, :]

    def _create_topic_embeddings(self) -> dict[str, dict[str, Union[torch.Tensor, str]]]:
        """Pre-computes embeddings and sources for all topics in the schema map."""
        topic_embeddings = {}
        for topic, content in self.schema_map.items():
            if not isinstance(content, dict):
                continue
            
            terms = [topic] + content.get('column_names', []) + content.get('unique_col_values', [])
            combined_text = " ".join(self._normalize_text(str(term)) for term in terms if term)
            
            if combined_text:
                topic_embeddings[topic] = {
                    "embedding": self._create_embedding(combined_text),
                    "source": content.get("source", topic)  # Store source, default to topic name
                }
        return topic_embeddings

    def bert_match(self, column_name: str, top_k: int = 5) -> list[tuple[str, float, str]]:
        """
        Matches a clinical data column to schema map topics using BERT embeddings.

        Args:
            column_name (str): The name of the column to match.
            top_k (int, optional): The number of top matches to return. Defaults to 5.

        Returns:
            list: A list of tuples containing (match_field, match_score, match_source).
        """
        if column_name not in self.clinical_data.columns:
            return []

        col_values = self.clinical_data[column_name].dropna().astype(str).tolist()
        value_sample = " ".join(col_values[:50])
        text_to_embed = f"{self._normalize_text(column_name)} {self._normalize_text(value_sample)}"

        if not text_to_embed.strip():
            return []

        col_embedding = self._create_embedding(text_to_embed)

        sims = {
            topic: util.cos_sim(col_embedding, data["embedding"]).item()
            for topic, data in self._topic_embeddings.items()
        }

        sorted_matches = sorted(sims.items(), key=lambda x: x[1], reverse=True)

        results = []
        for topic, score in sorted_matches[:top_k]:
            source = self._topic_embeddings[topic]["source"]
            results.append((topic, score, source))

        return results
