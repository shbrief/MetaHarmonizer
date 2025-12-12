import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from typing import List
import os
from src.CustomLogger.custom_logger import CustomLogger


class LLMReranker:
    """
    MonoT5-based reranker (T5 is a true LLM trained specifically for reranking).
    """

    def __init__(
        self,
        model_name: str = "castorini/monot5-3b-msmarco-10k",
        device: str = None,
        use_8bit: bool = False,
        max_length: int = 512,
        batch_size: int = 16,
        cache_dir: str = "model_cache",
    ):
        self.model_name = model_name
        self.device = device or ('cuda'
                                 if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        self.batch_size = batch_size

        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.logger.info(f"Loading MonoT5 Reranker: {model_name}")
        self.logger.info(f"Device: {self.device}, 8-bit: {use_8bit}")

        os.makedirs(cache_dir, exist_ok=True)

        self.tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            legacy=False,
        )

        load_kwargs = {
            "cache_dir": cache_dir,
            "torch_dtype": torch.float16,
        }

        if use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"

        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs)

        if not use_8bit and self.device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()

        self.token_false_id = self.tokenizer.encode('false')[0]
        self.token_true_id = self.tokenizer.encode('true')[0]

        self.logger.info("MonoT5 Reranker loaded successfully")

    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """Predict relevance scores for query-document pairs."""
        scores = []

        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            batch_scores = self._predict_batch(batch_pairs)
            scores.extend(batch_scores)

        return np.array(scores)

    def _predict_batch(self, pairs: List[List[str]]) -> List[float]:
        """MonoT5 batch prediction"""
        # MonoT5 input format: "Query: {query} Document: {doc} Relevant:"
        texts = [
            f"Query: {query} Document: {doc} Relevant:" for query, doc in pairs
        ]

        inputs = self.tokenizer(texts,
                                padding=True,
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # MonoT5 generates "true" or "false"
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )

            # Extract logits for "true" and "false"
            logits = outputs.scores[0]

            # Calculate relevance scores (probability of true)
            batch_scores = []
            for i in range(len(texts)):
                true_score = logits[i, self.token_true_id].item()
                false_score = logits[i, self.token_false_id].item()

                # Use softmax normalization
                exp_true = np.exp(true_score)
                exp_false = np.exp(false_score)
                score = exp_true / (exp_true + exp_false)

                batch_scores.append(score)

        return batch_scores

    def __call__(self, pairs: List[List[str]]) -> np.ndarray:
        return self.predict(pairs)
