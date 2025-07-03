from langchain.embeddings.base import Embeddings
from typing import List, Any
import numpy as np
from transformers import AutoTokenizer

import torch


class EmbeddingAdapter(Embeddings):
    """
    Generic embedding adapter that supports both SentenceTransformer and LangChain style models:
    - SentenceTransformer: encode(texts: List[str], normalize_embeddings=True)
    - LangChain: embed_documents(texts: List[str]) and embed_query(text: str)
    """

    def __init__(self, model: Any, om_strategy: str = "st"):
        self.model = model
        self.strategy = om_strategy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.strategy == "lm":
            if not hasattr(self.model.config, "name_or_path"):
                raise ValueError(
                    "For LM strategy, model must have a config with 'name_or_path' attribute"
                )
            model_id = getattr(self.model.config, "name_or_path", None)
            if model_id is None:
                raise ValueError(
                    "Cannot infer model_id from model.config.name_or_path")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model.to(self.device).eval()
        else:
            if not hasattr(self.model, "tokenizer"):
                raise ValueError("Model does not have a tokenizer attribute")

            self.tokenizer = getattr(self.model, "tokenizer", None)
            self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.strategy == "lm":
            # CLS pooling
            enc = self.tokenizer(texts,
                                 padding="max_length",
                                 max_length=64,
                                 truncation=True,
                                 return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**enc)

            hidden = getattr(outputs, "last_hidden_state", outputs[0])
            cls_emb = hidden[:, 0, :].cpu().numpy()  # shape (bs, dim)

            if not getattr(self.model, "_is_normalized", False):
                norms = np.linalg.norm(cls_emb, axis=1, keepdims=True) + 1e-12
                cls_emb = cls_emb / norms

            return cls_emb.tolist()
        else:
            embs = self.model.encode(texts,
                                     normalize_embeddings=True,
                                     batch_size=16,
                                     show_progress_bar=True,
                                     device=self.device)

            return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        if self.strategy == "lm":
            # CLS pooling for single query, can be optimized later
            return self.embed_documents([text])[0]

        if hasattr(self.model, "encode"):
            emb = self.model.encode([text],
                                    normalize_embeddings=True,
                                    batch_size=16,
                                    show_progress_bar=True,
                                    device=self.device)[0]
            return emb.tolist()
        if hasattr(self.model, "embed_query"):
            return self.model.embed_query(text)
        raise NotImplementedError(
            f"{self.model.__class__.__name__} not supported for query embeddings"
        )
