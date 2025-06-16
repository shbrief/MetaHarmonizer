from langchain.embeddings.base import Embeddings
from typing import List, Any


class EmbeddingAdapter(Embeddings):
    """
    Generic embedding adapter that supports both SentenceTransformer and LangChain style models:
    - SentenceTransformer: encode(texts: List[str], normalize_embeddings=True)
    - LangChain: embed_documents(texts: List[str]) and embed_query(text: str)
    """

    def __init__(self, model: Any):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.model, "encode"):
            embs = self.model.encode(texts, normalize_embeddings=True)
            return embs.tolist()
        if hasattr(self.model, "embed_documents"):
            return self.model.embed_documents(texts)
        raise NotImplementedError(
            f"{self.model.__class__.__name__} not supported for document-level embeddings"
        )

    def embed_query(self, text: str) -> List[float]:
        if hasattr(self.model, "encode"):
            emb = self.model.encode([text], normalize_embeddings=True)[0]
            return emb.tolist()
        if hasattr(self.model, "embed_query"):
            return self.model.embed_query(text)
        raise NotImplementedError(
            f"{self.model.__class__.__name__} not supported for query embeddings"
        )
