import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List
import os
import re
from src.CustomLogger.custom_logger import CustomLogger

RERANKER_TYPE_MAP = {
    # Cross-Encoders
    "minilm": "cross_encoder",
    "electra": "cross_encoder",
    "bge-reranker-base": "cross_encoder",
    "bge-reranker-large": "cross_encoder",
    "bge-ce": "cross_encoder",
    "medcpt-ce": "cross_encoder",

    # T5
    "monot5-base": "t5",
    "monot5-3b": "t5",

    # Generative LLMs
    "qwen-7b": "generative",
    "qwen-14b": "generative",
    "mistral-7b": "generative",
    "llama-8b": "generative",
}

RERANKER_BATCH_SIZE = {
    "cross_encoder": 32,
    "t5": 16,
    "generative": 1,
}


class Reranker:
    """
    Unified reranker. Auto-detects everything.
    """

    def __init__(
        self,
        model_name: str,
        method: str,
        use_8bit: bool = False,
        batch_size: int = None,
        cache_dir: str = "model_cache",
    ):
        self.model_name = model_name
        self.method = method
        self.cache_dir = cache_dir

        self.logger = CustomLogger().custlogger(loglevel='INFO')

        if method not in RERANKER_TYPE_MAP:
            raise ValueError(f"Unknown method: {method}. "
                             f"Choose from {list(RERANKER_TYPE_MAP.keys())}")

        self.reranker_type = RERANKER_TYPE_MAP[method]
        self.batch_size = batch_size or RERANKER_BATCH_SIZE[self.reranker_type]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if use_8bit is None and self.device == 'cuda':
            vram = torch.cuda.get_device_properties(0).total_memory
            use_8bit = (vram < 20e9) and (self.reranker_type
                                          in ["t5", "generative"])

        self.use_8bit = use_8bit

        self.logger.info(f"Loading {self.reranker_type}: {model_name}")
        self.logger.info(
            f"Device: {self.device}, 8-bit: {use_8bit}, batch: {self.batch_size}"
        )

        os.makedirs(cache_dir, exist_ok=True)

        # Load
        if self.reranker_type == "cross_encoder":
            self._load_cross_encoder()
        elif self.reranker_type == "t5":
            self._load_t5()
        elif self.reranker_type == "generative":
            self._load_generative()

        self.logger.info("Reranker loaded")

    def _load_cross_encoder(self):
        """Load Cross-Encoder"""
        self.model = CrossEncoder(self.model_name,
                                  device=self.device,
                                  cache_folder=self.cache_dir)

    def _load_t5(self):
        """Load T5"""
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        load_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": torch.float16
        }

        if self.use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs[
                "device_map"] = "auto" if self.device == "cuda" else None

        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs)

        if not self.use_8bit and self.device == "cuda":
            self.model = self.model.cuda()

        self.model.eval()

        self.token_false_id = self.tokenizer.encode('false')[0]
        self.token_true_id = self.tokenizer.encode('true')[0]

    def _load_generative(self):
        """Load Generative LLM"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }

        if self.use_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs)
        self.model.eval()

    # ==================== Prediction ====================

    def predict(self, pairs: List[List[str]]) -> np.ndarray:
        """Unified predict"""
        if self.reranker_type == "cross_encoder":
            return self._predict_cross_encoder(pairs)
        elif self.reranker_type == "t5":
            return self._predict_t5(pairs)
        else:  # generative
            return self._predict_listwise(pairs)

    def _predict_cross_encoder(self, pairs: List[List[str]]) -> np.ndarray:
        """Cross-Encoder"""
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores.extend(self.model.predict(batch))
        return np.array(scores)

    def _predict_t5(self, pairs: List[List[str]]) -> np.ndarray:
        """T5"""
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores.extend(self._t5_batch(batch))
        return np.array(scores)

    def _t5_batch(self, pairs: List[List[str]]) -> List[float]:
        texts = [f"Query: {q} Document: {d} Relevant:" for q, d in pairs]

        inputs = self.tokenizer(texts,
                                padding=True,
                                truncation=True,
                                return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs,
                                          max_new_tokens=1,
                                          return_dict_in_generate=True,
                                          output_scores=True)
            logits = outputs.scores[0]

            scores = []
            for i in range(len(texts)):
                true_s = logits[i, self.token_true_id].item()
                false_s = logits[i, self.token_false_id].item()
                score = np.exp(true_s) / (np.exp(true_s) + np.exp(false_s))
                scores.append(score)

        return scores

    def _predict_listwise(self, pairs: List[List[str]]) -> np.ndarray:
        """Listwise ranking"""
        query_groups = {}
        for i, (query, doc) in enumerate(pairs):
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append((i, doc))

        scores = [0.0] * len(pairs)

        for query, doc_list in query_groups.items():
            indices = [idx for idx, _ in doc_list]
            docs = [doc for _, doc in doc_list]

            # Chunked ranking
            if len(docs) > 20:
                ranked = self._rank_chunked(query, docs)
            else:
                ranked = self._rank_docs(query, docs)

            for rank, doc in enumerate(ranked):
                try:
                    orig_idx = docs.index(doc)
                    scores[indices[orig_idx]] = 1.0 - (rank / len(ranked))
                except ValueError:
                    pass

        return np.array(scores)

    def _rank_docs(self, query: str, docs: List[str]) -> List[str]:
        """Rank using LLM"""
        doc_list = "\n".join([f"{i}. {doc}" for i, doc in enumerate(docs)])

        prompt = f"""Rank by relevance to "{query}":

    {doc_list}

    Indices:"""

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)

        inputs = self.tokenizer([text], return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):],
                                         skip_special_tokens=True).strip()

        return self._parse_ranking(response, docs)

    def _rank_chunked(self, query: str, docs: List[str]) -> List[str]:
        """Chunked ranking"""
        chunks = [docs[i:i + 20] for i in range(0, len(docs), 20)]
        chunk_rankings = [self._rank_docs(query, c) for c in chunks]

        merged = []
        for ranked in chunk_rankings:
            merged.extend(ranked[:5])

        final = self._rank_docs(query, merged) if len(merged) > 20 else merged

        for doc in docs:
            if doc not in final:
                final.append(doc)

        return final

    def _parse_ranking(self, response: str, docs: List[str]) -> List[str]:
        """Parse ranking"""
        try:
            numbers = re.findall(r'\d+', response)
            indices = [int(x) for x in numbers if int(x) < len(docs)]

            seen = set()
            unique = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique.append(idx)

            ranked = [docs[i] for i in unique]
            for i, doc in enumerate(docs):
                if i not in unique:
                    ranked.append(doc)

            return ranked
        except:
            return docs

    def __call__(self, pairs: List[List[str]]) -> np.ndarray:
        return self.predict(pairs)
