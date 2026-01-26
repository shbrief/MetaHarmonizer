"""Loader for value dictionary."""
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from ..config import VALUE_DICT_PATH, NOISE_VALUES, FIELD_MODEL
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class ValueLoader:
    """Load and prepare value dictionary."""
    
    @staticmethod
    def load_value_dict(engine, json_path: str = VALUE_DICT_PATH):
        """
        Load value dictionary and prepare embeddings.
        Modifies engine in-place by setting:
        - engine.value_texts
        - engine.value_fields_list
        - engine.value_embs
        
        Args:
            engine: SchemaMapEngine instance to modify
            json_path: Path to value dictionary JSON file
        """
        engine.value_texts = []
        engine.value_fields_list = []
        
        if not os.path.exists(json_path):
            logger.warning(f"[ValueLoader] Missing value dictionary: {json_path}")
            engine.value_embs = None
            return
        
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for field, values in data.items():
            for v in values:
                v = str(v).strip()
                if not v or v.lower() in NOISE_VALUES:
                    continue
                engine.value_texts.append(v)
                engine.value_fields_list.append([field])
        
        if not engine.value_texts:
            engine.value_embs = None
            logger.warning("[ValueLoader] No valid values found in dictionary")
            return
        
        model = SentenceTransformer(FIELD_MODEL)
        with torch.no_grad():
            embs = model.encode(engine.value_texts, convert_to_tensor=True)
            engine.value_embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        
        logger.info(f"[ValueLoader] Loaded {len(engine.value_texts)} value entries from {json_path}")