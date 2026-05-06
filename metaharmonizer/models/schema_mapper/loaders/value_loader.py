"""Loader for value dictionary."""
import os
import json
import torch
from pathlib import Path
from typing import Optional, Union, Iterable
from sentence_transformers import SentenceTransformer
from .. import config as _config
from ..config import NOISE_VALUES, FIELD_MODEL
from metaharmonizer.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class ValueLoader:
    """Load and prepare value dictionary."""

    @staticmethod
    def load_value_dict(
        engine,
        json_path: Optional[Union[str, Path]] = None,
        allowed_fields: Optional[Iterable[str]] = None,
    ):
        """
        Load value dictionary and prepare embeddings.
        Modifies engine in-place by setting:
        - engine.value_texts
        - engine.value_fields_list
        - engine.value_embs

        Args:
            engine: SchemaMapEngine instance to modify.
            json_path: Path to value dictionary JSON file. If None, falls back to
                       config.VALUE_DICT_PATH. Pass "" to disable explicitly.
            allowed_fields: Iterable of curated schema field names. If provided,
                            only JSON entries whose key appears in this set are
                            kept. If the JSON has zero overlap, the value dict is
                            skipped entirely (even if the file is valid).
        """
        engine.value_texts = []
        engine.value_fields_list = []
        engine.value_embs = None

        resolved = json_path if json_path is not None else _config.VALUE_DICT_PATH

        # Explicit disable
        if not resolved:
            logger.info("[ValueLoader] Value dictionary disabled, skipping")
            return

        resolved = str(resolved)
        if not os.path.exists(resolved):
            logger.warning(f"[ValueLoader] Missing value dictionary: {resolved}")
            return

        with open(resolved, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Schema-consistency check: only keep fields present in the curated schema
        if allowed_fields is not None:
            allowed = set(allowed_fields)
            overlap = set(data.keys()) & allowed
            if not overlap:
                logger.warning(
                    f"[ValueLoader] Value dictionary keys disjoint from curated "
                    f"schema (value keys: {len(data)}, schema fields: {len(allowed)}); "
                    "skipping value-dict matching."
                )
                return
            if len(overlap) < len(data):
                logger.info(
                    f"[ValueLoader] Keeping {len(overlap)}/{len(data)} value-dict "
                    "fields that match the curated schema"
                )
            data = {k: v for k, v in data.items() if k in allowed}

        for field, values in data.items():
            for v in values:
                v = str(v).strip()
                if not v or v.lower() in NOISE_VALUES:
                    continue
                engine.value_texts.append(v)
                engine.value_fields_list.append([field])

        if not engine.value_texts:
            logger.warning("[ValueLoader] No valid values found in dictionary")
            return

        model = SentenceTransformer(FIELD_MODEL)
        with torch.no_grad():
            embs = model.encode(engine.value_texts, convert_to_tensor=True)
            engine.value_embs = torch.nn.functional.normalize(embs, p=2, dim=1)

        logger.info(f"[ValueLoader] Loaded {len(engine.value_texts)} value entries from {resolved}")