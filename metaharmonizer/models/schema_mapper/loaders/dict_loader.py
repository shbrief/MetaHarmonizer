"""Loaders for dictionary data."""
import pandas as pd
from pathlib import Path
from typing import Optional, Union
from sentence_transformers import SentenceTransformer
from metaharmonizer.utils.schema_mapper_utils import normalize
from .. import config as _config
from ..config import FIELD_MODEL
from metaharmonizer.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='WARNING')


class DictLoader:
    """Load and prepare dictionary data."""

    @staticmethod
    def load_standard_dict(path: Optional[Union[str, Path]] = None):
        """
        Load standard fields dictionary.

        Args:
            path: Path to curated CSV. If None, falls back to config.CURATED_DICT_PATH.

        Returns:
            tuple: (standard_fields, standard_fields_normed, normed_std_to_std, curated_df)
        """
        resolved = Path(path) if path else Path(_config.CURATED_DICT_PATH)
        if not resolved.exists():
            raise FileNotFoundError(
                f"Curated schema CSV not found at {resolved}. "
                "Pass curated_dict_path=... to SchemaMapEngine, "
                "or set METAHARMONIZER_DATA_DIR so the default path resolves."
            )
        curated_df = pd.read_csv(resolved)
        standard_fields = curated_df['field_name'].dropna().unique().tolist()
        standard_fields_normed = [normalize(f) for f in standard_fields]
        normed_std_to_std = {normalize(f): f for f in standard_fields}

        logger.info(f"[DictLoader] Loaded {len(standard_fields)} standard fields from {resolved}")
        return standard_fields, standard_fields_normed, normed_std_to_std, curated_df

    @staticmethod
    def load_alias_dict(path: Optional[Union[str, Path]] = None):
        """
        Load alias dictionary.

        Args:
            path: Path to alias CSV. If None, falls back to config.ALIAS_DICT_PATH.
                  Pass "" to disable alias matching explicitly.

        Returns:
            tuple: (sources_to_fields, sources_keys, normed_source_to_source, has_alias)
            If alias dict doesn't exist or is disabled, returns empty structures and has_alias=False.
        """
        resolved = path if path is not None else _config.ALIAS_DICT_PATH

        # Explicit disable: empty string or falsy (but not None, handled above)
        if not resolved:
            logger.info("[DictLoader] Alias dictionary disabled, skipping alias matching")
            return {}, [], {}, False

        resolved = Path(resolved)
        if not resolved.exists():
            logger.warning(f"[DictLoader] Alias dictionary not found at {resolved}, skipping alias matching")
            return {}, [], {}, False

        try:
            df_dict = pd.read_csv(resolved)
        except Exception as e:
            logger.warning(f"[DictLoader] Failed to load alias dictionary: {e}, skipping alias matching")
            return {}, [], {}, False
        
        if df_dict.empty:
            logger.warning(f"[DictLoader] Alias dictionary is empty, skipping alias matching")
            return {}, [], {}, False
        
        # Normalize alias to fields mapping
        sources_to_fields = (
            df_dict.groupby(df_dict['source'].map(normalize))['field_name']
            .apply(lambda s: sorted(set(s)))
            .to_dict()
        )
        
        sources_keys = list(sources_to_fields.keys())
        
        normed_source_to_source = {}
        for _, row in df_dict.dropna(subset=['source']).iterrows():
            norm_src = normalize(row['source'])
            if norm_src not in normed_source_to_source:
                normed_source_to_source[norm_src] = row['source']
        
        logger.info(f"[DictLoader] Loaded {len(sources_keys)} alias sources")
        return sources_to_fields, sources_keys, normed_source_to_source, True
    
    @staticmethod
    def load_numeric_dict(df_dict: pd.DataFrame):
        """
        Load numeric field dictionary.
        
        Args:
            df_dict: Full alias dictionary DataFrame
            
        Returns:
            tuple: (df_num, numeric_sources)
            If df_dict is None or empty, returns (None, [])
        """
        if df_dict is None or df_dict.empty:
            logger.warning("[DictLoader] No alias dictionary for numeric fields")
            return None, []
        
        df_num = df_dict[df_dict['is_numeric_field'] == 'yes']
        numeric_sources = df_num['source'].dropna().unique().tolist()
        
        logger.info(f"[DictLoader] Loaded {len(numeric_sources)} numeric sources")
        return df_num, numeric_sources
    
    @staticmethod
    def encode_fields(fields: list, model_name: str = FIELD_MODEL):
        """
        Encode fields using SentenceTransformer.
        
        Args:
            fields: List of field strings to encode
            model_name: Name of the sentence transformer model
            
        Returns:
            torch.Tensor: Encoded embeddings, or None if fields is empty
        """
        if not fields:
            logger.warning("[DictLoader] No fields to encode")
            return None
        
        model = SentenceTransformer(model_name)
        embeddings = model.encode(fields, convert_to_tensor=True)
        logger.info(f"[DictLoader] Encoded {len(fields)} fields using {model_name}")
        return embeddings