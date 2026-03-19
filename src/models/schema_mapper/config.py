"""Configuration constants for schema mapping."""
import os
from pathlib import Path
from src._paths import DATA_DIR
from src.utils.model_loader import load_method_model_dict

_method_model_dict = load_method_model_dict()

# === Paths ===
OUTPUT_DIR = Path(os.getenv("SM_OUTPUT_DIR", Path.cwd() / "schema_mapping_eval"))
# CURATED_DICT_PATH = DATA_DIR / "schema" / "curated_fields.csv"
CURATED_DICT_PATH = DATA_DIR / "schema" / "schema30.csv"
ALIAS_DICT_PATH = ""
# ALIAS_DICT_PATH = DATA_DIR / "schema" / "curated_fields_source_latest_with_flags.csv"
# ALIAS_DICT_PATH = DATA_DIR / "schema" / "heterogeneous_attribute_mapping_ver1.csv"
VALUE_DICT_PATH = os.getenv("FIELD_VALUE_JSON") or DATA_DIR / "schema" / "field_value_dict.json"
# VALUE_DICT_PATH = "data/schema/value_dictionary.json"

# === Models ===
FIELD_MODEL = _method_model_dict["minilm-l6"]
LLM_MODEL = _method_model_dict["gemma-27b"]

# === Thresholds ===
FUZZY_THRESH = 92
NUMERIC_THRESH = 0.6
FIELD_ALIAS_THRESH = 0.5
VALUE_DICT_THRESH = 0.85
VALUE_UNIQUE_CAP = 50
VALUE_PERCENTAGE_THRESH = 0.15
LLM_THRESHOLD = 0.5

# === Noise Values ===
NOISE_VALUES = {
    "yes", "no", "true", "false", "unknown", "not reported", "not available",
    "na", "n/a", "none", "other", "missing", "not evaluated", "uninformative",
    "pending", "undetermined", "positive", "negative", "not applicable"
}