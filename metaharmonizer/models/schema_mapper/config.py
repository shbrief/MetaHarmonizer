"""Configuration constants for schema mapping."""
import os
from pathlib import Path
from metaharmonizer._paths import DATA_DIR, resolve_data_file
from metaharmonizer.utils.model_loader import load_method_model_dict

_method_model_dict = load_method_model_dict()

# === Paths ===
OUTPUT_DIR = Path(os.getenv("SM_OUTPUT_DIR", DATA_DIR / "schema_mapping_eval"))
# Default curated schema: user data dir first, else the bundled copy that
# ships inside the wheel (resolve_data_file falls back automatically). Users
# typically override this via SchemaMapEngine(curated_dict_path=...).
CURATED_DICT_PATH = resolve_data_file("schema/curated_fields.csv")
# Alias dict is keyed to the bundled curated schema; no bundled copy, so only
# honored when METAHARMONIZER_DATA_DIR points at a dir that contains it.
ALIAS_DICT_PATH = DATA_DIR / "schema" / "curated_fields_source_latest_with_flags.csv"
VALUE_DICT_PATH = os.getenv("FIELD_VALUE_JSON") or DATA_DIR / "schema" / "field_value_dict.json"

# === Models ===
FIELD_MODEL = _method_model_dict["minilm-l6"]
LLM_MODEL = _method_model_dict["gemma-27b"]

# === Thresholds ===
FUZZY_THRESH = 92
NUMERIC_THRESH = 0.6
FIELD_ALIAS_THRESH = 0.5
VALUE_DICT_THRESH = 0.85
VALUE_UNIQUE_CAP = 50
VALUE_PERCENTAGE_THRESH = 0.2
LLM_THRESHOLD = 0.5

# === Noise Values ===
NOISE_VALUES = {
    "yes", "no", "true", "false", "unknown", "not reported", "not available",
    "na", "n/a", "none", "other", "missing", "not evaluated", "uninformative",
    "pending", "undetermined", "positive", "negative", "not applicable"
}