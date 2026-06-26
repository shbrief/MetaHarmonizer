"""Configuration constants for schema mapping.

Thresholds, model keys, and the noise-value set below are resolved through the
unified settings layer (:mod:`metaharmonizer.settings`), so a project-level
``metaharmonizer.toml`` / ``[tool.metaharmonizer]`` table can override them
without code changes (project file > built-in default). Paths remain
environment-driven via :mod:`metaharmonizer._paths` and the path env vars.
"""
import os
from pathlib import Path
from metaharmonizer._paths import DATA_DIR, resolve_data_file
from metaharmonizer.utils.model_loader import load_method_model_dict
from metaharmonizer.settings import get_settings, load_project_config

_method_model_dict = load_method_model_dict()
_settings = get_settings()
_project = load_project_config()


def _resolve_model(key: str) -> str:
    """Map a method key (e.g. ``"minilm-l6"``) to its model repo via the YAML."""
    if key not in _method_model_dict:
        raise KeyError(
            f"Unknown model key {key!r} (set via field_model/llm_model in the "
            f"project config). Known keys: {sorted(_method_model_dict)}"
        )
    return _method_model_dict[key]


# === Paths ===
OUTPUT_DIR = Path(os.getenv("SM_OUTPUT_DIR", DATA_DIR / "schema_mapping_eval"))
# Default curated schema: user data dir first, else the bundled copy that
# ships inside the wheel (resolve_data_file falls back automatically). Users
# typically override this via SchemaMapEngine(curated_dict_path=...).
CURATED_DICT_PATH = resolve_data_file("schema/cbio_target_attrs.csv")
# Alias dict is keyed to the bundled curated schema. engine.py disables it
# when the user supplies their own schema, so the bundled fallback is only
# read alongside the bundled cbio_target_attrs.csv.
ALIAS_DICT_PATH = resolve_data_file("schema/cbio_target_attrs_alias_manual.csv")
# Value dict is filtered against the active curated schema by ValueLoader
# (allowed_fields=standard_fields), so the bundled copy is safe to fall
# through to a user-supplied schema — disjoint keys are skipped automatically.
VALUE_DICT_PATH = os.getenv("FIELD_VALUE_JSON") or resolve_data_file("schema/field_value_dict.json")

# === Models === (project file may override the method key)
FIELD_MODEL = _resolve_model(_settings.field_model)
LLM_MODEL = _resolve_model(_settings.llm_model)

# === Thresholds === (project file > built-in default, via the resolver)
FUZZY_THRESH = _settings.fuzzy_thresh
NUMERIC_THRESH = _settings.numeric_thresh
FIELD_ALIAS_THRESH = _settings.field_alias_thresh
VALUE_DICT_THRESH = _settings.value_dict_thresh
VALUE_UNIQUE_CAP = _settings.value_unique_cap
VALUE_PERCENTAGE_THRESH = _settings.value_percentage_thresh
LLM_THRESHOLD = _settings.llm_threshold

# === Noise Values === (project file `noise_values` list replaces the default)
_DEFAULT_NOISE_VALUES = {
    "yes", "no", "true", "false", "unknown", "not reported", "not available",
    "na", "n/a", "none", "other", "missing", "not evaluated", "uninformative",
    "pending", "undetermined", "positive", "negative", "not applicable"
}
_noise_override = _project.get("noise_values")
NOISE_VALUES = (
    {str(v).lower() for v in _noise_override}
    if _noise_override else _DEFAULT_NOISE_VALUES
)
