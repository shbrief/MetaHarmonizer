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
# Default curated schema: the 736-field GDC schema. User data dir first, else
# the bundled copy that ships inside the wheel (resolve_data_file falls back
# automatically). Users typically override this via the ``schema`` preset arg
# or SchemaMapEngine(target_schema_path=...).
TARGET_SCHEMA_PATH = resolve_data_file("schema/gdc_target_attrs.csv")
# Alias dict is keyed to the default curated schema (GDC). engine.py disables it
# when the user supplies their own schema, so the bundled fallback is only
# read alongside the bundled gdc_target_attrs.csv.
ALIAS_DICT_PATH = resolve_data_file("schema/gdc_target_attrs_alias_haiku.csv")
# Value dict is filtered against the active curated schema by ValueLoader
# (allowed_fields=standard_fields), so the bundled copy is safe to fall
# through to a user-supplied schema — disjoint keys are skipped automatically.
VALUE_DICT_PATH = os.getenv("TARGET_ATTRS_ALLOWED_VALUES_JSON") or resolve_data_file("schema/gdc_target_attrs_allowed_values.json")

# === Bundled schema presets ===
# A preset bundles a curated schema with its matched alias + value dicts as a
# coherent set. Selecting one (``SchemaMapEngine(schema="gdc")``) supplies all
# three together, so the alias dict is NOT auto-disabled the way it is when only
# ``target_schema_path`` is overridden (the alias is keyed to *that* schema, not
# the default). Explicit ``*_dict_path`` args still win over a preset.
#   - cbio: 33-field cBioPortal schema + manually curated aliases
#   - gdc:  736-field GDC schema + Haiku-4.5-generated aliases (the default; see
#           the sibling ``gdc_target_attrs_alias_haiku.meta.json`` for provenance)
SCHEMA_PRESETS: dict[str, dict[str, str]] = {
    "cbio": {
        "target_schema_path": "schema/cbio_target_attrs.csv",
        "alias_dict_path": "schema/cbio_target_attrs_alias_manual.csv",
        "value_dict_path": "schema/cbio_target_attrs_allowed_values.json",
    },
    "gdc": {
        "target_schema_path": "schema/gdc_target_attrs.csv",
        "alias_dict_path": "schema/gdc_target_attrs_alias_haiku.csv",
        "value_dict_path": "schema/gdc_target_attrs_allowed_values.json",
    },
}


def resolve_schema_preset(name: str) -> dict[str, "Path"]:
    """Resolve a :data:`SCHEMA_PRESETS` entry to absolute (bundled or user) paths.

    Raises ``KeyError`` with the known preset names if ``name`` is unknown.
    """
    if name not in SCHEMA_PRESETS:
        raise KeyError(
            f"Unknown schema preset {name!r}. Known presets: {sorted(SCHEMA_PRESETS)}."
        )
    return {k: resolve_data_file(v) for k, v in SCHEMA_PRESETS[name].items()}

# === Models === (project file may override the method key)
FIELD_MODEL = _resolve_model(_settings.field_model)
LLM_MODEL = _resolve_model(_settings.llm_model)

# === Thresholds ===
# Resolved tunables (fuzzy_thresh, numeric_thresh, field_alias_thresh,
# value_dict_thresh, value_unique_cap, value_percentage_thresh, llm_threshold)
# live on the :class:`metaharmonizer.settings.Settings` snapshot, read live by
# the engine and matchers via ``self.settings.*``. This lets a per-run
# ``SchemaMapEngine(settings=...)`` override them; module-level constants here
# would freeze a single process-wide value at import. Build a snapshot with
# ``Settings.resolve(**overrides)`` (arg > env > project file > default).

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
