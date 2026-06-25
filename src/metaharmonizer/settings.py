"""Unified configuration resolver for MetaHarmonizer.

Implements the package-wide precedence chain:

    explicit argument  >  environment variable  >  project config file  >  built-in default

- **Arguments**: per-run engine/CLI parameters (handled at the call site, e.g.
  ``SchemaMapEngine(value_dict_path=...)`` or ``OntoMapEngine(corpus_hash=...)``).
- **Environment variables**: secrets + deployment/ops. Path env vars are
  resolved in :mod:`metaharmonizer._paths`; secrets are read lazily here.
- **Project file**: ``metaharmonizer.toml`` (whole file) or the
  ``[tool.metaharmonizer]`` table in ``pyproject.toml`` in the current
  working directory.
- **Built-in defaults**: the field defaults on :class:`Settings`.

Secrets are deliberately *not* stored on the :class:`Settings` object — read
them via the properties so they never land in ``repr()``, logs, or a file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from metaharmonizer import _paths

try:  # Python >= 3.11 ships tomllib in the stdlib
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - 3.10 fallback
    try:
        import tomli as _toml  # optional backport
    except ModuleNotFoundError:
        _toml = None


@lru_cache(maxsize=1)
def load_project_config() -> dict:
    """Return the ``[tool.metaharmonizer]`` table from the nearest project file.

    Looks in the current working directory for ``metaharmonizer.toml`` (the
    whole file is the config) and then ``pyproject.toml`` (the
    ``[tool.metaharmonizer]`` table). Returns an empty dict when no file is
    found, the table is absent/empty, or no TOML parser is available
    (Python < 3.11 without ``tomli`` installed).

    Cached for the life of the process; the project file is read once.
    """
    if _toml is None:
        return {}
    cwd = Path.cwd()
    for path, is_pyproject in ((cwd / "metaharmonizer.toml", False),
                               (cwd / "pyproject.toml", True)):
        if not path.exists():
            continue
        try:
            data = _toml.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        section = (data.get("tool", {}).get("metaharmonizer", {})
                   if is_pyproject else data)
        if section:
            return dict(section)
    return {}


def _pick(overrides: dict, file_cfg: dict, name: str,
          env_var: str | None, default, cast=None):
    """Resolve one value: arg > env > project-file > built-in default."""
    if overrides.get(name) is not None:
        val = overrides[name]
    elif env_var and os.getenv(env_var) not in (None, ""):
        val = os.getenv(env_var)
    elif name in file_cfg:
        val = file_cfg[name]
    else:
        val = default
    return cast(val) if (cast is not None and val is not None) else val


@dataclass(frozen=True)
class Settings:
    """Immutable, resolved configuration snapshot.

    Build via :meth:`resolve` (or :func:`get_settings` for the cached
    process-wide instance). Tunables below may be overridden by a project
    file; secrets are exposed as properties, not fields.
    """

    # --- Tier C tunables (project-file / built-in default) ---
    # NOTE: ``field_model`` is a *method key* from method_model.yaml (e.g.
    # "minilm-l6"), distinct from the raw FIELD_MODEL env var that
    # FieldSuggester reads as a model name. It is intentionally not wired to
    # that env var to avoid key/name collisions.
    field_model: str = "minilm-l6"
    llm_model: str = "gemma-27b"
    topk: int = 5
    fuzzy_thresh: int = 92
    numeric_thresh: float = 0.6
    field_alias_thresh: float = 0.5
    value_dict_thresh: float = 0.85
    value_unique_cap: int = 50
    value_percentage_thresh: float = 0.2
    llm_threshold: float = 0.5

    # --- Tier B deployment paths (defaults already env-resolved in _paths) ---
    data_dir: Path = field(default_factory=lambda: _paths.DATA_DIR)
    sm_output_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SM_OUTPUT_DIR", str(_paths.DATA_DIR / "schema_mapping_eval"))
        )
    )

    # --- Secrets: env-only, lazy, never stored or logged ---
    @property
    def umls_api_key(self) -> str | None:
        return os.getenv("UMLS_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.getenv("GEMINI_API_KEY")

    @classmethod
    def resolve(cls, **overrides) -> "Settings":
        """Resolve a :class:`Settings` via arg > env > project-file > default."""
        f = load_project_config()
        o = overrides
        return cls(
            field_model=_pick(o, f, "field_model", None, "minilm-l6"),
            llm_model=_pick(o, f, "llm_model", None, "gemma-27b"),
            topk=_pick(o, f, "topk", None, 5, int),
            fuzzy_thresh=_pick(o, f, "fuzzy_thresh", None, 92, int),
            numeric_thresh=_pick(o, f, "numeric_thresh", None, 0.6, float),
            field_alias_thresh=_pick(o, f, "field_alias_thresh", None, 0.5, float),
            value_dict_thresh=_pick(o, f, "value_dict_thresh", None, 0.85, float),
            value_unique_cap=_pick(o, f, "value_unique_cap", None, 50, int),
            value_percentage_thresh=_pick(
                o, f, "value_percentage_thresh", None, 0.2, float),
            llm_threshold=_pick(o, f, "llm_threshold", None, 0.5, float),
            data_dir=Path(_pick(o, f, "data_dir", "METAHARMONIZER_DATA_DIR",
                                str(_paths.DATA_DIR))),
            sm_output_dir=Path(_pick(o, f, "sm_output_dir", "SM_OUTPUT_DIR",
                                     str(_paths.DATA_DIR / "schema_mapping_eval"))),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached :class:`Settings` (env + project file read once)."""
    return Settings.resolve()
