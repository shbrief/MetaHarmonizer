"""FieldSuggester – standalone new-field discovery via hybrid NER + embedding clustering."""

from .field_suggester import FieldSuggester
from .integration import suggest_from_schema_mapper

__all__ = ["FieldSuggester", "suggest_from_schema_mapper"]
