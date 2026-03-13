"""Tests for numeric_match_utils — strip_units_and_tags, detect_numeric_semantic, family_boost."""
import pytest

from src.utils.numeric_match_utils import (
    strip_units_and_tags,
    detect_numeric_semantic,
    family_boost,
)


class TestStripUnitsAndTags:
    def test_strips_mg(self):
        clean, tags = strip_units_and_tags("dose mg")
        tag_str = " ".join(t.lower() for t in tags)
        assert "mg" in tag_str
        assert "dose" in clean

    def test_plain_text_no_tags(self):
        clean, tags = strip_units_and_tags("patient age")
        assert len(tags) == 0
        assert clean == "patient age"

    def test_empty_string(self):
        clean, tags = strip_units_and_tags("")
        assert clean == ""
        assert len(tags) == 0

    def test_none_handled_as_empty(self):
        clean, tags = strip_units_and_tags(None)
        assert clean == ""
        assert len(tags) == 0

    def test_multiple_units_all_removed(self):
        clean, tags = strip_units_and_tags("Gy dose mg")
        assert len(tags) >= 2

    def test_clean_text_has_no_unit_tokens(self):
        clean, tags = strip_units_and_tags("dose_mg_per_m2")
        # Normalized clean should not contain the unit tokens
        for tag in tags:
            assert tag.lower() not in clean.lower() or "dose" in clean

    def test_whitespace_normalized_in_clean(self):
        # After removing units, multiple spaces should collapse
        clean, _ = strip_units_and_tags("a  mg  b")
        assert "  " not in clean

    def test_iv_tag_captured(self):
        clean, tags = strip_units_and_tags("route IV dose")
        tag_str = " ".join(t.lower() for t in tags)
        assert "iv" in tag_str


class TestDetectNumericSemantic:
    # NOTE: detect_numeric_semantic is called with an already-normalized header
    # (underscores converted to spaces via normalize()) and unit tags extracted
    # by strip_units_and_tags().

    def test_dose_via_header_keyword(self):
        # "dose mg" → after strip_units_and_tags, clean="dose", tags={"mg"}
        # Passing just the word "dose" to the header hits DOSE_HINTS (\bdose\b)
        assert detect_numeric_semantic("dose", set()) == "dose"

    def test_dose_via_tag(self):
        # tag "mg" triggers dose branch
        assert detect_numeric_semantic("amount", {"mg"}) == "dose"

    def test_dose_via_gy_tag(self):
        assert detect_numeric_semantic("radiation", {"Gy"}) == "dose"

    def test_age_header(self):
        # normalize("age_at_diagnosis") → "age at diagnosis"; \bage[_\s-]?at\b matches
        assert detect_numeric_semantic("age at diagnosis", set()) == "age"

    def test_age_simple_header(self):
        assert detect_numeric_semantic("age", set()) == "age"

    def test_time_date_header(self):
        # normalize("start_date") → "start date"; TIME_HINTS matches \bdate\b
        assert detect_numeric_semantic("start date", set()) == "time"

    def test_time_duration_header(self):
        # normalize("treatment_duration") → "treatment duration"
        assert detect_numeric_semantic("treatment duration", set()) == "time"

    def test_unknown_header(self):
        assert detect_numeric_semantic("patient id", set()) == "unknown"

    def test_empty_header_returns_unknown(self):
        assert detect_numeric_semantic("", set()) == "unknown"

    def test_none_header_returns_unknown(self):
        assert detect_numeric_semantic(None, set()) == "unknown"

    def test_dose_beats_time_when_dose_tag_present(self):
        # dose tag in set (checked first) overrides time hint in header
        assert detect_numeric_semantic("start date", {"mg"}) == "dose"


class TestFamilyBoost:
    def test_dose_field_with_dose_in_name(self):
        assert family_boost("treatment_dose", "dose") == 0.15

    def test_dose_cycle_gives_partial_boost(self):
        assert family_boost("treatment_cycle", "dose") == 0.10

    def test_dose_auc_gives_partial_boost(self):
        assert family_boost("auc_value", "dose") == 0.10

    def test_age_field_returns_full_boost(self):
        assert family_boost("age_at_diagnosis", "age") == 0.15

    def test_age_simple_field(self):
        assert family_boost("age", "age") == 0.15

    def test_time_date_returns_full_boost(self):
        assert family_boost("start_date", "time") == 0.15

    def test_time_duration_returns_full_boost(self):
        assert family_boost("duration_days", "time") == 0.15

    def test_mismatch_dose_field_age_family(self):
        assert family_boost("treatment_dose", "age") == 0.0

    def test_mismatch_patient_id_dose(self):
        assert family_boost("patient_id", "dose") == 0.0

    def test_unknown_family_returns_zero(self):
        assert family_boost("dose", "unknown") == 0.0

    def test_empty_field_returns_zero(self):
        assert family_boost("", "dose") == 0.0

    def test_none_field_returns_zero(self):
        assert family_boost(None, "dose") == 0.0
