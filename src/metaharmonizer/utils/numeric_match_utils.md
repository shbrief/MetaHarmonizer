# Numeric Matching Rules — MetaHarmonizer

TL;DR
- Purpose: Identify and classify numeric-like fields (e.g., dose, age, time) by stripping unit tokens and using header/tags heuristics.
- How it works: `UNIT_PATTERNS` find and remove unit-like tokens (e.g., `mg/m2`, `AUC12`, `d1-3`) and return them as tags; `detect_numeric_semantic(header, tags)` then uses these tags plus header hint regexes to label fields as `"dose"`, `"age"`, `"time"`, or `"unknown"`; `family_boost(field_name, family)` gives small name-based score boosts.
- When to extend: Add new unit tokens to `UNIT_PATTERNS`, add new hint regexes for extra semantic families, or convert substring boosts to regexes for greater precision.

---

This document explains the rules, patterns, and heuristics implemented in `metaharmonizer/utils/numeric_match_utils.py` for identifying and classifying numeric-like fields (columns) in datasets. It describes the intent behind each regular expression and function, how they interact, and guidance for extending or tuning the behavior.

File referenced:
- `metaharmonizer/utils/numeric_match_utils.py`

---

## Overview

The numeric matching utilities perform two related tasks:

1. Strip out known unit tokens and unit-like tags from a text string (typically a column header or column sample value) so that numeric values and the remaining context are easier to analyze.
2. Use the header text and the extracted tags to detect the semantic category (family) of a numeric field: specifically `dose`, `age`, `time`, or `unknown`. A small scoring helper (`family_boost`) provides name-based boosts for matching.

This is intended to help downstream harmonization logic decide how to treat numeric columns (for example, whether a numeric column represents a drug dose vs an age vs a duration).

---

## Unit / Token Patterns

`UNIT_PATTERNS` is a list of regular expressions used to identify tokens that look like units, dosing schedules, or other metadata that should be removed from a header or text before further analysis.

| Pattern (literal) | Example matches | Intent / Notes |
|---|---:|---|
| `\bmg(?:/m2|/m²)?\b` | mg, mg/m2, mg/m² | Milligrams and mg per square meter (drug doses) |
| `\bµg\b` | µg | Microgram symbol |
| `\bmcg\b` | mcg | ASCII microgram |
| `\bIU\b` | IU | International Units |
| `\bU/mL\b` | U/mL | Units per mL |
| `\bAUC\d+\b` | AUC12, AUC0 | Area-under-curve tokens (AUC + digits) |
| `\bGy\b` | Gy | Radiation dose (Gray) |
| `\b%\b` | % | Percentage token |
| `\bmmHg\b` | mmHg | Blood pressure unit |
| `\bq\d+[wd]\b` | q12w, q3d | Frequency shorthand (q + digits + w/d) |
| `\bqd\b` | qd | Once daily shorthand |
| `\bbid\b` | bid | Twice daily shorthand |
| `\btid\b` | tid | Thrice daily shorthand |
| `\bd\d+(?:[,-]\d+)*\b` | d1, d1-3, d1,3,5 | Day numbering / ranges (cycle-day notation) |
| `\bx\d+\s*cycles?\b` | x4 cycles, x6 cycle | "xN cycles" notation |
| `\bIV\b` | IV | Intravenous administration route |
| `\bPO\b` | PO | Oral administration (per os) |
| `\bSC\b` | SC | Subcutaneous |
| `\bICD-10\b` | ICD-10 | Explicit code system token |

- Compiled into `UNIT_REGEX` using `re.IGNORECASE`.
- Matches are extracted as tags and removed from the cleaned header/text.
- Purpose: capture unit-like tokens so they can be removed from the cleaned text AND retained as tags for downstream semantic detection.

---

## Tag Extraction / Stripping

Function: `strip_units_and_tags(text: str) -> (clean, tags)`

- Runs `UNIT_REGEX.sub(_tagger, text)`:
  - Each match is replaced by a single space in the cleaned text.
  - The matched token is collected into a `set` called `tags` so the original unit token(s) are available for later semantic detection.
- After substitution, runs `re.sub(r"\s+", " ", ...)` and `.strip()` to normalize whitespace.
- Returns:
  - `clean`: the cleaned header/text with units removed.
  - `tags`: set of matched unit tokens (preserves original spelling/case as matched, though matching is case-insensitive).

Example:
- Input: `"Dose (mg/m2) at d1-3"`
- Clean: `"Dose (at)"` (roughly)
- Tags: `{"mg/m2", "d1-3"}`

Tags are used to boost or directly detect semantic families — e.g., presence of `mg` or `AUC...` strongly suggests `dose`.

---

## Semantic Detection

Function: `detect_numeric_semantic(header: str, tags: set) -> str`

Classifies a numeric field into one of: `"dose"`, `"age"`, `"time"`, or `"unknown"`. Uses both the header string and the extracted tags for robust detection.

Priority order:

| Priority | Check | Detail | Outcome |
|---:|---|---|---|
| 1 | Dose (tag or hint) | Any tag matches `(AUC\d+|Gy|mg\b|mg/m2|mg/m²)` OR `DOSE_HINTS` matches header | `"dose"` |
| 2 | Age (header) | `AGE_HINTS` matches header | `"age"` |
| 3 | Time (header) | `TIME_HINTS` matches header | `"time"` |
| 4 | Fallback | none of the above | `"unknown"` |

Notes:
- Tags get first-class consideration for `dose` because tokens like `mg`, `AUC`, or `Gy` are strong indicators a numeric column is a dose.
- Header searches use word-boundary anchored regexes to avoid false positives inside other words.
- All regexes compiled with `IGNORECASE`.
- The function is deliberately conservative — returns a single family string, not a probability.

---

## Name-based Boosting

Function: `family_boost(field_name: str, family: str) -> float`

Provides a small additive boost for a given `(field_name, family)` pair, intended to combine with other heuristic scores in a larger matching algorithm. Matching is simple substring membership in lowercased `field_name`.

| Family | Field name contains → boost | Rationale |
|---|---:|---|
| dose | `"treatment_dose"` or `"dose"` → 0.15 | Direct dose naming |
| dose | `"treatment_number"`, `"cycle"`, `"auc"`, or `"unit"` → 0.10 | Related dose/cycle naming |
| age | `"age"` → 0.15 | Direct age naming |
| age | `"age_group"` → 0.10 | Age-group naming |
| time | any of `["date","start","end","duration","frequency","time"]` → 0.15 | Time/date/duration naming |
| time | `"unit"` AND any of `["duration","frequency","start","end"]` → 0.10 | Unit-of-time fields |
| — | no match | 0.0 |

Notes:
- Matching is deliberately simple/fast; consider upgrading to regex token matching for stricter boundaries (e.g., avoid `"age-old"` triggering age).
- Boosts are small and meant to nudge an overall scoring pipeline, not be determinative alone.

---

## Regex Constants / Hints

| Constant | Pattern (literal) | Purpose |
|---|---:|---|
| `DOSE_HINTS` | `\b(dose|auc|gy|mg|m2|m²|regimen|cycle)\b` | Dose-related hints in header text |
| `AGE_HINTS` | `\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b` | Age-related headers (e.g., `age`, `age_at`, `diagnosis_age`) |
| `TIME_HINTS` | `\b(date|day|days|month|months|year|years|start|end|duration|freq(uency)?|interval)\b` | Time/date/duration headers |
| `UNIT_REGEX` | alternation of `UNIT_PATTERNS` | Matches unit-like tokens to remove and tag |

All compiled with `IGNORECASE`. `AGE_HINTS` and `TIME_HINTS` focus on header terms; `DOSE_HINTS` also overlaps with common dose unit tokens. Chosen to reflect common clinical dataset header tokens.

---

## Examples

| Header | Tags | `detect_numeric_semantic` | `family_boost` |
|---|---|---|---|
| `"Cumulative Dose (mg/m2)"` | `{"mg/m2"}` | `"dose"` | `("cumulative_dose","dose")` → 0.15 |
| `"Age at Diagnosis (years)"` | `{}` (`years` via `TIME_HINTS`, not `UNIT_REGEX`) | `"age"` (`AGE_HINTS` matches) | `("age_at_diagnosis","age")` → 0.15 |
| `"Start date"` | `{}` | `"time"` (`TIME_HINTS` matches `date`/`start`) | `("start_date","time")` → 0.15 |
| `"Visit Number"` | `{}` | `"unknown"` | `("visit_number","age")` → 0.0 |

---

## Extension and Tuning Guidance

- Adding units:
  - Append more unit regexes to `UNIT_PATTERNS` for tokens observed in your data (e.g., `\bng/mL\b`, `\bL\b`, `\bkg\b`, `\bcm\b`).
  - Keep tokens word-boundary anchored (`\b`) to avoid accidental substring matches.

- Improving semantic detection:
  - Add more families (e.g., `measurement`, `vital_sign`) via additional hint regexes, ordered by priority.
  - For subtle cases, consider producing a confidence score instead of a single label.

- More robust tagging:
  - Current tag extractor returns the literal matched string. To canonicalize (e.g., `mcg` → `µg` or `mg/m2` → `mg/m²`), post-process `tags` after extraction.
  - If values contain units as suffixes (e.g., `"12 mg/kg"`), run `strip_units_and_tags` on sample values, not only headers.

- Better name matching:
  - `family_boost` uses substrings. To avoid accidental matches (e.g., `"age-old"` triggering age), switch to regex with word boundaries or a tokenized approach.

- Internationalization:
  - For non-English headers, add localized tokens to `TIME_HINTS`, `AGE_HINTS`, and `UNIT_PATTERNS` (e.g., `"años"` for Spanish years).

---

## Implementation Notes & Limitations

- The rules are heuristic and tuned for clinical/biomedical datasets (hence many clinical dosing tokens).
- The code is conservative — prefers `"unknown"` when ambiguous rather than risk misclassification.
- `years` and `months` are detected as `time`, not `age` — `AGE_HINTS` requires the explicit `age` token. Adjust if downstream needs differ.
- Regex tokenization is case-insensitive, but tags preserve the matched substring; normalize case if comparing tags later.

---

## Where to Look in Code

- Core code: `metaharmonizer/utils/numeric_match_utils.py`
  - `strip_units_and_tags`
  - `detect_numeric_semantic`
  - `family_boost`
- Unit patterns and hint regexes are defined at the top of the module.
