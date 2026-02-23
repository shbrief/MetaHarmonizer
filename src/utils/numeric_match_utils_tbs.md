# Numeric Matching Rules — MetaHarmonizer

TL;DR
- Purpose: Identify and classify numeric-like fields (e.g., dose, age, time) by stripping unit tokens and using header/tags heuristics.
- How it works: UNIT_PATTERNS find and remove unit-like tokens (e.g., mg/m2, AUC12, d1-3) and return them as tags; detect_numeric_semantic(header, tags) then uses these tags plus header hint regexes to label fields as "dose", "age", "time", or "unknown"; family_boost(field_name, family) gives small name-based score boosts.
- When to extend: Add new unit tokens to UNIT_PATTERNS, add new hint regexes for extra semantic families, or convert substring boosts to regexes for greater precision.

---

This document explains the rules, patterns, and heuristics implemented in `src/utils/numeric_match_utils.py` for identifying and classifying numeric-like fields (columns) in datasets. The following sections present the same information as the original document but organized into tables for quick reference.

File referenced:
- `src/utils/numeric_match_utils.py`

---

## Unit / Token Patterns

Table of unit/token regex patterns and intent.

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

- Compiled into UNIT_REGEX using re.IGNORECASE.
- These matches are extracted as tags and removed from the cleaned header/text.

---

## Tag Extraction / Stripping (summary)

| Function | Input | Output | Behavior |
|---|---|---:|---|
| strip_units_and_tags(text: str) | Header or sample text | (clean: str, tags: set) | Replaces UNIT_REGEX matches with space, collects matched tokens into a set, normalizes whitespace, returns cleaned text and tags. |

Usage example:
- Input: "Dose (mg/m2) at d1-3"
- Clean: "Dose (at)"
- Tags: {"mg/m2", "d1-3"}

---

## Regex Hint Constants

| Constant | Pattern (literal) | Purpose |
|---|---:|---|
| AGE_HINTS | `\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b` | Identify age-related headers (e.g., "age", "age_at", "diagnosis_age") |
| TIME_HINTS | `\b(date|day|days|month|months|year|years|start|end|duration|freq(uency)?|interval)\b` | Identify time/date/duration-related headers |
| DOSE_HINTS | `\b(dose|auc|gy|mg|m2|m²|regimen|cycle)\b` | Identify dose-related hints in header text |
| UNIT_REGEX | alternation of UNIT_PATTERNS | Matches unit-like tokens to remove and tag |

Notes:
- All compiled with IGNORECASE to be case-insensitive.
- AGE_HINTS and TIME_HINTS focus on header terms; DOSE_HINTS also matches common dose unit tokens.

---

## Semantic Detection Rules (detect_numeric_semantic)

The classification logic follows a priority order. The following table summarizes the rule checks and outcomes.

| Priority | Condition | Check details | Outcome |
|---:|---|---|---|
| 1 | Dose from tags | Any tag matches `(AUC\d+|Gy|mg\b|mg/m2|mg/m²)` OR DOSE_HINTS matches header | "dose" |
| 2 | Age from header | AGE_HINTS matches header | "age" |
| 3 | Time from header | TIME_HINTS matches header | "time" |
| 4 | Fallback | none of the above | "unknown" |

Notes:
- Tags (extracted unit tokens) get first-class consideration for dose detection because tokens like `mg`, `AUC`, or `Gy` are strong indicators.
- The function returns one label (no probability).

---

## Name-based Boost Rules (family_boost)

Rules that provide small additive boosts based on field name substrings. All checks are case-insensitive (field_name is lowercased).

| Family | Field name contains -> boost | Rationale |
|---|---:|---:|
| dose | "treatment_dose" or "dose" -> 0.15 | Direct dose naming |
| dose | "treatment_number", "cycle", "auc", or "unit" -> 0.10 | Related dose/cycle naming |
| age | "age" -> 0.15 | Direct age naming |
| age | "age_group" -> 0.10 | Age-group naming |
| time | any of ["date","start","end","duration","frequency","time"] -> 0.15 | Time/date/duration naming |
| time | contains "unit" AND any of ["duration","frequency","start","end"] -> 0.10 | Unit-of-time fields |
| default | - -> 0.0 | No match |

Notes:
- Matching is simple substring membership; consider upgrading to regex token matching for stricter boundaries.
- Boosts are small and intended to nudge an overall scoring pipeline.

---

## Examples

| Header | Tags (example) | detect_numeric_semantic result | family_boost example |
|---|---:|---:|---|
| "Cumulative Dose (mg/m2)" | {"mg/m2"} | "dose" | family_boost("cumulative_dose","dose") -> 0.15 |
| "Age at Diagnosis (years)" | {} (years matched via TIME_HINTS) | "age" | family_boost("age_at_diagnosis","age") -> 0.15 |
| "Start date" | {} | "time" | family_boost("start_date","time") -> 0.15 |
| "Visit Number" | {} | "unknown" | family_boost("visit_number","age") -> 0.0 |

---

## Extension & Tuning (compact table)

| Topic | Action | Notes |
|---|---|---|
| Add units | Append to UNIT_PATTERNS (e.g., `\bng/mL\b`, `\bL\b`, `\bkg\b`) | Keep `\b` anchors to reduce false positives |
| More families | Add new HINT regexes and detection priority | Order by confidence; tags-first for unit-driven families |
| Canonical tags | Post-process extracted tags to normalize (e.g., "mcg" -> "µg") | Useful if you match tags downstream |
| Name matching | Replace substring checks with word-boundary regex or tokenization | Avoid accidental matches like "age-old" |
| Internationalization | Add localized tokens to hint regexes | e.g., "años" for Spanish years |

---

## Implementation Notes & Limitations

| Point | Explanation |
|---|---|
| Heuristic nature | Tuned for clinical/biomedical datasets; not exhaustive |
| Conservatism | Returns "unknown" when ambiguous to avoid misclassification |
| Years vs age | "years" is a TIME_HINT; AGE_HINTS requires explicit "age" token — adjust if needed |
| Tag case | Tags preserve matched form; normalize case when comparing later |

---

## Where to Look in Code

- `src/utils/numeric_match_utils.py`
  - strip_units_and_tags
  - detect_numeric_semantic
  - family_boost
- Regex and UNIT_PATTERNS defined at top of module.

---
