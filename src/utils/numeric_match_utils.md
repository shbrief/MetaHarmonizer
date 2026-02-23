# Numeric Matching Rules — MetaHarmonizer

TL;DR
- Purpose: Identify and classify numeric-like fields (e.g., dose, age, time) by stripping unit tokens and using header/tags heuristics.
- How it works: UNIT_PATTERNS find and remove unit-like tokens (e.g., mg/m2, AUC12, d1-3) and return them as tags; detect_numeric_semantic(header, tags) then uses these tags plus header hint regexes to label fields as "dose", "age", "time", or "unknown"; family_boost(field_name, family) gives small name-based score boosts.
- When to extend: Add new unit tokens to UNIT_PATTERNS, add new hint regexes for extra semantic families, or convert substring boosts to regexes for greater precision.

---

This document explains the rules, patterns, and heuristics implemented in `src/utils/numeric_match_utils.py` for identifying and classifying numeric-like fields (columns) in datasets. It describes the intent behind each regular expression and function, how they interact, and guidance for extending or tuning the behavior.

File referenced:
- `src/utils/numeric_match_utils.py`

---

## Overview

The numeric matching utilities perform two related tasks:

1. Strip out known unit tokens and unit-like tags from a text string (typically a column header or column sample value) so that numeric values and the remaining context are easier to analyze.
2. Use the header text and the extracted tags to detect the semantic category (family) of a numeric field: specifically `dose`, `age`, `time`, or `unknown`. A small scoring helper (`family_boost`) provides name-based boosts for matching.

This is intended to help downstream harmonization logic decide how to treat numeric columns (for example, whether a numeric column represents a drug dose vs an age vs a duration).

---

## Unit / Token Patterns

UNIT_PATTERNS is a list of regular expressions used to identify tokens that look like units, dosing schedules, or other metadata that should be removed from a header or text before further analysis.

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

Purpose:
- To capture unit-like tokens so they can be removed and also stored as extracted tags for downstream semantic detection.

---

## Tag Extraction / Stripping

Function: strip_units_and_tags(text: str) -> (clean, tags)

- Runs `UNIT_REGEX.sub(_tagger, text)`:
  - Each match is replaced by a single space in the cleaned text.
  - The matched token is collected into a `set` called `tags` so the original unit token(s) are available for later semantic detection.
- After substitution, runs `re.sub(r"\s+", " ", ...)` and `.strip()` to normalize whitespace.
- Returns:
  - `clean`: the cleaned header/text with units removed.
  - `tags`: set of matched unit tokens (preserves original spelling/case as matched, though matching is case-insensitive).

Usage note:
- Example: `"Dose (mg/m2) at d1-3"` -> cleaned: `"Dose (at)"` (roughly) and tags: `{"mg/m2", "d1-3"}`.
- The tags are used to boost or directly detect semantic families (e.g., presence of `mg` or `AUC...` strongly suggests `dose`).

---

## Semantic Detection

Function: detect_numeric_semantic(header: str, tags: set) -> str

This function classifies a numeric field into one of: `"dose"`, `"age"`, `"time"`, or `"unknown"`. It uses both the header string and the extracted tags for robust detection.

Detection logic (priority order):

1. Dose detection:
   - If any tag matches a dose-like token (via searching tags for regex `(AUC\d+|Gy|mg\b|mg/m2|mg/m²)`), OR
   - If `DOSE_HINTS` matches the header text: `\b(dose|auc|gy|mg|m2|m²|regimen|cycle)\b` (case-insensitive).
   - Then return `"dose"`.

2. Age detection:
   - If `AGE_HINTS` matches the header: `\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b`.
   - Then return `"age"`.

3. Time detection:
   - If `TIME_HINTS` matches the header: `\b(date|day|days|month|months|year|years|start|end|duration|freq(uency)?|interval)\b`.
   - Then return `"time"`.

4. Fallback:
   - If none of the above match, return `"unknown"`.

| Constant | Pattern (literal) | Purpose |
|---|---:|---|
| DOSE_HINTS | `\b(dose|auc|gy|mg|m2|m²|regimen|cycle)\b` | Identify dose-related hints in header text |
| AGE_HINTS | `\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b` | Identify age-related headers (e.g., "age", "age_at", "diagnosis_age") |
| TIME_HINTS | `\b(date|day|days|month|months|year|years|start|end|duration|freq(uency)?|interval)\b` | Identify time/date/duration-related headers |
| UNIT_REGEX | alternation of UNIT_PATTERNS | Matches unit-like tokens to remove and tag |


Notes:
- All compiled with IGNORECASE to be case-insensitive.
- AGE_HINTS and TIME_HINTS focus on header terms; DOSE_HINTS also matches common dose unit tokens.
- Tags get first-class consideration for dose because units like `mg`, `AUC`, or `Gy` are strong indicators a numeric column is a dose.
- The header search uses word-boundary anchored regexes to avoid false positives inside other words.
- `detect_numeric_semantic` is deliberately conservative: it returns a single family string and does not compute probabilities.

---

## Name-based Boosting

Function: family_boost(field_name: str, family: str) -> float

Purpose:
- Provide a small additive boost for a given (field_name, family) pair. Intended to combine with other heuristic scores in a larger matching algorithm.

Behavior:

- For family `"dose"`:
  - If `field_name` contains `"treatment_dose"` or `"dose"` -> return `0.15`.
  - Else if it contains `"treatment_number"`, `"cycle"`, `"auc"`, or `"unit"` -> return `0.10`.

- For family `"age"`:
  - If `field_name` contains `"age"` -> return `0.15`.
  - If it contains `"age_group"` -> return `0.10`.

- For family `"time"`:
  - If the lowercase `field_name` contains any of `["date", "start", "end", "duration", "frequency", "time"]` -> return `0.15`.
  - If it contains `"unit"` and also contains any of `["duration", "frequency", "start", "end"]` -> return `0.10`.

- Default: return `0.0` if no matches.

Notes:
- Matching is done by simple substring membership in lowercase `field_name`. This is intentionally simple and fast but can be extended to use regexes for more precise matching.
- Boosts are small; they are meant to nudge scoring in a larger pipeline rather than be determinative alone.

---

## Regex Constants / Hints

- UNIT_REGEX — compiled alternation of UNIT_PATTERNS (case-insensitive).
- AGE_HINTS — `\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b` (case-insensitive).
- TIME_HINTS — checks for date/day/month/year/start/end/duration/frequency/interval words using a single pattern (case-insensitive).
- DOSE_HINTS — contains tokens often used in dose contexts: dose, auc, gy, mg, m2, m², regimen, cycle.

These are chosen to reflect common clinical dataset header tokens and unit annotations.

---

## Examples

1. Header: `"Cumulative Dose (mg/m2)"`, tags found: `{"mg/m2"}`
   - `detect_numeric_semantic` -> `"dose"` (tags include `mg/m2` or DOSE_HINTS finds `dose`).
   - `family_boost("cumulative_dose", "dose")` -> 0.15 (because `"dose"` present).

2. Header: `"Age at Diagnosis (years)"`
   - Tag extraction might find `"years"` (if configured to match `%` style or not — note: years are detected via TIME_HINTS, not UNIT_REGEX).
   - `detect_numeric_semantic` -> because `AGE_HINTS` matches -> `"age"`.
   - `family_boost("age_at_diagnosis", "age")` -> 0.15.

3. Header: `"Start date"`
   - `detect_numeric_semantic` -> `"time"` (TIME_HINTS matches `date` or `start`).

4. Header: `"Visit Number"` or `"Patient ID"`
   - `detect_numeric_semantic` -> `"unknown"` (no hints matched).
   - `family_boost` will return 0.0 for typical field names here.

---

## Extension and Tuning Guidance

- Adding Units:
  - Add more unit regexes to `UNIT_PATTERNS` for tokens observed in your data (e.g., `\bng/mL\b`, `\bL\b`, `\bkg\b`, `\bcm\b`).
  - Keep tokens word-boundary anchored `\b` to avoid accidental substring matches.

- Improving Semantic Detection:
  - If more families are needed (e.g., `measurement`, `vital_sign`), add additional hint regexes and order them by priority.
  - For subtle cases, consider producing a confidence score instead of a single label.

- More Robust Tagging:
  - Current tag extractor returns the literal matched string. If you want canonical tags (e.g., normalize `mcg` -> `µg` or `mg/m2` -> `mg/m²`), post-process `tags` after extraction.
  - If values contain units as suffixes (e.g., `"12 mg/kg"`), consider running `strip_units_and_tags` on sample values, not only headers.

- Better Name Matching:
  - `family_boost` currently uses substrings. To avoid accidental matches (e.g., `"age-old"`), switch to regex matching with word boundaries or a tokenized approach.

- Internationalization:
  - If headers are in other languages, add language-specific tokens to `TIME_HINTS`, `AGE_HINTS`, and `UNIT_PATTERNS`.

---

## Implementation Notes & Limitations

- The rules are intentionally heuristic and tuned for clinical/biomedical datasets (hence many clinical dosing tokens).
- The code is conservative — it prefers to return `"unknown"` when ambiguous rather than risk a misclassification.
- Some tokens (e.g., `years`) are detected as `time` rather than `age` — the AGE_HINTS pattern focuses on presence of the word `age` or `age_at` phrases. Depending on downstream needs you may want to treat `years` or `months` alongside `age` when the header contains both numeric and context suggesting age.
- The regex tokenization is case-insensitive, but tags preserve the matched substring; if you compare tags later, normalize case first.

---

## Where to Look in Code

- Core code: `src/utils/numeric_match_utils.py`
  - strip_units_and_tags
  - detect_numeric_semantic
  - family_boost
- Unit patterns and hint regexes are defined at the top of the module.

---