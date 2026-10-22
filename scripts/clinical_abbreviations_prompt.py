"""Prompt used to generate the closed clinical abbreviation dictionary (Issue #87)."""

PROMPT = """\
You are a biomedical data curator.

<context>
We map clinical dataset column headers (raw field names) onto a standardized
schema. Many raw headers use domain abbreviations — e.g. AGE_AT_DX, OS_MONTHS,
BIOPSY_SITE — which stop exact / fuzzy / embedding matching from recognizing the
field. To fix this cheaply, we expand abbreviations to their full form before
matching.
</context>

<task>
Produce a CLOSED dictionary of clinical / oncology header abbreviations mapped to
their full form, so a single header token can be expanded before matching
(e.g. `dx` -> `diagnosis`, `os` -> `overall survival`).
</task>

<rules>
- Token-level only: one lowercase header token maps to its full form. The
  expansion may be multiple words.
- Include only UNAMBIGUOUS abbreviations that commonly appear in clinical /
  oncology column headers: diagnosis, treatment, staging, survival endpoints,
  dates / time units, and similar.
- EXCLUDE single letters and any token that collides with a common English word
  or carries an unrelated meaning — a wrong expansion is worse than none.
- EXCLUDE any abbreviation with more than one common clinical meaning
  (e.g. `ca` = calcium or cancer; `pt` = patient or prothrombin time) —
  ambiguity makes expansion unsafe.
- Prefer widely recognized abbreviations over rare or institution-specific ones.
- No duplicate abbreviation keys.
</rules>

<output_format>
Save the output as a file named `clinical_abbreviations_<your-model-id>.csv`
(replace <your-model-id> with your own model identifier). The file must contain
CSV only — no prose, no code fences — with exactly two columns and this header:
abbrev,expansion
One lowercase mapping per line.
</output_format>

<examples>
abbrev,expansion
dx,diagnosis
os,overall survival
bx,biopsy
neoadj,neoadjuvant
</examples>
"""
