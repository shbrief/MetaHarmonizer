import re

UNIT_PATTERNS = [
    r"\bmg(?:/m2|/m²)?\b", r"\bµg\b", r"\bmcg\b", r"\bIU\b", r"\bU/mL\b",
    r"\bAUC\d+\b", r"\bGy\b", r"\b%\b", r"\bmmHg\b", r"\bq\d+[wd]\b",
    r"\bqd\b", r"\bbid\b", r"\btid\b", r"\bd\d+(?:[,-]\d+)*\b",
    r"\bx\d+\s*cycles?\b", r"\bIV\b", r"\bPO\b", r"\bSC\b", r"\bICD-10\b"
]
UNIT_REGEX = re.compile("|".join(UNIT_PATTERNS), flags=re.IGNORECASE)
AGE_HINTS = re.compile(r"\bage\b|\bage[_\s-]?at\b|\bdiagnosis[_\s-]?age\b",
                       re.I)
TIME_HINTS = re.compile(
    r"\b(date|day|days|month|months|year|years|start|end|duration|freq(uency)?|interval)\b",
    re.I)
DOSE_HINTS = re.compile(r"\b(dose|auc|gy|mg|m2|m²|regimen|cycle)\b", re.I)


def strip_units_and_tags(text: str):
    """
    Remove units and tags from the text.
    """
    tags = set()

    def _tagger(m):
        tags.add(m.group(0))
        return " "

    clean = UNIT_REGEX.sub(_tagger, text or "")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean, tags


def detect_numeric_semantic(header: str, tags: set) -> str:
    """
    Detect the semantic category of a numeric column based on its header and tags.
    """
    h = header or ""
    if any(re.search(r"(AUC\d+|Gy|mg\b|mg/m2|mg/m²)", t, re.I)
           for t in tags) or DOSE_HINTS.search(h):
        return "dose"
    if AGE_HINTS.search(h):
        return "age"
    if TIME_HINTS.search(h):
        return "time"
    return "unknown"


def family_boost(field_name: str, family: str) -> float:
    """
    Assign a boost score based on the field name and its semantic family.
    """
    fname = (field_name or "").lower()
    if family == "dose":
        if "treatment_dose" in fname or "dose" in fname:
            return 0.15
        if "treatment_number" in fname or "cycle" in fname or "auc" in fname or "unit" in fname:
            return 0.10
    if family == "age":
        if "age" in fname:
            return 0.15
        if "age_group" in fname:
            return 0.10
    if family == "time":
        if any(k in fname for k in
               ["date", "start", "end", "duration", "frequency", "time"]):
            return 0.15
        if "unit" in fname and any(
                k in fname for k in ["duration", "frequency", "start", "end"]):
            return 0.10
    return 0.0
