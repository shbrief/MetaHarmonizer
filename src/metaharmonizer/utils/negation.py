"""Lightweight negation detection for ontology-mapping queries."""

import re


_NEGATION_PREFIX_RE = re.compile(
    r"^\s*(?P<trigger>no|none|not|without)\b", re.IGNORECASE
)
_TREATMENT_NAIVE_SUFFIX_RE = re.compile(
    r"(?:^|[\s-])(?P<trigger>(?:treatment[\s-]*)?na[iï]ve)"
    r"\s*[)\]}.,;:]*\s*$",
    re.IGNORECASE,
)


def detect_negation(query: str, *, category: str | None = None) -> str | None:
    """Return the explicit negation marker in an ontology query, if present.

    Prefix markers are safe across ontology categories. The -naive suffix is
    limited to the treatment category because in other domains, terms such as
    naive T cell can describe an affirmative concept.
    """
    text = str(query)
    prefix_match = _NEGATION_PREFIX_RE.search(text)
    if prefix_match:
        return prefix_match.group("trigger").lower()

    if (category or "").lower() == "treatment":
        naive_match = _TREATMENT_NAIVE_SUFFIX_RE.search(text)
        if naive_match:
            return naive_match.group("trigger").lower()

    return None


def polarity_compatible(
    query: str,
    candidate: str,
    *,
    category: str | None = None,
) -> bool:
    """Return whether query and candidate have the same negation polarity."""
    query_is_negated = detect_negation(query, category=category) is not None
    candidate_is_negated = (
        detect_negation(candidate, category=category) is not None
    )
    return query_is_negated == candidate_is_negated
