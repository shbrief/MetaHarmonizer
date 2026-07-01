"""Content-based hashing for user-uploaded corpus DataFrames.

Used to generate unique table/index name suffixes so that different
user-provided corpora are stored in separate SQLite tables and FAISS
indexes, preventing cross-contamination with official corpus data.
"""

import hashlib


def compute_corpus_hash(df, n: int = 8) -> str:
    """Return a short hex digest identifying the corpus content.

    Hashes sorted ``(clean_code, official_label)`` pairs so that
    row order and extra columns do not affect the result.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``clean_code`` and ``official_label`` columns
        (i.e. already normalised via ``_normalize_df``).
    n : int
        Number of hex characters to return (default 8 → 32-bit).

    Returns
    -------
    str
        Lowercase hex string, e.g. ``"a1b2c3d4"``.
    """
    pairs = sorted(
        zip(df["clean_code"].astype(str), df["official_label"].astype(str))
    )
    content = "\n".join(f"{code}\t{label}" for code, label in pairs)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:n]
