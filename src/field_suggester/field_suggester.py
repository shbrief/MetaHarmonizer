"""FieldSuggester – Hybrid NER + Embedding Clustering tool for suggesting new harmonized fields.

Given a set of unmapped columns (columns that could not be mapped to any
existing target field), this module:

1. **NER pass** – extracts biomedical entities from column names and sample values
   using scispaCy, producing coarse semantic groups.
2. **Embedding clustering** – encodes column names (+ optional descriptions/values)
   with a sentence transformer and clusters them with Agglomerative Clustering.
3. **Signal merging** – reconciles NER groups and embedding clusters, merging
   clusters that share the same NER entity type.
4. **Name generation** – produces readable suggested field names from NER entities
   and common tokens.
5. **Confidence scoring** – scores each suggestion based on NER/embedding agreement
   and intra-cluster cohesion.
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

# Default sentence-transformer model (env-var override supported)
FIELD_MODEL = os.environ.get("FIELD_MODEL", "all-MiniLM-L6-v2")


# ======================================================================
# Constants
# ======================================================================

_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "in", "at", "for", "and", "or", "to",
    "is", "by", "on", "with", "from", "as", "it", "its", "this",
    "that", "was", "are", "be", "has", "had", "have", "not", "but",
    "no", "yes", "if", "so", "do", "did", "my", "your", "their",
    "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "type", "value", "code", "id", "name", "num", "number", "date",
}

# Default scispaCy model for biomedical NER
_DEFAULT_NER_MODEL = "en_ner_bc5cdr_md"

# Clustering defaults
_DEFAULT_DISTANCE_THRESHOLD = 0.45
_DEFAULT_MIN_CLUSTER_SIZE = 2

# Value sampling
_DEFAULT_VALUE_SAMPLE_SIZE = 50

# Confidence weights
_W_EMBEDDING_AGREEMENT = 0.35
_W_NER_AGREEMENT = 0.30
_W_COHESION = 0.20
_W_SIZE = 0.15


# ======================================================================
# FieldSuggester
# ======================================================================


class FieldSuggester:
    """Hybrid NER + Embedding Clustering approach for suggesting new harmonized fields.

    Strategy
    --------
    1. NER pass  – tag each column name (and optionally sample values) with
       biomedical entity types to form coarse semantic groups.
    2. Embedding clustering – encode column names with a sentence transformer
       and cluster via Agglomerative Clustering for fine-grained grouping.
    3. Merge signals – reconcile NER groups and embedding clusters.
    4. Name generation – produce human-readable field names from NER entities
       and common column-name tokens.
    5. Confidence scoring – score each suggestion based on layer agreement
       and intra-cluster embedding cohesion.

    Parameters
    ----------
    embedding_model : str
        Sentence-transformer model id for column-name embeddings.
    ner_model : str
        scispaCy model name for biomedical NER.  Falls back to
        embedding-only mode if the model is not installed.
    distance_threshold : float
        Agglomerative clustering distance threshold (cosine).
        Smaller → more clusters; larger → fewer, bigger clusters.
    min_cluster_size : int
        Minimum number of columns to form a valid suggestion.
    value_sample_size : int
        Max unique values to sample per column for NER enrichment.
    """

    def __init__(
        self,
        embedding_model: str = FIELD_MODEL,
        ner_model: str = _DEFAULT_NER_MODEL,
        distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD,
        min_cluster_size: int = _DEFAULT_MIN_CLUSTER_SIZE,
        value_sample_size: int = _DEFAULT_VALUE_SAMPLE_SIZE,
    ) -> None:
        self.encoder = SentenceTransformer(embedding_model)
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.value_sample_size = value_sample_size

        # Attempt to load scispaCy NER model
        self._nlp = None
        self._ner_available = False
        try:
            import spacy

            self._nlp = spacy.load(ner_model)
            self._ner_available = True
        except Exception:
            pass  # Embedding-only mode

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest(
        self,
        unmapped_columns: List[str],
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, dict]:
        """Suggest new harmonized fields for unmapped columns.

        Parameters
        ----------
        unmapped_columns : list of str
            Column names that were not mapped to existing target fields.
        df : pd.DataFrame or None
            Original data table.  If provided, sample values are used to
            enrich NER tagging and to perform value-based validation.

        Returns
        -------
        dict
            ``{suggested_field_name: {
                "source_columns": [str, ...],
                "ner_entities": [str, ...],
                "confidence": float,
                "sample_values": {col: [val, ...], ...}   # if df provided
            }}``
            Sorted by confidence descending.
        """
        if not unmapped_columns:
            return {}

        # De-duplicate while preserving order
        seen: Set[str] = set()
        columns: List[str] = []
        for c in unmapped_columns:
            if c not in seen:
                columns.append(c)
                seen.add(c)

        # --- Step 1: NER tagging (coarse groups) ---
        ner_tags = self._ner_tag(columns, df)

        # --- Step 2: Embedding clustering (fine groups) ---
        clusters, embeddings = self._embed_and_cluster(columns)

        # --- Step 3: Merge NER + clustering signals ---
        merged_groups = self._merge_signals(columns, ner_tags, clusters)

        # --- Step 4: Filter by min cluster size ---
        merged_groups = {
            gid: members
            for gid, members in merged_groups.items()
            if len(members) >= self.min_cluster_size
        }

        # --- Step 5: Generate suggestions ---
        suggestions: Dict[str, dict] = {}
        for group_id, members in merged_groups.items():
            name = self._generate_name(members, ner_tags)
            confidence = self._compute_confidence(
                members, ner_tags, clusters, embeddings, columns
            )

            entry: dict = {
                "source_columns": sorted(members),
                "ner_entities": sorted(
                    {ent for col in members for ent in ner_tags.get(col, [])}
                ),
                "confidence": round(confidence, 4),
            }

            # Attach sample values if df is available
            if df is not None:
                sample_values: Dict[str, list] = {}
                for col in sorted(members):
                    if col in df.columns:
                        vals = (
                            df[col]
                            .dropna()
                            .astype(str)
                            .unique()[: self.value_sample_size]
                        )
                        sample_values[col] = list(vals)
                entry["sample_values"] = sample_values

            # Handle duplicate names by appending a suffix
            final_name = name
            suffix = 2
            while final_name in suggestions:
                final_name = f"{name}_{suffix}"
                suffix += 1

            suggestions[final_name] = entry

        # Sort by confidence descending
        suggestions = dict(
            sorted(
                suggestions.items(),
                key=lambda kv: kv[1]["confidence"],
                reverse=True,
            )
        )

        return suggestions

    def suggest_from_mapping_results(
        self,
        df: pd.DataFrame,
        mapping_results: Dict[str, list],
    ) -> Dict[str, dict]:
        """Convenience wrapper: extract unmapped columns from mapping results
        and suggest new fields.

        Parameters
        ----------
        df : pd.DataFrame
            Original data table.
        mapping_results : dict
            Any ``{col: [(target, score), ...]}`` mapping output.
            Columns with empty prediction lists are treated as unmapped.

        Returns
        -------
        dict
            Same format as :meth:`suggest`.
        """
        unmapped = [
            col for col, preds in mapping_results.items() if not preds
        ]
        return self.suggest(unmapped, df)

    def suggest_to_df(
        self,
        unmapped_columns: List[str],
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return suggestions as a tidy DataFrame.

        Columns: ``suggested_field``, ``source_column``, ``ner_entities``,
        ``confidence``.
        """
        raw = self.suggest(unmapped_columns, df)
        rows = []
        for field_name, info in raw.items():
            for col in info["source_columns"]:
                rows.append(
                    {
                        "suggested_field": field_name,
                        "source_column": col,
                        "ner_entities": ", ".join(info["ner_entities"]),
                        "confidence": info["confidence"],
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Step 1: NER Tagging
    # ------------------------------------------------------------------

    def _ner_tag(
        self,
        columns: List[str],
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, List[str]]:
        """Extract biomedical entities from column names and optionally sample values.

        Returns
        -------
        dict
            ``{column_name: [entity_label, ...]}``
        """
        tags: Dict[str, List[str]] = {}
        for col in columns:
            entities: List[str] = []
            if self._ner_available and self._nlp is not None:
                # --- Column name NER ---
                text = self._col_to_text(col)
                doc = self._nlp(text)
                entities.extend(ent.label_ for ent in doc.ents)

                # --- Sample value NER (if df provided) ---
                if df is not None and col in df.columns:
                    sample_vals = (
                        df[col]
                        .dropna()
                        .astype(str)
                        .unique()[: self.value_sample_size]
                    )
                    for val in sample_vals:
                        vdoc = self._nlp(str(val))
                        entities.extend(ent.label_ for ent in vdoc.ents)

            tags[col] = entities
        return tags

    # ------------------------------------------------------------------
    # Step 2: Embedding Clustering
    # ------------------------------------------------------------------

    def _embed_and_cluster(
        self,
        columns: List[str],
    ) -> Tuple[Dict[str, int], np.ndarray]:
        """Encode column names and cluster by cosine similarity.

        Returns
        -------
        clusters : dict
            ``{column_name: cluster_label}``
        embeddings : np.ndarray
            Shape ``(n_columns, dim)``.
        """
        # Convert column names to readable text for better embeddings
        texts = [self._col_to_text(c) for c in columns]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        if len(columns) < 2:
            return {col: 0 for col in columns}, embeddings

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        cluster_map = {col: int(lbl) for col, lbl in zip(columns, labels)}
        return cluster_map, embeddings

    # ------------------------------------------------------------------
    # Step 3: Merge NER + Clustering Signals
    # ------------------------------------------------------------------

    def _merge_signals(
        self,
        columns: List[str],
        ner_tags: Dict[str, List[str]],
        clusters: Dict[str, int],
    ) -> Dict[int, Set[str]]:
        """Reconcile NER groups and embedding clusters.

        Merging logic
        -------------
        - Start with embedding clusters as base groups.
        - For each NER entity type that spans multiple embedding clusters,
          merge those clusters into one group.
        - Columns with no NER entities remain in their embedding cluster.
        """
        # Base groups from embedding clusters
        cluster_groups: Dict[int, Set[str]] = defaultdict(set)
        for col, label in clusters.items():
            cluster_groups[label].add(col)

        if not self._ner_available:
            return dict(cluster_groups)

        # Build entity-type → columns mapping
        entity_to_cols: Dict[str, Set[str]] = defaultdict(set)
        for col, entities in ner_tags.items():
            for ent in set(entities):  # unique entity labels per column
                entity_to_cols[ent].add(col)

        # Merge clusters that share the same NER entity type
        next_id = max(cluster_groups.keys()) + 1 if cluster_groups else 0

        for ent_type, ent_cols in entity_to_cols.items():
            if len(ent_cols) < 2:
                continue

            # Which current groups contain these columns?
            involved_gids: Set[int] = set()
            for gid, members in cluster_groups.items():
                if members & ent_cols:
                    involved_gids.add(gid)

            if len(involved_gids) <= 1:
                continue  # Already in the same group

            # Merge all involved groups
            combined: Set[str] = set()
            for gid in involved_gids:
                combined |= cluster_groups.pop(gid)

            cluster_groups[next_id] = combined
            next_id += 1

        return dict(cluster_groups)

    # ------------------------------------------------------------------
    # Step 4: Name Generation
    # ------------------------------------------------------------------

    def _generate_name(
        self,
        columns: Set[str],
        ner_tags: Dict[str, List[str]],
    ) -> str:
        """Generate a human-readable harmonized field name.

        Naming strategy (priority order):
        1. If NER entities are available, use the most common entity text
           combined with distinguishing column-name tokens.
        2. Otherwise, extract the most frequent meaningful tokens from
           column names.
        """
        # Tokenise all column names
        all_tokens: List[str] = []
        for col in columns:
            tokens = self._tokenise(col)
            all_tokens.extend(tokens)

        token_counts = Counter(all_tokens)

        # Remove stopwords and very short tokens
        meaningful = [
            (tok, cnt)
            for tok, cnt in token_counts.most_common()
            if tok not in _STOPWORDS and len(tok) > 1
        ]

        # --- NER-informed naming ---
        if self._ner_available:
            # Collect actual entity *texts* (not just labels) for richer names
            ner_texts: List[str] = []
            if self._nlp is not None:
                for col in columns:
                    text = self._col_to_text(col)
                    doc = self._nlp(text)
                    for ent in doc.ents:
                        ner_texts.append(ent.text.lower().strip())

            if ner_texts:
                # Most common entity text
                ent_counter = Counter(ner_texts)
                top_ent = ent_counter.most_common(1)[0][0]
                ent_tokens = self._tokenise(top_ent)

                # Add distinguishing tokens not already in entity
                extra = [
                    tok
                    for tok, _ in meaningful
                    if tok not in set(ent_tokens) and tok not in _STOPWORDS
                ][:2]

                name_tokens = ent_tokens + extra
                return "_".join(name_tokens[:4])

        # --- Fallback: token-frequency naming ---
        name_tokens = [tok for tok, _ in meaningful[:3]]
        return "_".join(name_tokens) if name_tokens else "unknown_field"

    # ------------------------------------------------------------------
    # Step 5: Confidence Scoring
    # ------------------------------------------------------------------

    def _compute_confidence(
        self,
        members: Set[str],
        ner_tags: Dict[str, List[str]],
        clusters: Dict[str, int],
        embeddings: np.ndarray,
        all_columns: List[str],
    ) -> float:
        """Compute a confidence score in [0, 1] for a suggestion.

        Components
        ----------
        - **Embedding agreement** : do all members share the same embedding cluster?
        - **NER agreement**       : do all members share the same NER entity types?
        - **Cohesion**            : average pairwise cosine similarity of member embeddings.
        - **Size**                : larger groups (up to a point) are more confident.
        """
        n = len(members)

        # -- Embedding agreement --
        cluster_ids = {clusters[c] for c in members if c in clusters}
        emb_agreement = 1.0 if len(cluster_ids) == 1 else 1.0 / len(cluster_ids)

        # -- NER agreement --
        if self._ner_available:
            ent_signatures = [
                tuple(sorted(set(ner_tags.get(c, [])))) for c in members
            ]
            non_empty = [s for s in ent_signatures if s]
            if non_empty:
                unique_sigs = set(non_empty)
                ner_agreement = 1.0 / len(unique_sigs) if unique_sigs else 0.5
            else:
                ner_agreement = 0.5  # No NER signal – neutral
        else:
            ner_agreement = 0.5  # NER unavailable – neutral

        # -- Intra-cluster cohesion --
        col_to_idx = {c: i for i, c in enumerate(all_columns)}
        member_indices = [col_to_idx[c] for c in members if c in col_to_idx]

        if len(member_indices) >= 2:
            member_embs = embeddings[member_indices]
            sim_matrix = cosine_similarity(member_embs)
            # Mean of upper triangle (exclude diagonal)
            triu_indices = np.triu_indices(len(member_indices), k=1)
            cohesion = float(np.mean(sim_matrix[triu_indices]))
        else:
            cohesion = 0.5

        # -- Size factor (diminishing returns, plateaus at ~8) --
        size_score = min(1.0, n / 8.0)

        # -- Weighted combination --
        confidence = (
            _W_EMBEDDING_AGREEMENT * emb_agreement
            + _W_NER_AGREEMENT * ner_agreement
            + _W_COHESION * cohesion
            + _W_SIZE * size_score
        )

        return min(1.0, max(0.0, confidence))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _col_to_text(col: str) -> str:
        """Convert a column name to readable text for NER / embedding."""
        return col.replace("_", " ").replace("-", " ").replace(".", " ").strip()

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase tokenisation of a column name or text."""
        cleaned = text.lower().replace("_", " ").replace("-", " ").replace(".", " ")
        return [tok.strip() for tok in cleaned.split() if tok.strip()]
