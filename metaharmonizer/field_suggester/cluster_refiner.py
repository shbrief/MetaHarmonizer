"""Post-clustering refinement for FieldSuggester.

Detects and splits over-merged clusters by checking for internal
heterogeneity via silhouette analysis and optional 2-way splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import pdist

from metaharmonizer.CustomLogger.custom_logger import CustomLogger


@dataclass
class ClusterRefinerConfig:
    """Configuration for post-clustering refinement."""

    split_threshold: float = 0.15
    min_split_size: int = 2
    min_items_to_check: int = 4
    use_llm: bool = False


class ClusterRefiner:
    """Post-clustering refinement: detect and split over-merged clusters.

    For each cluster with more than ``min_items_to_check`` items:

    1. Compute intra-cluster pairwise cosine distances.
    2. Attempt a 2-way split via k-means (k=2).
    3. If the split produces two sub-clusters whose average intra-cluster
       similarity is significantly higher than the parent's, accept the split.
    4. Optionally verify with an LLM.

    Parameters
    ----------
    split_threshold : float
        Minimum improvement in average intra-cluster similarity required
        to accept a split (default: 0.15).
    min_split_size : int
        Minimum number of items in each sub-cluster after splitting.
    min_items_to_check : int
        Only attempt to split clusters with at least this many items.
    use_llm : bool
        If True, use LLM verification for split decisions.
    llm_client : callable, optional
        ``f(prompt: str) -> str`` for LLM verification.
    """

    def __init__(
        self,
        split_threshold: float = 0.15,
        min_split_size: int = 2,
        min_items_to_check: int = 4,
        use_llm: bool = False,
        llm_client: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._split_threshold = split_threshold
        self._min_split_size = min_split_size
        self._min_items_to_check = min_items_to_check
        self._use_llm = use_llm
        self._llm_client = llm_client

    def refine_cluster_map(
        self,
        cluster_map: Dict[str, int],
        all_columns: List[str],
        embeddings: np.ndarray,
    ) -> Dict[str, int]:
        """Refine a {column: cluster_label} map by splitting over-merged clusters.

        Parameters
        ----------
        cluster_map : dict
            ``{column_name: cluster_label}`` from initial clustering.
        all_columns : list of str
            Ordered column list matching ``embeddings`` rows.
        embeddings : np.ndarray
            Embedding matrix of shape ``(n_columns, dim)``.

        Returns
        -------
        dict
            Refined ``{column_name: cluster_label}`` with potentially more clusters.
        """
        col_to_idx = {c: i for i, c in enumerate(all_columns)}
        log = CustomLogger().custlogger("INFO")

        # Group columns by cluster
        groups: Dict[int, List[str]] = {}
        for col, label in cluster_map.items():
            groups.setdefault(label, []).append(col)

        next_label = max(cluster_map.values()) + 1 if cluster_map else 0
        refined: Dict[str, int] = {}

        for label, members in groups.items():
            if len(members) < self._min_items_to_check:
                for col in members:
                    refined[col] = label
                continue

            idxs = [col_to_idx[c] for c in members if c in col_to_idx]
            if len(idxs) < self._min_items_to_check:
                for col in members:
                    refined[col] = label
                continue

            member_embs = embeddings[idxs]
            split_result = self._try_split(members, member_embs)

            if split_result is not None:
                group_a, group_b = split_result
                log.info(
                    "Split cluster %d (%d items) into two sub-clusters "
                    "(%d + %d items).",
                    label, len(members), len(group_a), len(group_b),
                )
                for col in group_a:
                    refined[col] = label
                for col in group_b:
                    refined[col] = next_label
                next_label += 1
            else:
                for col in members:
                    refined[col] = label

        return refined

    def _try_split(
        self,
        members: List[str],
        embeddings: np.ndarray,
    ) -> Optional[Tuple[List[str], List[str]]]:
        """Attempt a 2-way split of a cluster.

        Returns (group_a, group_b) if the split improves cohesion,
        or None if the cluster should remain intact.
        """
        from sklearn.cluster import KMeans

        n = len(members)
        if n < 2 * self._min_split_size:
            return None

        # Parent average pairwise cosine similarity
        parent_sim = self._avg_cosine_similarity(embeddings)

        # 2-means split
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        group_a_idx = [i for i in range(n) if labels[i] == 0]
        group_b_idx = [i for i in range(n) if labels[i] == 1]

        if (
            len(group_a_idx) < self._min_split_size
            or len(group_b_idx) < self._min_split_size
        ):
            return None

        # Sub-cluster similarities
        sim_a = self._avg_cosine_similarity(embeddings[group_a_idx])
        sim_b = self._avg_cosine_similarity(embeddings[group_b_idx])
        avg_child_sim = (sim_a * len(group_a_idx) + sim_b * len(group_b_idx)) / n

        improvement = avg_child_sim - parent_sim
        if improvement < self._split_threshold:
            return None

        group_a = [members[i] for i in group_a_idx]
        group_b = [members[i] for i in group_b_idx]

        # Optional LLM verification
        if self._use_llm and self._llm_client is not None:
            if not self._llm_should_split(group_a, group_b):
                return None

        return group_a, group_b

    @staticmethod
    def _avg_cosine_similarity(embeddings: np.ndarray) -> float:
        """Average pairwise cosine similarity for a set of embeddings."""
        if embeddings.shape[0] <= 1:
            return 1.0
        dists = pdist(embeddings, metric="cosine")
        sims = 1.0 - dists
        return float(np.mean(sims))

    def _llm_should_split(
        self, group_a: List[str], group_b: List[str]
    ) -> bool:
        """Ask LLM whether two groups should be separate fields."""
        if self._llm_client is None:
            return True

        items_a = ", ".join(group_a[:10])
        items_b = ", ".join(group_b[:10])
        prompt = (
            "You are an expert biomedical data curator. These two groups of "
            "clinical metadata column names were clustered together but may "
            "represent different types of information.\n\n"
            f"Group A: {items_a}\n"
            f"Group B: {items_b}\n\n"
            "Should these be two SEPARATE harmonized fields, or do they belong "
            "to the SAME field? Answer ONLY 'separate' or 'same'."
        )
        try:
            response = self._llm_client(prompt).strip().lower()
            return "separate" in response
        except Exception:
            return True  # Default to splitting on failure
