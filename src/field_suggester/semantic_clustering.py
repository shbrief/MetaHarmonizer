"""
Semantic Clustering module for FieldSuggester.

Combines hierarchical clustering of embedding vectors with a configurable
Semantic Consistency Score (SCS) evaluation.  SCS can be computed via:

- **Embedding-based** (default) — average pairwise cosine similarity.
  No API keys or LLM calls required.
- **LLM-based** — queries an LLM to assess whether sampled items belong
  to the same semantic category.  Opt-in.

Algorithm
---------
1. Hierarchical clustering of embeddings (cosine distance, average linkage).
2. Breadth-first traversal of the resulting cluster tree.
3. At each node, compute SCS via the chosen mode.
4. If SCS >= threshold → accept node as a semantic cluster; stop exploring subtree.
5. If SCS < threshold → continue BFS to children.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.CustomLogger.custom_logger import CustomLogger

if TYPE_CHECKING:
    from src.utils.embedding_store import EmbeddingStore


# ======================================================================
# Enums & Config
# ======================================================================


class SCSMode(str, Enum):
    """Evaluation mode for Semantic Consistency Score."""

    LLM = "llm"
    EMBEDDING = "embedding"


@dataclass
class SemanticCluster:
    """Represents a semantically coherent cluster of metadata items."""

    items: List[str]
    indices: List[int]
    scs: float
    depth: int


@dataclass
class SemanticClusteringConfig:
    """Configuration for semantic clustering.

    Parameters
    ----------
    scs_mode : SCSMode
        How to compute the Semantic Consistency Score.
        ``EMBEDDING`` (default) uses average pairwise cosine similarity.
        ``LLM`` queries an LLM for richer semantic evaluation.
    scs_threshold : float
        Minimum SCS to accept a node as a coherent cluster.
    sample_size : int
        Maximum items sampled for SCS evaluation (LLM mode).
    pca_components : int
        PCA dimensions for stratified sampling (LLM mode).
    kmeans_n_clusters : int
        K-means clusters for stratified sampling (LLM mode).
    min_cluster_size : int
        Nodes with this many items or fewer are accepted without evaluation.
    llm_model : str
        LLM model identifier (only used when ``scs_mode == LLM``).
    llm_temperature : float
        LLM temperature (only used when ``scs_mode == LLM``).
    embedding_scs_metric : str
        Metric for embedding-based SCS.  ``"cosine"`` (average pairwise)
        or ``"centroid_cosine"`` (faster for large clusters).
    """

    scs_mode: SCSMode = SCSMode.EMBEDDING
    scs_threshold: float = 0.9
    sample_size: int = 100
    pca_components: int = 10
    kmeans_n_clusters: int = 10
    min_cluster_size: int = 2
    llm_model: str = "gpt-4-turbo"
    llm_temperature: float = 0.0
    embedding_scs_metric: str = "cosine"

    def __post_init__(self):
        if isinstance(self.scs_mode, str) and not isinstance(self.scs_mode, SCSMode):
            self.scs_mode = SCSMode(self.scs_mode)


# ======================================================================
# SemanticClusteringEngine
# ======================================================================


class SemanticClusteringEngine:
    """Hierarchical clustering with SCS-based adaptive stopping.

    Parameters
    ----------
    config : SemanticClusteringConfig, optional
        Clustering and SCS parameters.  Defaults to embedding-based SCS.
    llm_client : callable, optional
        ``f(prompt: str) -> str``.  Custom LLM backend for SCS evaluation
        in ``SCSMode.LLM``.  If *None*, falls back to ``openai.OpenAI()``.
    """

    def __init__(
        self,
        config: Optional[SemanticClusteringConfig] = None,
        llm_client: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.config = config or SemanticClusteringConfig()
        self._llm_client = llm_client
        self._api_call_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cluster(
        self,
        items: List[str],
        embeddings: np.ndarray,
    ) -> List[SemanticCluster]:
        """Run semantic clustering on a set of metadata items.

        Parameters
        ----------
        items : list of str
            Metadata item labels (column names, free-text terms, etc.).
        embeddings : np.ndarray
            Embedding matrix of shape ``(n_items, embedding_dim)``.

        Returns
        -------
        list of SemanticCluster
            Semantically coherent clusters.
        """
        if len(items) != embeddings.shape[0]:
            raise ValueError(
                f"Length mismatch: {len(items)} items vs "
                f"{embeddings.shape[0]} embeddings."
            )

        if len(items) <= self.config.min_cluster_size:
            return [
                SemanticCluster(
                    items=list(items),
                    indices=list(range(len(items))),
                    scs=1.0,
                    depth=0,
                )
            ]

        log = CustomLogger().custlogger("INFO")
        log.info(
            "Building hierarchical cluster tree for %d items "
            "(cosine distance, average linkage). SCS mode: %s",
            len(items),
            getattr(self.config.scs_mode, "value", self.config.scs_mode),
        )

        dist_matrix = pdist(embeddings, metric="cosine")
        # Clamp NaN / negative distances that can arise from floating-point
        # arithmetic on near-identical vectors.
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 2.0)

        linkage_matrix = linkage(dist_matrix, method="average")
        root: ClusterNode = to_tree(linkage_matrix)

        clusters = self._bfs_traverse(root, items, embeddings)

        log.info(
            "Semantic clustering complete: %d clusters found (%d LLM calls).",
            len(clusters),
            self._api_call_count,
        )
        return clusters

    def cluster_from_store(
        self,
        items: List[str],
        store: "EmbeddingStore",
    ) -> List[SemanticCluster]:
        """Convenience: retrieve embeddings from a shared store and cluster.

        Parameters
        ----------
        items : list of str
            Metadata item labels.
        store : EmbeddingStore
            Shared embedding store (embeddings may already be cached).

        Returns
        -------
        list of SemanticCluster
        """
        embeddings = store.embed(items, source="column_name")
        return self.cluster(items, embeddings)

    # ------------------------------------------------------------------
    # BFS Traversal
    # ------------------------------------------------------------------

    def _bfs_traverse(
        self,
        root: ClusterNode,
        items: List[str],
        embeddings: np.ndarray,
    ) -> List[SemanticCluster]:
        """Breadth-first traversal with SCS-based stopping."""
        result_clusters: List[SemanticCluster] = []
        queue: deque[Tuple[ClusterNode, int]] = deque()
        queue.append((root, 0))

        while queue:
            node, depth = queue.popleft()
            indices = self._get_leaf_indices(node)

            # Accept leaf / tiny nodes directly
            if len(indices) <= self.config.min_cluster_size:
                result_clusters.append(
                    SemanticCluster(
                        items=[items[i] for i in indices],
                        indices=indices,
                        scs=1.0,
                        depth=depth,
                    )
                )
                continue

            node_items = [items[i] for i in indices]
            node_embeddings = embeddings[indices]

            scs = self._compute_scs(node_items, node_embeddings)

            log = CustomLogger().custlogger("DEBUG")
            log.debug(
                "Depth %d, node size %d: SCS = %.3f", depth, len(indices), scs
            )

            if scs >= self.config.scs_threshold:
                result_clusters.append(
                    SemanticCluster(
                        items=node_items,
                        indices=indices,
                        scs=scs,
                        depth=depth,
                    )
                )
            else:
                left = node.get_left()
                right = node.get_right()
                if left is not None:
                    queue.append((left, depth + 1))
                if right is not None:
                    queue.append((right, depth + 1))

        return result_clusters

    # ------------------------------------------------------------------
    # Semantic Consistency Score — dispatcher
    # ------------------------------------------------------------------

    def _compute_scs(
        self,
        items: List[str],
        embeddings: np.ndarray,
    ) -> float:
        """Compute the Semantic Consistency Score.

        Dispatches to embedding-based or LLM-based evaluation depending
        on ``config.scs_mode``.
        """
        if self.config.scs_mode == SCSMode.EMBEDDING:
            return self._compute_scs_embedding(embeddings)

        # SCSMode.LLM
        if len(items) <= self.config.sample_size:
            sampled_items = items
        else:
            sampled_items = self._stratified_sample(items, embeddings)
        return self._query_llm_for_scs(sampled_items)

    # ------------------------------------------------------------------
    # Embedding-based SCS (no LLM required)
    # ------------------------------------------------------------------

    def _compute_scs_embedding(self, embeddings: np.ndarray) -> float:
        """Compute SCS from embedding geometry alone.

        Supported metrics (``config.embedding_scs_metric``):

        ``"cosine"``
            Average pairwise cosine similarity, rescaled from [-1, 1] to
            [0, 1] via ``(mean_sim + 1) / 2``.

        ``"centroid_cosine"``
            Average cosine similarity of each vector to the cluster
            centroid.  More efficient for large clusters.
        """
        if embeddings.shape[0] <= 1:
            return 1.0

        metric = self.config.embedding_scs_metric

        if metric == "cosine":
            return self._scs_avg_pairwise_cosine(embeddings)
        elif metric == "centroid_cosine":
            return self._scs_centroid_cosine(embeddings)
        else:
            log = CustomLogger().custlogger("WARNING")
            log.warning(
                "Unknown embedding_scs_metric %r; falling back to 'cosine'.",
                metric,
            )
            return self._scs_avg_pairwise_cosine(embeddings)

    @staticmethod
    def _scs_avg_pairwise_cosine(embeddings: np.ndarray) -> float:
        """Average pairwise cosine similarity, rescaled to [0, 1]."""
        cosine_dists = pdist(embeddings, metric="cosine")
        cosine_sims = 1.0 - cosine_dists
        mean_sim = float(np.mean(cosine_sims))
        return (mean_sim + 1.0) / 2.0

    @staticmethod
    def _scs_centroid_cosine(embeddings: np.ndarray) -> float:
        """Average cosine similarity to centroid, in [0, 1]."""
        centroid = embeddings.mean(axis=0, keepdims=True)
        norms_e = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms_c = np.linalg.norm(centroid, axis=1, keepdims=True)
        norms_e = np.where(norms_e == 0, 1e-10, norms_e)
        norms_c = np.where(norms_c == 0, 1e-10, norms_c)
        cosine_sims = (embeddings / norms_e) @ (centroid / norms_c).T
        mean_sim = float(np.mean(cosine_sims))
        return (mean_sim + 1.0) / 2.0

    # ------------------------------------------------------------------
    # Stratified Sampling (for LLM mode)
    # ------------------------------------------------------------------

    def _stratified_sample(
        self,
        items: List[str],
        embeddings: np.ndarray,
    ) -> List[str]:
        """PCA → K-means stratified sampling for a representative subset."""
        n_items = len(items)
        n_components = min(
            self.config.pca_components, n_items, embeddings.shape[1]
        )
        n_clusters = min(self.config.kmeans_n_clusters, n_items)

        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(reduced)

        samples_per_cluster = max(1, self.config.sample_size // n_clusters)
        sampled_indices: List[int] = []
        rng = np.random.default_rng(42)

        for cid in range(n_clusters):
            members = np.where(cluster_labels == cid)[0]
            if len(members) == 0:
                continue
            n_take = min(samples_per_cluster, len(members))
            chosen = rng.choice(members, size=n_take, replace=False)
            sampled_indices.extend(chosen.tolist())

        if len(sampled_indices) > self.config.sample_size:
            sampled_indices = rng.choice(
                sampled_indices,
                size=self.config.sample_size,
                replace=False,
            ).tolist()

        return [items[i] for i in sampled_indices]

    # ------------------------------------------------------------------
    # LLM-based SCS
    # ------------------------------------------------------------------

    def _query_llm_for_scs(self, items: List[str]) -> float:
        """Query an LLM to assess semantic consistency.  Returns 0.0–1.0."""
        self._api_call_count += 1
        prompt = self._build_scs_prompt(items)

        try:
            response = self._call_llm(prompt)
            return self._parse_scs_response(response)
        except Exception as exc:
            log = CustomLogger().custlogger("WARNING")
            log.warning("LLM SCS query failed: %s. Returning 0.0.", exc)
            return 0.0

    @staticmethod
    def _build_scs_prompt(items: List[str]) -> str:
        """Build the SCS evaluation prompt."""
        items_str = "\n".join(f"- {item}" for item in items)
        return (
            "You are an expert biomedical data curator. You are given a list of "
            "metadata field names (column headers) from clinical datasets.\n\n"
            "Assess whether ALL of the following items belong to the same "
            "semantic category (i.e., they describe the same type of clinical "
            "information, even if worded differently).\n\n"
            f"Items:\n{items_str}\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"score": <float between 0.0 and 1.0>, '
            '"category": "<brief category label if score > 0.7, else null>", '
            '"reasoning": "<one sentence explanation>"}\n\n'
            "A score of 1.0 means all items clearly belong to the same semantic "
            "category. A score of 0.0 means they are completely unrelated."
        )

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM.  Uses the injected client or falls back to OpenAI."""
        if self._llm_client is not None:
            return self._llm_client(prompt)

        try:
            import openai

            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biomedical metadata expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except ImportError:
            raise RuntimeError(
                "openai package not installed. Install it or provide a custom "
                "llm_client to SemanticClusteringEngine."
            )

    @staticmethod
    def _parse_scs_response(response: str) -> float:
        """Parse the SCS score from the LLM JSON response."""
        try:
            data = json.loads(response)
            return float(data["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)

        log = CustomLogger().custlogger("WARNING")
        log.warning("Could not parse SCS from response: %r", response)
        return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_leaf_indices(node: ClusterNode) -> List[int]:
        """Get all leaf indices under a ClusterNode."""
        if node.is_leaf():
            return [node.id]
        return list(node.pre_order(lambda x: x.id))
