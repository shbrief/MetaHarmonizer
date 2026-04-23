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
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.spatial.distance import pdist, squareform
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
    hybrid_similarity: Optional["HybridSimilarityConfig"] = None
    domain_thresholds: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if isinstance(self.scs_mode, str) and not isinstance(self.scs_mode, SCSMode):
            self.scs_mode = SCSMode(self.scs_mode)


@dataclass
class HybridSimilarityConfig:
    """Weights for combining multiple similarity signals into a distance matrix."""

    embedding_weight: float = 0.6
    token_jaccard_weight: float = 0.2
    edit_distance_weight: float = 0.1
    prefix_suffix_weight: float = 0.1


class HybridSimilarity:
    """Combine embedding cosine distance with lexical similarity signals.

    Produces a condensed distance matrix suitable for
    ``scipy.cluster.hierarchy.linkage``.
    """

    def __init__(self, config: Optional[HybridSimilarityConfig] = None) -> None:
        self.config = config or HybridSimilarityConfig()

    def compute_distance_matrix(
        self, items: List[str], embeddings: np.ndarray
    ) -> np.ndarray:
        """Return a condensed distance matrix combining 4 signals.

        Parameters
        ----------
        items : list of str
            Item labels (column names).
        embeddings : np.ndarray
            Embedding matrix of shape ``(n_items, dim)``.

        Returns
        -------
        np.ndarray
            Condensed distance vector (for ``scipy.cluster.hierarchy.linkage``).
        """
        n = len(items)
        cfg = self.config

        # 1. Embedding cosine distance
        emb_dist = pdist(embeddings, metric="cosine")
        emb_dist = np.nan_to_num(emb_dist, nan=0.0)
        emb_dist = np.clip(emb_dist, 0.0, 2.0)

        # Normalise to [0, 1] (cosine distance is already in [0, 2])
        emb_dist_norm = emb_dist / 2.0

        # 2. Token Jaccard distance
        token_sets = [self._tokenise(item) for item in items]
        jac_dist = np.zeros(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                jac_dist[idx] = 1.0 - self._token_jaccard(
                    token_sets[i], token_sets[j]
                )
                idx += 1

        # 3. Normalised edit distance
        edit_dist = np.zeros(n * (n - 1) // 2)
        normed_items = [item.lower().replace("_", "").replace(" ", "") for item in items]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edit_dist[idx] = self._normalized_edit_distance(
                    normed_items[i], normed_items[j]
                )
                idx += 1

        # 4. Prefix/suffix distance
        ps_dist = np.zeros(n * (n - 1) // 2)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                ps_dist[idx] = 1.0 - self._prefix_suffix_score(
                    token_sets[i], token_sets[j]
                )
                idx += 1

        # Weighted combination
        combined = (
            cfg.embedding_weight * emb_dist_norm
            + cfg.token_jaccard_weight * jac_dist
            + cfg.edit_distance_weight * edit_dist
            + cfg.prefix_suffix_weight * ps_dist
        )
        return combined

    @staticmethod
    def _tokenise(text: str) -> Set[str]:
        """Lowercase tokenise a column name."""
        cleaned = text.lower().replace("_", " ").replace("-", " ").replace(".", " ")
        return {tok.strip() for tok in cleaned.split() if tok.strip()}

    @staticmethod
    def _token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    @staticmethod
    def _normalized_edit_distance(a: str, b: str) -> float:
        """Normalised Levenshtein distance in [0, 1]."""
        if a == b:
            return 0.0
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 0.0
        # Simple DP Levenshtein
        m, n = len(a), len(b)
        prev = list(range(n + 1))
        for i in range(1, m + 1):
            curr = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[n] / max_len

    @staticmethod
    def _prefix_suffix_score(tokens_a: Set[str], tokens_b: Set[str]) -> float:
        """Score based on shared prefix/suffix tokens.

        Returns a value in [0, 1] reflecting what fraction of the smaller
        set's tokens appear in the larger set as prefix or suffix tokens.
        """
        if not tokens_a or not tokens_b:
            return 0.0
        shared = tokens_a & tokens_b
        smaller = min(len(tokens_a), len(tokens_b))
        return len(shared) / smaller if smaller > 0 else 0.0


# Domain keyword sets for adaptive thresholds
_DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "staging": {"stage", "staging", "tnm", "ajcc", "path", "clin", "clinical"},
    "site": {"site", "tissue", "location", "anatomic", "biopsy", "primary"},
    "status": {"status", "vital", "alive", "dead", "deceased"},
    "treatment": {"treatment", "therapy", "chemo", "radiation", "adjuvant", "neo"},
    "age": {"age", "years", "birth", "dob"},
    "smoking": {"smoking", "smoker", "tobacco", "pack"},
}


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
        use_hybrid = self.config.hybrid_similarity is not None
        log.info(
            "Building hierarchical cluster tree for %d items "
            "(%s distance, average linkage). SCS mode: %s",
            len(items),
            "hybrid" if use_hybrid else "cosine",
            getattr(self.config.scs_mode, "value", self.config.scs_mode),
        )

        if use_hybrid:
            hybrid = HybridSimilarity(self.config.hybrid_similarity)
            dist_matrix = hybrid.compute_distance_matrix(items, embeddings)
        else:
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

            threshold = self._resolve_threshold(node_items)

            if scs >= threshold:
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
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. "
                "Install with: pip install metaharmonizer[llm-openai], "
                "or provide a custom llm_client to SemanticClusteringEngine."
            ) from e

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

    def _resolve_threshold(self, node_items: List[str]) -> float:
        """Return the SCS threshold for a node, respecting domain overrides.

        If ``config.domain_thresholds`` is set and >50% of the node's items
        contain tokens from a recognised domain keyword set, the domain-specific
        threshold is used.  Otherwise falls back to ``config.scs_threshold``.
        """
        dt = self.config.domain_thresholds
        if not dt:
            return self.config.scs_threshold

        # Tokenise all items in the node
        all_tokens: Set[str] = set()
        for item in node_items:
            cleaned = item.lower().replace("_", " ").replace("-", " ")
            all_tokens.update(tok.strip() for tok in cleaned.split() if tok.strip())

        best_domain: Optional[str] = None
        best_overlap = 0
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            if domain not in dt:
                continue
            overlap = len(all_tokens & keywords)
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = domain

        if best_domain is not None and best_overlap >= max(1, len(node_items) * 0.3):
            return dt[best_domain]

        return self.config.scs_threshold

    @staticmethod
    def _get_leaf_indices(node: ClusterNode) -> List[int]:
        """Get all leaf indices under a ClusterNode."""
        if node.is_leaf():
            return [node.id]
        return list(node.pre_order(lambda x: x.id))
