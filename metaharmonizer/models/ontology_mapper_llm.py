"""Stage 4: LLM-based query rewriting + FAISS re-search for low-confidence ontology matches.

Uses an LLM to interpret ambiguous queries (abbreviations, typos, informal names)
with clinical context, then re-searches the existing Stage 2 FAISS index with the
rewritten terms.
"""
import os
import json
import numpy as np
import pandas as pd
from metaharmonizer.utils.model_loader import load_method_model_dict
from metaharmonizer.CustomLogger.custom_logger import CustomLogger

_S4_LLM_MODEL = load_method_model_dict()["gemma-12b"]

logger = CustomLogger()


class OntoMapLLM:
    """LLM query rewriter for low-confidence ontology matches.

    Rewrites queries using Gemma 12B, then re-searches the S2 FAISS index
    with the rewritten terms to find better matches.
    """

    def __init__(
        self,
        category: str,
        s2_model,
        query_df: pd.DataFrame = None,
        query_col: str = None,
        topk: int = 5,
        model_key: str = "gemma-12b",
        max_retries: int = 5,
    ):
        self.category = category
        self.s2_model = s2_model
        self.query_df = query_df
        self._query_col = query_col
        self.topk = topk
        self.max_retries = max_retries
        self.logger = logger.custlogger(loglevel='INFO')

        # Resolve LLM model name
        model_dict = load_method_model_dict()
        self._llm_model_name = model_dict.get(model_key, _S4_LLM_MODEL)

        # Init Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "OntoMapLLM requires google-generativeai. "
                "Install with: pip install metaharmonizer[llm-gemini]"
            ) from e
        genai.configure(api_key=api_key)
        self._genai_model = genai.GenerativeModel(self._llm_model_name)

        self.logger.info(
            f"Initialized OntoMapLLM (model={self._llm_model_name}, "
            f"category={category})"
        )

    # ── Context extraction ────────────────────────────────────────────────

    # Max characters for context string to avoid blowing token limits
    _MAX_CONTEXT_CHARS = 500
    # Skip columns whose unique-value count exceeds this (likely IDs / free text)
    _HIGH_CARDINALITY_THRESHOLD = 50

    def _get_context_for_query(self, query: str) -> str:
        """Extract clinical context string from query_df for a given query.

        Skips high-cardinality and identifier-like columns, and caps total
        context length to avoid exceeding LLM token limits.
        """
        if self.query_df is None or self._query_col is None:
            return ""

        mask = self.query_df[self._query_col].astype(str) == str(query)
        rows = self.query_df.loc[mask]
        if rows.empty:
            return ""

        row = rows.iloc[0]
        parts = []
        total_len = 0
        for col in self.query_df.columns:
            if col == self._query_col:
                continue
            # Skip high-cardinality columns (likely IDs or free text)
            if self.query_df[col].nunique() > self._HIGH_CARDINALITY_THRESHOLD:
                continue
            val = str(row.get(col, "")).strip()
            if not val or val.lower() in ("nan", "none", ""):
                continue
            label = col.replace("_", " ").title()
            part = f"{label}: {val}"
            total_len += len(part)
            if total_len > self._MAX_CONTEXT_CHARS:
                break
            parts.append(part)
        return "; ".join(parts)

    # ── Rate limit handling ────────────────────────────────────────────────

    @staticmethod
    def _extract_retry_delay(exc: Exception) -> float | None:
        """Extract retry delay (seconds) from a 429 rate-limit error."""
        import re
        msg = str(exc)
        if "429" not in msg:
            return None
        # Match "retry in 35.714s" or "retry_delay { seconds: 35 }"
        m = re.search(r'retry in (\d+\.?\d*)', msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r'seconds:\s*(\d+)', msg)
        if m:
            return float(m.group(1))
        return 40.0  # safe default for Gemini free tier (30 rpm)

    # ── LLM prompt & parsing ─────────────────────────────────────────────

    def _build_prompt(self, query: str, context: str, k: int = 5) -> str:
        """Build the query-rewriting prompt."""
        context_block = ""
        if context:
            context_block = f"\nClinical context: {context}"

        return (
            f"You are a biomedical ontology expert.\n"
            f"Category: {self.category}\n"
            f"Query term: \"{query}\"{context_block}\n\n"
            f"The query may contain abbreviations, typos, or informal names.\n"
            f"Interpret the query using the clinical context and your biomedical knowledge.\n"
            f"Suggest exactly {k} standard ontology terms that best match the intended meaning, "
            f"ordered from most likely to least likely.\n\n"
            f"Return JSON array only:\n"
            f"[{{\"term\": \"Standard Ontology Term\", \"reasoning\": \"brief explanation\"}}]\n\n"
            f"CRITICAL: Return PURE JSON only - no markdown code blocks, no explanations outside JSON."
        )

    def _parse_response(self, text: str) -> list[str]:
        """Parse LLM response, return list of suggested terms."""
        raw = text.strip()
        # Strip markdown code fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        try:
            items = json.loads(raw.strip())
        except json.JSONDecodeError:
            return []

        if not isinstance(items, list):
            return []

        terms = []
        for item in items:
            if isinstance(item, dict) and "term" in item:
                term = str(item["term"]).strip()
                if term:
                    terms.append(term)
            elif isinstance(item, str):
                item = item.strip()
                if item:
                    terms.append(item)
        return terms

    def _rewrite_single(self, query: str, context: str, k: int = 5) -> list[str]:
        """Call LLM for one query with retry loop and exponential backoff."""
        import time

        prompt = self._build_prompt(query, context, k=k)
        last_exc = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._genai_model.generate_content(prompt)
                terms = self._parse_response(resp.text)
                if terms:
                    self.logger.info(
                        f"[S4] LLM rewrote \"{query}\" → {terms} (attempt {attempt})"
                    )
                    return terms
                self.logger.warning(
                    f"[S4] LLM returned empty for \"{query}\" (attempt {attempt})"
                )
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    # Use server-suggested delay for 429, else exponential backoff
                    delay = self._extract_retry_delay(e) or (2 ** attempt)
                    self.logger.warning(
                        f"[S4] LLM attempt {attempt} failed for \"{query}\": {e}, "
                        f"retrying in {delay}s"
                    )
                    time.sleep(delay)

        self.logger.warning(
            f"[S4] LLM failed after {self.max_retries} attempts for \"{query}\": {last_exc}"
        )
        return []

    # ── FAISS re-search ──────────────────────────────────────────────────

    def _re_search(self, terms: list[str], topk: int) -> tuple:
        """Embed LLM suggestions using S2's model, search S2's FAISS index.

        Returns:
            (D, I) — FAISS distance and index matrices, shape (len(terms), topk)
        """
        if not terms:
            return np.array([]).reshape(0, topk), np.array([]).reshape(0, topk)

        strategy = self.s2_model.om_strategy

        if strategy == 'lm':
            embs = self.s2_model.model.embed_documents(terms)
            mat = np.array(embs, dtype="float32")
        elif strategy == 'st':
            embs = self.s2_model.model.encode(terms, convert_to_tensor=False)
            mat = np.array(embs, dtype="float32")
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (norms + 1e-12)
        else:
            raise ValueError(f"Unsupported S2 strategy for re-search: {strategy}")

        idx = self.s2_model.vector_store.index
        D, I = idx.search(mat, topk)
        return D, I

    def _get_id_term_map(self) -> dict:
        """Preload id→term mapping from SQLite (cached)."""
        if not hasattr(self, '_id_term_map'):
            vs = self.s2_model.vector_store
            rows = vs.cursor.execute(
                f"SELECT id, term FROM {vs.table_name}"
            ).fetchall()
            self._id_term_map = {row[0]: row[1] for row in rows}
        return self._id_term_map

    def _faiss_indices_to_terms(self, I_row, D_row) -> list[tuple]:
        """Map FAISS index positions to (term, score) pairs."""
        vs = self.s2_model.vector_store
        n_ids = len(vs._ids)
        id_term = self._get_id_term_map()
        results = []
        for faiss_idx, score in zip(I_row, D_row):
            if faiss_idx == -1 or faiss_idx >= n_ids:
                continue
            db_id = vs._ids[faiss_idx]
            term = id_term.get(db_id)
            if term:
                results.append((term, float(score)))
        return results

    # ── Main entry point ─────────────────────────────────────────────────

    def _build_row(self, query: str, suggested_terms: list[str], k: int) -> dict:
        """FAISS re-search + build result row. Runs on main thread."""
        if not suggested_terms:
            return self._empty_row(query, k)

        # Each LLM suggestion → FAISS top-1 → collect as final matches
        # e.g. LLM suggests [a, b, c, d, e] → FAISS finds [a1, b1, c1, d1, e1]
        D, I = self._re_search(suggested_terms, 1)  # top-1 per suggestion

        top_matches = []
        seen = set()
        for i in range(len(suggested_terms)):
            matches = self._faiss_indices_to_terms(I[i], D[i])
            if matches:
                term, score = matches[0]
                if term not in seen:
                    seen.add(term)
                    top_matches.append((term, score))

        top_matches = top_matches[:k]

        row = {"original_value": query}
        for i in range(k):
            if i < len(top_matches):
                row[f"match{i+1}"] = top_matches[i][0]
                row[f"match{i+1}_score"] = f"{top_matches[i][1]:.4f}"
            else:
                row[f"match{i+1}"] = "N/A"
                row[f"match{i+1}_score"] = "0.0000"

        self.logger.info(
            f"[S4] \"{query}\" → match1={row.get('match1', 'N/A')} "
            f"({row.get('match1_score', 'N/A')})"
        )
        return row

    def get_match_results(
        self,
        queries: list[str],
        topk: int = 5,
        max_workers: int = 5,
    ) -> pd.DataFrame:
        """For each query: rewrite via LLM → re-search FAISS → build result DataFrame.

        LLM API calls are parallelized with ThreadPoolExecutor.
        FAISS re-search runs on the main thread (SQLite is not thread-safe).

        Returns DataFrame with: original_value, match1..matchN, match1_score..matchN_score.
        Does NOT include curated_ontology or match_level — those are eval concerns
        handled by the engine.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        k = topk or self.topk

        # Phase 1: parallel LLM rewriting
        def _rewrite(query):
            context = self._get_context_for_query(query)
            return self._rewrite_single(query, context, k=k)

        if len(queries) <= 1:
            rewrites = [_rewrite(q) for q in queries]
        else:
            self.logger.info(
                f"[S4] Rewriting {len(queries)} queries with {max_workers} workers"
            )
            rewrites = [None] * len(queries)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_rewrite, q): i
                    for i, q in enumerate(queries)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        rewrites[idx] = future.result()
                    except Exception as e:
                        self.logger.warning(f"[S4] LLM error for \"{queries[idx]}\": {e}")
                        rewrites[idx] = []

        # Phase 2: FAISS re-search on main thread (SQLite not thread-safe)
        rows = []
        for query, suggested_terms in zip(queries, rewrites):
            rows.append(self._build_row(query, suggested_terms, k))

        return pd.DataFrame(rows)

    def _empty_row(self, query, k):
        """Build an empty result row when LLM produces no suggestions."""
        row = {"original_value": query}
        for i in range(1, k + 1):
            row[f"match{i}"] = "N/A"
            row[f"match{i}_score"] = "0.0000"
        return row
