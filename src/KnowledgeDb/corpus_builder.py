"""Build an ontology term corpus from a root term by collecting all descendants.

Analogous to LinkML dynamic enums (reachable_from) and the R-based corpus
building in ``data/corpus/cbio_disease/disease_corpus.R``.

Usage::

    builder = CorpusBuilder()
    corpus = builder.build_sync("NCIT:C3262")
    builder.save(corpus, "data/corpus/neoplasm_corpus.json")
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.CustomLogger.custom_logger import CustomLogger
from src.KnowledgeDb.db_clients.ols_client import OLSClient


class CorpusBuilder:
    """Collect all descendants of an ontology term and save as JSON.

    Parameters
    ----------
    page_size : int
        Number of terms per page when fetching descendants from OLS.
    rate_limit_calls : int
        Max OLS API requests per *rate_limit_period* seconds.
    rate_limit_period : float
        Rate-limit window in seconds.
    """

    def __init__(
        self,
        page_size: int = 200,
        rate_limit_calls: int = 10,
        rate_limit_period: float = 1.0,
    ):
        self._ols = OLSClient(
            page_size=page_size,
            rate_limit_calls=rate_limit_calls,
            rate_limit_period=rate_limit_period,
        )
        self.logger = CustomLogger().custlogger(loglevel="INFO")

    # ----------------------------- helpers ------------------------------------

    @staticmethod
    def _parse_term(raw: dict) -> dict:
        """Extract a flat record from a raw OLS term dict."""
        descriptions = raw.get("description") or []
        return {
            "iri": raw.get("iri", ""),
            "ontology_name": raw.get("ontology_name", ""),
            "ontology_prefix": raw.get("ontology_prefix", ""),
            "short_form": raw.get("short_form", ""),
            "label": raw.get("label", ""),
            "obo_id": raw.get("obo_id", ""),
            "description": descriptions[0] if descriptions else None,
            "type": "class",
        }

    # ----------------------------- core API -----------------------------------

    async def build(
        self,
        root_term_id: str,
        ontology: Optional[str] = None,
        include_root: bool = True,
    ) -> list[dict]:
        """Collect all descendant terms from *root_term_id*.

        Parameters
        ----------
        root_term_id : str
            OBO-format term ID, e.g. ``"NCIT:C3262"`` or ``"MONDO:0000001"``.
        ontology : str, optional
            Ontology short name (e.g. ``"ncit"``).  Inferred from the prefix
            of *root_term_id* when not provided.
        include_root : bool
            Whether to include the root term itself in the output.

        Returns
        -------
        list[dict]
            Each dict has keys: ``iri``, ``ontology_name``, ``ontology_prefix``,
            ``short_form``, ``label``, ``obo_id``, ``description``, ``type``.
        """
        ontology = ontology or OLSClient.infer_ontology(root_term_id)
        self.logger.info(
            f"Building corpus for {root_term_id} (ontology={ontology})"
        )

        records: list[dict] = []

        if include_root:
            root_raw = await self._ols.get_term(root_term_id, ontology)
            records.append(self._parse_term(root_raw))

        desc_raw = await self._ols.get_descendants(root_term_id, ontology)
        for raw in desc_raw:
            records.append(self._parse_term(raw))

        self.logger.info(
            f"Corpus complete: {len(records)} terms "
            f"(root={'included' if include_root else 'excluded'})"
        )
        return records

    def build_sync(
        self,
        root_term_id: str,
        ontology: Optional[str] = None,
        include_root: bool = True,
    ) -> list[dict]:
        """Synchronous wrapper around :meth:`build`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(
                self.build(root_term_id, ontology, include_root)
            )

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.build(root_term_id, ontology, include_root)
            )
        finally:
            loop.close()

    # ----------------------------- persistence --------------------------------

    def save(
        self,
        records: list[dict],
        output_path: str,
        root_term_id: str = "",
        root_term_label: str = "",
        ontology: str = "",
    ) -> Path:
        """Write corpus records to a JSON file.

        Parameters
        ----------
        records : list[dict]
            Term records returned by :meth:`build`.
        output_path : str
            Destination file path.
        root_term_id, root_term_label, ontology : str
            Optional metadata written into the ``metadata`` envelope.

        Returns
        -------
        pathlib.Path
            Absolute path of the written file.
        """
        if not root_term_label and records:
            root_term_label = records[0].get("label", "")
        if not ontology and records:
            ontology = records[0].get("ontology_name", "")

        output = {
            "metadata": {
                "root_term_id": root_term_id,
                "root_term_label": root_term_label,
                "ontology": ontology,
                "total_terms": len(records),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "terms": records,
        }

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(records)} terms to {path}")
        return path.resolve()
