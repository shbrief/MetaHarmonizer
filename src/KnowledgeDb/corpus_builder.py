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

import httpx

from src.CustomLogger.custom_logger import CustomLogger
from src.KnowledgeDb.db_clients.ols_db import OLSDb


async def _async_empty() -> list:
    """Return an empty list — used as a no-op coroutine in asyncio.gather."""
    return []


class CorpusBuilder:
    """Collect all descendants of an ontology term and save as JSON.

    Uses OLSDb for all OLS4 API interactions (shared rate limiter and retry).
    """

    def __init__(self):
        self._ols = OLSDb()
        self.logger = CustomLogger().custlogger(loglevel="INFO")

    # ----------------------------- helpers ------------------------------------

    @staticmethod
    def _parse_term(raw: dict) -> dict:
        """Extract a flat record from a raw OLS term dict.

        Captures label, obo_id, short_form, description, and synonyms so that
        the saved JSON is self-contained for building concept tables offline
        (no further API calls needed).
        """
        descriptions = raw.get("description") or []
        # OLS returns synonyms as a list of plain strings
        synonyms = [
            s for s in (raw.get("synonyms") or [])
            if isinstance(s, str) and s.strip()
        ]
        return {
            "iri": raw.get("iri", ""),
            "ontology_name": raw.get("ontology_name", ""),
            "ontology_prefix": raw.get("ontology_prefix", ""),
            "short_form": raw.get("short_form", ""),
            "label": raw.get("label", ""),
            "obo_id": raw.get("obo_id", ""),
            "description": descriptions[0] if descriptions else None,
            "synonyms": synonyms,
            "type": "class",
        }

    # ----------------------------- core API -----------------------------------

    async def _enrich_term(self, raw: dict, client: httpx.AsyncClient) -> dict:
        """Parse a raw OLS term and concurrently fetch its parent/child labels."""
        record = self._parse_term(raw)
        links = raw.get("_links", {})
        parents_href = (links.get("parents") or {}).get("href")
        children_href = (links.get("children") or {}).get("href")

        parent_labels, child_labels = await asyncio.gather(
            self._ols.get_term_neighbors(parents_href, client)
            if parents_href else _async_empty(),
            self._ols.get_term_neighbors(children_href, client)
            if children_href else _async_empty(),
        )
        record["parents"] = parent_labels
        record["children"] = child_labels
        return record

    async def build(
        self,
        root_term_id: str,
        ontology: Optional[str] = None,
        include_root: bool = True,
        include_hierarchy: bool = False,
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
        include_hierarchy : bool
            When True, enrich each term record with ``parents`` and ``children``
            label lists by fetching the OLS ``_links.parents`` / ``_links.children``
            endpoints concurrently.  Produces richer RAG context at the cost of
            additional API calls (roughly 2× the number of terms).

        Returns
        -------
        list[dict]
            Each dict has keys: ``iri``, ``ontology_name``, ``ontology_prefix``,
            ``short_form``, ``label``, ``obo_id``, ``description``,
            ``synonyms``, ``type``.
            When *include_hierarchy* is True, also includes ``parents`` and
            ``children`` (both ``list[str]`` of labels).
        """
        ontology = ontology or OLSDb.infer_ontology(root_term_id)
        self.logger.info(
            f"Building corpus for {root_term_id} "
            f"(ontology={ontology}, include_hierarchy={include_hierarchy})"
        )

        raw_terms: list[dict] = []

        if include_root:
            root_raw = await self._ols.get_term(root_term_id, ontology)
            raw_terms.append(root_raw)

        desc_raw = await self._ols.get_descendants(root_term_id, ontology)
        raw_terms.extend(desc_raw)

        if include_hierarchy:
            self.logger.info(
                f"Fetching parent/child labels for {len(raw_terms)} terms "
                f"(~{len(raw_terms) * 2} extra API calls) ..."
            )
            async with httpx.AsyncClient() as client:
                records = await asyncio.gather(
                    *[self._enrich_term(raw, client) for raw in raw_terms]
                )
            records = list(records)
        else:
            records = [self._parse_term(raw) for raw in raw_terms]

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
        include_hierarchy: bool = False,
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
                self.build(root_term_id, ontology, include_root, include_hierarchy)
            )

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.build(root_term_id, ontology, include_root, include_hierarchy)
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
