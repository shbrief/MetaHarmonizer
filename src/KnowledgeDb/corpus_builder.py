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
from src._async_utils import run_async

MAX_CONTEXT_NEIGHBORS = 10


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

        Captures label, obo_id, short_form, definitions, and synonyms so that
        the saved JSON is self-contained for building concept tables offline
        (no further API calls needed).
        """
        descriptions = raw.get("description") or []
        first_definition = ""
        for item in descriptions:
            if isinstance(item, str) and item.strip():
                first_definition = item.strip()
                break
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
            "definitions": [first_definition] if first_definition else [],
            "description": first_definition or None,
            "synonyms": synonyms,
            "parents": [],
            "children": [],
            "roles": [],
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
        record["parents"] = parent_labels[:MAX_CONTEXT_NEIGHBORS]
        record["children"] = child_labels[:MAX_CONTEXT_NEIGHBORS]
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
            sem = asyncio.Semaphore(20)
            chunk_size = 500

            async def _bounded_enrich(raw, client):
                async with sem:
                    return await self._enrich_term(raw, client)

            records = []
            async with httpx.AsyncClient() as client:
                for i in range(0, len(raw_terms), chunk_size):
                    chunk = raw_terms[i:i + chunk_size]
                    chunk_records = await asyncio.gather(
                        *[_bounded_enrich(raw, client) for raw in chunk]
                    )
                    records.extend(chunk_records)
                    self.logger.info(
                        f"Enriched {min(i + chunk_size, len(raw_terms))}"
                        f"/{len(raw_terms)} terms"
                    )
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
        return run_async(
            self.build(root_term_id, ontology, include_root, include_hierarchy)
        )

    # ----------------------------- persistence --------------------------------

    def save(
        self,
        records: list[dict],
        output_path: str,
        root_term_id: str = "",
        root_term_label: str = "",
        ontology: str = "",
        metadata_extra: Optional[dict] = None,
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

        metadata = {
            "root_term_id": root_term_id,
            "root_term_label": root_term_label,
            "ontology": ontology,
            "total_terms": len(records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if metadata_extra:
            metadata.update(metadata_extra)

        output = {"metadata": metadata, "terms": records}

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved {len(records)} terms to {path}")
        return path.resolve()
