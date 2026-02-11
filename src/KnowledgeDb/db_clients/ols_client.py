"""EBI OLS4 API client for ontology term and descendant retrieval."""

import asyncio
from urllib.parse import quote

import httpx
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.CustomLogger.custom_logger import CustomLogger

OLS_BASE_URL = "https://www.ebi.ac.uk/ols4/api"
OLS_CALLS = 10
OLS_PERIOD = 1.0


class OLSClient:
    """Async client for the EBI OLS4 REST API.

    Supports fetching single terms and paginated descendant traversal
    for any ontology hosted on OLS (NCIT, MONDO, UBERON, HP, etc.).
    """

    def __init__(
        self,
        base_url: str = OLS_BASE_URL,
        rate_limit_calls: int = OLS_CALLS,
        rate_limit_period: float = OLS_PERIOD,
        page_size: int = 200,
    ):
        self._base_url = base_url.rstrip("/")
        self.page_size = page_size
        self.rate_limiter = AsyncLimiter(rate_limit_calls, rate_limit_period)
        self.logger = CustomLogger().custlogger(loglevel="INFO")

    # ----------------------------- static helpers -----------------------------

    @staticmethod
    def obo_id_to_iri(obo_id: str) -> str:
        """Convert OBO ID to a full OBO Foundry IRI.

        Example: "NCIT:C3262" -> "http://purl.obolibrary.org/obo/NCIT_C3262"
        """
        prefix, local_id = obo_id.split(":", 1)
        return f"http://purl.obolibrary.org/obo/{prefix}_{local_id}"

    @staticmethod
    def double_encode_iri(iri: str) -> str:
        """Double-URL-encode an IRI for use in OLS4 path parameters."""
        return quote(quote(iri, safe=""), safe="")

    @staticmethod
    def infer_ontology(term_id: str) -> str:
        """Infer ontology short name from an OBO-format term ID prefix.

        Example: "NCIT:C3262" -> "ncit", "MONDO:0000001" -> "mondo"
        """
        prefix = term_id.split(":")[0]
        return prefix.lower()

    # ----------------------------- HTTP helpers -------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.RequestError)),
    )
    async def _fetch(self, client: httpx.AsyncClient, url: str) -> dict:
        """GET JSON from *url* with rate limiting and retry."""
        async with self.rate_limiter:
            resp = await client.get(url, timeout=30.0)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                self.logger.warning(f"OLS rate-limited; sleeping {retry_after}s")
                await asyncio.sleep(retry_after)
                resp.raise_for_status()
            resp.raise_for_status()
            return resp.json()

    # ----------------------------- public API ---------------------------------

    async def get_term(
        self,
        obo_id: str,
        ontology: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict:
        """Fetch metadata for a single ontology term.

        Args:
            obo_id: OBO-format ID, e.g. "NCIT:C3262".
            ontology: Ontology short name. Inferred from prefix if None.
            client: Optional shared httpx client.

        Returns:
            Raw OLS term dict with keys: iri, label, short_form, obo_id,
            ontology_name, ontology_prefix, description, synonyms, etc.

        Raises:
            httpx.HTTPStatusError: If the term is not found (404) or server error.
        """
        ontology = ontology or self.infer_ontology(obo_id)
        iri = self.obo_id_to_iri(obo_id)
        encoded = self.double_encode_iri(iri)
        url = f"{self._base_url}/ontologies/{ontology}/terms/{encoded}"

        if client is None:
            async with httpx.AsyncClient() as c:
                return await self._fetch(c, url)
        return await self._fetch(client, url)

    async def get_descendants(
        self,
        obo_id: str,
        ontology: str | None = None,
        page_size: int | None = None,
    ) -> list[dict]:
        """Fetch all descendant terms of a root term via paginated OLS4 API.

        Args:
            obo_id: Root term OBO ID, e.g. "NCIT:C3262".
            ontology: Ontology short name. Inferred from prefix if None.
            page_size: Results per page (default: self.page_size).

        Returns:
            List of raw OLS term dicts for every descendant.
        """
        ontology = ontology or self.infer_ontology(obo_id)
        iri = self.obo_id_to_iri(obo_id)
        encoded = self.double_encode_iri(iri)
        size = page_size or self.page_size
        url = (
            f"{self._base_url}/ontologies/{ontology}"
            f"/terms/{encoded}/descendants?size={size}"
        )

        all_terms: list[dict] = []
        async with httpx.AsyncClient() as client:
            while url:
                data = await self._fetch(client, url)
                terms = data.get("_embedded", {}).get("terms", [])
                all_terms.extend(terms)

                page_info = data.get("page", {})
                total = page_info.get("totalElements", "?")
                pages = page_info.get("totalPages", "?")
                current = page_info.get("number", 0) + 1
                self.logger.info(
                    f"Fetched page {current}/{pages} "
                    f"({len(all_terms)}/{total} terms)"
                )

                url = data.get("_links", {}).get("next", {}).get("href")

        return all_terms
