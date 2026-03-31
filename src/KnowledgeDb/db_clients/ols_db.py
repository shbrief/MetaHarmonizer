import asyncio
from typing import Dict, List
from urllib.parse import quote_plus
import httpx
from tenacity import (retry, stop_after_attempt, wait_exponential,
                      retry_if_exception_type)
from aiolimiter import AsyncLimiter
from src.CustomLogger.custom_logger import CustomLogger

OLS_CALLS = 10
OLS_PERIOD = 1

# Maps code prefix (as stored in clean_code, e.g. "EFO") to the OLS ontology id
PREFIX_TO_ONTOLOGY = {
    "EFO": "efo",
    "UBERON": "uberon",
    "HP": "hp",
    "Orphanet": "ordo",
    "MONDO": "mondo",
    "DOID": "doid",
    "CL": "cl",
    "CHEBI": "chebi",
    "PATO": "pato",
    "GO": "go",
    "BFO": "bfo",
    "OBI": "obi",
}

# Maps code prefix to the IRI base used to construct the full IRI
IRI_BASES = {
    "EFO": "http://www.ebi.ac.uk/efo/",
    "Orphanet": "http://www.orpha.net/ORDO/Orphanet_",
}
# Most OBO ontologies use this base
_OBO_BASE = "http://purl.obolibrary.org/obo/"


def _curie_to_iri(prefix: str, local_id: str) -> str:
    """Convert a prefix + local_id to a full IRI for OLS lookup."""
    base = IRI_BASES.get(prefix, _OBO_BASE)
    if prefix == "Orphanet":
        return f"{base}{local_id}"
    return f"{base}{prefix}_{local_id}"


def parse_code_prefix(code: str) -> str | None:
    """Extract the ontology prefix from a code.

    Handles: 'EFO:0000249' (colon), 'EFO_0000249' (underscore), 'C12345' (bare NCIT).
    """
    if ":" in code:
        return code.split(":", 1)[0]
    if "_" in code:
        return code.split("_", 1)[0]
    return None


def partition_codes(codes: list[str]) -> dict[str, list[str]]:
    """Split codes into NCI vs OLS groups based on prefix.

    NCIT codes look like 'C12345' (bare) or 'NCIT:C12345'.
    Everything with a recognized OLS prefix goes to OLS.
    """
    nci_codes = []
    ols_codes = []
    for code in codes:
        prefix = parse_code_prefix(code)
        if prefix is None:
            nci_codes.append(code)
        elif prefix == "NCIT":
            local = code.split(":", 1)[1] if ":" in code else code.split("_", 1)[1]
            nci_codes.append(local)
        elif prefix in PREFIX_TO_ONTOLOGY:
            ols_codes.append(code)
        else:
            nci_codes.append(code)
    return {"nci": nci_codes, "ols": ols_codes}


def _parse_clean_code(clean_code: str):
    """Parse a clean_code into (prefix, local_id).

    Handles both formats:
      - 'EFO:0000239'  (colon-separated, from corpus CSV)
      - 'EFO_0000239'  (underscore-separated, from _obo_to_clean_code)

    Returns (prefix, local_id) or (None, None) if unparseable.
    """
    if ":" in clean_code:
        prefix, local_id = clean_code.split(":", 1)
        return prefix, local_id
    if "_" in clean_code:
        prefix, local_id = clean_code.split("_", 1)
        return prefix, local_id
    return None, None


class OLSDb:
    """Client for the EMBL-EBI OLS4 REST API.

    Mirrors NCIDb's interface so it can be used as a drop-in data source
    in ConceptTableBuilder for non-NCIT ontology terms.
    """

    def __init__(self):
        self._base_url = "https://www.ebi.ac.uk/ols4/api"
        self.logger = CustomLogger().custlogger(loglevel="INFO")
        self.rate_limiter = AsyncLimiter(OLS_CALLS, time_period=OLS_PERIOD)
        self.batch_size = 50
        self.concurrency = 4

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.ConnectError,
             httpx.HTTPStatusError)),
        retry_error_callback=lambda retry_state: None,
    )
    async def fetch_one(self, client: httpx.AsyncClient, url: str):
        """Fetch a single URL with rate limiting and retry on transient errors."""
        async with self.rate_limiter:
            r = await client.get(url, timeout=15.0)
            if r.status_code == 429:
                self.logger.warning(
                    f"OLSDb: Rate limited for {url}, retrying after backoff")
                raise httpx.HTTPStatusError("429 Too Many Requests",
                                            request=r.request,
                                            response=r)
            if r.status_code >= 500:
                self.logger.warning(
                    f"OLSDb: Server error {r.status_code} for {url}, retrying")
                raise httpx.HTTPStatusError(f"{r.status_code} Server Error",
                                            request=r.request,
                                            response=r)
            if r.status_code != 200:
                self.logger.warning(
                    f"OLSDb: Non-200 response {r.status_code} for {url}")
                return None
            return r

    async def fetch_batch(self, client: httpx.AsyncClient,
                          urls: List[str]):
        """Fetch a batch of URLs concurrently."""
        tasks = [self.fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)

    @staticmethod
    def obo_id_to_iri(obo_id: str) -> str:
        """Convert an OBO ID (e.g. 'MONDO:0000001') to a full IRI."""
        prefix, local_id = obo_id.split(":", 1)
        return _curie_to_iri(prefix, local_id)

    @staticmethod
    def infer_ontology(term_id: str) -> str:
        """Infer ontology short name from OBO-format prefix."""
        prefix = term_id.split(":")[0]
        return prefix.lower()

    async def get_term(self, obo_id: str, ontology: str | None = None,
                       client: httpx.AsyncClient | None = None) -> dict:
        """Fetch raw metadata for a single term by OBO ID."""
        ontology = ontology or self.infer_ontology(obo_id)
        iri = self.obo_id_to_iri(obo_id)
        encoded = quote_plus(quote_plus(iri))
        url = f"{self._base_url}/ontologies/{ontology}/terms/{encoded}"
        if client is None:
            async with httpx.AsyncClient() as c:
                resp = await self.fetch_one(c, url)
        else:
            resp = await self.fetch_one(client, url)
        if resp is None:
            raise httpx.HTTPStatusError(
                f"Failed to fetch term {obo_id}", request=None, response=None)
        return resp.json()

    async def get_descendants(self, obo_id: str,
                              ontology: str | None = None,
                              page_size: int = 200) -> list[dict]:
        """Fetch all descendant terms of a root term via paginated OLS4 API."""
        ontology = ontology or self.infer_ontology(obo_id)
        iri = self.obo_id_to_iri(obo_id)
        encoded = quote_plus(quote_plus(iri))
        url = (f"{self._base_url}/ontologies/{ontology}"
               f"/terms/{encoded}/descendants?size={page_size}")

        all_terms: list[dict] = []
        async with httpx.AsyncClient() as client:
            while url:
                resp = await self.fetch_one(client, url)
                if resp is None:
                    break
                data = resp.json()
                terms = data.get("_embedded", {}).get("terms", [])
                all_terms.extend(terms)

                page_info = data.get("page", {})
                total = page_info.get("totalElements", "?")
                pages = page_info.get("totalPages", "?")
                current = page_info.get("number", 0) + 1
                self.logger.info(
                    f"Fetched page {current}/{pages} "
                    f"({len(all_terms)}/{total} terms)")

                url = data.get("_links", {}).get("next", {}).get("href")

        return all_terms

    async def get_term_neighbors(self, href: str,
                                 client: httpx.AsyncClient) -> list[str]:
        """Fetch labels of related terms from a pre-built _links href.

        Used for parent/child label retrieval in CorpusBuilder.
        Returns list of label strings; empty list on error.
        """
        try:
            resp = await self.fetch_one(client, href)
            if resp is None:
                return []
            data = resp.json()
            terms = data.get("_embedded", {}).get("terms", [])
            return [t["label"] for t in terms if t.get("label")]
        except Exception as exc:
            self.logger.debug(f"get_term_neighbors failed for {href}: {exc}")
            return []

    def _build_url(self, clean_code: str) -> str | None:
        """Build an OLS term-lookup URL from a clean_code like 'EFO_0000239'."""
        prefix, local_id = _parse_clean_code(clean_code)
        if prefix is None:
            return None
        ontology = PREFIX_TO_ONTOLOGY.get(prefix)
        if ontology is None:
            self.logger.warning(
                f"OLSDb: Unknown prefix '{prefix}' in code {clean_code}")
            return None
        iri = _curie_to_iri(prefix, local_id)
        encoded_iri = quote_plus(quote_plus(iri))
        return f"{self._base_url}/ontologies/{ontology}/terms/{encoded_iri}"

    def _normalize_response(self, data: dict) -> dict:
        """Convert an OLS term response into the NCIDb-compatible structure.

        Returns a dict with keys: name, synonyms, definitions, parents, children.
        """
        result = {}
        result["name"] = data.get("label", "")

        # Synonyms: OLS returns a flat list of strings
        raw_syns = data.get("synonyms") or []
        result["synonyms"] = [{"name": s} for s in raw_syns if s]

        # Definitions: OLS returns description as a list of strings
        raw_defs = data.get("description") or []
        result["definitions"] = [{"definition": d} for d in raw_defs if d]

        # OBO synonyms (exact/related/broad/narrow) may appear here
        for key in ("obo_synonym", "annotation"):
            if key in data and isinstance(data[key], dict):
                for syn_type_entries in data[key].values():
                    if isinstance(syn_type_entries, list):
                        for entry in syn_type_entries:
                            if isinstance(entry, str) and entry:
                                result["synonyms"].append({"name": entry})

        # Parents and children are fetched via embedded _links
        result["parents"] = []
        result["children"] = []
        result["roles"] = []

        return result

    async def _fetch_hierarchy(self, client: httpx.AsyncClient,
                               term_data: dict,
                               direction: str) -> list:
        """Fetch parent or child terms from OLS _links.

        Args:
            direction: 'parents' or 'children' (matching _links keys)
        """
        links = term_data.get("_links", {})
        href = links.get(direction, {}).get("href")
        if not href:
            return []

        resp = await self.fetch_one(client, href)
        if resp is None:
            return []
        try:
            payload = resp.json()
        except Exception:
            return []

        terms = (payload.get("_embedded", {}).get("terms") or [])
        return [{"name": t.get("label", "")} for t in terms if t.get("label")]

    async def get_custom_concepts_by_codes(
            self,
            codes: List[str],
            client: httpx.AsyncClient = None) -> Dict[str, dict]:
        """Fetch concept data for a list of clean_codes from the OLS API.

        Returns a dict mapping each code to an NCIDb-compatible concept dict.
        """
        if not codes:
            self.logger.info("OLSDb: No codes provided")
            return {}

        self.logger.info(
            f"OLSDb: Fetching concept data for {len(codes)} codes "
            f"in batches of {self.batch_size}")

        result = {}
        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch_and_parse(batch_codes, batch_idx, http_client):
            async with semaphore:
                urls = []
                valid_codes = []
                for code in batch_codes:
                    url = self._build_url(code)
                    if url:
                        urls.append(url)
                        valid_codes.append(code)
                    else:
                        self.logger.warning(
                            f"OLSDb: Could not build URL for code {code}")

                if not urls:
                    return {}

                responses = await self.fetch_batch(http_client, urls)
                batch_result = {}
                for code, response in zip(valid_codes, responses):
                    if response and response.status_code == 200:
                        try:
                            raw = response.json()
                            normalized = self._normalize_response(raw)

                            # Fetch parents and children
                            parents = await self._fetch_hierarchy(
                                http_client, raw, "parents")
                            children = await self._fetch_hierarchy(
                                http_client, raw, "children")
                            normalized["parents"] = parents
                            normalized["children"] = children

                            batch_result[code] = normalized
                        except Exception as e:
                            self.logger.warning(
                                f"OLSDb: Failed to parse response for "
                                f"{code}: {e}")
                    else:
                        status = response.status_code if response else "No response"
                        self.logger.warning(
                            f"OLSDb: Failed to fetch code {code}: {status}")

                self.logger.info(
                    f"OLSDb: Processed batch {batch_idx + 1} of "
                    f"{len(batches)}")
                return batch_result

        batches = [
            codes[i:i + self.batch_size]
            for i in range(0, len(codes), self.batch_size)
        ]

        if client is None:
            async with httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.concurrency *
                        self.batch_size)) as owned_client:
                tasks = [
                    fetch_and_parse(batch, i, owned_client)
                    for i, batch in enumerate(batches)
                ]
                all_results = await asyncio.gather(*tasks,
                                                   return_exceptions=True)
        else:
            tasks = [
                fetch_and_parse(batch, i, client)
                for i, batch in enumerate(batches)
            ]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(all_results):
            if isinstance(r, dict):
                result.update(r)
            else:
                self.logger.warning(f"OLSDb: Batch {i + 1} failed: {r}")

        self.logger.info(
            f"OLSDb: Retrieved {len(result)} concept data entries")
        return result

    def create_context_list(self, concept_data: dict) -> str:
        """Convert a concept dict into a structured context string.

        Mirrors NCIDb.create_context_list() output format.
        """
        context = []

        # Synonyms
        syns = concept_data.get("synonyms", [])
        syn_names = list({
            item.get("name", "")
            for item in syns if isinstance(item, dict) and item.get("name")
        })
        if syn_names:
            context.append(f"synonyms: {'; '.join(syn_names)}")

        # Definitions
        defs = concept_data.get("definitions", [])
        seen = set()
        def_parts = []
        for item in defs:
            if isinstance(item, dict):
                d = item.get("definition", "").strip()
                if d and d.lower() not in seen:
                    seen.add(d.lower())
                    def_parts.append(d)
        if def_parts:
            context.append(f"definitions: {'; '.join(def_parts)}")

        # Parents
        parents = concept_data.get("parents", [])
        parent_names = list({
            item.get("name", "")
            for item in parents
            if isinstance(item, dict) and item.get("name")
        })[:10]
        if parent_names:
            context.append(f"parents: {'; '.join(parent_names)}")

        # Children
        children = concept_data.get("children", [])
        child_names = list({
            item.get("name", "")
            for item in children
            if isinstance(item, dict) and item.get("name")
        })[:10]
        if child_names:
            context.append(f"children: {'; '.join(child_names)}")

        return ". ".join(context)
