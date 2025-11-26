import re
from typing import Dict, List
import asyncio
import httpx
import json
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
from src.CustomLogger.custom_logger import CustomLogger
from aiolimiter import AsyncLimiter

from src.KnowledgeDb.db_clients.umls_db import UMLSDb

NCI_CALLS = 18
NCI_PERIOD = 1
LIST_OF_CONCEPTS = [
    # "synonyms",
    "definitions",
    "parents",
    "children",
    "roles",
]


class NCIDb:
    """A class to interact with the NCI (National Cancer Institute) API for fetching concept data.
    This class provides methods to fetch concepts by their codes, create context strings from the fetched data,
    and manage rate limiting for API calls.
    Attributes:
        _base_url (str): The base URL for the NCI API.
        _umls_db (UMLSDb): An instance of the UMLSDb class for interacting with the UMLS database.
        logger (CustomLogger): An instance of CustomLogger for logging messages.
        rate_limiter (AsyncLimiter): An instance of AsyncLimiter to manage rate limiting for API calls.
        batch_size (int): The number of codes to fetch in a single batch.
        concurrency (int): The number of concurrent requests allowed.
    """

    def __init__(self, umls_api_key):
        self._base_url = "https://api-evsrest.nci.nih.gov/api/v1"
        self._umls_db = UMLSDb(umls_api_key)
        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.rate_limiter = AsyncLimiter(NCI_CALLS, time_period=NCI_PERIOD)
        self.batch_size = 50
        self.concurrency = 4
        self.list_of_concepts = LIST_OF_CONCEPTS

    async def fetch_one(self, client, url):
        """Fetch a single URL with rate limiting and error handling.
        Args:
            client (httpx.AsyncClient): The HTTP client to use for the request.
            url (str): The URL to fetch.
        Returns:
            httpx.Response: The response object if the request is successful.
            None: If the request fails or if the response status is not 200.
        """
        async with self.rate_limiter:
            try:
                r = await client.get(url, timeout=10.0)
                if r.status_code == 429:
                    self.logger.warning(
                        f"Rate limited for {url}: sleeping briefly before retry"
                    )
                    await asyncio.sleep(5)
                    raise httpx.HTTPStatusError("429 Too Many Requests",
                                                request=r.request,
                                                response=r)
                return r
            except httpx.HTTPStatusError as e:
                self.logger.warning(f"HTTP error for {url}: {e}")
                return None
            except Exception as e:
                self.logger.error(f"Failed to fetch {url}: {e}")
                return None

    @retry(stop=stop_after_attempt(3),
           wait=wait_fixed(2),
           retry=retry_if_exception_type(
               (httpx.HTTPStatusError, httpx.RequestError)))
    async def fetch_batch(self, client, urls):
        """Fetch a batch of URLs concurrently with error handling.
        Args:
            client (httpx.AsyncClient): The HTTP client to use for the requests.
            urls (List[str]): A list of URLs to fetch.
        Returns:
            List[httpx.Response]: A list of response objects for the fetched URLs.
        """
        tasks = [self.fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)

    async def get_custom_concepts_by_codes(self, codes, client=None):
        """Fetch concept data for a list of codes from the NCI API.
        Args:
            codes (List[str]): A list of NCI codes to fetch concept data for.
        Returns:
            Dict[str, dict]: A dictionary mapping each code to its corresponding concept data.
        """
        if not codes:
            self.logger.info("No codes provided for fetching concepts.")
            return {}

        self.logger.info(
            f"Fetching concept data for {len(codes)} codes in batches of {self.batch_size}"
        )
        result = {}
        semaphore = asyncio.Semaphore(self.concurrency)

        async def fetch_and_parse(batch_codes, batch_idx, http_client):
            async with semaphore:
                urls = [
                    f"{self._base_url}/concept/ncit/{code}?include={','.join(self.list_of_concepts)}"
                    for code in batch_codes
                ]
                responses = await self.fetch_batch(http_client, urls)
                batch_result = {}
                for code, response in zip(batch_codes, responses):
                    if response and response.status_code == 200:
                        try:
                            batch_result[code] = response.json()
                        except json.JSONDecodeError as e:
                            self.logger.warning(
                                f"Failed to parse JSON for code {code}: {e}")
                    else:
                        self.logger.warning(
                            f"Failed to fetch code {code}: "
                            f"{response.status_code if response else 'No response'}"
                        )
                self.logger.info(
                    f"Processed batch {batch_idx + 1} of {len(batches)}")
                return batch_result

        batches = [
            codes[i:i + self.batch_size]
            for i in range(0, len(codes), self.batch_size)
        ]

        if client is None:
            async with httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=self.concurrency *
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
                self.logger.warning(f"Batch {i + 1} failed: {r}")

        self.logger.info(f"Retrieved {len(result)} concept data entries")
        return result

    def create_context_list(self, concept_data: dict) -> str:
        """
        Convert a concept dictionary into a structured context string.
        Args:
            concept_data (dict): The concept data fetched from the NCI API.
        Returns:
            str: A string representation of the context, formatted as "concept: item1, item2, ...".
        """
        context = []
        excluded_prefixes = set([
            "Gene_", "Allele_", "Molecular_Abnormality_Involves_Gene",
            "Cytogenetic_Abnormality_Involves_Chromosome", "EO_Disease_",
            "Conceptual_Part_Of", "Chemotherapy_Regimen_Has_Component",
            "Biological_Process_"
        ])

        for concept in self.list_of_concepts:
            if concept not in concept_data:
                continue
            parts = []
            if concept == "roles":
                role_map = {}
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        role_type = item.get("type")
                        role_target = item.get("relatedName")
                        if any(
                                role_type.startswith(prefix)
                                for prefix in excluded_prefixes):
                            continue
                        if role_type and role_target:
                            role_map.setdefault(role_type,
                                                []).append(role_target)
                simplified_role_map = {}
                for role_type, targets in role_map.items():
                    key = re.sub(
                        r"^(?:Disease_|Procedure_)?(?:Has|Is|May_Have|Mapped_To|Excludes|Has_Accepted_Use_For)?_?",
                        "", role_type)
                    simplified_role_map.setdefault(key, set()).update(targets)
                for role_type, target_set in simplified_role_map.items():
                    target_list = list(target_set)[:10]  # Limit to 10 targets
                    parts.append(f"{role_type}: {'; '.join(target_list)}")
            elif concept == "definitions":
                seen_defs = set()
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        definition = item.get("definition", "")
                        if definition:
                            cleaned_def = definition.strip()
                            cleaned_def = re.sub(r'[.;\s]+$', '', cleaned_def)
                            norm_def = ' '.join(cleaned_def.split())
                            norm_def = re.sub(r'[\s\.;,:-]*\(?NCI\)?$',
                                              '',
                                              norm_def,
                                              flags=re.IGNORECASE)
                            norm_def_lower = norm_def.lower()
                            if norm_def_lower not in seen_defs:
                                seen_defs.add(norm_def_lower)
                                parts.append(norm_def)
            else:
                names = []
                for item in concept_data[concept]:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        if name:
                            names.append(name)
                parts = list(set(names)) if concept == "synonyms" else list(
                    set(names))[:10]
            if parts:
                context.append(f"{concept}: {'; '.join(parts)}")
        return ". ".join(context)

    async def get_context_map_by_codes(self,
                                       codes: List[str]) -> Dict[str, str]:
        """
        Fetch context data for a list of NCI codes and create a mapping of codes to context strings.
        Args:
            codes (List[str]): A list of NCI codes to fetch context data for.
        Returns:
            Dict[str, str]: A dictionary mapping each code to its corresponding context string.
        """
        concept_map = await self.get_custom_concepts_by_codes(codes)
        return {
            code: self.create_context_list(data)
            for code, data in concept_map.items()
        }

    async def get_labels_by_codes(self, codes: List[str]) -> Dict[str, str]:
        """
        Fetch labels for a list of NCI codes and create a mapping of codes to their preferred names.
        Args:
            codes (List[str]): A list of NCI codes to fetch labels for.
        Returns:
            Dict[str, str]: A dictionary mapping each code to its corresponding preferred name.
        """
        concept_data = await self.get_custom_concepts_by_codes(codes)
        code_to_label = {}
        for code, data in concept_data.items():
            label = data.get("name")
            if label:
                code_to_label[code] = label.strip()
        return code_to_label
