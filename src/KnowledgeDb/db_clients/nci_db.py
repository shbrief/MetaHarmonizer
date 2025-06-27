import re
from typing import Dict, List
import asyncio
import httpx
import json
import os
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
import json
from src.CustomLogger.custom_logger import CustomLogger
from aiolimiter import AsyncLimiter

from src.KnowledgeDb.db_clients.umls_db import UMLSDb

NCI_CALLS = 18
NCI_PERIOD = 1


class NCIDb:

    def __init__(self, umls_api_key):
        self._base_url = "https://api-evsrest.nci.nih.gov/api/v1"
        self._umls_db = UMLSDb(umls_api_key)
        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.rate_limiter = AsyncLimiter(NCI_CALLS, time_period=NCI_PERIOD)
        self.batch_size = 50
        self.concurrency = 4

    async def fetch_one(self, client, url):
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
        tasks = [self.fetch_one(client, url) for url in urls]
        return await asyncio.gather(*tasks)

    async def get_custom_concepts_by_codes(self, codes, list_of_concepts):
        if not codes:
            self.logger.info("No codes provided for fetching concepts.")
            return {}

        self.logger.info(
            f"Fetching concept data for {len(codes)} codes in batches of {self.batch_size}"
        )
        result = {}
        semaphore = asyncio.Semaphore(self.concurrency)

        async with httpx.AsyncClient(limits=httpx.Limits(
                max_connections=self.concurrency * self.batch_size)) as client:

            async def fetch_and_parse(batch_codes, batch_idx):
                async with semaphore:
                    urls = [
                        f"{self._base_url}/concept/ncit/{code}?include={','.join(list_of_concepts)}"
                        for code in batch_codes
                    ]
                    responses = await self.fetch_batch(client, urls)
                    batch_result = {}
                    for code, response in zip(batch_codes, responses):
                        if response and response.status_code == 200:
                            try:
                                batch_result[code] = response.json()
                            except json.JSONDecodeError as e:
                                self.logger.warning(
                                    f"Failed to parse JSON for code {code}: {e}"
                                )
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
            tasks = [
                fetch_and_parse(batch_codes, i)
                for i, batch_codes in enumerate(batches)
            ]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, r in enumerate(all_results):
                if isinstance(r, dict):
                    result.update(r)
                else:
                    self.logger.warning(f"Batch {i + 1} failed: {r}")

        self.logger.info(f"Retrieved {len(result)} concept data entries")
        return result

    def create_context_list(
        self,
        concept_data: dict,
        list_of_concepts: List[str] = [
            "synonyms",
            "definitions",
            "parents",
            "children",
            "roles",
        ],
    ) -> str:
        """
        Convert a concept dictionary into a structured context string.
        Args:
            concept_data (dict): The concept data fetched from the NCI API.
            list_of_concepts (list): List of concepts to include in the context. Defaults to ["synonyms", "children", "roles", "definitions", "parents"].
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

        for concept in list_of_concepts:
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

    async def get_context_map_by_codes(self, codes: List[str],
                                       fields: List[str]) -> Dict[str, str]:
        concept_map = await self.get_custom_concepts_by_codes(codes, fields)
        return {
            code: self.create_context_list(data, fields)
            for code, data in concept_map.items()
        }
