import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from src.CustomLogger.custom_logger import CustomLogger
from aiolimiter import AsyncLimiter

UMLS_CALLS = 18
UMLS_PERIOD = 1


class UMLSDb:

    def __init__(self, api_key):
        self.api_key = api_key
        self._base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.logger = CustomLogger().custlogger(loglevel='INFO')
        self.rate_limiter = AsyncLimiter(UMLS_CALLS, time_period=UMLS_PERIOD)
        self.semaphore = asyncio.Semaphore(15)

    @retry(stop=stop_after_attempt(3),
           wait=wait_fixed(2),
           retry=retry_if_exception_type(
               (httpx.HTTPStatusError, httpx.RequestError)))
    async def get_nci_code_by_term(self, term: str, client: httpx.AsyncClient):
        async with self.semaphore:
            async with self.rate_limiter:
                uri = f"{self._base_url}/search/current"
                query = {
                    "string": term,
                    "apiKey": self.api_key,
                    "pageNumber": 1,
                    "searchType": "exact",
                    "sabs": "NCI",
                    "returnIdType": "code",
                }
                try:
                    r = await client.get(uri, params=query, timeout=10.0)
                    r.raise_for_status()
                    results = r.json().get("result", {}).get("results", [])
                    codes = [
                        res["ui"] for res in results
                        if res.get("rootSource") == "NCI"
                    ]
                    if not codes:
                        self.logger.info(
                            f"No NCI code found for term '{term}'")
                        return []
                    return codes
                except httpx.HTTPStatusError as e:
                    self.logger.warning(f"HTTP error for term '{term}': {e}")
                    raise
                except Exception as e:
                    self.logger.error(f"Other error for term '{term}': {e}")
                    raise
