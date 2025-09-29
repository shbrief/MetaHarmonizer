# - Uses NCIt concept search (keeps synonym/pref-name matching on the API side)
# - Classifies by prebuilt descendants sets (O(1) membership)
# - Fetches synonyms ONLY when category conflicts or all candidates are unclassified
# - Optional on-disk cache for re-use across runs
from concurrent.futures import wait
import os
import re
import json
import threading
import time
import requests
from typing import Dict, List, Optional, Tuple, Set
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"
NCIT_DICT = {
    "C12219": "body_site",
    "C1909": "treatment_name",
    "C2991": "disease",
}
NCIT_DESC_PATH = "data/schema/ncit_descendants.json"


class NCIClientSync:

    def __init__(self):
        """
        Initialize the NCIClientSync.
        """
        # ---- Session with connection pool + retry ----
        self.session = requests.Session()
        retry = Retry(total=5,
                      connect=5,
                      read=5,
                      backoff_factor=0.5,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"GET"}))
        pool_size = int(
            os.getenv("NCIT_POOL_SIZE",
                      str(min(32, max(8, (os.cpu_count() or 8) * 2)))))
        adapter = HTTPAdapter(max_retries=retry,
                              pool_connections=pool_size,
                              pool_maxsize=pool_size)
        self.session.mount("https://", adapter)
        self.session.headers.update({"User-Agent": "ncit-sync-client/1.3"})
        descendants_json_path = NCIT_DESC_PATH
        cache_path = os.getenv("NCIT_CACHE_PATH", "data/ncit_cache.json")

        # Caches
        self.term2code: Dict[str, Optional[str]] = {
        }  # normalized term -> chosen code (or None)
        self.code2synonyms: Dict[str, Set[str]] = {
        }  # code -> set of synonyms (lowercased), includes pref name
        self.code2category: Dict[str, Optional[str]] = {
        }  # code -> category string or None

        # Load descendants sets (include roots themselves for convenience)
        with open(descendants_json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.desc_sets = {
            "C12219": set(raw.get("C12219", [])) | {"C12219"},
            "C1909": set(raw.get("C1909", [])) | {"C1909"},
            "C3262": set(raw.get("C3262", [])) | {"C3262"},
        }

        # Optional: load persisted cache
        self.cache_path = cache_path
        if cache_path:
            self.load_cache(cache_path)

        # ---- lightweight adaptive rate limiter (no config needed) ----
        base_rps = 8
        max_rps = 12
        self._rps = max(1.0, min(max_rps, base_rps))
        self._rps_cap = max(1.0, max_rps)
        self._success_streak = 0
        self._last_ts = 0.0
        self._limiter_lock = threading.Lock()

    # ----------------------------- utilities -----------------------------

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for caching/comparison (keep letters/digits, collapse spaces, lowercase)."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9]", " ",
                                          str(text))).lower().strip()

    # ---- limiter helpers ----
    def _throttle(self) -> None:
        with self._limiter_lock:
            interval = 1.0 / self._rps
            now = time.time()
            wait = self._last_ts + interval - now
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            self._last_ts = now

    def _on_success(self) -> None:
        with self._limiter_lock:
            self._success_streak += 1
            if self._success_streak >= 100 and self._rps < self._rps_cap:
                self._rps = min(self._rps_cap, self._rps + 1.0)
                self._success_streak = 0

    def _on_backoff(self, retry_after_sec: Optional[float]) -> None:
        with self._limiter_lock:
            self._rps = max(1.0, self._rps * 0.7)
            self._success_streak = 0
        if retry_after_sec and retry_after_sec > 0:
            time.sleep(min(retry_after_sec, 10.0))

    def _get_json(self,
                  url: str,
                  params: dict,
                  retries: int = 0) -> Optional[dict]:
        """GET JSON with adapter-level retries + local smoothing + 429 handling."""
        self._throttle()
        try:
            r = self.session.get(url, params=params, timeout=15)
            if r.status_code == 429:
                ra = r.headers.get("Retry-After")
                self._on_backoff(self._parse_retry_after(ra))
                return None
            r.raise_for_status()
            self._on_success()
            return r.json()
        except requests.RequestException:
            self._on_backoff(None)
            return None

    # Parse Retry-After header (seconds or HTTP-date)
    def _parse_retry_after(self, ra: Optional[str]) -> Optional[float]:
        if not ra:
            return 2.0
        try:
            return float(ra)
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(ra)
                if dt is None:
                    return 2.0
                # Convert to seconds from now (UTC safe)
                now = time.time()
                secs = (dt.timestamp() - now)
                return max(1.0, min(secs, 30.0))
            except Exception:
                return 2.0

    # ----------------------------- NCIt API ------------------------------

    def search_candidates(self,
                          term: str,
                          limit: int = 3) -> Optional[List[Tuple[str, str]]]:
        """
         Search NCIt concepts by term (covers synonyms/pref-name/code).
         Returns: list of (code, preferred_name)
         """
        params = {
            "term": term,
            "type": "match",
            "limit": limit,
            "terminology": "ncit"
        }
        data = self._get_json(f"{BASE_URL}/concept/search", params=params)
        if data is None:
            return None
        if not data.get("concepts"):
            return []
        return [(c.get("code"), c.get("name", "")) for c in data["concepts"]
                if c.get("code")]

    def get_synonyms(self, code: str) -> Set[str]:
        """
        Fetch synonyms for a code (also includes preferred name).
        Lowercased, deduplicated, cached.
        """
        if code in self.code2synonyms:
            return self.code2synonyms[code]
        data = self._get_json(f"{BASE_URL}/concept/ncit/{code}",
                              {"include": "synonyms"})
        names: Set[str] = set()
        if data:
            pref = (data.get("name") or "").strip()
            if pref:
                names.add(pref.lower())
            for s in data.get("synonyms", []) or []:
                nm = (s.get("name") or "").strip()
                if nm:
                    names.add(nm.lower())
        self.code2synonyms[code] = names
        return names

    # ----------------------------- classification ------------------------

    def classify_code(self, code: str) -> Optional[str]:
        """
        Map a code to a top category (body_site / treatment_name / disease) using descendants sets.
        Cached per code.
        """
        if code in self.code2category:
            return self.code2category[code]
        cat = None
        if code in self.desc_sets["C12219"]:
            cat = NCIT_DICT["C12219"]
        elif code in self.desc_sets["C1909"]:
            cat = NCIT_DICT["C1909"]
        elif code in self.desc_sets["C3262"]:
            cat = NCIT_DICT["C3262"]
        self.code2category[code] = cat
        return cat

    def map_value_to_schema(self, value: str) -> str:
        """
        Value -> category, tuned for speed:
          1) Search Top-K candidates (keeps synonym/pref-name matching on API side).
          2) First try to classify by descendants sets WITHOUT fetching synonyms.
             - If exactly one category matches -> pick it immediately.
             - If multiple candidates but all in the same category -> pick that.
             - If multiple candidates across different categories -> then fetch synonyms to disambiguate.
          3) If no candidate falls into any target category, try exact synonym/pref-name match once.
        """
        if not value or not isinstance(value, str):
            return "Unclassified"

        term_norm = self.normalize(value)

        # Cached decision for this normalized term
        if term_norm in self.term2code:
            code = self.term2code[term_norm]
            if not code:
                return "Unclassified"
            cat = self.classify_code(code)
            return cat if cat else "Unclassified"

        # 1) NCIt search (Top-K)
        cands = self.search_candidates(value, limit=5)
        if cands is None:
            return "Unclassified"
        if not cands:
            self.term2code[term_norm] = None
            return "Unclassified"

        # 2) Try fast classification by descendants (no synonyms yet)
        classified: List[Tuple[str, str, str]] = []  # (code, pref, category)
        for code, pref in cands:
            cat = self.classify_code(code)
            if cat:
                classified.append((code, pref, cat))

        if len(classified) == 1:
            code, _, cat = classified[0]
            self.term2code[term_norm] = code
            return cat

        if classified:
            uniq_cats = {c for _, _, c in classified}
            if len(uniq_cats) == 1:
                code, _, cat = classified[0]
                self.term2code[term_norm] = code
                return cat
            # Conflict: candidates span multiple categories -> disambiguate by exact synonym/pref-name
            term_l = term_norm
            for code, pref, cat in classified:
                syns = self.get_synonyms(code)  # only now we pay the cost
                if term_l == (pref or "").strip().lower() or term_l in syns:
                    self.term2code[term_norm] = code
                    return cat
            # No exact synonym hit -> pick the first classified candidate (or define your own priority)
            code, _, cat = classified[0]
            self.term2code[term_norm] = code
            return cat

        # 3) None of the Top-K candidates are in target categories:
        #    Try a single pass of exact synonym/pref-name match; if that code maps to a category, accept.
        term_l = term_norm
        for code, pref in cands:
            syns = self.get_synonyms(code)
            if term_l == (pref or "").strip().lower() or term_l in syns:
                cat = self.classify_code(code)
                if cat:
                    self.term2code[term_norm] = code
                    return cat

        # 4) Still unclassified
        self.term2code[term_norm] = None
        return "Unclassified"

    # ----------------------------- persistence (optional) ----------------

    def save_cache(self, path: Optional[str] = None) -> None:
        """Persist caches to disk (so future runs reuse prior lookups)."""
        if path is None:
            path = self.cache_path
        if not path:
            return
        data = {
            "term2code": self.term2code,
            "code2synonyms": {
                k: sorted(list(v))
                for k, v in self.code2synonyms.items()
            },
            "code2category": self.code2category,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load_cache(self, path: str) -> None:
        """Load caches from disk if available."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.term2code = data.get("term2code", {})
            self.code2synonyms = {
                k: set(v)
                for k, v in data.get("code2synonyms", {}).items()
            }
            self.code2category = data.get("code2category", {})
        except FileNotFoundError:
            pass
