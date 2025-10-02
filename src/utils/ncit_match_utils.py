# - Uses NCIt concept search (keeps synonym/pref-name matching on the API side)
# - Classifies by prebuilt descendants sets (O(1) membership) + BFS up to 5 hops via parents
from collections import defaultdict
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
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://api-evsrest.nci.nih.gov/api/v1"
NCIT_DICT = {
    "C12219": "body_site",
    "C1909": "treatment_name",
    "C2991": "disease",
    "C3262": "cancer_type"
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

        # Caches
        self.term2code: Dict[str, Optional[str]] = {
        }  # normalized term -> chosen code (or None)
        self.code2category: Dict[str, Optional[List[str]]] = {
        }  # code -> category list or None

        # Load descendants sets (include roots themselves for convenience)
        with open(NCIT_DESC_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.desc_sets = {
            "C12219": set(raw.get("C12219", [])) | {"C12219"},
            "C1909": set(raw.get("C1909", [])) | {"C1909"},
            "C2991": set(raw.get("C2991", [])) | {"C2991"},
            "C3262": set(raw.get("C3262", [])) | {"C3262"},
        }

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
                          limit: int = 1) -> Optional[List[Tuple[str, str]]]:
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
                if c.get("code")][:limit]

    # ----------------------------- classification ------------------------
    def _classify_local(self, code: str) -> Optional[str]:
        if code in self.desc_sets["C12219"]:
            return NCIT_DICT["C12219"]  # body_site
        if code in self.desc_sets["C1909"]:
            return NCIT_DICT["C1909"]  # treatment_name
        if code in self.desc_sets["C2991"]:
            return NCIT_DICT["C2991"]  # disease
        if code in self.desc_sets["C3262"]:
            return NCIT_DICT["C3262"]  # cancer_type
        return None

    def _fetch_parents(self, code: str) -> Optional[List[str]]:
        data = self._get_json(f"{BASE_URL}/concept/ncit/{code}",
                              {"include": "parents"})
        if not data:
            return None
        parents = data.get("parents") or []
        out = []
        for p in parents:
            c = p.get("code")
            if c:
                out.append(c)
        return out

    def _classify_via_api(self,
                          code: str,
                          max_hops: int = 5) -> Tuple[List[str], bool]:
        # ok=True means no network failure, False means network failure (None result)
        hits: List[str] = []

        # Direct hit
        if code in NCIT_DICT:
            return [NCIT_DICT[code]], True

        seen = {code}
        frontier = [code]
        hops = 0
        while frontier and hops < max_hops:
            nxt = []
            for c in frontier:
                parents = self._fetch_parents(c)
                if parents is None:
                    return [], False  # network failure
                if not parents:
                    continue
                for p in parents:
                    if p in seen:
                        continue
                    seen.add(p)
                    # 命中根
                    if p in NCIT_DICT:
                        hits.append(NCIT_DICT[p])
                    else:
                        local_cat = self._classify_local(p)
                        if local_cat:
                            hits.append(local_cat)
                        else:
                            nxt.append(p)
            frontier = nxt
            hops += 1

        return hits, True

    def classify_code(self, code: str) -> List[str]:
        if code in self.code2category:
            return self.code2category[code]

        hits = []
        for root, catname in NCIT_DICT.items():
            if code in self.desc_sets[root]:
                hits.append(catname)

        if hits:
            self.code2category[code] = hits
            return hits

        cats, ok = self._classify_via_api(code)
        if cats:
            self.code2category[code] = cats
            return cats

        if ok:
            self.code2category[code] = []  # negative cache

        return []

    def map_value_to_schema(self, values: List[str]) -> Dict[str, int]:
        """
        Value(s) -> category counts
        """
        if isinstance(values, str):
            values = [values]

        counts: Dict[str, int] = defaultdict(int)
        to_lookup = {}  # term_norm -> raw value

        # 1. Lookup cache
        for value in values:
            if not value or not isinstance(value, str):
                continue

            term_norm = self.normalize(value)
            if term_norm in self.term2code:
                code = self.term2code[term_norm]
                if code:
                    cats = self.classify_code(code)
                    for cat in cats:
                        counts[cat] += 1
            else:
                to_lookup[term_norm] = value

        # 2. Parallel lookup for the remaining
        def lookup(term_norm, raw_value):
            cands = self.search_candidates(raw_value)
            if cands is None:
                return term_norm, None  # Network failure, do not cache
            return term_norm, cands[0][0] if cands else None

        if to_lookup:
            with ThreadPoolExecutor(max_workers=16) as ex:
                futures = [
                    ex.submit(lookup, tn, rv) for tn, rv in to_lookup.items()
                ]
                for f in as_completed(futures):
                    term_norm, code = f.result()
                    # Cache
                    self.term2code[term_norm] = code
                    # Classify if code found
                    if code:
                        cats = self.classify_code(code)
                        for cat in cats:
                            counts[cat] += 1

        return dict(counts)
