from src.models import ontology_mapper_st as oms
from src.models import ontology_mapper_lm as oml
from src.models import ontology_mapper_rag as omr
from src.models import ontology_mapper_synonym as omsyn
from src.models import ontology_mapper_bi_encoder as ombe
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from src.CustomLogger.custom_logger import CustomLogger
from src._paths import DATA_DIR, corpus_path
from src._async_utils import run_async
from src.KnowledgeDb.concept_table_builder import ConceptTableBuilder
from src.KnowledgeDb.corpus_builder import CorpusBuilder
from src.KnowledgeDb.db_clients.ols_db import OLSDb, validate_identifier

logger = CustomLogger()
ABBR_DICT_PATH = DATA_DIR / "corpus" / "oncotree_code_to_name.csv"
SYNONYM_MIN_CONFIDENCE = 0.9
MAX_RETRIEVED_CONTEXT_ITEMS = 10

# Known (category, ontology_source) → OBO root term ID.
# NCIt entries use the NCI EVSREST API for corpus + concept tables.
# Non-ncit entries use OLS4 API via CorpusBuilder.
_CORPUS_REGISTRY: dict[tuple[str, str], str] = {
    # NCIt
    ("bodysite",  "ncit"):   "NCIT:C32221",
    ("disease",   "ncit"):   "NCIT:C3262",
    ("treatment", "ncit"):   "NCIT:C1909",
    # OLS-based
    ("disease",   "mondo"):  "MONDO:0000001",
    ("bodysite",  "uberon"): "UBERON:0001062",
}

_BRITISH_TO_AMERICAN = [
    (r"(?i)leukaemia",   "leukemia"),
    (r"(?i)tumour",      "tumor"),
    (r"(?i)haemato",     "hemato"),
    (r"(?i)oedema",      "edema"),
    (r"(?i)paediatric",  "pediatric"),
    (r"(?i)foetal",      "fetal"),
    (r"(?i)anaemia",     "anemia"),
    (r"(?i)haemoglobin", "hemoglobin"),
    (r"(?i)gynaecolog",  "gynecolog"),
    (r"(?i)diarrhoea",   "diarrhea"),
]


class OntoMapEngine:
    """
    A class to initialize and run the OntoMapEngine for ontology mapping.

    Attributes:
        method (str): The name of the method.
        query (list[str]): The list of queries.
        corpus (list[str]): The list of corpus.
        cura_map (dict): The dictionary containing the mapping of queries to curated values.
        topk (int): The number of top matches to return.
        yaml_path (str): The path to the YAML file.
        om_strategy (str): The strategy to use for OntoMap.
        other_params (dict): Other parameters to pass to the engine.
        _test_or_prod (str): Indicates whether the environment is test or production.
        _logger (CustomLogger): Logger instance.
    """

    def __init__(self,
                 category: str,
                 query: list[str] = None,
                 cura_map: dict = None,
                 corpus: list[str] = None,
                 topk: int = 5,
                 s2_method: str = 'sap-bert',
                 s2_strategy: str = 'lm',
                 s3_method: str = 'pubmed-bert',
                 s3_strategy: str = None,
                 s3_threshold: float = 0.9,
                 s4_strategy: str = None,
                 s4_threshold: float = 0.6,
                 s4_model: str = 'gemma-12b',
                 output_dir: str = None,
                 **other_params: dict) -> None:
        """
        Initializes the OntoMapEngine class.

        Args:
            method (str): The name of the method.
            query (list[str]): The list of queries.
            corpus (list[str]): The list of corpus.
            cura_map (dict): The dictionary containing the mapping of queries to curated values.
            topk (int, optional): The number of top matches to return. Defaults to 5.
            s2_strategy (str, optional): The strategy to use for stage 2 OntoMap. Defaults to 'lm'. Options are 'st' or 'lm'.
            s3_strategy (str, optional): The strategy to use for stage 3 OntoMap. Defaults to None. Options are 'rag', 'rag_bie', or None.
            s3_threshold (float, optional): The threshold for stage 3 OntoMap. Defaults to 0.9.
            s4_strategy (str, optional): The strategy to use for stage 4 LLM rewriting. Defaults to None. Options are 'llm' or None.
            s4_threshold (float, optional): The threshold for stage 4. Defaults to 0.6.
            s4_model (str, optional): The LLM model key for stage 4. Defaults to 'gemma-12b'.
            **other_params (dict): Other parameters to pass to the engine.
        """
        self.s2_method = s2_method
        self.s3_method = s3_method
        self.category = validate_identifier(category, "category")
        self.output_dir = output_dir
        self.topk = topk
        self.s2_strategy = s2_strategy
        self.s3_strategy = s3_strategy
        self.s3_threshold = s3_threshold
        self.s4_strategy = s4_strategy
        self.s4_threshold = s4_threshold
        self.s4_model = s4_model
        self.other_params = other_params
        if 'test_or_prod' not in self.other_params.keys():
            raise ValueError(
                "test_or_prod value must be defined in other_params dictionary"
            )

        self._test_or_prod = self.other_params['test_or_prod']

        # --- Parse query_df / query_col from other_params ---
        query_df = self.other_params.get('query_df', None)
        query_col = self.other_params.get('query_col', None)

        if query_df is not None:
            if query_col is None:
                raise ValueError(
                    "query_col must be specified when passing query_df")
            if query_col not in query_df.columns:
                raise ValueError(
                    f"Column '{query_col}' not found in query_df. "
                    f"Available: {list(query_df.columns)}")
            self.query_df = query_df
            self.query_col = query_col
            # DataFrame mode: extract query list from specified column
            if query is None:
                raw = query_df[query_col].dropna().astype(str).str.strip()
                self.query = (raw[~raw.isin(["", "nan", "NaN", "None"])]
                              .unique().tolist())
            else:
                self.query = query
        else:
            self.query_df = None
            self.query_col = None
            self.query = query

        if self.query is None or len(self.query) == 0:
            raise ValueError(
                "No queries provided. Pass query (list) or "
                "query_df + query_col via other_params.")

        # cura_map: required in test mode, auto-generated in prod mode
        if cura_map is not None:
            self.cura_map = cura_map
        elif self._test_or_prod == 'test':
            raise ValueError("cura_map is required in test mode")
        else:
            self.cura_map = {q: "Not Found" for q in self.query}
        self._logger = logger.custlogger(loglevel='INFO')

        if self.s2_strategy not in ('lm', 'st'):
            raise ValueError("s2_strategy must be 'lm' or 'st'")
        if self.s3_strategy is not None and self.s3_strategy not in ('rag', 'rag_bie'):
            raise ValueError("s3_strategy must be 'rag', 'rag_bie', or None")
        if self.s4_strategy is not None and self.s4_strategy not in ('llm',):
            raise ValueError("s4_strategy must be 'llm' or None")

        # Ontology source validation (also guards f-string SQL in table names)
        self._ontology_source = validate_identifier(
            self.other_params.get("ontology_source", "ncit").lower(),
            "ontology_source"
        )

        # Resolve corpus_df: use provided value or auto-load from CSV
        corpus_df = self.other_params.get("corpus_df", None)
        corpus_df_provided = corpus_df is not None
        self._corpus_df_provided = corpus_df_provided

        if corpus_df is None:
            # Auto-resolve: registry must have the (category, ontology_source)
            if (self.category, self._ontology_source) not in _CORPUS_REGISTRY:
                supported = [
                    f"(category='{c}', ontology_source='{o}')"
                    for c, o in _CORPUS_REGISTRY
                ]
                raise ValueError(
                    f"Unsupported combination: category='{self.category}', "
                    f"ontology_source='{self._ontology_source}'. "
                    f"Supported: {supported}."
                )
            corpus_df = self._resolve_corpus_df()
            csv_src = corpus_path(self.category, self._ontology_source, "_corpus.csv")
            self._logger.info(
                f"Auto-loaded corpus_df ({len(corpus_df)} terms) from {csv_src}"
            )
        else:
            self._logger.info(
                f"Using caller-provided corpus_df ({len(corpus_df)} rows)."
            )
            self._validate_user_corpus(corpus_df)

        corpus_df = self._normalize_df(corpus_df, need_code=True)
        self.other_params["corpus_df"] = corpus_df

        # Content-hash suffix for user-uploaded corpus isolation.
        # If the user corpus matches the official corpus, skip the suffix
        # so pre-built tables are reused.
        if corpus_df_provided:
            from src.utils.corpus_hash import compute_corpus_hash
            self._corpus_hash = compute_corpus_hash(corpus_df)
            official_hash = self._compute_official_corpus_hash()
            if official_hash and self._corpus_hash == official_hash:
                self._table_suffix = ""
                self._logger.info(
                    f"User corpus matches official corpus (hash={self._corpus_hash}), "
                    f"reusing standard tables."
                )
            else:
                self._table_suffix = f"_{self._corpus_hash}"
                self._logger.info(f"User corpus hash: {self._corpus_hash}")
        else:
            self._corpus_hash = None
            self._table_suffix = ""

        # When corpus_df is caller-provided, infer ontology_source from codes
        if corpus_df_provided:
            codes = corpus_df["clean_code"].dropna().unique().tolist()
            if not codes:
                raise ValueError(
                    "User-provided corpus_df has no valid clean_code values "
                    "after normalization."
                )
            groups = self._partition_codes(codes)
            detected = sorted(groups.keys())
            if len(detected) != 1:
                raise ValueError(
                    f"User-provided corpus_df contains codes from multiple "
                    f"ontology sources: {detected}. All codes must share "
                    f"the same prefix. Separate your corpus by ontology source."
                )
            inferred = detected[0]
            if inferred != self._ontology_source:
                import warnings
                warnings.warn(
                    f"ontology_source overridden: '{self._ontology_source}' → "
                    f"'{inferred}' (inferred from corpus_df code prefixes)",
                    UserWarning,
                    stacklevel=2,
                )
                self._logger.warning(
                    f"Overriding ontology_source: "
                    f"'{self._ontology_source}' → '{inferred}' "
                    f"(inferred from corpus_df code prefixes)"
                )
                self._ontology_source = validate_identifier(
                    inferred, "ontology_source")
        self.corpus_s3 = corpus_df["official_label"].astype(
            str).unique().tolist()

        if corpus_df_provided and self.other_params.get("persist_corpus", False):
            self._persist_corpus_csv(corpus_df)

        # Resolve corpus (stage 2 term list)
        if corpus is not None:
            self.corpus = list(dict.fromkeys(corpus))
        else:
            self.corpus = list(dict.fromkeys(
                corpus_df["official_label"].astype(str).tolist()
            ))
        # Filter obsolete terms from corpus list (mirrors _normalize_df filter)
        self.corpus = [
            t for t in self.corpus
            if not t.strip().lower().startswith("obsolete_")
        ]

        self._ensure_concept_tables(corpus_df)

        self._logger.info("Initialized OntoMap Engine")
        self._logger.info(f"Stage 1: Exact matching")
        self._logger.info(f"Stage 2: {self.s2_strategy.upper()}")
        self._logger.info(
            f"Stage 2.5: Synonym matching (confidence>={SYNONYM_MIN_CONFIDENCE})"
        )

        if self.s3_strategy is not None:
            self._logger.info(
                f"Stage 3: {self.s3_strategy.upper()} (threshold={self.s3_threshold})"
            )
        else:
            self._logger.info("Stage 3: Disabled")

        if self.s4_strategy is not None:
            self._logger.info(
                f"Stage 4: {self.s4_strategy.upper()} (threshold={self.s4_threshold}, model={self.s4_model})"
            )
        else:
            self._logger.info("Stage 4: Disabled")

    def _compute_official_corpus_hash(self) -> str | None:
        """Compute the content hash of the official corpus CSV, if it exists."""
        from src.utils.corpus_hash import compute_corpus_hash
        csv_path = corpus_path(self.category, self._ontology_source, "_corpus.csv")
        if not csv_path.exists():
            return None
        try:
            official_df = self._normalize_df(
                pd.read_csv(csv_path), need_code=True)
            return compute_corpus_hash(official_df)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Corpus resolution
    # ------------------------------------------------------------------

    def _resolve_corpus_df(self) -> pd.DataFrame:
        """Load or build the corpus CSV for (category, ontology_source).

        Lookup order:
        1. CSV at standard path → pd.read_csv
        2. Build from API → save CSV → return DataFrame

        For ncit: uses NCI EVSREST ``/descendants`` endpoint.
        For non-ncit: uses ``CorpusBuilder`` (OLS4 API).
        """
        csv_path = corpus_path(self.category, self._ontology_source, "_corpus.csv")
        if csv_path.exists():
            self._logger.info(f"Loading corpus CSV: {csv_path}")
            return pd.read_csv(csv_path)

        root_term = _CORPUS_REGISTRY[(self.category, self._ontology_source)]

        if self._ontology_source == "ncit":
            df = self._build_ncit_corpus_csv(root_term)
        else:
            df = self._build_ols_corpus_csv(root_term)

        return df

    def _retrieved_json_path(self):
        return corpus_path(self.category, self._ontology_source, ".json")

    def _persist_corpus_csv(self, corpus_df: pd.DataFrame) -> None:
        """Persist corpus_df to the standard CSV cache path."""
        csv_path = corpus_path(self.category, self._ontology_source, "_corpus.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_df.to_csv(csv_path, index=False)
        self._logger.info(f"Saved corpus CSV ({len(corpus_df)} terms) to {csv_path}")

    def _get_retrieved_ontology_metadata(self, root_term: str) -> dict:
        """Fetch ontology/version metadata for the retrieved JSON envelope."""
        # TODO: Use ontology_version/version_iri to detect stale cached corpora
        # and implement incremental refresh/update behavior.
        if self._ontology_source == "ncit":
            from src.KnowledgeDb.db_clients.nci_db import NCIDb

            nci = NCIDb(os.getenv("UMLS_API_KEY"))
            return run_async(nci.get_ontology_metadata("ncit"))

        return run_async(OLSDb().get_ontology_metadata(self._ontology_source))

    @staticmethod
    def _extract_names(items: list, key: str = "name",
                       limit: int = MAX_RETRIEVED_CONTEXT_ITEMS,
                       dedupe: bool = False) -> list[str]:
        """Extract string values from a list of dicts, with optional dedup and limit."""
        result = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            value = str(item.get(key) or "").strip()
            if not value:
                continue
            if dedupe:
                if value in seen:
                    continue
                seen.add(value)
            result.append(value)
            if len(result) >= limit:
                break
        return result

    def _nci_concepts_to_records(self, concept_map: dict[str, dict]) -> list[dict]:
        """Convert NCIt concept payloads into retrieved-ontology JSON records."""
        records = []
        for code, concept in concept_map.items():
            label = str(concept.get("name") or "").strip()
            if not label:
                continue

            first_definition = ""
            for item in concept.get("definitions", []):
                if isinstance(item, dict):
                    definition = str(item.get("definition") or "").strip()
                    if definition:
                        first_definition = definition
                        break

            synonyms = self._extract_names(
                concept.get("synonyms", []), dedupe=True,
                limit=MAX_RETRIEVED_CONTEXT_ITEMS)
            parents = self._extract_names(concept.get("parents", []))
            children = self._extract_names(concept.get("children", []))

            roles = []
            for item in concept.get("roles", []):
                if not isinstance(item, dict):
                    continue
                role_type = str(item.get("type") or "").strip()
                related_name = str(item.get("relatedName") or "").strip()
                related_code = str(item.get("relatedCode") or "").strip()
                if role_type or related_name or related_code:
                    roles.append({
                        "type": role_type,
                        "related_name": related_name,
                        "related_code": related_code,
                    })
                if len(roles) >= MAX_RETRIEVED_CONTEXT_ITEMS:
                    break

            records.append({
                "iri": f"http://purl.obolibrary.org/obo/NCIT_{code}",
                "ontology_name": "ncit",
                "ontology_prefix": "NCIT",
                "short_form": f"NCIT_{code}",
                "label": label,
                "obo_id": f"NCIT:{code}",
                "definitions": [first_definition] if first_definition else [],
                "description": first_definition or None,
                "synonyms": synonyms,
                "parents": parents,
                "children": children,
                "roles": roles,
                "type": "class",
            })

        return records

    def _save_retrieved_ontology_json(self,
                                      records: list[dict],
                                      root_term: str) -> None:
        """Persist fetched ontology terms to the canonical JSON cache path."""
        json_path = self._retrieved_json_path()
        metadata_extra = self._get_retrieved_ontology_metadata(root_term)
        CorpusBuilder().save(records,
                             str(json_path),
                             root_term_id=root_term,
                             ontology=self._ontology_source,
                             metadata_extra=metadata_extra)
        self._logger.info(
            f"Saved retrieved ontology JSON ({len(records)} terms) to {json_path}"
        )

    def _build_ncit_corpus_csv(self, root_term: str) -> pd.DataFrame:
        """Fetch NCI descendants via EVSREST API and save corpus CSV + JSON.

        1. ``get_descendants`` → list of codes
        2. ``get_custom_concepts_by_codes`` → full concept data (single pass)
        3. Derive both CSV (label + obo_id) and rich JSON from the concept data
        """
        from src.KnowledgeDb.db_clients.nci_db import NCIDb

        code = root_term.split(":")[-1]  # "NCIT:C32221" → "C32221"
        self._logger.info(
            f"Building NCIt corpus for {root_term} via EVSREST API ..."
        )

        umls_key = os.getenv("UMLS_API_KEY")
        nci = NCIDb(umls_key)
        descendants = run_async(nci.get_descendants(code))

        if not descendants:
            raise RuntimeError(
                f"NCI EVSREST returned no descendants for {root_term}. "
                "Check the code and network connectivity."
            )

        # Include root concept (consistent with OLS which includes root by default)
        concept_codes = [code] + [d["code"] for d in descendants]
        self._logger.info(
            f"Fetching full concept data for {len(concept_codes)} codes ..."
        )
        concept_map = run_async(
            nci.get_custom_concepts_by_codes(concept_codes))

        # Derive CSV rows: root + descendants
        rows = []
        all_entries = [{"code": code, "name": ""}] + descendants
        seen_codes = set()
        for d in all_entries:
            c = d["code"]
            if c in seen_codes:
                continue
            seen_codes.add(c)
            concept = concept_map.get(c, {})
            label = concept.get("name", d["name"]).strip()
            if not label:
                continue
            rows.append({
                "iri": f"http://purl.obolibrary.org/obo/NCIT_{c}",
                "ontology_name": "ncit",
                "ontology_prefix": "NCIT",
                "short_form": f"NCIT_{c}",
                "description": "",
                "label": label,
                "obo_id": f"NCIT:{c}",
                "type": "class",
            })

        df = pd.DataFrame(rows)
        self._persist_corpus_csv(df)

        # Save rich JSON from the same concept data
        self._save_retrieved_ontology_json(
            self._nci_concepts_to_records(concept_map), root_term)
        return df

    def _build_ols_corpus_csv(self, root_term: str) -> pd.DataFrame:
        """Fetch OLS descendants via CorpusBuilder and save as corpus CSV."""
        self._logger.info(
            f"Building OLS corpus for {root_term} via CorpusBuilder ..."
        )
        builder = CorpusBuilder()
        records = builder.build_sync(root_term_id=root_term,
                                     include_hierarchy=True)
        if not records:
            raise RuntimeError(
                f"OLS returned no terms for root '{root_term}'. "
                "Check the term ID and network connectivity."
            )

        df = pd.DataFrame(records)
        # Keep columns consistent with NCIt CSV format
        for col in ("iri", "ontology_name", "ontology_prefix", "short_form",
                     "description", "label", "obo_id", "type"):
            if col not in df.columns:
                df[col] = ""
        df = df[["iri", "ontology_name", "ontology_prefix", "short_form",
                  "description", "label", "obo_id", "type"]]

        self._persist_corpus_csv(df)
        self._save_retrieved_ontology_json(records, root_term)
        return df

    # ------------------------------------------------------------------
    # Concept table building (RAG + synonym)
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_codes(codes: list[str]) -> dict[str, list[str]]:
        """Partition clean_codes by ontology source.

        Bare codes (e.g. ``C12345``) → ``'ncit'``.
        Prefixed codes (e.g. ``UBERON_0001062``) → looked up in
        ``PREFIX_TO_ONTOLOGY`` (exact match first, then upper-cased
        fallback); raises ``ValueError`` for unrecognised prefixes.
        """
        from src.KnowledgeDb.db_clients.ols_db import PREFIX_TO_ONTOLOGY

        groups: dict[str, list[str]] = {}
        unknown: list[str] = []
        for code in codes:
            if "_" in code:
                prefix = code.split("_", 1)[0]
                ont = (PREFIX_TO_ONTOLOGY.get(prefix)
                       or PREFIX_TO_ONTOLOGY.get(prefix.upper()))
                if ont is None:
                    unknown.append(code)
                    continue
            else:
                ont = "ncit"
            groups.setdefault(ont, []).append(code)
        if unknown:
            raise ValueError(
                f"Unknown ontology prefix in codes: {unknown[:5]}"
                f"{'...' if len(unknown) > 5 else ''}. "
                f"Supported prefixes: {sorted(PREFIX_TO_ONTOLOGY.keys())} "
                f"and bare NCI codes (e.g. C12345)."
            )
        return groups

    def _ensure_concept_tables(self, corpus_df: pd.DataFrame) -> None:
        """Ensure RAG and synonym SQLite tables are populated for all codes.

        Build strategy depends on ontology source and whether corpus was
        user-provided:

        - **NCIt** (always): fetch synonym + RAG context from NCI EVSREST API.
        - **OLS + official corpus**: build from local JSON (contains full
          synonym + context data from OLS API).
        - **OLS + user-provided corpus**: fetch from OLS API (same as NCIt).
        """
        import sqlite3
        from os import getenv

        all_codes = corpus_df["clean_code"].dropna().unique().tolist()
        if not all_codes:
            return

        db_path = getenv("VECTOR_DB_PATH") or "src/KnowledgeDb/vector_db.sqlite"
        groups = self._partition_codes(all_codes)

        for ont_source, codes in groups.items():
            ont_source = validate_identifier(ont_source, "ontology_source")
            rag_table = f"{ont_source}_rag_{self.category}{self._table_suffix}"
            codes_set = set(codes)

            # 1. Check if tables already have all codes
            try:
                with sqlite3.connect(db_path) as conn:
                    stored = {r[0] for r in conn.execute(
                        f"SELECT DISTINCT code FROM {rag_table}"
                    ).fetchall()}
            except sqlite3.OperationalError:
                stored = set()

            if codes_set.issubset(stored):
                self._logger.info(
                    f"Concept tables for '{ont_source}_{self.category}' "
                    f"are up-to-date ({len(codes_set)} codes); skipping"
                )
                continue

            # 2. Build tables
            builder = ConceptTableBuilder(self.category, ont_source,
                                          table_suffix=self._table_suffix)
            missing = list(codes_set - stored)

            # OLS + official corpus: build from local JSON (has synonym + context)
            if (ont_source != "ncit"
                    and not self._corpus_df_provided):
                json_path = corpus_path(self.category, ont_source, ".json")
                if json_path.exists():
                    self._logger.info(
                        f"Building concept tables from local JSON: {json_path}")
                    builder.build_from_json(str(json_path))
                    continue

            # NCIt, or OLS + user corpus: fetch from API
            self._logger.info(
                f"Fetching {len(missing)} missing codes via API "
                f"(ontology_source={ont_source})"
            )
            run_async(builder.fetch_and_build_tables(missing))

    @staticmethod
    def _validate_user_corpus(df: pd.DataFrame) -> None:
        """Validate that a user-provided corpus_df has the required columns.

        Must contain a label column (``official_label`` or ``label``) and
        a code column (``clean_code`` or ``obo_id``).
        """
        has_label = "official_label" in df.columns or "label" in df.columns
        has_code = "clean_code" in df.columns or "obo_id" in df.columns
        missing = []
        if not has_label:
            missing.append("'official_label' or 'label'")
        if not has_code:
            missing.append("'clean_code' or 'obo_id'")
        if missing:
            raise ValueError(
                f"User-provided corpus_df is missing required columns: "
                f"{' and '.join(missing)}. "
                f"Available columns: {list(df.columns)}"
            )

    def _normalize_df(self, df: pd.DataFrame, need_code: bool) -> pd.DataFrame:
        """
        Normalizes the input DataFrame by ensuring the presence of necessary columns and cleaning data.
        - official_label: if missing, attempts to use 'label' column as fallback;
        - clean_code: if missing, attempts to extract from 'obo_id' (format NCIT:C123456 → C123456);
        - If still missing, raises an error;
        - Removes null and duplicate values to ensure clean data.
        """
        df = df.copy()

        # official_label fallback
        if "official_label" not in df.columns:
            if "label" in df.columns:
                df["official_label"] = df["label"]
                self._logger.info(
                    "`official_label` not found — using `label` as fallback.")
            else:
                raise ValueError(
                    "DataFrame must contain 'official_label' or 'label'")

        # clean_code fallback
        if need_code:
            def _curie_to_clean_code(x: str) -> str:
                """Normalize CURIE-form codes to underscore form.

                ``NCIT:C156482`` → ``C156482`` (strip prefix for NCI endpoints)
                ``UBERON:0001062`` → ``UBERON_0001062`` (preserve non-NCIT prefixes)
                Already-clean codes (no ``:``) pass through unchanged.
                """
                x = str(x)
                if ":" not in x:
                    return x
                prefix, local = x.split(":", 1)
                if prefix == "NCIT":
                    return local
                return f"{prefix}_{local}"

            if "clean_code" not in df.columns:
                if "obo_id" in df.columns:
                    df["clean_code"] = df["obo_id"].astype(str).apply(_curie_to_clean_code)
                    self._logger.info(
                        "`clean_code` not found — generated from `obo_id`.")
                else:
                    raise ValueError(
                        "DataFrame must contain 'clean_code' or 'obo_id' for RAG/RAG_BIE strategies"
                    )
            else:
                # Normalize existing clean_code values that may be in CURIE form
                df["clean_code"] = df["clean_code"].astype(str).apply(_curie_to_clean_code)

        # Basic cleaning
        keep = ["official_label"] + (["clean_code"] if need_code else [])
        df = df.dropna(subset=keep).drop_duplicates(subset=keep)
        df["official_label"] = df["official_label"].astype(str)
        if "clean_code" in df.columns:
            df["clean_code"] = df["clean_code"].astype(str)

        # Filter out obsolete ontology terms (e.g. "obsolete_AIDS" in EFO)
        obs_mask = (df["official_label"].str.strip().str.lower()
                    .str.startswith("obsolete_"))
        n_obs = obs_mask.sum()
        if n_obs:
            df = df[~obs_mask].reset_index(drop=True)
            self._logger.info(
                f"Filtered {n_obs} obsolete_ terms from corpus_df")

        return df

    def _exact_matching(self):
        """
        Performs exact matching of queries to the corpus.

        Returns:
            list: The list of exact matches from the query.
        """
        corpus_normalized = {c.strip().lower() for c in self.corpus}
        return [
            q for q in self.query if q.strip().lower() in corpus_normalized
        ]

    def _map_shortname_to_fullname(self, non_exact_list: list[str]) -> dict:
        """
        Return a dict: original_value -> updated_value
        (short name (code) → full name if exists, otherwise keep original)
        """
        try:
            mapping_df = pd.read_csv(ABBR_DICT_PATH)
            short_to_name = dict(
                zip(mapping_df["code"].str.strip().str.replace("_", " ").str.lower(),
                    mapping_df["name"].str.strip()))
        except FileNotFoundError:
            self._logger.warning(
                "Abbreviation mapping file not found. Skipping abbreviation replacement."
            )
            short_to_name = {}

        replaced = {}
        for q in non_exact_list:
            q_strip = q.strip()
            q_key = q_strip.lower()
            replaced[q] = short_to_name.get(q_key, q_strip)
            if q_key in short_to_name:
                self._logger.info(
                    f"Replaced: {q_strip} → {short_to_name[q_key]}")
        return replaced
    
    def _normalize_query(self, q: str, corpus_set: set) -> str:
        """
        Apply lightweight text normalization to improve embedding recall.

        Steps applied in order:
        1. Underscore → space
        2. British → American spelling
        3. Plural stripping (only when singular form exists in corpus)
        """
        # 1. Underscore → space
        q = q.replace("_", " ")

        # 2. British → American spelling
        for pattern, replacement in _BRITISH_TO_AMERICAN:
            q = re.sub(pattern, replacement, q)

        # 3. Plural stripping: only strip trailing 's' when singular exists in corpus
        if q.lower().endswith('s') and len(q) > 3:
            singular = q[:-1]
            if singular.lower() in corpus_set:
                q = singular

        return q

    def _finalize_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Finalize results for output by renaming columns and dropping internal columns.
        
        Args:
            df (pd.DataFrame): The combined results DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame with user-facing column names.
        """
        df = df.copy()
        
        # Rename columns
        df = df.rename(columns={
            'original_value': 'query',
            'curated_ontology': 'ref_match'
        })
        
        # Drop internal columns
        columns_to_drop = [
            'updated_value',
            'top1_accuracy',
            'top3_accuracy', 
            'top5_accuracy'
        ]
        
        # Only drop columns that exist (avoid KeyError)
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            self._logger.info(f"Dropped internal columns: {existing_cols_to_drop}")

        return df

    def _recompute_match_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recompute match_level from self.cura_map[original_value] and match columns.

        Called after merging back to original_value so match_level is always
        consistent with the original curated label, not the normalized/expanded
        query key used inside the model.  Only has effect in test mode.
        """
        if self._test_or_prod != 'test':
            return df

        match_cols = [f"match{i}" for i in range(1, self.topk + 1)]

        def _level(row):
            curated = self.cura_map.get(row["original_value"])
            if not curated:
                return 99
            for i, col in enumerate(match_cols, start=1):
                if row.get(col) == curated:
                    return i
            return 99

        df = df.copy()
        df["match_level"] = df.apply(_level, axis=1)
        return df

    def _om_model_from_strategy(self, strategy: str,
                                non_exact_query_list: list[str]):
        """
        Returns the OntoMap model based on the strategy.

        Args:
            strategy (str): The strategy to use ('lm', 'st', 'rag', 'rag_bie').
            non_exact_query_list (list[str]): The list of non-exact query strings.

        Returns:
            object: The OntoMap model instance.
        """
        query_df = self.query_df
        corpus_df = self.other_params.get('corpus_df', None)
        use_reranker = self.other_params.get('use_reranker', True)
        reranker_method = self.other_params.get('reranker_method', 'minilm')
        reranker_topk = self.other_params.get('reranker_topk', 50)

        if strategy == 'lm':
            return oml.OntoMapLM(method=self.s2_method,
                                 category=self.category,
                                 ontology_source=self._ontology_source,
                                 om_strategy='lm',
                                 query=non_exact_query_list,
                                 corpus=self.corpus,
                                 topk=self.topk,
                                 table_suffix=self._table_suffix)

        elif strategy == 'st':
            return oms.OntoMapST(method=self.s2_method,
                                 category=self.category,
                                 ontology_source=self._ontology_source,
                                 om_strategy='st',
                                 query=non_exact_query_list,
                                 corpus=self.corpus,
                                 topk=self.topk,
                                 from_tokenizer=False,
                                 table_suffix=self._table_suffix)
        elif strategy == 'syn':
            return omsyn.OntoMapSynonym(method=self.s2_method,
                                        category=self.category,
                                 ontology_source=self._ontology_source,
                                        om_strategy='syn',
                                        query=non_exact_query_list,
                                        corpus=self.corpus,
                                        topk=self.topk,
                                        corpus_df=corpus_df,
                                        table_suffix=self._table_suffix)
        elif strategy == 'rag':
            return omr.OntoMapRAG(method=self.s3_method,
                                  category=self.category,
                                 ontology_source=self._ontology_source,
                                  om_strategy='rag',
                                  query=non_exact_query_list,
                                  corpus=self.corpus_s3,
                                  topk=self.topk,
                                  corpus_df=corpus_df,
                                  use_reranker=use_reranker,
                                  reranker_method=reranker_method,
                                  reranker_topk=reranker_topk,
                                  table_suffix=self._table_suffix)
        elif strategy == 'rag_bie':
            return ombe.OntoMapBIE(method=self.s3_method,
                                   category=self.category,
                                 ontology_source=self._ontology_source,
                                   om_strategy='rag_bie',
                                   query=non_exact_query_list,
                                   corpus=self.corpus_s3,
                                   topk=self.topk,
                                   query_df=query_df,
                                   corpus_df=corpus_df,
                                   query_col=self.query_col,
                                   table_suffix=self._table_suffix)
        else:
            raise ValueError(
                f"strategy should be 'st', 'lm', 'rag', or 'rag_bie', got '{strategy}'"
            )

    def _log_final_summary(self, exact_df, stage2_results, s3_res=None, s4_res=None):
        """Log final summary statistics.

        Args:
            exact_df: Stage 1 exact match results
            stage2_results: Stage 2/2.5 results (may be filtered if Stage 3 exists)
            s3_res: Stage 3 results (optional)
            s4_res: Stage 4 results (optional)
        """
        self._logger.info("=" * 50)
        self._logger.info("FINAL SUMMARY")
        self._logger.info("=" * 50)
        self._logger.info(f"Stage 1 (Exact): {len(exact_df)} queries")

        s2_only = len(stage2_results[stage2_results['stage'] == 2])
        self._logger.info(
            f"Stage 2 ({self.s2_strategy.upper()}): {s2_only} queries")

        s25_boosted = len(stage2_results[stage2_results['stage'] == 2.5])
        self._logger.info(
            f"Stage 2.5 (Synonym boost): {s25_boosted} queries")

        if s3_res is not None:
            self._logger.info(
                f"Stage 3 ({self.s3_strategy.upper()}): {len(s3_res)} queries")

        if s4_res is not None:
            self._logger.info(
                f"Stage 4 ({self.s4_strategy.upper()}): {len(s4_res)} queries")

    def _run_stage4(self, prior_res, s2_model):
        """Run Stage 4 LLM query rewriting on low-confidence results.

        Args:
            prior_res: DataFrame with prior stage results (must have 'top1_score_float')
            s2_model: The Stage 2 model object (for FAISS re-search)

        Returns:
            (s4_valid, queries_for_s4): improved rows DataFrame and query list,
            or (None, []) if Stage 4 is disabled or no queries need it.
        """
        if self.s4_strategy is None:
            return None, []

        self._logger.info(f"Stage 4: {self.s4_strategy.upper()} Query Rewriting")

        low_conf_mask = prior_res['top1_score_float'] < self.s4_threshold
        queries_for_s4 = prior_res.loc[low_conf_mask, 'original_value'].tolist()

        self._logger.info(
            f"Queries with match1_score < {self.s4_threshold}: {len(queries_for_s4)}"
        )

        if not queries_for_s4:
            self._logger.info("No queries require Stage 4.")
            return None, []

        from src.models.ontology_mapper_llm import OntoMapLLM

        s4_model = OntoMapLLM(
            category=self.category,
            s2_model=s2_model,
            query_df=self.query_df,
            query_col=self.query_col,
            topk=self.topk,
            model_key=self.s4_model,
        )
        s4_res = s4_model.get_match_results(
            queries=queries_for_s4,
            topk=self.topk,
        )
        # Add eval columns (same as other stages — engine's responsibility)
        s4_res['stage'] = 4.0
        s4_res['curated_ontology'] = s4_res['original_value'].map(
            self.cura_map).fillna("Not Found")
        s4_res = self._recompute_match_level(s4_res)

        # Drop rows where LLM returned no results (keep prior for those)
        s4_valid = s4_res[s4_res['match1'] != 'N/A'].copy()

        self._logger.info(
            f"Stage 4 completed: {len(queries_for_s4)} queries processed, "
            f"{len(s4_valid)} with results"
        )
        return s4_valid, queries_for_s4

    def _apply_synonym_boost(self, s2_res):
        """
        Applies synonym verification to low-confidence results in Stage 2.

        Args:
            s2_res (pd.DataFrame): The DataFrame containing Stage 2 results.

        Returns:
            None (modifies s2_res in place)
        """
        self._logger.info(
            "Stage 2.5: Synonym Verification for Low Confidence Results")

        s2_res['top1_score_float'] = pd.to_numeric(
            s2_res['match1_score'], errors='coerce').fillna(0)

        low_conf_mask = s2_res['top1_score_float'] < SYNONYM_MIN_CONFIDENCE
        low_conf_queries = s2_res.loc[low_conf_mask,
                                      'original_value'].tolist()

        self._logger.info(
            f"Found {len(low_conf_queries)} low-confidence queries "
            f"(score < {SYNONYM_MIN_CONFIDENCE}) for synonym verification")

        syn_boosted = []

        if low_conf_queries:
            syn_model = self._om_model_from_strategy(
                'syn', low_conf_queries)
            syn_results = syn_model.get_match_results(
                cura_map=self.cura_map,
                topk=self.topk,
                test_or_prod=self._test_or_prod)

            syn_dict = {}
            for _, syn_row in syn_results.iterrows():
                orig_val = syn_row['original_value']
                syn_dict[orig_val] = syn_row

            for idx, row in s2_res[low_conf_mask].iterrows():
                orig_val = row['original_value']
                curated = row['curated_ontology']

                combined_candidates = {}
                for i in range(1, self.topk + 1):
                    match = row[f'match{i}']
                    score = float(row[f'match{i}_score'])
                    if pd.notna(match) and match:
                        combined_candidates[match] = score

                syn_row = syn_dict.get(orig_val)

                if syn_row is not None:
                    for i in range(1, self.topk + 1):
                        match = syn_row[f'match{i}']
                        score = float(syn_row[f'match{i}_score'])
                        if pd.notna(match) and match:
                            if match in combined_candidates:
                                combined_candidates[match] = max(
                                    combined_candidates[match], score)
                            else:
                                combined_candidates[match] = score

                    if not combined_candidates:
                        self._logger.warning(
                            f"No valid candidates for '{orig_val}' in Stage 2 or Synonym results, skipping boost"
                        )
                        continue

                    sorted_candidates = sorted(combined_candidates.items(),
                                               key=lambda x: x[1],
                                               reverse=True)[:self.topk]

                    combined_matches = [
                        match for match, _ in sorted_candidates
                    ]
                    combined_scores = [
                        score for _, score in sorted_candidates
                    ]

                    while len(combined_matches) < self.topk:
                        combined_matches.append(None)
                        combined_scores.append(0.0)

                    match_level = next(
                        (i + 1 for i, term in enumerate(combined_matches)
                         if (term or "").strip().lower() == (curated or "").strip().lower()), 99)

                    old_top1_score = float(row['match1_score'])
                    new_top1_score = combined_scores[0]
                    old_match_level = int(row['match_level'])

                    boosted = (new_top1_score > old_top1_score
                               or match_level < old_match_level)

                    if boosted:
                        self._logger.info(
                            f"Boosted '{orig_val}': "
                            f"S2_top1={row['match1']}({old_top1_score:.3f}) → "
                            f"Combined_top1={combined_matches[0]}({new_top1_score:.3f}), "
                            f"match_level: {old_match_level} → {match_level}"
                        )

                        for i in range(1, self.topk + 1):
                            s2_res.at[idx,
                                      f'match{i}'] = combined_matches[i -
                                                                      1]
                            s2_res.at[
                                idx,
                                f'match{i}_score'] = f"{combined_scores[i - 1]:.4f}"

                        s2_res.at[idx, 'match_level'] = match_level
                        s2_res.at[idx, 'stage'] = 2.5
                        s2_res.at[idx, 'top1_score_float'] = new_top1_score
                        syn_boosted.append(orig_val)

        self._logger.info(
            f"Stage 2.5: Boosted {len(syn_boosted)} queries with synonyms")

    def _save_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Save result DataFrame to output_dir with a timestamp if output_dir is set."""
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            s2_method_clean = self.s2_method.replace('-', '_')
            parts = [f"om_{self._ontology_source}_{self.category}", f"s2_{self.s2_strategy}_{s2_method_clean}"]

            if self.s3_strategy is not None:
                s3_method_clean = self.s3_method.replace('-', '_')
                parts.append(f"s3_{self.s3_strategy}_{s3_method_clean}")
                if self.other_params.get('use_reranker', True):
                    reranker_method = self.other_params.get('reranker_method', 'minilm')
                    parts.append(f"reranker_{reranker_method}")

            if self.s4_strategy is not None:
                s4_model_clean = re.sub(r'[^A-Za-z0-9_]', '_', self.s4_model)
                parts.append(f"s4_{self.s4_strategy}_{s4_model_clean}")

            parts.append(ts)
            fname = "_".join(parts) + ".csv"
            path = os.path.join(self.output_dir, fname)
            df.to_csv(path, index=False)
            self._logger.info(f"Saved results to {path}")
        return df

    def run(self):
        """
        Runs the OntoMap Engine with multi-stage cascade.

        Returns:
            pd.DataFrame: A DataFrame containing results from all stages.
        """
        self._logger.info("=" * 50)
        self._logger.info("Starting Ontology Mapping")
        self._logger.info("=" * 50)

        # ========== Query Normalization ==========
        corpus_set = {c.strip().lower() for c in self.corpus}
        norm_map = {q: self._normalize_query(q, corpus_set) for q in self.query}
        self._logger.info("Query normalization applied")

        # ========== Stage 1: Exact Matching ==========
        self._logger.info("Stage 1: Exact Matching")
        exact_matches = self._exact_matching()
        self._logger.info(f"Exact matches: {len(exact_matches)}")

        stage1_matches = exact_matches

        # Create DataFrame for Stage 1 matches
        exact_df = pd.DataFrame({'original_value': stage1_matches})
        exact_df['curated_ontology'] = exact_df['original_value'].map(
            self.cura_map).fillna(exact_df['original_value'])
        exact_df['match_level'] = 1
        exact_df['stage'] = 1.0
        exact_df['match1'] = exact_df['curated_ontology']
        exact_df['match1_score'] = 1.00

        # Remaining queries for Stage 2
        non_exact_matches_ls = list(np.setdiff1d(self.query, stage1_matches))
        self._logger.info(
            f"Remaining for Stage 2: {len(non_exact_matches_ls)}")

        if not non_exact_matches_ls:
            self._logger.info(
                "No queries for Stage 2. Returning Stage 1 results.")
            return self._save_result(self._finalize_results(exact_df))

        # ========== Stage 2: LM/ST ==========
        self._logger.info(f"Stage 2: {self.s2_strategy.upper()} Matching")

        # Normalize non-exact queries, then apply shortname expansion
        norm_non_exact = [norm_map[q] for q in non_exact_matches_ls]
        self._logger.info("Replacing shortNames using rule-based name mapping")
        mapping_dict = self._map_shortname_to_fullname(norm_non_exact)
        updated_queries = [mapping_dict[nq] for nq in norm_non_exact]

        replace_df = pd.DataFrame({
            "original_value": non_exact_matches_ls,  # original query for cura_map lookup
            "updated_value": updated_queries           # normalized + shortname-expanded for embedding
        })

        updated_cura_map = {
            mapping_dict[norm_map[k]]: v
            for k, v in self.cura_map.items()
            if k in norm_map and norm_map[k] in mapping_dict
        }

        # Run Stage 2 model
        s2_model = self._om_model_from_strategy(self.s2_strategy,
                                                updated_queries)
        self._s2_model = s2_model  # Retain for Stage 4 re-search
        s2_res = s2_model.get_match_results(cura_map=updated_cura_map,
                                            topk=self.topk,
                                            test_or_prod=self._test_or_prod)

        # Merge back to original_value
        s2_res.rename(columns={"original_value": "updated_value"},
                      inplace=True)
        s2_res = pd.merge(replace_df, s2_res, on="updated_value", how="left")
        s2_res["curated_ontology"] = s2_res["original_value"].map(
            self.cura_map).fillna("Not Found")
        s2_res = self._recompute_match_level(s2_res)
        s2_res['stage'] = 2.0

        self._logger.info(f"Stage 2 completed: {len(s2_res)} queries")

        # ========== Stage 2.5: Synonym Verification ==========
        self._apply_synonym_boost(s2_res)

        # ========== Stage 3: RAG/RAG_BIE (Optional) ==========
        if self.s3_strategy is None:
            # No Stage 3, combine Stage 1 + Stage 2
            self._logger.info("Stage 3: Disabled")

            # ========== Stage 4: LLM Query Rewriting (Optional) ==========
            s4_valid, queries_for_s4 = self._run_stage4(s2_res, s2_model)

            if s4_valid is not None and not s4_valid.empty:
                s2_res_filtered = s2_res[
                    ~s2_res['original_value'].isin(
                        s4_valid['original_value'])
                ].copy()
                s2_res_filtered.drop(columns=['top1_score_float'], inplace=True)
                combined_results = pd.concat(
                    [exact_df, s2_res_filtered, s4_valid], ignore_index=True)
                self._log_final_summary(exact_df, s2_res_filtered, s4_res=s4_valid)
            else:
                s2_res.drop(columns=['top1_score_float'], inplace=True)
                combined_results = pd.concat([exact_df, s2_res], ignore_index=True)
                self._log_final_summary(exact_df, s2_res)

            return self._save_result(self._finalize_results(combined_results))

        else:
            # Check which queries need Stage 3 (top1_score < threshold)
            self._logger.info(f"Stage 3: {self.s3_strategy.upper()} Matching")

            top1_score_col = 'match1_score'
            if top1_score_col not in s2_res.columns:
                self._logger.warning(
                    f"{top1_score_col} not found in Stage 2 results. Skipping Stage 3."
                )
                s2_res.drop(columns=['top1_score_float'],
                            inplace=True,
                            errors='ignore')
                combined_results = pd.concat([exact_df, s2_res],
                                             ignore_index=True)
                return self._save_result(self._finalize_results(combined_results))

            # Identify low-confidence queries for Stage 3
            # Use existing top1_score_float column
            low_confidence_mask = s2_res[
                'top1_score_float'] < self.s3_threshold
            queries_for_s3 = s2_res.loc[low_confidence_mask,
                                        'original_value'].tolist()

            self._logger.info(
                f"Queries with match1_score < {self.s3_threshold}: {len(queries_for_s3)}"
            )

            if not queries_for_s3:
                self._logger.info("No queries require Stage 3.")

                # Drop temp column
                s2_res.drop(columns=['top1_score_float'], inplace=True)

                combined_results = pd.concat([exact_df, s2_res],
                                             ignore_index=True)

                self._log_final_summary(exact_df, s2_res)
                return self._save_result(self._finalize_results(combined_results))

            # Normalize then apply shortname replacement for Stage 3 queries
            norm_queries_s3 = [norm_map[q] for q in queries_for_s3]
            mapping_dict_s3 = self._map_shortname_to_fullname(norm_queries_s3)
            updated_queries_s3 = [mapping_dict_s3[nq] for nq in norm_queries_s3]

            replace_df_s3 = pd.DataFrame({
                "original_value": queries_for_s3,
                "updated_value": updated_queries_s3
            })

            updated_cura_map_s3 = {
                mapping_dict_s3[norm_map[k]]: v
                for k, v in self.cura_map.items()
                if k in norm_map and norm_map[k] in mapping_dict_s3
            }

            # Run Stage 3 model
            s3_model = self._om_model_from_strategy(self.s3_strategy,
                                                    updated_queries_s3)
            s3_res = s3_model.get_match_results(
                cura_map=updated_cura_map_s3,
                topk=self.topk,
                test_or_prod=self._test_or_prod)

            # Merge back to original_value
            s3_res.rename(columns={"original_value": "updated_value"},
                          inplace=True)
            s3_res = pd.merge(replace_df_s3,
                              s3_res,
                              on="updated_value",
                              how="left")
            s3_res["curated_ontology"] = s3_res["original_value"].map(
                self.cura_map).fillna("Not Found")
            s3_res = self._recompute_match_level(s3_res)
            s3_res['stage'] = 3.0

            self._logger.info(f"Stage 3 completed: {len(s3_res)} queries")

            # Remove Stage 2 results for queries that went to Stage 3
            s2_res_filtered = s2_res[~s2_res['original_value'].
                                     isin(queries_for_s3)].copy()

            # ========== Stage 4: LLM Query Rewriting (Optional) ==========
            # Build combined prior results (S2 filtered + S3) for threshold check
            # Need top1_score_float on S3 results for Stage 4 filtering
            s3_res['top1_score_float'] = pd.to_numeric(
                s3_res['match1_score'], errors='coerce').fillna(0)
            prior_for_s4 = pd.concat(
                [s2_res_filtered, s3_res], ignore_index=True)

            s4_valid, queries_for_s4 = self._run_stage4(
                prior_for_s4, s2_model)

            if s4_valid is not None and not s4_valid.empty:
                # Remove prior results for queries improved by S4
                prior_filtered = prior_for_s4[
                    ~prior_for_s4['original_value'].isin(
                        s4_valid['original_value'])
                ].copy()
                prior_filtered.drop(
                    columns=['top1_score_float'], inplace=True, errors='ignore')
                combined_results = pd.concat(
                    [exact_df, prior_filtered, s4_valid], ignore_index=True)

                # For summary: split prior_filtered back to s2/s3 counts
                s2_for_log = prior_filtered[
                    prior_filtered['stage'].isin([2.0, 2.5])]
                s3_for_log = prior_filtered[prior_filtered['stage'] == 3.0]
                self._log_final_summary(
                    exact_df, s2_for_log, s3_for_log, s4_valid)
            else:
                # Drop temp columns
                s2_res_filtered.drop(
                    columns=['top1_score_float'], inplace=True, errors='ignore')
                s3_res.drop(
                    columns=['top1_score_float'], inplace=True, errors='ignore')

                combined_results = pd.concat(
                    [exact_df, s2_res_filtered, s3_res], ignore_index=True)
                self._log_final_summary(exact_df, s2_res_filtered, s3_res)

            return self._save_result(self._finalize_results(combined_results))
