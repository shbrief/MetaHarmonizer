import os
import sqlite3
import asyncio
import httpx
from typing import List, Set
from pathlib import Path
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.KnowledgeDb.db_clients.ols_db import OLSDb, validate_identifier
from src.CustomLogger.custom_logger import CustomLogger

BASE_DB = os.getenv("VECTOR_DB_PATH") or "src/KnowledgeDb/vector_db.sqlite"
UMLS_API_KEY = os.getenv("UMLS_API_KEY")


class ConceptTableBuilder:
    """Build synonym and RAG concept tables for a given (category, ontology_source).

    Supports both NCI EVSREST API (ontology_source='ncit') and EBI OLS4 API
    (non-ncit sources). Also supports offline building from pre-saved JSON
    via :meth:`build_from_json`.
    """

    def __init__(self, category: str, ontology_source: str = 'ncit'):
        self.category = validate_identifier(category, "category")
        self.ontology_source = validate_identifier(ontology_source, "ontology_source")
        self.syn_table = f"{ontology_source}_synonym_{category}"
        self.rag_table = f"{ontology_source}_rag_{category}"
        self.db_path = BASE_DB
        self.logger = CustomLogger().custlogger(loglevel="INFO")
        self.nci_db = NCIDb(UMLS_API_KEY)
        self.ols_db = OLSDb()

        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _ensure_tables(self):
        """Build synonym and RAG tables if they do not already exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Migrate legacy nci_code → code if needed
            self._migrate_nci_code_column(conn)

            # Synonym
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.syn_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    synonym TEXT NOT NULL,
                    official_label TEXT NOT NULL,
                    code TEXT NOT NULL
                );
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.syn_table}_code
                ON {self.syn_table}(code);
            """)

            # RAG (context)
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.rag_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    code TEXT NOT NULL UNIQUE,
                    context TEXT NOT NULL
                );
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.rag_table}_code 
                ON {self.rag_table}(code);
            """)

    def _migrate_nci_code_column(self, conn: sqlite3.Connection):
        """Rename legacy nci_code column to code if the old schema exists."""
        try:
            cols = [
                row[1]
                for row in conn.execute(
                    f"PRAGMA table_info({self.syn_table})").fetchall()
            ]
        except sqlite3.OperationalError:
            return  # table doesn't exist yet
        if "nci_code" in cols and "code" not in cols:
            self.logger.info(
                f"Migrating {self.syn_table}: renaming nci_code → code")
            conn.execute(
                f"ALTER TABLE {self.syn_table} RENAME COLUMN nci_code TO code")

    def get_stored_codes(self) -> Set[str]:
        """
        Get stored codes
        Check from either table (both should be in sync)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Get from rag table (because code is UNIQUE)
            rows = cursor.execute(
                f"SELECT DISTINCT code FROM {self.rag_table}").fetchall()
            return set(row[0] for row in rows)

    async def fetch_and_build_tables(self,
                                     codes: List[str],
                                     force_rebuild: bool = False):
        """
        Fetch data from NCI API and directly build synonym and rag tables.
        
        Args:
            codes: List of NCI codes
            force_rebuild: Whether to force rebuild (clear tables and refetch)
        """
        codes_set = set(codes)

        if force_rebuild:
            self.logger.info(
                f"🔄 Force rebuild: clearing tables and fetching all codes")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DELETE FROM {self.syn_table}")
                conn.execute(f"DELETE FROM {self.rag_table}")
            codes_to_fetch = list(codes_set)
        else:
            stored_codes = self.get_stored_codes()
            codes_to_fetch = list(codes_set - stored_codes)

            if not codes_to_fetch:
                self.logger.info(
                    f"✅ All {len(codes_set)} codes already in tables")
                return

            self.logger.info(
                f"📥 Fetching {len(codes_to_fetch)} new codes "
                f"(skipping {len(stored_codes & codes_set)} already stored)")

        # ===== Fetch via the API determined by ontology_source =====
        use_ols = self.ontology_source != "ncit"

        if use_ols:
            self.logger.info(
                f"Fetching {len(codes_to_fetch)} codes via OLS API "
                f"(ontology_source={self.ontology_source})")
            async with httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.ols_db.concurrency *
                        self.ols_db.batch_size)) as client:
                concept_data = await self.ols_db.get_custom_concepts_by_codes(
                    codes_to_fetch, client)
        else:
            self.logger.info(
                f"Fetching {len(codes_to_fetch)} codes via NCI API")
            async with httpx.AsyncClient(
                    limits=httpx.Limits(
                        max_connections=self.nci_db.concurrency *
                        self.nci_db.batch_size)) as client:
                concept_data = await self.nci_db.get_custom_concepts_by_codes(
                    codes_to_fetch, client)

        if not concept_data:
            self.logger.warning("No concept data fetched")
            return

        self.logger.info(f"Retrieved {len(concept_data)} concepts")

        # ===== Prepare data for tables =====
        syn_records = []
        rag_records = []

        # For NCIt RAG context: temporarily exclude synonyms
        if not use_ols:
            original_concepts = self.nci_db.list_of_concepts
            self.nci_db.list_of_concepts = [
                "definitions", "parents", "children", "roles"
            ]

        try:
            for code, data in concept_data.items():
                official_label = data.get('name', '').strip()
                if not official_label:
                    continue

                # 1. Extract Synonym records
                syn_set = {official_label}
                for syn_item in data.get('synonyms', []):
                    if isinstance(syn_item, dict):
                        syn_name = syn_item.get('name', '').strip()
                        if syn_name:
                            syn_set.add(syn_name)

                for syn in syn_set:
                    syn_records.append((syn, official_label, code))

                # 2. Build RAG records
                if use_ols:
                    context = self.ols_db.create_context_list(data)
                else:
                    context = self.nci_db.create_context_list(data)
                combined = f"{official_label}: {context}"
                rag_records.append((official_label, code, combined))
        finally:
            if not use_ols:
                self.nci_db.list_of_concepts = original_concepts
        # ===== Asynchronously batch insert into both tables =====
        def batch_insert():
            with sqlite3.connect(self.db_path) as conn:
                # Insert into synonym table
                conn.executemany(
                    f"INSERT INTO {self.syn_table} (synonym, official_label, code) VALUES (?, ?, ?)",
                    syn_records)

                # Insert into rag table
                conn.executemany(
                    f"INSERT OR IGNORE INTO {self.rag_table} (term, code, context) VALUES (?, ?, ?)",
                    rag_records)

                return len(syn_records), len(rag_records)

        loop = asyncio.get_event_loop()
        syn_count, rag_count = await loop.run_in_executor(None, batch_insert)

        self.logger.info(
            f"✅ Built tables: {syn_count} synonym records, {rag_count} rag records"
        )

    def build_from_json(
        self,
        json_path: str,
        force_rebuild: bool = False,
    ) -> None:
        """Read a CorpusBuilder JSON and populate synonym/RAG tables.

        This is the OLS offline path: a pre-saved JSON (produced by
        ``CorpusBuilder.save()``) is loaded directly — no external API calls.

        Parameters
        ----------
        json_path : str
            Path to the JSON file produced by ``CorpusBuilder.save()``.
        force_rebuild : bool
            If True, clear existing tables before inserting.
        """
        import json
        from pathlib import Path as _Path

        path = _Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus JSON not found: {json_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        terms = data.get("terms", [])
        self.logger.info(
            f"Loading {len(terms)} terms from {path.name} "
            f"into {self.syn_table} / {self.rag_table}"
        )

        if force_rebuild:
            self.logger.info("Force rebuild: clearing existing tables")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"DELETE FROM {self.syn_table}")
                conn.execute(f"DELETE FROM {self.rag_table}")
            stored_codes: set = set()
        else:
            stored_codes = self.get_stored_codes()

        syn_records: list = []
        rag_records: list = []

        for term in terms:
            label = (term.get("label") or "").strip()
            obo_id = (term.get("obo_id") or "").strip()
            description = (term.get("description") or "").strip()
            synonyms: list = term.get("synonyms") or []

            if not label or not obo_id:
                continue

            # Derive code from obo_id, matching _normalize_df logic:
            #   "NCIT:C156482"  → "C156482"       (strip NCIT prefix)
            #   "UBERON:0001062" → "UBERON_0001062" (keep non-NCIT prefix)
            if ":" in obo_id:
                prefix, local = obo_id.split(":", 1)
                code = local if prefix == "NCIT" else f"{prefix}_{local}"
            else:
                code = obo_id

            if code in stored_codes:
                continue

            # RAG context: mirrors NCIt format (definition + parents + children)
            parents: list = term.get("parents") or []
            children: list = term.get("children") or []

            parts = [f"{label}: {description}" if description else label]
            if parents:
                parts.append(f"parents: {', '.join(parents)}")
            if children:
                parts.append(f"children: {', '.join(children)}")
            context = ". ".join(parts)
            rag_records.append((label, code, context))

            # Synonym records
            syn_set = {label} | {s.strip() for s in synonyms if s.strip()}
            for syn in syn_set:
                syn_records.append((syn, label, code))

        if not rag_records:
            self.logger.info("No new terms to insert — tables already up to date")
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"INSERT OR IGNORE INTO {self.rag_table} "
                f"(term, code, context) VALUES (?, ?, ?)",
                rag_records,
            )
            conn.executemany(
                f"INSERT INTO {self.syn_table} "
                f"(synonym, official_label, code) VALUES (?, ?, ?)",
                syn_records,
            )

        self.logger.info(
            f"Inserted {len(rag_records)} RAG records and "
            f"{len(syn_records)} synonym records"
        )

    def get_stats(self) -> dict:
        """Get table statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            syn_count = cursor.execute(
                f"SELECT COUNT(*) FROM {self.syn_table}").fetchone()[0]

            rag_count = cursor.execute(
                f"SELECT COUNT(*) FROM {self.rag_table}").fetchone()[0]

            unique_codes = cursor.execute(
                f"SELECT COUNT(DISTINCT code) FROM {self.syn_table}"
            ).fetchone()[0]

        return {
            'synonym_records': syn_count,
            'rag_records': rag_count,
            'unique_codes': unique_codes
        }
