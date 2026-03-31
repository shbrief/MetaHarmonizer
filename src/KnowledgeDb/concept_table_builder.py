import os
import sqlite3
import asyncio
import httpx
from typing import Dict, List, Set
from pathlib import Path
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.KnowledgeDb.db_clients.ols_db import OLSDb, PREFIX_TO_ONTOLOGY
from src.CustomLogger.custom_logger import CustomLogger

BASE_DB = os.getenv("VECTOR_DB_PATH") or "src/KnowledgeDb/vector_db.sqlite"
UMLS_API_KEY = os.getenv("UMLS_API_KEY")


class ConceptTableBuilder:
    """
    Build synonym and RAG tables for a given category using NCI API data.
    """

    def __init__(self, category: str):
        self.category = category
        self.syn_table = f"synonym_{category}"
        self.rag_table = f"rag_{category}"
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

    @staticmethod
    def _parse_prefix(code: str) -> str | None:
        """Extract the ontology prefix from a code.

        Handles both formats:
          - 'EFO:0000249'  (colon-separated, from corpus CSV)
          - 'EFO_0000249'  (underscore-separated, from _obo_to_clean_code)
          - 'C12345'       (bare NCIT code, no prefix)
        """
        if ":" in code:
            return code.split(":", 1)[0]
        if "_" in code:
            return code.split("_", 1)[0]
        return None

    @staticmethod
    def _partition_codes(codes: List[str]) -> Dict[str, List[str]]:
        """Split codes into NCI vs OLS groups based on prefix.

        NCIT codes look like 'C12345' (bare, no prefix) or 'NCIT:C12345'.
        Everything with a recognized OLS prefix goes to OLS.
        """
        nci_codes = []
        ols_codes = []
        for code in codes:
            prefix = ConceptTableBuilder._parse_prefix(code)
            if prefix is None:
                # Bare code like 'C12345' — NCIT
                nci_codes.append(code)
            elif prefix == "NCIT":
                # Strip NCIT prefix for NCI API
                local = code.split(":", 1)[1] if ":" in code else code.split("_", 1)[1]
                nci_codes.append(local)
            elif prefix in PREFIX_TO_ONTOLOGY:
                ols_codes.append(code)
            else:
                # Unknown prefix — try NCI as fallback
                nci_codes.append(code)
        return {"nci": nci_codes, "ols": ols_codes}

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

        # ===== Route codes to NCI or OLS API =====
        partitioned = self._partition_codes(codes_to_fetch)
        nci_codes = partitioned["nci"]
        ols_codes = partitioned["ols"]

        concept_data = {}

        async with httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=max(
                        self.nci_db.concurrency * self.nci_db.batch_size,
                        self.ols_db.concurrency * self.ols_db.batch_size,
                    ))) as client:
            if nci_codes:
                self.logger.info(
                    f"Fetching {len(nci_codes)} NCIT codes via NCI API")
                nci_data = await self.nci_db.get_custom_concepts_by_codes(
                    nci_codes, client)
                for code, data in nci_data.items():
                    concept_data[code] = ("nci", data)

            if ols_codes:
                self.logger.info(
                    f"Fetching {len(ols_codes)} non-NCIT codes via OLS API")
                ols_data = await self.ols_db.get_custom_concepts_by_codes(
                    ols_codes, client)
                for code, data in ols_data.items():
                    concept_data[code] = ("ols", data)

        if not concept_data:
            self.logger.warning("No concept data fetched from NCI or OLS")
            return

        self.logger.info(
            f"Retrieved {len(concept_data)} concepts "
            f"(NCI: {len(nci_codes)}, OLS: {len(ols_codes)})")

        # ===== Prepare data for tables =====
        syn_records = []
        rag_records = []

        # Temporarily modify list_of_concepts for RAG (excluding synonyms)
        original_concepts = self.nci_db.list_of_concepts
        self.nci_db.list_of_concepts = [
            "definitions", "parents", "children", "roles"
        ]

        try:
            for code, (source, data) in concept_data.items():
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

                # 2. Build RAG records (context excludes synonyms)
                if source == "ols":
                    context = self.ols_db.create_context_list(data)
                else:
                    context = self.nci_db.create_context_list(data)
                combined = f"{official_label}: {context}"
                rag_records.append((official_label, code, combined))
        finally:
            # Restore list_of_concepts even if an error occurs above
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
