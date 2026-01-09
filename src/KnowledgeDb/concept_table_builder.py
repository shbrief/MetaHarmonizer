import os
import sqlite3
import asyncio
import httpx
from typing import List, Set
from pathlib import Path
from src.KnowledgeDb.db_clients.nci_db import NCIDb
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

        Path(os.path.dirname(self.db_path)).mkdir(parents=True, exist_ok=True)
        self._ensure_tables()

    def _ensure_tables(self):
        """B synonym and rag tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Synonym
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.syn_table} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    synonym TEXT NOT NULL,
                    official_label TEXT NOT NULL,
                    nci_code TEXT NOT NULL
                );
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.syn_table}_code 
                ON {self.syn_table}(nci_code);
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

        # ===== Use NCI API to fetch data =====
        async with httpx.AsyncClient(
                limits=httpx.Limits(max_connections=self.nci_db.concurrency *
                                    self.nci_db.batch_size)) as client:
            concept_data = await self.nci_db.get_custom_concepts_by_codes(
                codes_to_fetch, client)

        if not concept_data:
            self.logger.warning("No concept data fetched from NCI")
            return

        self.logger.info(
            f"Retrieved {len(concept_data)} concepts from NCI API")

        # ===== Prepare data for tables =====
        syn_records = []
        rag_records = []

        # Temporarily modify list_of_concepts for RAG (excluding synonyms)
        original_concepts = self.nci_db.list_of_concepts
        self.nci_db.list_of_concepts = [
            "definitions", "parents", "children", "roles"
        ]

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

            # 2. Build RAG records (context excludes synonyms)
            context = self.nci_db.create_context_list(data)
            combined = f"{official_label}: {context}"
            rag_records.append((official_label, code, combined))

        # Restore list_of_concepts
        self.nci_db.list_of_concepts = original_concepts

        # ===== Asynchronously batch insert into both tables =====
        def batch_insert():
            with sqlite3.connect(self.db_path) as conn:
                # Insert into synonym table
                conn.executemany(
                    f"INSERT INTO {self.syn_table} (synonym, official_label, nci_code) VALUES (?, ?, ?)",
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
                f"SELECT COUNT(DISTINCT nci_code) FROM {self.syn_table}"
            ).fetchone()[0]

        return {
            'synonym_records': syn_count,
            'rag_records': rag_count,
            'unique_codes': unique_codes
        }
