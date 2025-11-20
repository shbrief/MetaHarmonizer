import sqlite3
import asyncio
import os
import math
from typing import List, Tuple, Optional, Set
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.CustomLogger.custom_logger import CustomLogger

BASE_DB = os.getenv("VECTOR_DB_PATH")
UMLS_API_KEY = os.getenv("UMLS_API_KEY")


class FTSSynonymDb:
    """
    SQLite FTS5-based synonym database.
    Simplified version without metadata table.
    """

    def __init__(self, category: str):
        self.category = category
        self.nci_db = NCIDb(UMLS_API_KEY)
        self.db_path = BASE_DB
        self.table_name = f"fts_synonyms_{category}"
        self.logger = CustomLogger().custlogger(loglevel='INFO')

        self.conn = None
        self._connect()
        self._create_table()

    def _connect(self):
        """Create or connect to SQLite database"""
        if self.conn is None:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info(f"Connected to FTS database: {self.db_path}")

    def _create_table(self):
        """Create FTS5 virtual table"""
        cursor = self.conn.cursor()
        cursor.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.table_name} USING fts5(
                synonym,
                standard_term,
                nci_code,
                tokenize='trigram'
            )
        ''')
        self.conn.commit()

    def get_indexed_codes(self) -> Set[str]:
        """
        Retrieve set of NCI codes already indexed in the FTS table.
        """
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT nci_code FROM {self.table_name}")

        return set(row[0] for row in cursor.fetchall())

    def index_exists(self) -> bool:
        """Check if FTS index has data"""
        cursor = self.conn.cursor()
        count = cursor.execute(
            f"SELECT COUNT(*) FROM {self.table_name}").fetchone()[0]
        return count > 0

    async def build_index_from_codes(self,
                                     codes: List[str],
                                     force_rebuild: bool = False):
        """
        Build FTS5 index from NCI codes.
        Only fetch codes that are not yet indexed.
        """
        codes_set = set(codes)

        if force_rebuild:
            self.logger.info(
                f"🔄 Force rebuild: fetching all {len(codes_set)} codes")
            cursor = self.conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            self.conn.commit()
            codes_to_fetch = list(codes_set)
        else:
            indexed_codes = self.get_indexed_codes()
            codes_to_fetch = list(codes_set - indexed_codes)

            if not codes_to_fetch:
                self.logger.info(
                    f"✅ All {len(codes_set)} codes already indexed. Skipping API calls."
                )
                return

            self.logger.info(
                f"📥 Fetching synonyms for {len(codes_to_fetch)} new codes "
                f"(skipping {len(indexed_codes & codes_set)} already indexed)")

        # Fetch concept data from NCI API
        concept_data = await self.nci_db.get_custom_concepts_by_codes(
            codes_to_fetch)

        if not concept_data:
            self.logger.warning("No concept data fetched from NCI")
            return

        # Extract synonyms and insert into FTS table
        cursor = self.conn.cursor()
        insert_count = 0

        for code, data in concept_data.items():
            if not data:
                continue

            standard_term = data.get("name", "").strip()
            if not standard_term:
                continue

            synonym_set = set()
            synonym_set.add(standard_term)

            if "synonyms" in data:
                for syn_item in data["synonyms"]:
                    if isinstance(syn_item, dict):
                        syn_name = syn_item.get("name", "").strip()
                        if syn_name:
                            synonym_set.add(syn_name)

            for synonym in synonym_set:
                cursor.execute(
                    f'INSERT INTO {self.table_name}(synonym, standard_term, nci_code) VALUES (?, ?, ?)',
                    (synonym, standard_term, code))
                insert_count += 1

        self.conn.commit()
        self.logger.info(
            f"✅ Built FTS index: {insert_count} synonyms from {len(concept_data)} codes"
        )

    def search(self,
               query: str,
               limit: int = 5) -> List[Tuple[str, str, float]]:
        """
        Search synonyms using FTS5 with BM25 ranking.
        
        Args:
            query: Query string to match
            limit: Maximum number of results
        
        Returns:
            List of (standard_term, nci_code, confidence_score) tuples
        """
        if not self.index_exists():
            self.logger.warning("FTS index is empty")
            return []

        cursor = self.conn.cursor()
        query_clean = query.strip()

        if not query_clean:
            return []

        try:
            results = cursor.execute(
                f'''
                SELECT standard_term, nci_code, rank
                FROM {self.table_name}
                WHERE {self.table_name} MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query_clean, limit)).fetchall()

            if results:
                return self._normalize_results(results)

        except sqlite3.OperationalError as e:
            try:
                results = cursor.execute(
                    f'''
                    SELECT standard_term, nci_code, rank
                    FROM {self.table_name}
                    WHERE {self.table_name} MATCH ?
                    ORDER BY rank
                    LIMIT ?
                ''', (f'"{query_clean}"', limit)).fetchall()

                if results:
                    return self._normalize_results(results)

            except sqlite3.OperationalError as e2:
                self.logger.warning(f"FTS search failed for '{query}': {e2}")

        return []

    def _normalize_results(
            self, results: List[Tuple]) -> List[Tuple[str, str, float]]:
        """
        Normalize BM25 scores using sigmoid-like mapping.
        """
        if not results:
            return []

        normalized = []
        for standard, code, rank in results:
            abs_rank = abs(rank)

            # confidence = e^(-abs_rank / scale)
            scale = 2.0
            confidence = math.exp(-abs_rank / scale)
            confidence = max(0.0, min(1.0, confidence))

            normalized.append((standard, code, confidence))

        return normalized

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
