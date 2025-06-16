from gevent import monkey

monkey.patch_all()

import csv
import os
from src.KnowledgeDb.faiss_sqlite_pipeline import FAISSSQLiteSearch

umls_api_key = os.getenv("UMLS_API_KEY")


def read_terms_from_csv(csv_path: str, term_column: str = "term") -> list:
    """Read terms from a specific column in a CSV file."""
    terms = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if term_column in row:
                term = row[term_column].strip()
                if term:
                    terms.append(term)
    return terms


def main():
    # === Configuration ===
    csv_path = "data/corpus/cbio_treatment_name/trt_name_query_for_NCIT:C1909.csv"
    # csv_path = "data/corpus/cbio_body_site/sample_50.csv"
    term_column_name = "curated_ontology"
    # options: pubmed-bert, mt-sap-bert, sap-bert
    method = "mt-sap-bert"

    print("Reading terms from CSV...")
    terms = read_terms_from_csv(csv_path, term_column=term_column_name)
    print(f"Found {len(terms)} terms.")

    table_name = f"{method.replace('-', '_')}_term_info"
    # table_name = f"{method.replace('-', '_')}_term_info_origin"
    index_path = f"src/KnowledgeDb/faiss_{method}.index"
    # index_path = f"src/KnowledgeDb/faiss_{method}_origin.index"
    db_path = "src/KnowledgeDb/vector_db.sqlite"

    print("Initializing FAISS + SQLite builder...")
    vector_builder = FAISSSQLiteSearch(db_path=db_path,
                                       index_path=index_path,
                                       table_name=table_name)

    print("Encoding terms and saving to vector DB...")
    vector_builder.fetch_and_store_terms(terms=terms,
                                         api_key=umls_api_key,
                                         method=method)

    print("Done. Vector database and SQLite have been updated.")


if __name__ == "__main__":
    main()
