import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()


def cleanup_vector_store(om_strategy, method, category) -> None:
    """
    Drops the specified table and index from the SQLite database.
    
    Args:
        om_strategy (str): The ontology mapping strategy.
        method (str): The method name.
        category (str): The category name.
    """

    BASE_DB = os.getenv("VECTOR_DB_PATH")
    BASE_IDX_DIR = os.getenv("FAISS_INDEX_DIR")

    if BASE_DB is None or BASE_IDX_DIR is None:
        raise ValueError(
            "Missing required environment variables: VECTOR_DB_PATH or FAISS_INDEX_DIR"
        )

    db_path = BASE_DB
    index_path = os.path.join(BASE_IDX_DIR,
                              f"{om_strategy}_{method}_{category}.index")
    table_name = f"{om_strategy}_{method.replace('-', '_')}_{category}"

    drop_table(db_path, table_name)
    delete_index(index_path)


def drop_table(db_path: str, table_name: str) -> None:

    if not os.path.exists(db_path):
        print(f"[Warning] Database not found: {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"[Success] Table '{table_name}' dropped from {db_path}")
    except Exception as e:
        print(f"[Error] Failed to drop table '{table_name}': {e}")
    finally:
        conn.close()


def delete_index(index_path: str) -> None:
    """
    Deletes the FAISS index file if it exists.
    
    Args:
        index_path (str): The path to the FAISS index file.
    """
    if os.path.exists(index_path):
        try:
            os.remove(index_path)
            print(f"[Success] Index file '{index_path}' deleted.")
        except Exception as e:
            print(f"[Error] Failed to delete index file '{index_path}': {e}")
    else:
        print(f"[Warning] Index file '{index_path}' does not exist.")
