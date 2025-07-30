import pandas as pd
import asyncio
import os
from src.KnowledgeDb.db_clients.nci_db import NCIDb
from src.CustomLogger.custom_logger import CustomLogger

logger = CustomLogger().custlogger(loglevel='INFO')


def update_ncit_labels(
        input_path: str,
        ncit_code_column: str,
        label_column: str,
        save_csv: bool = False,
        output_path: str = "updated_with_official_label.csv") -> pd.DataFrame:
    """
    Append a new 'official_label' column using NCIT labels via UMLS API, without modifying original labels.
    
    Args:
        input_path (str): Path to the input CSV file.
        ncit_code_column (str): Column with NCIT codes (with or without 'NCIT:' prefix).
        label_column (str): Column name of the current (curated) label (kept unchanged).
        save_csv (bool): Whether to save updated dataframe as CSV.
        output_path (str): Path to save output CSV if save_csv is True.
        
    Returns:
        pd.DataFrame: DataFrame with an additional 'official_label' column.
    """
    # Load API key
    api_key = os.getenv("UMLS_API_KEY")
    if not api_key:
        raise ValueError("Please set the UMLS_API_KEY environment variable.")

    # Load CSV
    df = pd.read_csv(input_path)

    # Clean NCIT code
    df["__clean_code__"] = df[ncit_code_column].astype(str).str.replace(
        "NCIT:", "", regex=False).str.strip()
    codes = df["__clean_code__"].dropna().unique().tolist()

    # Fetch official labels
    nci_db = NCIDb(api_key)
    labels = asyncio.run(nci_db.get_labels_by_codes(codes))

    # Append new column without modifying the original
    df["official_label"] = df["__clean_code__"].map(labels)

    # Report mismatches
    if label_column in df.columns:
        mismatches = df[(df["official_label"].notna())
                        & (df["official_label"] != df[label_column])]
        mismatch_count = len(mismatches)
        total_count = len(df[df["official_label"].notna()])
        mismatch_ratio = (mismatch_count /
                          total_count) * 100 if total_count > 0 else 0.0
        logger.info(
            f"🟡 Found {mismatch_count} mismatches out of {total_count} ({mismatch_ratio:.2f}%)"
        )

    # Drop helper column
    df.drop(columns=["__clean_code__"], inplace=True, errors="ignore")

    # Save to file if needed
    if save_csv:
        df.to_csv(output_path, index=False)
        logger.info(f"✅ Saved with 'official_label' to {output_path}")

    return df
