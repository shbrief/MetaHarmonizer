import pandas as pd
import asyncio
import os
from pathlib import Path
from src.KnowledgeDb.db_clients.nci_db import NCIDb

# Get UMLS API key from environment
UMLS_API_KEY = os.getenv("UMLS_API_KEY")
if not UMLS_API_KEY:
    raise ValueError("Please set the UMLS_API_KEY environment variable.")

# Resolve relative file path safely
BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data/corpus/cbio_disease/disease_corpus_from_NCIT:C3262.csv"
OUTPUT_PATH = BASE_DIR / "disease_corpus_updated.csv"

# Load data
df = pd.read_csv(INPUT_PATH)

# Initialize NCI API client
nci_db = NCIDb(UMLS_API_KEY)

# Clean NCI codes (remove "NCIT:" prefix)
df["clean_code"] = df["obo_id"].astype(str).str.replace(
    "NCIT:", "", regex=False).str.strip()
codes = df["clean_code"].dropna().unique().tolist()

# Fetch official labels from NCI
labels = asyncio.run(nci_db.get_labels_by_codes(codes))

# Map labels back to dataframe
df["official_label"] = df["clean_code"].map(labels)

# Update label if mismatched or missing
df["label"] = df["official_label"].combine_first(df["label"])

# Save cleaned version
df.drop(columns=["clean_code", "official_label"],
        errors="ignore").to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved updated file to {OUTPUT_PATH}")
