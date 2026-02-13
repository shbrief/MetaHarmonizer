from collections import defaultdict
from typing import List, Dict, Set
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np

def suggest_new_harmonized_fields(
    df: pd.DataFrame,
    unmapped_columns: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    eps: float = 0.3,
    min_samples: int = 2
) -> Dict[str, Set[str]]:
    """Cluster unmapped columns to suggest new harmonized fields (legacy DBSCAN approach).
    
    NOTE: This is a legacy implementation using DBSCAN clustering.
    For the current recommended approach, use FieldSuggester from src.field_suggester
    which combines NER + agglomerative clustering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Clinical metadata table
    unmapped_columns : List[str]
        Columns that could not be mapped to existing fields
    model_name : str
        Sentence transformer model for embeddings
    eps : float
        DBSCAN epsilon (max distance between points in a cluster)
    min_samples : int
        Minimum cluster size
        
    Returns
    -------
    Dict[str, Set[str]]
        Suggested new field names -> set of source columns
    """
    if not unmapped_columns:
        return {}
    
    # Create embeddings for unmapped columns
    model = SentenceTransformer(model_name)
    embeddings = model.encode(unmapped_columns)
    
    # Cluster similar column names
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    
    # Group columns by cluster
    clusters: Dict[int, Set[str]] = defaultdict(set)
    for col, label in zip(unmapped_columns, labels):
        if label != -1:  # Ignore noise points
            clusters[label].add(col)
    
    # Generate suggested field names from clusters
    suggestions = {}
    for cluster_id, columns in clusters.items():
        # Extract common terms from column names
        suggested_name = _generate_field_name(columns)
        suggestions[suggested_name] = columns
    
    return suggestions


def _generate_field_name(columns: Set[str]) -> str:
    """Generate a harmonized field name from clustered columns."""
    # Convert to lowercase and split into tokens
    all_tokens = []
    for col in columns:
        # Split on common delimiters
        tokens = col.lower().replace('_', ' ').replace('-', ' ').split()
        all_tokens.extend(tokens)
    
    # Find most common meaningful tokens
    from collections import Counter
    token_counts = Counter(all_tokens)
    
    # Remove generic tokens
    generic = {'stage', 'clinical', 'pathologic', 'value', 'type', 'code'}
    meaningful = [(tok, cnt) for tok, cnt in token_counts.items() 
                  if tok not in generic and cnt >= 2]
    
    if meaningful:
        # Sort by frequency
        meaningful.sort(key=lambda x: x[1], reverse=True)
        key_tokens = [tok for tok, _ in meaningful[:3]]
        return "_".join(key_tokens)
    
    # Fallback: use most common token
    most_common = token_counts.most_common(3)
    return "_".join([tok for tok, _ in most_common])


# Example usage
if __name__ == "__main__":
    # NOTE: This script uses a legacy DBSCAN-based approach.
    # For the current recommended approach, use FieldSuggester from src.field_suggester
    # which combines NER + agglomerative clustering instead of DBSCAN.
    
    from src.models.schema_mapper.engine import SchemaMapEngine
    
    # Load data
    df = pd.read_csv("data/schema/hnsc_tcga_gdc_clinical_data.tsv", sep = "\t")
    
    # Run schema mapping
    engine = SchemaMapEngine(
        clinical_data_path="data/schema/hnsc_tcga_gdc_clinical_data.tsv",
        mode="auto"
    )
    results_df = engine.run_schema_mapping()
    
    # Find columns that weren't confidently mapped
    unmapped = results_df[
        (results_df['match1_score'].isna()) | (results_df['match1_score'] < 0.5)
    ]['query'].tolist()
    
    print(f"Found {len(unmapped)} unmapped columns")
    print(f"Unmapped: {unmapped}\n")
    
    # Suggest new harmonized fields using legacy DBSCAN approach
    suggestions = suggest_new_harmonized_fields(df, unmapped)
    
    print("Suggested New Harmonized Fields:")
    for new_field, source_cols in suggestions.items():
        print(f"\n{new_field}:")
        for col in sorted(source_cols):
            print(f"  - {col}")
