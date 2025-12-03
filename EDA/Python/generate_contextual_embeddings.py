import torch
from typing import List, Dict, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer


def create_combined_strings(
    terms: List[str],
    metadata_list: List[Dict],
    context_fields: Optional[List[str]] = None
) -> List[str]:
    """Create 'term: context' strings by concatenating metadata fields."""
    combined = []
    
    for term, metadata in zip(terms, metadata_list):
        # Get relevant fields
        fields = context_fields or metadata.keys()
        
        # Filter valid values and join
        context_parts = [
            str(metadata[f]) for f in fields 
            if f in metadata and pd.notna(metadata[f]) and str(metadata[f]).strip()
        ]
        context = " ".join(context_parts)
        
        # Combine term with context
        combined.append(f"{term}: {context}" if context else term)
    
    return combined


def generate_contextual_embeddings(
    terms: List[str],
    metadata_list: List[Dict],
    model: SentenceTransformer,
    context_fields: Optional[List[str]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Generate embeddings with context following CC's approach."""
    combined_strings = create_combined_strings(terms, metadata_list, context_fields)
    
    with torch.no_grad():
        embeddings = model.encode(
            combined_strings,
            convert_to_tensor=True,
            normalize_embeddings=normalize
        )
    
    return embeddings


# === Example Usage ===
if __name__ == "__main__":
    model = SentenceTransformer("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    
    unique_values = ["diabetes", "hypertension", "asthma"]
    metadata_list = [
        {"definition": "metabolic disorder", "category": "endocrine disease"},
        {"definition": "elevated blood pressure", "category": "cardiovascular disease"},
        {"definition": "chronic respiratory condition", "category": "respiratory disease"}
    ]
    
    # Generate embeddings with context
    embeddings = generate_contextual_embeddings(
        terms=unique_values,
        metadata_list=metadata_list,
        model=model,
        context_fields=["definition", "category"]
    )
    
    print(f"Shape: {embeddings.shape}")
    print(f"Example: '{create_combined_strings(unique_values[:1], metadata_list[:1])[0]}'")
