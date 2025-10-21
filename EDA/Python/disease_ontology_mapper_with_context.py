#!/usr/bin/env python3
"""
Context-aware ontology mapping for clinical cancer metadata.
Maps input terms to disease ontology labels using multiple strategies.
"""

import pandas as pd
import re
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

def parse_context(context_str):
    """Parse the context string into a dictionary of key-value pairs."""
    if pd.isna(context_str) or context_str == 'NA':
        return {}
    
    context_dict = {}
    pairs = context_str.split(';')
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':', 1)
            context_dict[key.strip()] = value.strip()
    return context_dict

def extract_keywords(text):
    """Extract meaningful keywords from text."""
    if not text or pd.isna(text):
        return set()
    
    # Convert to lowercase and split on non-alphanumeric characters
    text = str(text).lower()
    words = re.findall(r'\b[a-z][a-z0-9]*\b', text)
    
    # Filter out common stop words
    stop_words = {'with', 'the', 'and', 'or', 'in', 'of', 'to', 'a', 'an', 'for', 'by', 
                  'on', 'at', 'from', 'type', 'yes', 'no', 'na', 'other', 'nos'}
    
    keywords = set(w for w in words if len(w) > 2 and w not in stop_words)
    return keywords

def normalize_term(term):
    """Normalize a term for comparison."""
    if pd.isna(term):
        return ""
    
    # Convert to lowercase
    term = str(term).lower()
    
    # Remove special characters and extra spaces
    term = re.sub(r'[^\w\s-]', ' ', term)
    term = re.sub(r'\s+', ' ', term).strip()
    
    return term

def calculate_similarity_score(input_term, input_context, ontology_label):
    """
    Calculate similarity score between input term and ontology label.
    Uses multiple strategies including:
    - Direct substring matching
    - Keyword overlap
    - NO CONTEXT USED (for comparison purposes)
    """
    score = 0.0
    
    input_norm = normalize_term(input_term)
    label_norm = normalize_term(ontology_label)
    
    # Strategy 1: Exact match (highest priority)
    if input_norm == label_norm:
        return 100.0
    
    # Strategy 2: Substring containment
    if input_norm in label_norm:
        score += 50.0
    elif label_norm in input_norm:
        score += 40.0
    
    # Strategy 3: Keyword overlap
    input_keywords = extract_keywords(input_term)
    label_keywords = extract_keywords(ontology_label)
    
    if input_keywords and label_keywords:
        common_keywords = input_keywords & label_keywords
        keyword_score = len(common_keywords) / max(len(input_keywords), len(label_keywords)) * 30.0
        score += keyword_score
    
    # Strategy 4: Context-based scoring - DISABLED
    # (Not using context for this run)
    
    # Strategy 5: Common cancer term matching
    cancer_terms = {
        'adenocarcinoma': ['adenocarcinoma', 'adca'],
        'carcinoma': ['carcinoma', 'ca'],
        'leukemia': ['leukemia', 'leukaemia', 'aml', 'all', 'cml', 'cll'],
        'lymphoma': ['lymphoma'],
        'melanoma': ['melanoma'],
        'sarcoma': ['sarcoma'],
        'myeloma': ['myeloma'],
        'glioma': ['glioma'],
        'blastoma': ['blastoma'],
    }
    
    for canonical, variants in cancer_terms.items():
        if canonical in label_norm:
            for variant in variants:
                if variant in input_norm:
                    score += 15.0
                    break
    
    # Strategy 6: Anatomical site matching - DISABLED
    # (Not using context for this run)
    
    # Strategy 7: Abbreviation expansion
    abbreviations = {
        'acc': 'adrenocortical carcinoma',
        'acyc': 'adenoid cystic',
        'aml': 'acute myeloid leukemia',
        'all': 'acute lymphoblastic leukemia',
        'ucs': 'uterine carcinosarcoma',
        'um': 'uveal melanoma',
        'uvm': 'uveal melanoma',
    }
    
    input_lower = input_norm.replace(' ', '').replace('-', '')
    if input_lower in abbreviations:
        expanded = abbreviations[input_lower]
        if expanded in label_norm:
            score += 40.0
        else:
            expanded_keywords = extract_keywords(expanded)
            overlap = expanded_keywords & label_keywords
            if overlap:
                score += len(overlap) * 10.0
    
    return score

def find_top_matches(input_term, input_context, ontology_labels, top_n=5):
    """Find top N matching ontology labels for an input term."""
    matches = []
    
    for label in ontology_labels:
        score = calculate_similarity_score(input_term, input_context, label)
        if score > 0:  # Only include non-zero scores
            matches.append((label, score))
    
    # Sort by score (descending)
    matches.sort(key=lambda x: x[1], reverse=True)
    
    return matches[:top_n]

def main():
    # Load input data
    print("Loading input data...")
    input_df = pd.read_csv('/mnt/user-data/uploads/input_with_context.csv')
    
    # Load ontology labels
    print("Loading ontology labels...")
    ontology_df = pd.read_csv('/mnt/user-data/uploads/disease_corpus_label.csv')
    ontology_labels = ontology_df['x'].tolist()
    
    print(f"Input terms: {len(input_df)}")
    print(f"Ontology labels: {len(ontology_labels)}")
    
    # Process each input term
    results = []
    
    for idx, row in input_df.iterrows():
        term = row['term']
        context = row['context']
        
        print(f"\nProcessing {idx+1}/{len(input_df)}: {term}")
        
        # Find top matches
        top_matches = find_top_matches(term, context, ontology_labels, top_n=5)
        
        # Prepare result row
        result = {
            'input_term': term,
            'has_context': 'Yes' if not pd.isna(context) and context != 'NA' else 'No'
        }
        
        # Add top 5 matches
        for i, (label, score) in enumerate(top_matches, 1):
            result[f'match_{i}'] = label
            result[f'score_{i}'] = round(score, 2)
        
        # Fill empty slots if fewer than 5 matches
        for i in range(len(top_matches) + 1, 6):
            result[f'match_{i}'] = ''
            result[f'score_{i}'] = 0.0
        
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    output_path = '/mnt/user-data/outputs/ontology_mapping_results_no_context.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    # Print summary statistics
    print("\n=== Mapping Summary ===")
    print(f"Total terms processed: {len(results_df)}")
    print(f"Terms with context: {sum(results_df['has_context'] == 'Yes')}")
    print(f"Terms without context: {sum(results_df['has_context'] == 'No')}")
    
    # Count high-confidence matches (score > 30)
    high_conf = sum(results_df['score_1'] > 30)
    print(f"High-confidence matches (score > 30): {high_conf}")
    
    # Show some examples
    print("\n=== Sample Results ===")
    for idx in [0, 1, 2]:
        if idx < len(results_df):
            row = results_df.iloc[idx]
            print(f"\nInput: {row['input_term']}")
            print(f"Top match: {row['match_1']} (score: {row['score_1']})")

if __name__ == '__main__':
    main()
