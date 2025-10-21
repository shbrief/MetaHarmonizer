#!/usr/bin/env python3
"""
Generalized Ontology Mapper - Domain-agnostic version
Maps input terms to ontology labels using configurable strategies.

This version can be adapted to any domain (diseases, drugs, procedures, etc.)
by providing domain-specific configurations.
"""

import pandas as pd
import re
import json
from typing import List, Tuple, Dict, Optional, Set
import argparse


class OntologyMapper:
    """
    A generalized ontology mapping class that can be configured for different domains.
    """
    
    def __init__(self,
                 domain_terms: Optional[Dict[str, List[str]]] = None,
                 abbreviations: Optional[Dict[str, str]] = None,
                 context_fields: Optional[List[str]] = None,
                 stop_words: Optional[Set[str]] = None,
                 scoring_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the mapper with domain-specific configurations.
        
        Args:
            domain_terms: Dictionary mapping canonical terms to variants
                         Example: {'medication': ['medication', 'med', 'drug']}
            abbreviations: Dictionary mapping abbreviations to full terms
                          Example: {'ASA': 'acetylsalicylic acid'}
            context_fields: List of context field names to use for matching
                           Example: ['DRUG_CLASS', 'ROUTE', 'FORMULATION']
            stop_words: Set of words to ignore during keyword extraction
            scoring_weights: Dictionary of scoring weights for each strategy
        """
        self.domain_terms = domain_terms or {}
        self.abbreviations = abbreviations or {}
        self.context_fields = context_fields or []
        self.stop_words = stop_words or self._default_stop_words()
        self.scoring_weights = scoring_weights or self._default_weights()
    
    @staticmethod
    def _default_stop_words() -> Set[str]:
        """Default stop words that work across most domains."""
        return {
            'with', 'the', 'and', 'or', 'in', 'of', 'to', 'a', 'an', 'for', 
            'by', 'on', 'at', 'from', 'type', 'yes', 'no', 'na', 'other', 'nos'
        }
    
    @staticmethod
    def _default_weights() -> Dict[str, float]:
        """Default scoring weights."""
        return {
            'exact_match': 100.0,
            'substring_in_label': 50.0,
            'substring_in_input': 40.0,
            'keyword_overlap': 30.0,
            'context_overlap': 20.0,
            'domain_terms': 15.0,
            'anatomical_site': 10.0,
            'abbreviation_exact': 40.0,
            'abbreviation_keyword': 10.0
        }
    
    @classmethod
    def from_config(cls, config_path: str) -> 'OntologyMapper':
        """
        Create mapper from a JSON configuration file.
        
        Example config.json:
        {
            "domain_terms": {
                "carcinoma": ["carcinoma", "ca"]
            },
            "abbreviations": {
                "acc": "adrenocortical carcinoma"
            },
            "context_fields": ["CANCER_TYPE", "PRIMARY_SITE"],
            "stop_words": ["with", "the"],
            "scoring_weights": {
                "exact_match": 100.0,
                "keyword_overlap": 30.0
            }
        }
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            domain_terms=config.get('domain_terms'),
            abbreviations=config.get('abbreviations'),
            context_fields=config.get('context_fields'),
            stop_words=set(config.get('stop_words', [])) if config.get('stop_words') else None,
            scoring_weights=config.get('scoring_weights')
        )
    
    def parse_context(self, context_str: str) -> Dict[str, str]:
        """Parse context string into dictionary."""
        if pd.isna(context_str) or context_str == 'NA':
            return {}
        
        context_dict = {}
        pairs = str(context_str).split(';')
        for pair in pairs:
            if ':' in pair:
                key, value = pair.split(':', 1)
                context_dict[key.strip()] = value.strip()
        return context_dict
    
    def extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        if not text or pd.isna(text):
            return set()
        
        text = str(text).lower()
        words = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        keywords = set(w for w in words if len(w) > 2 and w not in self.stop_words)
        return keywords
    
    def normalize_term(self, term: str) -> str:
        """Normalize a term for comparison."""
        if pd.isna(term):
            return ""
        
        term = str(term).lower()
        term = re.sub(r'[^\w\s-]', ' ', term)
        term = re.sub(r'\s+', ' ', term).strip()
        return term
    
    def calculate_similarity_score(self, 
                                   input_term: str, 
                                   input_context: str, 
                                   ontology_label: str) -> float:
        """
        Calculate similarity score between input term and ontology label.
        Uses configurable strategies.
        """
        score = 0.0
        
        input_norm = self.normalize_term(input_term)
        label_norm = self.normalize_term(ontology_label)
        
        # Strategy 1: Exact match
        if input_norm == label_norm:
            return self.scoring_weights['exact_match']
        
        # Strategy 2: Substring containment
        if input_norm in label_norm:
            score += self.scoring_weights['substring_in_label']
        elif label_norm in input_norm:
            score += self.scoring_weights['substring_in_input']
        
        # Strategy 3: Keyword overlap
        input_keywords = self.extract_keywords(input_term)
        label_keywords = self.extract_keywords(ontology_label)
        
        if input_keywords and label_keywords:
            common_keywords = input_keywords & label_keywords
            overlap_ratio = len(common_keywords) / max(len(input_keywords), len(label_keywords))
            score += overlap_ratio * self.scoring_weights['keyword_overlap']
        
        # Strategy 4: Context-based scoring (if configured)
        if self.context_fields:
            context_dict = self.parse_context(input_context)
            
            # Extract values from specified context fields
            context_values = [context_dict.get(field, '') for field in self.context_fields]
            context_text = ' '.join(context_values).lower()
            context_keywords = self.extract_keywords(context_text)
            
            if context_keywords and label_keywords:
                context_overlap = label_keywords & context_keywords
                overlap_ratio = len(context_overlap) / len(label_keywords) if label_keywords else 0
                score += overlap_ratio * self.scoring_weights['context_overlap']
        
        # Strategy 5: Domain-specific term matching (if configured)
        if self.domain_terms:
            for canonical, variants in self.domain_terms.items():
                if canonical in label_norm:
                    for variant in variants:
                        if variant in input_norm:
                            score += self.scoring_weights['domain_terms']
                            break
        
        # Strategy 6: Abbreviation expansion (if configured)
        if self.abbreviations:
            input_lower = input_norm.replace(' ', '').replace('-', '')
            if input_lower in self.abbreviations:
                expanded = self.abbreviations[input_lower]
                if expanded in label_norm:
                    score += self.scoring_weights['abbreviation_exact']
                else:
                    expanded_keywords = self.extract_keywords(expanded)
                    overlap = expanded_keywords & label_keywords
                    if overlap:
                        score += len(overlap) * self.scoring_weights['abbreviation_keyword']
        
        return score
    
    def find_top_matches(self, 
                        input_term: str, 
                        input_context: str, 
                        ontology_labels: List[str], 
                        top_n: int = 5) -> List[Tuple[str, float]]:
        """Find top N matching ontology labels for an input term."""
        matches = []
        
        for label in ontology_labels:
            score = self.calculate_similarity_score(input_term, input_context, label)
            if score > 0:
                matches.append((label, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_n]
    
    def map_file(self,
                input_file: str,
                ontology_file: str,
                output_file: str,
                input_term_col: str = 'term',
                input_context_col: str = 'context',
                ontology_label_col: str = 'label',
                top_n: int = 5):
        """
        Map terms from input file to ontology labels and save results.
        
        Args:
            input_file: Path to CSV with terms to map
            ontology_file: Path to CSV with ontology labels
            output_file: Path to save mapping results
            input_term_col: Column name for input terms
            input_context_col: Column name for context (optional)
            ontology_label_col: Column name for ontology labels
            top_n: Number of top matches to return per term
        """
        print("Loading input data...")
        input_df = pd.read_csv(input_file)
        
        print("Loading ontology labels...")
        ontology_df = pd.read_csv(ontology_file)
        ontology_labels = ontology_df[ontology_label_col].tolist()
        
        print(f"Input terms: {len(input_df)}")
        print(f"Ontology labels: {len(ontology_labels)}")
        
        results = []
        
        for idx, row in input_df.iterrows():
            term = row[input_term_col]
            context = row.get(input_context_col, '') if input_context_col in row.index else ''
            
            print(f"\nProcessing {idx+1}/{len(input_df)}: {term}")
            
            top_matches = self.find_top_matches(term, context, ontology_labels, top_n)
            
            result = {
                'input_term': term,
                'has_context': 'Yes' if context and not pd.isna(context) and context != 'NA' else 'No'
            }
            
            for i, (label, score) in enumerate(top_matches, 1):
                result[f'match_{i}'] = label
                result[f'score_{i}'] = round(score, 2)
            
            for i in range(len(top_matches) + 1, top_n + 1):
                result[f'match_{i}'] = ''
                result[f'score_{i}'] = 0.0
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)
        
        print(f"\nResults saved to {output_file}")
        print(f"\n=== Mapping Summary ===")
        print(f"Total terms processed: {len(results_df)}")
        print(f"Terms with context: {sum(results_df['has_context'] == 'Yes')}")
        print(f"High-confidence matches (score > 50): {sum(results_df['score_1'] > 50)}")


def main():
    parser = argparse.ArgumentParser(description='Generalized Ontology Mapper')
    parser.add_argument('input_file', help='Input CSV file with terms to map')
    parser.add_argument('ontology_file', help='Ontology CSV file with labels')
    parser.add_argument('output_file', help='Output CSV file for results')
    parser.add_argument('--config', help='JSON configuration file', default=None)
    parser.add_argument('--input-term-col', default='term', help='Input term column name')
    parser.add_argument('--input-context-col', default='context', help='Context column name')
    parser.add_argument('--ontology-label-col', default='label', help='Ontology label column name')
    parser.add_argument('--top-n', type=int, default=5, help='Number of top matches to return')
    
    args = parser.parse_args()
    
    # Create mapper from config or use defaults
    if args.config:
        mapper = OntologyMapper.from_config(args.config)
    else:
        mapper = OntologyMapper()
    
    # Perform mapping
    mapper.map_file(
        input_file=args.input_file,
        ontology_file=args.ontology_file,
        output_file=args.output_file,
        input_term_col=args.input_term_col,
        input_context_col=args.input_context_col,
        ontology_label_col=args.ontology_label_col,
        top_n=args.top_n
    )


if __name__ == '__main__':
    main()
