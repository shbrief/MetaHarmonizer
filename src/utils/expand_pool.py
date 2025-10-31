import pandas as pd
import os, json, time
from collections import defaultdict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def expand_terms_with_gemini(input_terms, api_key=None, batch_size=50):
    """
    Expand terms using Gemini API.
    Fixed model: models/gemini-2.5-flash-preview-05-20
    Handles batching + simple JSON extraction.
    
    Args:
        input_terms (list[str]): Terms to expand
        api_key (str, optional): Gemini API key (if not set via env)
        batch_size (int, optional): Number of terms per API request

    Returns:
        list of dict: [{ "original_term": "...", "variants": [...] }]
    """
    # Load API key
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please provide API key or set GEMINI_API_KEY")

    genai.configure(api_key=api_key)
    print("✓ Gemini API configured")

    # Fixed model selection
    model_name = "models/gemini-2.5-flash-preview-05-20"
    model = genai.GenerativeModel(model_name)
    print(f"✓ Using model: {model_name}\n")

    all_expanded = []
    total_batches = (len(input_terms) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(input_terms), batch_size):
        batch = input_terms[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} terms)..."
        )

        # Prompt
        prompt = """You are a terminology expert. For each term, generate:
- Synonyms
- Abbreviations
- Common misspellings
- Alternative names (including layman/common forms)

Output ONLY a valid JSON array with this format:
[
  {
    "original_term": "...",
    "variants": ["...", "..."]
  }
]

Terms to expand:
"""
        for i, t in enumerate(batch, 1):
            prompt += f"{i}. {t}\n"
        prompt += "\nReturn ONLY the JSON array:"

        try:
            resp = model.generate_content(prompt)
            text = resp.text.strip()

            # Strip code fences if present
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()

            expanded = json.loads(text)
            all_expanded.extend(expanded)
            print(f"  ✓ Expanded {len(expanded)} terms")

            if batch_num < total_batches:
                time.sleep(2)

        except json.JSONDecodeError:
            print("  ✗ JSON parse error — skipping batch")
            print(f"  Raw response preview: {text[:200]}")
            continue

        except Exception as e:
            if "429" in str(e):
                print("  ✗ Rate limit — waiting 60s...")
                time.sleep(60)
                continue
            print(f"  ✗ Error: {e}")
            continue

    print(f"\n✓ Total expanded: {len(all_expanded)} terms")

    with open("llm_expanded_terms.json", "w", encoding="utf-8") as f:
        json.dump(all_expanded, f, indent=2, ensure_ascii=False)

    print("✓ Saved to llm_expanded_terms.json")

    return all_expanded


def create_variant_mapping(expanded_data, output_json='variant_mapping.json'):
    """
    Create variant -> canonical term mapping dictionary for direct lookup
    Input: expanded_data from LLM (list of dicts with 'original_term' and 'variants')
    Output: Dictionary where keys are variants and values are canonical terms
    """
    mapping = {}

    for item in expanded_data:
        canonical = item['original_term']

        # Add the original term to itself
        mapping[canonical.lower().strip()] = canonical

        # Add all variants
        for variant in item.get('variants', []):
            if variant and variant.strip():
                variant_key = variant.strip().lower()
                mapping[variant_key] = canonical

    # Save to JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Created mapping with {len(mapping)} variant -> canonical mappings")
    print(f"Saved to {output_json}")

    return mapping


def apply_variant_mapping(raw_queries, mapping_dict):
    """
    Match raw queries against variant mapping dictionary
    
    Args:
        raw_queries: list of query terms or pandas Series
        mapping_dict: variant -> canonical mapping
    
    Returns:
        DataFrame with columns: raw_query, matched_term, match_found
    """
    results = []

    for query in raw_queries:
        query_lower = str(query).lower().strip()
        matched = mapping_dict.get(query_lower)

        results.append({
            'raw_query': query,
            'matched_term': matched if matched else None,
            'match_found': matched is not None
        })

    results_df = pd.DataFrame(results)

    print(f"Total queries: {len(results_df)}")
    print(f"Matched: {results_df['match_found'].sum()}")
    print(f"Unmatched: {(~results_df['match_found']).sum()}")
    print(f"Match rate: {results_df['match_found'].mean():.2%}")

    return results_df
