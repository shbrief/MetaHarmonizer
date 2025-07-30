import pandas as pd
import re

# Load curated metadata
df = pd.read_csv('data/cBioPortal_curated_metadata.csv')

# Identify all *_source columns
source_cols = [col for col in df.columns if col.endswith('_source')]

# Prepare result list
rows = []


# Normalize function for later deduplication
def normalize(text):
    return re.sub(r'\W+', '', str(text)).lower()


# Iterate over *_source columns
for col in source_cols:
    value_col = col.replace('_source', '')
    field_name = value_col.replace('curated_', '')

    # If the value_col doesn't exist, skip (for safety)
    if value_col not in df.columns:
        continue

    # Get the valid source values (remove NA and NaN)
    source_series = df[col].dropna()
    source_series = source_series[source_series != 'NA']

    # Collect source terms
    source_terms = set()
    for val in source_series:
        # Split and strip
        for s in re.split(r';|::|<;>', str(val)):
            s = s.strip()
            if s and s.upper() != 'NA':
                source_terms.add(s)

    # Always include self-mapping
    source_terms.add(field_name)

    for s in source_terms:
        rows.append({'field_name': field_name, 'source': s})

# Also handle curated fields that have no *_source column at all
curated_fields = [
    col.replace('curated_', '') for col in df.columns
    if col.startswith('curated_') and not col.endswith('_source')
]

# Find fields already included
existing_field_source_set = set(
    (normalize(r['field_name']), normalize(r['source'])) for r in rows)

# Ensure even missing source fields are included as self-mapping
for field in curated_fields:
    pair = (normalize(field), normalize(field))
    if pair not in existing_field_source_set:
        rows.append({'field_name': field, 'source': field})

# Convert to DataFrame and drop duplicates
df_expanded = pd.DataFrame(rows).drop_duplicates()

# Add flags
dict_df = pd.read_csv('data/cBioPortal_data_dictionary.csv')
numeric_df = pd.read_csv('data/numeric_fields_with_sources.csv')

dict_colnames = set(dict_df['ColName'].unique())
numeric_fields = set(numeric_df['numeric_field'].unique())

df_expanded['in_data_dictionary'] = df_expanded['field_name'].apply(
    lambda x: 'yes' if x in dict_colnames else 'no')

df_expanded['is_numeric_field'] = df_expanded['field_name'].apply(
    lambda x: 'yes' if x in numeric_fields else 'no')

# Save final result
df_expanded.to_csv('data/curated_fields_source_latest_with_flags.csv',
                   index=False)
print("✅ Done. Final shape:", df_expanded.shape)
print(df_expanded.head())
