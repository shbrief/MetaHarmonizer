
# METAHARMONIZER DEMO USAGE EXAMPLES

## 1. SCHEMA MAPPING EXAMPLE
from src.Engine import SchemaMapEngine

# Map the demo clinical data columns
engine = SchemaMapEngine(
    path_to_input_table="demo_data/clinical_metadata_demo.tsv",
    mode="manual",  # Change to "auto" for automatic mapping
    top_k=5
)

results = engine.run_schema_mapping()
print("Schema mapping completed! Check the output CSV file.")

## 2. ONTOLOGY MAPPING EXAMPLES

import pandas as pd
from src.Engine import ontology_mapping_engine as ome

### 2a. Map Disease Terms
disease_queries = pd.read_csv('demo_data/disease_query_terms.csv')['disease_terms'].tolist()
disease_corpus = pd.read_csv('demo_data/disease_corpus_terms.csv')['disease_corpus'].tolist()

disease_engine = ome.OntoMapEngine(
    method='mt-sap-bert',
    category='disease',
    topk=10,
    query=disease_queries,
    corpus=disease_corpus,
    om_strategy='st',
    test_or_prod="prod"
)

disease_results = disease_engine.run()
print("Disease mapping completed!")

### 2b. Map Treatment Terms
treatment_queries = pd.read_csv('demo_data/treatment_query_terms.csv')['treatment_terms'].tolist()
treatment_corpus = pd.read_csv('demo_data/treatment_corpus_terms.csv')['treatment_corpus'].tolist()

treatment_engine = ome.OntoMapEngine(
    method='mt-sap-bert',
    category='treatment',
    topk=10,
    query=treatment_queries,
    corpus=treatment_corpus,
    om_strategy='st',
    test_or_prod="prod"
)

treatment_results = treatment_engine.run()
print("Treatment mapping completed!")

### 2c. Map Body Site Terms
body_site_queries = pd.read_csv('demo_data/body_site_query_terms.csv')['body_site_terms'].tolist()
body_site_corpus = pd.read_csv('demo_data/body_site_corpus_terms.csv')['body_site_corpus'].tolist()

body_site_engine = ome.OntoMapEngine(
    method='mt-sap-bert',
    category='body_site',
    topk=10,
    query=body_site_queries,
    corpus=body_site_corpus,
    om_strategy='st',
    test_or_prod="prod"
)

body_site_results = body_site_engine.run()
print("Body site mapping completed!")

## 3. WORKING WITH REAL DATA
# Extract unique values from your clinical data for ontology mapping

# Load the demo clinical data
clinical_data = pd.read_csv('demo_data/clinical_metadata_demo.tsv', sep='\t')

# Extract unique drug names for treatment mapping
unique_drugs = clinical_data['Treatment_Drug_Name'].dropna().unique().tolist()
print(f"Found {len(unique_drugs)} unique drug names to map")

# Extract unique tumor locations for body site mapping
unique_locations = clinical_data['Primary_Tumor_Location'].dropna().unique().tolist()
print(f"Found {len(unique_locations)} unique tumor locations to map")

# Now you can use these lists as queries in the ontology mapping engines
