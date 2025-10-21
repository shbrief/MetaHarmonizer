### 1553_values_with_context.csv
1553 disease-related values ('term' column) from cBioPortalData and 
concatenated values ('context' column) from all non-NA columns of samples/rows 
(only one row per value retrieved).

### concatenated_cols_for_189439_values.csv
Concatenated columns of all the samples. Exclude id and count-related columns
as well as NA column.

### config_disease.json
Disease-specific configuration file for `generalized_ontology_mapper` function

### disease_mapping_evaluation.csv
Disease-term ontology mapping results from MetaHarmonizer were evaluated by
Claude. [(report)](https://docs.google.com/document/d/1eygmJGqkQa5pMr0qNCPjMrhPyK1qIHhJIjRK6DsHdLY/edit?usp=sharing)

### disease_mapping_with_context.csv

### disease_mapping_without_context.xlsx

### embeddings.csv 
SapBERT embeddings of 673 column names in cBioPortalData that were harmonized

### RAG_vs_Lexical_Comparison.csv
Claude's evaluation on RAG vs. Lexicol mapping outputs

### st_sapbert_disease.csv
Mapping results from OntologyMapper

### st_sub_for_LLM_evaluation.csv
Subset of `st_sapbert_disease.csv` for LLM evaluation (without curated_ontology
and similarity score for matches)

### unique_values.txt
673 column names from cBioPortal data that were harmonized