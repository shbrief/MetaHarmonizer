## Select dynamic enum nodes ----------------------
dir <- "~/OmicsMLRepo/OmicsMLRepoData/cBioPortalData/maps"
disease_map_url <- file.path(dir, "cBioPortal_disease_map.csv")
disease_map <- readr::read_csv(disease_map_url)
disease_map_long <- OmicsMLRepoR::getLongMetaTb(
    disease_map,
    targetCols = c("curated_ontology", "curated_ontology_term_id"),
    delim = "<;>")

disease_enums <- mapNodes(unique(disease_map_long$curated_ontology_term_id), cutoff = 0.5)
readr::write_csv(disease_enums, "disease_enums_0.5.csv") # save the dynamic enum node table


## Build a comprehensive corpus -------------
library(rols)
ol <- Ontologies()
ncit <- Ontology("ncit")
trm <- Term(ncit, "NCIT:C3262") # covering 845 out of 1,152 curated terms
allDesc <- descendants(trm)
allTerms <- termId(allDesc) %>% names # 13,924 terms under this enum node

disease_corpus <- getOntoInfo(allTerms[1], ontology = "ncit", exact = TRUE)
for (i in 2:length(allTerms)) {
    print(i)
    res <- getOntoInfo(allTerms[i], ontology = "ncit", exact = TRUE)
    disease_corpus <- rbind(disease_corpus, res)
}

## Query terms for the selected corpus ------------------
q_terms <- disease_enums %>% 
    filter(ontology_term_id == "NCIT:C3262") %>% 
    pull(original_covered) %>% 
    strsplit(., ";") %>%
    unlist()

disease_map_sub <- disease_map %>%
    filter(curated_ontology_term_id %in% q_terms) # 1,562 out of 2,208 original values

## Save the comprehensive bodysite corpus and query
write.csv(disease_corpus, "disease_corpus_from_NCIT:C3262.csv", row.names = FALSE) 
write.csv(disease_map_sub, "disease_query_for_NCIT:C3262.csv", row.names = FALSE) 
