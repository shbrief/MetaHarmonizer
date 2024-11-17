## Select dynamic enum nodes ----------------------
dir <- "~/OmicsMLRepo/OmicsMLRepoData/cBioPortalData/maps"
trt_map_url <- file.path(dir, "cBioPortal_treatment_name_map.csv")
trt_map <- readr::read_csv(trt_map_url)
trt_enums <- mapNodes(unique(trt_map$curated_ontology_term_id), cutoff = 0.5)
readr::write_csv(trt_enums, "trt_name_enums_0.5.csv") # save the dynamic enum node table

## Build a comprehensive corpus -------------
library(rols)
ol <- Ontologies()
ncit <- Ontology("ncit")
trm <- Term(ncit, "NCIT:C1909") # covering 311 out of 557 curated terms
allDesc <- descendants(trm)
allTerms <- termId(allDesc) %>% names # 22,738 terms

trt_name_corpus <- getOntoInfo(allTerms[1], ontology = "ncit", exact = TRUE)
for (i in 2:length(allTerms)) {
    print(i)
    res <- getOntoInfo(allTerms[i], ontology = "ncit", exact = TRUE)
    trt_name_corpus <- rbind(trt_name_corpus, res)
}

## Query terms for the selected corpus ------------------
q_terms <- trt_enums %>% 
    filter(ontology_term_id == "NCIT:C1909") %>% 
    pull(original_covered) %>% 
    strsplit(., ";") %>%
    unlist()

trt_map_sub <- trt_map %>%
    filter(curated_ontology_term_id %in% q_terms) # 578 out of 918 original values

## Save the comprehensive bodysite corpus and query
write.csv(trt_name_corpus, "trt_name_corpus_from_NCIT:C1909.csv", row.names = FALSE) 
write.csv(trt_map_sub, "trt_name_query_for_NCIT:C1909.csv", row.names = FALSE) 
