## Select dynamic enum nodes ----------------------
library(OmicsMLRepoCuration)
dir <- "~/OmicsMLRepo/OmicsMLRepoData/cBioPortalData/maps"
bs_map_fpath <- file.path(dir, "cBioPortal_body_site_map.csv")
bs_map <- readr::read_csv(bs_map_fpath)
bs_map_long <- OmicsMLRepoR::getLongMetaTb(bs_map, 
                                           targetCols = c("curated_ontology", "curated_ontology_term_id"), 
                                           delim = "<;>")
bs_enums <- mapNodes(unique(bs_map_long$curated_ontology_term_id), cutoff = 0.5)
readr::write_csv(bs_enums, "bodysite_enums_0.5.csv") # save the dynamic enum node table

## Build a comprehensive corpus -------------
library(rols)
ol <- Ontologies()
ncit <- Ontology("ncit")
trm <- Term(ncit, "NCIT:C32221") # Body Part (covering 177 out of 369 terms)
allDesc <- descendants(trm)
allTerms <- termId(allDesc) %>% names

bs_corpus <- getOntoInfo(allTerms[1], ontology = "ncit", exact = TRUE)
for (i in 2:length(allTerms)) {
    print(i)
    res <- getOntoInfo(allTerms[i], ontology = "ncit", exact = TRUE)
    bs_corpus <- rbind(bs_corpus, res)
}

## Query terms for the selected corpus ------------------
q_terms <- bs_enums %>% 
    filter(ontology_term_id == "NCIT:C32221") %>% 
    pull(original_covered) %>% 
    strsplit(., ";") %>%
    unlist()

bs_map_sub <- bs_map %>%
    filter(curated_ontology_term_id %in% q_terms) # 591 out of 1671 original values

# all_query <- getOntoInfo(q_terms[1], ontology = "ncit", exact = TRUE)
# for (i in 2:length(q_terms)) {
#     print(i)
#     res <- getOntoInfo(q_terms[i], ontology = "ncit", exact = TRUE)
#     all_query <- rbind(all_query, res)
# }

## Save the comprehensive bodysite corpus and query
write.csv(bs_corpus, "bodysite_corpus_from_NCIT:C32221.csv", row.names = FALSE) 
write.csv(bs_map_sub, "bodysite_query_for_NCIT:C32221.csv", row.names = FALSE) 
