## Select dynamic enum nodes ----------------------
library(OmicsMLRepoCuration)
bs_map_url <- "data/cBioPortal_bodysite_map.csv"
bs_map <- readr::read_csv(bs_map_url)
bs_map_long <- OmicsMLRepoR::getLongMetaTb(bs_map, 
                                           targetCols = c("curated_ontology", "curated_ontology_term_id"), 
                                           delim = "<;>")
bs_enums <- mapNodes(unique(bs_map_long$curated_ontology_term_id), cutoff = 0.5)
readr::write_csv(bs_enums, "bodysite_enums_0.5.csv") # save enum node table

## Build a comprehensive corpus -------------
library(rols)
ol <- Ontologies()
ncit <- Ontology("ncit")
trm <- Term(ncit, "NCIT:C32221") # Body Part (covering 177 out of 369 terms)
allDesc <- descendants(trm)
allTerms <- termId(allDesc) %>% names

bs_corpus <- vector(mode = "character", length = length(allTerms))

for (i in seq_along(allTerms)) {
    print(i)
    bs_corpus[[i]] <- getOntoInfo(allTerms[i], ontology = "ncit", exact = TRUE)$label
}

## Save the comprehensive bodysite corpus
writeLines(bs_corpus, "bodysite_corpus_from_BodyPart.csv") 

