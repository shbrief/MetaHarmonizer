#' Build comprehensive corpus from the dynamic enum node
#' 
#' @import rols
#' @importFrom OmicsMLRepoR get_ontologies getOntoInfo
#' 
#' @param enum
#' @param save
#' @param targetDir 
#' 
build_corpus <- function(enum, save = FALSE, targetDir = "") {
    ol <- Ontologies()
    onto <- get_ontologies(enum)
    obj <- Ontology(onto)
    trm <- Term(object = obj, id = enum)
    allDesc <- descendants(trm)
    allTerms <- termId(allDesc) %>% names
    return(allTerms)
    
    corpus <- vector(mode = "character", length = length(allTerms))

    for (i in seq_along(allTerms)) {
        msg <- paste(i, "out of", length(allTerms))
        print(msg)
        corpus[[i]] <- getOntoInfo(allTerms[i],
                                   ontology = onto,
                                   exact = TRUE)$label
    }
    
    ## Save the comprehensive corpus
    if (isTRUE(save)) {
        fname <- paste0("corpus_from_", enum, ".csv")
        writeLines(corpus, file.path(targetDir, fname))
    } else {return(corpus)}
}