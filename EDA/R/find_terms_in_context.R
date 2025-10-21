#' Find Terms in Context
#'
#' Searches for terms within a context vector and returns the first matching
#' context string for each term.
#'
#' @param context A character vector containing the text to search within.
#'   Each element represents a separate context (e.g., sentence, paragraph, or document).
#' @param terms A character vector of search terms or patterns to find.
#'   Supports regular expressions when using special regex characters.
#'
#' @return A data frame with two columns:
#'   \describe{
#'     \item{term}{The search term from the input}
#'     \item{context}{The first matching context string, or NA if no match found}
#'   }
#'
#' @details
#' The function performs case-insensitive pattern matching using Perl-compatible
#' regular expressions (PCRE). For each term, only the first matching context
#' is returned. If a term appears in multiple contexts, subsequent matches are
#' ignored.
#'
#' @examples
#' # Basic usage
#' context <- c("The quick brown fox", "jumps over the lazy dog", "The end")
#' terms <- c("fox", "lazy", "cat")
#' find_terms_in_context(context, terms)
#' #   term                 context
#' # 1  fox  The quick brown fox
#' # 2 lazy jumps over the lazy dog
#' # 3  cat                    <NA>
#'
#' # Using regex patterns
#' context <- c("Error: file not found", "Warning: low memory", "Info: success")
#' terms <- c("error", "warn.*memory", "debug")
#' find_terms_in_context(context, terms)
#'
#' @seealso \code{\link{grepl}}, \code{\link{grep}}
#'
#' @export
find_terms_in_context <- function(context, terms) {
    
    # Ensure context is a character vector
    context <- as.character(context)
    
    # Pre-compile regex patterns if case-insensitive matching is expensive
    matches <- vapply(terms, function(term) {
        match_idx <- which(grepl(term, context, ignore.case = TRUE, perl = TRUE))
        if (length(match_idx) > 0) {
            context[match_idx[1]]
        } else {
            NA_character_
        }
    }, character(1), USE.NAMES = FALSE)
    
    data.frame(
        term = terms,
        context = matches,
        stringsAsFactors = FALSE
    )
}