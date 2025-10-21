#' Find First Match for Multiple Patterns in a Character Vector
#'
#' @description
#' Iteratively searches a character vector for multiple patterns and returns
#' the first match for each pattern. The function stops searching for each
#' pattern as soon as a match is found, making it efficient for large vectors.
#'
#' @param char_vector A character vector to search in.
#' @param patterns A character vector of patterns to search for. Can be plain
#'   text or regular expressions.
#' @param ignore_case Logical. Should pattern matching be case-insensitive?
#'   Default is \code{TRUE}.
#'
#' @return A data frame with two columns:
#'   \describe{
#'     \item{pattern}{The search pattern that was matched.}
#'     \item{matched_value}{The first value from \code{char_vector} that
#'       contains the pattern.}
#'   }
#'   Only patterns that have matches are included in the output. If no matches
#'   are found for any pattern, an empty data frame is returned.
#'
#' @details
#' The function uses a nested loop approach where it searches for each pattern
#' sequentially. For each pattern, it iterates through \code{char_vector} and
#' stops immediately upon finding the first match (using \code{break}), then
#' moves to the next pattern. This makes it significantly faster than
#' vectorized approaches when:
#' \itemize{
#'   \item Matches occur early in the vector
#'   \item The vector is very large
#'   \item Only the first match per pattern is needed
#' }
#'
#' Pattern matching is performed using \code{\link[base]{grepl}}, so both
#' plain text and regular expressions are supported.
#'
#' @seealso
#' \code{\link[base]{grepl}}, \code{\link[base]{grep}},
#' \code{\link[base]{regex}} for pattern matching syntax
#'
#' @examples
#' # Example 1: Basic usage
#' char_vector <- c(
#'   "The apple is red",
#'   "I like bananas",
#'   "Cherry blossoms are beautiful",
#'   "Dates are sweet",
#'   "Elderberries are tart"
#' )
#'
#' patterns <- c("apple", "banana", "fig", "elderberry")
#' results <- find_first_iterative(char_vector, patterns)
#' print(results)
#'
#' # Example 2: Case-sensitive search
#' results_case <- find_first_iterative(char_vector, c("Apple", "BANANA"),
#'                                      ignore_case = FALSE)
#' print(results_case)
#'
#' # Example 3: Using regular expressions
#' patterns_regex <- c("^The", "berr", "\\d+")
#' results_regex <- find_first_iterative(char_vector, patterns_regex)
#' print(results_regex)
#'
#' # Example 4: Large vector performance
#' large_vector <- c(rep("no match here", 10000),
#'                   "this has the keyword",
#'                   rep("more text", 10000))
#' result <- find_first_iterative(large_vector, "keyword")
#' # Stops at position 10001 instead of searching all 20001 elements
#'
#' @author Your Name
#'
#' @keywords character iteration pattern-matching
#'
#' @export
find_first_iterative <- function(char_vector, patterns,
                                 ignore_case = TRUE) {
    
    # Input validation
    if (!is.character(char_vector)) {
        stop("'char_vector' must be a character vector")
    }
    if (!is.character(patterns)) {
        stop("'patterns' must be a character vector")
    }
    if (!is.logical(ignore_case) || length(ignore_case) != 1) {
        stop("'ignore_case' must be a single logical value")
    }
    
    results <- data.frame(
        pattern = character(),
        matched_value = character(),
        stringsAsFactors = FALSE
    )
    
    for (pattern in patterns) {
        # Search for this pattern
        for (i in seq_along(char_vector)) {
            if (grepl(pattern, char_vector[i], ignore.case = ignore_case)) {
                results <- rbind(results, data.frame(
                    input = pattern,
                    context = char_vector[i],
                    stringsAsFactors = FALSE
                ))
                break  # Move to next pattern immediately
            }
        }
    }
    
    return(results)
}