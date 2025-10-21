library(dplyr)

#' Subset Data Based on Top N Match Columns
#'
#' This function filters a dataset by comparing a curated/reference column against
#' multiple match columns. It returns rows where the curated value either appears
#' in the top N matches ("matched") or does not appear in any of them ("not_matched").
#'
#' @param data A data frame containing the curated column and match columns
#' @param n Integer. The number of match columns to consider (default: 5).
#'   Match columns are identified by combining `match_prefix`, a number (1 to n),
#'   and `match_suffix`.
#' @param type Character. Either "matched" or "not_matched" (default: "not_matched").
#'   - "matched": Returns rows where the curated value appears in any of the n match columns
#'   - "not_matched": Returns rows where the curated value does not appear in any match columns
#'     (includes rows where curated value is NA)
#' @param curated_col Character. The name of the column containing the reference/curated
#'   values to compare against (default: "curated_field")
#' @param match_prefix Character. The prefix used in match column names (default: "match")
#' @param match_suffix Character. The suffix used in match column names (default: "_field")
#'
#' @return A data frame containing only the rows that meet the specified matching criteria
#'
#' @details
#' The function constructs match column names by concatenating:
#' `match_prefix` + number (1 to n) + `match_suffix`
#' 
#' For example, with defaults and n=3, it looks for columns:
#' "match1_field", "match2_field", "match3_field"
#'
#' Rows with NA in the curated column are always considered "not_matched".
#'
#' @examples
#' # Default usage (looks for columns: match1_field, match2_field, match3_field)
#' dat_not_matched <- subset_by_top_n(dat, n = 3, type = "not_matched")
#'
#' # Get rows where curated value appears in top 5 matches
#' dat_matched <- subset_by_top_n(
#'     data = dat,
#'     n = 5,
#'     type = "matched"
#' )
#'
#' # With custom column names (e.g., "best_match1", "best_match2", "best_match3")
#' dat_subset <- subset_by_top_n(
#'     data = dat,
#'     n = 3,
#'     type = "not_matched",
#'     curated_col = "reference_field",
#'     match_prefix = "best_match",
#'     match_suffix = ""
#' )
#'
#' # With pattern like "candidate_1_name", "candidate_2_name"
#' dat_subset <- subset_by_top_n(
#'     data = dat,
#'     n = 4,
#'     type = "matched",
#'     curated_col = "true_name",
#'     match_prefix = "candidate_",
#'     match_suffix = "_name"
#' )
#'
#' @export
subset_by_top_n <- function(data, 
                            n = 5, 
                            type = "not_matched",
                            curated_col = "curated_field",
                            match_prefix = "match",
                            match_suffix = "_field") {
    # Ensure n is at least 1
    if (n < 1) {
        stop("n must be at least 1")
    }
    
    # Validate type parameter
    if (!type %in% c("matched", "not_matched")) {
        stop("type must be either 'matched' or 'not_matched'")
    }
    
    # Check if curated column exists
    if (!curated_col %in% names(data)) {
        stop(paste("Column", curated_col, "not found in data"))
    }
    
    # Get the relevant match field column names
    match_cols <- paste0(match_prefix, 1:n, match_suffix)
    
    # Check which match columns exist
    existing_match_cols <- match_cols[match_cols %in% names(data)]
    
    if (length(existing_match_cols) == 0) {
        stop(paste("No match columns found with pattern:", 
                   paste0(match_prefix, "1-", n, match_suffix)))
    }
    
    if (length(existing_match_cols) < n) {
        warning(paste("Only", length(existing_match_cols), "of", n, 
                      "requested match columns found"))
    }
    
    # Filter rows based on type
    data_subset <- data %>%
        rowwise() %>%
        filter({
            curated <- .data[[curated_col]]
            matches <- c_across(all_of(existing_match_cols))
            
            if (is.na(curated)) {
                # NA rows are considered "not_matched"
                type == "not_matched"
            } else {
                is_matched <- curated %in% matches
                if (type == "matched") {
                    is_matched
                } else {
                    !is_matched
                }
            }
        }) %>%
        ungroup()
    
    return(data_subset)
}