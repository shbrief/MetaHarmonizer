#' Check Mapping of Original Fields to Curated Fields Using Similarity Matrix or Match Table
#'
#' This function checks whether original fields (and their top n most similar 
#' curated fields) from a similarity matrix or match table are present in the 
#' curated_field column of a schema table. It's useful for validating field 
#' mappings based on similarity scores between original data fields and 
#' standardized/curated field names.
#'
#' @param similarity A numeric matrix, data frame, or match table. Can be in two formats:
#'   \itemize{
#'     \item \strong{Matrix format}: Row names represent original field names and column 
#'           names represent curated/candidate field names. Each cell contains a 
#'           similarity score (e.g., string distance, cosine similarity). Higher 
#'           values indicate greater similarity.
#'     \item \strong{Long format}: A data frame with columns including 
#'           \code{original_column}, \code{match1_field}, \code{match1_score}, 
#'           \code{match2_field}, \code{match2_score}, etc. The function will 
#'           automatically detect this format.
#'   }
#'
#' @param schema A data frame or matrix containing the many-to-many mapping between
#'   curated fields and original fields. Typically contains columns like
#'   \code{curated_field} and \code{original_field}.
#'
#' @param top_n Integer specifying the number of top similar fields to return
#'   for each original field. Default is \code{1} (returns only the most similar field).
#'
#' @param schema_curated_col Character string or numeric index specifying the 
#'   column in \code{schema} that contains the curated field names to check against. 
#'   Default is \code{"curated_field"}.
#'
#' @param remove_na Logical indicating whether to remove rows with NA similarity
#'   values for all curated fields. Default is \code{TRUE}.
#'
#' @param similarity_threshold Numeric value specifying the minimum similarity score
#'   to consider a match valid. If NULL (default), no threshold is applied.
#'   Only matches with similarity >= this threshold will be included.
#'
#' @return A data frame with four columns:
#'   \itemize{
#'     \item \code{original_field}: The original field names (from row names of \code{similarity})
#'     \item \code{centroid}: The top similar curated field name (from column names)
#'     \item \code{similarity_score}: The similarity score between original and centroid fields
#'     \item \code{is_mapped_to_curated}: Logical indicating whether the centroid 
#'           exists in the curated fields column of \code{schema}
#'     \item \code{rank}: The rank of this match (1 = most similar, 2 = second most similar, etc.)
#'   }
#'   
#'   If \code{top_n > 1}, each original field will have multiple rows (up to \code{top_n} rows).
#'
#' @details
#' The function performs the following steps:
#' \enumerate{
#'   \item Detects the input format (matrix or long format)
#'   \item Converts long format to matrix format if necessary
#'   \item For each original field (row), identifies the top n most similar curated fields (columns)
#'   \item Retrieves the similarity scores for these top matches
#'   \item Checks if each top match exists in the specified curated field column of \code{schema}
#'   \item Applies similarity threshold filter if specified
#'   \item Returns a summary data frame with mapping validation results
#' }
#'
#' @examples
#' \dontrun{
#' # Example 1: Get the single best match for each original field (matrix format)
#' mapping_results <- check_mapping(res$similarity, schema, top_n = 1)
#' 
#' # Example 2: Get top 3 matches for each original field (long format)
#' mapping_results <- check_mapping(match_table, schema, top_n = 3)
#' 
#' # Example 3: Only consider matches with similarity >= 0.3
#' mapping_results <- check_mapping(res$similarity, schema, 
#'                                  top_n = 5, 
#'                                  similarity_threshold = 0.3)
#' 
#' # Example 4: Using column index for schema
#' mapping_results <- check_mapping(res$similarity, schema, 
#'                                  top_n = 2, 
#'                                  schema_curated_col = 1)
#' 
#' # View results
#' print(mapping_results)
#' 
#' # Summary statistics
#' cat("Total original fields:", length(unique(mapping_results$original_field)), "\n")
#' cat("Total matches:", nrow(mapping_results), "\n")
#' cat("Mapped to curated:", sum(mapping_results$is_mapped_to_curated), "\n")
#' 
#' # View best matches per field
#' best_matches <- subset(mapping_results, rank == 1)
#' print(best_matches)
#' }
#'
#' @export
check_mapping <- function(similarity, 
                          schema, 
                          top_n = 1,
                          schema_curated_col = "curated_field",
                          remove_na = TRUE,
                          similarity_threshold = NULL) {
    
    # Convert to data frame if matrix (preserve row/col names)
    if (is.matrix(similarity)) {
        similarity <- as.data.frame(similarity, stringsAsFactors = FALSE)
    }
    if (is.matrix(schema)) {
        schema <- as.data.frame(schema, stringsAsFactors = FALSE)
    }
    
    # Validate inputs
    if (!is.data.frame(similarity)) {
        stop("similarity must be a data frame or matrix")
    }
    if (!is.data.frame(schema)) {
        stop("schema must be a data frame or matrix")
    }
    
    # Detect format and convert if necessary
    is_long_format <- "original_column" %in% names(similarity) && 
        any(grepl("^match[0-9]+_field$", names(similarity)))
    
    if (is_long_format) {
        message("Detected long format input. Converting to matrix format...")
        similarity <- convert_long_to_matrix(similarity)
    }
    
    # Check for row names in similarity matrix
    if (is.null(rownames(similarity)) || all(rownames(similarity) == as.character(1:nrow(similarity)))) {
        stop("similarity matrix must have row names representing original field names")
    }
    
    # Check for column names in similarity matrix
    if (is.null(colnames(similarity)) || all(colnames(similarity) == paste0("V", 1:ncol(similarity)))) {
        stop("similarity matrix must have column names representing curated field names")
    }
    
    # Validate top_n
    if (!is.numeric(top_n) || length(top_n) != 1 || top_n < 1) {
        stop("top_n must be a positive integer")
    }
    top_n <- as.integer(top_n)
    
    if (top_n > ncol(similarity)) {
        warning(sprintf("top_n (%d) is greater than number of curated fields (%d). Using all fields.",
                        top_n, ncol(similarity)))
        top_n <- ncol(similarity)
    }
    
    # Validate similarity_threshold
    if (!is.null(similarity_threshold)) {
        if (!is.numeric(similarity_threshold) || length(similarity_threshold) != 1) {
            stop("similarity_threshold must be a single numeric value or NULL")
        }
    }
    
    # Validate schema column
    if (is.numeric(schema_curated_col)) {
        if (schema_curated_col > ncol(schema) || schema_curated_col < 1) {
            stop(sprintf("Column index %d is out of bounds for schema (ncol = %d)",
                         schema_curated_col, ncol(schema)))
        }
    } else if (!schema_curated_col %in% names(schema)) {
        stop(sprintf("Column '%s' not found in schema. Available columns: %s",
                     schema_curated_col, paste(names(schema), collapse = ", ")))
    }
    
    # Extract curated fields from schema
    curated_fields <- schema[[schema_curated_col]]
    
    # Get original field names from row names
    original_fields <- rownames(similarity)
    
    # Remove rows with all NA if requested
    if (remove_na) {
        non_na_rows <- apply(similarity, 1, function(x) !all(is.na(x)))
        if (!any(non_na_rows)) {
            warning("All rows have only NA values. Returning empty data frame.")
            return(data.frame(
                original_field = character(0),
                centroid = character(0),
                similarity_score = numeric(0),
                is_mapped_to_curated = logical(0),
                rank = integer(0),
                stringsAsFactors = FALSE
            ))
        }
        similarity <- similarity[non_na_rows, , drop = FALSE]
        original_fields <- original_fields[non_na_rows]
    }
    
    # Initialize result lists
    result_original <- character()
    result_centroid <- character()
    result_similarity <- numeric()
    result_rank <- integer()
    result_mapped <- logical()
    
    # For each original field, find top n most similar curated fields
    for (i in seq_along(original_fields)) {
        orig_field <- original_fields[i]
        similarity_row <- as.numeric(similarity[i, ])
        curated_names <- colnames(similarity)
        
        # Remove NA values for this row
        valid_idx <- !is.na(similarity_row)
        if (!any(valid_idx)) {
            next  # Skip this row if all similarities are NA
        }
        
        valid_similarities <- similarity_row[valid_idx]
        valid_names <- curated_names[valid_idx]
        
        # Sort by similarity (descending) and get top n
        sorted_idx <- order(valid_similarities, decreasing = TRUE)
        n_to_take <- min(top_n, length(sorted_idx))
        
        # Accuracy for top 1:n_to_take
        correct_results <- schema %>% filter(original_field == orig_field) %>% pull(curated_field)   
        is_mapped <- vector(mode = "logical", length = n_to_take)
        
        for (j in seq_len(n_to_take)) {
            top_idx <- sorted_idx[seq_len(j)]
            top_names <- valid_names[top_idx]
            top_scores <- valid_similarities[top_idx]
            is_mapped[j] <- any(top_names %in% correct_results)
        }
        
        
        # Apply similarity threshold if specified
        if (!is.null(similarity_threshold)) {
            threshold_mask <- top_scores >= similarity_threshold
            if (!any(threshold_mask)) {
                next  # Skip if no matches meet threshold
            }
            top_names <- top_names[threshold_mask]
            top_scores <- top_scores[threshold_mask]
            n_to_take <- length(top_names)
        }
        
        
        # Append results
        result_original <- c(result_original, rep(orig_field, n_to_take))
        result_centroid <- c(result_centroid, top_names)
        result_similarity <- c(result_similarity, top_scores)
        result_rank <- c(result_rank, 1:n_to_take)
        result_mapped <- c(result_mapped, is_mapped)
        
    }
    
    # Create result data frame
    results <- data.frame(
        original_field = result_original,
        centroid = result_centroid,
        similarity_score = result_similarity,
        is_mapped_to_curated = result_mapped,
        rank = result_rank,
        stringsAsFactors = FALSE
    )
    
    # Return empty data frame with correct structure if no results
    if (nrow(results) == 0) {
        warning("No valid mappings found. Returning empty data frame.")
        return(data.frame(
            original_field = character(0),
            centroid = character(0),
            similarity_score = numeric(0),
            is_mapped_to_curated = logical(0),
            rank = integer(0),
            stringsAsFactors = FALSE
        ))
    }
    
    return(results)
}


#' Convert Long Format Match Table to Similarity Matrix
#'
#' Internal helper function to convert a long-format match table with columns
#' like match1_field, match1_score, match2_field, match2_score, etc. into a
#' wide similarity matrix format.
#'
#' @param long_data A data frame in long format with columns including
#'   \code{original_column} and match columns (\code{match1_field}, 
#'   \code{match1_score}, \code{match2_field}, \code{match2_score}, etc.)
#'
#' @return A data frame in matrix format with original fields as row names
#'   and curated fields as column names, containing similarity scores.
#'
#' @keywords internal
convert_long_to_matrix <- function(long_data) {
    
    # Ensure dplyr is loaded
    if (!requireNamespace("dplyr", quietly = TRUE)) {
        stop("Package 'dplyr' is required for long format conversion. Please install it.")
    }
    if (!requireNamespace("tidyr", quietly = TRUE)) {
        stop("Package 'tidyr' is required for long format conversion. Please install it.")
    }
    
    # Extract match columns
    match_field_cols <- grep("^match[0-9]+_field$", names(long_data), value = TRUE)
    match_score_cols <- grep("^match[0-9]+_score$", names(long_data), value = TRUE)
    
    if (length(match_field_cols) == 0 || length(match_score_cols) == 0) {
        stop("Long format data must contain match*_field and match*_score columns")
    }
    
    # Create a list to store field-score pairs for each original column
    match_list <- list()
    
    for (i in seq_len(nrow(long_data))) {
        orig_col <- long_data$original_column[i]
        
        # Extract all non-empty matches for this row
        for (j in seq_along(match_field_cols)) {
            field_col <- match_field_cols[j]
            score_col <- match_score_cols[j]
            
            field_value <- long_data[[field_col]][i]
            score_value <- long_data[[score_col]][i]
            
            # Skip if field or score is NA or empty
            if (!is.na(field_value) && !is.na(score_value) && 
                field_value != "" && nchar(as.character(field_value)) > 0) {
                
                match_list[[length(match_list) + 1]] <- data.frame(
                    original_field = orig_col,
                    curated_field = field_value,
                    score = as.numeric(score_value),
                    stringsAsFactors = FALSE
                )
            }
        }
    }
    
    # Combine all matches
    if (length(match_list) == 0) {
        stop("No valid matches found in long format data")
    }
    
    matches_df <- do.call(rbind, match_list)
    
    # Get unique original and curated fields
    unique_original <- unique(matches_df$original_field)
    unique_curated <- unique(matches_df$curated_field)
    
    # Create matrix with appropriate dimensions
    sim_matrix <- matrix(
        NA,
        nrow = length(unique_original),
        ncol = length(unique_curated),
        dimnames = list(unique_original, unique_curated)
    )
    
    # Fill in the matrix with scores
    for (i in seq_len(nrow(matches_df))) {
        orig <- matches_df$original_field[i]
        curat <- matches_df$curated_field[i]
        score <- matches_df$score[i]
        sim_matrix[orig, curat] <- score
    }
    
    # Convert to data frame
    sim_df <- as.data.frame(sim_matrix, stringsAsFactors = FALSE)
    
    return(sim_df)
}