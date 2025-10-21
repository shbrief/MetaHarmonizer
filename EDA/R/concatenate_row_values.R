concatenate_row_values <- function(df, 
                                   separator = ":",
                                   collapse_char = ";") {
    
    result_list <- lapply(seq_len(nrow(df)), function(i) {
        row_data <- df[i, ]
        
        # Create column:value pairs
        pairs <- paste(names(df), row_data, sep = separator)
        
        # Keep only non-NA values (check original values, not the pairs)
        pairs <- pairs[!is.na(unlist(row_data))]
        
        # Collapse into single string
        if (length(pairs) > 0) {
            paste(pairs, collapse = collapse_char)
        } else {
            NA_character_
        }
    })
    
    names(result_list) <- paste0("row_", seq_len(nrow(df)))
    return(result_list)
}
