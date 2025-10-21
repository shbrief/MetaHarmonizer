remove_id_count_na_columns <- function(df) {
    # Check if input is a data frame
    if (!is.data.frame(df)) {
        stop("Input must be a data frame")
    }
    
    # Identify columns to keep
    cols_to_keep <- sapply(names(df), function(col_name) {
        # Remove if column name contains "id" or "count" (case-insensitive)
        if (grepl("id|count", col_name, ignore.case = TRUE)) {
            return(FALSE)
        }
        
        # Remove if column is all NA
        if (all(is.na(df[[col_name]]))) {
            return(FALSE)
        }
        
        return(TRUE)
    })
    
    # Return filtered data frame
    return(df[, cols_to_keep, drop = FALSE])
}