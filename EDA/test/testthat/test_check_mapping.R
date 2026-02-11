library(testthat)
library(dplyr)
source("~/OmicsMLRepo/MetaHarmonizer/EDA/R/check_mapping.R")

# Test suite for check_mapping function
test_that("check_mapping handles basic valid input correctly", {
    # Create sample similarity matrix
    similarity <- matrix(
        c(0.9, 0.3, 0.1,
          0.2, 0.8, 0.4,
          0.1, 0.2, 0.95),
        nrow = 3, byrow = TRUE,
        dimnames = list(
            c("orig_field1", "orig_field2", "orig_field3"),
            c("curated_A", "curated_B", "curated_C")
        )
    )
    
    # Create sample schema
    schema <- data.frame(
        curated_field = c("curated_A", "curated_B", "curated_C", "curated_A"),
        original_field = c("orig_field1", "orig_field2", "orig_field3", "orig_field4"),
        stringsAsFactors = FALSE
    )
    
    # Run function with top_n = 1
    result <- check_mapping(similarity, schema, top_n = 1)
    
    # Assertions
    expect_s3_class(result, "data.frame")
    expect_equal(nrow(result), 3)
    expect_equal(ncol(result), 5)
    expect_true(all(c("original_field", "centroid", "similarity_score", 
                      "is_mapped_to_curated", "rank") %in% names(result)))
    expect_equal(result$original_field, c("orig_field1", "orig_field2", "orig_field3"))
    expect_equal(result$centroid, c("curated_A", "curated_B", "curated_C"))
    expect_equal(result$rank, c(1, 1, 1))
})

test_that("check_mapping returns top_n matches per original field", {
    similarity <- matrix(
        c(0.9, 0.7, 0.5, 0.3,
          0.8, 0.6, 0.4, 0.2),
        nrow = 2, byrow = TRUE,
        dimnames = list(
            c("field1", "field2"),
            c("cur_A", "cur_B", "cur_C", "cur_D")
        )
    )
    
    schema <- data.frame(
        curated_field = c("cur_A", "cur_B"),
        original_field = c("field1", "field2"),
        stringsAsFactors = FALSE
    )
    
    # Test with top_n = 3
    result <- check_mapping(similarity, schema, top_n = 3)
    
    expect_equal(nrow(result), 6)  # 2 fields × 3 matches each
    expect_equal(sum(result$original_field == "field1"), 3)
    expect_equal(sum(result$original_field == "field2"), 3)
    expect_true(all(result$rank %in% 1:3))
    
    # Check that ranks are correctly assigned
    field1_results <- result[result$original_field == "field1", ]
    expect_equal(field1_results$rank, c(1, 2, 3))
    expect_equal(field1_results$centroid, c("cur_A", "cur_B", "cur_C"))
})

test_that("check_mapping correctly identifies mapped fields", {
    similarity <- matrix(
        c(0.9, 0.3,
          0.2, 0.8),
        nrow = 2, byrow = TRUE,
        dimnames = list(
            c("orig1", "orig2"),
            c("cur_X", "cur_Y")
        )
    )
    
    schema <- data.frame(
        curated_field = c("cur_X", "cur_Z"),
        original_field = c("orig1", "orig2"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 1)
    
    # orig1 should map to cur_X (which is in schema)
    # orig2 should map to cur_Y (which is NOT in schema)
    expect_true(result$is_mapped_to_curated[1])
    expect_false(result$is_mapped_to_curated[2])
})

test_that("check_mapping applies similarity threshold correctly", {
    similarity <- matrix(
        c(0.9, 0.5, 0.2,
          0.8, 0.4, 0.1),
        nrow = 2, byrow = TRUE,
        dimnames = list(
            c("field1", "field2"),
            c("cur_A", "cur_B", "cur_C")
        )
    )
    
    schema <- data.frame(
        curated_field = c("cur_A", "cur_B", "cur_C"),
        original_field = c("field1", "field2", "field3"),
        stringsAsFactors = FALSE
    )
    
    # Apply threshold of 0.5
    result <- check_mapping(similarity, schema, top_n = 3, similarity_threshold = 0.5) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< debug
    
    # field1 should have 2 matches (0.9, 0.5), field2 should have 1 match (0.8)
    expect_equal(sum(result$original_field == "field1"), 2)
    expect_equal(sum(result$original_field == "field2"), 1)
    expect_true(all(result$similarity_score >= 0.5))
})

test_that("check_mapping handles NA values correctly with remove_na=TRUE", {
    similarity <- matrix(
        c(0.9, NA, 0.3,
          NA, NA, NA,
          0.5, 0.7, 0.2),
        nrow = 3, byrow = TRUE,
        dimnames = list(
            c("field1", "field2", "field3"),
            c("cur_A", "cur_B", "cur_C")
        )
    )
    
    schema <- data.frame(
        curated_field = c("cur_A", "cur_B"),
        original_field = c("field1", "field3"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 2, remove_na = TRUE)
    
    # field2 should be excluded (all NA)
    expect_false("field2" %in% result$original_field)
    expect_true("field1" %in% result$original_field)
    expect_true("field3" %in% result$original_field)
    
    # field1 should have 2 matches (ignoring NA)
    field1_results <- result[result$original_field == "field1", ]
    expect_equal(nrow(field1_results), 2)
    expect_false(any(is.na(field1_results$similarity_score)))
})

test_that("check_mapping handles NA values correctly with remove_na=FALSE", {
    similarity <- matrix(
        c(0.9, NA, 0.3),
        nrow = 1, byrow = TRUE,
        dimnames = list(
            c("field1"),
            c("cur_A", "cur_B", "cur_C")
        )
    )
    
    schema <- data.frame(
        curated_field = c("cur_A"),
        original_field = c("field1"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 2, remove_na = FALSE)
    
    # Should still get results but NA values are skipped
    expect_true("field1" %in% result$original_field)
    expect_false(any(is.na(result$similarity_score)))
})

test_that("check_mapping works with data.frame input", {
    similarity_df <- data.frame(
        cur_A = c(0.9, 0.2),
        cur_B = c(0.3, 0.8),
        row.names = c("field1", "field2"),
        stringsAsFactors = FALSE
    )
    
    schema <- data.frame(
        curated_field = c("cur_A", "cur_B"),
        original_field = c("field1", "field2"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity_df, schema, top_n = 1)
    
    expect_s3_class(result, "data.frame")
    expect_equal(nrow(result), 2)
    expect_equal(result$centroid, c("cur_A", "cur_B"))
})

test_that("check_mapping works with numeric column index for schema", {
    similarity <- matrix(
        c(0.9, 0.3),
        nrow = 1, byrow = TRUE,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    
    schema <- data.frame(
        curated_field = c("cur_A"),
        original_field = c("field1"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 1, schema_curated_col = 1)
    
    expect_s3_class(result, "data.frame")
    expect_equal(nrow(result), 1)
})

test_that("check_mapping throws error for invalid similarity input", {
    expect_error(
        check_mapping("not a matrix", data.frame(curated_field = "A")),
        "similarity must be a data frame or matrix"
    )
    
    expect_error(
        check_mapping(list(a = 1, b = 2), data.frame(curated_field = "A")),
        "similarity must be a data frame or matrix"
    )
})

test_that("check_mapping throws error for invalid schema input", {
    similarity <- matrix(
        c(0.9),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A"))
    )
    
    expect_error(
        check_mapping(similarity, "not a dataframe"),
        "schema must be a data frame or matrix"
    )
})

test_that("check_mapping throws error for missing row names", {
    similarity <- matrix(c(0.9, 0.3), nrow = 1)
    colnames(similarity) <- c("cur_A", "cur_B")
    
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_error(
        check_mapping(similarity, schema),
        "similarity matrix must have row names representing original field names"
    )
})

test_that("check_mapping throws error for missing column names", {
    similarity <- matrix(c(0.9, 0.3), nrow = 1)
    rownames(similarity) <- c("field1")
    
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_error(
        check_mapping(similarity, schema),
        "similarity matrix must have column names representing curated field names"
    )
})

test_that("check_mapping throws error for invalid top_n", {
    similarity <- matrix(
        c(0.9),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A"))
    )
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_error(
        check_mapping(similarity, schema, top_n = -1),
        "top_n must be a positive integer"
    )
    
    expect_error(
        check_mapping(similarity, schema, top_n = "not a number"),
        "top_n must be a positive integer"
    )
    
    expect_error(
        check_mapping(similarity, schema, top_n = c(1, 2)),
        "top_n must be a positive integer"
    )
})

test_that("check_mapping warns when top_n exceeds available fields", {
    similarity <- matrix(
        c(0.9, 0.3),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    schema <- data.frame(curated_field = c("cur_A"),
                         original_field = c("field2"))
    
    expect_warning(
        check_mapping(similarity, schema, top_n = 10),
        " top_n (10) is greater than number of curated fields (2). Using all fields." #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< debug
    )
})

test_that("check_mapping throws error for invalid similarity_threshold", {
    similarity <- matrix(
        c(0.9),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A"))
    )
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_error(
        check_mapping(similarity, schema, similarity_threshold = "not a number"),
        "similarity_threshold must be a single numeric value or NULL"
    )
    
    expect_error(
        check_mapping(similarity, schema, similarity_threshold = c(0.5, 0.7)),
        "similarity_threshold must be a single numeric value or NULL"
    )
})

test_that("check_mapping throws error for invalid schema column", {
    similarity <- matrix(
        c(0.9),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A"))
    )
    schema <- data.frame(col1 = c("cur_A"))
    
    expect_error(
        check_mapping(similarity, schema, schema_curated_col = "nonexistent"),
        "Column 'nonexistent' not found in schema"
    )
    
    expect_error(
        check_mapping(similarity, schema, schema_curated_col = 10),
        "Column index 10 is out of bounds"
    )
})

test_that("check_mapping returns empty dataframe when all rows are NA", {
    similarity <- matrix(
        c(NA, NA, NA, NA),
        nrow = 2, byrow = TRUE,
        dimnames = list(c("field1", "field2"), c("cur_A", "cur_B"))
    )
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_warning(
        result <- check_mapping(similarity, schema, remove_na = TRUE),
        "All rows have only NA values"
    )
    
    expect_equal(nrow(result), 0)
    expect_equal(names(result), c("original_field", "centroid", "similarity_score", 
                                  "is_mapped_to_curated", "rank"))
})

test_that("check_mapping returns empty dataframe when threshold eliminates all matches", {
    similarity <- matrix(
        c(0.1, 0.2),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    schema <- data.frame(curated_field = c("cur_A"))
    
    expect_warning(
        result <- check_mapping(similarity, schema, similarity_threshold = 0.5),
        "No valid mappings found"
    )
    
    expect_equal(nrow(result), 0)
})

test_that("check_mapping handles ties in similarity scores correctly", {
    similarity <- matrix(
        c(0.9, 0.9, 0.5),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B", "cur_C"))
    )
    schema <- data.frame(
        curated_field = c("cur_A", "cur_B"),
        original_field = c("field1", "field1"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 2)
    
    expect_equal(nrow(result), 2)
    # Both should have similarity of 0.9
    expect_true(all(result$similarity_score == 0.9))
})

test_that("check_mapping preserves correct ordering by similarity", {
    similarity <- matrix(
        c(0.3, 0.9, 0.5, 0.7),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B", "cur_C", "cur_D"))
    )
    schema <- data.frame(curated_field = c("cur_B"), original_field = c("field4"))
    
    result <- check_mapping(similarity, schema, top_n = 4)
    
    # Check descending order
    expect_equal(result$centroid, c("cur_B", "cur_D", "cur_C", "cur_A"))
    expect_equal(result$similarity_score, c(0.9, 0.7, 0.5, 0.3))
    expect_equal(result$rank, c(1, 2, 3, 4))
})

test_that("check_mapping handles empty schema gracefully", {
    similarity <- matrix(
        c(0.9, 0.3),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    schema <- data.frame(curated_field = character(0), 
                         original_field = character(0),
                         stringsAsFactors = FALSE)
    
    result <- check_mapping(similarity, schema, top_n = 1)
    
    expect_equal(nrow(result), 1)
    expect_false(result$is_mapped_to_curated)
})

test_that("check_mapping handles single row and single column matrices", {
    # Single row, multiple columns
    similarity1 <- matrix(
        c(0.9, 0.3),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    
    # Multiple rows, single column
    similarity2 <- matrix(
        c(0.9, 0.3),
        nrow = 2,
        dimnames = list(c("field1", "field2"), c("cur_A"))
    )
    
    schema <- data.frame(curated_field = c("cur_A"),
                         original_field = c("field3"),
                         stringsAsFactors = FALSE)
    
    result1 <- check_mapping(similarity1, schema, top_n = 2)
    result2 <- check_mapping(similarity2, schema, top_n = 1)
    
    expect_equal(nrow(result1), 2)
    expect_equal(nrow(result2), 2)
})

test_that("check_mapping handles special characters in field names", {
    similarity <- matrix(
        c(0.9, 0.3),
        nrow = 1,
        dimnames = list(c("field_1@test"), c("cur-A$100", "cur.B#2"))
    )
    schema <- data.frame(
        curated_field = c("cur-A$100"),
        original_field = c("field_1@test"),
        stringsAsFactors = FALSE
    )
    
    result <- check_mapping(similarity, schema, top_n = 1)
    
    expect_equal(result$original_field, "field_1@test")
    expect_equal(result$centroid, "cur-A$100")
    expect_true(result$is_mapped_to_curated)
})

test_that("check_mapping result structure is consistent across different scenarios", {
    similarity <- matrix(
        c(0.9, 0.3),
        nrow = 1,
        dimnames = list(c("field1"), c("cur_A", "cur_B"))
    )
    schema <- data.frame(curated_field = c("cur_A"))
    
    result <- check_mapping(similarity, schema, top_n = 1) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< debug
    
    # Check data types
    expect_type(result$original_field, "character")
    expect_type(result$centroid, "character")
    expect_type(result$similarity_score, "double")
    expect_type(result$is_mapped_to_curated, "logical")
    expect_type(result$rank, "integer")
})

# Run all tests
test_file(test_file = "test_check_mapping.R")
