# Custom cosine similarity function
.cosine_similarity <- function(A, B) {
    # A: matrix (rows = observations, cols = features)
    # B: matrix (rows = centroids, cols = features)
    
    # Normalize rows to unit vectors
    A_norm <- A / sqrt(rowSums(A^2))
    B_norm <- B / sqrt(rowSums(B^2))
    
    # Compute dot product (similarity matrix)
    similarity <- A_norm %*% t(B_norm)
    
    return(similarity)
}

#' Assign Data Points to Centroids Based on Cosine Similarity
#'
#' This function assigns each data point to the most similar centroid using
#' cosine similarity as the distance metric ('Nearest centroid classification 
#' with cosine similarity'). It returns cluster assignments along with a 
#' detailed matching table and similarity scores.
#'
#' @param data A numeric matrix where rows represent data points and columns
#'   represent features. Row names, if present, will be used to identify
#'   data points in the output matching table.
#' @param centroids A numeric matrix where rows represent centroids and columns
#'   represent features. Must have the same number of columns as \code{data}.
#'   Row names, if present, will be used to identify centroids in the output.
#'
#' @return A list containing four elements:
#'   \describe{
#'     \item{matching}{A data frame with two columns:
#'       \itemize{
#'         \item \code{original_field}: Character vector of data point identifiers
#'           (from row names of \code{data})
#'         \item \code{centroid}: Character vector of assigned centroid
#'           identifiers (from row names of \code{centroids})
#'       }
#'     }
#'     \item{cluster}{An integer vector indicating the cluster assignment
#'       for each data point (indices correspond to centroid rows)}
#'     \item{centers}{The original centroids matrix passed as input}
#'     \item{similarity}{A numeric matrix of cosine similarity scores where
#'       rows correspond to data points and columns to centroids}
#'   }
#'
#' @details
#' The function uses cosine similarity to measure the similarity between each
#' data point and each centroid. Cosine similarity ranges from -1 to 1, where
#' 1 indicates perfect similarity, 0 indicates orthogonality, and -1 indicates
#' perfect dissimilarity. Each data point is assigned to the centroid with
#' the highest cosine similarity score.
#'
#' @examples
#' # Create sample data
#' data <- matrix(rnorm(100), nrow = 10, ncol = 10)
#' rownames(data) <- paste0("point_", 1:10)
#' 
#' # Create centroids
#' centroids <- matrix(rnorm(30), nrow = 3, ncol = 10)
#' rownames(centroids) <- paste0("centroid_", 1:3)
#' 
#' # Assign data points to centroids
#' result <- centroid_cluster_by_cosine(data, centroids)
#' 
#' # View matching table
#' print(result$matching)
#' 
#' # View cluster assignments
#' print(result$cluster)
#'
#' @export
centroid_cluster_by_cosine <- function(data, centroids) {
    # data: matrix (rows = data points, cols = features)
    # centroids: matrix (rows = centroids, cols = features)
    
    # Calculate cosine similarity
    similarities <- .cosine_similarity(data, centroids)
    
    # Assign each data point to most similar centroid
    cluster <- apply(similarities, 1, which.max)
    
    # Create matching table with row names
    matching_table <- data.frame(
        original_field = rownames(data),
        centroid = rownames(centroids)[cluster],
        stringsAsFactors = FALSE
    )
    
    return(list(
        matching = matching_table,
        cluster = cluster, 
        centers = centroids,
        similarity = similarities
    ))
}