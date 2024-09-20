# LOAD NECESSARY LIBRARIES
pacman::p_load(tidyverse, httr, jsonlite, irlba, Rtsne, plotly, cluster)

# SET SEED FOR REPRODUCIBILITY
set.seed(123)

# READ AND FILTER HORROR MOVIE DATA
horror_movies <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-11-01/horror_movies.csv") %>%
  filter(!is.na(overview), original_language == "en") %>%
  slice_sample(n = 1000)   


# DISPLAY A GLIMPSE OF THE DATA
glimpse(horror_movies)

# SAMPLE A FEW OVERVIEWS FOR DEMONSTRATION
set.seed(234)
sample(horror_movies$overview, size = 3)

# API DETAILS
api_url <- "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"
api_key <- "myAPIkey"

# FUNCTION TO GET EMBEDDINGS FOR A SINGLE TEXT WITH RETRY LOGIC
get_embeddings <- function(text) {
  max_retries <- 3
  retry_delay <- 1 # Initial delay in seconds
  
  for (attempt in 1:max_retries) {
    request_body <- list(
      model = "models/text-embedding-004",
      content = list(
        parts = list(
          list(text = text)
        )
      )
    )
    
    response <- tryCatch(
      {
        POST(
          url = paste0(api_url, "?key=", api_key),
          body = request_body,
          encode = "json",
          add_headers("Content-Type" = "application/json")
        )
      },
      error = function(e) {
        message(paste("Error during API call:", e$message))
        return(NULL) # Return NULL on error
      }
    )
    
    if (!is.null(response) && status_code(response) == 200) {
      embeddings <- content(response, as = "text", encoding = "UTF-8") %>%
        fromJSON(flatten = TRUE) %>%
        pluck("embedding", "values")
      return(embeddings)
    } else {
      message(paste("API call failed. Retrying in", retry_delay, "seconds..."))
      Sys.sleep(retry_delay)
      retry_delay <- retry_delay * 2 # Exponential backoff
    }
  }
  
  message("Failed to get embeddings after multiple retries.")
  return(NULL) # Return NULL if all retries fail
}

# GET EMBEDDINGS FOR EACH MOVIE OVERVIEW
horror_embeddings <- horror_movies %>%
  mutate(embeddings = map(overview, possibly(get_embeddings, otherwise = NULL)))

# FILTER OUT ROWS WHERE EMBEDDINGS ARE NULL (FAILED API CALLS)
horror_embeddings_clean <- horror_embeddings %>%
  filter(!map_lgl(embeddings, is.null))

# SELECT AND DISPLAY RELEVANT COLUMNS
horror_embeddings_clean %>%
  select(id, original_title, embeddings)

# CREATE EMBEDDINGS MATRIX
embeddings_mat <- matrix(
  unlist(horror_embeddings$embeddings),
  ncol = 768, byrow = TRUE
)

# DISPLAY DIMENSIONS OF EMBEDDINGS MATRIX
dim(embeddings_mat)

# CALCULATE COSINE SIMILARITY
embeddings_similarity <- embeddings_mat / sqrt(rowSums(embeddings_mat * embeddings_mat))
embeddings_similarity <- embeddings_similarity %*% t(embeddings_similarity)   


# DISPLAY DIMENSIONS OF SIMILARITY MATRIX
dim(embeddings_similarity)

# DISPLAY A SPECIFIC MOVIE AND ITS OVERVIEW
horror_movies %>%
  slice(4) %>%
  select(title, overview)

# FIND SIMILAR MOVIES BASED ON EMBEDDINGS
enframe(embeddings_similarity[4, ], name = "movie", value = "similarity") %>%
  arrange(-similarity)

# DISPLAY THE MOST SIMILAR MOVIES
horror_movies %>%
  slice(c(935, 379, 380)) %>%
  select(title, overview)

# PERFORM PCA ON EMBEDDINGS
set.seed(234)
horror_pca <- irlba::prcomp_irlba(embeddings_mat, n = 32)

# COMBINE PCA RESULTS WITH ORIGINAL DATA
augmented_pca <-
  as_tibble(horror_pca$x) %>%
  bind_cols(horror_movies)

# VISUALIZE PCA RESULTS WITH GGPLOT
augmented_pca %>%
  ggplot(aes(PC1, PC2, color = vote_average)) +
  geom_point(size = 1.3, alpha = 0.8) +
  scale_color_viridis_c()

# VISUALISE THE EMBEDDING MATRIX ------------------------------------------

# FIND DUPLICATE ROWS
duplicate_rows <- duplicated(embeddings_mat)

# REMOVE DUPLICATES AND KEEP ONLY THE FIRST OCCURRENCE
embeddings_mat_unique <- embeddings_mat[!duplicate_rows, ]

# YOU MIGHT ALSO WANT TO KEEP TRACK OF WHICH MOVIES WERE REMOVED
removed_movies <- horror_movies[duplicate_rows, ]

# SET A SEED FOR REPRODUCIBILITY
set.seed(123)

# PERFORM T-SNE ON THE UNIQUE EMBEDDINGS
tsne_out <- Rtsne(embeddings_mat_unique, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 500)

# CREATE A DATA FRAME FOR PLOTTING
tsne_df <- data.frame(tsne_out$Y)
colnames(tsne_df) <- c("X", "Y")

# ADD MOVIE TITLES (MAKE SURE TO USE THE FILTERED DATA AFTER REMOVING DUPLICATES)
tsne_df$title <- horror_movies$original_title[!duplicate_rows]

# CREATE THE PLOTLY PLOT
plot_ly(tsne_df,
        x = ~X, y = ~Y, text = ~title, type = "scatter", mode = "markers",
        marker = list(size = 10, opacity = 0.7)
) %>%
  layout(
    title = "t-SNE Visualization of Horror Movie Embeddings",
    xaxis = list(title = "t-SNE Dimension 1"),
    yaxis = list(title = "t-SNE Dimension 2")
  )


# PERFORM KMEANS CLUSTERING -----------------------------------------------

# CALCULATE WCSS FOR DIFFERENT K VALUES
wcss <- sapply(1:15, function(k) {
  kmeans(embeddings_mat_unique, centers = k, nstart = 25)$tot.withinss
})

# PLOT THE ELBOW METHOD
plot(1:15, wcss,
     type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Total within-clusters sum of squares"
)

# CALCULATE AVERAGE SILHOUETTE WIDTH FOR DIFFERENT K VALUES
sil <- sapply(2:15, function(k) {
  km <- kmeans(embeddings_mat_unique, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(embeddings_mat_unique))
  mean(ss[, 3])
})

# PLOT SILHOUETTE ANALYSIS
plot(2:15, sil,
     type = "b", pch = 19, frame = FALSE,
     xlab = "Number of clusters K",
     ylab = "Average silhouette width"
)

# PERFORM K-MEANS CLUSTERING WITH THE CHOSEN K (REPLACE 'OPTIMAL_K' WITH THE VALUE YOU SELECTED)
optimal_k <- 13
kmeans_result <- kmeans(embeddings_mat_unique, centers = optimal_k, nstart = 25)

# CREATE A NEW COLUMN FOR CLUSTER ASSIGNMENTS, INITIALIZED WITH NAS
horror_movies$cluster <- NA

# ASSIGN CLUSTER LABELS TO THE ROWS CORRESPONDING TO THE UNIQUE EMBEDDINGS
horror_movies$cluster[!duplicate_rows] <- kmeans_result$cluster
