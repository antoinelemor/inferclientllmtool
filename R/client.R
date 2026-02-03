#' Connect to LLM Tool Inference API
#'
#' Create a connection object for the inference API.
#'
#' @param base_url The base URL of your inference API server
#' @param api_key Your API key for authentication
#' @return An infer_client object
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' }
infer_connect <- function(base_url, api_key) {
  if (missing(base_url) || is.null(base_url) || base_url == "") {
    cli::cli_abort("base_url is required")
  }
  if (missing(api_key) || is.null(api_key) || api_key == "") {
    cli::cli_abort("api_key is required")
  }

  base_url <- sub("/$", "", base_url)

  client <- list(
    base_url = base_url,
    api_key = api_key
  )
  class(client) <- "infer_client"

  tryCatch({
    health <- infer_health(client)
    cli::cli_alert_success("Connected to {.url {base_url}} - v{health$version}, {health$models_count} model(s)")
  }, error = function(e) {
    cli::cli_abort("Failed to connect: {e$message}")
  })

  invisible(client)
}

#' Check API health
#'
#' @param client An infer_client object
#' @return A list with API status information
#' @export
infer_health <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/health")) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' List available models
#'
#' @param client An infer_client object
#' @return A list of available models
#' @export
infer_models <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/models")) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Get model information
#'
#' @param client An infer_client object
#' @param model_id The model identifier
#' @return A list with model metadata
#' @export
infer_model_info <- function(client, model_id) {
  resp <- httr2::request(paste0(client$base_url, "/models/", model_id)) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Classify text(s)
#'
#' Run inference on one or more texts using a model trained with LLM Tool.
#'
#' @param client An infer_client object
#' @param texts A character vector of texts to classify
#' @param model Optional model ID (uses default if not specified)
#' @return A list with classification results
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # Single text
#' result <- infer_classify(client, "The economy is improving")
#'
#' # Multiple texts
#' results <- infer_classify(client, c("Good news", "Bad news"), model = "sentiment")
#' }
infer_classify <- function(client, texts, model = NULL) {
  if (length(texts) == 1) {
    body <- list(text = texts)
  } else {
    body <- list(texts = as.list(texts))
  }

  if (!is.null(model)) {
    body$model <- model
  }

  resp <- httr2::request(paste0(client$base_url, "/infer")) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Classify a data frame column
#'
#' Classify texts in a data frame column and add result columns.
#'
#' @param df A data frame
#' @param client An infer_client object
#' @param text_column Name of the column containing texts
#' @param model Optional model ID
#' @param batch_size Number of texts per API call (default 50)
#' @return The data frame with added columns: label, confidence, and probability columns
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' df <- data.frame(text = c("Good news", "Bad news", "Neutral"))
#' df_classified <- infer_classify_df(df, client, "text")
#' }
infer_classify_df <- function(df, client, text_column, model = NULL, batch_size = 50) {
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    cli::cli_abort("Package 'dplyr' is required for infer_classify_df()")
  }

  texts <- df[[text_column]]
  n <- length(texts)

  all_results <- vector("list", n)

  for (i in seq(1, n, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n)
    batch_texts <- texts[i:end_idx]

    response <- infer_classify(client, batch_texts, model = model)

    for (j in seq_along(response$results)) {
      all_results[[i + j - 1]] <- response$results[[j]]
    }
  }

  labels <- sapply(all_results, function(r) r$label)
  confidences <- sapply(all_results, function(r) r$confidence)

  df$label <- labels
  df$confidence <- confidences

  if (length(all_results) > 0 && !is.null(all_results[[1]]$probabilities)) {
    prob_names <- names(all_results[[1]]$probabilities)
    for (pname in prob_names) {
      col_name <- paste0("prob_", pname)
      df[[col_name]] <- sapply(all_results, function(r) r$probabilities[[pname]])
    }
  }

  df
}

#' @export
print.infer_client <- function(x, ...) {
  cat("<infer_client>\n")
  cat("  URL:", x$base_url, "\n")
  cat("  Key:", substr(x$api_key, 1, 12), "...\n")
  invisible(x)
}
