#' Connect to LLM Tool Inference API
#'
#' Create a connection object for the inference API.
#'
#' @param base_url The base URL of your inference API server
#' @param api_key Your API key for authentication
#' @param timeout_seconds Request timeout in seconds (default: 600)
#' @return An infer_client object
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' # For large models that take longer to respond:
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY", timeout_seconds = 600)
#' }
infer_connect <- function(base_url, api_key, timeout_seconds = 600) {
  if (missing(base_url) || is.null(base_url) || base_url == "") {
    cli::cli_abort("base_url is required")
  }
  if (missing(api_key) || is.null(api_key) || api_key == "") {
    cli::cli_abort("api_key is required")
  }

  base_url <- sub("/$", "", base_url)

  client <- list(
    base_url = base_url,
    api_key = api_key,
    timeout_seconds = timeout_seconds
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
#' @return A list of available models with training_mode and multi_label info
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

#' Get optimal inference configuration for a model
#'
#' @param client An infer_client object
#' @param model_id The model identifier
#' @param n_texts Expected number of texts (affects parallel recommendation)
#' @return A list with batch_size, device_mode, use_parallel, training_mode, etc.
#' @export
infer_model_config <- function(client, model_id, n_texts = 100) {
  resp <- httr2::request(paste0(client$base_url, "/models/", model_id, "/config")) |>
    httr2::req_url_query(n_texts = n_texts) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Classify text(s)
#'
#' Run inference on one or more texts using a model trained with LLM Tool.
#'
#' For multi-label models, each result contains 'labels' (list) instead of 'label'.
#' Use threshold to override the model's default multi-label threshold.
#'
#' @param client An infer_client object
#' @param texts A character vector of texts to classify
#' @param model Optional model ID (uses default if not specified)
#' @param threshold Override multi-label threshold (0.0-1.0)
#' @param parallel Use parallel GPU+CPU inference (for large batches)
#' @param device_mode Device mode for parallel: 'cpu', 'gpu', or 'both'
#' @return A list with classification results, including training_mode and multi_label
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # Single text
#' result <- infer_classify(client, "The economy is improving")
#'
#' # Multiple texts with parallel inference
#' results <- infer_classify(client, texts, model = "sentiment", parallel = TRUE)
#'
#' # Multi-label classification with custom threshold
#' results <- infer_classify(client, texts, model = "themes", threshold = 0.3)
#' }
infer_classify <- function(client, texts, model = NULL, threshold = NULL,
                           parallel = FALSE, device_mode = "both") {
  if (length(texts) == 1) {
    body <- list(text = texts)
  } else {
    body <- list(texts = as.list(texts))
  }

  # Parallel inference options
  if (parallel) {
    body$parallel <- TRUE
    body$device_mode <- device_mode
  }

  # Multi-label threshold override
  if (!is.null(threshold)) {
    body$threshold <- threshold
  }

  if (!is.null(model)) {
    url <- paste0(client$base_url, "/models/", model, "/infer")
  } else {
    url <- paste0(client$base_url, "/infer")
  }

  resp <- httr2::request(url) |>
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
#' For single-label models:
#'   Adds columns: label, confidence, and prob_* columns for each class.
#'
#' For multi-label models:
#'   Adds columns: labels (comma-separated), label_count, threshold,
#'   and prob_* columns for each class.
#'
#' @param df A data frame
#' @param client An infer_client object
#' @param text_column Name of the column containing texts
#' @param model Optional model ID
#' @param batch_size Number of texts per API call (default 50)
#' @param threshold Override multi-label threshold (0.0-1.0)
#' @param parallel Use parallel GPU+CPU inference
#' @return The data frame with added columns
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' df <- data.frame(text = c("Good news", "Bad news", "Neutral"))
#'
#' # Single-label classification
#' df_classified <- infer_classify_df(df, client, "text")
#'
#' # Multi-label classification with custom threshold
#' df_classified <- infer_classify_df(df, client, "text", model = "themes", threshold = 0.3)
#' }
infer_classify_df <- function(df, client, text_column, model = NULL, batch_size = 50,
                              threshold = NULL, parallel = FALSE) {
  if (!requireNamespace("dplyr", quietly = TRUE)) {
    cli::cli_abort("Package 'dplyr' is required for infer_classify_df()")
  }

  texts <- df[[text_column]]
  n <- length(texts)

  all_results <- vector("list", n)
  multi_label <- FALSE

  for (i in seq(1, n, by = batch_size)) {
    end_idx <- min(i + batch_size - 1, n)
    batch_texts <- texts[i:end_idx]

    response <- infer_classify(client, batch_texts, model = model,
                               threshold = threshold, parallel = parallel)

    # Track if this is a multi-label model
    multi_label <- isTRUE(response$multi_label)

    for (j in seq_along(response$results)) {
      all_results[[i + j - 1]] <- response$results[[j]]
    }
  }

  if (multi_label) {
    # Multi-label results: 'labels' is a list of strings
    df$labels <- sapply(all_results, function(r) {
      if (is.null(r$labels) || length(r$labels) == 0) {
        ""
      } else {
        paste(unlist(r$labels), collapse = ",")
      }
    })
    df$label_count <- sapply(all_results, function(r) {
      if (is.null(r$label_count)) 0 else r$label_count
    })
    df$threshold <- sapply(all_results, function(r) {
      if (is.null(r$threshold)) 0.5 else r$threshold
    })
  } else {
    # Single-label results
    df$label <- sapply(all_results, function(r) {
      if (is.null(r$label)) "" else r$label
    })
    df$confidence <- sapply(all_results, function(r) {
      if (is.null(r$confidence)) 0 else r$confidence
    })
  }

  # Add probability columns for both modes
  if (length(all_results) > 0 && !is.null(all_results[[1]]$probabilities)) {
    prob_names <- names(all_results[[1]]$probabilities)
    for (pname in prob_names) {
      col_name <- paste0("prob_", pname)
      df[[col_name]] <- sapply(all_results, function(r) r$probabilities[[pname]])
    }
  }

  df
}

#' Get server resources
#'
#' @param client An infer_client object
#' @return A list with CPU, GPU, memory, storage, and capacity info
#' @export
infer_resources <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/resources")) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' @export
print.infer_client <- function(x, ...) {
  cat("<infer_client>\n")
  cat("  URL:", x$base_url, "\n")
  cat("  Key:", substr(x$api_key, 1, 12), "...\n")
  cat("  Timeout:", x$timeout_seconds, "seconds\n")
  invisible(x)
}
