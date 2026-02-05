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
#' @param mc_samples MC Dropout forward passes for confidence intervals (0=disabled)
#' @param ci_level Confidence interval level (default 0.95 for 95\% CI)
#' @return A list with classification results, including training_mode and multi_label.
#'   When mc_samples > 0, results include confidence_interval and probabilities_ci.
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
#'
#' # With 95\% confidence intervals (30 MC Dropout passes)
#' results <- infer_classify(client, texts, model = "sentiment", mc_samples = 30)
#' }
infer_classify <- function(client, texts, model = NULL, threshold = NULL,
                           parallel = FALSE, device_mode = "both",
                           mc_samples = 0, ci_level = 0.95) {
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

  # MC Dropout confidence intervals
  if (mc_samples > 0) {
    body$mc_samples <- mc_samples
    body$ci_level <- ci_level
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
    httr2::req_timeout(client$timeout_seconds) |>
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
#' @param mc_samples MC Dropout forward passes for confidence intervals (0=disabled)
#' @param ci_level Confidence interval level (default 0.95 for 95\% CI)
#' @return The data frame with added columns. When mc_samples > 0, also adds
#'   ci_lower_* and ci_upper_* columns for each class.
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' df <- data.frame(text = c("Good news", "Bad news", "Neutral"))
#'
#' # Single-label classification
#' df_classified <- infer_classify_df(df, client, "text")
#'
#' # With 95% confidence intervals (30 MC Dropout passes)
#' df_classified <- infer_classify_df(df, client, "text", mc_samples = 30)
#'
#' # Multi-label classification with custom threshold
#' df_classified <- infer_classify_df(df, client, "text", model = "themes", threshold = 0.3)
#' }
infer_classify_df <- function(df, client, text_column, model = NULL, batch_size = 50,
                              threshold = NULL, parallel = FALSE,
                              mc_samples = 0, ci_level = 0.95) {
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
                               threshold = threshold, parallel = parallel,
                               mc_samples = mc_samples, ci_level = ci_level)

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

  # Add CI columns when mc_samples > 0
  if (mc_samples > 0 && length(all_results) > 0 && !is.null(all_results[[1]]$probabilities_ci)) {
    ci_names <- names(all_results[[1]]$probabilities_ci)
    for (cname in ci_names) {
      df[[paste0("ci_lower_", cname)]] <- sapply(all_results, function(r) r$probabilities_ci[[cname]]$lower)
      df[[paste0("ci_upper_", cname)]] <- sapply(all_results, function(r) r$probabilities_ci[[cname]]$upper)
    }
    df$ci_level <- ci_level
    df$mc_samples <- mc_samples
  }

  df
}

#' Segment text into sentences using WTPSPLIT
#'
#' WTPSPLIT is a fast multilingual sentence segmentation model that supports 85+ languages.
#'
#' @param client An infer_client object
#' @param texts A character vector of texts to segment
#' @param model WTPSPLIT model ID (default: "wtpsplit")
#' @param mode Segmentation mode: 'sentence' (default) or 'newline'
#' @return A list with segmentation results
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # Segment a single text
#' result <- infer_segment_sentences(client, "First sentence. Second sentence.")
#' print(result$results[[1]]$sentences)
#'
#' # Batch segmentation
#' texts <- c("Hello world. How are you?", "Another text. With sentences.")
#' results <- infer_segment_sentences(client, texts)
#'
#' # Preserve newlines mode
#' result <- infer_segment_sentences(client, "Para one.\n\nPara two.", mode = "newline")
#' }
infer_segment_sentences <- function(client, texts, model = "wtpsplit", mode = "sentence") {
  if (length(texts) == 1) {
    body <- list(text = texts, mode = mode)
  } else {
    body <- list(texts = as.list(texts), mode = mode)
  }

  url <- paste0(client$base_url, "/models/", model, "/segment")

  resp <- httr2::request(url) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(client$timeout_seconds) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Extract named entities using GLiNER zero-shot NER
#'
#' GLiNER is a third-party model (https://github.com/urchade/GLiNER) that
#' supports multilingual entity extraction with custom labels. It can extract
#' ANY entity type without training.
#'
#' @param client An infer_client object
#' @param texts Character vector of texts to analyze
#' @param labels Character vector of entity types to extract. Can be ANY entity type:
#'   "person", "organization", "location", "political party", "disease", "product", etc.
#' @param model NER model ID (default: "gliner")
#' @param threshold Confidence threshold (0.0-1.0, default: 0.5)
#' @param flat_ner Resolve overlapping entities (default: TRUE)
#' @return A list of results, each containing:
#'   \itemize{
#'     \item text: Input text
#'     \item entities: List of found entities with text, label, start, end, score
#'     \item entity_count: Number of entities found
#'     \item labels_used: Labels that were searched for
#'   }
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # Extract people and organizations
#' results <- infer_extract_entities(
#'   client,
#'   "Apple Inc. was founded by Steve Jobs",
#'   labels = c("person", "organization")
#' )
#'
#' # Extract custom entity types
#' results <- infer_extract_entities(
#'   client,
#'   c("The Democratic Party won the election",
#'     "Joe Biden met with Emmanuel Macron"),
#'   labels = c("political party", "politician", "event")
#' )
#'
#' # Multilingual support (12+ languages)
#' results <- infer_extract_entities(
#'   client,
#'   "Emmanuel Macron est président de la France",
#'   labels = c("person", "country", "job title")
#' )
#'
#' # Access entities from first result
#' entities <- results[[1]]$entities
#' for (entity in entities) {
#'   cat(entity$text, "is a", entity$label, "\n")
#' }
#' }
#' @details
#' Supports 12+ languages: EN, FR, DE, ES, IT, PT, NL, RU, ZH, JA, AR
#'
#' Context window: 512 tokens
#'
#' GLiNER model credit: urchade/GLiNER (not trained by LLM Tool)
infer_extract_entities <- function(client, texts, labels, model = "gliner",
                                   threshold = 0.5, flat_ner = TRUE) {
  if (length(labels) == 0) {
    cli::cli_abort("Must provide at least one label")
  }

  body <- list(
    texts = as.list(texts),
    labels = as.list(labels),
    threshold = threshold,
    flat_ner = flat_ner
  )

  url <- paste0(client$base_url, "/models/", model, "/infer")

  resp <- httr2::request(url) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(client$timeout_seconds) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  result$results
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
