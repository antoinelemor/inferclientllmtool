# Multi-head binary classification (one-vs-all model families).
#
# The inference API serves families of one-vs-all binary heads whose model
# ids share a prefix (e.g. cap_theme_health, cap_theme_immigration, ...,
# and party_* once deployed). These helpers fan one batch of texts out
# over every head of a family, apply each head's calibrated decision
# threshold (shipped in the model metadata as multi_label_threshold), and
# return one tidy data frame.

.multihead_cache <- new.env(parent = emptyenv())

#' Discover the heads of a binary model family and their thresholds
#'
#' Lists the models whose id starts with `prefix` and fetches each head's
#' calibrated decision threshold from its metadata. Results are cached on
#' the client object environment for the session.
#'
#' @param client A client object from [infer_connect()].
#' @param prefix Model id prefix of the family (default `"cap_theme_"`).
#' @param refresh Set to TRUE to bypass the session cache.
#' @return A data frame with columns `model_id`, `category`, `threshold`.
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://gate.llm-tool.org", api_key = Sys.getenv("INFER_API_KEY"))
#' infer_binary_heads(client)
#' }
infer_binary_heads <- function(client, prefix = "cap_theme_", refresh = FALSE) {
  cache_key <- paste0(client$base_url, "|", prefix)
  if (!refresh && !is.null(.multihead_cache[[cache_key]])) {
    return(.multihead_cache[[cache_key]])
  }
  models <- infer_models(client)
  if (!is.null(models$models)) models <- models$models
  ids <- vapply(models, function(m) m$model_id, character(1))
  ids <- sort(ids[startsWith(ids, prefix)])
  if (length(ids) == 0) {
    cli::cli_abort("No model with prefix {.val {prefix}} is served by the API")
  }
  thresholds <- vapply(ids, function(id) {
    info <- infer_model_info(client, id)
    th <- info$multi_label_threshold
    if (is.null(th) || is.na(as.numeric(th))) 0.5 else as.numeric(th)
  }, numeric(1))
  heads <- data.frame(
    model_id  = ids,
    category  = substring(ids, nchar(prefix) + 1L),
    threshold = unname(thresholds),
    stringsAsFactors = FALSE
  )
  .multihead_cache[[cache_key]] <- heads
  heads
}

#' Classify texts against every head of a binary model family
#'
#' Sends one request per (head, chunk) pair, all in flight concurrently
#' (`httr2::req_perform_parallel()`), applies each head's calibrated
#' threshold to the positive-class probability, and returns a tidy data
#' frame. Chunks are capped at 100 texts per request: beyond that the
#' server switches to a parallel engine that degrades sharply.
#'
#' @param client A client object from [infer_connect()].
#' @param texts Character vector of texts to classify.
#' @param prefix Model id prefix of the family (default `"cap_theme_"`).
#' @param heads Optional character vector restricting to some model ids.
#' @param chunk_size Texts per request (default 64, measured optimum; hard
#'   cap 100).
#' @param max_concurrent Max requests in flight (default 8).
#' @return A data frame with columns `text_index`, `model_id`, `category`,
#'   `score` (positive-class probability) and `decision` (score at or
#'   above the head's calibrated threshold).
#' @export
#' @examples
#' \dontrun{
#' res <- infer_binary_family(client, df$title)
#' # Themes retained for sentence 3:
#' res$category[res$decision & res$text_index == 3]
#' }
infer_binary_family <- function(client, texts, prefix = "cap_theme_",
                                heads = NULL, chunk_size = 64L,
                                max_concurrent = 8L) {
  stopifnot(length(texts) > 0)
  chunk_size <- min(as.integer(chunk_size), 100L)
  fam <- infer_binary_heads(client, prefix = prefix)
  if (!is.null(heads)) {
    fam <- fam[fam$model_id %in% heads, , drop = FALSE]
    if (nrow(fam) == 0) cli::cli_abort("No matching head")
  }

  chunks <- split(seq_along(texts), ceiling(seq_along(texts) / chunk_size))
  jobs <- list()
  for (hi in seq_len(nrow(fam))) {
    for (ci in seq_along(chunks)) {
      jobs[[length(jobs) + 1L]] <- list(head = fam[hi, ], idx = chunks[[ci]])
    }
  }

  reqs <- lapply(jobs, function(job) {
    httr2::request(paste0(client$base_url, "/models/", job$head$model_id, "/infer")) |>
      .add_auth(client) |>
      httr2::req_headers("Content-Type" = "application/json") |>
      httr2::req_body_json(list(texts = as.list(as.character(texts[job$idx])))) |>
      httr2::req_retry(max_tries = 4, is_transient = function(resp) {
        httr2::resp_status(resp) %in% c(429L, 502L, 503L, 504L)
      })
  })

  resps <- httr2::req_perform_parallel(reqs, max_active = max_concurrent,
                                       on_error = "stop")

  rows <- vector("list", length(jobs))
  for (ji in seq_along(jobs)) {
    job <- jobs[[ji]]
    parsed <- httr2::resp_body_json(resps[[ji]])
    preds <- parsed$results
    if (is.null(preds)) preds <- parsed$predictions
    scores <- vapply(seq_along(job$idx), function(j) {
      p <- preds[[j]]
      s <- p$probabilities[["1"]]
      if (is.null(s)) s <- p$confidence
      if (is.null(s)) NA_real_ else as.numeric(s)
    }, numeric(1))
    rows[[ji]] <- data.frame(
      text_index = job$idx,
      model_id   = job$head$model_id,
      category   = job$head$category,
      score      = scores,
      decision   = !is.na(scores) & scores >= job$head$threshold,
      stringsAsFactors = FALSE
    )
  }
  out <- do.call(rbind, rows)
  out[order(out$text_index, out$category), , drop = FALSE]
}

#' Classify texts against the 21 CAP theme heads
#'
#' Convenience wrapper around [infer_binary_family()] with
#' `prefix = "cap_theme_"`. The heads' decision thresholds were calibrated
#' on the 4-annotator press benchmark, so decisions are tuned for
#' vitrine/news text.
#'
#' @inheritParams infer_binary_family
#' @return See [infer_binary_family()].
#' @export
#' @examples
#' \dontrun{
#' res <- infer_cap_themes(client, c("Ottawa hausse les transferts en santé."))
#' subset(res, decision)
#' }
infer_cap_themes <- function(client, texts, heads = NULL, chunk_size = 64L,
                             max_concurrent = 8L) {
  infer_binary_family(client, texts, prefix = "cap_theme_", heads = heads,
                      chunk_size = chunk_size, max_concurrent = max_concurrent)
}
