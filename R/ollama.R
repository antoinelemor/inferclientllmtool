#' Connect to Ollama
#'
#' Create a connection object for a local Ollama server.
#'
#' @param base_url The Ollama server URL (default: "http://localhost:11434")
#' @return An ollama_client object
#' @export
#' @examples
#' \dontrun{
#' ollama <- ollama_connect()
#' ollama <- ollama_connect("http://localhost:11434")
#' }
ollama_connect <- function(base_url = "http://localhost:11434") {
  base_url <- sub("/$", "", base_url)

  client <- list(base_url = base_url)
  class(client) <- "ollama_client"

  tryCatch({
    resp <- httr2::request(paste0(base_url, "/api/tags")) |>
      httr2::req_perform()
    models <- httr2::resp_body_json(resp)
    n_models <- length(models$models)
    cli::cli_alert_success("Connected to Ollama at {.url {base_url}} - {n_models} model(s) available")
  }, error = function(e) {
    cli::cli_abort("Failed to connect to Ollama at {base_url}: {e$message}")
  })

  invisible(client)
}

#' List available Ollama models
#'
#' @param client An ollama_client object
#' @return A list of available models
#' @export
ollama_models <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/api/tags")) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  result$models
}

#' Generate text with Ollama
#'
#' Generate text using a local LLM via Ollama.
#'
#' @param client An ollama_client object
#' @param model The model name (e.g., "llama3", "mistral", "phi3")
#' @param prompt The prompt to send to the model
#' @param system Optional system message
#' @param stream Whether to stream the response (default FALSE)
#' @param options Optional model parameters (temperature, top_p, etc.)
#' @return The generated text (character string)
#' @export
#' @examples
#' \dontrun{
#' ollama <- ollama_connect()
#'
#' # Simple generation
#' response <- ollama_generate(ollama, "llama3", "What is R?")
#'
#' # With system prompt
#' response <- ollama_generate(ollama, "llama3",
#'   prompt = "Extract key entities from: The stock market crashed today",
#'   system = "You are an NLP expert. Return only a comma-separated list."
#' )
#'
#' # With custom parameters
#' response <- ollama_generate(ollama, "mistral",
#'   prompt = "Write a haiku about data science",
#'   options = list(temperature = 0.7)
#' )
#' }
ollama_generate <- function(client, model, prompt, system = NULL, stream = FALSE, options = NULL) {
  body <- list(
    model = model,
    prompt = prompt,
    stream = stream
  )

  if (!is.null(system)) {
    body$system <- system
  }

  if (!is.null(options)) {
    body$options <- options
  }

  resp <- httr2::request(paste0(client$base_url, "/api/generate")) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(300) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  result$response
}

#' Chat with Ollama
#'
#' Have a multi-turn conversation with a local LLM via Ollama.
#'
#' @param client An ollama_client object
#' @param model The model name
#' @param messages A list of message objects with role and content
#' @param stream Whether to stream the response (default FALSE)
#' @param options Optional model parameters
#' @return A list with the assistant's response and updated messages
#' @export
#' @examples
#' \dontrun{
#' ollama <- ollama_connect()
#'
#' messages <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "What is machine learning?")
#' )
#'
#' result <- ollama_chat(ollama, "llama3", messages)
#' print(result$response)
#'
#' # Continue conversation
#' messages <- result$messages
#' messages <- append(messages, list(list(role = "user", content = "Give me an example")))
#' result <- ollama_chat(ollama, "llama3", messages)
#' }
ollama_chat <- function(client, model, messages, stream = FALSE, options = NULL) {
  body <- list(
    model = model,
    messages = messages,
    stream = stream
  )

  if (!is.null(options)) {
    body$options <- options
  }

  resp <- httr2::request(paste0(client$base_url, "/api/chat")) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(300) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)

  updated_messages <- append(messages, list(result$message))

  list(
    response = result$message$content,
    messages = updated_messages,
    model = result$model,
    done = result$done
  )
}

#' @export
print.ollama_client <- function(x, ...) {
  cat("<ollama_client>\n")
  cat("  URL:", x$base_url, "\n")
  invisible(x)
}
