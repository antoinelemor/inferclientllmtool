#' Get Ollama server status
#'
#' Check the status of the Ollama service on the inference API server.
#'
#' @param client An infer_client object
#' @return A list with Ollama availability status and model count
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' status <- infer_ollama_status(client)
#' print(status)
#' }
infer_ollama_status <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/ollama/status")) |>
    httr2::req_headers("X-API-Key" = client$api_key) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' List available Ollama models
#'
#' Get the list of models available on the server's Ollama instance.
#'
#' @param client An infer_client object
#' @return A list of available Ollama models
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' models <- infer_ollama_models(client)
#' print(models)
#' }
infer_ollama_models <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/ollama/models")) |>
    httr2::req_headers("X-API-Key" = client$api_key) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' Generate text with Ollama
#'
#' Generate text using an LLM via the server's Ollama instance.
#'
#' @param client An infer_client object
#' @param model The model name (e.g., "llama3", "mistral", "gemma3:27b")
#' @param prompt The prompt to send to the model
#' @param system Optional system message
#' @param options Optional model parameters (temperature, top_p, etc.)
#' @return The generated text (character string)
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # Simple generation
#' response <- infer_ollama_generate(client, "gemma3:27b", "What is R?")
#' cat(response)
#'
#' # With system prompt
#' response <- infer_ollama_generate(client, "gemma3:27b",
#'   prompt = "Extract key entities from: The stock market crashed today",
#'   system = "You are an NLP expert. Return only a comma-separated list."
#' )
#'
#' # With custom parameters
#' response <- infer_ollama_generate(client, "gemma3:27b",
#'   prompt = "Write a haiku about data science",
#'   options = list(temperature = 0.7)
#' )
#' }
infer_ollama_generate <- function(client, model, prompt, system = NULL, options = NULL) {
  body <- list(
    model = model,
    prompt = prompt
  )

  if (!is.null(system)) {
    body$system <- system
  }

  if (!is.null(options)) {
    body$options <- options
  }

  resp <- httr2::request(paste0(client$base_url, "/ollama/generate")) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(client$timeout_seconds) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)
  result$response
}

#' Chat with Ollama
#'
#' Have a multi-turn conversation with an LLM via the server's Ollama instance.
#'
#' @param client An infer_client object
#' @param model The model name (e.g., "gemma3:27b", "llama3")
#' @param messages A list of message objects with role and content
#' @param options Optional model parameters (temperature, top_p, etc.)
#' @return A list with the assistant's response and updated messages
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' messages <- list(
#'   list(role = "system", content = "You are a helpful assistant."),
#'   list(role = "user", content = "What is machine learning?")
#' )
#'
#' result <- infer_ollama_chat(client, "gemma3:27b", messages)
#' cat(result$response)
#'
#' # Continue conversation
#' messages <- result$messages
#' messages <- append(messages, list(list(role = "user", content = "Give me an example")))
#' result <- infer_ollama_chat(client, "gemma3:27b", messages)
#' }
#' Translate text with TranslateGemma
#'
#' Translate text using TranslateGemma via the server's Ollama instance.
#' Supports 130+ languages with regional variants (BCP-47 codes).
#'
#' @param client An infer_client object
#' @param text The text to translate
#' @param source_lang Source language code (e.g., "en", "fr", "zh-Hans")
#' @param target_lang Target language code (e.g., "fr", "en", "es")
#' @param model TranslateGemma model variant (default: "translategemma:12b")
#' @param options Optional model parameters (temperature, top_p, etc.)
#' @return A list with translation, source_lang, target_lang, model, etc.
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#'
#' # English to French
#' result <- infer_translate(client, "Hello, how are you?", "en", "fr")
#' cat(result$translation)
#'
#' # French to Spanish
#' result <- infer_translate(client, "Bonjour le monde", "fr", "es")
#'
#' # With custom parameters
#' result <- infer_translate(client, "The economy is growing", "en", "de",
#'   options = list(temperature = 0.3)
#' )
#' }
infer_translate <- function(client, text, source_lang, target_lang,
                            model = "translategemma:12b", options = NULL) {
  body <- list(
    text = text,
    source_lang = source_lang,
    target_lang = target_lang,
    model = model
  )

  if (!is.null(options)) {
    body$options <- options
  }

  resp <- httr2::request(paste0(client$base_url, "/ollama/translate")) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(client$timeout_seconds) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

#' List TranslateGemma supported languages
#'
#' Get the list of all languages supported by TranslateGemma.
#'
#' @param client An infer_client object
#' @return A list with 'languages' (list of code/name pairs) and 'count'
#' @export
#' @examples
#' \dontrun{
#' client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")
#' langs <- infer_translate_languages(client)
#' print(paste(langs$count, "languages supported"))
#' }
infer_translate_languages <- function(client) {
  resp <- httr2::request(paste0(client$base_url, "/ollama/translate/languages")) |>
    httr2::req_headers("X-API-Key" = client$api_key) |>
    httr2::req_perform()

  httr2::resp_body_json(resp)
}

infer_ollama_chat <- function(client, model, messages, options = NULL) {
  body <- list(
    model = model,
    messages = messages
  )

  if (!is.null(options)) {
    body$options <- options
  }

  resp <- httr2::request(paste0(client$base_url, "/ollama/chat")) |>
    httr2::req_headers(
      "X-API-Key" = client$api_key,
      "Content-Type" = "application/json"
    ) |>
    httr2::req_body_json(body) |>
    httr2::req_timeout(client$timeout_seconds) |>
    httr2::req_perform()

  result <- httr2::resp_body_json(resp)

  # Build updated messages list
  assistant_message <- list(role = "assistant", content = result$response)
  updated_messages <- append(messages, list(assistant_message))

  list(
    response = result$response,
    messages = updated_messages,
    model = result$model,
    done = result$done
  )
}
