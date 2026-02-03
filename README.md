# inferclientllmtool

R client SDK for the [Transformer Inference API](https://github.com/antoinelemor/infer-api-llm-tool).

Classify texts at scale using models trained with [LLM Tool](https://github.com/antoinelemor/LLM_Tool), or generate text with decoder LLMs via the server's Ollama integration.

## Installation

```r
# Install from GitHub
remotes::install_github("antoinelemor/inferclientllmtool")
```

## Quick start

### Classification with LLM Tool models

Connect to your inference API and classify texts using any model trained with LLM Tool:

```r
library(inferclientllmtool)

# Connect to your inference API
client <- infer_connect(
  base_url = "https://your-server.example.com",
  api_key = "YOUR_API_KEY"
)

# Check available models
infer_models(client)

# Classify single text
result <- infer_classify(client, "The economy is improving")
result$results[[1]]$label
result$results[[1]]$confidence

# Classify multiple texts
results <- infer_classify(client,
  c("Great news for investors", "Market crash expected", "Stocks unchanged"),
  model = "sentiment"
)

# Classify a data frame column
df <- data.frame(text = c("Good news", "Bad news", "Neutral statement"))
df_classified <- infer_classify_df(df, client, "text")
# Returns: text, label, confidence, prob_*, ...
```

### LLM inference with Ollama (via server)

Generate text using decoder models via the server's Ollama integration:

```r
library(inferclientllmtool)

# Connect to your inference API (same client for classification and Ollama)
client <- infer_connect(
  base_url = "https://your-server.example.com",
  api_key = "YOUR_API_KEY"
)

# Check Ollama status
infer_ollama_status(client)

# List available Ollama models
infer_ollama_models(client)

# Generate text
response <- infer_ollama_generate(client, "gemma3:27b", "What is machine learning?")
cat(response)

# With system prompt (useful for extraction tasks)
response <- infer_ollama_generate(client, "gemma3:27b",
  prompt = "Extract key entities from: The Federal Reserve raised interest rates",
  system = "You are an NLP expert. Return only a comma-separated list of entities."
)

# Chat (multi-turn conversation)
messages <- list(
  list(role = "system", content = "You are a helpful assistant."),
  list(role = "user", content = "Explain R in one sentence.")
)
result <- infer_ollama_chat(client, "gemma3:27b", messages)
cat(result$response)

# Continue conversation
messages <- result$messages
messages <- append(messages, list(list(role = "user", content = "Give an example")))
result <- infer_ollama_chat(client, "gemma3:27b", messages)
```

## Integration example: Radar+ pipeline

Use remote Ollama inference for extraction tasks:

```r
library(inferclientllmtool)
library(dplyr)
library(glue)

# Connect to API (classification + Ollama)
client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")

# Extract salient objects from headlines
df_objects <- df_clean |>
  rowwise() |>
  mutate(
    extracted_objects = infer_ollama_generate(client, "gemma3:27b",
      prompt = glue("
        Article title: {title}
        Extract: {body}

        Extract the main salient objects (people, places, organizations, events).
        Return only a comma-separated list in English, lowercase, no punctuation.
      "),
      system = "You are a journalist expert in extracting salient objects from news articles."
    )
  ) |>
  ungroup()
```

Or use your trained classification models:

```r
library(inferclientllmtool)

# Connect to your inference API
client <- infer_connect("https://your-server.example.com", "YOUR_API_KEY")

# Classify sentiment of headlines
df_classified <- df_clean |>
  infer_classify_df(client, "title", model = "sentiment")
```

## API Reference

### Inference API (LLM Tool models)

| Function | Description |
|----------|-------------|
| `infer_connect(base_url, api_key)` | Connect to inference API |
| `infer_health(client)` | Check API status |
| `infer_models(client)` | List available models |
| `infer_model_info(client, model_id)` | Get model metadata |
| `infer_classify(client, texts, model)` | Classify text(s) |
| `infer_classify_df(df, client, text_column, model)` | Classify data frame column |

### Ollama (via server)

| Function | Description |
|----------|-------------|
| `infer_ollama_status(client)` | Check Ollama availability |
| `infer_ollama_models(client)` | List available models |
| `infer_ollama_generate(client, model, prompt, system, options)` | Generate text |
| `infer_ollama_chat(client, model, messages, options)` | Multi-turn chat |

## License

MIT
