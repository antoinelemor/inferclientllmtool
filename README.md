# inferclientllmtool

R client SDK for the [Transformer Inference API](https://github.com/antoinelemor/infer-api-llm-tool).

Classify texts at scale using models trained with [LLM Tool](https://github.com/antoinelemor/LLM_Tool), or generate text with decoder LLMs via the server's Ollama integration.

## Features

- **Multi-model support** — target any model registered on the inference server
- **Multi-label classification** — supports binary, multi-class, multi-label, and one-vs-all training modes
- **Parallel GPU+CPU inference** — hybrid inference for large batches using server-side parallelization
- **Configurable thresholds** — customize multi-label prediction thresholds
- **Zero-shot NER** — extract entities with custom labels using GLiNER (requires server with `pip install 'infer-api[ner]'`)
  - **Multilingual**: 12+ languages (EN, FR, DE, ES, IT, PT, NL, RU, ZH, JA, AR)
  - **Custom labels**: ANY entity type ("political party", "disease", "product", etc.)
  - **Credit**: [urchade/GLiNER](https://github.com/urchade/GLiNER) (third-party model)
- **Ollama integration** — generate text and chat with server-side LLMs
- **DataFrame classification** — classify data frame columns with automatic result binding

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

# Check available models (includes training_mode, multi_label info)
infer_models(client)

# Get optimal inference configuration
config <- infer_model_config(client, "sentiment", n_texts = 1000)
# Returns: batch_size, use_parallel, device_mode, training_mode, etc.

# Server resources
infer_resources(client)

# Classify single text
result <- infer_classify(client, "The economy is improving")
result$results[[1]]$label
result$results[[1]]$confidence

# Classify multiple texts
results <- infer_classify(client,
  c("Great news for investors", "Market crash expected", "Stocks unchanged"),
  model = "sentiment"
)

# Parallel GPU+CPU inference (for large batches)
results <- infer_classify(client, texts, model = "sentiment",
                          parallel = TRUE, device_mode = "both")

# Multi-label classification with custom threshold
results <- infer_classify(client, "Economic policy affects markets",
                          model = "themes", threshold = 0.3)
# results$results[[1]]$labels = c("economy", "politics")
# results$results[[1]]$label_count = 2

# Classify a data frame column (single-label)
df <- data.frame(text = c("Good news", "Bad news", "Neutral statement"))
df_classified <- infer_classify_df(df, client, "text")
# Returns: text, label, confidence, prob_*, ...

# Classify a data frame column (multi-label)
df_classified <- infer_classify_df(df, client, "text", model = "themes", threshold = 0.3)
# Returns: text, labels (comma-separated), label_count, threshold, prob_*, ...
```

### Named Entity Recognition (requires server with NER support)

Extract entities with custom labels using zero-shot NER:

```r
library(inferclientllmtool)

# Connect to your inference API
client <- infer_connect(
  base_url = "https://your-server.example.com",
  api_key = "YOUR_API_KEY"
)

# Extract entities with standard labels
results <- infer_extract_entities(
  client,
  "Apple Inc. was founded by Steve Jobs in Cupertino",
  labels = c("person", "organization", "location")
)

# Access entities from first result
entities <- results[[1]]$entities
for (entity in entities) {
  cat(entity$text, "is a", entity$label, "(", entity$score, ")\n")
}
# Output:
#   Apple Inc. is a organization ( 0.91 )
#   Steve Jobs is a person ( 0.99 )
#   Cupertino is a location ( 0.99 )

# Extract custom entity types (zero-shot, no training needed)
results <- infer_extract_entities(
  client,
  c(
    "The Democratic Party won against the Republican Party",
    "Joe Biden met with Emmanuel Macron in Paris"
  ),
  labels = c("political party", "politician", "location")
)

# Multilingual extraction (12+ languages)
results <- infer_extract_entities(
  client,
  "Emmanuel Macron est président de la France",
  labels = c("person", "country", "job title"),
  threshold = 0.4  # Adjust confidence threshold
)

# Process results
for (result in results) {
  cat("Text:", result$text, "\n")
  cat("Found", result$entity_count, "entities:\n")
  for (entity in result$entities) {
    cat(sprintf("  - %s (%s, %.2f)\n", entity$text, entity$label, entity$score))
  }
}
```

**Output format**: List of results, each containing `text`, `entities` (list), `entity_count`, `labels_used`, `threshold`. Each entity has `text`, `label`, `start`, `end`, `score`.

**Credit**: Uses [GLiNER](https://github.com/urchade/GLiNER) or [GLiNER2](https://huggingface.co/fastino/gliner2-large-v1) (third-party models, not LLM Tool trained)

**Note**: GLiNER2 supports 2048 token context (4x larger) for longer documents. Use `model="gliner2"` to access.

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
| `infer_models(client)` | List available models with training mode info |
| `infer_model_info(client, model_id)` | Get model metadata |
| `infer_model_config(client, model_id, n_texts)` | Get optimal inference configuration |
| `infer_resources(client)` | Get server resource status |
| `infer_classify(client, texts, model, threshold, parallel, device_mode)` | Classify text(s) |
| `infer_classify_df(df, client, text_column, model, threshold, parallel)` | Classify data frame column |
| `infer_extract_entities(client, texts, labels, model, threshold, flat_ner)` | Extract named entities (zero-shot NER) |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | model default | Multi-label threshold (0.0-1.0) |
| `parallel` | `FALSE` | Enable parallel GPU+CPU inference |
| `device_mode` | `"both"` | Device for parallel: `"cpu"`, `"gpu"`, or `"both"` |

### Ollama (via server)

| Function | Description |
|----------|-------------|
| `infer_ollama_status(client)` | Check Ollama availability |
| `infer_ollama_models(client)` | List available models |
| `infer_ollama_generate(client, model, prompt, system, options)` | Generate text |
| `infer_ollama_chat(client, model, messages, options)` | Multi-turn chat |

### Model info response

`infer_model_info(client, model_id)` returns comprehensive metadata:

| Field | Description |
|-------|-------------|
| `model_id` | Model identifier |
| `base_model` | HuggingFace base model name |
| `training_mode` | `binary`, `multi-class`, `multi-label`, `one-vs-all` |
| `multi_label` | Whether model uses multi-label classification |
| `labels` | List of class labels |
| `languages` | Supported languages |
| `metrics` | Overall metrics (macro_f1, accuracy, train_loss, val_loss) |
| `per_class_metrics` | Per-class precision, recall, f1, support |
| `per_language_metrics` | Per-language breakdown |
| `hyperparameters` | Training hyperparameters |
| `raw_config` | Full config.json contents |
| `raw_training_metadata` | Full training_metadata.json contents |

## License

MIT
