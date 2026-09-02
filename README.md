# Grinning Cat Plus

`Grinning Cat Plus` is a plugin for the Grinning Cat / Cheshire Cat ecosystem that extends the default factory with additional:

- LLM providers
- embedders (including multimodal embedders)
- chunkers
- file managers
- parsers for richer ingestion workflows

The plugin also adds multimodal parsing behavior (notably image handling) when the active embedder supports multimodality.

## Features

- Registers extra providers through factory hooks in `factories.py`
- Registers additional rabbit-hole parsers through `rabbithole.py`
- Supports cloud and local storage backends
- Supports semantic, token, hierarchical, and math-aware chunking strategies
- Enables image-aware document ingestion when using multimodal embedders

## Project layout

- `plugin.json`: plugin metadata and dependency declaration (`base_plugin`)
- `requirements.txt`: Python dependencies for all integrations
- `factories.py`: hook registration for allowed LLMs, embedders, chunkers, and file managers
- `rabbithole.py`: parser wiring and multimodal parser switching
- `llms/`: LLM config models and custom adapters
- `embedder/`: embedder config models and custom adapters
- `chunker/`: chunker config models and implementations
- `file_manager/`: storage/file manager config models and implementations
- `parsers/`: parser implementations for PowerPoint, unstructured content, and YouTube

## Registered providers

### LLMs

Enabled via `factory_allowed_llms` in `factories.py`:

- Cohere
- OpenAI Chat
- OpenAI Completions
- OpenAI-compatible APIs
- Ollama
- Google Gemini
- Azure OpenAI (chat + completion)
- Hugging Face Endpoint
- Hugging Face Text Generation Inference
- Anthropic
- Mistral AI
- Groq
- OpenRouter (one `api_key` unlocks models from many providers)

### OpenRouter LLM (`LLMOpenRouterConfig`)

- The `model` field is populated dynamically: the settings schema exposes the
  currently-available OpenRouter model ids as an **enum** (rendered by the admin
  as a searchable combo), fetched from `https://openrouter.ai/api/v1/models`
  (cached 15 min in `llms/openrouter.py`).
- On save, the plugin's `before_llm_settings_update` hook auto-enriches the stored
  settings with the selected model's capabilities and costs (best-effort — the
  save never blocks, even if the catalog is unreachable):
  - `is_multimodal` / `input_modalities` — for the multimodal pipeline;
  - `max_token_context` / `max_completion_tokens` — for context-window splitting;
  - `prompt_cost_per_1m`, `completion_cost_per_1m`, `input_cache_read_cost_per_1m`,
    `request_cost` — per-1M USD prices used for accounting.
- The core hook `before_llm_settings_update` (no-op in `base_plugin`, called by
  `ServiceUpdater` before storing LLM settings) is the extension point.

### Embedders

Enabled via `factory_allowed_embedders` in `factories.py`:

- Qdrant FastEmbed (local)
- OpenAI
- Azure OpenAI
- Cohere embeddings
- Gemini embeddings
- OpenAI-compatible embeddings
- Fake/default embedder
- Mistral embeddings
- Ollama embeddings
- Jina embeddings
- Qwen3 local embeddings
- Qwen3 via Ollama
- Qwen3 via DeepInfra (OpenAI-compatible)
- Qwen3 via Text Embeddings Inference (TEI)
- Jina multimodal embedder
- Jina CLIP multimodal embedder
- VoyageAI embeddings

### File managers

Enabled via `factory_allowed_file_managers` in `factories.py`:

- AWS S3
- Azure Blob Storage
- Google Cloud Storage
- DigitalOcean Spaces

### Chunkers

Enabled via `factory_allowed_chunkers` in `factories.py`:

- Semantic chunker
- HTML semantic chunker
- JSON chunker
- spaCy token chunker
- NLTK token chunker
- Hierarchical chunker
- Math-aware hierarchical chunker

## Parser behavior and supported content types

The parser hook `rabbithole_instantiates_parsers` dynamically switches behavior depending on whether the active embedder is multimodal.

### Always wired

- Word (`application/msword`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
- PowerPoint (`application/vnd.ms-powerpoint`, `application/vnd.openxmlformats-officedocument.presentationml.presentation`)
- Excel (`application/vnd.ms-excel`, `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`)
- Video (`video/mp4`) via YouTube transcript extraction
- Audio (`audio/mpeg`, `audio/mp3`, `audio/ogg`, `audio/wav`, `audio/webm`) via `FasterWhisperParser`

### Conditional behavior

- PDF (`application/pdf`): uses `UnstructuredParser` when embedder is multimodal; otherwise falls back to the existing handler
- Images (`image/png`, `image/jpeg`, `image/jpg`, `image/gif`, `image/bmp`, `image/tiff`, `image/webp`): only added when embedder is multimodal

### Notable parser details

- `parsers/unstructured_parser.py` enriches metadata with element type, table HTML, formula data, coordinates, page number, and optional image payload
  - to support all mime types supported by unstructured you must install the packages
    - libmagic-dev poppler-utils tesseract-ocr tesseract-ocr-{eng,ita} pandoc qpdf "libreoffice-*-nogui"
- `parsers/youtube_parser.py` fetches transcript text for YouTube sources (languages set to `en` and `it`)
- `rabbithole.py` downloads NLTK assets (`punkt`, `averaged_perceptron_tagger`) at import time

## Installation

Because this is a plugin, installation depends on your Grinning Cat/Cheshire Cat deployment strategy.

Typical local flow:

```bash
apt install -y --no-install-recommends libmagic-dev poppler-utils tesseract-ocr tesseract-ocr-{eng,ita} pandoc qpdf "libreoffice-*-nogui"

# from your plugin root
pip install -r requirements.txt
```

Then install/enable the plugin in your host application and select providers from the admin settings UI.

## Configuration

Each provider is configured through its corresponding Pydantic settings class under:

- `llms/configs.py`
- `embedders/configs.py`
- `chunkers/configs.py`
- `file_managers/configs.py`

In the admin UI, these appear using each class `humanReadableName` and expose the required fields (API keys, endpoints, model names, chunking params, etc.).

## vLLM multimodal embedder (server)

When using the vLLM multimodal embedder with a Qwen3-VL model (e.g. `Qwen/Qwen3-VL-Embedding-2B`), the server must be launched with the correct pooling/embedding flags. Multimodal embedding is supported for `Qwen3VLForConditionalGeneration`.

Correct server launch:

```bash
vllm serve Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling \
  --convert embed \
  --mm-processor-kwargs '{"min_pixels":4096,"max_pixels":1310720}' \
  --limit-mm-per-prompt '{"image":8}' \
  --max-model-len <sufficient>
```

> **Warning:** `--task embed` is **NOT** a vLLM flag. The correct flag to enable the pooling/embedding runner is `--runner pooling`. `--mm-processor-kwargs` overrides the multimodal processor defaults (Qwen3-VL-Embedding-2B ships with `min_pixels=4096`, `max_pixels=1310720`).

### Client behavior

- The plugin always pairs an image with a text part when building the `/v1/embeddings` request.
- It applies a Qwen3-VL pixel budget on the client side (`max_pixels` default `1_310_720`).
- If an image is rejected, the plugin retries once with a halved pixel-budget ceiling; if it still fails, the image is skipped (returns `None`) and ingestion continues.

### Troubleshooting `400 Failed to apply Qwen3VLProcessor`

This error is a vLLM wrapper around the Hugging Face processor call. The real cause is chained as `exc.__cause__` in the server log — always read the server log to find the actual exception. Common causes:

- **Insufficient `--max-model-len`**: the image token cost must fit within the model context. As a reference, a `1040x518` image costs roughly `512` tokens. Raise `--max-model-len` accordingly.
- **Tokenizer `truncation` / `max_length` mismatch**: the served `tokenizer.json` may carry `truncation`/`max_length` settings that conflict with the processor's image-token accounting, producing a "Mismatch in image token count between text and input_ids". Known upstream issues: `vllm-project/vllm#36653` and `llm-compressor#1725`.
- **Transformers version incompatibility** on the server: verify the installed `transformers` version is compatible with the model and vLLM build.

## Compatibility notes

- Some integrations are intentionally commented out due to compatibility concerns (see `requirements.txt` and config files)
- Multimodal parsing paths rely on `unstructured` extras and related dependencies
- Cloud providers require valid credentials and service-specific setup

## Development

Useful files while extending the plugin:

- Add/remove available providers: `factories.py`
- Change ingestion/parser routing: `rabbithole.py`
- Add a new provider config: `*/configs.py`
- Implement provider logic: `*/custom.py`
- Add parser modules: `parsers/`

## Metadata

From `plugin.json`:

- Name: `Grinning Cat Plus`
- Version: `0.0.1`
- Author: `Matteo Cacciola`
- URL: `https://github.com/matteocacciola/grinning_cat_plus`

## System dependencies

The following system packages must be present in the Docker image or on the host
running the Cheshire Cat AI instance.

### Required for all deployments

| Package | Purpose |
|---|---|
| `libmagic1` / `libmagic-dev` | MIME-type detection used by `unstructured` |
| `poppler-utils` | PDF rendering (`pdftoppm`, `pdfinfo`) |
| `libreoffice` | Converts legacy/ODF formats (ODP, ODT, PPT, DOC) to modern equivalents before text extraction |

### Required for OCR and multimodal embedders

| Package | Purpose |
|---|---|
| `tesseract-ocr` | OCR engine for images and scanned PDFs |
| `tesseract-ocr-ita` | Italian language data (add other `tesseract-ocr-*` packs as needed) |

### Required for additional document formats

| Package | Purpose |
|---|---|
| `pandoc` ≥ 2.14.2 | Enables `.epub`, `.odt`, and `.rtf` support inside `unstructured` |
| `ffmpeg` | Audio/video pre-processing for `FasterWhisperParser` |

### Ubuntu / Debian one-liner

```bash
apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    libreoffice \
    tesseract-ocr tesseract-ocr-ita \
    pandoc \
    ffmpeg
```

### Dockerfile snippet

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    libreoffice \
    tesseract-ocr tesseract-ocr-ita \
    pandoc \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*
```

> **Note:** `libreoffice` adds roughly 300 MB to the image. If you never upload
> ODP / ODS / ODT files you may omit it, but the plugin will raise a
> `RuntimeError` if those MIME types are encountered at runtime.
