from typing import Type, Any, Literal
from fastembed import TextEmbedding
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings
from pydantic import ConfigDict, Field
from cat.services.factory.embedder import EmbedderSettings, EmbedderMultimodalSettings
from cat.utils import Enum

from .custom import (
    CustomFastEmbedEmbeddings,
    CustomOpenAIEmbeddings,
    CustomOllamaEmbeddings,
    CustomJinaEmbedder,
    Qwen3LocalEmbeddings,
    Qwen3OllamaEmbeddings,
    Qwen3DeepInfraEmbeddings,
    Qwen3TEIEmbeddings,
    CustomJinaMultimodalEmbedder,
    JinaCLIPEmbeddings,
    CustomVllmMultimodalEmbedder,
)


class EmbedderOpenAICompatibleConfig(EmbedderSettings):
    api_key: str | None = None
    model: str
    url: str
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI-compatible API embedder",
            "description": "Configuration for OpenAI-compatible API embeddings",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomOpenAIEmbeddings]:
        return CustomOpenAIEmbeddings


class EmbedderOpenAIConfig(EmbedderSettings):
    openai_api_key: str
    model: str = "text-embedding-ada-002"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "OpenAI Embedder",
            "description": "Configuration for OpenAI embeddings",
            "link": "https://platform.openai.com/docs/models/overview",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[OpenAIEmbeddings]:
        return OpenAIEmbeddings


# https://python.langchain.com/en/latest/_modules/langchain/embeddings/openai.html#OpenAIEmbeddings
class EmbedderAzureOpenAIConfig(EmbedderSettings):
    openai_api_key: str
    model: str
    azure_endpoint: str
    openai_api_type: str = "azure"
    openai_api_version: str
    deployment: str

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Azure OpenAI Embedder",
            "description": "Configuration for Azure OpenAI embeddings",
            "link": "https://azure.microsoft.com/en-us/products/ai-services/openai-service",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[AzureOpenAIEmbeddings]:
        return AzureOpenAIEmbeddings


class EmbedderCohereConfig(EmbedderSettings):
    cohere_api_key: str
    model: str = "embed-multilingual-v2.0"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Cohere Embedder",
            "description": "Configuration for Cohere embeddings",
            "link": "https://docs.cohere.com/docs/models",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CohereEmbeddings]:
        return CohereEmbeddings


# Enum for menu selection in the admin!
FastEmbedModels = Enum(
    "FastEmbedModels",
    {
        item["model"].replace("/", "_").replace("-", "_"): item["model"]
        for item in TextEmbedding.list_supported_models()
    },
)


class EmbedderQdrantFastEmbedConfig(EmbedderSettings):
    model_name: FastEmbedModels = Field(title="Model name", default="BAAI/bge-base-en")  # type: ignore
    # Unknown behavior for values > 512.
    max_length: int = 512
    # as suggest on fastembed documentation, "passage" is the best option for documents.
    doc_embed_type: str = "passage"
    cache_dir: str = "cat/data/models/fast_embed"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Qdrant FastEmbed (Local)",
            "description": "Configuration for Qdrant FastEmbed",
            "link": "https://qdrant.github.io/fastembed/",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomFastEmbedEmbeddings]:
        return CustomFastEmbedEmbeddings


class EmbedderGeminiChatConfig(EmbedderSettings):
    """Configuration for Gemini Chat Embedder.

    This class contains the configuration for the Gemini Embedder.
    """
    google_api_key: str
    # Default model https://python.langchain.com/docs/integrations/text_embedding/google_generative_ai
    model: str = "models/embedding-001"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Google Gemini Embedder",
            "description": "Configuration for Gemini Embedder",
            "link": "https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/text-embeddings?hl=en",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[GoogleGenerativeAIEmbeddings]:
        return GoogleGenerativeAIEmbeddings


class EmbedderMistralAIChatConfig(EmbedderSettings):
    """
    Configuration for Mistral AI Chat Embedder.

    This class contains the configuration for the Mistral AI Embedder.
    """
    api_key: str
    model: str = "mistral-embed"
    max_retries: int = 5
    max_concurrent_requests: int = 64

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Mistral AI Embedder",
            "description": "Configuration for MistralAI Embedder",
            "link": "https://docs.mistral.ai/capabilities/embeddings/",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[MistralAIEmbeddings]:
        return MistralAIEmbeddings


class EmbedderVoyageAIChatConfig(EmbedderSettings):
    """
    Configuration for Voyage AI Chat Text Embedder.

    This class contains the configuration for the Voyage AI Text Embedder.
    """
    api_key: str
    model: str = "voyage-3"
    batch_size: int

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Voyage AI Embedder",
            "description": "Configuration for Voyage AI Embedder",
            "link": "https://docs.voyageai.com/docs/embeddings",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[VoyageAIEmbeddings]:
        return VoyageAIEmbeddings


class EmbedderOllamaConfig(EmbedderSettings):
    base_url: str
    model: str = "mxbai-embed-large"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Ollama embedding models",
            "description": "Configuration for Ollama embeddings API",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomOllamaEmbeddings]:
        return CustomOllamaEmbeddings


class EmbedderJinaConfig(EmbedderSettings):
    base_url: str
    model: str
    api_key: str
    task: str | None = "text-matching"
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Jina Embedder",
            "description": "Configuration for Jina embeddings",
            "link": "https://docs.jina.ai/api/jina/hub/index.html?highlight=embeddings#jina.hub.encoders.text.TextEncoder",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomJinaEmbedder]:
        return CustomJinaEmbedder


class Qwen3LocalEmbeddingsConfig(EmbedderSettings):
    model_name: str
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Local Qwen3 embeddings using HuggingFace Sentence Transformers",
            "description": "Configuration for Local Qwen3 embeddings using HuggingFace Sentence Transformers",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[Qwen3LocalEmbeddings]:
        return Qwen3LocalEmbeddings


class Qwen3OllamaEmbeddingsConfig(EmbedderSettings):
    model_name: str
    base_url: str
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Qwen3 embeddings via Ollama",
            "description": "Configuration for Qwen3 embeddings via Ollama",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[Qwen3OllamaEmbeddings]:
        return Qwen3OllamaEmbeddings


class Qwen3DeepInfraEmbeddingsConfig(EmbedderSettings):
    model_name: str
    base_url: str
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Qwen3 embeddings via DeepInfra API (OpenAI-compatible)",
            "description": "Configuration for Qwen3 embeddings via DeepInfra API (OpenAI-compatible) embeddings",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[Qwen3DeepInfraEmbeddings]:
        return Qwen3DeepInfraEmbeddings


class Qwen3TEIEmbeddingsConfig(EmbedderSettings):
    base_url: str
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Qwen3 embeddings via Text Embeddings Inference",
            "description": "Configuration for Qwen3 embeddings via Text Embeddings Inference embeddings",
            "link": "",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[Qwen3TEIEmbeddings]:
        return Qwen3TEIEmbeddings


class EmbedderJinaMultimodalConfig(EmbedderMultimodalSettings):
    base_url: str
    model: str
    api_key: str
    task: str | None = "text-matching"
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Jina Embedder",
            "description": "Configuration for Jina embeddings",
            "link": "https://docs.jina.ai/api/jina/hub/index.html?highlight=embeddings#jina.hub.encoders.text.TextEncoder",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomJinaMultimodalEmbedder]:
        return CustomJinaMultimodalEmbedder


class JinaCLIPEmbeddingsConfig(EmbedderMultimodalSettings):
    api_key: str
    model_name: str = "jina-clip-v2"
    base_url: str = "https://api.jina.ai/v1/embeddings"
    max_input_tokens: int | None = Field(
        default=None,
        description="Maximum input tokens accepted by the embedding model. Used to size chunks that never exceed the embedder. None = unknown/unlimited.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Jina CLIP Embedder",
            "description": "Configuration for Jina CLIP embeddings",
            "link": "https://docs.jina.ai/",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[JinaCLIPEmbeddings]:
        return JinaCLIPEmbeddings


class VllmMultimodalConfiguration(EmbedderMultimodalSettings):
    model: str
    base_url: str = "http://localhost:8000"
    api_key: str | None = None
    timeout: float = 300.0
    max_image_tokens: int = Field(
        default=2048,
        ge=64,
        description="Max image tokens the embedding model accepts before images are downscaled. Jina v5 omni rejects grids over ~2048; other multimodal models may allow more.",
    )
    query_prefix: str = Field(
        default="Query: ",
        description="Prefix prepended to query-side text/image inputs (Jina v5 default 'Query: '). Set to '' to disable.",
    )
    document_prefix: str = Field(
        default="Document: ",
        description="Prefix prepended to document-side text/image inputs (Jina v5 default 'Document: '; some setups use 'Passage: '). Set to '' to disable.",
    )
    max_input_tokens: int | None = Field(
        default=None,
        ge=256,
        description="Raw model context window (max_model_len). None (default) auto-detects the value from the vLLM /v1/models endpoint on first use and stores it back here; the per-request budget is then max_input_tokens * context_margin. Set explicitly to override auto-detection (e.g. a smaller window if the server limits it).",
    )
    context_margin: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
        description="Fraction of the model's max_model_len used as the per-request token budget when auto-detecting. Headroom absorbs tokenizer overestimation and the input=[[conv]] structure overhead.",
    )
    min_pixels: int | None = Field(
        default=None,
        ge=1,
        description="Minimum image pixels the processor keeps (Qwen3-VL preprocessor min_pixels). None (default) uses 4096. Consulted only when image_budget resolves to the pixel budget.",
    )
    max_pixels: int | None = Field(
        default=None,
        ge=1,
        description="Maximum image pixels the processor accepts before images are downscaled (Qwen3-VL preprocessor max_pixels). None (default) uses 1_310_720, mirroring Qwen/Qwen3-VL-Embedding-2B preprocessor_config.json; Instruct variants may use 1_505_280 — the client budget must stay <= the server budget. Consulted only when image_budget resolves to the pixel budget.",
    )
    image_budget: Literal["auto", "pixels", "grid_tokens"] = Field(
        default="auto",
        description="Which image budget guards downscaling. 'auto' uses the pixel budget when the model name contains 'qwen' (case-insensitive) and the legacy grid-token guard (max_image_tokens) otherwise; 'pixels' forces the pixel budget; 'grid_tokens' forces the legacy Jina grid-token guard.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "vLLM Multimodal Embedder",
            "description": "Multimodal embeddings via vLLM's OpenAI-compatible /v1/embeddings endpoint. Sends text chunks and image data URIs as the batch-chat form; returns one vector per input. Auto-detects the model's max_model_len (context length) from /v1/models to keep requests within the running window. Supports Jina v5 omni (and other VLM) embedding models.",
            "link": "https://docs.vllm.ai/en/latest/models/pooling_models/embed.html",
        }
    )

    @classmethod
    def pyclass(cls) -> Type[CustomVllmMultimodalEmbedder]:
        return CustomVllmMultimodalEmbedder
