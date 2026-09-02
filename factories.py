from typing import List, Type
from cat import hook, BillTheLizard
from cat.db.cruds import settings as crud_settings
from cat.services.factory.chunker import ChunkerSettings
from cat.services.factory.embedder import EmbedderSettings
from cat.services.factory.file_manager import FileManagerConfig
from cat.services.factory.llm import LLMSettings

from .chunkers.configs import (
    SemanticChunkerSettings,
    HTMLSemanticChunkerSettings,
    JSONChunkerSettings,
    TokenSpacyChunkerSettings,
    TokenNLTKChunkerSettings,
    HierarchicalChunkerSettings,
    MathAwareHierarchicalChunkerSettings,
    MathAwareSemanticChunkerSettings,
)
from .embedders.configs import (
    EmbedderQdrantFastEmbedConfig,
    EmbedderOpenAIConfig,
    EmbedderAzureOpenAIConfig,
    EmbedderGeminiChatConfig,
    EmbedderOpenAICompatibleConfig,
    EmbedderCohereConfig,
    EmbedderMistralAIChatConfig,
    EmbedderVoyageAIChatConfig,
    EmbedderOllamaConfig,
    EmbedderJinaConfig,
    Qwen3LocalEmbeddingsConfig,
    Qwen3OllamaEmbeddingsConfig,
    Qwen3DeepInfraEmbeddingsConfig,
    Qwen3TEIEmbeddingsConfig,
    EmbedderJinaMultimodalConfig,
    JinaCLIPEmbeddingsConfig,
    VllmMultimodalConfiguration,
)
from .file_managers.configs import (
    AWSFileManagerConfig,
    AzureFileManagerConfig,
    GoogleFileManagerConfig,
    DigitalOceanFileManagerConfig,
)
from .llms.configs import (
    LLMOpenAIChatConfig,
    LLMOpenAIConfig,
    LLMOpenAICompatibleConfig,
    LLMOllamaConfig,
    LLMGeminiChatConfig,
    LLMCohereConfig,
    LLMAzureOpenAIConfig,
    LLMAzureChatOpenAIConfig,
    LLMHuggingFaceEndpointConfig,
    LLMHuggingFaceTextGenInferenceConfig,
    LLMAnthropicChatConfig,
    LLMMistralAIChatConfig,
    LLMGroqChatConfig,
    LLMOpenRouterBaseConfig,
)
from .llms.openrouter import model_ids


@hook(priority=1)
def factory_allowed_llms(allowed: List[LLMSettings], cat) -> List:
    return allowed + [
        LLMOpenAIChatConfig,
        LLMOpenAIConfig,
        LLMOpenAICompatibleConfig,
        LLMOllamaConfig,
        LLMGeminiChatConfig,
        LLMCohereConfig,
        LLMAzureOpenAIConfig,
        LLMAzureChatOpenAIConfig,
        LLMHuggingFaceEndpointConfig,
        LLMHuggingFaceTextGenInferenceConfig,
        LLMAnthropicChatConfig,
        LLMMistralAIChatConfig,
        LLMGroqChatConfig,
        _build_openrouter_config(),
    ]


def _build_openrouter_config() -> Type[LLMOpenRouterBaseConfig]:
    """Return the OpenRouter config class, with ``model`` carrying the current catalog as UI-only enum.

    OpenRouter's model list is dynamic, so the config class cannot hardcode an enum.
    Every time the settings schema is requested, this builds a subclass of
    ``LLMOpenRouterBaseConfig`` whose ``model`` field exposes the currently-available
    model ids via ``json_schema_extra["enum"]`` — the admin UI renders that as a
    searchable combo.

    The enum is deliberately NOT a ``Literal``: the save path must never be able to
    reject a model id (no server-side validation), and ``get_from_config`` would
    hard-fail on an out-of-catalog id. The enum is UI metadata only.

    The generated class keeps the SAME ``__name__`` (``LLMOpenRouterConfig``) so the
    settings save path (``ServiceFactory`` matches by class name) stays stable across
    refetches. If the catalog is unreachable, the base class (``model: str``, no enum)
    is returned so the user is never locked out.
    """
    from pydantic import Field

    ids = model_ids()
    model_field = str
    if ids:
        model_field = Field(
            description=LLMOpenRouterBaseConfig.model_fields["model"].description,
            json_schema_extra={"enum": ids},
        )
        return type(
            "LLMOpenRouterConfig",
            (LLMOpenRouterBaseConfig,),
            {
                "__annotations__": {"model": str},
                "model": model_field,
                "__module__": LLMOpenRouterBaseConfig.__module__,
            },
        )
    return LLMOpenRouterBaseConfig


@hook(priority=1)
def factory_allowed_embedders(allowed: List[EmbedderSettings], lizard) -> List:
    return allowed + [
        EmbedderQdrantFastEmbedConfig,
        EmbedderOpenAIConfig,
        EmbedderAzureOpenAIConfig,
        EmbedderGeminiChatConfig,
        EmbedderOpenAICompatibleConfig,
        EmbedderCohereConfig,
        EmbedderMistralAIChatConfig,
        EmbedderVoyageAIChatConfig,
        EmbedderOllamaConfig,
        EmbedderJinaConfig,
        Qwen3LocalEmbeddingsConfig,
        Qwen3OllamaEmbeddingsConfig,
        Qwen3DeepInfraEmbeddingsConfig,
        Qwen3TEIEmbeddingsConfig,
        EmbedderJinaMultimodalConfig,
        JinaCLIPEmbeddingsConfig,
        VllmMultimodalConfiguration,
    ]


@hook(priority=1)
def factory_allowed_file_managers(allowed: List[FileManagerConfig], cat) -> List:
    return allowed + [
        AWSFileManagerConfig,
        AzureFileManagerConfig,
        GoogleFileManagerConfig,
        DigitalOceanFileManagerConfig,
    ]


@hook(priority=1)
def factory_allowed_chunkers(allowed: List[ChunkerSettings], cat) -> List:
    return allowed + [
        SemanticChunkerSettings,
        HTMLSemanticChunkerSettings,
        JSONChunkerSettings,
        TokenSpacyChunkerSettings,
        TokenNLTKChunkerSettings,
        HierarchicalChunkerSettings,
        MathAwareHierarchicalChunkerSettings,
        MathAwareSemanticChunkerSettings,
    ]


@hook(priority=1)
async def lizard_notify_plugin_installation(plugin_id: str, plugin_path: str, lizard: BillTheLizard):
    this_plugin_id = lizard.mad_hatter.get_plugin().id
    if this_plugin_id != plugin_id:
        return

    # for each Cheshire Cat, activate this plugin
    ccat_ids = await crud_settings.get_agents_main_keys()
    for ccat_id in ccat_ids:
        if (ccat := await lizard.get_cheshire_cat(ccat_id)) is None:
            continue

        await ccat.plugin_manager.toggle_plugin(plugin_id)
