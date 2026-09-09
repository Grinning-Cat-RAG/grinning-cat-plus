from typing import Any, List
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fields that are NOT ChatOpenAI fields: popped from the constructor kwargs and
# assigned after the Pydantic init for runtime use (multimodal dispatch,
# context-window splitting, accounting). Module-level on purpose: an underscore
# class variable would be treated by Pydantic as a private attribute.
_OPENROUTER_EXTRA_FIELDS = (
    "is_multimodal",
    "max_token_context",
    "input_modalities",
    "prompt_cost_per_1m",
    "completion_cost_per_1m",
    "input_cache_read_cost_per_1m",
    "request_cost",
)


class CustomOpenAI(ChatOpenAI):
    def __init__(self, **kwargs):
        super().__init__(model_kwargs={}, **kwargs)


class CustomOllama(ChatOllama):
    def __init__(self, **kwargs: Any) -> None:
        if kwargs.get("base_url", "").endswith("/"):
            kwargs["base_url"] = kwargs["base_url"][:-1]
        super().__init__(**kwargs)


class OpenRouterLLM(CustomOpenAI):
    """ChatOpenAI bound to OpenRouter's OpenAI-compatible API.

    The ``LLMOpenRouterConfig`` settings carry extra fields that LangChain's
    ``ChatOpenAI`` does not know about (``is_multimodal``, ``max_token_context``,
    the per-1M-token costs used for accounting, ...). This class pops those fields
    out of the constructor kwargs before delegating to ``ChatOpenAI``, then assigns
    them back so the rest of the framework (multimodal dispatch, context-window
    splitting, accounting) can read the model capabilities off the instance at
    runtime.

    ``max_completion_tokens`` is deliberately not among them: it is a real
    ``ChatOpenAI`` field (alias of ``max_tokens``) and must reach LangChain.
    """

    # Declared so that the post-init `setattr` below is accepted: Pydantic v2
    # rejects assignment to an undeclared name. Assignment does NOT revalidate
    # (`validate_assignment` is False by default), so the stored values reach the
    # instance exactly as the config produced them. They do not leak into the
    # request payload, which ChatOpenAI builds from its own params.
    is_multimodal: bool | None = None
    max_token_context: int | None = None
    input_modalities: List[str] | None = None
    prompt_cost_per_1m: float | None = None
    completion_cost_per_1m: float | None = None
    input_cache_read_cost_per_1m: float | None = None
    request_cost: float | None = None

    def __init__(self, **kwargs: Any) -> None:
        # Pop model-capability fields BEFORE the Pydantic init: they are not
        # ChatOpenAI constructor arguments.
        extra = {field: kwargs.pop(field, None) for field in _OPENROUTER_EXTRA_FIELDS}

        # OpenRouter attribution headers (optional, recommended by OpenRouter).
        referer = kwargs.pop("referer", None)
        site_title = kwargs.pop("site_title", None)
        default_headers = dict(kwargs.pop("default_headers", None) or {})
        if referer:
            default_headers["HTTP-Referer"] = referer
        if site_title:
            default_headers["X-Title"] = site_title
        if default_headers:
            kwargs["default_headers"] = default_headers

        kwargs.setdefault("base_url", OPENROUTER_BASE_URL)

        # Pydantic init: must complete before touching instance attributes.
        # NOTE: do NOT pass model_kwargs here — CustomOpenAI.__init__ already
        # injects model_kwargs={} into the ChatOpenAI constructor.
        super().__init__(**kwargs)

        # Store the capability fields AFTER the Pydantic init.
        for field, value in extra.items():
            setattr(self, field, value)
