from typing import Any
from langchain_ollama import ChatOllama
from langchain_openai.chat_models import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Fields that are NOT ChatOpenAI fields: popped from the constructor kwargs and
# stored as plain instance attributes for runtime use (multimodal dispatch,
# context-window splitting, accounting). Module-level on purpose: an underscore
# class variable would be treated by Pydantic as a private attribute and would
# not be readable through `self` before the Pydantic init has completed.
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
    out of the constructor kwargs before delegating to ``ChatOpenAI``, then stores
    them as plain attributes so the rest of the framework (multimodal dispatch,
    context-window splitting, accounting) can read the model capabilities at
    runtime.

    ``max_completion_tokens`` is intentionally NOT popped: it is a real
    ``ChatOpenAI`` field (alias of ``max_tokens``) and must reach LangChain.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Pop model-capability fields BEFORE the Pydantic init: ChatOpenAI would
        # reject unknown kwargs, and pydantic attributes are not writable before
        # the init has completed.
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

        # Store the capability fields AFTER the Pydantic init, bypassing Pydantic's
        # attribute interception (works regardless of extra config).
        for field, value in extra.items():
            object.__setattr__(self, field, value)