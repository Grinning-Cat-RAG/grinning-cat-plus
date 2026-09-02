"""OpenRouter model catalog support.

The OpenRouter public catalog (GET https://openrouter.ai/api/v1/models) lists every
model id exposed by the provider together with its capabilities (context length,
max completion tokens, input modalities) and its per-token pricing. This module
fetches and normalizes that catalog so that MyPLUS can:

* build an ``LLMOpenRouterConfig`` whose ``model`` field is a ``Literal`` of the
  real, currently-available model ids (rendered by the admin as a searchable combo);
* validate (and enrich) the payload on save, storing in the settings the model
  capabilities and the costs needed for accounting.

Network calls are cached in-memory with a TTL: the catalog changes rarely and the
settings schema is regenerated on every ``GET /llm/settings``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from cat.log import log

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Catalog TTL: 15 minutes.
_CATALOG_TTL_SECONDS = 15 * 60

# In-memory cache shared across agents (single-process assumption, like the rest of
# the in-process caches in the Cat). Keyed by nothing: the catalog is global.
_catalog_cache: Tuple[float, Optional[List[Dict[str, Any]]]] = (0.0, None)
# Whether the last catalog fetch succeeded (distinguishes "catalog down" from
# "model truly absent" on the save path).
_last_fetch_ok: bool = False

# Settings fields -> OpenRouter pricing keys. The API returns USD per token; the
# settings store the per-1M values (readable, stable) for accounting.
_PRICING_FIELDS: Dict[str, str] = {
    "prompt_cost_per_1m": "prompt",
    "completion_cost_per_1m": "completion",
    "input_cache_read_cost_per_1m": "input_cache_read",
}


def _parse_catalog(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize the raw OpenRouter /models payload into a compact list.

    Each entry keeps, for a model id, only the fields MyPLUS consumes:

    * ``id`` — the OpenRouter model id (e.g. ``qwen/qwen3.8-27b``)
    * ``context_length`` — the model context window (input tokens)
    * ``max_completion_tokens`` — the provider's max output tokens
      (``top_provider.max_completion_tokens``), when advertised
    * ``input_modalities`` — ``architecture.input_modalities`` (e.g.
      ``["text"]``, ``["text", "image", "video"]``)
    * ``pricing`` — flattened per-1M-token costs: ``prompt_cost_per_1m``,
      ``completion_cost_per_1m``, ``input_cache_read_cost_per_1m``,
      ``request_cost`` (each ``float | None``)

    Models whose ``id`` is missing are dropped; models with a ``context_length``
    of ``None`` are kept (the field is nullable).
    """
    catalog: List[Dict[str, Any]] = []
    for model in payload.get("data", []):
        model_id = model.get("id")
        if not model_id:
            continue

        arch = model.get("architecture") or {}
        top_provider = model.get("top_provider") or {}
        pricing = model.get("pricing") or {}

        def _per_1m(key: str) -> Optional[float]:
            raw = pricing.get(key)
            if raw is None:
                return None
            try:
                return round(float(raw) * 1_000_000, 6)
            except (TypeError, ValueError):
                return None

        catalog.append(
            {
                "id": model_id,
                "context_length": model.get("context_length"),
                "max_completion_tokens": top_provider.get("max_completion_tokens"),
                "input_modalities": list(arch.get("input_modalities") or []),
                **{field: _per_1m(api_key) for field, api_key in _PRICING_FIELDS.items()},
                "request_cost": _per_1m("request"),
            }
        )
    return catalog


def fetch_catalog() -> List[Dict[str, Any]]:
    """Fetch and normalize the OpenRouter model catalog.

    Returns the list of model descriptors (see :func:`_parse_catalog`). A network
    or parsing failure logs a warning and returns ``[]`` — callers must treat an
    empty catalog as "unknown", not as "no models".
    """
    global _last_fetch_ok
    try:
        resp = httpx.get(OPENROUTER_MODELS_URL, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        _last_fetch_ok = True
        return _parse_catalog(resp.json())
    except Exception as e:  # noqa: BLE001 - any failure must degrade gracefully
        log.warning(f"OPENROUTER failed to fetch model catalog: {e!r}")
        _last_fetch_ok = False
        return []


def catalog_last_ok() -> bool:
    """Whether the most recent catalog fetch succeeded.

    Lets save-path validation distinguish "catalog unreachable" (don't block the
    user) from "model truly absent from a reachable catalog" (validation error).
    """
    return _last_fetch_ok


def get_catalog(force: bool = False) -> List[Dict[str, Any]]:
    """Return the cached catalog, fetching it on first use or when stale.

    ``force=True`` bypasses the cache (used by the save-path validation). On a
    failed fetch the previous cache is kept (or, if there was none, the next call
    retries immediately): a transient network issue must not poison the cache for
    the whole TTL.
    """
    global _catalog_cache
    cached_at, cached = _catalog_cache
    if force or cached is None or (time.monotonic() - cached_at) > _CATALOG_TTL_SECONDS:
        fresh = fetch_catalog()
        if fresh or catalog_last_ok():
            _catalog_cache = (time.monotonic(), fresh)
    return _catalog_cache[1] or []


def find_model(model_id: str, force: bool = False) -> Optional[Dict[str, Any]]:
    """Look up a single model in the catalog, or ``None`` if absent/unknown."""
    catalog = get_catalog(force=force)
    return next((m for m in catalog if m["id"] == model_id), None)


def model_ids() -> List[str]:
    """All currently-available model ids (for the ``Literal``/``enum``)."""
    return [m["id"] for m in get_catalog()]