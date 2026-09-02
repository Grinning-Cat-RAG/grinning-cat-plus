import base64
import json
import os
from typing import Any, Dict, List
import httpx
import requests
from sentence_transformers import SentenceTransformer

from langchain_community.embeddings import FastEmbedEmbeddings
from cat import Embeddings, MultimodalEmbeddings
from cat.utils import retrieve_image

from cat import log
from cat.db.database import DEFAULT_SYSTEM_KEY, get_sync_db
from cat.db.cruds.settings import format_key

_EMBEDDERS_MODELS_CACHE = {}


class CustomFastEmbedEmbeddings(Embeddings):
    """Wrapper for FastEmbedEmbeddings that inherits from cat.Embeddings.

    FastEmbedEmbeddings from langchain_community inherits from langchain_core.embeddings.Embeddings,
    which is a sibling (not parent) of cat.Embeddings. The factory validation in
    BaseFactoryConfigModel.get_from_config() requires issubclass(pyclass, cat.Embeddings),
    so a wrapper is needed to pass the check.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en",
        max_length: int = 512,
        doc_embed_type: str = "passage",
        cache_dir: str = "cat/data/models/fast_embed",
        max_input_tokens: int | None = None,
    ):
        self.max_input_tokens = max_input_tokens or max_length
        self._inner = FastEmbedEmbeddings(
            model_name=model_name,
            max_length=max_length,
            doc_embed_type=doc_embed_type,
            cache_dir=cache_dir,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._inner.embed_query(text)


class CustomOpenAIEmbeddings(Embeddings):
    """Use OpenAI-compatible API as embedder (like llama-cpp-python)."""
    def __init__(self, url: str, model: str, api_key: str | None = None, max_input_tokens: int | None = None):
        self.url = os.path.join(url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.max_input_tokens = max_input_tokens

    @property
    def headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # OpenAI API expects JSON payload, not form data
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": texts, "model": self.model},
            timeout=300,
        )
        response.raise_for_status()

        to_return = [e["embedding"] for e in response.json()["data"]]
        return to_return

    def embed_query(self, text: str) -> List[float]:
        # OpenAI API expects JSON payload, not form data
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": text, "model": self.model},
            timeout=300,
        )
        response.raise_for_status()

        to_return = response.json()["data"][0]["embedding"]
        return to_return


class CustomOllamaEmbeddings(Embeddings):
    """Use Ollama to serve embedding models."""
    def __init__(self, base_url: str, model: str, max_input_tokens: int | None = None):
        self.url = os.path.join(base_url, "api/embeddings")
        self.model = model
        self.max_input_tokens = max_input_tokens

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Ollama doesn't support batch processing, so we need to process one by one
        embeddings = []
        for text in texts:
            ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
            ret.raise_for_status()
            embeddings.append(ret.json()["embedding"])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        ret = httpx.post(self.url, json={"model": self.model, "prompt": text}, timeout=300.0)
        ret.raise_for_status()
        return ret.json()["embedding"]


class CustomJinaEmbedder(Embeddings):
    """Use Jina AI to serve embedding models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching", max_input_tokens: int | None = None):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task
        self.max_input_tokens = max_input_tokens

    def _embed(self, texts: List[str]) -> List[List[float]]:
        ret = httpx.post(
            self.url,
            data={"model": self.model, "input": texts, "task": self.task},
            timeout=300.0,
            headers=self.headers,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]


class Qwen3LocalEmbeddings(Embeddings):
    """
    Local Qwen3 embeddings using HuggingFace Sentence Transformers.
    Best for: Full control, no external dependencies, offline usage
    """
    def __init__(self, model_name: str, max_input_tokens: int | None = None):
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens

    def _load_model(self) -> SentenceTransformer:
        """Lazy load the model"""
        global _EMBEDDERS_MODELS_CACHE

        model = _EMBEDDERS_MODELS_CACHE.get("Qwen3LocalEmbeddings", {}).get(self.model_name)
        if model is None:
            model = SentenceTransformer(self.model_name, trust_remote_code=True)
            _EMBEDDERS_MODELS_CACHE.setdefault("Qwen3LocalEmbeddings", {})[self.model_name] = model
        return model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        model = self._load_model()
        embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return embedding.tolist()


class Qwen3OllamaEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Ollama.
    Best for: Easy local deployment, minimal setup
    """
    def __init__(self, model_name: str, base_url: str, max_input_tokens: int | None = None):
        self.model_name = model_name
        self.base_url = base_url
        self.max_input_tokens = max_input_tokens

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embedding(text)


class Qwen3DeepInfraEmbeddings(Embeddings):
    """
    Qwen3 embeddings via DeepInfra API (OpenAI-compatible).
    Best for: Production deployment, no GPU required, pay-as-you-go
    """
    def __init__(self, model_name: str, base_url: str, api_key: str, max_input_tokens: int | None = None):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.max_input_tokens = max_input_tokens

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from DeepInfra"""
        if not self.api_key:
            raise ValueError("DeepInfra API key is required")

        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "input": texts,
                    "model": self.model_name,
                    "encoding_format": "float"
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            # Sort by index to maintain order
            sorted_embeddings = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_embeddings]
        except requests.RequestException as e:
            raise RuntimeError(f"DeepInfra embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        # DeepInfra supports batch processing
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class Qwen3TEIEmbeddings(Embeddings):
    """
    Qwen3 embeddings via Text Embeddings Inference (self-hosted).
    Best for: High-throughput production, full control, optimized inference
    """
    def __init__(self, base_url: str, max_input_tokens: int | None = None):
        self.base_url = base_url
        self.max_input_tokens = max_input_tokens

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from TEI server"""
        try:
            response = requests.post(
                f"{self.base_url}/embed",
                json={"inputs": texts},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"TEI embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._get_embeddings([text])[0]


class CustomJinaMultimodalEmbedder(MultimodalEmbeddings):
    """Use Jina AI to serve embedding multimodal models."""
    def __init__(self, base_url: str, model: str, api_key: str, task: str = "text-matching", max_input_tokens: int | None = None):
        self.url = os.path.join(base_url, "v1/embeddings")
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task
        self.max_input_tokens = max_input_tokens

    def _embed(
        self,
        texts: List[str] | None = None,
        images: List[str | bytes] | None = None
    ) -> List[List[float]]:
        def parse_image(image: str | bytes) -> str:
            if isinstance(image, bytes):
                return base64.b64encode(image).decode("utf-8")
            image = retrieve_image(image)
            # remove "data:image/...;base64," prefix if present
            if image is not None and image.startswith("data:image"):
                image = image.split(",", 1)[1]
            return image

        payload = (
            [{"text": t} for t in texts] if texts else []
        ) + (
            [{"image": parse_image(i)} for i in images if i] if images else []
        )

        if not payload:
            return []

        ret = httpx.post(
            self.url,
            json={"model": self.model, "input": payload, "task": self.task},
            headers=self.headers,
            timeout=300.0,
        )
        ret.raise_for_status()
        return [e["embedding"] for e in ret.json()["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed(texts=[text])[0]

    def embed_image(self, image: str | bytes) -> List[float]:
        return self._embed(images=[image])[0]

    def embed_images(self, images: List[str | bytes]) -> List[List[float]]:
        return self._embed(images=images)


class JinaCLIPEmbeddings(MultimodalEmbeddings):
    """
    Jina CLIP v2 multimodal embeddings.
    Handles both text and images in same vector space.
    """
    def __init__(self, api_key: str, model_name: str, base_url: str, max_input_tokens: int | None = None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.max_input_tokens = max_input_tokens

    def _get_embeddings(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Get embeddings from Jina API.

        Args:
            inputs: List of {"text": str} or {"image": bytes/url}
        """
        if not self.api_key:
            raise ValueError("Jina API key required")

        try:
            # Prepare input for Jina API
            prepared_inputs = []
            for inp in inputs:
                if "text" in inp:
                    prepared_inputs.append({"text": inp["text"]})
                elif "image" in inp:
                    # Handle image bytes or URL
                    tmp = inp["image"]
                    if isinstance(inp["image"], bytes):
                        img_b64 = base64.b64encode(inp["image"]).decode()
                        tmp = f"data:image/png;base64,{img_b64}"

                    prepared_inputs.append({"image": tmp})

            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": self.model_name,
                    "input": prepared_inputs
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            return [item["embedding"] for item in data["data"]]

        except requests.RequestException as e:
            raise RuntimeError(f"Jina embedding failed: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents"""
        inputs = [{"text": text} for text in texts]
        return self._get_embeddings(inputs)

    def embed_query(self, text: str) -> List[float]:
        """Embed single text query"""
        return self._get_embeddings([{"text": text}])[0]

    def embed_image(self, image: bytes) -> List[float]:
        """Embed single image"""
        return self._get_embeddings([{"image": image}])[0]

    def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """Embed multiple images"""
        inputs = [{"image": img} for img in images]
        return self._get_embeddings(inputs)


class CustomVllmMultimodalEmbedder(MultimodalEmbeddings):
    """Multimodal embeddings via vLLM's OpenAI-compatible /v1/embeddings endpoint.

    Uses vLLM's batch-chat request form: ``input`` is a list of conversations,
    one per item, each a single ``user`` message whose ``content`` is typed
    parts (``{"type": "text", "text": ...}`` / ``{"type": "image_url",
    "image_url": {"url": "data:...;base64,<...>"}}``). vLLM returns ONE vector
    per conversation (aligned by ``index``) and decodes images as real images
    (vision tokens) — required by the chunkers that call ``embed_documents`` on
    many chunks and by ``embed_images`` for PDF-extracted images.

    NOTE: the plain string-list ``input`` form is NOT usable for images: vLLM
    tokenizes the whole base64 URI as text, blowing the context window, and the
    single-conversation chat form pools the whole request into one vector.
    """

    # magic-byte prefixes -> mime type, used to build data URIs for bytes input
    _MAGIC_TO_MIME = (
        (b"\x89PNG", "image/png"),
        (b"\xff\xd8", "image/jpeg"),
        (b"GIF8", "image/gif"),
        (b"RIFF", "image/webp"),
        (b"BM", "image/bmp"),
    )

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        task: str | None = None,
        timeout: float = 300.0,
        max_image_tokens: int = 2048,
        query_prefix: str = "Query: ",
        document_prefix: str = "Document: ",
        max_input_tokens: int | None = None,
        context_margin: float = 0.9,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        image_budget: str = "auto",
    ):
        self.base_url = base_url.rstrip("/")
        self.url = self.base_url + "/v1/embeddings"
        self.model = model
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.task = task
        self.timeout = timeout
        # Max image tokens the embedding model accepts (Jina v5 omni: ~2048).
        # Other multimodal models may accept more; used by the downscaler.
        self.max_image_tokens = max_image_tokens
        # Qwen3-VL pixel budget (mirrors Qwen/Qwen3-VL-Embedding-2B
        # preprocessor_config.json). None -> effective defaults applied at
        # resize time: min_pixels=4096, max_pixels=1_310_720. Instruct variants
        # may raise the server budget to 1_505_280; the client budget must stay
        # <= the server budget.
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        # Budget discriminator: "auto" = pixel budget when the model name
        # contains "qwen" (case-insensitive), else the legacy Jina grid-token
        # guard; "pixels" forces the pixel budget; "grid_tokens" forces the
        # legacy path. Keeps Jina byte-identical while protecting Qwen3-VL
        # out-of-the-box.
        self.image_budget = image_budget
        # Retrieval-side prefixes (Jina v5 defaults). Queries vs documents are
        # embedded differently by retrieval models; prepending the matching side
        # prefix aligns with encode_query()/encode_document(). Overridable for
        # other models (e.g. "Passage: " for document side).
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        # Token budget semantics:
        #  - max_input_tokens is None -> the value is UNRESOLVED: it means "ask
        #    the embedder (vLLM /v1/models) for the model's real max_model_len"
        #    and derive the safe per-request budget from it.
        #  - max_input_tokens is an int -> an explicit admin override: the raw
        #    model window is that value; the same safety factor still applies.
        # The resolution is lazy (see _ensure_max_input_tokens): if the model
        # endpoint is not reachable yet at construction, we keep None and retry
        # on first embed, so a transient startup race never bakes in a wrong
        # budget. Exposed publicly so the rabbit-hole oversized-split and
        # budget-aware chunkers can size chunks to this embedder's ceiling.
        self.agent_id: str | None = None
        self._last_requested_max_model_len: int | None = None
        self._persist_pending = False
        self._persisted = False
        self._context_margin = context_margin
        self._override_max_input_tokens = max_input_tokens
        # real tokenizer (lazy): None until _load_tokenizer() runs
        self._tokenizer = None
        self.max_input_tokens: int | None = self._resolve_initial_max_input_tokens(
            override=max_input_tokens,
            margin=context_margin,
        )
        if self.max_input_tokens is None:
            log.warning(
                "VLLM_EMBEDDINGS max_input_tokens unset and /v1/models not "
                "reachable at init: will ask the embedder for the model's max "
                "input size on first embed."
            )
        # internal alias used by the request batching below
        self._max_input_tokens = self.max_input_tokens

    def _fetch_max_model_len(self) -> int | None:
        """Read the loaded model's context length from vLLM's /v1/models.

        Returns ``max_model_len`` (post auto-fit) for ``self.model``, or the
        first entry as a fallback; ``None`` if the endpoint is unreachable or
        the value is missing. vLLM exposes the effective window here (not raw
        HF keys), which is exactly the cap it enforces on /v1/embeddings.
        """
        try:
            resp = httpx.get(
                self.base_url + "/v1/models",
                headers=self.headers,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.warning(
                    f"VLLM_EMBEDDINGS /v1/models status {resp.status_code}; "
                    f"using fallback token budget"
                )
                return None
            data = resp.json().get("data") or []
            for entry in data:
                if entry.get("id") == self.model:
                    return entry.get("max_model_len")
            if data:
                # unknown model id: assume the first served one rather than failing
                return data[0].get("max_model_len")
        except Exception as exc:  # noqa: BLE001 - network hiccup, fall back to override
            log.warning(f"VLLM_EMBEDDINGS failed to query /v1/models: {exc}")
        return None

    def _resolve_initial_max_input_tokens(
        self,
        override: int | None,
        margin: float,
    ) -> int | None:
        """Best-effort resolution of the per-request budget at construction.

        When ``override`` is None we ASK the embedder (``/v1/models``) for the
        model's real ``max_model_len`` and derive the budget from it. If the
        endpoint is not reachable yet, return ``None`` (unresolved): the same
        question is re-asked lazily on first embed via
        ``_ensure_max_input_tokens``, so a transient startup race never freezes
        a wrong or invented budget.

        With exact token counting (the model's real tokenizer, see
        ``_estimate_tokens``) no conservative fudge factor is needed: the raw
        window scaled by ``margin`` is a hard, exact ceiling.
        """
        if override is not None:
            return max(1, int(override * margin))
        max_model_len = self._fetch_max_model_len()
        if max_model_len is None:
            return None
        self._last_requested_max_model_len = max_model_len
        self._persist_pending = True
        budget = max(1, int(max_model_len * margin))
        log.info(
            f"VLLM_EMBEDDINGS max_input_tokens not set: asked embedder "
            f"({self.base_url}/v1/models) -> max_model_len={max_model_len}, "
            f"applying margin={margin} -> max_input_tokens={budget}"
        )
        return budget

    def _persist_auto_budget_to_config(self) -> None:
        """Persist an auto-detected context budget to the system settings.

        One-shot and fire-and-forget: runs at most once per instance, never
        overwrites an explicit admin override or an already-stored value, and
        never blocks embedding on a Redis failure.
        """
        if self._override_max_input_tokens is not None:
            return
        if self._persisted:
            return
        raw = self._last_requested_max_model_len
        if raw is None:
            return
        try:
            key = format_key(DEFAULT_SYSTEM_KEY)  # "system:agent"
            path = '$[?(@.name=="VllmMultimodalConfiguration")].value.max_input_tokens'
            # sync handle is core-owned (cat.db.database.get_sync_db); the
            # DB-swap seam for settings lives in the core, so plugins use this
            # shared handle rather than opening their own connection
            db = get_sync_db()
            res = db.json().get(key, path)
            current = res[0] if isinstance(res, list) and res else None
            if current is not None:
                self._persisted = True
                return
            db.json().set(key, path, raw)
            log.debug(
                f"VLLM_EMBEDDINGS persisted auto-detected "
                f"max_input_tokens={raw} to {key}"
            )
            self._persisted = True
        except Exception as exc:  # noqa: BLE001 - never block embedding on Redis
            log.warning(
                f"VLLM_EMBEDDINGS failed to persist auto-detected "
                f"max_input_tokens: {exc}"
            )

    def _ensure_max_input_tokens(self) -> None:
        """Ask the embedder for the model's input size if not resolved yet.

        Called before every embed when ``max_input_tokens`` is still ``None``
        (unresolved). Retries ``/v1/models`` -- this heals the transient
        startup race where the vLLM server was not ready when the embedder was
        constructed. If it STILL fails, logs loudly and falls back to a
        conservative 16384 hard budget (provably safe for any context window
        >= 32768; avoids inventing a number larger than the real one).
        """
        if self.max_input_tokens is not None:
            # already resolved (at init or a previous call): flush now
            self._persist_auto_budget_to_config()
            return
        budget = self._resolve_initial_max_input_tokens(
            override=self._override_max_input_tokens,
            margin=self._context_margin,
        )
        if budget is None:
            # conservative fallback: never larger than the models we serve
            budget = int(0.5 * 0.9 * 32768)  # ~14745, safe for any 32k-window model
            log.warning(
                f"VLLM_EMBEDDINGS still cannot reach /v1/models; using "
                f"conservative max_input_tokens={budget}"
            )
        self.max_input_tokens = budget
        self._max_input_tokens = budget
        self._persist_auto_budget_to_config()

    def _load_tokenizer(self):
        """Lazily obtain the model's REAL tokenizer for exact token counting.

        The model usually lives on a DIFFERENT host (the vLLM server), so we
        cannot assume a local HF cache. Resolution order:

        1. local HF cache (``AutoTokenizer``) if the model is cached here;
        2. the vLLM server's own ``/tokenize`` endpoint -- the authoritative
           tokenizer of the serving host, used at runtime to count each text;
        3. ``None`` -> callers fall back to a conservative chars/2 estimate.

        The result is memoized; the remote fallback transparently marks that
        counting must go through the HTTP endpoint (see _estimate_text_tokens).
        """
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=True
            )
            log.info(f"VLLM_EMBEDDINGS loaded real tokenizer for {self.model} (local cache)")
            return self._tokenizer
        except Exception as exc:  # noqa: BLE001 - non-fatal, fall back to remote
            log.warning(
                f"VLLM_EMBEDDINGS no local tokenizer for {self.model} "
                f"(model is remote): will use the vLLM /tokenize endpoint. ({exc})"
            )
            self._tokenizer = "remote"  # sentinel: count via the server
        return self._tokenizer

    def _estimate_text_tokens(self, text: str) -> int:
        """EXACT token count of ``text`` using the model's real tokenizer.

        Uses the local HF tokenizer when available; otherwise asks the remote
        vLLM server's ``/tokenize`` endpoint, which returns the exact count
        (``{"count": N, ...}``) for the model it serves. Falls back to a
        conservative ~2 chars/token estimate only when both are unavailable.
        """
        tok = self._tokenizer
        if tok is None:
            tok = self._load_tokenizer()

        if tok is not None and tok != "remote":
            try:
                return max(1, len(tok.encode(text)))
            except Exception:  # noqa: BLE001 - non-fatal, fall back to remote
                pass

        if tok == "remote" or tok is None:
            try:
                r = httpx.post(
                    self.base_url + "/tokenize",
                    json={"model": self.model, "prompt": text},
                    headers=self.headers,
                    timeout=min(self.timeout, 10),
                )
                if r.status_code == 200:
                    count = r.json().get("count")
                    if count:
                        return max(1, int(count))
            except Exception as exc:  # noqa: BLE001 - non-fatal
                log.debug(f"VLLM_EMBEDDINGS /tokenize failed: {exc}")

        return max(1, len(text) // 2)

    def _estimate_tokens(self, text: str) -> int:
        """Soldier base override: real-tokenizer counting for core/chunker.

        The core rabbit-hole's oversized-split and the semantic chunker both
        call ``embedder._estimate_tokens``: giving them the SAME exact count
        the model uses keeps chunking, batching and splitting consistent.
        """
        return self._estimate_text_tokens(text)

    def _split_batches(self, items: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split items into request batches that stay within the context window.

        Images go one per request because their token cost scales with pixels and
        cannot be estimated cheaply; texts are grouped up to the token budget.

        ``_max_input_tokens`` already embeds the safety factor against the
        char-based estimator, so the same ceiling keeps a single chunk safe when
        sent alone AND keeps the accumulated batch within the model window.
        """
        batch_budget = self._max_input_tokens

        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_tokens = 0

        for it in items:
            if "image" in it:
                if current:
                    batches.append(current)
                    current, current_tokens = [], 0
                batches.append([it])
                continue

            tokens = self._estimate_text_tokens(it.get("text", ""))
            if current and current_tokens + tokens > batch_budget:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(it)
            current_tokens += tokens

        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _sniff_mime(data: bytes) -> str:
        for magic, mime in CustomVllmMultimodalEmbedder._MAGIC_TO_MIME:
            if data.startswith(magic):
                return mime
        return "image/png"

    def _uses_pixel_budget(self) -> bool:
        """Resolve the image_budget discriminator to a concrete budget kind.

        ``auto`` picks the pixel budget when the model name contains "qwen"
        (case-insensitive) and the legacy grid-token guard otherwise;
        ``pixels`` and ``grid_tokens`` force their respective budget.
        """
        if self.image_budget == "pixels":
            return True
        if self.image_budget == "grid_tokens":
            return False
        # auto
        return "qwen" in self.model.lower()

    def _image_grid_tokens(self, width: int, height: int) -> int:
        """Estimate the number of image tokens the Jina processor will produce.

        Jina-calibrated grid-token estimate: each side is rounded to a multiple
        of patch_size(16) * merge_size(2) = 32, then divided by 16 to get the
        grid dims; tokens = grid_h * grid_w. This is the legacy formula used by
        the Jina grid-token guard (max_image_tokens=2048) and MUST stay
        byte-identical to preserve Jina v5 behavior. The Qwen pixel-budget path
        does NOT use this method (it uses an area check).
        """
        grid_w = round(width / 32) * 2
        grid_h = round(height / 32) * 2
        return grid_w * grid_h

    def _resize_image_if_needed(self, image: bytes) -> bytes:
        """Downscale an image so its processor budget fits the model.

        Two budgets, selected by ``_uses_pixel_budget``:

        - Pixel budget (Qwen3-VL): if the raw area ``w*h`` exceeds
          ``max_pixels`` (effective default 1_310_720), scale preserving aspect
          ratio so the output area is <= ``max_pixels``.
        - Grid-token budget (Jina v5 omni): the embedding model rejects images
          whose processor grid exceeds ~2048 image tokens (observed: <=2024 OK,
          >=2052 FAIL), regardless of aspect ratio. Preserve aspect ratio and
          shrink just enough to land under ``self.max_image_tokens``.

        Falls back to the original bytes if PIL is unavailable or the decode
        fails.
        """
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image))
            img.load()
            w, h = img.size
            if self._uses_pixel_budget():
                max_pixels = self.max_pixels if self.max_pixels is not None else 1_310_720
                if w * h <= max_pixels:
                    return image
                # scale preserving aspect so area <= max_pixels (floor keeps
                # the invariant exact: floor(w*s)*floor(h*s) <= w*h*s*s)
                scale = (max_pixels / (w * h)) ** 0.5
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                img = img.resize((nw, nh), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue()

            # legacy Jina grid-token guard (byte-identical behavior)
            if self._image_grid_tokens(w, h) <= self.max_image_tokens:
                return image

            # binary search the largest scale keeping tokens under the budget
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                nw = max(1, round(w * mid))
                nh = max(1, round(h * mid))
                if self._image_grid_tokens(nw, nh) <= self.max_image_tokens:
                    lo = mid
                else:
                    hi = mid

            nw = max(1, round(w * lo))
            nh = max(1, round(h * lo))
            img = img.resize((nw, nh), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        except Exception as exc:  # noqa: BLE001 - best-effort resize
            log.debug(f"VLLM_EMBEDDINGS image resize skipped: {exc}")
            return image

    def _to_data_uri(self, image: str | bytes) -> str:
        """Return a full ``data:<mime>;base64,<...>`` URI for image input."""
        if isinstance(image, bytes):
            image = self._resize_image_if_needed(image)
            return f"data:{self._sniff_mime(image)};base64,"\
                   f"{base64.b64encode(image).decode('utf-8')}"
        uri = retrieve_image(image)
        if isinstance(uri, bytes):
            return self._to_data_uri(uri)
        if uri is None:
            raise ValueError(f"Unable to read image: {image!r}")
        uri = str(uri)
        if uri.startswith("data:"):
            return uri
        # treat as raw base64 (e.g. retrieve_image returned bare base64)
        return f"data:image/png;base64,{uri}"

    @staticmethod
    def _prefixed(text: str, prefix: str) -> str:
        """Join a prefix and the text ensuring they are space-separated."""
        text = text.strip()
        if not prefix or not text:
            return text
        return prefix.rstrip() + " " + text if text else prefix

    def _embed(
        self,
        items: List[Dict[str, Any]],
        prefix: str = "",
    ) -> List[List[float]]:
        if not items:
            return []

        # vLLM batch-chat form: input = one conversation per item; each
        # conversation is a single message whose content is typed parts
        # (text / image_url). This returns ONE vector per conversation and
        # decodes images as real images (vision tokens), not as base64 text.
        conversations = []
        for it in items:
            content = []
            if "text" in it:
                text = it["text"] if it["text"] is not None else ""
                # vLLM rejects empty prompts; placeholder keeps alignment
                text = text if text.strip() else " "
                content.append({"type": "text", "text": self._prefixed(text, prefix)})
            elif "image" in it:
                # The retrieval prefix (Query:/Document:) applies to media too:
                # prepend it as a text part right before the image (the chat
                # template renders it next to the media placeholder).
                if prefix and prefix.strip():
                    content.append({"type": "text", "text": prefix.rstrip()})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._to_data_uri(it["image"])},
                })
                # vLLM builds an mm-only dummy prompt when a message is
                # image-only, which some processors (e.g. Qwen3-VL) reject
                # with a 400. Always pair the image with a real non-whitespace
                # text part so the chat template never emits an image-only
                # message. When a prefix is present it already serves as that
                # text part (before the image); otherwise append a placeholder
                # after the image.
                if not prefix or not prefix.strip():
                    content.append({"type": "text", "text": "Document"})
            conversations.append({"role": "user", "content": content})

        payload = {"model": self.model, "input": [[c] for c in conversations]}

        # Debug-only request log: build it only when DEBUG is enabled (cheap
        # check on the cat log engine's configured level) and keep each text to
        # its first 100 chars (images ~20 b64 chars) so the line stays small
        # even for huge batches.
        if log.LOG_LEVEL == "DEBUG":
            debug_input = []
            for conv in conversations:
                debug_content = []
                for part in conv["content"]:
                    if part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        head, _, b64 = url.partition(";base64,")
                        if len(b64) > 20:
                            b64 = b64[:20] + f"...({len(b64)} b64 chars)"
                        debug_content.append({"type": "image_url",
                                              "image_url": {"url": f"{head};base64,{b64}"}})
                    else:
                        text = part.get("text", "")
                        debug_content.append({**part, "text": text[:100]})
                debug_input.append([{"role": "user", "content": debug_content}])
            log.debug(f"VLLM_EMBEDDINGS request to {self.url}: "
                      f"{json.dumps({'model': payload['model'], 'input': debug_input}, default=str)}")

        ret = httpx.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        if ret.status_code != 200:
            raise RuntimeError(
                f"vLLM embedding failed with status {ret.status_code}: {ret.text[:500]}"
            )
        data = ret.json().get("data")
        if not data:
            raise RuntimeError(f"vLLM embedding returned no data: {ret.text[:200]}")
        # vLLM returns one entry per conversation, ordered by index
        return [entry["embedding"] for entry in sorted(data, key=lambda e: e.get("index", 0))]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self._ensure_max_input_tokens()
        results = []
        for batch in self._split_batches([{"text": t} for t in texts]):
            # A single input can still exceed the per-item budget (e.g. a whole
            # document passed to a chunker's clustering step before splitting).
            # The 1:1 vector-per-input contract must hold, so an oversized item
            # cannot be split into multiple returned vectors: instead embed its
            # budget-sized sub-parts and average them into ONE vector. This keeps
            # clustering/retrieval semantics intact while never sending more than
            # the model window accepts.
            if len(batch) == 1:
                item = batch[0]
                text = item.get("text", "")
                if text and self._estimate_text_tokens(text) > self._max_input_tokens:
                    vectors = [
                        v for v in self._embed_sub_parts(text, prefix=self.document_prefix)
                    ]
                    results.append(self._average_vectors(vectors))
                    continue
            results.extend(self._embed(batch, prefix=self.document_prefix))
        return results

    def _embed_sub_parts(self, text: str, prefix: str) -> List[List[float]]:
        """Embed ``text`` split into budget-sized sub-parts, one vector each.

        Words are folded into sub-parts of at most ``max_input_tokens`` (chars
        estimate); each sub-part is embedded (batch-safe) and its vector
        returned in order. Used only to embed a single oversized input.
        """
        part_chars = max(1, self._max_input_tokens * 2)  # chars ≈ 2 * token-budget
        words = text.split()
        parts: List[str] = []
        current: List[str] = []
        current_chars = 0
        for word in words:
            sep = 1 if current else 0
            if current and current_chars + sep + len(word) > part_chars:
                parts.append(" ".join(current))
                current = [word]
                current_chars = len(word)
            else:
                current.append(word)
                current_chars += sep + len(word)
        if current:
            parts.append(" ".join(current))

        vectors: List[List[float]] = []
        for part in parts:
            vectors.extend(self._embed([{"text": part}], prefix=prefix))
        return vectors

    @staticmethod
    def _average_vectors(vectors: List[List[float]]) -> List[float]:
        """Element-wise mean of vectors (one embedding for many sub-parts)."""
        if not vectors:
            return []
        n = len(vectors)
        return [sum(col) / n for col in zip(*vectors)]

    def embed_query(self, text: str) -> List[float]:
        self._ensure_max_input_tokens()
        if self._estimate_text_tokens(text) > self._max_input_tokens:
            # a single over-limit query: embed sub-parts and average
            return self._average_vectors(self._embed_sub_parts(text, prefix=self.query_prefix))
        return self._embed([{"text": text}], prefix=self.query_prefix)[0]

    @staticmethod
    def _is_retryable_failure(exc: Exception) -> bool:
        """True if a per-image embed failure is worth ONE retry.

        Retryable: httpx network/timeout errors (``httpx.HTTPError``) and the
        vLLM 4xx ``RuntimeError`` we raise from ``_embed`` (message contains
        "status 4", the format ``f"vLLM embedding failed with status
        {ret.status_code}: ..."``). Non-retryable errors (ValueError / config
        errors, or a 5xx / no-data RuntimeError) must propagate, not be
        swallowed.
        """
        if isinstance(exc, httpx.HTTPError):
            return True
        return isinstance(exc, RuntimeError) and "status 4" in str(exc)

    def _halved_budget(self) -> tuple[int, int]:
        """Return ``(max_pixels, max_image_tokens)`` with the ceilings halved.

        Used for the single retry after a retryable image-embed failure so
        ``_resize_image_if_needed`` produces a strictly smaller payload -- the
        only lever against a server-side processor rejection (e.g. a Qwen3-VL
        pixel-budget 400). The effective pixel ceiling is halved (None resolves
        to the 1_310_720 default first); the Jina grid-token ceiling is halved
        too so both budget paths shrink.
        """
        eff_pixels = self.max_pixels if self.max_pixels is not None else 1_310_720
        return (max(1, eff_pixels // 2), max(1, self.max_image_tokens // 2))

    def _embed_image_isolated(self, image: str | bytes) -> List[float] | None:
        """Embed a single image in isolation, retrying once with a smaller payload.

        A retryable failure (vLLM 4xx ``RuntimeError`` or ``httpx.HTTPError``)
        triggers exactly ONE retry with a halved pixel/grid ceiling so the
        payload is strictly smaller; if that also fails, returns ``None`` so the
        caller can skip the image and continue. Non-retryable errors (ValueError
        / config errors) propagate.
        """
        try:
            return self._embed([{"image": image}], prefix=self.document_prefix)[0]
        except (httpx.HTTPError, RuntimeError) as exc:
            if not self._is_retryable_failure(exc):
                raise
            log.warning(
                f"VLLM_EMBEDDINGS image embed failed (retrying once with halved "
                f"ceiling): {exc}"
            )
            orig_pixels, orig_tokens = self.max_pixels, self.max_image_tokens
            try:
                self.max_pixels, self.max_image_tokens = self._halved_budget()
                return self._embed([{"image": image}], prefix=self.document_prefix)[0]
            except (httpx.HTTPError, RuntimeError) as exc2:
                if not self._is_retryable_failure(exc2):
                    raise
                log.warning(
                    f"VLLM_EMBEDDINGS image embed failed again after retry; "
                    f"skipping image: {exc2}"
                )
                return None
            finally:
                self.max_pixels, self.max_image_tokens = orig_pixels, orig_tokens

    def embed_image(self, image: str | bytes) -> List[float] | None:
        return self._embed_image_isolated(image)

    def embed_images(self, images: List[str | bytes]) -> List[List[float] | None]:
        # Each image is embedded in isolation (one request per image, matching
        # the previous _split_batches image-one-per-request behavior) so a
        # single failure never aborts the whole batch. The returned list keeps
        # the SAME length and order as the input, with None placeholders for
        # images that fail twice, so MyCAT's zip alignment holds.
        return [self._embed_image_isolated(img) for img in images]
