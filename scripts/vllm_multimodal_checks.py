"""Import-safe, self-contained check for the VLLM multimodal embedder's
payload composition.

Verifies ``CustomVllmMultimodalEmbedder._embed``: every image part in a vLLM
chat-form embedding request is ALWAYS accompanied by a non-whitespace text
part (never an image-only message). vLLM builds an mm-only dummy prompt when a
message is image-only, which some processors (e.g. Qwen3-VL) reject with a 400
"Failed to apply Qwen3VLProcessor"; the client must therefore never emit an
image-only prompt.

Runs standalone with plain ``python3`` (NOT a pytest-collected module). The
ENTIRE body -- all imports (including the module under test, sys, unittest.mock,
and the fake-httpx classes), all fakes, all helpers, and all assert runs -- lives
inside ``if __name__ == "__main__":``. Importing this file therefore executes
ZERO code and emits ZERO stdout, which is the required import-safety contract
for files shipped under a cat plugin folder (the host import_module()s every
.py recursively at activation).
"""

if __name__ == "__main__":
    import os
    import sys
    from unittest import mock

    # Running this file directly puts embedders/ on sys.path, not the repo
    # root, so the top-level `embedders` package is unresolvable. Bootstrap the
    # repo root so `python3 embedders/vllm_multimodal_checks.py` works from the
    # repo root. Inside __main__, so importing the module still runs zero code.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import embedders.custom as embedder_module
    from embedders.custom import CustomVllmMultimodalEmbedder

    class _FakeResponse:
        """Minimal httpx.Response stand-in: 200 with one embedding per item."""

        def __init__(self, n_items):
            self.status_code = 200
            self.text = "{}"
            self._data = [
                {"index": i, "embedding": [0.1, 0.2, 0.3]} for i in range(n_items)
            ]

        def json(self):
            return {"data": self._data}

    def _capture_payload(embedder, items, prefix):
        """Run _embed with httpx.post stubbed to capture the request payload.

        Returns the JSON payload dict that would be sent to /v1/embeddings.
        _to_data_uri is stubbed to a fixed data URI so no network / PIL work
        happens; httpx.post is stubbed to record the payload and return a fake
        200 response.
        """
        captured = {}

        def _fake_post(url, json=None, headers=None, timeout=None):
            captured["payload"] = json
            return _FakeResponse(len((json or {})["input"]))

        with mock.patch.object(embedder_module, "httpx") as fake_httpx:
            fake_httpx.post.side_effect = _fake_post
            with mock.patch.object(
                embedder, "_to_data_uri", lambda image: "data:image/png;base64,AAAA"
            ):
                embedder._embed(items, prefix=prefix)
        return captured["payload"]

    def _text_parts(content):
        return [p for p in content if p["type"] == "text"]

    def _assert_image_item_has_text(payload, label):
        # payload["input"] is [[conv]] per item; each conv is a single message.
        for i, conv in enumerate(payload["input"]):
            content = conv[0]["content"]
            text_parts = _text_parts(content)
            assert text_parts, (
                f"FAIL {label}: image item {i} has NO text part; "
                f"content={content!r}"
            )
            assert any(p["text"].strip() for p in text_parts), (
                f"FAIL {label}: image item {i} has only whitespace text parts; "
                f"content={content!r}"
            )
            # every image part must be paired with a non-whitespace text part
            image_parts = [p for p in content if p["type"] == "image_url"]
            assert image_parts, (
                f"FAIL {label}: item {i} has no image part; content={content!r}"
            )
            for p in text_parts:
                assert p["text"].strip() != "", (
                    f"FAIL {label}: image item {i} carries a whitespace-only "
                    f"text part; content={content!r}"
                )

    # (a) document_prefix="" -> the fallback placeholder text part must be
    #     appended after the image_url so the message is never image-only.
    e_empty = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="m",
        document_prefix="",
    )
    payload = _capture_payload(e_empty, [{"image": b"fake"}], prefix=e_empty.document_prefix)
    _assert_image_item_has_text(payload, "document_prefix=''")
    content = payload["input"][0][0]["content"]
    assert content[-1] == {"type": "text", "text": "Document"}, (
        f"FAIL document_prefix='': expected trailing 'Document' placeholder, "
        f"got content={content!r}"
    )
    print("PASS image item always carries a non-whitespace text part (document_prefix='')")

    # (b) document_prefix="Document: " -> the prefix text part (before the
    #     image) already satisfies the pairing; no placeholder is appended.
    e_prefixed = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="m",
        document_prefix="Document: ",
    )
    payload = _capture_payload(
        e_prefixed, [{"image": b"fake"}], prefix=e_prefixed.document_prefix
    )
    _assert_image_item_has_text(payload, "document_prefix='Document: '")
    content = payload["input"][0][0]["content"]
    assert content[0] == {"type": "text", "text": "Document:"}, (
        f"FAIL document_prefix='Document: ': expected leading prefix text part, "
        f"got content={content!r}"
    )
    assert content[-1]["type"] == "image_url", (
        f"FAIL document_prefix='Document: ': expected image_url to be last, "
        f"got content={content!r}"
    )
    print("PASS image item always carries a non-whitespace text part (document_prefix='Document: ')")

    # (d) Jina-regression: _image_grid_tokens must match the ORIGINAL
    #     Jina-calibrated formula (round(w/32)*2, round(h/32)*2, product) so the
    #     Jina grid-token guard stays byte-identical. 1040x518 -> 2048 (would
    #     resize under max_image_tokens=2048), 2000x2000 -> 15376
    #     (round(62.5)=62 banker's -> 124*124).
    e_jina = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="jinaai/jina-clip-v2",
        max_image_tokens=2048,
    )
    assert e_jina._image_grid_tokens(1040, 518) == 2048, (
        f"FAIL Jina regression: expected 2048 tokens for 1040x518, "
        f"got {e_jina._image_grid_tokens(1040, 518)}"
    )
    assert e_jina._image_grid_tokens(2000, 2000) == 15376, (
        f"FAIL Jina regression: expected 15376 tokens for 2000x2000, "
        f"got {e_jina._image_grid_tokens(2000, 2000)}"
    )
    # a non-qwen model with image_budget=auto must resolve to grid_tokens and
    # therefore use the Jina formula (not the pixel-budget area check)
    assert e_jina._uses_pixel_budget() is False, (
        "FAIL image_budget=auto: non-qwen model must resolve to grid_tokens"
    )
    print("PASS Jina regression: _image_grid_tokens matches original formula (1040x518 -> 2048, 2000x2000 -> 15376)")

    # (e) image_budget=auto discriminator: qwen model names -> pixel budget,
    #     others -> legacy grid-token guard; explicit overrides force either.
    e_qwen = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="Qwen/Qwen3-VL-Embedding-2B",
    )
    assert e_qwen._uses_pixel_budget() is True, (
        "FAIL image_budget=auto: qwen model must resolve to pixel budget"
    )
    e_forced_pixels = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="jinaai/jina-clip-v2",
        image_budget="pixels",
    )
    assert e_forced_pixels._uses_pixel_budget() is True, (
        "FAIL image_budget=pixels must force the pixel budget"
    )
    e_forced_grid = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="Qwen/Qwen3-VL-Embedding-2B",
        image_budget="grid_tokens",
    )
    assert e_forced_grid._uses_pixel_budget() is False, (
        "FAIL image_budget=grid_tokens must force the legacy grid-token guard"
    )
    print("PASS image_budget=auto picks pixels for qwen model names and grid_tokens for others")

    # (f) image resized when over max_pixels: 4000x2000 -> output area <= 1310720.
    from PIL import Image as _PILImage
    import io as _io

    _big = _io.BytesIO()
    _PILImage.new("RGB", (4000, 2000), (255, 255, 255)).save(_big, format="PNG")
    _resized = e_qwen._resize_image_if_needed(_big.getvalue())
    _out = _PILImage.open(_io.BytesIO(_resized))
    _out.load()
    _ow, _oh = _out.size
    assert _ow * _oh <= 1310720, (
        f"FAIL image resized when over max_pixels: output {_ow}x{_oh} "
        f"area {_ow * _oh} > 1310720"
    )
    print("PASS image resized when over max_pixels (4000x2000 -> area <= 1310720)")

    # (g) config defaults: max_pixels None -> effective 1310720, image_budget auto.
    from embedders.configs import VllmMultimodalConfiguration

    _cfg = VllmMultimodalConfiguration(model="Qwen/Qwen3-VL-Embedding-2B")
    assert (_cfg.max_pixels or 1310720) == 1310720, (
        f"FAIL config: expected effective max_pixels 1310720, got {_cfg.max_pixels}"
    )
    assert _cfg.image_budget == "auto", (
        f"FAIL config: expected image_budget 'auto', got {_cfg.image_budget!r}"
    )
    print("PASS config defaults (max_pixels None -> 1310720, image_budget auto)")

    # (h) embed_images returns an ALIGNED list (same length/order as input) with
    #     None placeholders when every image fails twice (retryable failure).
    import httpx as _httpx

    e_iso = CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:1",
        model="Qwen/Qwen3-VL-Embedding-2B",
    )

    def _always_connect_error(url, json=None, headers=None, timeout=None):
        raise _httpx.ConnectError("connection refused")

    with mock.patch.object(embedder_module.httpx, "post", side_effect=_always_connect_error):
        out = e_iso.embed_images([b"img1", b"img2", b"img3"])
    assert len(out) == 3, (
        f"FAIL embed_images aligned length: expected 3, got {len(out)}"
    )
    assert all(v is None for v in out), (
        f"FAIL embed_images double-failure: expected all None, got {out!r}"
    )
    print("PASS embed_images returns aligned list length == input, None on double failure")

    # (i) retry happens exactly ONCE with a strictly smaller payload: first
    #     attempt raises a vLLM 400 RuntimeError, the retry (halved ceiling)
    #     succeeds and returns a vector; the retried image data URI is smaller.
    import io as _io2
    from PIL import Image as _PILImage2

    _big = _io2.BytesIO()
    _PILImage2.new("RGB", (4000, 2000), (255, 255, 255)).save(_big, format="PNG")
    _big_bytes = _big.getvalue()

    _captured = []
    _calls = {"n": 0}

    def _fail_once_then_ok(url, json=None, headers=None, timeout=None):
        _calls["n"] += 1
        _captured.append(json)
        if _calls["n"] == 1:
            raise RuntimeError("vLLM embedding failed with status 400: processor rejected")
        return _FakeResponse(1)

    with mock.patch.object(embedder_module.httpx, "post", side_effect=_fail_once_then_ok):
        out = e_iso.embed_images([_big_bytes])
    assert _calls["n"] == 2, (
        f"FAIL retry count: expected exactly 2 HTTP attempts, got {_calls['n']}"
    )
    assert out[0] == [0.1, 0.2, 0.3], (
        f"FAIL retry success: expected the vector after retry, got {out[0]!r}"
    )
    _uri1 = _captured[0]["input"][0][0]["content"][-1]["image_url"]["url"]
    _uri2 = _captured[1]["input"][0][0]["content"][-1]["image_url"]["url"]
    assert len(_uri2) < len(_uri1), (
        f"FAIL retry smaller payload: attempt2 URI len {len(_uri2)} not < "
        f"attempt1 URI len {len(_uri1)}"
    )
    print("PASS retry happens exactly once with smaller payload")

    # (j) ValueError (config / unreadable image) is NOT swallowed: it propagates.
    with mock.patch.object(
        e_iso, "_to_data_uri", side_effect=ValueError("Unable to read image")
    ):
        try:
            e_iso.embed_images([b"bad"])
            raise AssertionError("FAIL: ValueError was swallowed by embed_images")
        except ValueError:
            pass
    print("PASS ValueError is not swallowed")

    # (c) IMPORT-SAFETY: the module must be import-safe (zero stdout on import).
    #     We cannot re-import ourselves cleanly here, so we assert the file's
    #     top-level structure: the only non-indented executable statement is
    #     the `if __name__ == "__main__":` guard. This pins the import-safety
    #     contract structurally.
    import ast as _ast

    _tree = _ast.parse(open(__file__).read())
    _executable_top = [
        n for n in _tree.body
        if not (isinstance(n, _ast.Expr) and isinstance(n.value, _ast.Constant))
    ]
    assert len(_executable_top) == 1, (
        f"FAIL import-safety: expected exactly one top-level executable "
        f"statement (the __main__ guard), got {len(_executable_top)}"
    )
    assert isinstance(_executable_top[0], _ast.If), (
        "FAIL import-safety: the single top-level executable must be an if"
    )
    print("PASS import-safety")
