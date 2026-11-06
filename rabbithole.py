import importlib.util
import shutil
from typing import Dict, Iterable, Set
import nltk
from cat import hook, BillTheLizard, EmbedderSettings
from cat.services.service_factory import ServiceFactory

from .parsers import FasterWhisperParser, ExcelParser, OdpParser, OdsParser, UnstructuredParser, YoutubeParser
from .parsers.html_preserving_parser import RawHTMLParser
from .chunkers.custom import HTMLSemanticChunker

import ctranslate2

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

_AUDIO_MIME_TYPES: Set[str] = {
    "audio/flac", "audio/x-flac",
    "audio/mp3", "audio/mpeg", "audio/x-mp3", "audio/x-mpeg",
    "audio/mp4", "audio/x-m4a",
    "audio/ogg", "audio/x-ogg",
    "audio/opus",
    "audio/wav", "audio/vnd.wav", "audio/vnd.wave", "audio/wave",
    "audio/x-pn-wav", "audio/x-wav",
    "audio/webm",
}
_VIDEO_MIME_TYPES: Set[str] = {"video/mp4"}


def _filetype_is_image(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "image"


def _filetype_is_audio(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "audio"


def _find_spec_or_binary(name: str) -> bool:
    return importlib.util.find_spec(name) is not None or shutil.which(name) is not None


def _get_unstructured_supported_mimetypes() -> Dict[str, object]:
    try:
        from unstructured.file_utils.model import FileType
    except ImportError:
        return {}

    supported: Dict[str, object] = {}
    for ft in FileType:
        if not ft.is_partitionable:
            continue
        deps: Iterable[str] = ft.importable_package_dependencies
        if not all(_find_spec_or_binary(pkg) for pkg in deps):
            continue
        supported[ft.mime_type] = ft
        for alias in getattr(ft, "_alias_mime_types", ()):
            supported[alias] = ft
    return supported


@hook(priority=1)
async def rabbithole_instantiates_parsers(file_handlers: Dict, cat) -> Dict:
    lizard = BillTheLizard()
    sp = ServiceFactory(
        agent_key=lizard.agent_key,
        hook_manager=lizard.plugin_manager,
        factory_allowed_handler_name="factory_allowed_embedders",
        setting_category="embedder",
        schema_name="languageEmbedderName",
    )

    embedder_config: EmbedderSettings | None = await sp.get_config_class_from_adapter(await lizard.embedder())
    if not embedder_config:
        return file_handlers

    is_multimodal = embedder_config.is_multimodal()
    up_options = {
        "strategy": "hi_res",
        "extract_image_block_to_payload": True,
        "extract_image_block_types": ["Image", "Table"],
    } if is_multimodal else {
        "strategy": "fast",
        "extract_images_in_pdf": False,
        "infer_table_structure": False,
        "extract_image_block_types": [],
    }
    word_parser = UnstructuredParser(**up_options)
    powerpoint_parser = UnstructuredParser(**up_options)

    supported = _get_unstructured_supported_mimetypes()
    for mime_type, ft in supported.items():
        if mime_type in _AUDIO_MIME_TYPES or _filetype_is_audio(ft):
            continue
        if mime_type in _VIDEO_MIME_TYPES:
            continue
        if _filetype_is_image(ft) and not is_multimodal:
            continue
        if mime_type in file_handlers:
            continue
        file_handlers[mime_type] = UnstructuredParser(**up_options)

    file_handlers.update({
        "application/msword": word_parser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": word_parser,
        "application/vnd.ms-powerpoint": powerpoint_parser,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": powerpoint_parser,
        "application/vnd.ms-excel": ExcelParser(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ExcelParser(),
        "application/vnd.oasis.opendocument.text": word_parser,
        # unstructured has no ODP partitioner: convert to PPTX, then parse like PowerPoint.
        "application/vnd.oasis.opendocument.presentation": OdpParser(powerpoint_parser),
        "application/vnd.oasis.opendocument.spreadsheet": OdsParser(),
    })

    if is_multimodal:
        file_handlers["application/pdf"] = UnstructuredParser(**up_options)
        for mime_type, ft in supported.items():
            if _filetype_is_image(ft):
                file_handlers[mime_type] = UnstructuredParser(**up_options)

    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    file_handlers.update({
        "video/mp4": YoutubeParser(),
        "audio/mpeg": FasterWhisperParser(device=device),
        "audio/mp3": FasterWhisperParser(device=device),
        "audio/ogg": FasterWhisperParser(device=device),
        "audio/wav": FasterWhisperParser(device=device),
        "audio/webm": FasterWhisperParser(device=device),
        "video/webm": FasterWhisperParser(device=device),
    })

    # Preserve raw HTML when HTMLSemanticChunker is active,
    # so the chunker can split semantically on heading structure.
    if hasattr(cat, "chunker") and isinstance(cat.chunker, HTMLSemanticChunker):
        file_handlers["text/html"] = RawHTMLParser()

    return file_handlers
