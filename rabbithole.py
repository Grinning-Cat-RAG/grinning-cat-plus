import importlib.util
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Set

import nltk
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain_community.document_loaders.parsers.msword import MsWordParser
from cat import hook, BillTheLizard, EmbedderSettings
from cat.services.service_factory import ServiceFactory

from .parsers import ExcelParser, OdsParser, PowerPointParser, UnstructuredParser, YoutubeParser
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
_ZIP_BASED_MIME_TYPES: Set[str] = {
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.graphics",
}
_ODP_MIME = "application/vnd.oasis.opendocument.presentation"
_ODS_MIME = "application/vnd.oasis.opendocument.spreadsheet"
_ODG_MIME = "application/vnd.oasis.opendocument.graphics"


def _filetype_is_image(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "image"


def _filetype_is_audio(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "audio"


def _find_spec_or_binary(name: str) -> bool:
    return importlib.util.find_spec(name) is not None or shutil.which(name) is not None


def _libreoffice_bin() -> str | None:
    return shutil.which("libreoffice") or shutil.which("soffice")


def _libreoffice_convert(source_path: str, target_ext: str) -> str:
    binary = _libreoffice_bin()
    if not binary:
        raise RuntimeError(
            "LibreOffice/soffice non trovato nel PATH: necessario per convertire file ODF"
        )

    src = Path(source_path)
    outdir = Path(tempfile.mkdtemp(prefix="odf-convert-"))
    try:
        cmd = [
            binary,
            "--headless",
            "--convert-to",
            target_ext,
            "--outdir",
            str(outdir),
            str(src),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Conversione LibreOffice fallita ({proc.returncode}): "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
            )

        converted = outdir / f"{src.stem}.{target_ext}"
        if not converted.exists():
            files = ", ".join(p.name for p in outdir.iterdir()) if outdir.exists() else "<none>"
            raise RuntimeError(
                f"File convertito non trovato: atteso {converted.name}, presenti: {files}"
            )
        return str(converted)
    except Exception:
        shutil.rmtree(outdir, ignore_errors=True)
        raise





def _register_odf_filetypes() -> None:
    try:
        from unstructured.file_utils.filetype import add_file_type
        from unstructured.file_utils.model import FileType
    except ImportError:
        return

    existing_mimes = {getattr(ft, "mime_type", None) for ft in FileType}
    module_name = __name__

    if _ODP_MIME not in existing_mimes:
        add_file_type(
            file_type_name="ODP",
            canonical_mime_type=_ODP_MIME,
            importable_package_dependencies=[],
            extra_name="odf",
            extensions=[".odp"],
            partitioner_shortname=module_name,
        )
    if _ODS_MIME not in existing_mimes:
        add_file_type(
            file_type_name="ODS",
            canonical_mime_type=_ODS_MIME,
            importable_package_dependencies=[],
            extra_name="odf",
            extensions=[".ods"],
            partitioner_shortname=module_name,
        )
    if _ODG_MIME not in existing_mimes:
        add_file_type(
            file_type_name="ODG",
            canonical_mime_type=_ODG_MIME,
            importable_package_dependencies=[],
            extra_name="odf",
            extensions=[".odg"],
            partitioner_shortname=module_name,
        )


def _get_unstructured_supported_mimetypes() -> Dict[str, object]:
    _register_odf_filetypes()
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
    up_options = {  'strategy':"hi_res", 
                    'extract_image_block_to_payload':True, 
                    'extract_image_block_types':["Image", "Table"], 
                  } if is_multimodal else {
                    'strategy':"fast",
                    'extract_images_in_pdf':False,
                    'infer_table_structure':False,
                    'extract_image_block_types':[],
                  }
    word_parser = MsWordParser() if not is_multimodal else UnstructuredParser(**up_options)
    powerpoint_parser = PowerPointParser() if not is_multimodal else UnstructuredParser(**up_options)


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
        "application/vnd.oasis.opendocument.presentation": powerpoint_parser,
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
