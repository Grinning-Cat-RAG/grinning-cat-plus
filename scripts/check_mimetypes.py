"""
Diagnostic script: lists all FileType entries from unstructured (including
ODP/ODS/ODG registered by this plugin) and compares them with what the plugin
would actually register in rabbithole_instantiates_parsers.

Run inside the container:
    python check_mimetypes.py

Expected output sections:
  1. Full FileType table (all partitionable types, dep check columns)
  2. Plugin view: what mime types the plugin would register, with which parser
  3. Diff: types present in unstructured but NOT mapped by the plugin
  4. Diff: types mapped by the plugin but NOT coming from unstructured scan
"""
import importlib.util
import shutil
from typing import Dict, Set


# ── helpers (mirrored from rabbithole.py) ─────────────────────────────────────

def _find_spec_safe(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _shutil_which_safe(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _dep_available(pkg: str) -> bool:
    return _find_spec_safe(pkg) or _shutil_which_safe(pkg)


def _filetype_is_image(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "image"


def _filetype_is_audio(ft) -> bool:
    return getattr(ft, "_partitioner_shortname", None) == "audio"


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

# Parser labels used in the "plugin view" table
_WORD_PARSERS       = {"MsWordParser", "UnstructuredParser(word)"}
_PPT_PARSERS        = {"PowerPointParser", "UnstructuredParser(ppt)"}
_EXCEL_PARSERS      = {"TableParser", "UnstructuredParser(excel)"}


# ── unstructured scan ─────────────────────────────────────────────────────────

def _get_all_filetypes():
    """Return dict mime→FileType for every partitionable FileType (env deps satisfied)."""
    try:
        # Try to call plugin's registration first
        from cat.plugins.grinning_cat_plus.rabbithole import (
            _register_odf_filetypes,
            _get_unstructured_supported_mimetypes,
        )
        _register_odf_filetypes()
        return _get_unstructured_supported_mimetypes()
    except ImportError:
        pass

    # Fallback: inline implementation
    try:
        from unstructured.file_utils.model import FileType
    except ImportError:
        return {}

    supported: Dict[str, object] = {}
    for ft in FileType:
        if not ft.is_partitionable:
            continue
        deps = list(ft.importable_package_dependencies)
        if not all(_dep_available(pkg) for pkg in deps):
            continue
        supported[ft.mime_type] = ft
        for alias in getattr(ft, "_alias_mime_types", ()):
            supported[alias] = ft
    return supported


# ── plugin view simulation ────────────────────────────────────────────────────

def _simulate_plugin(supported: Dict[str, object], is_multimodal: bool = False) -> Dict[str, str]:
    """
    Simulate what rabbithole_instantiates_parsers would put in file_handlers,
    using string labels instead of real parser instances.
    Returns dict mime→parser_label.
    """
    word_label  = "UnstructuredParser" if is_multimodal else "MsWordParser"
    ppt_label   = "UnstructuredParser" if is_multimodal else "PowerPointParser"
    excel_label = "UnstructuredParser" if is_multimodal else "TableParser"
    fast_label  = "UnstructuredParser" if is_multimodal else "UnstructuredParser(fast)"

    handlers: Dict[str, str] = {}

    # Auto-discovery pass
    for mime_type, ft in supported.items():
        if mime_type in _AUDIO_MIME_TYPES or _filetype_is_audio(ft):
            continue
        if mime_type in _VIDEO_MIME_TYPES:
            continue
        if _filetype_is_image(ft) and not is_multimodal:
            continue
        if mime_type in handlers:
            continue
        handlers[mime_type] = "UnstructuredParser" if is_multimodal else fast_label

    # Explicit overrides
    handlers.update({
        "application/msword": word_label,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": word_label,
        "application/vnd.ms-powerpoint": ppt_label,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ppt_label,
        "application/vnd.ms-excel": excel_label,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": excel_label,
        "application/vnd.oasis.opendocument.text": word_label,
        "application/vnd.oasis.opendocument.spreadsheet": excel_label,
        "application/vnd.oasis.opendocument.presentation": ppt_label,
        "application/vnd.oasis.opendocument.graphics": "UnstructuredParser" if is_multimodal else fast_label,
    })

    if is_multimodal:
        handlers["application/pdf"] = "UnstructuredParser"
        for mime_type, ft in supported.items():
            if _filetype_is_image(ft):
                handlers[mime_type] = "UnstructuredParser"

    # Audio / video
    handlers.update({
        "video/mp4":   "YoutubeParser",
        "audio/mpeg":  "FasterWhisperParser",
        "audio/mp3":   "FasterWhisperParser",
        "audio/ogg":   "FasterWhisperParser",
        "audio/wav":   "FasterWhisperParser",
        "audio/webm":  "FasterWhisperParser",
        "video/webm":  "FasterWhisperParser",
    })

    return handlers


# ── output helpers ────────────────────────────────────────────────────────────

SEP = "-" * 140


def _print_section(title: str):
    print()
    print("=" * 140)
    print(f"  {title}")
    print("=" * 140)


def main():
    try:
        from unstructured.file_utils.model import FileType
    except ImportError:
        print("ERROR: unstructured not installed")
        return

    # ── Section 1: full FileType table ────────────────────────────────────────
    _print_section("1. All partitionable FileTypes (env check)")
    print(f"{'FileType':<12} {'MIME type':<65} {'Deps':<38} {'PyOK':<7} {'BinOK':<7} IN")
    print(SEP)
    for ft in FileType:
        if not ft.is_partitionable:
            continue
        deps = list(ft.importable_package_dependencies)
        py_ok  = all(_find_spec_safe(d) for d in deps) if deps else True
        bin_ok = all(_dep_available(d) for d in deps) if deps else True
        print(
            f"{ft.name:<12} {ft.mime_type:<65} {str(deps):<38} "
            f"{str(py_ok):<7} {str(bin_ok):<7} {'✓' if bin_ok else '✗'}"
        )

    # ── ODF summary ───────────────────────────────────────────────────────────
    supported = _get_all_filetypes()
    print()
    print(f"ODP presente : {'application/vnd.oasis.opendocument.presentation' in supported}")
    print(f"ODT presente : {'application/vnd.oasis.opendocument.text' in supported}")
    print(f"ODS presente : {'application/vnd.oasis.opendocument.spreadsheet' in supported}")
    odf_keys = sorted(k for k in supported if "opendocument" in k)
    print(f"Chiavi ODF trovate: {odf_keys}")

    # ── Section 2: plugin view (text-only embedder) ───────────────────────────
    for is_multimodal, label in [(False, "text-only"), (True, "multimodal")]:
        _print_section(f"2. Plugin view — embedder={label}")
        handlers = _simulate_plugin(supported, is_multimodal=is_multimodal)
        print(f"{'MIME type':<65} {'Parser'}")
        print(SEP)
        for mime in sorted(handlers):
            print(f"{mime:<65} {handlers[mime]}")

        # ── Section 3: unstructured → not in plugin ───────────────────────────
        _print_section(f"3. In unstructured scan but NOT in plugin handlers ({label})")
        missing = sorted(
            m for m in supported
            if m not in handlers
            and m not in _AUDIO_MIME_TYPES
            and m not in _VIDEO_MIME_TYPES
            and not (not is_multimodal and _filetype_is_image(supported[m]))
        )
        if missing:
            for m in missing:
                ft = supported[m]
                print(f"  {m:<65} FileType={getattr(ft, 'name', '?')}")
        else:
            print("  (nessuno — copertura completa ✓)")

        # ── Section 4: plugin → not in unstructured scan ──────────────────────
        _print_section(f"4. In plugin handlers but NOT in unstructured scan ({label})")
        extra = sorted(
            m for m in handlers
            if m not in supported
            and m not in _AUDIO_MIME_TYPES
            and m not in _VIDEO_MIME_TYPES
            and handlers[m] not in {"YoutubeParser", "FasterWhisperParser"}
        )
        if extra:
            for m in extra:
                print(f"  {m:<65} parser={handlers[m]}  ← registrato manualmente nel plugin")
        else:
            print("  (nessuno)")


if __name__ == "__main__":
    main()
