import threading
from typing import Dict, Iterator, Tuple

from langchain_core.documents import Document
from faster_whisper import WhisperModel

from cat import log


class FasterWhisperParser:
    """Audio transcription with faster-whisper.

    The Whisper model is NOT loaded at construction: ``rabbithole_instantiates_parsers``
    builds one parser per audio MIME type on every call, so loading there meant several
    model loads (and several Hugging Face Hub requests) per hook call, even when no audio
    was ever ingested. The model is loaded on the first transcription instead, once per
    process per ``(model_size, device, compute_type)``, and shared by every parser.

    Loading is offline-first: the local Hugging Face cache is tried with
    ``local_files_only=True``, which makes no network request. Only when the model is not
    cached is it downloaded, once. A failed download is not memoized, so the next audio
    file retries it.
    """

    _models: Dict[Tuple[str, str, str], WhisperModel] = {}
    _models_lock = threading.Lock()

    def __init__(self, device: str = "cpu", model_size: str = "base", compute_type: str = "int8"):
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type

    @property
    def model(self) -> WhisperModel:
        key = (self.model_size, self.device, self.compute_type)
        model = self._models.get(key)
        if model is not None:
            return model

        with self._models_lock:
            model = self._models.get(key)
            if model is not None:
                return model
            kwargs = {"device": self.device, "compute_type": self.compute_type}
            try:
                model = WhisperModel(self.model_size, local_files_only=True, **kwargs)
            except Exception as exc:  # noqa: BLE001 - not in the local cache: download once
                log.info(
                    f"Whisper model {self.model_size!r} not in the local Hugging Face cache "
                    f"({type(exc).__name__}: {exc}), downloading it"
                )
                model = WhisperModel(self.model_size, **kwargs)
            self._models[key] = model
            return model

    def lazy_parse(self, blob) -> Iterator[Document]:
        source = getattr(blob, "source", "") or ""
        with blob.as_bytes_io() as audio_file:
            segments, info = self.model.transcribe(audio_file, beam_size=5)
            full_text = " ".join(segment.text for segment in segments)
            yield Document(page_content=full_text, metadata={"source": source})
