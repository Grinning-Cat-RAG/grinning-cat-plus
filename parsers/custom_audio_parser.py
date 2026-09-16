from typing import Iterator
from langchain_core.documents import Document
from faster_whisper import WhisperModel


class FasterWhisperParser:
    def __init__(self, device: str = "cpu", model_size: str = "base"):
        self.device = device
        self.model_size = model_size
        self.model = WhisperModel(model_size, device=device, compute_type="int8")

    def lazy_parse(self, blob) -> Iterator[Document]:
        source = getattr(blob, "source", "") or ""
        with blob.as_bytes_io() as audio_file:
            segments, info = self.model.transcribe(audio_file, beam_size=5)
            full_text = " ".join(segment.text for segment in segments)
            yield Document(page_content=full_text, metadata={"source": source})
