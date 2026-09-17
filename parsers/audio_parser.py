from typing import Iterator
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents import Document
from faster_whisper import WhisperModel

class FasterWhisperParser(BaseBlobParser):
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Initialize model here or lazily?
        # langchain_community initializes inside lazy_parse or __init__ depending on implementation
        # For safety, let's initialize lazily to save memory if not used
        self.model = None

    def lazy_parse(self, blob) -> Iterator[Document]:
        if self.model is None:
            self.model = WhisperModel("base", device=self.device, compute_type="int8")
        
        # faster_whisper transcribe can take a file-like object or a path
        # blob.as_bytes_io() returns a BytesIO object which faster-whisper accepts
        with blob.as_bytes_io() as f:
            segments, _ = self.model.transcribe(f, beam_size=5)
            text = " ".join(segment.text for segment in segments)
            
        yield Document(page_content=text, metadata={"source": blob.source})
