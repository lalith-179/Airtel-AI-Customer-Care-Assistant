"""
Speech-to-Text service.

Defines a small interface (`SpeechToTextEngine`) so the concrete engine
(faster-whisper today) can be swapped later without touching the rest of
the app. Only `transcribe(audio_bytes)` is required by callers.
"""
import abc
import io
import logging
import time
from typing import Optional

from config import settings

logger = logging.getLogger("voicerag.stt")


class TranscriptionResult:
    def __init__(self, text: str, language: str, duration_seconds: float, confidence: float = 0.0):
        self.text = text
        self.language = language
        self.duration_seconds = duration_seconds
        self.confidence = confidence

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 3),
            "confidence": round(self.confidence, 3),
        }


class SpeechToTextEngine(abc.ABC):
    """Interface every STT backend must implement."""

    @abc.abstractmethod
    def transcribe(self, audio_bytes: bytes, filename_hint: str = "audio.webm") -> TranscriptionResult:
        raise NotImplementedError

    @abc.abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError


class WhisperSTTEngine(SpeechToTextEngine):
    """faster-whisper backed implementation.

    faster-whisper's Whisper models are multilingual out of the box and
    handle Telugu + English (including code-mixed audio reasonably well for
    a small/base model), matching the project's language requirement without
    any custom fine-tuning.
    """

    def __init__(self, model_size: str = None, device: str = None, compute_type: str = None):
        self.model_size = model_size or settings.STT_MODEL
        self.device = device or settings.STT_DEVICE
        self.compute_type = compute_type or settings.STT_COMPUTE_TYPE
        self._model = None
        self._load_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel  # imported lazily - heavy dependency

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            logger.info(
                "Loaded faster-whisper model=%s device=%s compute_type=%s",
                self.model_size, self.device, self.compute_type,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load faster-whisper model '{self.model_size}': {exc}"
            logger.error(self._load_error)

    def is_ready(self) -> bool:
        return self._model is not None

    def transcribe(self, audio_bytes: bytes, filename_hint: str = "audio.webm") -> TranscriptionResult:
        if not self.is_ready():
            raise RuntimeError(self._load_error or "STT engine is not loaded.")

        start = time.time()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.name = filename_hint  # faster-whisper/ffmpeg uses this for format sniffing

        segments, info = self._model.transcribe(
            audio_buffer,
            beam_size=5,
            vad_filter=True,
            language=None,  # auto-detect between Telugu / English / mixed
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed = time.time() - start

        logger.info(
            "stt.transcribe language=%s prob=%.2f took=%.2fs chars=%d",
            info.language, info.language_probability, elapsed, len(text),
        )
        return TranscriptionResult(
            text=text,
            language=info.language,
            duration_seconds=elapsed,
            confidence=info.language_probability,
        )


def get_stt_engine() -> SpeechToTextEngine:
    """Factory so app.py doesn't need to know the concrete engine class."""
    return WhisperSTTEngine()


# Module-level singleton - the whisper model is loaded once at startup.
stt_engine = get_stt_engine()
