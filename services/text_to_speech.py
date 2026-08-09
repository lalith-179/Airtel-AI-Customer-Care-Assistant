"""
Text-to-Speech service.

Defines a small interface (`TextToSpeechEngine`) so the concrete engine
(Piper today) can be swapped later without touching the rest of the app.
Only `synthesize(text, language)` is required by callers, and it always
returns raw WAV bytes.
"""
import abc
import io
import logging
import time
import wave
from typing import Optional

from config import settings

logger = logging.getLogger("voicerag.tts")


class SynthesisResult:
    def __init__(self, audio_bytes: bytes, sample_rate: int, duration_seconds: float):
        self.audio_bytes = audio_bytes
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds


class TextToSpeechEngine(abc.ABC):
    """Interface every TTS backend must implement."""

    @abc.abstractmethod
    def synthesize(self, text: str, language: str = "en") -> SynthesisResult:
        raise NotImplementedError

    @abc.abstractmethod
    def is_ready(self) -> bool:
        raise NotImplementedError


class PiperTTSEngine(TextToSpeechEngine):
    """Piper-backed implementation.

    Piper voices are per-language ONNX models. We keep an English voice and
    a Telugu voice loaded and pick one based on the detected language from
    the STT/intent stages, falling back to English for code-mixed text
    (Piper does not support mixed-script synthesis natively).
    """

    def __init__(self, voice_en: str = None, voice_te: str = None):
        self.voice_en_name = voice_en or settings.PIPER_VOICE_EN
        self.voice_te_name = voice_te or settings.PIPER_VOICE_TE
        self._voices: dict = {}
        self._load_error: Optional[str] = None
        self._load_voices()

    def _load_voices(self) -> None:
        try:
            from piper import PiperVoice  # imported lazily - heavy dependency
            from pathlib import Path

            models_dir = Path(settings.PIPER_MODELS_DIR)
            for lang, voice_name in (("en", self.voice_en_name), ("te", self.voice_te_name)):
                onnx_path = models_dir / f"{voice_name}.onnx"
                if onnx_path.exists():
                    self._voices[lang] = PiperVoice.load(str(onnx_path))
                    logger.info("Loaded Piper voice '%s' for language=%s", voice_name, lang)
                else:
                    logger.warning(
                        "Piper voice file not found for language=%s at %s "
                        "(download it into %s to enable TTS for this language)",
                        lang, onnx_path, models_dir,
                    )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"Failed to load Piper voices: {exc}"
            logger.error(self._load_error)

    def is_ready(self) -> bool:
        return len(self._voices) > 0

    def synthesize(self, text: str, language: str = "en") -> SynthesisResult:
        if not self.is_ready():
            raise RuntimeError(self._load_error or "TTS engine is not loaded.")

        voice_key = language if language in self._voices else "en"
        voice = self._voices.get(voice_key)
        if voice is None:
            raise RuntimeError(f"No Piper voice available for language='{language}'")

        start = time.time()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            voice.synthesize(text, wav_file)
        elapsed = time.time() - start

        audio_bytes = buffer.getvalue()
        sample_rate = getattr(voice.config, "sample_rate", 22050)
        # rough duration estimate from PCM16 mono byte length
        duration = (len(audio_bytes) - 44) / (sample_rate * 2) if len(audio_bytes) > 44 else 0.0

        logger.info(
            "tts.synthesize language=%s chars=%d took=%.2fs duration=%.2fs",
            voice_key, len(text), elapsed, duration,
        )
        return SynthesisResult(audio_bytes=audio_bytes, sample_rate=sample_rate, duration_seconds=duration)


def get_tts_engine() -> TextToSpeechEngine:
    """Factory so app.py doesn't need to know the concrete engine class."""
    return PiperTTSEngine()


# Module-level singleton - Piper voices are loaded once at startup.
tts_engine = get_tts_engine()
