"""
Centralized configuration for VoiceRAG.

Every tunable value in the project should live here (or be derived from
environment variables here) rather than being hardcoded elsewhere.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (falls back to real environment variables / defaults)
load_dotenv(BASE_DIR / ".env")


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma3:1b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:3b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

OLLAMA_REQUEST_TIMEOUT = _get_int("OLLAMA_REQUEST_TIMEOUT", 60)

# ---------------------------------------------------------------------------
# ChromaDB / Knowledge base (already built - runtime only queries it)
# ---------------------------------------------------------------------------
CHROMA_PATH = str(BASE_DIR / os.getenv("CHROMA_PATH", "knowledge_base/chroma"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "airtel_support")
METADATA_DIR = str(BASE_DIR / "knowledge_base" / "metadata")

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = _get_int("TOP_K", 5)
SIMILARITY_THRESHOLD = _get_float("SIMILARITY_THRESHOLD", 0.70)

# ---------------------------------------------------------------------------
# Chunking (only used by the one-time offline indexing script)
# ---------------------------------------------------------------------------
CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 100)

# ---------------------------------------------------------------------------
# Speech-to-Text
# ---------------------------------------------------------------------------
STT_MODEL = os.getenv("STT_MODEL", "base")
STT_DEVICE = os.getenv("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")

# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
PIPER_VOICE_EN = os.getenv("PIPER_VOICE_EN", "en_US-lessac-medium")
PIPER_VOICE_TE = os.getenv("PIPER_VOICE_TE", "te_IN-medium")
PIPER_MODELS_DIR = str(BASE_DIR / "tts_models")

# ---------------------------------------------------------------------------
# Flask / app
# ---------------------------------------------------------------------------
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = _get_int("FLASK_PORT", 5000)
FLASK_DEBUG = _get_bool("FLASK_DEBUG", False)

# ---------------------------------------------------------------------------
# Audio handling
# ---------------------------------------------------------------------------
PERSIST_RAW_AUDIO = _get_bool("PERSIST_RAW_AUDIO", False)
AUDIO_TMP_DIR = str(BASE_DIR / os.getenv("AUDIO_TMP_DIR", "tmp_audio"))

# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------
MAX_HISTORY_TURNS = _get_int("MAX_HISTORY_TURNS", 8)
CONVERSATION_TTL_SECONDS = _get_int("CONVERSATION_TTL_SECONDS", 60 * 60)

# ---------------------------------------------------------------------------
# Grounding / escalation
# ---------------------------------------------------------------------------
GROUNDING_CONFIDENCE_THRESHOLD = _get_float("GROUNDING_CONFIDENCE_THRESHOLD", 0.60)
INTENT_CONFIDENCE_THRESHOLD = _get_float("INTENT_CONFIDENCE_THRESHOLD", 0.45)
MAX_RETRIEVAL_RETRIES = _get_int("MAX_RETRIEVAL_RETRIES", 1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = str(BASE_DIR / "logs")

# ---------------------------------------------------------------------------
# Supported intents / services (used for structured-output validation)
# ---------------------------------------------------------------------------
SUPPORTED_SERVICES = [
    "prepaid",
    "postpaid",
    "broadband",
    "wifi",
    "recharge",
    "billing",
    "sim",
    "5g",
    "porting",
    "dth",
    "airtel_app",
    "plans",
    "support",
    "complaints",
    "general_faq",
]

DISCLAIMER = (
    "This is an independent prototype using publicly available Airtel "
    "support information. It is not an official Airtel customer-care system."
)

SENSITIVE_TERMS = [
    "otp", "pin", "upi pin", "atm pin", "password",
    "credit card", "debit card", "cvv", "card number",
]
