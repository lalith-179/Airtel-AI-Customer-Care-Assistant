"""
Intent Agent.

Classifies the user's message into (intent, service, language, confidence)
using structured JSON output from Gemma3:1B via Ollama. Never answers the
question itself - that is the Response Agent's job.
"""
import logging

from config import settings
from prompts.intent_prompt import INTENT_SYSTEM_PROMPT, build_intent_prompt
from services.ollama_service import OllamaError, ollama_service

logger = logging.getLogger("voicerag.agents.intent")

_VALID_SERVICES = set(settings.SUPPORTED_SERVICES)


def classify_intent(user_input: str, conversation_history: list, current_service: str | None) -> dict:
    """Returns {"intent", "service", "language", "confidence"}.

    Falls back to a safe, low-confidence generic classification if the
    model call fails or returns malformed data - the graph still moves
    forward, it just treats the query as low-confidence general FAQ.
    """
    prompt = build_intent_prompt(user_input, conversation_history, current_service)

    try:
        result = ollama_service.generate_json(
            prompt=prompt,
            system=INTENT_SYSTEM_PROMPT,
            model=settings.LLM_MODEL,
            temperature=0.1,
            retries=1,
        )
    except OllamaError as exc:
        logger.warning("Intent classification failed, using fallback: %s", exc)
        return _fallback(current_service)

    intent = str(result.get("intent") or "general_faq").strip().lower().replace(" ", "_")
    service = str(result.get("service") or (current_service or "general_faq")).strip().lower()
    language = str(result.get("language") or "en").strip().lower()
    try:
        confidence = float(result.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    if service not in _VALID_SERVICES:
        service = current_service if current_service in _VALID_SERVICES else "general_faq"

    if language not in {"te", "en", "mixed"}:
        language = "en"

    return {
        "intent": intent,
        "service": service,
        "language": language,
        "confidence": confidence,
    }


def _fallback(current_service: str | None) -> dict:
    return {
        "intent": "general_faq",
        "service": current_service or "general_faq",
        "language": "en",
        "confidence": 0.3,
    }
