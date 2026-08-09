"""
Grounding / Safety Validator Agent.

Reviews a generated answer against the retrieved context using Gemma3:1B
and additional deterministic safety checks (sensitive-term scanning), and
decides whether the answer is safe to speak to the user and/or whether the
conversation should be escalated.
"""
import logging

from config import settings
from prompts.grounding_prompt import GROUNDING_SYSTEM_PROMPT, build_grounding_prompt
from services.ollama_service import OllamaError, ollama_service

logger = logging.getLogger("voicerag.agents.grounding")


def validate_grounding(generated_answer: str, context: str, user_input: str) -> dict:
    """Returns {"grounded", "confidence", "reason", "should_escalate"}."""

    # Deterministic safety net: never let a request for credentials through,
    # regardless of what the LLM validator decides.
    lowered_answer = generated_answer.lower()
    requests_sensitive_info = any(term in lowered_answer for term in settings.SENSITIVE_TERMS)

    if not context.strip():
        return {
            "grounded": False,
            "confidence": 0.0,
            "reason": "No context was retrieved from the knowledge base.",
            "should_escalate": True,
        }

    prompt = build_grounding_prompt(generated_answer, context, user_input)

    try:
        result = ollama_service.generate_json(
            prompt=prompt,
            system=GROUNDING_SYSTEM_PROMPT,
            model=settings.LLM_MODEL,
            temperature=0.0,
            retries=1,
        )
    except OllamaError as exc:
        logger.warning("Grounding validation call failed, defaulting to ungrounded: %s", exc)
        return {
            "grounded": False,
            "confidence": 0.0,
            "reason": f"Validator unavailable: {exc}",
            "should_escalate": True,
        }

    grounded = bool(result.get("grounded", False))
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(result.get("reason", "")).strip() or "No reason provided."
    should_escalate = bool(result.get("should_escalate", False))

    if requests_sensitive_info:
        grounded = False
        should_escalate = True
        reason = "Response referenced sensitive credential terms and was blocked."

    if confidence < settings.GROUNDING_CONFIDENCE_THRESHOLD:
        grounded = False

    return {
        "grounded": grounded,
        "confidence": confidence,
        "reason": reason,
        "should_escalate": should_escalate,
    }
