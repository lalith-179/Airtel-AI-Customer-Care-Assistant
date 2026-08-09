"""
Response Generation Agent.

Generates a grounded answer using Gemma3:1B, the retrieved Airtel
documents, source information, and conversation context. Enforces the
"never claim account access / never ask for credentials" rules at the
prompt level (grounding_agent.py double-checks this after generation).
"""
import logging

from config import settings
from prompts.response_prompt import RESPONSE_SYSTEM_PROMPT, build_response_prompt
from services.ollama_service import OllamaError, ollama_service

logger = logging.getLogger("voicerag.agents.response")

NO_CONTEXT_FALLBACK = (
    "I couldn't find enough information in the available Airtel support "
    "knowledge to answer that reliably. Please try the official Airtel "
    "app or airtel.in support pages, or rephrase your question."
)


def generate_response(
    user_input: str,
    conversation_history: list,
    context: str,
    sources: list,
    language: str,
) -> str:
    if not context.strip():
        logger.info("No retrieved context available; returning safe fallback answer.")
        return NO_CONTEXT_FALLBACK

    prompt = build_response_prompt(
        user_input=user_input,
        conversation_history=conversation_history,
        context=context,
        sources=sources,
        language=language,
    )

    try:
        answer = ollama_service.generate(
            prompt=prompt,
            system=RESPONSE_SYSTEM_PROMPT,
            model=settings.LLM_MODEL,
            temperature=0.4,
        )
    except OllamaError as exc:
        logger.error("Response generation failed: %s", exc)
        return (
            "I'm having trouble reaching the assistant model right now. "
            "Please try again in a moment."
        )

    answer = answer.strip()
    return answer if answer else NO_CONTEXT_FALLBACK
