"""Prompt template for the Intent Agent."""
from config.settings import SUPPORTED_SERVICES

SERVICE_LIST = ", ".join(SUPPORTED_SERVICES)

INTENT_SYSTEM_PROMPT = f"""You are an intent-classification engine for an Airtel
telecom customer-care assistant. You NEVER answer the user's question.
You ONLY classify it.

Allowed "service" values: {SERVICE_LIST}

Respond with STRICT JSON only, no markdown, no commentary, no code fences.
Schema:
{{
  "intent": "<short_snake_case_intent_label>",
  "service": "<one value from the allowed list>",
  "language": "<'te' for Telugu, 'en' for English, 'mixed' for Telugu-English code-mixed>",
  "confidence": <float between 0 and 1>
}}

Rules:
- If the user message continues a previous topic (e.g. answering a clarifying
  question), infer the service from the conversation context provided.
- If you are unsure, still return your best guess with a lower confidence score.
- Never include any text outside the JSON object.
"""


def build_intent_prompt(user_input: str, conversation_history: list, current_service: str | None) -> str:
    history_lines = []
    for turn in conversation_history[-6:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        history_lines.append(f"{role}: {turn.get('text', '')}")
    history_text = "\n".join(history_lines) if history_lines else "(no prior turns)"

    return f"""Conversation so far:
{history_text}

Known current service (may be None if not yet established): {current_service}

Latest user message:
\"\"\"{user_input}\"\"\"

Classify this message according to the schema."""
