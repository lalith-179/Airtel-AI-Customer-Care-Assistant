"""
VoiceRAG LangGraph Workflow.

    START
      |
      v
  Input Understanding
      |
      v
  Intent Detection
      |
      v
  RAG Retrieval  <---------------------+
      |                                |
      v                                |
  Response Generation                  |
      |                                |
      v                                |
  Grounding Validation                 |
      |          |                     |
     PASS       FAIL --(retry left)----+
      |          |
      |    (no retries left)
      |          |
      v          v
  Escalation Decision <-- (fallback response used)
      |
      v
     TTS (only if synthesize_speech=True)
      |
      v
     END

This module builds ONE reusable compiled graph at import time. Both the
text and voice entry points in app.py invoke the exact same graph - only
STT (before) and TTS (inside, gated by a flag) differ per mode.
"""
import logging
import time

from config import settings
from agents.intent_agent import classify_intent
from agents.response_agent import NO_CONTEXT_FALLBACK, generate_response
from agents.grounding_agent import validate_grounding
from models.state import GraphState
from services.rag_service import get_context_and_sources
from services.retrieval_service import KnowledgeBaseUnavailableError
from services.text_to_speech import tts_engine

from langgraph.graph import END, StateGraph

logger = logging.getLogger("voicerag.workflow")

ACCOUNT_SPECIFIC_HINTS = (
    "my account", "my balance", "my bill", "check my", "raise a complaint",
    "process my", "cancel my", "activate my", "my recharge status",
)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def input_understanding_node(state: GraphState) -> GraphState:
    text = (state.get("user_input") or "").strip()
    timings = state.get("timings", {})
    state["user_input"] = text
    state["stage"] = "input_understanding"
    state["retrieval_attempts"] = 0
    state["error"] = None
    state["timings"] = timings
    return state


def intent_detection_node(state: GraphState) -> GraphState:
    start = time.time()
    result = classify_intent(
        user_input=state["user_input"],
        conversation_history=state.get("conversation_history", []),
        current_service=state.get("current_service"),
    )
    state["intent"] = result["intent"]
    state["service"] = result["service"]
    state["language"] = result["language"]
    state["intent_confidence"] = result["confidence"]
    state["stage"] = "intent_detection"
    state.setdefault("timings", {})["intent_detection"] = round(time.time() - start, 3)
    return state


def retrieval_node(state: GraphState) -> GraphState:
    start = time.time()
    query = state["user_input"]
    try:
        context, sources, matches = get_context_and_sources(query)
        state["context"] = context
        state["sources"] = sources
        state["retrieved_documents"] = matches
        state["error"] = None
    except KnowledgeBaseUnavailableError as exc:
        logger.error("Knowledge base unavailable during retrieval: %s", exc)
        state["context"] = ""
        state["sources"] = []
        state["retrieved_documents"] = []
        state["error"] = str(exc)
    state["stage"] = "retrieval"
    state.setdefault("timings", {})["retrieval"] = round(time.time() - start, 3)
    return state


def response_generation_node(state: GraphState) -> GraphState:
    start = time.time()
    if state.get("error"):
        # Knowledge base is down - do not call the LLM with empty context,
        # go straight to the safe fallback message.
        state["response"] = (
            "I'm unable to reach the Airtel knowledge base right now, so I "
            "can't answer reliably. Please try again shortly."
        )
    else:
        state["response"] = generate_response(
            user_input=state["user_input"],
            conversation_history=state.get("conversation_history", []),
            context=state.get("context", ""),
            sources=state.get("sources", []),
            language=state.get("language", "en"),
        )
    state["stage"] = "response_generation"
    state.setdefault("timings", {})["response_generation"] = round(time.time() - start, 3)
    return state


def grounding_validation_node(state: GraphState) -> GraphState:
    start = time.time()
    result = validate_grounding(
        generated_answer=state.get("response", ""),
        context=state.get("context", ""),
        user_input=state["user_input"],
    )
    state["grounded"] = result["grounded"]
    state["grounding_confidence"] = result["confidence"]
    state["grounding_reason"] = result["reason"]
    state["should_escalate"] = result["should_escalate"]
    state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
    state["stage"] = "grounding_validation"
    state.setdefault("timings", {})["grounding_validation"] = round(time.time() - start, 3)
    return state


def fallback_node(state: GraphState) -> GraphState:
    """Used when grounding fails and no retries remain."""
    state["response"] = NO_CONTEXT_FALLBACK
    state["grounded"] = False
    state["stage"] = "fallback"
    return state


def escalation_decision_node(state: GraphState) -> GraphState:
    lowered = state["user_input"].lower()
    account_specific = any(hint in lowered for hint in ACCOUNT_SPECIFIC_HINTS)

    should_escalate = state.get("should_escalate", False) or account_specific
    reason = None
    if account_specific:
        reason = "Request appears to be account-specific; prototype has no account access."
        state["response"] = (
            "I don't have access to your Airtel account or transaction history, "
            "but I can explain how to do this using the official Airtel app or "
            "support process. " + state.get("response", "")
        ).strip()
    elif should_escalate:
        reason = state.get("grounding_reason", "Low confidence answer.")

    state["should_escalate"] = should_escalate
    state["escalation_reason"] = reason
    state["confidence"] = state.get("grounding_confidence", 0.0)
    state["stage"] = "escalation_decision"
    return state


def tts_node(state: GraphState) -> GraphState:
    if not state.get("synthesize_speech"):
        state["stage"] = "tts_skipped"
        return state

    start = time.time()
    try:
        tts_language = state.get("language", "en")
        if tts_language == "mixed":
            tts_language = "en"
        result = tts_engine.synthesize(state.get("response", ""), language=tts_language)
        state["audio_bytes"] = result.audio_bytes
        state["audio_sample_rate"] = result.sample_rate
    except Exception as exc:  # noqa: BLE001 - never crash the graph on TTS failure
        logger.error("TTS synthesis failed: %s", exc)
        state["audio_bytes"] = None
        state["error"] = state.get("error") or f"TTS failed: {exc}"
    state["stage"] = "tts"
    state.setdefault("timings", {})["tts"] = round(time.time() - start, 3)
    return state


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_after_grounding(state: GraphState) -> str:
    if state.get("grounded"):
        return "escalation_decision"
    if state.get("retrieval_attempts", 0) <= settings.MAX_RETRIEVAL_RETRIES:
        return "retrieval"
    return "fallback"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("input_understanding", input_understanding_node)
    graph.add_node("intent_detection", intent_detection_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("response_generation", response_generation_node)
    graph.add_node("grounding_validation", grounding_validation_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("escalation_decision", escalation_decision_node)
    graph.add_node("tts", tts_node)

    graph.set_entry_point("input_understanding")
    graph.add_edge("input_understanding", "intent_detection")
    graph.add_edge("intent_detection", "retrieval")
    graph.add_edge("retrieval", "response_generation")
    graph.add_edge("response_generation", "grounding_validation")

    graph.add_conditional_edges(
        "grounding_validation",
        route_after_grounding,
        {
            "escalation_decision": "escalation_decision",
            "retrieval": "retrieval",
            "fallback": "fallback",
        },
    )
    graph.add_edge("fallback", "escalation_decision")
    graph.add_edge("escalation_decision", "tts")
    graph.add_edge("tts", END)

    return graph.compile()


# Compiled once at import time and reused for every request.
voice_customer_care_graph = build_graph()


def run_workflow(
    conversation_id: str,
    user_input: str,
    conversation_history: list,
    current_service: str | None,
    synthesize_speech: bool = False,
) -> GraphState:
    initial_state: GraphState = {
        "conversation_id": conversation_id,
        "user_input": user_input,
        "conversation_history": conversation_history,
        "current_service": current_service,
        "synthesize_speech": synthesize_speech,
        "timings": {},
    }
    final_state = voice_customer_care_graph.invoke(initial_state)
    return final_state
