"""
Shared state object passed through the LangGraph workflow.

Using a TypedDict keeps the graph nodes decoupled - each node reads the
keys it needs and writes the keys it produces.
"""
from typing import List, Optional, TypedDict


class RetrievedDocument(TypedDict):
    text: str
    source_url: str
    source_title: str
    category: str
    document_type: str
    score: float


class ConversationTurn(TypedDict):
    role: str  # "user" | "assistant"
    text: str


class GraphState(TypedDict, total=False):
    # Identity / input
    conversation_id: str
    user_input: str
    language: str

    # Conversation memory
    conversation_history: List[ConversationTurn]
    current_intent: Optional[str]
    current_service: Optional[str]

    # Intent understanding
    intent: Optional[str]
    service: Optional[str]
    intent_confidence: float

    # Retrieval
    retrieved_documents: List[RetrievedDocument]
    context: str
    sources: List[dict]

    # Generation
    response: str
    confidence: float

    # Grounding
    grounded: bool
    grounding_confidence: float
    grounding_reason: str
    retrieval_attempts: int

    # Escalation
    should_escalate: bool
    escalation_reason: Optional[str]

    # Pipeline bookkeeping
    stage: str
    error: Optional[str]
    timings: dict

    # Voice output (populated only when synthesize_speech=True)
    synthesize_speech: bool
    audio_bytes: Optional[bytes]
    audio_sample_rate: Optional[int]
