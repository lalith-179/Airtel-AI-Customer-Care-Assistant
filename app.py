"""
VoiceRAG - Airtel AI Customer Care Voice Assistant
Flask application entry point.

Runtime responsibilities ONLY:
    - Serve the chat UI.
    - Accept text messages (REST) and voice audio (WebSocket) from the browser.
    - Run STT on incoming audio.
    - Invoke the shared LangGraph workflow (text and voice share this).
    - Return the generated text answer, sources, and (for voice) synthesized
      speech audio.

This file NEVER builds or rebuilds the knowledge base - that happens only
in scripts/build_knowledge_base.py.
"""
import base64
import logging
import os
import sys
import time
import uuid

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from config import settings
from services.conversation_service import conversation_store
from services.ollama_service import ollama_service
from services.rag_service import knowledge_base_status
from services.retrieval_service import retrieval_service
from services.speech_to_text import stt_engine
from services.text_to_speech import tts_engine
from workflows.voice_customer_care_graph import run_workflow

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(settings.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(settings.LOG_DIR, "voicerag.log")),
    ],
)
logger = logging.getLogger("voicerag.app")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = settings.FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB cap on uploaded audio

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", max_http_buffer_size=20 * 1024 * 1024)

os.makedirs(settings.AUDIO_TMP_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_log_conversation_event(conversation_id: str, stage: str, **extra):
    """Structured logging that never includes sensitive fields."""
    scrubbed = {k: v for k, v in extra.items() if k.lower() not in {
        "password", "otp", "pin", "card_number", "cvv",
    }}
    logger.info("conversation_id=%s stage=%s %s", conversation_id, stage, scrubbed)


def _run_pipeline(conversation_id: str, user_text: str, synthesize_speech: bool) -> dict:
    record = conversation_store.get(conversation_id)

    if not user_text.strip():
        return {"error": "empty_query", "message": "I didn't catch a question - please try again."}

    conversation_store.append_turn(conversation_id, "user", user_text)

    start = time.time()
    final_state = run_workflow(
        conversation_id=conversation_id,
        user_input=user_text,
        conversation_history=record["history"],
        current_service=record.get("current_service"),
        synthesize_speech=synthesize_speech,
    )
    total_time = time.time() - start

    conversation_store.append_turn(conversation_id, "assistant", final_state.get("response", ""))
    conversation_store.update_context(
        conversation_id,
        intent=final_state.get("intent"),
        service=final_state.get("service"),
    )

    _safe_log_conversation_event(
        conversation_id,
        "pipeline_complete",
        intent=final_state.get("intent"),
        service=final_state.get("service"),
        grounded=final_state.get("grounded"),
        grounding_confidence=final_state.get("grounding_confidence"),
        should_escalate=final_state.get("should_escalate"),
        retrieved_count=len(final_state.get("retrieved_documents", [])),
        total_time_s=round(total_time, 3),
        timings=final_state.get("timings"),
    )

    audio_b64 = None
    if synthesize_speech and final_state.get("audio_bytes"):
        audio_b64 = base64.b64encode(final_state["audio_bytes"]).decode("ascii")

    return {
        "conversation_id": conversation_id,
        "transcript": user_text,
        "response": final_state.get("response", ""),
        "intent": final_state.get("intent"),
        "service": final_state.get("service"),
        "language": final_state.get("language"),
        "grounded": final_state.get("grounded"),
        "confidence": round(final_state.get("confidence", 0.0), 3),
        "should_escalate": final_state.get("should_escalate", False),
        "escalation_reason": final_state.get("escalation_reason"),
        "sources": final_state.get("sources", []),
        "audio_base64": audio_b64,
        "warning": final_state.get("error"),
    }


# ---------------------------------------------------------------------------
# Routes - UI
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", disclaimer=settings.DISCLAIMER)


# ---------------------------------------------------------------------------
# Routes - REST API
# ---------------------------------------------------------------------------
@app.route("/api/conversation/start", methods=["POST"])
def start_conversation():
    conversation_id = conversation_store.create_conversation()
    return jsonify({"conversation_id": conversation_id})


@app.route("/api/chat/text", methods=["POST"])
def chat_text():
    payload = request.get_json(silent=True) or {}
    user_text = str(payload.get("message", "")).strip()
    conversation_id = payload.get("conversation_id") or conversation_store.create_conversation()
    synthesize_speech = bool(payload.get("synthesize_speech", False))

    if not user_text:
        return jsonify({"error": "empty_query", "message": "Please type a question."}), 400
    if len(user_text) > 2000:
        return jsonify({"error": "input_too_long", "message": "Please shorten your question."}), 400

    try:
        result = _run_pipeline(conversation_id, user_text, synthesize_speech)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in text pipeline")
        return jsonify({
            "error": "internal_error",
            "message": "Something went wrong on our side. Please try again.",
        }), 500

    return jsonify(result)


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "ollama_available": ollama_service.is_available(),
        "knowledge_base": knowledge_base_status(),
        "stt_ready": stt_engine.is_ready(),
        "tts_ready": tts_engine.is_ready(),
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "vision_model": settings.VISION_MODEL,
        "disclaimer": settings.DISCLAIMER,
    })


@app.route("/api/knowledge-base/reload", methods=["POST"])
def reload_knowledge_base():
    """Re-opens the (already-built) ChromaDB collection. Does NOT rebuild it.

    Use this after running scripts/build_knowledge_base.py to pick up a
    fresh collection without restarting the Flask process.
    """
    retrieval_service.reload()
    return jsonify(knowledge_base_status())


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "not_found"}), 404


@app.errorhandler(500)
def internal_error(_e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "internal_error", "message": "Something went wrong on our side."}), 500


# ---------------------------------------------------------------------------
# WebSocket - voice interaction
# ---------------------------------------------------------------------------
@socketio.on("connect")
def handle_connect():
    logger.info("Socket connected: sid=%s", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Socket disconnected: sid=%s", request.sid)


@socketio.on("voice_message")
def handle_voice_message(data):
    """
    Expects: { conversation_id, audio_base64, mime_type }
    Emits progressively:
        status: "transcribing" -> "transcript" -> status: "thinking" ->
        status: "searching" -> status: "generating" -> status: "speaking" ->
        "voice_response"
    """
    conversation_id = data.get("conversation_id") or conversation_store.create_conversation()
    audio_b64 = data.get("audio_base64")
    mime_type = data.get("mime_type", "audio/webm")

    if not audio_b64:
        emit("error", {"message": "No audio received."})
        return

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:  # noqa: BLE001
        emit("error", {"message": "Audio could not be decoded."})
        return

    if len(audio_bytes) == 0:
        emit("error", {"message": "Empty audio recording."})
        return

    if settings.PERSIST_RAW_AUDIO:
        raw_path = os.path.join(settings.AUDIO_TMP_DIR, f"{uuid.uuid4()}.webm")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

    emit("status", {"stage": "transcribing", "message": "Transcribing..."})
    try:
        transcription = stt_engine.transcribe(audio_bytes, filename_hint=f"audio.{mime_type.split('/')[-1]}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("STT failed")
        emit("error", {"message": "Sorry, I couldn't understand the audio. Please try again."})
        return

    user_text = transcription.text.strip()
    emit("transcript", {"text": user_text, "language": transcription.language})

    if not user_text:
        emit("error", {"message": "I didn't catch that - please try speaking again."})
        return

    emit("status", {"stage": "searching", "message": "Searching knowledge..."})
    emit("status", {"stage": "thinking", "message": "Thinking..."})

    try:
        result = _run_pipeline(conversation_id, user_text, synthesize_speech=True)
    except Exception:  # noqa: BLE001
        logger.exception("Unhandled error in voice pipeline")
        emit("error", {"message": "Something went wrong while generating a response."})
        return

    emit("status", {"stage": "speaking", "message": "Speaking..."})
    emit("voice_response", result)


if __name__ == "__main__":
    logger.info("Starting VoiceRAG on %s:%s", settings.FLASK_HOST, settings.FLASK_PORT)
    logger.info("Knowledge base status: %s", knowledge_base_status())
    socketio.run(app, host=settings.FLASK_HOST, port=settings.FLASK_PORT, debug=settings.FLASK_DEBUG)
