import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.grounding_agent import validate_grounding
from agents.intent_agent import classify_intent
from services.ollama_service import OllamaError


@patch("agents.intent_agent.ollama_service")
def test_classify_intent_happy_path(mock_ollama):
    mock_ollama.generate_json.return_value = {
        "intent": "recharge_failed",
        "service": "prepaid",
        "language": "te",
        "confidence": 0.92,
    }
    result = classify_intent("recharge fail అయింది", [], None)
    assert result["intent"] == "recharge_failed"
    assert result["service"] == "prepaid"
    assert result["language"] == "te"
    assert 0.0 <= result["confidence"] <= 1.0


@patch("agents.intent_agent.ollama_service")
def test_classify_intent_falls_back_on_ollama_error(mock_ollama):
    mock_ollama.generate_json.side_effect = OllamaError("connection refused")
    result = classify_intent("hello", [], "broadband")
    assert result["service"] == "broadband"
    assert result["confidence"] < 0.5


@patch("agents.intent_agent.ollama_service")
def test_classify_intent_rejects_invalid_service(mock_ollama):
    mock_ollama.generate_json.return_value = {
        "intent": "unknown",
        "service": "not_a_real_service",
        "language": "en",
        "confidence": 0.8,
    }
    result = classify_intent("test", [], "wifi")
    assert result["service"] == "wifi"


@patch("agents.grounding_agent.ollama_service")
def test_validate_grounding_empty_context_is_never_grounded(mock_ollama):
    result = validate_grounding("Some answer", context="", user_input="question")
    assert result["grounded"] is False
    assert result["should_escalate"] is True
    mock_ollama.generate_json.assert_not_called()


@patch("agents.grounding_agent.ollama_service")
def test_validate_grounding_blocks_sensitive_terms_even_if_model_says_grounded(mock_ollama):
    mock_ollama.generate_json.return_value = {
        "grounded": True,
        "confidence": 0.95,
        "reason": "looks fine",
        "should_escalate": False,
    }
    result = validate_grounding(
        "Please share your OTP so I can help.", context="some context", user_input="q"
    )
    assert result["grounded"] is False
    assert result["should_escalate"] is True
