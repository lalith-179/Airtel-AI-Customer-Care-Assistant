"""
Thin client around the local Ollama HTTP API.

Every module that needs an LLM completion, a structured-JSON completion, or
an embedding goes through this service instead of calling `requests` itself.
This keeps model names, timeouts, and the Ollama endpoint contract in one
place, and reuses a single `requests.Session` (connection pooling) across
calls instead of opening a new connection per request.
"""
import json
import logging
import time
from typing import List, Optional

import requests

from config import settings

logger = logging.getLogger("voicerag.ollama")


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an unexpected response."""


class OllamaService:
    def __init__(self, base_url: str = None, timeout: int = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.OLLAMA_REQUEST_TIMEOUT
        self._session = requests.Session()

    # -- health -------------------------------------------------------
    def is_available(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.RequestException as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False

    # -- generation -----------------------------------------------------
    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> str:
        """Call /api/generate and return the raw text response."""
        payload = {
            "model": model or settings.LLM_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        start = time.time()
        try:
            resp = self._session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama generate() request failed: {exc}") from exc

        elapsed = time.time() - start
        logger.info("ollama.generate model=%s took=%.2fs", payload["model"], elapsed)

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON response: {resp.text[:200]}") from exc

        return data.get("response", "")

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
        retries: int = 1,
    ) -> dict:
        """Call generate() in JSON mode and parse the result, with one retry
        on malformed JSON (models sometimes wrap JSON in stray text)."""
        last_error = None
        for attempt in range(retries + 1):
            raw = self.generate(
                prompt=prompt, system=system, model=model, json_mode=True, temperature=temperature
            )
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError) as exc:
                last_error = exc
                extracted = _extract_json_object(raw)
                if extracted is not None:
                    try:
                        return json.loads(extracted)
                    except json.JSONDecodeError as exc2:
                        last_error = exc2
                logger.warning(
                    "Malformed JSON from model (attempt %d/%d): %s",
                    attempt + 1, retries + 1, raw[:200],
                )
        raise OllamaError(f"Model did not return valid JSON after retries: {last_error}")

    # -- embeddings -------------------------------------------------------
    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        payload = {"model": model or settings.EMBEDDING_MODEL, "prompt": text}
        try:
            resp = self._session.post(
                f"{self.base_url}/api/embeddings", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama embed() request failed: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned non-JSON embedding response: {resp.text[:200]}") from exc

        embedding = data.get("embedding")
        if not embedding:
            raise OllamaError(f"Ollama embedding response missing 'embedding' field: {data}")
        return embedding


def _extract_json_object(text: str) -> Optional[str]:
    """Best-effort extraction of the first {...} block from noisy model output."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


# Module-level singleton reused across the app (connection pooling)
ollama_service = OllamaService()
