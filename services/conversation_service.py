"""
Conversation Service.

Keeps short-term conversational memory per conversation_id so the LangGraph
workflow can resolve references like "slow speed" back to "broadband" from
an earlier turn.

This is an in-memory prototype store (dict + TTL sweep). Swapping this for
Redis or a database later only requires changing this module.
"""
import logging
import threading
import time
import uuid
from typing import Optional

from config import settings

logger = logging.getLogger("voicerag.conversation")


class ConversationStore:
    def __init__(self, ttl_seconds: int = None, max_turns: int = None):
        self.ttl_seconds = ttl_seconds or settings.CONVERSATION_TTL_SECONDS
        self.max_turns = max_turns or settings.MAX_HISTORY_TURNS
        self._lock = threading.Lock()
        self._store: dict[str, dict] = {}

    def create_conversation(self) -> str:
        conversation_id = str(uuid.uuid4())
        with self._lock:
            self._store[conversation_id] = {
                "history": [],
                "current_intent": None,
                "current_service": None,
                "updated_at": time.time(),
            }
        return conversation_id

    def get(self, conversation_id: str) -> dict:
        with self._lock:
            self._sweep_expired_locked()
            record = self._store.get(conversation_id)
            if record is None:
                self._store[conversation_id] = {
                    "history": [],
                    "current_intent": None,
                    "current_service": None,
                    "updated_at": time.time(),
                }
                record = self._store[conversation_id]
            return record

    def append_turn(self, conversation_id: str, role: str, text: str) -> None:
        with self._lock:
            record = self._store.setdefault(
                conversation_id,
                {"history": [], "current_intent": None, "current_service": None, "updated_at": time.time()},
            )
            record["history"].append({"role": role, "text": text})
            record["history"] = record["history"][-(self.max_turns * 2) :]
            record["updated_at"] = time.time()

    def update_context(
        self, conversation_id: str, intent: Optional[str] = None, service: Optional[str] = None
    ) -> None:
        with self._lock:
            record = self._store.setdefault(
                conversation_id,
                {"history": [], "current_intent": None, "current_service": None, "updated_at": time.time()},
            )
            if intent:
                record["current_intent"] = intent
            if service:
                record["current_service"] = service
            record["updated_at"] = time.time()

    def _sweep_expired_locked(self) -> None:
        now = time.time()
        expired = [
            cid for cid, rec in self._store.items() if now - rec["updated_at"] > self.ttl_seconds
        ]
        for cid in expired:
            del self._store[cid]
        if expired:
            logger.info("Swept %d expired conversation(s)", len(expired))


# Module-level singleton
conversation_store = ConversationStore()
