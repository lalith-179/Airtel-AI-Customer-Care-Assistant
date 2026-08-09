import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.conversation_service import ConversationStore


def test_create_and_append_turn():
    store = ConversationStore(ttl_seconds=3600, max_turns=4)
    cid = store.create_conversation()
    store.append_turn(cid, "user", "My broadband is slow.")
    store.append_turn(cid, "assistant", "Is it slow speed or no connectivity?")
    record = store.get(cid)
    assert len(record["history"]) == 2
    assert record["history"][0]["role"] == "user"


def test_update_context_tracks_service_and_intent():
    store = ConversationStore(ttl_seconds=3600, max_turns=4)
    cid = store.create_conversation()
    store.update_context(cid, intent="slow_speed", service="broadband")
    record = store.get(cid)
    assert record["current_service"] == "broadband"
    assert record["current_intent"] == "slow_speed"


def test_history_is_trimmed_to_max_turns():
    store = ConversationStore(ttl_seconds=3600, max_turns=2)
    cid = store.create_conversation()
    for i in range(10):
        store.append_turn(cid, "user", f"message {i}")
    record = store.get(cid)
    assert len(record["history"]) <= 4  # max_turns * 2 (user+assistant)


def test_unknown_conversation_id_returns_fresh_record():
    store = ConversationStore(ttl_seconds=3600, max_turns=4)
    record = store.get("does-not-exist")
    assert record["history"] == []
