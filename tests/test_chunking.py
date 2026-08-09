import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_knowledge_base import chunk_text, clean_text, infer_category


def test_clean_text_collapses_whitespace():
    dirty = "Hello    world\r\n\r\n\r\n\r\nNext   line"
    cleaned = clean_text(dirty)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "    " not in cleaned


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 100


def test_chunk_text_empty_input_returns_empty_list():
    assert chunk_text("", chunk_size=100, chunk_overlap=10) == []


def test_chunk_text_rejects_overlap_gte_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=50, chunk_overlap=50)


def test_infer_category_from_filename():
    assert infer_category("broadband_faq") == "broadband"
    assert infer_category("recharge_faq") == "recharge"
    assert infer_category("random_doc") == "general"
