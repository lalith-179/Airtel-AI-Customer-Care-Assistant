"""
Evaluation script.

Runs a small evaluation dataset through the same LangGraph workflow used at
runtime and reports:
    - retrieval relevance (expected source found in retrieved sources)
    - grounding pass rate
    - approximate hallucination rate (1 - grounding pass rate)
    - intent accuracy
    - end-to-end response latency
    - escalation accuracy

STT/TTS accuracy are reported as manual/optional checks (they require real
audio samples, which are outside the scope of an automated text-based
evaluation run) - see the printed notes at the end of the report.

Usage:
    python scripts/evaluate.py --dataset scripts/eval_dataset.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workflows.voice_customer_care_graph import run_workflow  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(dataset: list[dict]) -> dict:
    total = len(dataset)
    retrieval_hits = 0
    grounded_count = 0
    intent_hits = 0
    escalation_hits = 0
    latencies = []

    rows = []
    for case in dataset:
        conversation_id = f"eval-{case['id']}"
        start = time.time()
        state = run_workflow(
            conversation_id=conversation_id,
            user_input=case["question"],
            conversation_history=[],
            current_service=None,
            synthesize_speech=False,
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        retrieved_titles = {s.get("title", "").lower() for s in state.get("sources", [])}
        expected_source = case.get("expected_source", "").lower()
        retrieval_hit = bool(expected_source) and any(expected_source in t for t in retrieved_titles)
        retrieval_hits += int(retrieval_hit)

        grounded = bool(state.get("grounded"))
        grounded_count += int(grounded)

        intent_hit = state.get("intent") == case.get("expected_intent")
        intent_hits += int(intent_hit)

        escalation_hit = state.get("should_escalate") == case.get("expected_escalate", False)
        escalation_hits += int(escalation_hit)

        rows.append({
            "id": case["id"],
            "question": case["question"],
            "intent": state.get("intent"),
            "expected_intent": case.get("expected_intent"),
            "grounded": grounded,
            "retrieval_hit": retrieval_hit,
            "should_escalate": state.get("should_escalate"),
            "latency_s": round(elapsed, 2),
        })

    summary = {
        "total_cases": total,
        "retrieval_relevance": round(retrieval_hits / total, 3) if total else 0,
        "grounding_pass_rate": round(grounded_count / total, 3) if total else 0,
        "approx_hallucination_rate": round(1 - (grounded_count / total), 3) if total else 0,
        "intent_accuracy": round(intent_hits / total, 3) if total else 0,
        "escalation_accuracy": round(escalation_hits / total, 3) if total else 0,
        "avg_latency_s": round(sum(latencies) / total, 2) if total else 0,
        "rows": rows,
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the VoiceRAG pipeline")
    parser.add_argument("--dataset", type=str, default="scripts/eval_dataset.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path

    dataset = load_dataset(dataset_path)
    results = evaluate(dataset)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nNotes:")
    print("- STT accuracy: run manually by comparing transcripts to known audio samples.")
    print("- TTS success: run manually by confirming audio playback for a few sample answers.")
