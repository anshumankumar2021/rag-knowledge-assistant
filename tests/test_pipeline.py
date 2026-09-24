"""End-to-end checks against the committed index and models."""
import json
from pathlib import Path

from api.ask import respond
from rag.corpus import load_questions
from rag.pipeline import get_assistant

ROOT = Path(__file__).resolve().parent.parent


def test_every_answer_is_a_verbatim_quote_from_its_cited_passage():
    a = get_assistant()
    answered = 0
    for q in load_questions("eval")[:50]:
        r = a.ask(q["q"])
        if r["answer"] is None:
            continue
        answered += 1
        src = a.idx.passages[r["answer"]["pid"]]
        assert src.text[r["answer"]["start"]:r["answer"]["end"]] == r["answer"]["text"]
        assert r["answer"]["pid"] in [p["pid"] for p in r["passages"]]
    assert answered >= 30   # it answers most in-KB questions (about 82% on the full test set)


def test_declines_question_outside_knowledge_base():
    r = get_assistant().ask("What is the recommended torque for a 2019 Honda Civic lug nut?")
    assert r["abstained"] and r["answer"] is None


def test_all_modes_and_api_contract():
    for mode in ("reranked", "hybrid", "bm25", "dense"):
        status, body = respond({"q": "Who designed the Eiffel Tower?", "mode": mode})
        assert status == 200 and len(body["passages"]) == 5
    assert respond({"q": ""})[0] == 400
    assert respond({"q": "x", "mode": "bad"})[0] == 400
    status, body = respond({"random": "out"})
    assert status == 200 and body["in_kb"] is False


def test_reported_results_show_reranker_beats_first_stage():
    res = json.loads((ROOT / "results" / "results.json").read_text())["retrieval"]
    assert res["reranked"]["R@1"] > res["bm25"]["R@1"] > res["hybrid"]["R@1"] > res["dense"]["R@1"]


def test_opening_example_is_answered_correctly():
    for _ in range(3):
        status, item = respond({"random": "in", "good": "1"})
        assert status == 200
        r = get_assistant().ask(item["question"])
        assert r["answer"] and any(a in r["answer"]["text"] for a in item["answers"])
