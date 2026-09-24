"""Vercel serverless endpoint for the knowledge-base assistant.

GET /api/ask?q=...&mode=reranked|hybrid|bm25|dense   -> answer, citations, ranked passages
GET /api/ask?random=in|out                           -> a random test question (in or out of the knowledge base)
GET /api/ask?random=in&good=1                        -> a random test question it answers correctly (page's opening example)
GET /api/ask?meta=1                                  -> evaluation results and knowledge-base stats
"""
from __future__ import annotations

import json
import os
import random
import sys
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.corpus import load_questions  # noqa: E402
from rag.pipeline import get_assistant  # noqa: E402
from rag.text import tokens  # noqa: E402

MODES = ("reranked", "hybrid", "bm25", "dense")
MAX_Q = 300


@lru_cache(maxsize=2)
def questions(kind: str) -> list[dict]:
    return load_questions("eval" if kind == "in" else "heldout_test")


def meta() -> dict:
    a = get_assistant()
    res = json.loads((ROOT / "results" / "results.json").read_text())
    titles = sorted({p.title for p in a.idx.passages})
    return {"results": res, "kb": {"passages": len(a.idx.passages), "articles": len(titles),
                                   "sample_titles": random.Random(0).sample(titles, 12)}}


def ask(q: str, mode: str) -> dict:
    r = get_assistant().ask(q, mode=mode, k=5)
    qterms = set(tokens(q))
    for p in r["passages"]:  # which words to highlight in the UI
        p["match_terms"] = sorted({w for w in tokens(p["text"]) if w in qterms})
    r["query_terms"] = sorted(qterms)
    return r


def respond(params: dict) -> tuple[int, dict]:
    if params.get("meta"):
        return 200, meta()
    kind = params.get("random")
    if kind in ("in", "out"):
        item = random.choice(questions(kind))
        if kind == "in" and params.get("good"):
            # opening example for the page: a random test question the assistant answers correctly
            for _ in range(30):
                r = get_assistant().ask(item["q"])
                if r["answer"] and any(a in r["answer"]["text"] for a in item["answers"]):
                    break
                item = random.choice(questions(kind))
        return 200, {"question": item["q"], "answers": item["answers"], "in_kb": kind == "in",
                     "article": item["title"]}
    q = (params.get("q") or "").strip()[:MAX_Q]
    if not q:
        return 400, {"error": "missing q"}
    mode = params.get("mode", "reranked")
    if mode not in MODES:
        return 400, {"error": f"mode must be one of {MODES}"}
    return 200, ask(q, mode)


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        status, body = respond(params)
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        cacheable = status == 200 and "random" not in params
        self.send_header("Cache-Control", "public, s-maxage=86400, max-age=600" if cacheable else "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if os.environ.get("VERCEL"):
    get_assistant()  # load index at cold start, not on the first request
