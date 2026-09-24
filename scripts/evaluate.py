"""Evaluate retrieval, answer selection and abstention on data the models never saw.

In-KB:     SQuAD v1.1 dev questions (10,570); gold passage is in the index.
Out-of-KB: questions about 20 held-out Wikipedia articles that are not indexed (heldout_test).
Writes results/results.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rag.corpus import load_questions
from rag.pipeline import Assistant

OUT = Path(__file__).resolve().parent.parent / "results" / "results.json"


def pct(xs, p):
    return float(np.percentile(xs, p)) if xs else 0.0


def run(limit: int | None = None) -> dict:
    a = Assistant()
    ev = load_questions("eval")[:limit]
    out_kb = load_questions("heldout_test")[:limit]
    a.ask("warm up the caches")
    res = {"n_eval": len(ev), "n_out_of_kb": len(out_kb), "retrieval": {}}

    for mode in ("bm25", "dense", "hybrid", "reranked"):
        ranks, lat = [], []
        answered = correct = 0
        for q in ev:
            t = time.perf_counter()
            r = a.ask(q["q"], mode=mode, k=10)
            lat.append((time.perf_counter() - t) * 1000)
            pids = [p["pid"] for p in r["passages"]]
            ranks.append(pids.index(q["pid"]) if q["pid"] in pids else None)
            if mode == "reranked" and r["answer"] is not None:
                answered += 1
                correct += any(ans in r["answer"]["text"] for ans in q["answers"])
        n = len(ev)
        row = {
            "R@1": sum(x is not None and x < 1 for x in ranks) / n,
            "R@5": sum(x is not None and x < 5 for x in ranks) / n,
            "R@10": sum(x is not None for x in ranks) / n,
            "MRR@10": sum(1 / (x + 1) for x in ranks if x is not None) / n,
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
        }
        if mode == "reranked":
            row |= {"answer_rate": answered / n, "answer_acc_when_answered": correct / max(answered, 1),
                    "answer_acc_overall": correct / n}
        res["retrieval"][mode] = row
        print(f"{mode:9s} " + " ".join(f"{k} {v:.3f}" for k, v in row.items()))

    # questions whose answer is not in the KB: the assistant should decline
    abstained = sum(a.ask(q["q"])["abstained"] for q in out_kb)
    res["out_of_kb_abstain_rate"] = abstained / len(out_kb)
    print(f"out-of-KB abstain rate {res['out_of_kb_abstain_rate']:.3f} on {len(out_kb):,} questions")
    return res


def table(res: dict) -> str:
    names = {"bm25": "BM25 (keyword)", "dense": "Dense (WordLlama)", "hybrid": "Hybrid (RRF)",
             "reranked": "Hybrid + learned re-ranker"}
    lines = ["| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 latency | p95 latency |",
             "|---|---|---|---|---|---|---|"]
    for k, r in res["retrieval"].items():
        lines.append(f"| {names[k]} | {r['R@1']:.1%} | {r['R@5']:.1%} | {r['R@10']:.1%} | {r['MRR@10']:.3f} | "
                     f"{r['p50_ms']:.1f} ms | {r['p95_ms']:.1f} ms |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    res = run(args.limit)
    if not args.limit:
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res, indent=2))
    print()
    print(table(res))
