"""End-to-end question answering: retrieve -> re-rank -> pick answer sentence -> answer or abstain."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from rag.embed import embed
from rag.features import PASSAGE_FEATURES, passage_features, sentence_features
from rag.index import Index, get_index
from rag.ranker import ListRanker, Logistic
from rag.text import sentences

MODELS_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "models.json"
CANDIDATES = 50     # first-stage depth per retriever fed to the re-ranker
TOP_PASSAGES = 3    # passages whose sentences compete for the answer


def candidates(idx: Index, q: str, depth: int = CANDIDATES):
    """Union of BM25 and dense top-`depth`, plus everything the re-ranker needs about them."""
    qvec = embed([q])[0]
    sb, sd = idx.bm25_scores(q), idx.dense_scores(q, qvec)
    tb, td = idx.top(sb, depth), idx.top(sd, depth)
    rank_b = {int(p): r for r, p in enumerate(tb)}
    rank_d = {int(p): r for r, p in enumerate(td)}
    pids = np.array(list(dict.fromkeys([*map(int, tb), *map(int, td)])), dtype=np.int64)
    X = passage_features(q, pids, idx.info, sb, sd, rank_b, rank_d, idx.bm25.term_idf)
    return pids, X, qvec, sb, sd


def sentence_candidates(idx: Index, pids: np.ndarray, probs: np.ndarray) -> list[dict]:
    out = []
    for r, (pid, pp) in enumerate(zip(pids[:TOP_PASSAGES], probs[:TOP_PASSAGES])):
        text = idx.passages[pid].text
        for i, (a, z) in enumerate(sentences(text)):
            out.append({"pid": int(pid), "start": a, "end": z, "text": text[a:z], "p_prob": float(pp),
                        "p_rank": r, "s_pos": i})
    return out


def abstain_features(top_row: np.ndarray, top_prob: float, sent_best: float) -> np.ndarray:
    f = dict(zip(PASSAGE_FEATURES, top_row))
    return np.array([f["bm25_raw"], f["dense"], f["coverage"], f["bigram_cov"], f["best_sent_cov"],
                     top_prob, sent_best], dtype=np.float64)


class Assistant:
    def __init__(self, idx: Index | None = None, models: dict | None = None):
        self.idx = idx or get_index()
        if models is None:
            d = json.loads(MODELS_PATH.read_text())
            models = {"passage": ListRanker.from_dict(d["passage"]), "sentence": ListRanker.from_dict(d["sentence"]),
                      "abstain": Logistic.from_dict(d["abstain"]), "threshold": d["threshold"]}
        self.m = models

    def ask(self, q: str, mode: str = "reranked", k: int = 5) -> dict:
        t0 = time.perf_counter()
        idx = self.idx
        if mode in ("bm25", "dense", "hybrid"):
            res = idx.retrieve(q, mode=mode, k=k)
            pids = np.asarray(res["pids"][:k])
            probs = np.full(len(pids), np.nan)
            t_ret = time.perf_counter()
            top = {"pids": pids, "probs": probs}
            answer = None
            conf = None
        else:
            pids, X, qvec, _, _ = candidates(idx, q)
            s = self.m["passage"].score(X)
            order = np.argsort(-s)
            pids, X, s = pids[order], X[order], s[order]
            probs = np.exp(s - s.max())
            probs /= probs.sum()
            t_ret = time.perf_counter()
            cands = sentence_candidates(idx, pids, probs)
            svecs = embed([c["text"] for c in cands])
            ss = self.m["sentence"].score(sentence_features(q, cands, qvec, svecs, idx.bm25.term_idf))
            sp = np.exp(ss - ss.max())
            sp /= sp.sum()
            best = int(np.argmax(ss))
            conf = float(self.m["abstain"].prob(abstain_features(X[0], float(probs[0]), float(sp[best]))[None])[0])
            answer = cands[best] | {"sentence_prob": float(sp[best])}
            top = {"pids": pids[:k], "probs": probs[:k]}
        t_end = time.perf_counter()
        abstained = conf is not None and conf < self.m["threshold"]
        return {
            "question": q,
            "mode": mode,
            "answer": None if (answer is None or abstained) else answer,
            "abstained": abstained,
            "confidence": conf,
            "threshold": self.m["threshold"],
            "passages": [{"pid": int(p), "title": idx.passages[p].title, "text": idx.passages[p].text,
                          "prob": None if np.isnan(pr) else float(pr)} for p, pr in zip(top["pids"], top["probs"])],
            "timing_ms": {"retrieve": round((t_ret - t0) * 1000, 2), "total": round((t_end - t0) * 1000, 2)},
        }


@lru_cache(maxsize=1)
def get_assistant() -> Assistant:
    return Assistant()
