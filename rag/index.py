"""Loads the persisted index and runs first-stage retrieval (BM25, dense, hybrid)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from rag.bm25 import BM25
from rag.corpus import load_passages
from rag.embed import embed
from rag.features import passage_info

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index" / "index.npz"
RRF_K = 60


class Index:
    def __init__(self):
        a = np.load(INDEX_PATH, allow_pickle=False)
        self.bm25 = BM25.from_arrays(a)
        self.emb = a["emb"].astype(np.float32)
        self.passages = load_passages()
        self._info: dict[int, dict] = {}

    def info(self, pid: int) -> dict:
        d = self._info.get(pid)
        if d is None:
            d = self._info[pid] = passage_info(self.passages[pid])
        return d

    def bm25_scores(self, q: str) -> np.ndarray:
        return self.bm25.scores(q)

    def dense_scores(self, q: str, qvec: np.ndarray | None = None) -> np.ndarray:
        v = embed([q])[0] if qvec is None else qvec
        return self.emb @ v

    @staticmethod
    def top(scores: np.ndarray, k: int) -> np.ndarray:
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        return idx[np.argsort(-scores[idx], kind="stable")]

    def retrieve(self, q: str, mode: str = "hybrid", k: int = 10, depth: int = 100) -> dict:
        """First-stage retrieval. Returns ranked pids plus the raw signals the re-ranker uses."""
        sb = self.bm25_scores(q) if mode in ("bm25", "hybrid") else None
        sd = self.dense_scores(q) if mode in ("dense", "hybrid") else None
        if mode == "bm25":
            return {"pids": self.top(sb, k), "bm25": sb, "dense": None}
        if mode == "dense":
            return {"pids": self.top(sd, k), "bm25": None, "dense": sd}
        # Reciprocal rank fusion of the two lists: robust to their very different score scales.
        fused: dict[int, float] = {}
        for s in (sb, sd):
            for r, pid in enumerate(self.top(s, depth)):
                fused[int(pid)] = fused.get(int(pid), 0.0) + 1.0 / (RRF_K + r + 1)
        pids = np.array(sorted(fused, key=fused.get, reverse=True)[:max(k, depth)])
        return {"pids": pids[:k] if k < depth else pids, "bm25": sb, "dense": sd, "rrf": fused}


@lru_cache(maxsize=1)
def get_index() -> Index:
    return Index()
