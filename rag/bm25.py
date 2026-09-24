"""Okapi BM25 over an inverted index held in flat numpy arrays (fast to load, no scipy needed)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from rag.text import tokens


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> "BM25":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(docs), dtype=np.float32)
        for i, d in enumerate(docs):
            tf = Counter(tokens(d))
            lengths[i] = sum(tf.values())
            for t, c in tf.items():
                postings[t].append((i, c))
        self.vocab = {t: j for j, t in enumerate(sorted(postings))}
        offs, ids, tfs = [0], [], []
        for t in sorted(postings):
            ps = postings[t]
            ids.extend(p[0] for p in ps)
            tfs.extend(p[1] for p in ps)
            offs.append(len(ids))
        self.offsets = np.asarray(offs, dtype=np.int64)
        self.doc_ids = np.asarray(ids, dtype=np.int32)
        self.tfs = np.asarray(tfs, dtype=np.float32)
        self.lengths = lengths
        self._finish()
        return self

    def _finish(self) -> None:
        n = len(self.lengths)
        df = np.diff(self.offsets).astype(np.float32)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        self.norm = (self.k1 * (1 - self.b + self.b * self.lengths / self.lengths.mean())).astype(np.float32)

    def term_idf(self, term: str) -> float:
        j = self.vocab.get(term)
        return float(self.idf[j]) if j is not None else 0.0

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.lengths), dtype=np.float32)
        for t in set(tokens(query)):
            j = self.vocab.get(t)
            if j is None:
                continue
            a, z = self.offsets[j], self.offsets[j + 1]
            ids, tf = self.doc_ids[a:z], self.tfs[a:z]
            s[ids] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.norm[ids])
        return s

    # --- persistence
    def to_arrays(self) -> dict:
        terms = np.array(sorted(self.vocab, key=self.vocab.get))
        return {"bm25_terms": terms, "bm25_offsets": self.offsets, "bm25_doc_ids": self.doc_ids,
                "bm25_tfs": self.tfs.astype(np.uint16), "bm25_lengths": self.lengths}

    @classmethod
    def from_arrays(cls, a) -> "BM25":
        m = cls()
        m.vocab = {t: j for j, t in enumerate(a["bm25_terms"].tolist())}
        m.offsets, m.doc_ids = a["bm25_offsets"], a["bm25_doc_ids"]
        m.tfs = a["bm25_tfs"].astype(np.float32)
        m.lengths = a["bm25_lengths"]
        m._finish()
        return m
