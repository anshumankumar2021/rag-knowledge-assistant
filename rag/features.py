"""Hand-built ranking features for (question, passage) and (question, sentence) pairs."""
from __future__ import annotations

import math
import re

import numpy as np

from rag.text import sentences, tokens

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUM = re.compile(r"\b\d[\d,.]*\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion)\b", re.I)
_MONTH = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\b")
_CAP = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")

PASSAGE_FEATURES = ["bm25_norm", "bm25_raw", "dense", "dense_gap", "bm25_rank", "dense_rank", "coverage",
                    "bigram_cov", "title_cov", "best_sent_cov", "log_len", "bm25_x_dense",
                    "best_pair_cov", "best_sent_idf", "q_len", "coverage_gap"]
SENTENCE_FEATURES = ["s_coverage", "s_bigram", "s_dense", "p_prob", "p_rank", "s_pos", "s_len",
                     "type_match", "novel_caps", "s_numbers"]


def qtype(q: str) -> str:
    ql = q.lower()
    if re.search(r"\b(what|which) year\b|\bwhen\b", ql):
        return "time"
    if re.search(r"\bhow (many|much|long|far|old|big|large)\b|\bwhat (percentage|percent|number)\b", ql):
        return "number"
    if re.search(r"\bwho\b|\bwhose\b|\bwhom\b", ql):
        return "person"
    if re.search(r"\bwhere\b", ql):
        return "place"
    return "other"


def _cov(qterms: list[str], text_terms: set[str], idf) -> float:
    tot = sum(idf(t) for t in qterms) or 1.0
    return sum(idf(t) for t in qterms if t in text_terms) / tot


def _bigrams(ts: list[str]) -> set[tuple[str, str]]:
    return set(zip(ts, ts[1:]))


def passage_info(p) -> dict:
    """Token data for one passage, computed once and cached by the index."""
    body = tokens(p.text)
    return {"set": set(body), "big": _bigrams(body), "len": len(body), "title": set(tokens(p.title)),
            "sents": [set(tokens(p.text[a:z])) for a, z in sentences(p.text)]}


def passage_features(q: str, pids: np.ndarray, info, sb: np.ndarray, sd: np.ndarray,
                     rank_b: dict, rank_d: dict, idf) -> np.ndarray:
    qt = list(dict.fromkeys(tokens(q)))
    qbi = _bigrams(tokens(q))
    bmax = float(max(sb[pids].max(), 1e-6))
    dmax = float(sd[pids].max())
    qidf = {t: idf(t) for t in qt}
    rows = []
    for pid in pids:
        pi = info(int(pid))
        sents = pi["sents"]
        best = max((_cov(qt, st, idf) for st in sents), default=0.0)
        # how tightly the question's terms cluster: best coverage within two adjacent sentences
        pair = max((_cov(qt, sents[i] | sents[i + 1], idf) for i in range(len(sents) - 1)), default=best)
        best_idf = max((sum(qidf[t] for t in qt if t in st) for st in sents), default=0.0)
        b, d = float(sb[pid]), float(sd[pid])
        cov = _cov(qt, pi["set"], idf)
        rows.append([
            b / bmax, b / 30.0, d, d - dmax,
            math.log1p(rank_b.get(int(pid), 200)), math.log1p(rank_d.get(int(pid), 200)),
            cov,
            len(qbi & pi["big"]) / max(len(qbi), 1),
            _cov(qt, pi["title"], idf),
            best, math.log(pi["len"] + 1) / 6.0, (b / bmax) * d,
            pair, best_idf / 20.0, len(qt) / 10.0, 0.0,
        ])
    X = np.asarray(rows, dtype=np.float32)
    X[:, 15] = X[:, 6] - X[:, 6].max()   # coverage relative to the best candidate for this question
    return X


def sentence_features(q: str, cands: list[dict], qvec: np.ndarray, svecs: np.ndarray, idf) -> np.ndarray:
    qt = list(dict.fromkeys(tokens(q)))
    qbi = _bigrams(tokens(q))
    qcaps = set(w.lower() for w in _CAP.findall(q))
    kind = qtype(q)
    rows = []
    for c, sv in zip(cands, svecs):
        s = c["text"]
        st = tokens(s)
        caps = [w for w in _CAP.findall(s[1:]) if w.lower() not in qcaps]  # skip sentence-initial capital
        if kind == "time":
            tm = float(bool(_YEAR.search(s) or _MONTH.search(s)))
        elif kind == "number":
            tm = float(bool(_NUM.search(s)))
        elif kind in ("person", "place"):
            tm = float(bool(caps))
        else:
            tm = 0.5
        rows.append([
            _cov(qt, set(st), idf), len(qbi & _bigrams(st)) / max(len(qbi), 1), float(sv @ qvec),
            c["p_prob"], math.log1p(c["p_rank"]), c["s_pos"] / 10.0, math.log(len(st) + 1) / 4.0,
            tm, min(len(caps), 5) / 5.0, float(bool(_NUM.search(s))),
        ])
    return np.asarray(rows, dtype=np.float32)
