"""Train the passage re-ranker, answer-sentence selector and abstain classifier.

Uses only TRAIN questions (from indexed train articles) and HELDOUT_CALIB questions (from non-indexed articles).
The SQuAD dev questions used for reporting are never touched here.
"""
from __future__ import annotations

import random
import time

import numpy as np

from rag.corpus import load_questions
from rag.embed import embed
from rag.features import SENTENCE_FEATURES, sentence_features
from rag.index import get_index
from rag.pipeline import MODELS_PATH, abstain_features, candidates, sentence_candidates
from rag.ranker import ListRanker, Logistic, save_models

N_PASSAGE, N_SENTENCE, N_VALID = 15_000, 12_000, 2_000


def contains_answer(text: str, answers: list[str]) -> bool:
    return any(a in text for a in answers)


def main():
    idx = get_index()
    train = load_questions("train")
    random.Random(1).shuffle(train)
    p_set = train[:N_PASSAGE]
    s_set = train[N_PASSAGE:N_PASSAGE + N_SENTENCE]
    v_set = train[N_PASSAGE + N_SENTENCE:N_PASSAGE + N_SENTENCE + N_VALID]
    a_pos = train[N_PASSAGE + N_SENTENCE + N_VALID:]

    # 1) passage re-ranker
    t = time.time()
    groups, miss = [], 0
    for q in p_set:
        pids, X, *_ = candidates(idx, q["q"])
        y = (pids == q["pid"]).astype(np.float32)
        if y.sum() == 0:
            miss += 1
            continue
        groups.append((X, y))
    print(f"passage groups: {len(groups):,} (gold not in candidates for {miss:,}) in {time.time() - t:.0f}s")
    pr = ListRanker(groups[0][0].shape[1], hidden=32).fit(groups, epochs=20, lr=0.003)

    def top1(qs, ranker):
        hit = 0
        for q in qs:
            pids, X, *_ = candidates(idx, q["q"])
            hit += int(pids[int(np.argmax(ranker.score(X)))] == q["pid"])
        return hit / len(qs)

    print(f"validation R@1 (train-article questions, not used for fitting): {top1(v_set, pr):.3f}")

    def rerank(q):
        pids, X, qvec, _, _ = candidates(idx, q)
        s = pr.score(X)
        o = np.argsort(-s)
        p = np.exp(s[o] - s[o].max())
        return pids[o], X[o], p / p.sum(), qvec

    # 2) answer-sentence selector
    t = time.time()
    sgroups = []
    for q in s_set:
        pids, _, probs, qvec = rerank(q["q"])
        cands = sentence_candidates(idx, pids, probs)
        y = np.array([contains_answer(c["text"], q["answers"]) for c in cands], np.float32)
        if y.sum() == 0:
            continue
        svecs = embed([c["text"] for c in cands])
        sgroups.append((sentence_features(q["q"], cands, qvec, svecs, idx.bm25.term_idf), y))
    print(f"sentence groups: {len(sgroups):,} in {time.time() - t:.0f}s ({len(SENTENCE_FEATURES)} features)")
    sr = ListRanker(len(SENTENCE_FEATURES), hidden=32, seed=1).fit(sgroups, epochs=20, lr=0.003)

    # 3) abstain classifier: in-KB questions vs questions about articles that are not in the KB
    def afeats(q):
        pids, X, probs, qvec = rerank(q)
        cands = sentence_candidates(idx, pids, probs)
        ss = sr.score(sentence_features(q, cands, qvec, embed([c["text"] for c in cands]), idx.bm25.term_idf))
        sp = np.exp(ss - ss.max())
        sp /= sp.sum()
        return abstain_features(X[0], float(probs[0]), float(sp.max()))

    neg = load_questions("heldout_calib")
    pos = a_pos[:len(neg)]
    XA = np.array([afeats(q["q"]) for q in pos + neg])
    yA = np.array([1.0] * len(pos) + [0.0] * len(neg))
    ab = Logistic().fit(XA, yA)
    p = ab.prob(XA)
    best_t, best_bal = 0.5, 0.0
    for thr in np.linspace(0.05, 0.95, 91):
        bal = 0.5 * (np.mean(p[yA == 1] >= thr) + np.mean(p[yA == 0] < thr))
        if bal > best_bal:
            best_t, best_bal = float(thr), float(bal)
    print(f"abstain threshold {best_t:.2f} (calibration balanced accuracy {best_bal:.3f})")

    save_models(MODELS_PATH, passage=pr, sentence=sr, abstain=ab)
    import json
    d = json.loads(MODELS_PATH.read_text())
    d["threshold"] = best_t
    MODELS_PATH.write_text(json.dumps(d))
    print(f"wrote {MODELS_PATH}")


if __name__ == "__main__":
    main()
