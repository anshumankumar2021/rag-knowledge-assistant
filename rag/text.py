"""Tokenization and sentence splitting shared by indexing, ranking and answering."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with would you your yours yourself yourselves also many much one s t
""".split())


def _stem(w: str) -> str:
    """Tiny suffix stripper: enough to match plurals and simple verb forms (e.g. 'founded' ~ 'found')."""
    if len(w) <= 3 or w.isdigit():
        return w
    for suf in ("ational", "ization", "ingly", "ments", "ings", "ness", "ies", "ing", "ed", "es", "ly", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            return w + ("y" if suf == "ies" else "")
    return w


def tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    ws = _WORD.findall(text.lower())
    return [_stem(w) for w in ws if keep_stopwords or w not in STOPWORDS]


def sentences(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences, so answers can be quoted verbatim."""
    spans, start = [], 0
    for m in _SENT.finditer(text):
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]
