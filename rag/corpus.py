"""Build the knowledge base and labelled question sets from SQuAD v1.1.

Splits (all by *article*, so no passage is shared between splits):
  KB      = all 48 dev articles + train articles except the held-out ones  -> indexed
  EVAL    = SQuAD dev questions (10,570). Gold passage is in the KB. Never used for training or tuning.
  TRAIN   = questions from KB train articles, used to fit the re-ranker and answer selector.
  HELDOUT = train articles NOT indexed. Their questions have no answer in the KB and should be abstained on.
            Split in two: calibration (tunes the abstain threshold) and test (reported).
"""
from __future__ import annotations

import gzip
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
KB_DIR = ROOT / "data" / "kb"
SQUAD_URL = "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/{}"
N_HELDOUT_ARTICLES = 40


@dataclass
class Passage:
    pid: int
    title: str
    text: str


def _load_raw(name: str) -> list[dict]:
    path = RAW / name
    if not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL.format(name), path)
    return json.loads(path.read_text())["data"]


def build(seed: int = 13) -> dict:
    train, dev = _load_raw("train-v1.1.json"), _load_raw("dev-v1.1.json")
    rng = random.Random(seed)
    heldout_titles = set(rng.sample(sorted(a["title"] for a in train), N_HELDOUT_ARTICLES))

    passages: list[dict] = []
    splits: dict[str, list[dict]] = {"eval": [], "train": [], "heldout": []}

    def add_article(article: dict, split: str, index: bool):
        title = article["title"].replace("_", " ")
        for para in article["paragraphs"]:
            pid = None
            if index:
                pid = len(passages)
                passages.append({"pid": pid, "title": title, "text": para["context"]})
            for qa in para["qas"]:
                answers = sorted({a["text"] for a in qa["answers"]})
                splits[split].append({"q": qa["question"], "pid": pid, "answers": answers, "title": title})

    for a in dev:
        add_article(a, "eval", index=True)
    for a in train:
        if a["title"] in heldout_titles:
            add_article(a, "heldout", index=False)
        else:
            add_article(a, "train", index=True)

    held = sorted({q["title"] for q in splits["heldout"]})
    rng.shuffle(held)
    calib_titles = set(held[: len(held) // 2])
    splits["heldout_calib"] = [q for q in splits["heldout"] if q["title"] in calib_titles]
    splits["heldout_test"] = [q for q in splits["heldout"] if q["title"] not in calib_titles]
    del splits["heldout"]
    return {"passages": passages, "splits": splits}


def save(kb: dict) -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    with gzip.open(KB_DIR / "passages.jsonl.gz", "wt", encoding="utf-8") as f:
        for p in kb["passages"]:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    for name, qs in kb["splits"].items():
        with gzip.open(KB_DIR / f"questions_{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")


def load_passages() -> list[Passage]:
    with gzip.open(KB_DIR / "passages.jsonl.gz", "rt", encoding="utf-8") as f:
        return [Passage(**json.loads(line)) for line in f]


def load_questions(split: str) -> list[dict]:
    with gzip.open(KB_DIR / f"questions_{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]
