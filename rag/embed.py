"""Dense embeddings with WordLlama: a small static-embedding model that ships inside its pip package.

No GPU, no torch, no model download at runtime, which is what lets the demo run in a serverless function.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def model():
    import wordllama
    from wordllama import WordLlama

    # WordLlama 0.4 looks for its bundled tokenizer under the cache dir's "tokenizers/" folder, so pointing the
    # cache at the package itself makes it load fully offline.
    return WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)


def embed(texts: list[str], batch_size: int = 512) -> np.ndarray:
    out = [model().embed(texts[i:i + batch_size], norm=True) for i in range(0, len(texts), batch_size)]
    return np.vstack(out).astype(np.float32) if out else np.zeros((0, 256), np.float32)
