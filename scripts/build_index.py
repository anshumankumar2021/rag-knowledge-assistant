"""Build the BM25 index and passage embeddings into data/index/."""
import time

import numpy as np

from rag.bm25 import BM25
from rag.corpus import load_passages
from rag.embed import embed
from rag.index import BM25_PATH, EMB_PATH

if __name__ == "__main__":
    ps = load_passages()
    t = time.time()
    bm = BM25().fit([f"{p.title} {p.text}" for p in ps])
    print(f"bm25: {len(bm.vocab):,} terms, {len(bm.doc_ids):,} postings in {time.time() - t:.1f}s")
    t = time.time()
    E = embed([f"{p.title}. {p.text}" for p in ps])
    print(f"embeddings: {E.shape} in {time.time() - t:.1f}s")
    EMB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # two files, each small enough for GitHub's web uploader
    np.savez_compressed(EMB_PATH, emb=E.astype(np.float16))
    np.savez_compressed(BM25_PATH, **bm.to_arrays())
    for p in (EMB_PATH, BM25_PATH):
        print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
