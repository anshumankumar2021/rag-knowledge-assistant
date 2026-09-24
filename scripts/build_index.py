"""Build the BM25 index and passage embeddings into data/index/index.npz."""
import time

import numpy as np

from rag.bm25 import BM25
from rag.corpus import load_passages
from rag.embed import embed
from rag.index import INDEX_PATH

if __name__ == "__main__":
    ps = load_passages()
    t = time.time()
    bm = BM25().fit([f"{p.title} {p.text}" for p in ps])
    print(f"bm25: {len(bm.vocab):,} terms, {len(bm.doc_ids):,} postings in {time.time() - t:.1f}s")
    t = time.time()
    E = embed([f"{p.title}. {p.text}" for p in ps])
    print(f"embeddings: {E.shape} in {time.time() - t:.1f}s")
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX_PATH, emb=E.astype(np.float16), **bm.to_arrays())
    print(f"wrote {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
