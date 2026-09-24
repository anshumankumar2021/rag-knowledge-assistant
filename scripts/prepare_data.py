"""Download SQuAD v1.1 and write the knowledge base + question splits to data/kb/."""
from rag.corpus import build, save

if __name__ == "__main__":
    kb = build()
    save(kb)
    print(f"passages: {len(kb['passages']):,}")
    for k, v in kb["splits"].items():
        print(f"{k:14s} {len(v):>7,} questions")
