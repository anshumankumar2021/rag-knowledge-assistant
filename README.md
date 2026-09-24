# Knowledge Base Assistant (RAG)

[![ci](https://github.com/anshumankumar2021/rag-knowledge-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/anshumankumar2021/rag-knowledge-assistant/actions/workflows/ci.yml)

**Live demo: [rag-knowledge-assistant-beta.vercel.app](https://rag-knowledge-assistant-beta.vercel.app)**

A retrieval-augmented question-answering assistant over a knowledge base of 19,391 passages from 450 Wikipedia
articles. It combines keyword and embedding search, re-ranks the candidates with a small learned model, and answers
by quoting the most relevant sentence with a citation. When the knowledge base doesn't contain the answer, it
declines instead of guessing.

Everything runs on CPU with no external API: the embedding model ships inside its pip package, and the whole
pipeline fits in one serverless function.

```
question ─┬─► BM25 (67K-term inverted index) ─── top 50 ─┐
          └─► dense embeddings (WordLlama 256-d) ─ top 50 ─┴─► learned re-ranker ─► answer-sentence selector ─► answer + citation
                                                                                                   └─► confidence model ─► or decline
```

## Results

Evaluated on the **10,570 SQuAD v1.1 dev questions**, which are never used for training or tuning. Each
question has a known source passage, so retrieval is measured exactly.

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| **Hybrid + learned re-ranker** | 76.8% | 91.1% | 94.0% | 0.830 | 6.9 ms | 13.2 ms |
| BM25 (keyword) | 71.5% | 88.0% | 91.4% | 0.785 | 0.1 ms | 0.2 ms |
| Hybrid (reciprocal rank fusion) | 58.7% | 80.8% | 88.2% | 0.683 | 0.8 ms | 1.0 ms |
| Dense embeddings (WordLlama) | 43.3% | 66.0% | 74.7% | 0.531 | 0.6 ms | 0.7 ms |

End to end, with the re-ranker:

| | |
|---|---|
| In-KB questions answered | **81.8%** (the rest are declined as low confidence) |
| Answers whose quoted sentence contains SQuAD's answer | **75.7%** |
| Out-of-KB questions correctly declined | **83.3%** of 3,660 questions about 20 articles left out of the index |
| Answers that are verbatim quotes from the cited passage | 100% by construction (tested) |

Latency is per query on a 2-vCPU machine. The re-ranked mode includes feature extraction and sentence embedding.

### What the numbers say

- **Keyword search beats embeddings here.** SQuAD questions reuse a lot of the passage's wording, so BM25 is strong
  and a small static embedding model is weak on its own.
- **Naively blending them hurt.** Reciprocal rank fusion dropped Recall@1 from 71.5% to 58.7%, although it
  puts the right passage in the combined top 100 for about 99% of questions.
- **The learned re-ranker is what makes the combination pay off:** 76.8% Recall@1, +5.3 points over BM25 and
  +18.1 over the naive hybrid.
- **Where the wrong answers come from:** in a 3,000-question sample, 43% of wrong answers cite a passage that does
  contain the answer but quote the wrong sentence; the other 57% cite the wrong passage. A neural cross-encoder
  re-ranker and a neural reader (or an LLM over the retrieved passages) would target those two in turn.

## How it works

| Stage | Detail |
|---|---|
| Knowledge base | SQuAD v1.1: all 48 dev articles plus 402 train articles. 40 train articles are held out of the index to test abstention. |
| BM25 | Own implementation over flat numpy arrays (no scipy), light stemming, stopwords. 0.1 ms per query. |
| Dense | [WordLlama](https://github.com/dleemiller/WordLlama) 256-d static embeddings, cosine similarity over a float16 matrix. |
| Re-ranker | 16 features per candidate: both scores and ranks, idf-weighted term coverage, bigram overlap, title overlap, best one- and two-sentence coverage. A one-hidden-layer network trained **listwise** (softmax cross-entropy over each question's candidate list, ListNet-style) on 15,000 training questions. Pure numpy. |
| Answer selector | Same model type over sentences from the top 3 passages, with answer-type cues (a "when" question favours sentences with dates, "how many" favours numbers). Trained on 12,000 questions. |
| Abstention | Logistic regression on the top result's evidence, trained on in-KB questions vs. questions about held-out articles. Its threshold is set on a calibration split; results above are on a separate test split. |

Splits are by article, so no passage or question leaks between training and evaluation.

## Run it

```bash
pip install -r requirements-dev.txt
pytest -q                          # unit and end-to-end tests against the committed index
python -m scripts.dev_server       # demo at http://localhost:3002
```

Rebuild everything from scratch (about 10 minutes on a laptop CPU):

```bash
python -m scripts.prepare_data     # downloads SQuAD v1.1 from GitHub, writes data/kb/
python -m scripts.build_index      # BM25 + embeddings -> data/index/{bm25,embeddings}.npz
python -m scripts.train            # re-ranker, answer selector, abstain model -> data/index/models.json
python -m scripts.evaluate         # results/results.json
```

CI runs the tests, then rebuilds the index, retrains every model, re-scores the full test set, and fails if the
reproduced numbers drift from the committed ones.

## Layout

```
rag/text.py          tokenizer, stemmer, sentence splitter
rag/bm25.py          BM25 inverted index
rag/embed.py         WordLlama embeddings (offline)
rag/index.py         first-stage retrieval: BM25, dense, hybrid (RRF)
rag/features.py      re-ranking and answer-selection features
rag/ranker.py        listwise ranker and logistic regression in numpy
rag/pipeline.py      retrieve -> re-rank -> answer or decline
api/ask.py           Vercel serverless endpoint
public/index.html    demo UI
scripts/             data prep, index build, training, evaluation, dev server
```

## Limitations

- Answers are single extracted sentences. That keeps them grounded, but they aren't phrased as direct replies.
- About 18% of answerable questions are declined. The threshold trades that against wrongly answering
  out-of-KB questions and can be moved in `data/index/models.json`.
- The knowledge base is general Wikipedia. Pointing it at another document set means rerunning the scripts with a
  new corpus and some labelled questions to retrain the ranker.

## Data

Knowledge base and questions from [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) (Rajpurkar et al., 2016),
CC BY-SA 4.0, with passages from Wikipedia (CC BY-SA). The processed files in `data/kb/` are redistributed under the
same license.
