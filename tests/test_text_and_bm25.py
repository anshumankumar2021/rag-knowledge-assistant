import numpy as np

from rag.bm25 import BM25
from rag.text import sentences, tokens


def test_tokens_drop_stopwords_and_stem():
    assert tokens("The Cats were running quickly") == ["cat", "runn", "quick"]
    assert "the" in tokens("the cat", keep_stopwords=True)


def test_sentence_spans_are_verbatim():
    text = "Paris is in France. It has the Eiffel Tower! Is it big? Yes."
    spans = sentences(text)
    assert [text[a:z] for a, z in spans] == ["Paris is in France.", "It has the Eiffel Tower!", "Is it big?", "Yes."]


def test_bm25_ranks_relevant_doc_first_and_round_trips():
    docs = ["the eiffel tower is in paris", "bananas are yellow fruit", "paris is the capital of france"]
    bm = BM25().fit(docs)
    s = bm.scores("eiffel tower location")
    assert int(np.argmax(s)) == 0
    bm2 = BM25.from_arrays(bm.to_arrays())
    assert np.allclose(bm2.scores("capital of france"), bm.scores("capital of france"))
    assert bm.scores("zzzz").sum() == 0
