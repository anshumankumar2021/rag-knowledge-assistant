import numpy as np

from rag.ranker import ListRanker, Logistic


def test_listranker_learns_to_put_positive_first():
    rng = np.random.default_rng(0)
    groups = []
    for _ in range(300):
        X = rng.standard_normal((10, 3)).astype(np.float32)
        y = np.zeros(10, np.float32)
        y[int(np.argmax(X[:, 0] - X[:, 1]))] = 1   # the "relevant" item follows a simple rule
        groups.append((X, y))
    m = ListRanker(3, hidden=8).fit(groups, epochs=15, lr=0.01)
    acc = np.mean([np.argmax(m.score(X)) == np.argmax(y) for X, y in groups])
    assert acc > 0.9
    m2 = ListRanker.from_dict(m.to_dict())
    assert np.allclose(m2.score(groups[0][0]), m.score(groups[0][0]))


def test_logistic_separates_and_round_trips():
    X = np.vstack([np.random.default_rng(1).normal(2, 1, (200, 2)), np.random.default_rng(2).normal(-2, 1, (200, 2))])
    y = np.array([1.0] * 200 + [0.0] * 200)
    m = Logistic().fit(X, y)
    assert np.mean((m.prob(X) > 0.5) == y) > 0.95
    assert np.allclose(Logistic.from_dict(m.to_dict()).prob(X), m.prob(X))
