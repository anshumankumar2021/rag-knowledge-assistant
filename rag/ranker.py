"""A tiny listwise ranking model in numpy: standardize -> 1 hidden layer (tanh) -> score.

Trained with softmax cross-entropy over each query's candidate list (ListNet-style), so it learns to put the
right item first rather than to classify items in isolation. Inference is two small matrix multiplies.
"""
from __future__ import annotations

import json

import numpy as np


class ListRanker:
    def __init__(self, n_features: int, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.mu = np.zeros(n_features, np.float32)
        self.sd = np.ones(n_features, np.float32)
        self.W1 = (rng.standard_normal((n_features, hidden)) * 0.3).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.w2 = (rng.standard_normal(hidden) * 0.3).astype(np.float32)
        self.b2 = np.float32(0)

    def score(self, X: np.ndarray) -> np.ndarray:
        H = np.tanh(((X - self.mu) / self.sd) @ self.W1 + self.b1)
        return H @ self.w2 + self.b2

    def fit(self, groups: list[tuple[np.ndarray, np.ndarray]], epochs: int = 30, lr: float = 0.01,
            l2: float = 1e-4, seed: int = 0, verbose: bool = False) -> "ListRanker":
        """groups: list of (features [n_i, f], target [n_i] with >=1 positive)."""
        allX = np.vstack([g[0] for g in groups])
        self.mu, self.sd = allX.mean(0), allX.std(0) + 1e-6
        params = [self.W1, self.b1, self.w2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        rng = np.random.default_rng(seed)
        step = 0
        for ep in range(epochs):
            order = rng.permutation(len(groups))
            tot = 0.0
            for gi in order:
                X, y = groups[gi]
                Z = (X - self.mu) / self.sd
                A = Z @ self.W1 + self.b1
                H = np.tanh(A)
                s = H @ self.w2
                p = np.exp(s - s.max())
                p /= p.sum()
                t = y / y.sum()
                tot += -float(np.sum(t * np.log(p + 1e-12)))
                ds = p - t                                 # d loss / d scores
                gw2 = H.T @ ds + l2 * self.w2
                dA = np.outer(ds, self.w2) * (1 - H ** 2)
                gW1 = Z.T @ dA + l2 * self.W1
                gb1 = dA.sum(0)
                step += 1
                for i, g in enumerate((gW1, gb1, gw2)):   # Adam
                    m[i] = 0.9 * m[i] + 0.1 * g
                    v[i] = 0.999 * v[i] + 0.001 * g * g
                    params[i] -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
            if verbose:
                print(f"  epoch {ep + 1:2d} loss {tot / len(groups):.4f}")
        return self

    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist() for k in ("mu", "sd", "W1", "b1", "w2")} | {"b2": float(self.b2)}

    @classmethod
    def from_dict(cls, d: dict) -> "ListRanker":
        m = cls(len(d["mu"]), len(d["b1"]))
        for k in ("mu", "sd", "W1", "b1", "w2"):
            setattr(m, k, np.asarray(d[k], np.float32))
        m.b2 = np.float32(d["b2"])
        return m


class Logistic:
    """Binary logistic regression (standardized inputs) used for the answer/abstain decision."""

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 3000, lr: float = 0.1):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Z @ self.w + self.b)))
            self.w -= lr * (Z.T @ (p - y) / len(y) + l2 * self.w)
            self.b -= lr * float(np.mean(p - y))
        return self

    def prob(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(((X - self.mu) / self.sd) @ self.w + self.b)))

    def to_dict(self) -> dict:
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "w": self.w.tolist(), "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "Logistic":
        m = cls()
        m.mu, m.sd, m.w = np.asarray(d["mu"]), np.asarray(d["sd"]), np.asarray(d["w"])
        m.b = float(d["b"])
        return m


def save_models(path, **models) -> None:
    path.write_text(json.dumps({k: v.to_dict() for k, v in models.items()}))
