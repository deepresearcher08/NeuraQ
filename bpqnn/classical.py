"""Classical baselines for the QNN classifier (E21).

The QNN in ``bpqnn.classify`` has ``M * n`` trainable angles (42 for E6/E17).
The fair comparison is a classical model with a comparable parameter count, so
the main baseline is a one-hidden-layer tanh MLP with 41 parameters.  It is
trained with the *same* protocol as the QNN: full-batch Adam, lr 0.05, 200
steps, MSE loss and the same eight seeds.  Wider/non-parametric classical
models (logistic regression, a larger MLP, an RBF SVM, k-NN and a random
forest) are reported alongside as reference points.

These are 2-D toy tasks with 16 training points.  Classical models have a much
stronger inductive bias here, so this is a sanity baseline, not evidence for
or against quantum advantage.
"""

from __future__ import annotations

import time

import numpy as np


def mlp_param_count(n_features: int, hidden: int, n_out: int = 1) -> int:
    """Trainable parameters of a one-hidden-layer MLP (tanh hidden)."""
    return int(n_features * hidden + hidden + hidden * n_out + n_out)


class NumpyMLP:
    """One-hidden-layer tanh MLP trained with full-batch Adam and MSE.

    ``predict`` returns the (unbounded) real-valued output; accuracy uses its
    sign.  The initialisation, optimiser, learning rate and step count mirror
    ``bpqnn.classify.adam_classify`` so the comparison with the QNN is matched.
    """

    def __init__(self, n_features, hidden=10, seed=0, lr=0.05, steps=200,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.n_features = int(n_features)
        self.hidden = int(hidden)
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 1.0 / np.sqrt(self.n_features),
                             size=(self.n_features, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0.0, 1.0 / np.sqrt(self.hidden),
                             size=(self.hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = float(lr)
        self.steps = int(steps)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.history: list[float] = []

    @property
    def n_params(self) -> int:
        return mlp_param_count(self.n_features, self.hidden)

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _forward(self, X):
        A = np.asarray(X, dtype=np.float64) @ self.W1 + self.b1
        Z = np.tanh(A)
        out = (Z @ self.W2 + self.b2)[:, 0]
        return out, Z

    def predict(self, X) -> np.ndarray:
        return self._forward(X)[0]

    def loss(self, X, y) -> float:
        pred = self.predict(X)
        return float(np.mean((np.asarray(y, dtype=np.float64) - pred) ** 2))

    def accuracy(self, X, y) -> float:
        pred = self.predict(X)
        return float(np.mean(np.sign(pred) == np.asarray(y, dtype=np.float64)))

    def _grads(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        S = X.shape[0]
        out, Z = self._forward(X)
        dout = -2.0 * (y - out) / S
        gW2 = (Z.T @ dout)[:, None]
        gb2 = np.array([dout.sum()])
        dZ = np.outer(dout, self.W2[:, 0]) * (1.0 - Z ** 2)
        gW1 = X.T @ dZ
        gb1 = dZ.sum(axis=0)
        return [gW1, gb1, gW2, gb2]

    def fit(self, X, y):
        params = self._params()
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for t in range(1, self.steps + 1):
            self.history.append(self.loss(X, y))
            grads = self._grads(X, y)
            for p, g, mi, vi in zip(params, grads, m, v):
                mi[...] = self.beta1 * mi + (1.0 - self.beta1) * g
                vi[...] = self.beta2 * vi + (1.0 - self.beta2) * g * g
                mhat = mi / (1.0 - self.beta1 ** t)
                vhat = vi / (1.0 - self.beta2 ** t)
                p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        self.history.append(self.loss(X, y))
        return self

    def predict_on(self, X) -> np.ndarray:
        return self.predict(X)


def evaluate_numpy_mlp(Xtr, ytr, Xte, yte, hidden=10, seeds=range(8),
                       lr=0.05, steps=200):
    """Fit ``NumpyMLP`` once per seed and return median/sample metrics."""
    n_features = int(np.asarray(Xtr).shape[1])
    train_acc, test_acc, losses = [], [], []
    t0 = time.perf_counter()
    for seed in seeds:
        model = NumpyMLP(n_features, hidden=hidden, seed=seed,
                         lr=lr, steps=steps).fit(Xtr, ytr)
        train_acc.append(model.accuracy(Xtr, ytr))
        test_acc.append(model.accuracy(Xte, yte))
        losses.append(model.loss(Xtr, ytr))
    return {
        "model": f"mlp_hidden_{hidden}",
        "n_params": mlp_param_count(n_features, hidden),
        "median_train_acc": float(np.median(train_acc)),
        "median_test_acc": float(np.median(test_acc)),
        "train_acc": [float(x) for x in train_acc],
        "test_acc": [float(x) for x in test_acc],
        "final_loss": [float(x) for x in losses],
        "n_seeds": len(train_acc),
        "fit_seconds": float(time.perf_counter() - t0),
    }


def evaluate_sklearn(make_model, Xtr, ytr, Xte, yte, seeds=(0,)):
    """Fit sklearn-style estimators and return median/sample metrics.

    ``make_model(seed)`` must return an object with ``fit``/``predict``.
    Labels are ``{-1, +1}``; MSE is computed in that coding.
    """
    ytr = np.asarray(ytr, dtype=np.float64)
    yte = np.asarray(yte, dtype=np.float64)
    train_acc, test_acc, losses = [], [], []
    t0 = time.perf_counter()
    for seed in seeds:
        model = make_model(seed).fit(Xtr, ytr)
        ptr = np.asarray(model.predict(Xtr), dtype=np.float64)
        pte = np.asarray(model.predict(Xte), dtype=np.float64)
        train_acc.append(float(np.mean(np.sign(ptr) == ytr)))
        test_acc.append(float(np.mean(np.sign(pte) == yte)))
        losses.append(float(np.mean((ytr - ptr) ** 2)))
    return {
        "median_train_acc": float(np.median(train_acc)),
        "median_test_acc": float(np.median(test_acc)),
        "train_acc": [float(x) for x in train_acc],
        "test_acc": [float(x) for x in test_acc],
        "final_loss": [float(x) for x in losses],
        "n_seeds": len(train_acc),
        "fit_seconds": float(time.perf_counter() - t0),
    }
