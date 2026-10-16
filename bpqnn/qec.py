"""Repetition-code "learn to decode" task for the QNN (SPEC E10).

Combine-with-another-track stretch: a quantum neural network that *learns to
decode* a bit-flip repetition code. A logical bit ``b`` is encoded as ``b``
repeated ``d`` times; each physical bit is flipped independently with
probability ``p``; the received ``d``-bit word (computational-basis readout) is
the classifier input, and the target is the logical bit.

The Bayes-optimal decoder for the repetition code is **majority vote**, which is
a *linear* function of the received bits, so a logistic classifier is already
optimal. This is therefore a **mechanism / transfer** study of the barren-
plateau machinery on a decoding task, **not** a competitive decoder and **not**
a quantum-advantage claim (SPEC section 12). The point of interest is whether
the local-vs-global readout and init story (E2/E6) transfers to decoding, and
how the loss-gradient variance scales with code distance ``d``.

Label convention: ``y = 1 - 2*b`` (``+1`` = logical 0). The sign is chosen so
the natural local observable ``mean_q <Z_q>`` of the encoded received word is
**aligned** with ``y``: a received word with more 0s (logical 0) gives
``mean <Z> > 0``.
"""

from __future__ import annotations

from math import comb

import numpy as np


def repetition_samples(d: int, p: float, size: int, rng) -> tuple:
    """Sample ``(received word, label)`` pairs for a bit-flip repetition code.

    Returns ``X`` of shape ``(size, d)`` with entries in ``{0,1}`` and ``y`` of
    shape ``(size,)`` with entries in ``{-1, +1}`` (``+1`` = logical 0).
    """
    d = int(d)
    if d < 1:
        raise ValueError("d must be >= 1")
    b = rng.integers(0, 2, size=size)
    codeword = np.repeat(b[:, None], d, axis=1)
    flips = (rng.random((size, d)) < p).astype(np.int64)
    X = (codeword ^ flips).astype(np.float64)
    y = (1.0 - 2.0 * b).astype(np.float64)
    return X, y


def majority_accuracy(d: int, p: float) -> float:
    """Analytic Bayes-optimal accuracy of the repetition decoder (odd ``d``).

    P(fewer than ``d/2`` physical flips) = sum_{k=0}^{floor(d/2)} C(d,k)
    p^k (1-p)^{d-k}.
    """
    d = int(d)
    if d % 2 == 0:
        raise ValueError(f"repetition decoder requires odd d, got {d}")
    half = d // 2
    return float(sum(comb(d, k) * p**k * (1.0 - p) ** (d - k)
                     for k in range(half + 1)))


def majority_predict(X) -> np.ndarray:
    """Majority-vote logical bit as ``{-1, +1}`` (``+1`` = logical 0)."""
    X = np.asarray(X, dtype=np.float64)
    frac_ones = X.mean(axis=1)
    b = (frac_ones > 0.5).astype(np.int64)
    return (1.0 - 2.0 * b).astype(np.float64)


def train_logreg(X, y, steps: int = 1000, lr: float = 0.5, l2: float = 1e-3):
    """Minimal L2-regularised logistic regression on ``{0,1}`` features.

    Returns the weight vector (with a trailing bias entry).
    """
    X = np.asarray(X, dtype=np.float64)
    y01 = (np.asarray(y, dtype=np.float64) > 0).astype(np.float64)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    N = Xb.shape[0]
    for _ in range(steps):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (p - y01) / N + l2 * w
        w -= lr * g
    return w


def logreg_accuracy(w, X, y) -> float:
    X = np.asarray(X, dtype=np.float64)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    pred = np.where(Xb @ np.asarray(w) > 0, 1.0, -1.0)
    return float(np.mean(pred == np.asarray(y, dtype=np.float64)))
