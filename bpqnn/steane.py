"""Steane [[7,1,3]] code utilities for a syndrome-decoding task (E12-A).

The Steane code is CSS and self-dual: the same ``[7,4,3]`` Hamming matrix ``H``
defines both the X- and Z-type stabilisers, so we model the **X-error channel**
only (the Z channel is identical by symmetry) and decode the logical X class.

Data error = bit vector ``e`` (7 qubits, X-type). Syndrome ``s = H e mod 2``
(3 bits). The logical value is ``L(e) = e . X_L mod 2`` with the weight-3
logical ``X_L = (1,1,1,0,0,0,0)`` (in the code's kernel of ``H`` of weight 3).

Noise model: each qubit carries an X error with probability ``p``; each of the
3 measured syndrome bits is flipped by a **measurement error** with probability
``q``. The *standard* decoder assumes ``q = 0`` (minimum-weight coset leader),
the *Bayes* decoder (exact, by enumeration over the 2**7 errors) accounts for
``q``; the QNN decoder is trained on the measured syndrome. The whole error
space has only 128 elements, so Bayes is exact and cheap.
"""

from __future__ import annotations

import numpy as np

H = np.array(
    [[0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1],
     [1, 0, 1, 0, 1, 0, 1]],
    dtype=np.uint8,
)
X_L = np.array([1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
N_QUBITS = 7
N_CHECKS = 3

ALL_ERRORS = np.array(
    [[(i >> b) & 1 for b in range(N_QUBITS)] for i in range(1 << N_QUBITS)],
    dtype=np.uint8,
)
ALL_SYNDROMES = (ALL_ERRORS @ H.T) % 2          # (128, 3)
ALL_LOGICALS = (ALL_ERRORS @ X_L) % 2            # (128,)
ALL_WEIGHTS = ALL_ERRORS.sum(axis=1)


def syndrome(e):
    return (np.asarray(e) @ H.T) % 2


def logical(e):
    return int((np.asarray(e) @ X_L) % 2)


def sample_data(p, size, rng):
    """(size, 7) X-error vectors, each qubit flipped with prob p."""
    return (rng.random((size, N_QUBITS)) < p).astype(np.uint8)


def sample_measurement(q, size, rng):
    """(size, 3) syndrome-measurement error masks, each bit flipped w.p. q."""
    return (rng.random((size, N_CHECKS)) < q).astype(np.uint8)


def error_weights(p):
    w = ALL_WEIGHTS
    return (p ** w) * ((1.0 - p) ** (N_QUBITS - w))


def standard_predict(p, s_meas):
    """Minimum-weight coset-leader decoder (assumes q = 0). Returns L_pred."""
    s_meas = np.asarray(s_meas, dtype=np.uint8)
    match = (ALL_SYNDROMES == s_meas).all(axis=1)
    w = np.where(match, ALL_WEIGHTS, 1 << 30)
    m = int(w.min())
    cand = np.where(w == m)[0]
    ls = ALL_LOGICALS[cand]
    return int(ls.mean() >= 0.5)


def bayes_predict(p, q, s_meas):
    """Exact P(L=1 | measured syndrome) over the data+measurement model."""
    s_meas = np.asarray(s_meas, dtype=np.uint8)
    pw = error_weights(p)
    d = (ALL_SYNDROMES != s_meas).sum(axis=1)          # Hamming(s_meas, H e)
    lik = ((1.0 - q) ** (N_CHECKS - d)) * (q ** d)
    post = pw * lik
    p1 = post[ALL_LOGICALS == 1].sum()
    p0 = post[ALL_LOGICALS == 0].sum()
    return 1 if p1 > p0 else 0


def random_syndrome_features(p, q, size, rng):
    """Return (features (size,3) uint8, logical errors (size,) int)."""
    e = sample_data(p, size, rng)
    s_true = (e @ H.T) % 2
    m = sample_measurement(q, size, rng)
    s_meas = (s_true ^ m) % 2
    L = (e @ X_L) % 2
    return s_meas.astype(np.uint8), L.astype(np.int64)
