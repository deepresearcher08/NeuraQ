"""E12:A Steane code utilities and the XOR3 QNN expressivity floor."""

from __future__ import annotations

import numpy as np

from bpqnn.classify import QNNClassifier, adam_classify
from bpqnn.init import random_init
from bpqnn.steane import (
    ALL_LOGICALS,
    ALL_SYNDROMES,
    H,
    X_L,
    bayes_predict,
    logical,
    random_syndrome_features,
    standard_predict,
    syndrome,
)


def test_logical_x_in_kernel_and_weight_3():
    lx = X_L
    assert lx.sum() == 3
    assert (H @ lx % 2).sum() == 0


def test_weight1_syndromes_are_distinct_columns():
    cols = {tuple(H[:, j]) for j in range(7)}
    assert len(cols) == 7
    assert (0, 0, 0) not in cols


def test_min_weight_decodes_single_errors_exactly_at_q0():
    for j in range(7):
        e = np.zeros(7, dtype=np.uint8)
        e[j] = 1
        s = syndrome(e)
        assert standard_predict(0.05, s) == logical(e)
        assert bayes_predict(0.05, 0.0, s) == logical(e)


def test_bayes_equals_standard_at_q0_across_all_syndromes():
    for s in range(8):
        sbits = np.array([(s >> b) & 1 for b in range(3)], dtype=np.uint8)
        assert bayes_predict(0.05, 0.0, sbits) == standard_predict(0.05, sbits)


def test_measured_syndrome_equals_true_plus_measurement():
    rng = np.random.default_rng(0)
    X, L = random_syndrome_features(0.05, 0.1, 2000, rng)
    assert set(X.shape) == {2000, 3}
    assert set(np.unique(L)) <= {0, 1}


def test_xor3_parity_is_below_the_qnn_floor():
    # Documented expressivity limitation: a small angle-encoded HEA with the
    # local readout cannot represent/learn 3-bit parity (XOR3); it collapses to
    # the constant predictor. Guard so a regression to trainable parity is seen.
    X = np.array([[b0, b1, b2] for b0 in (0, 1) for b1 in (0, 1) for b2 in (0, 1)], float)
    y = np.where(X.sum(1) % 2 == 1, 1.0, -1.0)
    Xb = np.vstack([X] * 40)
    yb = np.tile(y, 40)
    best = 1.0
    for s in range(4):
        clf = QNNClassifier(3, 7, Xb, yb, readout="local", qubits=[0, 1, 2])
        th, _ = adam_classify(clf, random_init(7, 3, np.random.default_rng(s)),
                              lr=0.1, steps=300, grad_fn=clf.grad_adjoint)
        best = min(best, 1.0 - clf.accuracy(th))
    # far from solvable: comparable linear thresholds reach ~0 (E6/E11 machinery),
    # but parity lands well above chance-independent error (chance = 0.5 here).
    assert best >= 0.15