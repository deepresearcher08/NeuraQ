"""Tests for the repetition-code decoder QNN (SPEC E10)."""

import json
import pathlib

import numpy as np
import pytest

from bpqnn.classify import QNNClassifier
from bpqnn.grad import finite_difference
from bpqnn.init import random_init, zero_init
from bpqnn.qec import (
    logreg_accuracy,
    majority_accuracy,
    majority_predict,
    repetition_samples,
    train_logreg,
)

_RESULT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "results" / "e10_decoder.json"
)


def test_repetition_samples_label_and_codeword():
    rng = np.random.default_rng(0)
    d, p, size = 5, 0.2, 4000
    X, y = repetition_samples(d, p, size, rng)
    assert X.shape == (size, d) and y.shape == (size,)
    assert set(np.unique(X)).issubset({0.0, 1.0})
    assert set(np.unique(y)).issubset({-1.0, 1.0})
    # label +1 <=> logical 0 <=> codeword all zeros
    b = ((1.0 - y) / 2.0).astype(int)
    codeword = np.repeat(b[:, None], d, axis=1)
    flips = (X.astype(int) ^ codeword) == 1
    # empirical flip rate ~ p
    assert abs(flips.mean() - p) < 0.02


def test_zero_noise_no_flips():
    rng = np.random.default_rng(1)
    X, y = repetition_samples(7, 0.0, 100, rng)
    b = ((1.0 - y) / 2.0).astype(int)
    assert np.array_equal(X, np.repeat(b[:, None], 7, axis=1))


def test_majority_accuracy_endpoints():
    assert majority_accuracy(5, 0.0) == pytest.approx(1.0)
    assert majority_accuracy(5, 0.5) == pytest.approx(0.5)
    assert majority_accuracy(3, 0.1) == pytest.approx(
        0.9**3 + 3 * 0.1 * 0.9**2
    )


def test_majority_accuracy_matches_empirical():
    rng = np.random.default_rng(2)
    d, p = 7, 0.15
    X, y = repetition_samples(d, p, 200000, rng)
    acc = float(np.mean(majority_predict(X) == y))
    assert abs(acc - majority_accuracy(d, p)) < 5e-3


def test_logreg_matches_majority():
    rng = np.random.default_rng(3)
    d, p = 7, 0.15
    Xtr, ytr = repetition_samples(d, p, 4000, rng)
    Xte, yte = repetition_samples(d, p, 20000, rng)
    w = train_logreg(Xtr, ytr)
    assert abs(logreg_accuracy(w, Xte, yte) - majority_accuracy(d, p)) < 5e-3


def test_zero_init_local_readout_is_majority_optimal():
    rng = np.random.default_rng(4)
    d, p, M = 5, 0.2, 11
    X, y = repetition_samples(d, p, 4000, rng)
    clf = QNNClassifier(d, M, X, y, readout="local", qubits=list(range(d)))
    acc = clf.accuracy(zero_init(M, d))
    assert abs(acc - majority_accuracy(d, p)) < 5e-3


def test_decoder_gradient_matches_finite_difference():
    rng = np.random.default_rng(5)
    d, M = 3, 7
    X, y = repetition_samples(d, 0.15, 32, rng)
    clf = QNNClassifier(d, M, X, y, readout="local", qubits=[0, 1, 2])
    theta = random_init(M, d, rng)
    g = clf.grad(theta)
    gfd = finite_difference(theta, clf.loss)
    assert np.max(np.abs(g - gfd)) < 1e-6


@pytest.mark.parametrize("readout,qubits", [("local", [0, 1, 2]), ("global", None)])
def test_adjoint_gradient_matches_parameter_shift(readout, qubits):
    rng = np.random.default_rng(6)
    d, M = 3, 7
    X, y = repetition_samples(d, 0.15, 32, rng)
    clf = QNNClassifier(d, M, X, y, readout=readout, qubits=qubits)
    theta = random_init(M, d, rng)
    g_ps = clf.grad(theta)
    g_ad = clf.grad_adjoint(theta)
    assert np.max(np.abs(g_ps - g_ad)) < 1e-9


@pytest.mark.skipif(not _RESULT_PATH.exists(), reason="E10 results not generated")
def test_e10_local_decoder_beats_global():
    out = json.load(open(_RESULT_PATH))
    for d in out["config"]["d"]:
        arms = out["cases"][f"d{d}"]["arms"]
        for init in ("random", "identity"):
            assert (arms[f"local_{init}"]["median_test_acc"]
                    > arms[f"global_{init}"]["median_test_acc"])
