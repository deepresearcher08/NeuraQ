import json
import pathlib

import numpy as np
import pytest

from bpqnn.classify import (
    QNNClassifier,
    additive_grid,
    encode_states,
    parity_expectation,
    xor_grid,
    z_expectation,
)
from bpqnn.grad import finite_difference
from bpqnn.init import random_init

_RESULT_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "results" / "e6_classify.json"
)


def test_additive_and_xor_labels():
    Xa, ya = additive_grid()
    assert Xa.shape == (16, 2) and ya.shape == (16,)
    assert set(np.unique(ya)) == {-1.0, 1.0}
    assert ya[(Xa[:, 0] == 1.0) & (Xa[:, 1] == 1.0)][0] == 1.0
    assert ya[(Xa[:, 0] == 0.0) & (Xa[:, 1] == 0.0)][0] == -1.0

    Xx, yx = xor_grid()
    assert set(np.unique(yx)) == {-1.0, 1.0}
    assert yx[(Xx[:, 0] == 1.0) & (Xx[:, 1] == 1.0)][0] == -1.0
    assert yx[(Xx[:, 0] == 1.0) & (Xx[:, 1] == 0.0)][0] == 1.0


def test_parity_and_z_expectations():
    # |0...0> and |1...1>
    zero = np.zeros((1, 4), dtype=np.complex128)
    zero[0, 0] = 1.0
    one = np.zeros((1, 4), dtype=np.complex128)
    one[0, 3] = 1.0  # |11>
    assert parity_expectation(zero, 2)[0] == pytest.approx(1.0)
    assert parity_expectation(one, 2)[0] == pytest.approx(1.0)
    ten = np.zeros((1, 4), dtype=np.complex128)
    ten[0, 1] = 1.0  # |10>
    assert parity_expectation(ten, 2)[0] == pytest.approx(-1.0)
    # Bell state (|00> + |11>)/sqrt2 -> Z0 Z1 = +1, Z0 = Z1 = 0
    bell = np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.complex128) / np.sqrt(2)
    assert parity_expectation(bell, 2)[0] == pytest.approx(1.0)
    assert z_expectation(bell, 0, 2)[0] == pytest.approx(0.0, abs=1e-12)


def test_encode_feature_map():
    psi = encode_states(np.array([[1.0, 0.0]]), 2)
    assert z_expectation(psi, 0, 2)[0] == pytest.approx(-1.0, abs=1e-9)
    assert z_expectation(psi, 1, 2)[0] == pytest.approx(1.0, abs=1e-9)
    assert parity_expectation(psi, 2)[0] == pytest.approx(-1.0, abs=1e-9)


def test_classifier_gradient_matches_finite_difference():
    X, y = additive_grid()
    clf = QNNClassifier(3, 3, X, y, readout="local", qubits=[0, 1])
    rng = np.random.default_rng(0)
    theta = random_init(3, 3, rng)
    g = clf.grad(theta)
    gfd = finite_difference(theta, clf.loss)
    assert np.max(np.abs(g - gfd)) < 1e-6
    # accuracy is in [0, 1]; loss is non-negative
    assert 0.0 <= clf.accuracy(theta) <= 1.0
    assert clf.loss(theta) >= 0.0


def test_global_readout_gradient_matches_finite_difference():
    X, y = xor_grid()
    clf = QNNClassifier(3, 3, X, y, readout="global")
    rng = np.random.default_rng(1)
    theta = random_init(3, 3, rng)
    gfd = finite_difference(theta, clf.loss)
    assert np.max(np.abs(clf.grad(theta) - gfd)) < 1e-6


@pytest.mark.skipif(not _RESULT_PATH.exists(), reason="E6 results not generated")
def test_e6_local_cost_trains_better_than_global():
    out = json.load(open(_RESULT_PATH))
    assert (out["init_signals"]["local"]["grad_var"]
            > out["init_signals"]["global"]["grad_var"])
    arms = out["arms"]
    for init in ("random", "identity"):
        assert (arms[f"local_{init}"]["median_acc"]
                > arms[f"global_{init}"]["median_acc"])
