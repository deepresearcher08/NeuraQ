"""E17 checks: the parity dataset, the held-out split, and out-of-sample scoring."""

import numpy as np
import pytest

from bpqnn.classify import (
    QNNClassifier,
    held_out_grid,
    make_dataset,
    train_test_datasets,
)


def test_parity_labels_are_checkerboard():
    X, y = make_dataset("parity")
    assert X.shape == (16, 2)
    assert y[(X[:, 0] == 1.0) & (X[:, 1] == 1.0)][0] == -1.0
    assert y[(X[:, 0] == 1.0) & (X[:, 1] == 0.0)][0] == 1.0


def test_additive_labels_still_linear():
    X, y = make_dataset("additive")
    assert y[(X[:, 0] == 0.0) & (X[:, 1] == 0.0)][0] == -1.0
    assert y[(X[:, 0] == 1.0) & (X[:, 1] == 1.0)][0] == 1.0


def test_held_out_grid_is_offset_from_training_knots():
    knots = {0.0, 1.0 / 3, 2.0 / 3, 1.0}
    for v in held_out_grid(20, 0.025, 0.975):
        assert v not in knots


def test_train_test_split_and_out_of_sample_accuracy():
    Xtr, ytr, Xte, yte = train_test_datasets("parity")
    assert Xte.shape[0] > Xtr.shape[0]
    clf = QNNClassifier(3, 3, Xtr, ytr, readout="global")
    theta = np.random.default_rng(0).uniform(0.0, 2 * np.pi, size=(3, 3))
    acc = clf.accuracy_on(theta, Xte, yte)
    assert 0.0 <= acc <= 1.0
    assert clf.loss_on(theta, Xte, yte) >= 0.0


def test_predict_on_length_matches_grid():
    Xtr, ytr, Xte, yte = train_test_datasets("additive")
    clf = QNNClassifier(3, 3, Xtr, ytr, readout="local", qubits=[0, 1])
    theta = np.zeros((3, 3))
    pred = clf.predict_on(theta, Xte)
    assert pred.shape == (Xte.shape[0],)
