"""E21 checks: matched classical baselines and the head-to-head result file."""

import json
import pathlib

import numpy as np
import pytest

from bpqnn.classical import (
    NumpyMLP,
    evaluate_numpy_mlp,
    mlp_param_count,
)
from bpqnn.classify import train_test_datasets

_RESULT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "results"
    / "e21_classical_baseline.json"
)


def test_matched_mlp_parameter_count_matches_qnn_budget():
    assert mlp_param_count(2, 10) == 41
    assert mlp_param_count(2, 32) == 129
    # QNN in E6/E17 is 6 qubits x depth 7 = 42 angles.
    assert abs(mlp_param_count(2, 10) - 6 * 7) <= 1


def test_numpy_mlp_learns_additive_and_parity():
    for task in ("additive", "parity"):
        Xtr, ytr, Xte, yte = train_test_datasets(task)
        model = NumpyMLP(2, hidden=10, seed=0, lr=0.05, steps=300)
        model.fit(Xtr, ytr)
        assert model.accuracy(Xtr, ytr) >= 0.9
        pred = model.predict(Xte)
        assert pred.shape == (Xte.shape[0],)
        assert np.all(np.isfinite(pred))
        assert 0.0 <= model.accuracy(Xte, yte) <= 1.0


def test_evaluate_numpy_mlp_reports_seed_samples():
    Xtr, ytr, Xte, yte = train_test_datasets("parity")
    out = evaluate_numpy_mlp(Xtr, ytr, Xte, yte, hidden=8, seeds=range(3))
    assert out["n_params"] == mlp_param_count(2, 8)
    assert len(out["test_acc"]) == 3
    assert len(out["final_loss"]) == 3
    assert 0.0 <= out["median_test_acc"] <= 1.0


@pytest.mark.skipif(not _RESULT_PATH.exists(), reason="E21 results not generated")
def test_e21_head_to_head_file_is_consistent():
    with open(_RESULT_PATH) as f:
        out = json.load(f)

    assert out["param_counts"]["qnn"] == 42
    assert out["param_counts"]["mlp_matched"] == 41
    assert out["config"]["n_train"] == 16
    assert out["config"]["n_test"] == 400

    add = out["tasks"]["additive"]
    par = out["tasks"]["parity"]
    # Every model is scored on the same held-out grid.
    for rows in (add, par):
        for row in rows.values():
            assert 0.0 <= row["median_test_acc"] <= 1.0
            assert row["n_seeds"] >= 1
    # The matched MLP is a real baseline: it learns both tasks.
    assert add["mlp_matched"]["median_test_acc"] >= 0.8
    assert par["mlp_matched"]["median_test_acc"] >= 0.8
    # Parity is not linearly separable; a QNN with a global readout is.
    assert par["logreg"]["median_test_acc"] <= 0.65
    assert par["qnn_global"]["median_test_acc"] >= 0.95
    assert par["qnn_local"]["median_test_acc"] <= 0.65

    assert out["verdict"]["linear_fails_parity"] is True
    assert out["verdict"]["qnn_competitive_parity"] is True
