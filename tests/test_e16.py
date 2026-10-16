"""E16 checks: the screening report, the verdict thresholds, and the claim
that a local cost carries more init-time signal than a global one. No hardware.
"""

import pytest

from bpqnn.screen import (
    classify,
    format_report,
    gradient_variance,
    recommend,
    screen,
)


def test_report_structure_and_verdict():
    rep = screen(n=4, M=5, cost="global", init="random", N=32, seed=0)
    assert set(rep) == {
        "config",
        "wide_ref",
        "slope_per_qubit",
        "verdict",
        "mitigation",
        "recommendation",
    }
    assert rep["verdict"] in {"trainable", "marginal", "barren plateau"}
    assert rep["config"]["grad_var"] > 0.0
    assert rep["mitigation"]["cost"] == "local"
    assert rep["mitigation"]["improvement_x"] > 0.0


def test_local_cost_beats_global_at_shallow_depth():
    kw = dict(M=5, init="random", N=64, seed=5)
    vg = gradient_variance(6, cost="global", **kw)["grad_var"]
    vl = gradient_variance(6, cost="local", **kw)["grad_var"]
    assert vl > vg


def test_classify_thresholds():
    assert classify(-2.0, 1e-6) == "barren plateau"
    assert classify(-0.5, 1e-6) == "marginal"
    assert classify(-0.1, 1e-3) == "trainable"
    assert classify(None, 0.0).startswith("degenerate")


def test_recommendation_follows_verdict():
    assert recommend("global", "random", "trainable") is None
    assert "local cost" in recommend("global", "random", "barren plateau")
    assert "identity" in recommend("local", "random", "marginal")


def test_identity_init_requires_odd_depth():
    with pytest.raises(ValueError):
        gradient_variance(4, M=4, init="identity", N=8)


def test_format_report_is_readable():
    text = format_report(screen(n=4, M=5, cost="global", N=16, seed=1))
    assert "NeuraQ trainability screen" in text
    assert "verdict" in text
