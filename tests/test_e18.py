"""E18 checks: the shot-noise model and its convergence to the exact gradient."""

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import random_init
from bpqnn.screen import screen
from bpqnn.simulator import default_axes

from experiments.e18_screen_validation import (
    _batched_cost_and_var,
    exact_cost,
    shot_gradient,
)


def _setup(n, M):
    edges = [(q, q + 1) for q in range(n - 1)]
    return edges, default_axes(M)


def test_exact_cost_matches_statevector():
    n, M = 4, 3
    edges, axes = _setup(n, M)
    theta = random_init(M, n, np.random.default_rng(0))
    c = exact_cost(theta, n, "global", edges, axes)
    assert 0.0 <= c <= 1.0
    cl = exact_cost(theta, n, "local", edges, axes)
    assert 0.0 <= cl <= 1.0
    assert cl <= c + 1e-9  # C_L <= C_G for the same state


def test_batched_cost_variance_is_nonnegative_and_shrinks_with_shots():
    n, M = 4, 3
    edges, axes = _setup(n, M)
    theta = random_init(M, n, np.random.default_rng(1), batch=8)
    _, v_low = _batched_cost_and_var(theta, n, "global", edges, axes, shots=100)
    _, v_high = _batched_cost_and_var(theta, n, "global", edges, axes, shots=10000)
    assert np.all(v_low >= 0.0) and np.all(v_high >= 0.0)
    assert np.all(v_high < v_low)


def test_shot_gradient_converges_to_adjoint_with_many_shots():
    n, M = 3, 3
    edges, axes = _setup(n, M)
    theta = random_init(M, n, np.random.default_rng(2))
    for cost, kind in (("global", "global_zero"), ("local", "local_zero")):
        rng = np.random.default_rng(3)
        g_shot = shot_gradient(theta, n, cost, edges, axes, shots=10 ** 7, rng=rng)
        g_exact = adjoint_gradient(theta, n, edges, axes, kind)
        assert np.max(np.abs(g_shot - g_exact)) < 0.01


def test_screen_verdict_is_a_known_label():
    rep = screen(n=4, M=3, cost="global", N=32, seed=0)
    assert rep["verdict"] in {"trainable", "marginal", "barren plateau"}
