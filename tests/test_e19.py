"""E19 checks: the blended-cost gradient, the normalisation bound, and the arm."""

import numpy as np

from bpqnn.csa import adam_mix, mixed_cost_of, mixed_grad
from bpqnn.grad import finite_difference
from bpqnn.init import random_init
from bpqnn.simulator import default_axes

from experiments.e19_lambda_mix import bound_check


def test_mixed_gradient_matches_finite_difference():
    n, M = 4, 5
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    theta = random_init(M, n, np.random.default_rng(0))
    g = mixed_grad(theta, n, 0.4, edges, axes).reshape(M, n)
    gfd = finite_difference(
        theta, lambda t: mixed_cost_of(t, n, 0.4, edges, axes)
    ).reshape(M, n)
    assert np.max(np.abs(g - gfd)) < 1e-6


def test_normalisation_bound_holds():
    b = bound_check(6, seed=2)
    assert b["max_violation_low"] < 1e-9          # C_L <= C_G
    assert b["max_violation_high"] < 1e-9          # C_G <= n C_L
    assert b["max_ratio_global_over_local"] <= 6 + 1e-6


def test_local_endpoint_recovers_local_gradient():
    n, M = 4, 5
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    theta = random_init(M, n, np.random.default_rng(1))
    g0 = mixed_grad(theta, n, 0.0, edges, axes)
    from bpqnn.grad import adjoint_gradient
    g_local = adjoint_gradient(theta, n, edges, axes, "local_zero")
    assert np.max(np.abs(g0.reshape(M, n) - g_local)) < 1e-9


def test_adam_mix_reduces_global_cost():
    n, M = 6, 6
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    th0 = random_init(M, n, np.random.default_rng(4))
    c0 = mixed_cost_of(th0, n, 1.0, edges, axes)
    res = adam_mix(th0, n, steps=120, lr=0.05, lam_max=1.0, schedule="linear",
                   edges=edges, axes=axes)
    assert res["finals"]["cost_G"] < c0 - 0.1
