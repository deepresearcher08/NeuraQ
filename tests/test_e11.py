"""E11: support-radius cost family and its adjoint gradient."""

from __future__ import annotations

import numpy as np
import pytest

from bpqnn.costs import global_cost_zero, local_cost_zero, single_qubit_zero_probs
from bpqnn.grad import adjoint_gradient, finite_difference, max_abs_diff
from bpqnn.init import random_init
from bpqnn.observables import (
    block_cost,
    block_observable,
    support_radius_cost,
    support_radius_observable,
    window_masks,
)
from bpqnn.simulator import StatevectorSimulator, default_axes


def _random_states(n, batch, seed=0):
    rng = np.random.default_rng(seed)
    sim = StatevectorSimulator(n, batch)
    sim.apply_circuit(random_init(3, n, rng, batch=batch))
    return sim.state


@pytest.mark.parametrize("n", [4, 6])
def test_r1_matches_local_cost(n):
    st = _random_states(n, 5)
    assert np.allclose(support_radius_cost(st, 1, n), local_cost_zero(st, n))


@pytest.mark.parametrize("n", [4, 6])
def test_rn_matches_global_cost(n):
    st = _random_states(n, 5)
    assert np.allclose(support_radius_cost(st, n, n), global_cost_zero(st))


def test_observable_matches_cost():
    n = 5
    st = _random_states(n, 4, seed=3)
    for r in range(1, n + 1):
        lam = support_radius_observable(st, r, n)
        val = np.real(np.sum(np.conj(st) * lam, axis=1))
        assert np.allclose(val, support_radius_cost(st, r, n))


def test_block_observable_matches_cost_and_single_qubit():
    n = 6
    st = _random_states(n, 4, seed=5)
    for r in range(1, n + 1):
        lam = block_observable(st, r, n)
        val = np.real(np.sum(np.conj(st) * lam, axis=1))
        assert np.allclose(val, block_cost(st, r, n))
    single = block_cost(st, 1, n, start=2)
    assert np.allclose(single, 1.0 - single_qubit_zero_probs(st, n)[:, 2])


def test_window_count():
    assert len(window_masks(6, 1)) == 6
    assert len(window_masks(6, 3)) == 4
    assert len(window_masks(6, 6)) == 1
    with pytest.raises(ValueError):
        window_masks(6, 7)


def test_adjoint_matches_finite_difference_custom_observable():
    n, M, r = 4, 5, 2
    rng = np.random.default_rng(1)
    theta = random_init(M, n, rng)
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)

    def cost_fn(pp):
        pp = np.asarray(pp)
        out = np.empty(pp.shape[0])
        for i in range(pp.shape[0]):
            sim = StatevectorSimulator(n, 1)
            sim.apply_circuit(pp[i], edges, axes)
            out[i] = support_radius_cost(sim.state, r, n)[0]
        return out

    g = adjoint_gradient(
        theta, n, edges, axes,
        observable=lambda s: support_radius_observable(s, r, n),
    )
    gfd = finite_difference(theta, cost_fn)
    assert max_abs_diff(g, gfd) < 1e-6


def test_variance_decays_with_radius():
    n, M, N = 8, 8, 256
    rng = np.random.default_rng(0)
    theta = random_init(M, n, rng, batch=N)
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)

    def var(r):
        g = adjoint_gradient(
            theta, n, edges, axes,
            observable=lambda s: support_radius_observable(s, r, n),
        )
        return g.var(axis=0).mean()

    assert var(n) < var(4) < var(1)
