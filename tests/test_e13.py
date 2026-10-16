"""E13: Cost-Support Annealing module checks."""

from __future__ import annotations

import numpy as np

from bpqnn.csa import (
    Linear,
    Staged,
    VarGated,
    adam_block,
    adam_window,
    block_cost_of,
    block_grad,
    csa_adam_train,
    window_grad,
)
from bpqnn.grad import finite_difference
from bpqnn.init import random_init
from bpqnn.simulator import default_axes


def test_schedules_are_monotone_and_bounded():
    for sched in [Linear(5, 100), Staged(5, 10), VarGated(5, gate_every=5, floor_frac=0.25)]:
        rs = [sched.r(t) for t in range(100)]
        assert rs[0] == 1
        assert all(1 <= r <= 5 for r in rs)
        assert all(rs[i] <= rs[i + 1] for i in range(len(rs) - 1))


def test_linear_ramps_linearly():
    s = Linear(5, 200)
    assert s.r(0) == 1 and s.r(199) == 5
    assert [s.r(t) for t in (0, 50, 100, 150)] == [1, 2, 3, 4]


def test_var_gated_grows_only_when_affordable():
    s = VarGated(5, gate_every=1, floor_frac=0.5)
    s.base = 1.0
    s.update({"pow_current": 0.8, "pow_prospective": 0.9})
    assert s.cur == 2
    s.update({"pow_current": 0.1, "pow_prospective": 0.9})
    assert s.cur == 2  # current power below floor -> no growth
    s.update({"pow_current": 0.8, "pow_prospective": 0.1})
    assert s.cur == 2  # prospective below floor -> no growth


def test_block_grad_matches_finite_difference():
    n, M, r = 4, 4, 2
    axes = default_axes(M)
    edges = [(q, q + 1) for q in range(n - 1)]
    theta = random_init(M, n, np.random.default_rng(0))
    cost = lambda t: block_cost_of(t, n, r, edges, axes)
    g_adj = block_grad(theta, n, r, edges, axes)
    g_fd = finite_difference(theta, cost, eps=1e-6).reshape(M, n)
    assert np.max(np.abs(g_adj - g_fd)) < 1e-5


def test_window_grad_matches_finite_difference():
    from bpqnn.observables import support_radius_cost
    n, M, r = 4, 4, 2
    axes = default_axes(M)
    edges = [(q, q + 1) for q in range(n - 1)]
    from bpqnn.simulator import StatevectorSimulator

    def cost(t):
        sim = StatevectorSimulator(n, 1)
        sim.apply_circuit(np.asarray(t, np.float64), edges, axes)
        return float(support_radius_cost(sim.state, r, n)[0])

    theta = random_init(M, n, np.random.default_rng(3))
    g_adj = window_grad(theta, n, r, edges, axes)
    g_fd = finite_difference(theta, cost, eps=1e-6).reshape(M, n)
    assert np.max(np.abs(g_adj - g_fd)) < 1e-5


def test_window_cost_is_faithful_for_allzero_target():
    # window-averaged C_w(r) at ANY r only vanishes when every qubit is |0>,
    # so a well-trained window arm drives the *global* all-zero probability.
    n, M = 6, 6
    axes = default_axes(M)
    edges = [(q, q + 1) for q in range(n - 1)]
    th0 = random_init(M, n, np.random.default_rng(4))
    res = adam_window(th0, n, 2, steps=200, lr=0.05, axes=axes)
    c0 = block_cost_of(th0, n, n, edges, axes)
    c1 = block_cost_of(res["theta"], n, n, edges, axes)
    assert c1 < c0 - 0.5


def test_adam_block_reduces_its_own_cost():
    n, M, r = 6, 6, 3
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    th0 = random_init(M, n, np.random.default_rng(1))
    res = adam_block(th0, n, r, steps=150, lr=0.05, axes=axes)
    assert block_cost_of(res["theta"], n, r, edges, axes) < \
           block_cost_of(th0, n, r, edges, axes) - 0.05


def test_csa_wins_or_ties_vs_local_on_regional_block():
    # regional task on n=6: target block r*=3; the r=1 local arm is blind to
    # qubits 1,2, so CSA (growing 1->3) should do at least as well as the
    # local arm, and strictly better on the *target* cost.
    n, M, r_star = 6, 6, 3
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    csa_best, local = np.inf, []
    for s in range(5):
        th0 = random_init(M, n, np.random.default_rng(10 + s))
        res = csa_adam_train(th0, n, r_star, schedule="linear", steps=150,
                             lr=0.05, axes=axes)
        csa_best = min(csa_best, block_cost_of(res["theta"], n, r_star, edges, axes))
        rl = adam_block(th0, n, 1, steps=150, lr=0.05, axes=axes)
        local.append(block_cost_of(rl["theta"], n, r_star, edges, axes))
    assert csa_best < np.mean(local) + 1e-3


def test_csa_targets_are_the_given_radius():
    res = csa_adam_train(random_init(6, 6, np.random.default_rng(2)), 6, 4,
                         schedule="staged", hold=15, steps=90)
    assert set(res["finals"]) == {"cost_target", "cost_1", "cost_n", "final_r"}
    assert res["finals"]["final_r"] == 4