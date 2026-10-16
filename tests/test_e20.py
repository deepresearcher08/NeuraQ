"""E20 checks: perturbed identity equals the identity at eps=0 and adds signal."""

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import identity_block_init, perturbed_identity_init
from bpqnn.simulator import default_axes


def test_eps_zero_matches_identity_block():
    M, n = 7, 4
    a = identity_block_init(M, n, np.random.default_rng(0), batch=4)
    b = perturbed_identity_init(M, n, np.random.default_rng(0), eps=0.0, batch=4)
    assert np.allclose(a, b)


def test_perturbation_is_bounded():
    M, n = 7, 4
    base = identity_block_init(M, n, np.random.default_rng(1))
    p = perturbed_identity_init(M, n, np.random.default_rng(1), eps=0.2)
    assert p.shape == base.shape
    assert np.max(np.abs(p - base)) <= 0.2 + 1e-12


def test_perturbation_restores_first_layer_signal():
    M, n = 7, 6
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    rng = np.random.default_rng(5)
    t0 = identity_block_init(M, n, rng, batch=32)
    t1 = perturbed_identity_init(M, n, rng, eps=0.2, batch=32)
    g0 = adjoint_gradient(t0, n, edges, axes, "ising")
    g1 = adjoint_gradient(t1, n, edges, axes, "ising")
    v0 = g0.var(axis=0).mean(axis=1)[0]
    v1 = g1.var(axis=0).mean(axis=1)[0]
    assert v1 > v0
