"""Tests for the SQD/QSCI probe (SPEC E8)."""

import numpy as np

from bpqnn.hamiltonian import ising_ground_state, ising_hamiltonian
from bpqnn.sqd import (
    embed_subspace_vector,
    half_chain_entropy,
    qsci_indices,
    rel_energy_error,
    sample_counts,
    sample_pool,
    sampled_indices,
    sqd_ground_energy,
    state_probabilities,
)


def test_full_space_recovers_ground_state():
    n, h = 4, 1.0
    H = ising_hamiltonian(n, h=h)
    E0, gs, w = ising_ground_state(n, h=h)
    idx = np.arange(1 << n)
    Emin, vec, _ = sqd_ground_energy(H, idx)
    assert abs(Emin - E0) < 1e-10
    psi = embed_subspace_vector(vec, idx, 1 << n)
    assert abs(abs(np.vdot(gs, psi)) - 1.0) < 1e-8


def test_subspace_projection_is_restriction():
    n, h = 5, 1.0
    H = ising_hamiltonian(n, h=h)
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(1 << n, size=12, replace=False))
    Emin, _, _ = sqd_ground_energy(H, idx)
    Hsub = H[np.ix_(idx, idx)]
    w = np.linalg.eigvalsh(Hsub)
    assert abs(Emin - w[0]) < 1e-12


def test_energy_is_variational_upper_bound():
    n, h = 6, 1.0
    H = ising_hamiltonian(n, h=h)
    E0, _, _ = ising_ground_state(n, h=h)
    rng = np.random.default_rng(1)
    for _ in range(5):
        idx = np.sort(rng.choice(1 << n, size=20, replace=False))
        Emin, _, _ = sqd_ground_energy(H, idx)
        assert Emin >= E0 - 1e-9


def test_qsci_indices_include_single_flips():
    n = 4
    idx = qsci_indices([0], n)
    assert set(idx.tolist()) == {0, 1, 2, 4, 8}
    idx2 = qsci_indices([0], n, include_flips=False)
    assert idx2.tolist() == [0]


def test_sample_counts_total_and_pool():
    n, h = 4, 1.0
    _, gs, _ = ising_ground_state(n, h=h)
    rng = np.random.default_rng(2)
    c = sample_counts(gs, n, 1000, rng)
    assert c.sum() == 1000
    assert c.shape == (1 << n,)
    pool = sample_pool(gs, n, 500, rng)
    assert pool.shape == (500,) and pool.max() < (1 << n)
    assert np.allclose(state_probabilities(gs, n).sum(), 1.0)


def test_entropy_product_and_bell():
    n = 4
    product = np.zeros(1 << n, dtype=complex)
    product[0] = 1.0
    assert abs(half_chain_entropy(product, n)) < 1e-12
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    assert abs(half_chain_entropy(bell, 2) - 1.0) < 1e-12


def test_sqd_recovers_energy_from_ground_state_samples():
    n, h = 6, 1.0
    H = ising_hamiltonian(n, h=h)
    E0, gs, w = ising_ground_state(n, h=h)
    E1 = float(w[1])
    rng = np.random.default_rng(3)
    counts = sample_counts(gs, n, 4096, rng)
    idx = qsci_indices(sampled_indices(counts), n)
    Emin, _, _ = sqd_ground_energy(H, idx)
    assert rel_energy_error(Emin, E0, E1) < 1e-2


def test_rel_energy_error_zero_at_ground():
    assert abs(rel_energy_error(2.0, 2.0, 5.0)) < 1e-15
