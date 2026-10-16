"""Tests for the MPS chi-sweep surrogate (SPEC D9 / E9)."""

import numpy as np

from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.mps import (
    bond_dimensions,
    compress,
    max_bond_dimension,
    mps_to_statevector,
    schmidt_spectrum,
    significant_rank,
    statevector_to_mps,
    truncation_weight,
)


def _random_state(n, seed):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    return psi / np.linalg.norm(psi)


def test_exact_roundtrip():
    n = 5
    psi = _random_state(n, 0)
    mps = statevector_to_mps(psi, n, chi=None)
    back = mps_to_statevector(mps, n)
    assert back.shape == (1 << n,)
    assert np.allclose(back, psi, atol=1e-12)


def test_bond_dims_bounded_by_schmidt_rank():
    n = 6
    psi = _random_state(n, 1)
    mps = statevector_to_mps(psi, n, chi=None)
    bonds = bond_dimensions(mps)
    assert bonds[-1] == 1
    for k, d in enumerate(bonds[:-1], start=1):
        assert d <= min(1 << k, 1 << (n - k))


def test_chi_one_product_state_exact():
    n = 4
    a = np.array([np.cos(0.3), np.sin(0.3)])
    b = np.array([np.cos(1.1), np.sin(1.1)])
    c = np.array([np.cos(2.0), np.sin(2.0)])
    d = np.array([np.cos(0.7), np.sin(0.7)])
    psi = np.einsum("a,b,c,d->abcd", a, b, c, d).reshape(-1).astype(complex)
    chi1 = compress(psi, n, chi=1)
    assert abs(abs(np.vdot(psi, chi1)) - 1.0) < 1e-12
    assert max_bond_dimension(statevector_to_mps(psi, n, chi=1)) == 1


def test_bell_needs_chi_two():
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    chi1 = compress(bell, 2, chi=1)
    fid1 = abs(np.vdot(bell, chi1)) ** 2
    assert abs(fid1 - 0.5) < 1e-10
    chi2 = compress(bell, 2, chi=2)
    assert abs(abs(np.vdot(bell, chi2)) - 1.0) < 1e-12


def test_schmidt_spectrum_matches_direct_svd():
    n = 6
    psi = _random_state(n, 2)
    spec = schmidt_spectrum(psi, n)
    assert len(spec) == n - 1
    k = 3
    direct = np.linalg.svd(psi.reshape(1 << k, 1 << (n - k)), compute_uv=False)
    assert np.allclose(spec[k - 1], direct)


def test_significant_rank_bell():
    spec = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    assert significant_rank(spec, tol=1e-8) == 2
    assert significant_rank(np.array([1.0, 0.0]), tol=1e-8) == 1


def test_truncation_weight_positive_for_chi_one():
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1 / np.sqrt(2)
    chi1 = mps_to_statevector(statevector_to_mps(bell, 2, chi=1), 2)
    assert abs(truncation_weight(bell, chi1) - 0.5) < 1e-10
    full = mps_to_statevector(statevector_to_mps(bell, 2, chi=None), 2)
    assert truncation_weight(bell, full) < 1e-14


def test_gs_compression_reaches_energy_at_full_chi():
    n, h = 8, 1.0
    E0, gs, _ = ising_ground_state(n, h=h)
    full = compress(gs, n, chi=None)
    assert abs(float(ising_energy(full, n, h=h)[0]) - E0) < 1e-10
    small = compress(gs, n, chi=8)
    assert abs(float(ising_energy(small, n, h=h)[0]) - E0) < 1e-3


def test_error_decreases_with_chi():
    n, h = 8, 1.0
    E0, gs, _ = ising_ground_state(n, h=h)
    errs = []
    for chi in (1, 2, 4, 8, 16):
        st = compress(gs, n, chi=chi)
        errs.append(abs(float(ising_energy(st, n, h=h)[0]) - E0))
    assert errs[-1] < 1e-10
    assert errs[0] >= errs[-1]
    assert min(errs) == errs[-1]
