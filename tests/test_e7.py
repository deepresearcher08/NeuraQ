"""Tests for the density-matrix / noisy simulator (SPEC E7)."""

import numpy as np
import pytest

from bpqnn.noise import (
    run_circuit,
    depolarise,
    parameter_shift_grads,
    local_cost_zero,
    global_cost_zero,
    single_qubit_zero_probs,
)
from bpqnn.simulator import StatevectorSimulator, default_axes
from bpqnn.init import random_init

I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def test_noiseless_matches_statevector():
    n, M = 3, 4
    th = random_init(M, n, np.random.default_rng(0))[None, ...]
    sv = StatevectorSimulator(n, 1)
    sv.apply_circuit(th[0])
    psi = sv.state[0]
    rho = run_circuit(th, n, p=0.0)[0]
    assert np.max(np.abs(rho - np.outer(psi, np.conj(psi)))) < 1e-12


def test_depolarise_analytic_single_qubit():
    p = 0.1
    plus = np.array([1, 1], complex) / np.sqrt(2)
    rho_in = np.outer(plus, np.conj(plus))[None]
    got = depolarise(rho_in, p, 1)[0]
    want = (1 - p) * rho_in[0] + p * I2 / 2
    assert np.max(np.abs(got - want)) < 1e-12
    assert abs(np.trace(got) - 1.0) < 1e-12


def test_depolarise_pauli_sum_identity():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2) + 1j * rng.normal(size=2)
    a /= np.linalg.norm(a)
    rho = np.outer(a, np.conj(a))[None]
    p = 0.13
    via_direct = (1 - p) * rho[0] + p / 4 * (
        rho[0] + X @ rho[0] @ X + Y @ rho[0] @ Y + Z @ rho[0] @ Z
    )
    assert np.max(np.abs(depolarise(rho, p, 1)[0] - via_direct)) < 1e-12


def test_matches_aer_single_qubit():
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    th, p = 0.9, 0.1
    qc = QuantumCircuit(1)
    qc.ry(th, 0)
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["ry"])
    rho_aer = np.asarray(
        AerSimulator(method="density_matrix", noise_model=nm).run(qc).result().data()["density_matrix"]
    )
    rho = run_circuit(np.array([[[th]]]), 1, p=p)[0]
    assert np.max(np.abs(rho - rho_aer)) < 1e-10


def test_matches_aer_two_qubit_circuit():
    aer = pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    n, M, p = 2, 2, 0.05
    axes = default_axes(M)
    th = random_init(M, n, np.random.default_rng(3))
    qc = QuantumCircuit(n)
    for l in range(M):
        for q in range(n):
            assert axes[l] == "Y"
            qc.ry(float(th[l, q]), q)
        if l < M - 1:
            qc.cz(0, 1)
    qc.save_density_matrix()
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["ry"])
    rho_aer = np.asarray(
        AerSimulator(method="density_matrix", noise_model=nm).run(qc).result().data()["density_matrix"]
    )
    rho = run_circuit(th[None, ...], n, [(0, 1)], axes, p=p)[0]
    assert np.max(np.abs(rho - rho_aer)) < 1e-10


def test_parameter_shift_matches_finite_difference():
    n, M, p = 3, 3, 0.03
    th = random_init(M, n, np.random.default_rng(1))[None, ...]
    g = parameter_shift_grads(th, n, p=p, kind="local_zero")[0]
    eps = 1e-6
    gfd = np.zeros_like(g)
    for l in range(M):
        for q in range(n):
            tp = th.copy(); tp[0, l, q] += eps
            tm = th.copy(); tm[0, l, q] -= eps
            cp = local_cost_zero(run_circuit(tp, n, p=p), n)[0]
            cm = local_cost_zero(run_circuit(tm, n, p=p), n)[0]
            gfd[l, q] = (cp - cm) / (2 * eps)
    assert np.max(np.abs(g - gfd)) < 1e-6


def test_costs_in_range_and_trace_one():
    n, M, p = 3, 4, 0.02
    th = random_init(M, n, np.random.default_rng(2), batch=8)
    rho = run_circuit(th, n, p=p)
    assert np.max(np.abs(np.einsum("bii->b", rho) - 1.0)) < 1e-11
    cl = local_cost_zero(rho, n)
    cg = global_cost_zero(rho, n)
    assert np.all(cl >= -1e-12) and np.all(cl <= 1 + 1e-12)
    assert np.all(cg >= -1e-12) and np.all(cg <= 1 + 1e-12)
