"""E15 checks: the circuit matches the simulator, counts give the right costs,
and the shifted gradients match the adjoint reference. All local aer, no QPU.
"""

import numpy as np
import pytest

from bpqnn.hardware import (
    build_circuit,
    costs_from_counts,
    flat_params,
    marginals_from_counts,
    run_gradients,
)
from bpqnn.init import random_init
from bpqnn.simulator import StatevectorSimulator, default_axes


def _aer_sampler(seed=1234):
    from qiskit_aer.primitives import SamplerV2
    return SamplerV2(seed=seed)


def test_circuit_matches_simulator_statevector():
    from qiskit.quantum_info import Statevector

    for n, M in [(2, 3), (3, 2), (4, 4), (5, 3)]:
        edges = [(q, q + 1) for q in range(n - 1)]
        axes = default_axes(M)
        rng = np.random.default_rng(10 * n + M)
        theta = random_init(M, n, rng)
        qc, params = build_circuit(n, M, edges, axes, measure=False)
        flat = flat_params(params)
        bound = qc.assign_parameters({p: v for p, v in zip(flat, theta.reshape(-1))})
        sv_q = Statevector(bound).data
        sim = StatevectorSimulator(n, 1)
        sim.apply_circuit(theta, edges, axes)
        sv_o = sim.state[0]
        assert np.max(np.abs(sv_q - sv_o)) < 1e-12


def test_bit_order_marginals():
    from qiskit import QuantumCircuit

    n = 3
    qc = QuantumCircuit(n)
    qc.x(0)
    qc.measure_all()
    sampler = _aer_sampler()
    res = sampler.run([(qc,)], shots=2000).result()
    counts = res[0].data.meas.get_counts()
    p_all, pi0 = marginals_from_counts(counts, n)
    assert p_all == pytest.approx(0.0, abs=0.02)
    assert pi0[0] == pytest.approx(0.0, abs=0.02)   # qubit 0 is '1'
    assert pi0[1] == pytest.approx(1.0, abs=0.02)
    assert pi0[2] == pytest.approx(1.0, abs=0.02)


def test_counts_costs_track_analytic():
    n, M = 4, 3
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    rng = np.random.default_rng(3)
    theta = random_init(M, n, rng)
    qc, params = build_circuit(n, M, edges, axes, measure=True)
    flat = flat_params(params)
    sampler = _aer_sampler()
    res = sampler.run([(qc, theta.reshape(-1))], shots=60000).result()
    counts = res[0].data.meas.get_counts()
    cG, cL = costs_from_counts(counts, n)
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(theta, edges, axes)
    from bpqnn import costs as C
    cG_exact = float(C.global_cost_zero(sim.state)[0])
    cL_exact = float(C.local_cost_zero(sim.state, n)[0])
    assert cG == pytest.approx(cG_exact, abs=0.01)
    assert cL == pytest.approx(cL_exact, abs=0.01)


def test_run_gradients_matches_adjoint_noiseless():
    from bpqnn.grad import adjoint_gradient

    n, M = 4, 3
    R = 8
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    rng = np.random.default_rng(21)
    theta = random_init(M, n, rng, batch=R)
    qc, params = build_circuit(n, M, edges, axes, measure=True)
    probes = [(1, 0), (1, 1), (1, 2), (1, 3)]
    sampler = _aer_sampler()
    gG, gL = run_gradients(qc, params, theta, probes, sampler, shots=100000)
    aG = adjoint_gradient(theta, n, edges, axes, "global_zero")
    aL = adjoint_gradient(theta, n, edges, axes, "local_zero")
    for p_i, (l0, q0) in enumerate(probes):
        errG = np.max(np.abs(gG[:, p_i] - aG[:, l0, q0]))
        errL = np.max(np.abs(gL[:, p_i] - aL[:, l0, q0]))
        assert errG < 0.02, (l0, q0, errG)
        assert errL < 0.02, (l0, q0, errL)


def test_shifted_sets_layout():
    from bpqnn.hardware import SHIFT, shifted_param_sets

    R, M, n = 3, 2, 2
    theta = np.arange(R * M * n, dtype=float).reshape(R, M, n)
    arr = shifted_param_sets(theta, 1, 1, shift=SHIFT)
    assert arr.shape == (2 * R, M * n)
    idx = 1 * n + 1
    base = theta.reshape(R, -1)
    for r in range(R):
        assert arr[2 * r, idx] == pytest.approx(base[r, idx] + SHIFT)
        assert arr[2 * r + 1, idx] == pytest.approx(base[r, idx] - SHIFT)
        keep = [j for j in range(M * n) if j != idx]
        assert np.allclose(arr[2 * r, keep], base[r, keep])
