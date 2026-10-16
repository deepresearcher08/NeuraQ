"""Tests for the CTQD module and E14 experiment (SPEC §14r).

CTQD = "train classically, deploy quantumly": the IQP-style phase ansatz
``U(theta) = H^xn D(theta) H^xn`` whose marginal correlators
``<Z_a> = 2^{-n} sum_t cos(phi(t^a)-phi(t))`` are classically computable
without building the state, and whose spectral MMD is minimised in pure NumPy
with ZERO quantum gradient evaluations.
"""

import numpy as np

from bpqnn.ctqd import (
    classical_adam_train,
    correlator_set,
    dense_phase_signs,
    expectation_z,
    expectation_z_from_state,
    iqp_state,
    mmd_loss_and_grad,
    phase_signs,
    sampler_frontier,
    sample_from_theta,
    wht,
    xor_table,
)
from bpqnn.hamiltonian import ising_ground_state, ising_hamiltonian
from bpqnn.sqd import (
    half_chain_entropy,
    qsci_indices,
    rel_energy_error,
    sqd_ground_energy,
)


def _theta(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    signs, _ = phase_signs(n)
    return rng.uniform(-scale, scale, size=signs.shape[0])


def test_xor_table_correct():
    n = 5
    masks = np.array([1, 3, 7, 17])
    t = np.arange(1 << n)
    idx = xor_table(masks, n)
    assert idx.shape == (4, 1 << n)
    assert np.array_equal(idx[0], masks[0] ^ t)
    assert np.array_equal(idx[3], masks[3] ^ t)


def test_iqp_expectation_formula_matches_state():
    n = 5
    theta = _theta(n, 1)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K=2)
    idx_ta = xor_table(cs["masks"], n)
    z_formula = expectation_z(theta, signs, idx_ta)
    z_state = expectation_z_from_state(iqp_state(theta, n, signs), n, cs["masks"])
    assert np.allclose(z_formula, z_state, atol=1e-10)
    assert z_state.shape == cs["masks"].shape


def test_iqp_state_normalized_and_matches_formula():
    n = 6
    theta = _theta(n, 2)
    signs, _ = phase_signs(n)
    psi = iqp_state(theta, n, signs)
    assert abs(np.linalg.norm(psi) ** 2 - 1.0) < 1e-10
    masks = correlator_set(n, K=2)["masks"]
    assert np.allclose(
        expectation_z_from_state(psi, n, masks),
        expectation_z(theta, signs, xor_table(masks, n)),
        atol=1e-10,
    )


def test_wht_is_two_energy():
    rng = np.random.default_rng(3)
    v = rng.normal(size=1 << 6) + 1j * rng.normal(size=1 << 6)
    assert np.allclose(np.linalg.norm(wht(v)) ** 2, 2 ** 6 * np.linalg.norm(v) ** 2)


def test_zero_phase_is_delta():
    n = 4
    signs, _ = phase_signs(n)
    theta = np.zeros(signs.shape[0])
    psi = iqp_state(theta, n, signs)
    p = np.abs(psi.reshape(-1)) ** 2
    assert p.argmax() == 0
    assert p[0] > 1 - 1e-12
    cs = correlator_set(n, K=2)
    z = expectation_z(theta, signs, xor_table(cs["masks"], n))
    assert np.allclose(z, 1.0, atol=1e-10)


def test_mmd_gradient_finite_difference():
    n = 5
    rng = np.random.default_rng(4)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K=3)
    idx_ta = xor_table(cs["masks"], n)
    target_z = rng.uniform(-0.3, 0.3, size=cs["masks"].size)
    theta = _theta(n, 5, scale=0.5)
    loss, grad = mmd_loss_and_grad(theta, signs, idx_ta, cs["weights"], target_z)
    eps = 1e-6
    fd = np.zeros_like(grad)
    for g in range(theta.size):
        tp = theta.copy()
        tm = theta.copy()
        tp[g] += eps
        tm[g] -= eps
        lp, _ = mmd_loss_and_grad(tp, signs, idx_ta, cs["weights"], target_z)
        lm, _ = mmd_loss_and_grad(tm, signs, idx_ta, cs["weights"], target_z)
        fd[g] = (lp - lm) / (2 * eps)
    assert np.allclose(grad, fd, atol=1e-5)


def test_classical_training_reduces_mmd():
    n = 6
    h = 1.0
    _, gs, _ = ising_ground_state(n, h=h)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K=2)
    idx_ta = xor_table(cs["masks"], n)
    target_z = expectation_z_from_state(gs, n, cs["masks"])
    rng = np.random.default_rng(6)
    theta0 = rng.uniform(-0.5, 0.5, size=signs.shape[0])
    res = classical_adam_train(
        theta0, signs=signs, idx_ta=idx_ta, weights=cs["weights"],
        target_z=target_z, lr=0.1, max_steps=80,
    )
    assert res["quantum_grad_evals"] == 0
    loss0 = res["history"][0]["loss"]
    assert res["loss"] < loss0 * 0.5


def test_samples_draw_valid_ids():
    n = 6
    theta = _theta(n, 7)
    signs, _ = phase_signs(n)
    rng = np.random.default_rng(8)
    ids = sample_from_theta(theta, n, signs, 1000, rng)
    assert ids.shape == (1000,)
    assert ids.min() >= 0 and ids.max() < (1 << n)
    assert np.allclose(np.bincount(ids, minlength=1 << n).sum(), 1000)


def test_sampler_frontier_delta_vs_random_phase():
    n = 8
    dim = 1 << n
    delta = np.zeros(dim, dtype=complex)
    delta[0] = 1.0
    fd = sampler_frontier(delta, n)
    assert fd["entropy_half"] < 1e-9
    assert fd["significant_rank_mid"] == 1
    rng = np.random.default_rng(9)
    signs, _ = dense_phase_signs(n, rng)
    theta = rng.uniform(-2 * np.pi, 2 * np.pi, size=signs.shape[0])
    fr = sampler_frontier(iqp_state(theta, n, signs), n)
    assert fr["entropy_half"] > 0.35 * n
    assert fr["significant_rank_mid"] >= (1 << (n // 2)) // 2


def test_sqd_recovery_from_ctqd_samples():
    """Trained-model samples must feed SQD orders of magnitude better than a
    uniform random-phase sampler at the same sample budget (fixed seeds)."""
    n = 8
    h = 1.0
    E0, gs, w = ising_ground_state(n, h=h)
    E1 = float(w[1])
    H = ising_hamiltonian(n, h=h)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K=3)
    idx_ta = xor_table(cs["masks"], n)
    target_z = expectation_z_from_state(gs, n, cs["masks"])
    rng = np.random.default_rng(31008)
    best = None
    for seed in range(3):
        theta0 = rng.uniform(-0.5, 0.5, size=signs.shape[0])
        res = classical_adam_train(
            theta0, signs=signs, idx_ta=idx_ta, weights=cs["weights"],
            target_z=target_z, lr=0.05, max_steps=250,
        )
        if best is None or res["loss"] < best["loss"]:
            best = res
    theta = np.asarray(best["theta"])
    rng_u = np.random.default_rng(11)
    theta_u = rng_u.uniform(0, 2 * np.pi, size=signs.shape[0])

    budget = 512
    trials = 5
    def recover(p, seeds):
        errs = []
        for s in seeds:
            r = np.random.default_rng(s)
            ids = r.choice(p.size, size=budget, p=p)
            idx = qsci_indices(np.unique(ids), n)
            Emin, _, _ = sqd_ground_energy(H, idx)
            errs.append(rel_energy_error(Emin, E0, E1))
        return float(np.mean(errs))

    p_ctqd = np.abs(iqp_state(theta, n, signs).reshape(-1)) ** 2
    p_uni = np.abs(iqp_state(theta_u, n, signs).reshape(-1)) ** 2
    seeds = [81000 + 100 * n + 13 * t for t in range(trials)]
    e_ctqd = recover(p_ctqd, seeds)
    e_uni = recover(p_uni, seeds)
    assert best["quantum_grad_evals"] == 0
    assert e_ctqd < e_uni
    assert e_ctqd < 0.2


def test_frontier_entropy_target_vs_ctqd():
    """The trained CTQD state must stay far outside the Haar/volume-law regime
    reached by the dense random-phase ansatz (the 'expensive' frontier end)."""
    n = 8
    E0, gs, _ = ising_ground_state(n, h=1.0)
    f_gs = sampler_frontier(gs, n)
    assert f_gs["entropy_half"] < 2.5
    rng = np.random.default_rng(12)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K=2)
    idx_ta = xor_table(cs["masks"], n)
    target_z = expectation_z_from_state(gs, n, cs["masks"])
    theta0 = rng.uniform(-0.5, 0.5, size=signs.shape[0])
    res = classical_adam_train(
        theta0, signs=signs, idx_ta=idx_ta, weights=cs["weights"],
        target_z=target_z, lr=0.1, max_steps=60,
    )
    f_ctqd = sampler_frontier(iqp_state(np.asarray(res["theta"]), n, signs), n)
    signs_dense, _ = dense_phase_signs(n, rng)
    theta_d = rng.uniform(-2 * np.pi, 2 * np.pi, size=signs_dense.shape[0])
    f_dense = sampler_frontier(iqp_state(theta_d, n, signs_dense), n)
    assert f_ctqd["entropy_half"] < f_dense["entropy_half"]