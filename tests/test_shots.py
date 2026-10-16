import numpy as np

from bpqnn.hamiltonian import ising_ground_state
from bpqnn.shots import estimate_ising_energy


def test_shot_energy_estimator_unbiased_on_ground_state():
    n = 8
    E0, gs, _ = ising_ground_state(n, h=1.0)
    rng = np.random.default_rng(5)
    est, used, se = estimate_ising_energy(
        gs, n, h=1.0, shots=4096, n_batches=20, rng=rng)
    assert abs(est - E0) < 6.0 * se  # within shot-noise error
    assert abs(est - E0) < 0.15  # and small in absolute terms
    assert used == (1 + n) * 4096  # budgeted per basis (D8)


def test_shot_energy_estimator_accurate_at_many_shots():
    n = 6
    E0, gs, _ = ising_ground_state(n, h=1.0)
    rng = np.random.default_rng(7)
    est, _, se = estimate_ising_energy(
        gs, n, h=1.0, shots=8192, n_batches=30, rng=rng)
    assert abs(est - E0) < 4.0 * se