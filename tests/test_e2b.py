import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.init import identity_block_init, random_init
from bpqnn.train import adam_train, layerwise_adam_train
from experiments.e2b_init_layerwise import axes_folded

N = 8
H = 1.0
M = 2 * N + 1
BLOCK = 6
MAX_STEPS = 600
EDGES = [(q, q + 1) for q in range(N - 1)]
AXES = axes_folded(M)


def _med_rel_err(records, E0, E1):
    return float(np.median([(r["cost"] - E0) / (E1 - E0) for r in records]))


def test_e2b_identity_init_is_the_computational_basis_state():
    rng = np.random.default_rng(0)
    t0 = identity_block_init(M, N, rng)
    sim = __import__("bpqnn.simulator", fromlist=["StatevectorSimulator"]).StatevectorSimulator(
        N, 1)
    sim.apply_circuit(np.asarray(t0, dtype=np.float64), EDGES, AXES)
    E = float(ising_energy(sim.state, N, h=H)[0])
    assert abs(E - (-7.0)) < 1e-9  # E(|0...0>) for n=8 open Ising, h=1


def test_e2b_identity_edge_grad_variance_is_zero():
    rng = np.random.default_rng(3)
    t = identity_block_init(M, N, rng, batch=64)
    g = adjoint_gradient(t, N, EDGES, AXES, "ising", h=H)
    first_var = float(g.var(axis=0)[0].mean())
    last_var = float(g.var(axis=0)[-1].mean())
    assert first_var < 1e-15
    assert last_var < 1e-15


def test_e2b_training_order_equal_budget():
    E0, _, w = ising_ground_state(N, h=H)
    E1 = float(w[1])
    records = {"random_full": [], "identity_full": [], "identity_layerwise": []}
    for s in range(4):
        rng = np.random.default_rng(1000 + s)
        t0_rd = random_init(M, N, rng)
        t0_id = identity_block_init(M, N, rng)
        rs = (
            adam_train(t0_rd, n=N, kind="ising", h=H, edges=EDGES, axes=AXES,
                       max_steps=MAX_STEPS),
            adam_train(t0_id, n=N, kind="ising", h=H, edges=EDGES, axes=AXES,
                       max_steps=MAX_STEPS),
            layerwise_adam_train(t0_id, n=N, kind="ising", h=H, edges=EDGES,
                                 axes=AXES, block=BLOCK, max_steps=MAX_STEPS,
                                 grow="output"),
        )
        for name, r in zip(records, rs):
            records[name].append(r)
    med_id = _med_rel_err(records["identity_full"], E0, E1)
    med_rd = _med_rel_err(records["random_full"], E0, E1)
    med_lw = _med_rel_err(records["identity_layerwise"], E0, E1)
    assert med_id < med_rd  # identity init is more budget-efficient
    assert med_lw > 3.0 * med_id  # layerwise (fixed hyperparams) lags clearly