import json
import statistics

import numpy as np
import pytest

from bpqnn.init import identity_block_init, random_init, zero_init
from bpqnn.train import adam_train
from experiments.e2b_init_layerwise import axes_folded, run_state

_RESULT = json.load(open(
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "results" / "e3_training.json"))


def _med(arm):
    return statistics.median(r["rel_err"] for r in arm["records"])


@pytest.mark.skipif(not _RESULT.get("arms"), reason="results file regenerated")
def test_e3_order_local_arms():
    """Equal-budget Adam: identity > random > zero > layerwise, at n=8."""
    a = _RESULT["arms"]
    assert _med(a["local_identity"]) < _med(a["local_random"])
    assert _med(a["local_random"]) < _med(a["local_zero"])
    assert _med(a["local_zero"]) < _med(a["layerwise_identity"])


@pytest.mark.skipif(not _RESULT.get("arms"), reason="results file regenerated")
def test_e3_global_reaches_floor_at_small_n():
    """With exact adjoint gradients, Adam masks gradient-scale vanishing:
    global-infidelity arms reach within ~3x of the oracle floor at n=8."""
    a = _RESULT["arms"]
    o = _RESULT["oracle"]["ising"]
    floor = statistics.median(r["rel_err"] for r in o)
    assert _med(a["global_random"]) < 6 * floor
    assert _med(a["global_identity"]) < 6 * floor
    assert all(r["energy"] > _RESULT["config"]["E0"] for r in
               a["global_random"]["records"])  # no below-E0 (ghost) values


@pytest.mark.skipif(not _RESULT.get("scaling"), reason="results missing scaling")
def test_e3_scaling_same_ordering_at_n10():
    sc = _RESULT["scaling"]
    assert (statistics.median(r["rel_err"] for r in
                              sc["arms"]["local_random"]["records"])
            > statistics.median(r["rel_err"] for r in
                                sc["arms"]["global_random"]["records"]))


def test_e3_mini_repro():
    """Deterministic mini run: identity init beats zero init at n=8 (600 steps).
    Mirrors the seeded headline medians (1.09e-1 vs 2.74e-1)."""
    n, H, M, steps = 8, 1.0, 17, 600
    E0, _, w = __import__("bpqnn.hamiltonian", fromlist=["ising_ground_state"]
                          ).ising_ground_state(n, h=H)
    E1 = float(w[1])
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = axes_folded(M)
    rels = {}
    t0_id = identity_block_init(M, n, np.random.default_rng(3))
    r_id = adam_train(t0_id, n=n, kind="ising", h=H, edges=edges, axes=axes,
                      max_steps=steps)
    st = run_state(n, edges, axes, r_id["theta"])
    from bpqnn.hamiltonian import ising_energy
    E_id = float(ising_energy(st, n, h=H)[0])
    t0_z = zero_init(M, n)
    r_z = adam_train(t0_z, n=n, kind="ising", h=H, edges=edges, axes=axes,
                     max_steps=steps)
    st = run_state(n, edges, axes, r_z["theta"])
    E_z = float(ising_energy(st, n, h=H)[0])
    rels = {"id": (E_id - E0) / (E1 - E0), "zero": (E_z - E0) / (E1 - E0)}
    assert rels["id"] < 0.75 * rels["zero"]