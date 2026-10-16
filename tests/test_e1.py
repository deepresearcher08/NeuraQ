import numpy as np

from bpqnn.reference import two_design_var
from experiments.e1_plateau import ansatz_gradients, depth_log


def test_e1_variance_decays_exponentially_with_n():
    ns = [2, 3, 4, 5, 6]
    aggs = []
    for n in ns:
        g = ansatz_gradients(n, depth_log(n), N=256, seed=1000 + n)
        aggs.append(g.var(axis=0).mean())
    logv = np.log(aggs)
    slope = np.polyfit(ns, logv, 1)[0]
    assert slope < 0  # decaying with n
    assert slope > -3.0  # not absurdly steep


def test_e1_same_order_as_two_design_reference():
    n = 5
    g = ansatz_gradients(n, 3, N=1024, seed=7)
    ansatz_var = g.var(axis=0).mean()
    ref_var, _, _ = two_design_var(n, n_samples=40000, rng=np.random.default_rng(0))
    ratio = ansatz_var / ref_var
    assert 0.2 < ratio < 50.0  # same order of magnitude
