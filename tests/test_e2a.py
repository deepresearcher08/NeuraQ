import numpy as np

from experiments.e2a_cost_locality import ansatz_gradients, depth_log, depth_linear


def _agg(n, M, kind, seed):
    g = ansatz_gradients(n, M, kind, N=512, seed=seed)
    return g.var(axis=0).mean()


def test_e2a_global_cost_stays_exponential():
    ns = list(range(2, 11))
    logv = []
    for n in ns:
        M = depth_log(n)
        logv.append(np.log(_agg(n, M, "global_zero", 3000 + n)))
    slope = np.polyfit(ns, logv, 1)[0]
    assert slope < -1.0  # ~2^{-1.8n}: clearly exponential (replicates E1)


def test_e2a_local_cost_log_regime_not_exponential():
    ns = list(range(2, 11))
    logl = []
    for n in ns:
        M = depth_log(n)
        logl.append(np.log(_agg(n, M, "local_zero", 3001 + n)))
    slope = np.polyfit(ns, logl, 1)[0]
    assert slope > -0.8  # consistent with ~1/n^2 (-0.33 on log-linear), not 2^{-1.8n}


def test_e2a_local_cost_dramatically_gentler_than_global():
    ns = list(range(2, 11))
    logg, logl = [], []
    for n in ns:
        M = depth_log(n)
        logg.append(np.log(_agg(n, M, "global_zero", 3002 + n)))
        logl.append(np.log(_agg(n, M, "local_zero", 3003 + n)))
    s_g = np.polyfit(ns, logg, 1)[0]
    s_l = np.polyfit(ns, logl, 1)[0]
    assert s_l - s_g > 0.5  # local decay is at least ~2.5x gentler


def test_e2a_local_depth_boundary_beyond_sampled_range():
    n = 8
    v1 = _agg(n, 1, "local_zero", 4000)
    vdeep = _agg(n, 2 * n, "local_zero", 4001)
    assert vdeep > v1 / 10.0  # no order-of-magnitude collapse by L = 2n