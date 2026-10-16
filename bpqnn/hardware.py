"""QPU realisation of the E2a global-vs-local gradient protocol.

Same layered ansatz as ``bpqnn.simulator``, built as a Qiskit circuit so both
costs can be read off the measured bitstrings:

    C_G = 1 - P(0...0)              global
    C_L = 1 - (1/n) sum_i P_i(0)    local

One counts dictionary gives both. Gradients use parameter shift,
g = 0.5 * (C(t + pi/2) - C(t - pi/2)).
"""

from __future__ import annotations

import numpy as np

from .simulator import default_axes

SHIFT = np.pi / 2.0


def build_circuit(n: int, M: int, edges=None, axes=None,
                  measure: bool = True):
    """Qiskit circuit for the simulator ansatz.

    Returns ``(circuit, params)`` with ``params[l][q]`` the Parameter for layer
    ``l``, qubit ``q``. Flatten row-major to match ``theta.reshape(-1)``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)
    if len(axes) != M:
        raise ValueError(f"axes has length {len(axes)}, expected {M}")

    params = [[Parameter(f"th_{l}_{q}") for q in range(n)] for l in range(M)]
    qc = QuantumCircuit(n)
    for l in range(M):
        for q in range(n):
            if axes[l] == "Y":
                qc.ry(params[l][q], q)
            elif axes[l] == "Z":
                qc.rz(params[l][q], q)
            else:
                raise ValueError(f"unknown axis: {axes[l]!r}")
        if l < M - 1:
            for (a, b) in edges:
                qc.cz(a, b)
    if measure:
        qc.measure_all()
    return qc, params


def flat_params(params) -> list:
    """Row-major flatten of the (M, n) parameter grid."""
    M = len(params)
    n = len(params[0])
    return [params[l][q] for l in range(M) for q in range(n)]


def marginals_from_counts(counts, n: int):
    """``(P(0...0), [P_i(0)]_i)`` from a counts dict.

    Qiskit prints the highest qubit first, so qubit ``i`` is ``key[n-1-i]``.
    """
    total = float(sum(counts.values()))
    if total <= 0:
        raise ValueError("empty counts")
    p_all = 0.0
    pi0 = np.zeros(n, dtype=np.float64)
    for key, c in counts.items():
        k = key.replace(" ", "")
        if len(k) != n:
            raise ValueError(f"key {key!r} does not have {n} bits")
        if "1" not in k:
            p_all += c
        for i in range(n):
            if k[n - 1 - i] == "0":
                pi0[i] += c
    return p_all / total, pi0 / total


def costs_from_counts(counts, n: int):
    """``(C_G, C_L)`` for one measured distribution."""
    p_all, pi0 = marginals_from_counts(counts, n)
    return 1.0 - p_all, 1.0 - float(pi0.mean())


def counts_to_array(counts, n: int, dtype=object):
    """Counts -> little-endian hit array. Helper for the tests."""
    hits = np.zeros(1 << n, dtype=dtype)
    for key, c in counts.items():
        k = key.replace(" ", "")
        val = sum((1 if ch == "1" else 0) << (n - 1 - i) for i, ch in enumerate(k))
        hits[val] += c
    return hits


def shifted_param_sets(thetas, l0: int, q0: int, shift: float = SHIFT):
    """``(2R, M*n)`` bindings: row ``2r`` shifts ``(l0, q0)`` by ``+shift``, row ``2r+1`` by ``-shift``."""
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.ndim != 3:
        raise ValueError("thetas must have shape (R, M, n)")
    R, M, n = thetas.shape
    flat = thetas.reshape(R, M * n).copy()
    idx = l0 * n + q0
    out = np.empty((2 * R, M * n), dtype=np.float64)
    plus = flat.copy(); plus[:, idx] += shift
    minus = flat.copy(); minus[:, idx] -= shift
    out[0::2] = plus
    out[1::2] = minus
    return out


def gradient_components(count_plus, count_minus, n: int):
    """Parameter-shift gradient of ``(C_G, C_L)`` for one parameter."""
    cG_p, cL_p = costs_from_counts(count_plus, n)
    cG_m, cL_m = costs_from_counts(count_minus, n)
    return 0.5 * (cG_p - cG_m), 0.5 * (cL_p - cL_m)


def run_gradients(circuit, params, thetas, probes, sampler, shots: int,
                  creg: str = None):
    """Run the shifted circuits, return ``gG, gL`` shaped ``(R, P)``.

    ``probes`` is a list of ``(l0, q0)`` component indices. One PUB per
    (init, probe, sign): batching a parameter array into a single PUB merges
    the counts, so keep them separate.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    R, M, n = thetas.shape
    if creg is None:
        creg = circuit.cregs[0].name if circuit.cregs else "meas"
    pubs = []
    meta = []
    for p_i, (l0, q0) in enumerate(probes):
        arr = shifted_param_sets(thetas, l0, q0)
        for r in range(2 * R):
            pubs.append((circuit, arr[r]))
            meta.append((p_i, r))
    result = sampler.run(pubs, shots=shots).result()
    cnt = [result[k].data[creg].get_counts() for k in range(len(pubs))]
    gG = np.empty((R, len(probes)))
    gL = np.empty((R, len(probes)))
    for k, (p_i, r) in enumerate(meta):
        init = r // 2
        sign = r % 2
        other = k + 1 if sign == 0 else k - 1
        c_a, c_b = cnt[k], cnt[other]
        if sign == 0:
            gG[init, p_i], gL[init, p_i] = gradient_components(c_a, c_b, n)
    return gG, gL


def sample_variance(g, subtract_shot_var=None):
    """Sample variance over inits (ddof=1), averaged over the probe components."""
    g = np.asarray(g, dtype=np.float64)
    v = g.var(axis=0, ddof=1)
    if subtract_shot_var is not None:
        v = np.clip(v - np.asarray(subtract_shot_var), 0.0, None)
    return float(v.mean())


def analytic_shot_var(n, probes, shots, p_all=None, p_i=None):
    """Variance the shot noise adds to the shifted gradient, per component.

    ``g = 0.5 (C+ - C-)`` so ``Var = 0.5 Var(C)``. The global cost uses
    ``p = P(0...0)``; the local cost averages ``n`` marginals, each with
    per-qubit variance ``p_i(1-p_i)/shots`` (marginals treated as independent).
    """
    if p_all is None:
        p_all = 2.0 ** (-n)
    if p_i is None:
        p_i = 0.5
    vG = 0.5 * p_all * (1.0 - p_all) / shots
    vL = 0.5 * (p_i * (1.0 - p_i) / shots) / n
    return {l0q0: (vG, vL) for l0q0 in probes}
