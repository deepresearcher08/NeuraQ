"""Shot-noise helpers for the Ising task (SPEC D8).

A measurement estimator for ``<H>`` that budgets shots *per basis*:

- all ``(n-1)`` ZZ terms share a single Z-basis sample set (``shots`` total);
- each of the ``n`` X terms needs its own X-basis rotation (apply H to that
  qubit) and its own sample set.

So one ``<H>`` evaluation costs ``(1 + n) * shots`` physical shots, with the
budget split ``1`` vs ``n`` basis settings. Everything here samples from the
exact statevector; the sampled outcome is the only source of shot noise.
"""

from __future__ import annotations

import numpy as np


def rotate_single(state, q: int, n: int) -> np.ndarray:
    """Apply a Hadamard to qubit ``q`` (X-basis rotation for Z measurement)."""
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    dim = 1 << n
    high = dim >> (q + 1)
    low = 1 << q
    v = state.reshape(high, 2, low).copy()
    a = v[:, 0, :].copy()
    b = v[:, 1, :]
    v[:, 0, :] = (a + b) / np.sqrt(2.0)
    v[:, 1, :] = (a - b) / np.sqrt(2.0)
    return v.reshape(dim)


def sample_overlap(state, target, shots, rng):
    """Bernoulli readout of the projector ``|target><target|``.

    Models a device that rotates the computational basis so that the target
    state maps to ``|0...0>`` and then does a single Z-basis event:
    ``P(click) = |<target|state>|^2``. Returns a batch of ``shots`` bits
    (``1`` = state collapsed onto the target) for one projection event.
    """
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    state = np.asarray(state, dtype=np.complex128).reshape(-1)
    a = np.vdot(target, state)
    p = float(abs(a) ** 2)
    return (rng.random(int(shots)) < p).astype(np.float64)


def sampled_infidelity_value(state, target, shots, rng):
    """Sampled ``1 - |<target|state>|^2`` (one projector readout)."""
    return float(1.0 - np.mean(sample_overlap(state, target, shots, rng)))


def sampled_infidelity_grad_cost(theta, n, target, shots, rng,
                                 shift=np.pi / 2.0, edges=None, axes=None):
    """Parameter-shift gradient of the *sampled* infidelity (shot-noisy)."""
    M, nq = theta.shape
    grad = np.empty_like(theta)
    from bpqnn.simulator import StatevectorSimulator, default_axes

    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)

    def _F(t):
        sim = StatevectorSimulator(n, 1)
        sim.apply_circuit(t, edges, axes)
        return sampled_infidelity_value(sim.state, target, shots, rng)

    for l in range(M):
        for q in range(nq):
            pp = theta.copy()
            pm = theta.copy()
            pp[l, q] += shift
            pm[l, q] -= shift
            grad[l, q] = 0.5 * (_F(pp) - _F(pm))
    return grad


def sample_bits(state, n: int, shots: int, rng: np.random.Generator):
    """Sample ``shots`` bitstrings (little-endian) from the Born rule."""
    p = np.abs(np.asarray(state).reshape(-1)) ** 2
    cs = np.cumsum(p)
    idx = np.searchsorted(cs, rng.random(shots))
    bits = (idx[:, None] >> np.arange(n)) & 1
    return bits.astype(np.int64)


def estimate_ising_energy(
    state, n: int, h: float = 1.0, shots: int = 8192, rng=None, n_batches: int = 10
):
    """Shot-based ``<H>`` estimate with a standard error over ``n_batches``.

    Returns ``(est, shots_used, se)`` where ``shots_used = (1 + n) * shots``.
    """
    rng = np.random.default_rng() if rng is None else rng
    batch = max(1, shots // n_batches)
    estimates = np.empty(n_batches)
    bond_terms = [(q, q + 1) for q in range(n - 1)]
    for b in range(n_batches):
        E = 0.0
        bits = sample_bits(state, n, batch, rng)
        for (i, j) in bond_terms:
            E -= np.mean((1 - 2 * bits[:, i]) * (1 - 2 * bits[:, j]))
        for i in range(n):
            rot = rotate_single(state, i, n)
            bits_i = sample_bits(rot, n, batch, rng)
            E -= h * np.mean(1 - 2 * bits_i[:, i])
        estimates[b] = E
    est = float(np.mean(estimates))
    se = float(np.std(estimates, ddof=1)) if n_batches > 1 else 0.0
    return est, (1 + n) * shots, se


def sampled_ising_grad_cost(theta, n, h, shots, rng, shift=np.pi / 2.0,
                            edges=None, axes=None):
    """Parameter-shift gradient of the *sampled* energy (for shot-noisy training)."""
    M, nq = theta.shape
    grad = np.empty_like(theta)
    from bpqnn.simulator import StatevectorSimulator, default_axes

    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)

    def _E(t):
        sim = StatevectorSimulator(n, 1)
        sim.apply_circuit(t, edges, axes)
        return estimate_ising_energy(sim.state, n, h, shots, rng,
                                     n_batches=1)[0]

    for l in range(M):
        for q in range(nq):
            pp = theta.copy()
            pm = theta.copy()
            pp[l, q] += shift
            pm[l, q] -= shift
            grad[l, q] = 0.5 * (_E(pp) - _E(pm))
    return grad