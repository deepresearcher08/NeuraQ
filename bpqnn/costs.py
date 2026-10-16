"""Cost functions.

- ``global_cost_zero`` / ``local_cost_zero``: the plateau-protocol costs with
  target |0...0> (SPEC section 5).
- ``infidelity``: global training cost for the Ising task.
- ``ising_energy`` lives in ``hamiltonian.py`` (local training cost).
"""

from __future__ import annotations

import numpy as np


def _as_batch(state):
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    return state


def zero_prob(state):
    state = _as_batch(state)
    return np.abs(state[:, 0]) ** 2


def global_cost_zero(state):
    """C_G = 1 - P(0...0)."""
    return 1.0 - zero_prob(state)


def single_qubit_zero_probs(state, n=None):
    state = _as_batch(state)
    if n is None:
        n = int(round(np.log2(state.shape[1])))
    dim = 1 << n
    batch = state.shape[0]
    out = np.empty((batch, n), dtype=np.float64)
    for q in range(n):
        high = dim >> (q + 1)
        low = 1 << q
        v = state.reshape(batch, high, 2, low)
        out[:, q] = np.sum(np.abs(v[:, :, 0, :]) ** 2, axis=(1, 2))
    return out


def local_cost_zero(state, n=None):
    """C_L = 1 - (1/n) sum_i P_i(0)."""
    return 1.0 - single_qubit_zero_probs(state, n).mean(axis=1)


def infidelity(state, target):
    """1 - |<target|psi>|^2. Target is a full statevector (dim,)."""
    state = _as_batch(state)
    target = np.asarray(target).reshape(-1)
    overlap = state @ np.conj(target)
    return 1.0 - np.abs(overlap) ** 2


# -- observables O for the adjoint method ---------------------------------
# Every cost above is C = <psi|O|psi> for a Hermitian O; the adjoint gradient
# needs lambda = O|psi>.
def global_zero_observable(state):
    """lambda = (I - |0><0|)|psi>."""
    state = _as_batch(state)
    lam = state.copy()
    lam[:, 0] = 0.0
    return lam


def local_zero_observable(state, n=None):
    """lambda = (I - (1/n) sum_i Pi_i^(0))|psi> = psi * popcount(b)/n."""
    state = _as_batch(state)
    dim = state.shape[1]
    if n is None:
        n = int(round(np.log2(dim)))
    idx = np.arange(dim)
    ones = np.zeros(dim, dtype=np.float64)
    for q in range(n):
        ones += (idx >> q) & 1
    return state * (ones / n)[None, :]


def infidelity_observable(state, target):
    """lambda = (I - |target><target|)|psi>."""
    state = _as_batch(state)
    target = np.asarray(target).reshape(-1)
    overlap = state @ np.conj(target)
    return state - overlap[:, None] * target[None, :]
