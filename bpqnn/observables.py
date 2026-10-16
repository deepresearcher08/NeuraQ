"""Support-radius cost family (E11).

The E1/E2 story is binary: a global cost plateaus, a local cost does not. Here
we make the "locality" a *tunable* knob. For target ``|0...0>`` define

    C_r = 1 - (1/W) sum_w P(qubits in window w are all measured 0),

where ``w`` ranges over the ``W = n - r + 1`` contiguous length-``r`` windows of
the ``n``-qubit chain. The two endpoints reproduce the earlier costs exactly:

    r = 1  ->  C_1 = 1 - (1/n) sum_i P_i(0)   (the E2a "local" cost)
    r = n  ->  C_n = 1 - P(0...0)             (the E1 "global" cost)

so sweeping ``r`` interpolates continuously between the local and global
regimes. The associated Hermitian observable is
``O_r = I - (1/W) sum_w P_w`` with ``P_w = |0..0>_w <0..0|_w``.
"""

from __future__ import annotations

import numpy as np

_MASK_CACHE: dict = {}


def _as_batch(state):
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    return state


def _dim_to_n(dim: int) -> int:
    return int(round(np.log2(dim)))


def subset_all_zero_mask(n: int, subset) -> np.ndarray:
    """Boolean mask (dim,) selecting basis states with every ``q in subset`` 0."""
    key = (n, tuple(sorted(subset)))
    if key not in _MASK_CACHE:
        idx = np.arange(1 << n)
        m = np.ones(1 << n, dtype=bool)
        for q in subset:
            m &= ((idx >> q) & 1) == 0
        _MASK_CACHE[key] = m
    return _MASK_CACHE[key]


def window_masks(n: int, r: int):
    """Contiguous length-``r`` window masks; ``r = n`` gives a single mask."""
    if not 1 <= r <= n:
        raise ValueError(f"r must be in 1..{n}, got {r}")
    return [subset_all_zero_mask(n, range(s, s + r)) for s in range(n - r + 1)]


def support_radius_cost(state, r: int, n: int | None = None) -> np.ndarray:
    """``C_r`` for a batched state. Returns shape (B,)."""
    state = _as_batch(state)
    if n is None:
        n = _dim_to_n(state.shape[1])
    masks = window_masks(n, r)
    acc = np.zeros(state.shape[0], dtype=np.float64)
    for m in masks:
        acc += np.sum(np.abs(state[:, m]) ** 2, axis=1)
    return 1.0 - acc / len(masks)


def block_observable(state, r: int, n: int | None = None, start: int | None = None) -> np.ndarray:
    """``lambda = O_b |psi>`` for a *single* contiguous block of ``r`` qubits.

    ``O_b = I - P_b`` with ``P_b = |0..0>_b <0..0|_b`` (no window averaging, so
    ``r`` is exactly the observable's support). Default block is centred.
    """
    state = _as_batch(state)
    if n is None:
        n = _dim_to_n(state.shape[1])
    if start is None:
        start = (n - r) // 2
    m = subset_all_zero_mask(n, range(start, start + r))
    return state - state * m[None, :]


def block_cost(state, r: int, n: int | None = None, start: int | None = None) -> np.ndarray:
    """``C_b = 1 - P(block of r qubits all 0)``. Returns shape (B,)."""
    state = _as_batch(state)
    if n is None:
        n = _dim_to_n(state.shape[1])
    if start is None:
        start = (n - r) // 2
    m = subset_all_zero_mask(n, range(start, start + r))
    return 1.0 - np.sum(np.abs(state[:, m]) ** 2, axis=1)


def support_radius_observable(state, r: int, n: int | None = None) -> np.ndarray:
    """``lambda = O_r |psi>`` for a batched state, shape (B, dim)."""
    state = _as_batch(state)
    if n is None:
        n = _dim_to_n(state.shape[1])
    masks = window_masks(n, r)
    proj = np.zeros_like(state)
    for m in masks:
        proj += state * m[None, :]
    return state - proj / len(masks)
