"""Transverse-field Ising chain.

H = -sum_i Z_i Z_{i+1} - h sum_i X_i   (open chain by default)

Ground state by dense diagonalisation (fine for n <= ~12).
"""

from __future__ import annotations

import numpy as np


def ising_hamiltonian(n: int, h: float = 1.0, periodic: bool = False) -> np.ndarray:
    dim = 1 << n
    idx = np.arange(dim)
    diag = np.zeros(dim, dtype=np.float64)
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    for (i, j) in bonds:
        diag -= (1 - 2 * ((idx >> i) & 1)) * (1 - 2 * ((idx >> j) & 1))
    H = np.diag(diag).astype(np.complex128)
    for i in range(n):
        flipped = idx ^ (1 << i)
        H[idx, flipped] -= h
    return H


def ising_ground_state(n: int, h: float = 1.0, periodic: bool = False):
    """Return (E0, ground_state, all_eigenvalues)."""
    H = ising_hamiltonian(n, h=h, periodic=periodic)
    w, v = np.linalg.eigh(H)
    return float(w[0]), v[:, 0].astype(np.complex128), w


def _x_expectation(state: np.ndarray, q: int, n: int) -> np.ndarray:
    dim = 1 << n
    high = dim >> (q + 1)
    low = 1 << q
    v = state.reshape(state.shape[0], high, 2, low)
    return 2.0 * np.real(np.sum(np.conj(v[:, :, 0, :]) * v[:, :, 1, :], axis=(1, 2)))


def ising_energy(state, n: int, h: float = 1.0, periodic: bool = False) -> np.ndarray:
    """<H> computed from a batched statevector (no dense operator needed)."""
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    dim = 1 << n
    idx = np.arange(dim)
    prob = np.abs(state) ** 2
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    E = np.zeros(state.shape[0], dtype=np.float64)
    for (i, j) in bonds:
        si = 1 - 2 * ((idx >> i) & 1)
        sj = 1 - 2 * ((idx >> j) & 1)
        E -= prob @ (si * sj)
    for i in range(n):
        E -= h * _x_expectation(state, i, n)
    return E


def ising_hamiltonian_apply(state, n: int, h: float = 1.0, periodic: bool = False):
    """Batched H|psi> without forming the dense operator (for the adjoint)."""
    state = np.asarray(state, dtype=np.complex128)
    if state.ndim == 1:
        state = state[None, :]
    B, dim = state.shape
    idx = np.arange(dim)
    out = np.zeros_like(state)
    bonds = [(i, i + 1) for i in range(n - 1)]
    if periodic and n > 2:
        bonds.append((n - 1, 0))
    for (i, j) in bonds:
        si = 1 - 2 * ((idx >> i) & 1)
        sj = 1 - 2 * ((idx >> j) & 1)
        out -= state * (si * sj)[None, :]
    for i in range(n):
        high = dim >> (i + 1)
        low = 1 << i
        v = state.reshape(B, high, 2, low)
        w = np.concatenate([v[:, :, 1, :], v[:, :, 0, :]], axis=2)
        out -= h * w.reshape(B, dim)
    return out
