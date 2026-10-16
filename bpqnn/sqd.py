"""SQD / QSCI: sample-based quantum diagonalisation (SPEC E8).

Sample-based quantum diagonalisation (SQD) and quantum-selected CI (QSCI)
turn a set of computational-basis bitstrings -- typically sampled from a
quantum circuit -- into a classical subspace: build the space spanned by the
sampled configurations and (in QSCI) the configurations connected to them by
the Hamiltonian, project ``H`` into it, and diagonalise.

For the transverse-field Ising chain
``H = -sum_i Z_i Z_{i+1} - h sum_i X_i`` the Hamiltonian couples a bitstring to
itself (diagonal ``ZZ``) and to every single-bit flip (the ``X`` terms), so the
connected space is the sampled set plus all single-bit flips.

The point of interest for this study is *simulability* (SPEC D9): if a small
classically-diagonalised subspace reproduces the energy, then the quantum
device is providing samples into a cheap classical post-processing (a QESIM
style algorithm), and gradient-free SQD can sidestep the optimisation plateau
entirely.

Basis ordering is little-endian, matching :mod:`bpqnn.simulator`.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import ising_hamiltonian


def state_probabilities(state, n: int) -> np.ndarray:
    psi = np.asarray(state).reshape(-1)
    p = np.abs(psi) ** 2
    s = p.sum()
    if s <= 0:
        raise ValueError("state has zero norm")
    return p / s


def sample_pool(state, n: int, num_samples: int, rng) -> np.ndarray:
    """Draw ``num_samples`` computational-basis indices from |psi|^2."""
    p = state_probabilities(state, n)
    return rng.choice(p.size, size=int(num_samples), p=p)


def sample_counts(state, n: int, num_samples: int, rng) -> np.ndarray:
    """Multinomial counts over the 2**n basis states."""
    p = state_probabilities(state, n)
    return rng.multinomial(int(num_samples), p)


def sampled_indices(counts: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(counts) > 0)


def qsci_indices(indices, n: int, include_flips: bool = True) -> np.ndarray:
    """QSCI subspace: sampled states plus (optionally) all single-bit flips."""
    base = np.unique(np.asarray(indices, dtype=np.int64))
    if not include_flips:
        return base
    dim = 1 << n
    extra = np.empty((base.size, n), dtype=np.int64)
    for i in range(n):
        extra[:, i] = base ^ (1 << i)
    return np.unique(np.concatenate([base, extra.reshape(-1)]).astype(np.int64))


def sqd_ground_energy(H: np.ndarray, indices: np.ndarray):
    """Lowest eigenpair of ``H`` restricted to ``indices``.

    Returns ``(E_min, vec_sub, eigvals)`` where ``vec_sub`` is in the subspace.
    """
    idx = np.asarray(indices, dtype=np.int64)
    Hsub = H[np.ix_(idx, idx)]
    w, v = np.linalg.eigh(Hsub)
    return float(w[0]), v[:, 0].astype(np.complex128), w


def embed_subspace_vector(vec_sub, indices, dim: int) -> np.ndarray:
    psi = np.zeros(dim, dtype=np.complex128)
    psi[np.asarray(indices, dtype=np.int64)] = vec_sub
    return psi


def rel_energy_error(E_min: float, E0: float, E1: float) -> float:
    return (E_min - E0) / (E1 - E0)


def half_chain_entropy(state, n: int) -> float:
    """Von Neumann entropy (base 2) of the |0..n/2> / |n/2+1..n> cut."""
    psi = np.asarray(state).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    nA = n // 2
    m = psi.reshape(1 << nA, 1 << (n - nA))
    s = np.linalg.svd(m, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def ising_dense(n: int, h: float, periodic: bool = False) -> np.ndarray:
    return ising_hamiltonian(n, h=h, periodic=periodic)
