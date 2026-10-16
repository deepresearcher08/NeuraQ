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

:func:`sqd_metrics` bundles the energy error, subspace size, and half-chain
entropy of a sampled run for the surrogate benchmark.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import ising_hamiltonian
from .mps import half_chain_entropy


def state_probabilities(state, n: int) -> np.ndarray:
    """Normalized |psi|^2 probability vector over the 2**n basis states."""
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
    """Relative energy error ``(E_min - E0) / (E1 - E0)``.

    Returns ``inf`` when the reference gap ``E1 - E0`` is non-positive, avoiding
    a division by zero for degenerate reference spectra.
    """
    if E1 - E0 <= 0:
        return float("inf")
    return (E_min - E0) / (E1 - E0)


def ising_dense(n: int, h: float, periodic: bool = False) -> np.ndarray:
    return ising_hamiltonian(n, h=h, periodic=periodic)


def sqd_metrics(state_sampled, n, H, E0, E1, num_samples, rng, include_flips=True):
    """Sample, diagonalise the QSCI subspace, and collect surrogate metrics.

    Draws ``num_samples`` bitstrings from ``state_sampled``, builds the QSCI
    subspace (sampled states plus, by default, all single-bit flips), finds the
    subspace ground energy, and returns the energy error, number of unique
    configurations used, and the half-chain entropy of the embedded vector.
    """
    counts = sample_counts(state_sampled, n, num_samples, rng)
    idx = qsci_indices(sampled_indices(counts), n, include_flips=include_flips)
    E_min, vsub, _ = sqd_ground_energy(H, idx)
    psi = embed_subspace_vector(vsub, idx, 1 << n)
    psi = psi / np.linalg.norm(psi)
    return {
        "E_min": E_min,
        "rel_err": rel_energy_error(E_min, E0, E1),
        "unique_configs": int(len(idx)),
        "S": half_chain_entropy(psi, n),
    }
