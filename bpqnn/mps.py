"""Matrix-product-state (MPS) surrogate for the simulability test (SPEC D9 / E9).

An MPS writes a state over ``n`` sites as a chain of tensors
``A[k][a, s, b]`` (left bond ``a``, physical index ``s``, right bond ``b``).
We convert a dense statevector to MPS by a **sequential Schmidt decomposition**
(SVD) and, for the surrogate test, *truncate* every bond to a maximum
dimension ``chi``. Reconstructing the truncated MPS gives an approximate
state whose energy/fidelity we compare against the target.

The point (D9): *does a small-``chi`` MPS reproduce the trained QNN's energy to
within the QNN's own error?* If yes, the trained model is classically
simulable as a low-bond-dimension MPS -- for the TFIM the answer is expected to
be yes (critical ``S ~ (1/3) log n``), and this module quantifies it.

Conventions
-----------
- Basis ordering is little-endian, matching :mod:`bpqnn.simulator`.
- Sites are the ``n`` tensor axes of ``psi.reshape((2,) * n)`` in C order, i.e.
  site ``0`` is the most-significant index (a contiguous physical ordering of
  the chain). Bond dimensions depend on this ordering; for a uniform
  nearest-neighbour chain any contiguous ordering is equivalent.
- ``chi=None`` keeps the exact ranks (an exact MPS); the maximal bond of an
  exact ``n``-qubit state is ``2**floor(n/2)``.

The truncation here is a **single-pass left-to-right** SVD compression (each
bond truncated as the sweep reaches it). It is a valid approximation but **not
the variationally optimal** rank-``chi`` MPS; energies are therefore upper
bounds on ``E0`` and the reported ``chi`` thresholds are conservative. A
two-site variational (DMRG) compression would tighten them.
"""

from __future__ import annotations

import numpy as np


def statevector_to_mps(psi, n: int, chi: int | None = None) -> list:
    """Sequential-SVD MPS; bond ``k`` truncated to ``chi`` when given.

    Returns a list of ``n`` tensors, each shape ``(d_l, 2, d_r)``.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    dim = 1 << n
    if psi.size != dim:
        raise ValueError(f"state has size {psi.size}, expected {dim}")
    if chi is not None and chi < 1:
        raise ValueError(f"chi must be >= 1, got {chi}")

    phi = psi.reshape((1,) + (2,) * n)
    tensors = []
    d = 1
    for k in range(n - 1):
        phi = phi.reshape(d * 2, -1)
        U, S, Vh = np.linalg.svd(phi, full_matrices=False)
        r = U.shape[1]
        if chi is not None and r > chi:
            r = chi
            U, S, Vh = U[:, :r], S[:r], Vh[:r, :]
        tensors.append(U.reshape(d, 2, r))
        phi = (S[:, None] * Vh).reshape((r,) + (2,) * (n - k - 1))
        d = r
    tensors.append(phi.reshape(d, 2, 1))
    return tensors


def mps_to_statevector(tensors, n: int) -> np.ndarray:
    """Contract an MPS back to a dense (little-endian) statevector."""
    acc = np.array([1.0], dtype=np.complex128)
    for A in tensors:
        A = np.asarray(A)
        acc = np.tensordot(acc, A, axes=([-1], [0]))
    return acc.reshape(1 << n)


def bond_dimensions(tensors) -> list:
    """Bond dimension after each site (the last entry is always 1)."""
    return [int(np.asarray(A).shape[2]) for A in tensors]


def max_bond_dimension(tensors) -> int:
    return max(bond_dimensions(tensors), default=1)


def schmidt_spectrum(psi, n: int) -> list:
    """Singular values at each cut ``k = 1 .. n-1`` (cut after ``k`` sites).

    Entry ``k-1`` is the descending Schmidt spectrum across the bipartition of
    the first ``k`` sites (most-significant indices) versus the rest.
    """
    psi = np.asarray(psi, dtype=np.complex128).reshape(-1)
    out = []
    for k in range(1, n):
        m = psi.reshape(1 << k, 1 << (n - k))
        out.append(np.linalg.svd(m, compute_uv=False))
    return out


def significant_rank(spectrum, tol: float = 1e-8) -> int:
    """Number of Schmidt values with squared weight ``>= tol``."""
    s2 = np.asarray(spectrum) ** 2
    return int(np.count_nonzero(s2 >= tol))


def truncation_weight(psi_ref, psi_chi) -> float:
    """Relative weight lost by the approximation: ``1 - <chi|chi>/<ref|ref>``."""
    ref = np.asarray(psi_ref).reshape(-1)
    chi = np.asarray(psi_chi).reshape(-1)
    nr = float(np.vdot(ref, ref).real)
    nc = float(np.vdot(chi, chi).real)
    if nr <= 0:
        raise ValueError("reference state has zero norm")
    return max(0.0, 1.0 - nc / nr)


def compress(psi, n: int, chi: int | None = None) -> np.ndarray:
    """Convenience: statevector -> (truncated) MPS -> normalized statevector."""
    tensors = statevector_to_mps(psi, n, chi=chi)
    out = mps_to_statevector(tensors, n)
    norm = np.linalg.norm(out)
    if norm > 0:
        out = out / norm
    return out
