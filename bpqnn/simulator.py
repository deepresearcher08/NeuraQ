"""Batched NumPy statevector simulator.

Conventions
-----------
- Basis ordering is **little-endian**, matching Qiskit: basis index
  ``b = sum_q bit_q * 2**q``, i.e. qubit ``q`` has stride ``2**q``.
- ``state`` has shape ``(batch, 2**n)``, dtype ``complex128``.

Ansatz template
---------------
``params`` has shape ``(M, n)`` (or ``(batch, M, n)``): ``M`` rotation
layers, one angle per qubit per layer. Layer ``l`` rotates every qubit about
a **single** axis ``axes[l]`` (``"Y"`` or ``"Z"``); a fixed entangling CZ
layer ``E`` is applied *between* consecutive rotation layers (no end
entanglers)::

    R_1  E  R_2  E  ...  E  R_M

Performance notes
-----------------
- A single-axis rotation layer acts on every qubit; a ``Z`` layer is
  diagonal and applied as one contiguous phase multiply using a precomputed
  bits matrix, a ``Y`` layer as n tensor matmuls.
- The whole CZ chain is one cached diagonal sign vector (one multiply).
"""

from __future__ import annotations

import numpy as np


# -- kernels operating directly on a (batch, dim) state array -------------
def default_axes(M: int) -> list:
    """Palindromic single-axis schedule.

    For odd ``M`` this is the alternation ``Y,Z,Y,Z,...`` (already
    palindromic). For even ``M`` a period-4 ``Y,Z,Z,Y`` pattern is folded
    about the centre. Palindromicity (``axes[j] == axes[M-1-j]``) is required
    so that an angle negation at mirrored positions inverts the layer.
    """
    if M % 2 == 1:
        return ["Y" if j % 2 == 0 else "Z" for j in range(M)]
    base = ["Y", "Z", "Z", "Y"]
    axes = [base[j % 4] for j in range(M)]
    for j in range(M // 2):
        axes[M - 1 - j] = axes[j]
    return axes


def bits_matrix(n: int) -> np.ndarray:
    """(n, 2**n) float array, ``bits[q, b] = bit q of b``."""
    dim = 1 << n
    idx = np.arange(dim)
    return np.array([(idx >> q) & 1 for q in range(n)], dtype=np.float64)


def entangler_sign(n: int, edges) -> np.ndarray:
    """Diagonal (+/-1) of the product of CZ gates on ``edges``."""
    dim = 1 << n
    idx = np.arange(dim)
    sign = np.ones(dim, dtype=np.float64)
    for (a, b) in edges:
        sign *= 1.0 - 2.0 * (((idx >> a) & 1) & ((idx >> b) & 1))
    return sign


def apply_rotation_layer(state, theta, axis, n, bits) -> None:
    """In-place single-axis rotation on every qubit. ``state`` (B, dim)."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 1:
        theta = theta[None, :]
    B, dim = state.shape
    if axis == "Z":
        total = theta.sum(axis=1)                 # (B,)
        weighted = theta @ bits                    # (B, dim)
        state *= np.exp(-0.5j * total[:, None] + 1j * weighted)
    elif axis == "Y":
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        G = np.empty((B, n, 2, 2), dtype=np.complex128)
        G[:, :, 0, 0] = c
        G[:, :, 0, 1] = -s
        G[:, :, 1, 0] = s
        G[:, :, 1, 1] = c
        psi = state.reshape((B,) + (2,) * n)
        for q in range(n):
            ax = n - q                             # qubit q -> tensor axis n-q
            a = np.moveaxis(psi, ax, 1).reshape(B, 2, -1)
            a = np.matmul(G[:, q], a)
            psi = np.moveaxis(a.reshape((B, 2) + (2,) * (n - 1)), 1, ax)
        state[...] = psi.reshape(B, dim)
    else:
        raise ValueError(f"unknown axis: {axis!r}")


def apply_entangler(state, sign) -> None:
    """In-place diagonal CZ chain: ``state *= sign`` (sign is (dim,))."""
    state *= sign


def pauli_inner(lam, psi, axis, n) -> np.ndarray:
    """Per-qubit ``<lam| P_q |psi>`` for P = Y or Z. Returns (B, n)."""
    lam = np.asarray(lam)
    psi = np.asarray(psi)
    B, dim = psi.shape
    out = np.empty((B, n), dtype=np.complex128)
    for q in range(n):
        high = dim >> (q + 1)
        low = 1 << q
        L = lam.reshape(B, high, 2, low)
        P = psi.reshape(B, high, 2, low)
        if axis == "Z":
            # <lam|Z|psi> = sum_{bit=0} cL P - sum_{bit=1} cL P
            val = np.sum(np.conj(L[:, :, 0, :]) * P[:, :, 0, :], axis=(1, 2))
            val -= np.sum(np.conj(L[:, :, 1, :]) * P[:, :, 1, :], axis=(1, 2))
        elif axis == "Y":
            # Y = [[0,-i],[i,0]] -> i( sum cL1 P0 - sum cL0 P1 )
            t0 = np.sum(np.conj(L[:, :, 1, :]) * P[:, :, 0, :], axis=(1, 2))
            t1 = np.sum(np.conj(L[:, :, 0, :]) * P[:, :, 1, :], axis=(1, 2))
            val = 1j * (t0 - t1)
        else:
            raise ValueError(f"unknown axis: {axis!r}")
        out[:, q] = val
    return out


class StatevectorSimulator:
    def __init__(self, n_qubits: int, batch_size: int = 1, dtype=np.complex128):
        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.n = int(n_qubits)
        self.dim = 1 << self.n
        self.batch = int(batch_size)
        self.state = np.zeros((self.batch, self.dim), dtype=dtype)
        self.state[:, 0] = 1.0
        self._bits = bits_matrix(self.n)
        self._esign_cache: dict = {}

    def reset(self) -> "StatevectorSimulator":
        self.state[...] = 0.0
        self.state[:, 0] = 1.0
        return self

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def esign(self, edges) -> np.ndarray:
        key = tuple(edges)
        if key not in self._esign_cache:
            self._esign_cache[key] = entangler_sign(self.n, edges)
        return self._esign_cache[key]

    def apply_rotation_layer(self, theta, axis: str) -> "StatevectorSimulator":
        apply_rotation_layer(self.state, theta, axis, self.n, self._bits)
        return self

    def apply_entangler(self, edges) -> "StatevectorSimulator":
        apply_entangler(self.state, self.esign(edges))
        return self

    def apply_circuit(self, params, edges=None, axes=None) -> "StatevectorSimulator":
        """Apply the full circuit. ``params`` is (M, n) or (batch, M, n)."""
        params = np.asarray(params, dtype=np.float64)
        if params.ndim == 2:
            params = params[None, ...]
        if params.shape[0] != self.batch:
            raise ValueError(
                f"batch mismatch: params has {params.shape[0]}, simulator has {self.batch}"
            )
        M, n = params.shape[1], params.shape[2]
        if n != self.n:
            raise ValueError(f"params has {n} qubits, simulator has {self.n}")
        if axes is None:
            axes = default_axes(M)
        if len(axes) != M:
            raise ValueError(f"axes has length {len(axes)}, expected {M}")
        if edges is None:
            edges = [(q, q + 1) for q in range(self.n - 1)]
        sign = self.esign(edges)
        for l in range(M):
            self.apply_rotation_layer(params[:, l, :], axes[l])
            if l < M - 1:
                apply_entangler(self.state, sign)
        return self
