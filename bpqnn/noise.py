"""Batched density-matrix simulator with per-qubit depolarising noise (SPEC E7).

Same circuit template as ``simulator.StatevectorSimulator``
(``R_1 E R_2 E ... E R_M``, one single axis per layer, CZ chain between layers)
but evolved on mixed states, so **local** noise can be applied and the
noise-induced barren plateau (NIBP) measured.

Noise model
-----------
Per-qubit depolarising applied after every rotation layer::

    E(rho) = (1 - p) rho + p * I_q/2 (x) Tr_q(rho)

i.e. qubit ``q`` is replaced by the maximally mixed state with probability
``p`` (equivalently each of ``I, X, Y, Z`` is applied with probability
``p/4``). This is exactly Qiskit's ``depolarizing_error(p, 1)`` convention, so
``p = 0`` is noiseless and the simulator is cross-validated against
``qiskit-aer``.

Representation: ``rho`` has shape ``(batch, 2**n, 2**n)``, ``complex128``,
little-endian basis (qubit ``q`` has stride ``2**q``).
"""

from __future__ import annotations

import numpy as np

from .simulator import default_axes, entangler_sign


def _rot_mats(theta, axis, n):
    """Per-qubit rotation matrices, list of (B, 2, 2) for q = 0..n-1."""
    B = theta.shape[0]
    mats = []
    for q in range(n):
        th = theta[:, q]
        R = np.zeros((B, 2, 2), dtype=np.complex128)
        if axis == "Y":
            c = np.cos(th / 2.0)
            s = np.sin(th / 2.0)
            R[:, 0, 0] = c
            R[:, 0, 1] = -s
            R[:, 1, 0] = s
            R[:, 1, 1] = c
        elif axis == "Z":
            R[:, 0, 0] = np.exp(-0.5j * th)
            R[:, 1, 1] = np.exp(0.5j * th)
        else:
            raise ValueError(f"unknown axis: {axis!r}")
        mats.append(R)
    return mats


def _ket(rho, R, q, n):
    """Left-multiply by R on qubit q: rho -> R rho. rho (B, d, d)."""
    B = rho.shape[0]
    d = 1 << n
    high = d >> (q + 1)
    low = 1 << q
    v = rho.reshape(B, high, 2, low, d)
    v = np.einsum("bij,bhjld->bhild", R, v, optimize=True)
    return v.reshape(B, d, d)


def _bra(rho, R, q, n):
    """Right-multiply by R-dagger on qubit q: rho -> rho R^dag."""
    B = rho.shape[0]
    d = 1 << n
    high = d >> (q + 1)
    low = 1 << q
    v = rho.reshape(B, d, high, 2, low)
    v = np.einsum("bij,bdhjl->bdhil", np.conj(R), v, optimize=True)
    return v.reshape(B, d, d)


def _apply_gate(rho, R, q, n):
    """rho -> R rho R^dag (R is (B, 2, 2))."""
    return _bra(_ket(rho, R, q, n), R, q, n)


def apply_rotation_layer(rho, theta, axis, n):
    mats = _rot_mats(theta, axis, n)
    for q in range(n):
        rho = _apply_gate(rho, mats[q], q, n)
    return rho


def apply_entangler(rho, sign):
    outer = np.outer(sign, sign)
    return rho * outer[None, :, :]


def _partial_trace_qubit(rho, q, n):
    """Tr_q(rho) as an operator on the other n-1 qubits. Returns (B, d/2, d/2)."""
    B = rho.shape[0]
    d = 1 << n
    high = d >> (q + 1)
    low = 1 << q
    v = rho.reshape(B, high, 2, low, high, 2, low)
    return np.einsum("bhilBiL->bhlBL", v, optimize=True)


def _embed_identity(red, q, n):
    """Embed an (n-1)-qubit operator as I_q tensor red. Returns (B, d, d)."""
    B = red.shape[0]
    d = 1 << n
    high = d >> (q + 1)
    low = 1 << q
    red = red.reshape(B, high, low, high, low)
    out = np.zeros((B, high, 2, low, high, 2, low), dtype=np.complex128)
    for j in (0, 1):
        out[:, :, j, :, :, j, :] = red
    return out.reshape(B, d, d)


def depolarise(rho, p, n):
    """Per-qubit depolarising channel: ``E(rho) = (1-p) rho + p * I_q/2 (x) Tr_q(rho)``.

    This is Qiskit's ``depolarizing_error(p, 1)`` convention (equivalently each
    Pauli -- including the identity -- is applied with probability ``p/4``); it
    is cross-validated against ``qiskit-aer`` in ``tests/test_e7.py``. Uses
    ``sum_{P in X,Y,Z} P rho P = 2 I_q (x) Tr_q(rho) - rho`` so each qubit needs
    only one partial trace and one embedding.
    """
    if p <= 0.0:
        return rho
    for q in range(n):
        red = _partial_trace_qubit(rho, q, n)
        rho = (1.0 - p) * rho + 0.5 * p * _embed_identity(red, q, n)
    return rho


def run_circuit(thetas, n, edges=None, axes=None, p=0.0):
    """Apply the noisy circuit to ``|0...0>``. ``thetas`` (B, M, n) or (M, n)."""
    thetas = np.asarray(thetas, dtype=np.float64)
    if thetas.ndim == 2:
        thetas = thetas[None, ...]
    B, M, nq = thetas.shape
    if nq != n:
        raise ValueError(f"params has {nq} qubits, expected {n}")
    d = 1 << n
    rho = np.zeros((B, d, d), dtype=np.complex128)
    rho[:, 0, 0] = 1.0
    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)
    sign = entangler_sign(n, edges)
    for l in range(M):
        rho = apply_rotation_layer(rho, thetas[:, l, :], axes[l], n)
        if p > 0.0:
            rho = depolarise(rho, p, n)
        if l < M - 1:
            rho = apply_entangler(rho, sign)
    return rho


def single_qubit_zero_probs(rho, n):
    """P(qubit q measured 0) for every q, batched. Returns (B, n)."""
    B = rho.shape[0]
    d = 1 << n
    diag = np.real(np.einsum("bii->bi", rho))
    out = np.empty((B, n), dtype=np.float64)
    for q in range(n):
        high = d >> (q + 1)
        low = 1 << q
        v = diag.reshape(B, high, 2, low)
        out[:, q] = v[:, :, 0, :].sum(axis=(1, 2))
    return out


def local_cost_zero(rho, n):
    """C_L = 1 - (1/n) sum_i P_i(0), batched. Returns (B,)."""
    return 1.0 - single_qubit_zero_probs(rho, n).mean(axis=1)


def global_cost_zero(rho, n):
    """C_G = 1 - P(0...0), batched. Returns (B,)."""
    return 1.0 - np.real(rho[:, 0, 0])


def cost_value(rho, n, kind="local_zero"):
    if kind == "local_zero":
        return local_cost_zero(rho, n)
    if kind == "global_zero":
        return global_cost_zero(rho, n)
    raise ValueError(f"unknown cost kind: {kind!r}")


def parameter_shift_grads(theta, n, *, edges=None, axes=None, p=0.0,
                          kind="local_zero", shift=np.pi / 2.0):
    """Exact parameter-shift gradients of a noisy cost. ``theta`` (B, M, n)."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim == 2:
        theta = theta[None, ...]
    B, M, nq = theta.shape

    def cost(th):
        return cost_value(run_circuit(th, n, edges, axes, p), n, kind)

    grad = np.empty((B, M, nq), dtype=np.float64)
    for l in range(M):
        for q in range(nq):
            tp = theta.copy()
            tp[:, l, q] += shift
            tm = theta.copy()
            tm[:, l, q] -= shift
            grad[:, l, q] = 0.5 * (cost(tp) - cost(tm))
    return grad
