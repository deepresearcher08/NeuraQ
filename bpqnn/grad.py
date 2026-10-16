"""Gradients: exact parameter-shift, finite differences, and adjoint.

The parameter-shift and finite-difference routines accept parameters of shape
``(M, n)`` or ``(batch, M, n)``; the cost callable receives batched parameters
and returns an array of shape (batch,).

Parameter-shift for RY/RZ (generators with eigenvalues +/-1/2):
    g = 0.5 * (C(theta + pi/2) - C(theta - pi/2))

``adjoint_gradient`` implements reverse-mode (adjoint) differentiation
(Jones & Gacon, arXiv:2009.02823): one forward and one reverse sweep give the
full gradient in O(P) gates, O(1) statevectors.
"""

from __future__ import annotations

import numpy as np

from . import costs, hamiltonian
from .simulator import (
    apply_entangler,
    apply_rotation_layer,
    bits_matrix,
    default_axes,
    entangler_sign,
    pauli_inner,
)

SHIFT = np.pi / 2.0


def _as_batch(params):
    params = np.asarray(params, dtype=np.float64)
    single = params.ndim == 2
    return (params[None, ...] if single else params), single


def parameter_shift(params, cost_fn) -> np.ndarray:
    p, single = _as_batch(params)
    B, M, n = p.shape
    grad = np.empty_like(p)
    for l in range(M):
        for q in range(n):
            pp = p.copy()
            pp[:, l, q] += SHIFT
            pm = p.copy()
            pm[:, l, q] -= SHIFT
            grad[:, l, q] = 0.5 * (
                np.asarray(cost_fn(pp)) - np.asarray(cost_fn(pm))
            )
    return grad[0] if single else grad


def finite_difference(params, cost_fn, eps: float = 1e-6) -> np.ndarray:
    p, single = _as_batch(params)
    B, M, n = p.shape
    grad = np.empty_like(p)
    for l in range(M):
        for q in range(n):
            pp = p.copy()
            pp[:, l, q] += eps
            pm = p.copy()
            pm[:, l, q] -= eps
            grad[:, l, q] = (
                np.asarray(cost_fn(pp)) - np.asarray(cost_fn(pm))
            ) / (2.0 * eps)
    return grad[0] if single else grad


def max_abs_diff(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _observable_apply(state, kind, n, h, periodic, target):
    """lambda = O|state> for the cost ``kind`` (Hermitian O)."""
    if kind == "global_zero":
        return costs.global_zero_observable(state)
    if kind == "local_zero":
        return costs.local_zero_observable(state, n)
    if kind == "infidelity":
        if target is None:
            raise ValueError("infidelity cost requires a target statevector")
        return costs.infidelity_observable(state, target)
    if kind == "ising":
        return hamiltonian.ising_hamiltonian_apply(state, n, h=h, periodic=periodic)
    raise ValueError(f"unknown cost kind: {kind!r}")


def adjoint_gradient(
    params,
    n: int,
    edges=None,
    axes=None,
    kind: str = "global_zero",
    h: float = 1.0,
    periodic: bool = False,
    target=None,
    observable=None,
) -> np.ndarray:
    """Reverse-mode gradient of ``C = <psi|O|psi>`` w.r.t. every angle.

    ``params`` is ``(M, n)`` or ``(batch, M, n)``; the return has the same
    shape. ``kind`` selects the observable: ``"global_zero"``, ``"local_zero"``,
    ``"infidelity"`` (needs ``target``), or ``"ising"``. Pass ``observable``,
    a callable ``state -> O|state>`` (state is ``(B, dim)``), to use a custom
    Hermitian observable instead of ``kind``.
    """
    params = np.asarray(params, dtype=np.float64)
    single = params.ndim == 2
    p = params[None, ...] if single else params
    B, M, nq = p.shape
    dim = 1 << n
    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)
    bits = bits_matrix(n)
    sign = entangler_sign(n, edges)

    psi = np.zeros((B, dim), dtype=np.complex128)
    psi[:, 0] = 1.0
    for l in range(M):
        apply_rotation_layer(psi, p[:, l, :], axes[l], n, bits)
        if l < M - 1:
            apply_entangler(psi, sign)

    lam = (
        observable(psi)
        if observable is not None
        else _observable_apply(psi, kind, n, h, periodic, target)
    )

    grad = np.empty_like(p)
    for l in range(M - 1, -1, -1):
        grad[:, l, :] = pauli_inner(lam, psi, axes[l], n).imag
        apply_rotation_layer(psi, -p[:, l, :], axes[l], n, bits)
        apply_rotation_layer(lam, -p[:, l, :], axes[l], n, bits)
        if l > 0:
            apply_entangler(psi, sign)
            apply_entangler(lam, sign)
    return grad[0] if single else grad
