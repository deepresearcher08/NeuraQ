"""Ansatz metadata: parameter shapes, entangling topologies, Qiskit export.

Template (see ``simulator``): ``params`` has shape ``(M, n)``, one angle per
qubit per single-axis rotation layer, with the CZ edge layer applied between
consecutive layers: ``R_1 E R_2 E ... E R_M`` (no end entanglers).
"""

from __future__ import annotations

import numpy as np

from .simulator import default_axes


def param_shape(M: int, n: int) -> tuple:
    return (M, n)


def num_params(M: int, n: int) -> int:
    return M * n


def make_edges(n: int, topology: str = "line") -> list:
    """Entangling CZ edge list for a topology (unweighted, q < q' pairs)."""
    if topology == "line":
        return [(q, q + 1) for q in range(n - 1)]
    if topology == "ring":
        edges = [(q, q + 1) for q in range(n - 1)]
        if n > 2:
            edges.append((n - 1, 0))
        return edges
    if topology == "star":
        return [(0, q) for q in range(1, n)]
    if topology == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown topology: {topology!r}")


def to_qiskit(params, edges=None, axes=None):
    """Build the equivalent single-circuit Qiskit ``QuantumCircuit``.

    Intended for display and cross-validation (batch size must be 1).
    """
    from qiskit import QuantumCircuit

    params = np.asarray(params)
    if params.ndim == 2:
        params = params[None, ...]
    if params.shape[0] != 1:
        raise ValueError("to_qiskit handles a single circuit (batch == 1)")
    _, M, n = params.shape
    if edges is None:
        edges = make_edges(n, "line")
    if axes is None:
        axes = default_axes(M)
    qc = QuantumCircuit(n)
    for l in range(M):
        for q in range(n):
            angle = float(params[0, l, q])
            if axes[l] == "Y":
                qc.ry(angle, q)
            elif axes[l] == "Z":
                qc.rz(angle, q)
            else:
                raise ValueError(f"unknown axis: {axes[l]!r}")
        if l < M - 1:
            for (a, b) in edges:
                qc.cz(a, b)
    return qc
