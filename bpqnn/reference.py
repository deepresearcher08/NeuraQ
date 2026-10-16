"""2-design (Haar) gradient-variance reference for the global |0...0> cost.

The universal barren-plateau benchmark: the derivative of ``C = 1 - |psi_0|^2``
with respect to a parameter whose generator ``V = P_q/2`` sits between two
independent Haar-random unitaries (interior position, so both sides form
2-designs). Its variance is a function of ``n`` only.

Rather than a dense Haar average, we sample it exactly with a projection
trick that scales to n = 12 and beyond:

    |psi(th)> = U_+ e^{-i th V} U_- |0>,      V = P_q / 2
    dC/dth    = 2 Im <a| O |b>,
        a = U_+ phi,  b = U_+ (V phi),  phi = U_- |0>,

where ``(a, a_perp)`` is a Haar-random orthonormal pair and
``V phi = alpha phi + beta phi_perp`` gives ``b = alpha a + beta a_perp``.
This needs no dense unitaries (validated against dense Haar for n <= 6).
"""

from __future__ import annotations

import numpy as np


def haar_vectors(n_samples: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n_samples, dim)) + 1j * rng.normal(size=(n_samples, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _pauli_apply(v: np.ndarray, pauli: str, q: int, n: int) -> np.ndarray:
    nsamp, dim = v.shape
    high = dim >> (q + 1)
    low = 1 << q
    a = v.reshape(nsamp, high, 2, low)
    out = np.empty_like(a)
    if pauli == "X":
        out[:, :, 0, :] = a[:, :, 1, :]
        out[:, :, 1, :] = a[:, :, 0, :]
    elif pauli == "Y":
        out[:, :, 0, :] = -1j * a[:, :, 1, :]
        out[:, :, 1, :] = 1j * a[:, :, 0, :]
    elif pauli == "Z":
        out[:, :, 0, :] = a[:, :, 0, :]
        out[:, :, 1, :] = -a[:, :, 1, :]
    else:
        raise ValueError(f"unknown pauli: {pauli!r}")
    return out.reshape(nsamp, dim)


def two_design_var(
    n: int,
    pauli: str = "Y",
    q: int = 0,
    n_samples: int = 200000,
    rng: np.random.Generator | None = None,
    chunk: int = 8192,
):
    """Empirical variance (and s.e.) of the 2-design gradient for cost C_G."""
    rng = np.random.default_rng() if rng is None else rng
    dim = 1 << n
    s = 0.0
    s2 = 0.0
    done = 0
    while done < n_samples:
        m = min(chunk, n_samples - done)
        phi = haar_vectors(m, dim, rng)
        a = haar_vectors(m, dim, rng)
        g = rng.normal(size=(m, dim)) + 1j * rng.normal(size=(m, dim))
        g -= np.sum(np.conj(a) * g, axis=1, keepdims=True) * a
        a_perp = g / np.linalg.norm(g, axis=1, keepdims=True)

        v_phi = _pauli_apply(phi, pauli, q, n) / 2.0
        alpha = np.sum(np.conj(phi) * v_phi, axis=1).real
        w_minus = v_phi - alpha[:, None] * phi
        beta = np.linalg.norm(w_minus, axis=1)
        phi_perp = w_minus / beta[:, None]

        b = alpha[:, None] * a + beta[:, None] * a_perp
        z = np.sum(np.conj(a) * b, axis=1) - np.conj(a[:, 0]) * b[:, 0]
        x = 2.0 * np.imag(z)
        s += float(x.sum())
        s2 += float((x * x).sum())
        done += m
    mean = s / n_samples
    var = s2 / n_samples - mean * mean
    se = float(np.sqrt(max(var, 0.0) / n_samples))
    return float(var), float(mean), se


def two_design_curve(ns, **kwargs):
    """Array of 2-design variances for a list of ``n`` (fresh RNG each)."""
    return np.array([two_design_var(int(n), **kwargs)[0] for n in ns])
