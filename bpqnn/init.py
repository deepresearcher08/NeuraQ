"""Initialisation strategies (SPEC section 6).

Parameter shape is ``(M, n)`` (or ``(batch, M, n)``).

- ``random``: uniform ``[0, 2*pi)``.
- ``zero``: all zeros (baseline for the Ising task; degenerate on |0...0>).
- ``identity_block``: Grant et al. 2019 style exact identity. For the
  palindromic template ``R_1 E R_2 E ... E R_M`` (M odd) with palindromic
  axes, setting the centre layer to zero and mirroring the rest as
  ``theta[M-1-k] = -theta[k]`` makes the whole circuit telescope to the
  identity. On the |0...0> target the cost is then exactly zero (so the
  gradient -- not the cost -- carries the signal); use it on the Ising /
  random-target protocol. WARNING: on |0...0> the cost at init is 0.
"""

from __future__ import annotations

import numpy as np


def _shape(M: int, n: int, batch):
    return (M, n) if batch is None else (batch, M, n)


def random_init(M: int, n: int, rng: np.random.Generator, batch=None) -> np.ndarray:
    return rng.uniform(0.0, 2.0 * np.pi, size=_shape(M, n, batch))


def zero_init(M: int, n: int, batch=None) -> np.ndarray:
    return np.zeros(_shape(M, n, batch), dtype=np.float64)


def identity_block_init(
    M: int, n: int, rng: np.random.Generator, batch=None
) -> np.ndarray:
    """Exact identity circuit for palindromic axes.

    Requires ``M`` odd (a mirror-centre layer exists). The centre layer is
    set to zero and every layer ``k`` is mirrored by ``-theta[k]`` at
    ``M-1-k``. Assumes ``axes = default_axes(M)`` (palindromic).
    """
    if M % 2 == 0:
        raise ValueError(f"identity block requires odd M, got M={M}")
    lead = () if batch is None else (batch,)
    theta = np.empty(lead + (M, n), dtype=np.float64)
    K = M // 2
    r = rng.uniform(0.0, 2.0 * np.pi, size=lead + (K, n))
    for k in range(K):
        theta[..., k, :] = r[..., k, :]
    theta[..., K, :] = 0.0
    for k in range(K):
        theta[..., M - 1 - k, :] = -r[..., k, :]
    return theta


def perturbed_identity_init(
    M: int, n: int, rng: np.random.Generator, eps: float = 0.1, batch=None
) -> np.ndarray:
    """Identity block plus uniform noise of half-width ``eps``.

    The exact identity block leaves the first and last layers with almost no
    gradient signal at init (E2b/E20). Perturbing every angle by
    ``U(-eps, eps)`` keeps the circuit near identity while re-injecting signal.
    """
    base = identity_block_init(M, n, rng, batch)
    return base + rng.uniform(-eps, eps, size=base.shape)
