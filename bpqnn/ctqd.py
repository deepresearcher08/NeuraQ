"""CTQD: classically-trained, quantum-deployed sampling (SPEC E14).

The plateau-mitigation framework -- local costs, identity init, layerwise
training, support annealing -- never escapes the Cerezo-Larocca dictum:
provably-trainable losses live in polynomial subspaces and are classically
simulable (SPEC D9 / E8-E9 measured this for our own ansatz). E14 *inverts*
the question: keep ALL optimisation classical, and reserve the quantum role
for enumerative *sampling* at deployment ("train on classical, deploy on
quantum", Krovi et al. 2025).

The family is a parametrised IQP-style phase ansatz

    U(theta) = H^{x n}  D(theta)  H^{x n},     D(theta) = diag_x exp(i phi_theta(x)),
    phi_theta(x) = sum_g theta_g (-1)^{q_g(x)},

with ``q_g`` the parity product over the gate's support (single-qubit ``Z``
phases and ``ZZ`` phase gates on a chain graph). For this family the
computational-basis marginal correlators are classically computable *without
ever constructing the state* (the den-Nest / IQP classical-evaluability
identity):

    <Z_a>_theta = 2^{-n} sum_{t} cos( phi_theta(t xor a) - phi_theta(t) ).

The spectral MMD between model ``q_theta`` and a target ``p`` over the
low-order correlators is therefore a classical, differentiable function:

    MMD^2(q,p) = sum_{|a| <= K} w_a ( <Z_a>_q - <Z_a>_p )^2.

It is minimised in pure NumPy with ZERO quantum gradient evaluations. Only AT
DEPLOYMENT is the state built (a Walsh-Hadamard transform of ``e^{i phi}``)
and sampled. The experiment (``experiments/e14_ctqd.py``) feeds the samples
into SQD/QSCI for the critical TFIM and measures the classical-surrogate
(MPS) frontier -- see ``sampler_frontier``.

Everything in this module is a classical algorithm; it never touches
:mod:`bpqnn.simulator` or :mod:`bpqnn.grad`.

Basis ordering is little-endian, matching :mod:`bpqnn.simulator`.
"""

from __future__ import annotations

import numpy as np


# -- phase-ansatz geometry -------------------------------------------------
def chain_zz_edges(n: int) -> list:
    return [(j, j + 1) for j in range(n - 1)]


def dense_phase_signs(n: int, rng=None) -> tuple:
    """A gate set dense enough to reach the volume-law / Haar regime.

    Single-qubit ``Z`` phases, ``ZZ`` on the complete graph, and ~2n triples of
    ``ZZZ``. With ``theta ~ U(-2pi, 2pi)`` the deployed state has half-chain
    entropy within a few % of a Haar-random state (the "expensive" end of the
    E14 sampling-surrogate frontier). Deterministic for a fixed ``rng`` seed.
    """
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng() if rng is None else rng
    triples = set()
    for i in range(n):
        triples.add((i, (i + 1) % n, (i + 2) % n))
    while len(triples) < 2 * n:
        t = tuple(sorted(rng.choice(n, size=3, replace=False)))
        triples.add(t)
    return phase_signs(n, edges=edges, triples=sorted(triples))


def phase_signs(n: int, edges=None, triples=None) -> tuple:
    """Return ``(signs, labels)``.

    ``signs`` has shape ``(G, 2**n)``, entry ``signs[g, x] = (-1)^{q_g(x)}``
    with ``q_g`` the parity product on the gate's support (``Z_j`` phases,
    ``ZZ`` phase gates on ``edges``, optional ``ZZZ`` phase gates on
    ``triples``). ``labels`` names each gate.
    """
    dim = 1 << n
    idx = np.arange(dim)
    bits = np.array([(idx >> q) & 1 for q in range(n)], dtype=np.int8)
    rows = []
    labels = []
    for q in range(n):
        rows.append(1 - 2 * bits[q, :].astype(np.float64))
        labels.append(f"Z{q}")
    for (i, j) in (edges if edges is not None else chain_zz_edges(n)):
        rows.append(1 - 2 * (bits[i, :].astype(np.float64)
                             * bits[j, :].astype(np.float64)))
        labels.append(f"ZZ{i}{j}")
    for (i, j, k) in (triples or []):
        rows.append(1 - 2 * (bits[i, :].astype(np.float64)
                             * bits[j, :].astype(np.float64)
                             * bits[k, :].astype(np.float64)))
        labels.append(f"ZZZ{i}{j}{k}")
    return np.asarray(rows, dtype=np.float64), labels


def correlator_set(n: int, K: int, tau: float = 1.5) -> dict:
    """Masks a with ``|a| <= K`` and spectral weights ``w_a = exp(-|a|/tau)``.

    ``masks`` are little-endian integers (bit ``q`` of the mask selects ``Z_q``),
    ``sizes`` the Hamming weight, ``weights`` the spectral mass.
    """
    dim = 1 << n
    masks, sizes, weights = [], [], []
    for a in range(dim):
        k = a.bit_count()
        if 0 < k <= K:
            masks.append(a)
            sizes.append(k)
            weights.append(float(np.exp(-k / tau)))
    return {
        "masks": np.asarray(masks, dtype=np.int64),
        "sizes": np.asarray(sizes, dtype=np.int64),
        "weights": np.asarray(weights, dtype=np.float64),
    }


# -- classical correlator evaluation --------------------------------------
def xor_table(masks: np.ndarray, n: int) -> np.ndarray:
    """``idx_ta[a, t] = masks[a] xor t`` -- shape ``(|A|, 2**n)``."""
    t = np.arange(1 << n)
    return np.bitwise_xor(masks[:, None], t[None, :])


def phase_vector(theta, signs: np.ndarray) -> np.ndarray:
    """``phi[t] = sum_g theta_g signs[g, t]`` -- shape ``(2**n,)``."""
    return signs.T @ np.asarray(theta, dtype=np.float64)


def expectation_z(theta, signs, idx_ta) -> np.ndarray:
    """\\(\\langle Z_a\\rangle_\\theta\\) for every mask, classically.

    Uses the identity ``<Z_a> = 2^{-n} sum_t cos(phi(t xor a) - phi(t))`` --
    no statevector is built.
    """
    phi = phase_vector(theta, signs)
    inner = phi[idx_ta] - phi[None, :]
    return np.cos(inner).mean(axis=1)


def mmd_loss_and_grad(theta, signs, idx_ta, weights, target_z) -> tuple:
    """Spectral MMD loss and its analytic gradient (vectorised).

    ``loss = sum_a w_a (<Z_a>_q - target_z_a)^2``. The gradient follows from
    ``dZ_a/dtheta_g = -2^{-n} sum_t sin(phi(t^a)-phi(t)) (s_g[t^a] - s_g[t])``
    and contracts to ``grad_g = -2^{1-n} <v, s_g>`` with
    ``v[s] = sum_a (u[a, s^a] - u[a, s])`` and ``u = w_a (Z_a - t_a) sin(inner)``
    -- no loop over gates.
    """
    theta = np.asarray(theta, dtype=np.float64)
    phi = phase_vector(theta, signs)
    n = int(round(np.log2(signs.shape[1])))
    inner = phi[idx_ta] - phi[None, :]                     # (|A|, 2^n)
    zq = np.cos(inner).mean(axis=1)                        # (|A|,)
    diff = zq - np.asarray(target_z, dtype=np.float64)     # (|A|,)
    loss = float(np.sum(weights * diff ** 2))

    u = (weights * diff)[:, None] * np.sin(inner)          # (|A|, 2^n)
    rows = np.arange(u.shape[0])[:, None]
    sheared = u[rows, idx_ta].sum(axis=0)                  # sum_a u[a, s^a]
    colsum = u.sum(axis=0)                                 # sum_a u[a, s]
    v = sheared - colsum
    grad = -(2.0 ** (1 - n)) * (signs @ v)
    return loss, grad


def classical_adam_train(
    theta0,
    *,
    signs,
    idx_ta,
    weights,
    target_z,
    lr: float = 0.05,
    max_steps: int = 300,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    tol: float = 1e-12,
    rel_drop: float | None = None,
):
    """Classical Adam over the spectral MMD. ZERO quantum gradient evals.

    Returns a result dict including ``quantum_grad_evals`` (always 0) and the
    classical work in ``classical_correlator_sums`` (each of size ~2^n).
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    best = None
    for t in range(1, max_steps + 1):
        loss, grad = mmd_loss_and_grad(theta, signs, idx_ta, weights, target_z)
        history.append({"step": t, "loss": float(loss),
                        "grad_norm": float(np.linalg.norm(grad))})
        if best is None or loss < best[0]:
            best = (float(loss), theta.copy())
        if loss <= tol:
            break
        if rel_drop is not None and len(history) >= 10:
            past = history[-10]["loss"]
            if past > 0 and abs(past - loss) / past < rel_drop:
                break
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad * grad
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
    return {
        "theta": best[1],
        "theta_best": best[1].tolist(),
        "loss": best[0],
        "history": history,
        "steps": len(history),
        "quantum_grad_evals": 0,
        "classical_correlator_sums": len(history) * (len(weights) + 1),
    }


# -- deployment: build the state and sample --------------------------------
def wht(v: np.ndarray) -> np.ndarray:
    """In-place Walsh-Hadamard transform of a length-2**n vector."""
    v = np.asarray(v, dtype=np.complex128).copy()
    n = int(round(np.log2(v.size)))
    seg = 1
    while seg < v.size:
        half = seg << 1
        idx = np.arange(seg)
        for base in range(0, v.size, half):
            a = v[base + idx].copy()
            b = v[base + seg + idx].copy()
            v[base + idx] = a + b
            v[base + seg + idx] = a - b
        seg = half
    return v


def iqp_state(theta, n: int, signs) -> np.ndarray:
    """The deployed statevector ``|psi> = H^xn D(theta) H^xn |0>``.

    Amplitude ``A[x] = 2^{-n} sum_s e^{i phi(s)} (-1)^{x.s}``, evaluated with
    the (unnormalised) Walsh-Hadamard transform of ``e^{i phi}``: the raw WHT
    returns ``w[x] = sum_s (-1)^{x.s} u[s]`` with ``|w|^2 = 2^n |u|^2``, so a
    factor ``2^{-n}`` restores the unit norm.
    """
    phi = phase_vector(theta, signs)
    v = wht(np.exp(1j * phi))
    return v * (2.0 ** (-n))


def sample_from_theta(theta, n: int, signs, num_samples: int, rng) -> np.ndarray:
    """Draw ``num_samples`` computational-basis indices from the deployed model."""
    phi = phase_vector(theta, signs)
    v = wht(np.exp(1j * phi))
    p = np.abs(2.0 ** (-n) * v) ** 2
    p = p / p.sum()
    return rng.choice(p.size, size=int(num_samples), p=p)


def expectation_z_from_state(state, n: int, masks: np.ndarray) -> np.ndarray:
    """Cross-check evaluator: ``<Z_a> = sum_x (-1)^{a.x} p(x)`` from a statevector."""
    p = np.abs(np.asarray(state).reshape(-1)) ** 2
    q = np.arange(n)
    abits = (masks[:, None] >> q[None, :]) & 1                 # (|A|, n)
    xbits = (np.arange(1 << n)[:, None] >> q[None, :]) & 1     # (2^n, n)
    parity = (abits @ xbits.T) & 1                             # (|A|, 2^n)
    return ((1 - 2 * parity.astype(np.float64)) @ p).astype(np.float64)


# -- classical-surrogate frontier (D9/E14) ----------------------------------
def sampler_frontier(state, n: int) -> dict:
    """Entanglement / MPS-reactance stats for the deployed distribution."""
    from .mps import significant_rank, schmidt_spectrum, statevector_to_mps
    from .sqd import half_chain_entropy

    psi = np.asarray(state).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    spec = schmidt_spectrum(psi, n)
    tensors = statevector_to_mps(psi, n)
    bonds = [t.shape[2] for t in tensors]
    return {
        "entropy_half": float(half_chain_entropy(psi, n)),
        "significant_rank_mid": int(significant_rank(spec[(n - 1) // 2])),
        "exact_bond": int(max(bonds)),
    }