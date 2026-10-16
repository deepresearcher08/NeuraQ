"""Batched QNN binary classifier on the single-axis ansatz (SPEC E6, Core Task 3).

Pipeline
--------
1. **Feature map:** angle-encode ``d`` features into the first ``d`` qubits with
   ``RY(pi * x_i)`` on the reference ``|0...0>`` (a product state).
2. **Ansatz:** the same single-axis HEA as everywhere else (params ``(M, n)``,
   palindromic axes, CZ chain between layers).
3. **Readout decides the cost locality:**
   - ``"global"``: parity of all qubits ``P = Z_0 Z_1 ... Z_{n-1}`` (acts on
     every qubit -> expected barren plateau),
   - ``"local"``:  single-qubit ``Z_q`` (few-qubit observable -> trainable).

Prediction ``p = <O>`` in ``[-1, 1]``; labels ``y in {-1, +1}``; loss is the
mean squared error ``(1/S) sum_s (y_s - p_s)^2``. The gradient of the loss
w.r.t. every angle is exact by parameter shift (the observable is Hermitian and
the feature map is parameter-independent).
"""

from __future__ import annotations

import numpy as np

from .simulator import (
    StatevectorSimulator,
    apply_entangler,
    apply_rotation_layer,
    bits_matrix,
    default_axes,
    entangler_sign,
    pauli_inner,
)

SHIFT = np.pi / 2.0


def xor_grid(vals=(0.0, 1.0 / 3, 2.0 / 3, 1.0)):
    """2-feature checkerboard (XOR) grid: label = sign(x0>0.5) XOR sign(x1>0.5)."""
    X = np.array([[a, b] for a in vals for b in vals], dtype=np.float64)
    y = np.where((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5), 1.0, -1.0)
    return X, y


def additive_grid(vals=(0.0, 1.0 / 3, 2.0 / 3, 1.0)):
    """2-feature additive grid: label = sign(x0 + x1 - 1) (a diagonal boundary)."""
    X = np.array([[a, b] for a in vals for b in vals], dtype=np.float64)
    y = np.where(X[:, 0] + X[:, 1] > 1.0, 1.0, -1.0)
    return X, y


_LABELS = {
    "additive": lambda X: np.where(X[:, 0] + X[:, 1] > 1.0, 1.0, -1.0),
    "parity": lambda X: np.where((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5), 1.0, -1.0),
}


def feature_grid(vals=(0.0, 1.0 / 3, 2.0 / 3, 1.0)) -> np.ndarray:
    """All pairs over ``vals`` as a 2-feature grid."""
    return np.array([[a, b] for a in vals for b in vals], dtype=np.float64)


def make_dataset(task: str, vals=(0.0, 1.0 / 3, 2.0 / 3, 1.0)):
    """Grid features and labels for ``"additive"`` or ``"parity"``."""
    if task not in _LABELS:
        raise ValueError(f"unknown task {task!r}; pick from {tuple(_LABELS)}")
    X = feature_grid(vals)
    return X, _LABELS[task](X)


def held_out_grid(n_vals: int = 20, lo: float = 0.025, hi: float = 0.975) -> tuple:
    """A denser grid offset from the training knots (a genuine held-out set)."""
    return tuple(np.linspace(lo, hi, n_vals))


def train_test_datasets(task: str, test_vals=None):
    """Train on the 4x4 knot grid, test on an offset denser grid."""
    if test_vals is None:
        test_vals = held_out_grid()
    Xtr, ytr = make_dataset(task)
    Xte, yte = make_dataset(task, test_vals)
    return Xtr, ytr, Xte, yte


def encode_states(X: np.ndarray, n: int) -> np.ndarray:
    """Angle-encode features into qubits 0..d-1 of ``|0...0>``. Returns (S, 2**n)."""
    X = np.asarray(X, dtype=np.float64)
    S, d = X.shape
    if d > n:
        raise ValueError(f"{d} features do not fit on {n} qubits")
    dim = 1 << n
    psi = np.zeros((S, dim), dtype=np.complex128)
    psi[:, 0] = 1.0
    angles = np.zeros((S, n), dtype=np.float64)
    angles[:, :d] = np.pi * X
    apply_rotation_layer(psi, angles, "Y", n, bits_matrix(n))
    return psi


def parity_expectation(state: np.ndarray, n: int) -> np.ndarray:
    """<Z_0 Z_1 ... Z_{n-1}> from a batched statevector."""
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    idx = np.arange(1 << n)
    bits = ((idx[:, None] >> np.arange(n)) & 1).sum(axis=1)
    parity = (1.0 - 2.0 * (bits % 2)).astype(np.float64)
    return np.abs(state) ** 2 @ parity


def z_expectation(state: np.ndarray, q: int, n: int) -> np.ndarray:
    """<Z_q> from a batched statevector."""
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[None, :]
    idx = np.arange(1 << n)
    s = (1.0 - 2.0 * ((idx >> q) & 1)).astype(np.float64)
    return np.abs(state) ** 2 @ s


def expected_value(state, n, readout: str, qubits=None) -> np.ndarray:
    if readout == "global":
        return parity_expectation(state, n)
    if readout == "local":
        qubits = [0] if qubits is None else list(qubits)
        zs = np.stack([z_expectation(state, q, n) for q in qubits], axis=-1)
        return zs.mean(axis=-1)
    raise ValueError(f"unknown readout: {readout!r}")


class QNNClassifier:
    """Binary QNN classifier with exact parameter-shift gradients."""

    def __init__(self, n, M, X, y, readout="local", qubits=None, axes=None, edges=None):
        self.n = int(n)
        self.M = int(M)
        self.X = np.asarray(X, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.readout = readout
        self.qubits = [0] if qubits is None else list(qubits)
        self.edges = edges if edges is not None else [(i, i + 1) for i in range(n - 1)]
        self.axes = axes if axes is not None else default_axes(M)
        self.encoded = encode_states(self.X, self.n)
        self.S = self.X.shape[0]
        self._bits = bits_matrix(self.n)
        self._esign = entangler_sign(self.n, self.edges)

    def _expectations_encoded(self, thetas: np.ndarray, encoded) -> np.ndarray:
        """thetas (K, M, n) against encoded (S, dim) -> (K, S) predictions."""
        thetas = np.asarray(thetas, dtype=np.float64)
        if thetas.ndim == 2:
            thetas = thetas[None, ...]
        K, S = thetas.shape[0], encoded.shape[0]
        tb = np.broadcast_to(
            thetas[:, None, :, :], (K, S, self.M, self.n)
        ).reshape(K * S, self.M, self.n)
        sim = StatevectorSimulator(self.n, K * S)
        sim.state[:] = np.tile(encoded, (K, 1))
        sim.apply_circuit(tb, self.edges, self.axes)
        return expected_value(
            sim.state, self.n, self.readout, self.qubits
        ).reshape(K, S)

    def _expectations(self, thetas: np.ndarray) -> np.ndarray:
        """thetas (K, M, n) -> (K, S) predictions on the training features."""
        return self._expectations_encoded(thetas, self.encoded)

    def predict(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 3:
            theta = theta[0]
        return self._expectations(theta[None, ...])[0]

    def loss(self, theta: np.ndarray) -> float:
        pred = self.predict(theta)
        return float(np.mean((self.y - pred) ** 2))

    def accuracy(self, theta: np.ndarray) -> float:
        pred = self.predict(theta)
        return float(np.mean(np.sign(pred) == self.y))

    def predict_on(self, theta, X) -> np.ndarray:
        """Predictions on arbitrary features ``X`` (e.g. a held-out grid)."""
        enc = encode_states(np.asarray(X, dtype=np.float64), self.n)
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 2:
            theta = theta[None, ...]
        return self._expectations_encoded(theta, enc)[0]

    def loss_on(self, theta, X, y) -> float:
        pred = self.predict_on(theta, X)
        return float(np.mean((np.asarray(y, dtype=np.float64) - pred) ** 2))

    def accuracy_on(self, theta, X, y) -> float:
        pred = self.predict_on(theta, X)
        return float(np.mean(np.sign(pred) == np.asarray(y, dtype=np.float64)))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        """Exact parameter-shift gradient of the MSE loss, shape (M, n)."""
        theta = np.asarray(theta, dtype=np.float64)
        shifts = [theta]
        index = []
        for l in range(self.M):
            for q in range(self.n):
                tp = theta.copy()
                tp[l, q] += SHIFT
                shifts.append(tp)
                tm = theta.copy()
                tm[l, q] -= SHIFT
                shifts.append(tm)
                index.append((l, q))
        ex = self._expectations(np.array(shifts))  # (1 + 2P, S)
        base = ex[0]
        g = np.empty((self.M, self.n), dtype=np.float64)
        for k, (l, q) in enumerate(index):
            ep = ex[1 + 2 * k]
            em = ex[2 + 2 * k]
            g[l, q] = np.mean(-2.0 * (self.y - base) * 0.5 * (ep - em))
        return g

    def observable_apply(self, state: np.ndarray) -> np.ndarray:
        """Apply the readout observable ``O`` to a batched state: ``O |psi>``."""
        state = np.asarray(state)
        if state.ndim == 1:
            state = state[None, :]
        idx = np.arange(1 << self.n)
        if self.readout == "global":
            bits = ((idx[:, None] >> np.arange(self.n)) & 1).sum(axis=1)
            sign = (1.0 - 2.0 * (bits % 2)).astype(np.float64)
            return state * sign[None, :]
        out = np.zeros_like(state)
        for q in self.qubits:
            s = (1.0 - 2.0 * ((idx >> q) & 1)).astype(np.float64)
            out += state * s[None, :]
        return out / len(self.qubits)

    def grad_adjoint(self, theta: np.ndarray) -> np.ndarray:
        """Reverse-mode gradient of the MSE loss, shape (M, n).

        One forward + one reverse sweep (Jones & Gacon, arXiv:2009.02823),
        batched over the ``S`` samples. ~``2M`` circuit applications instead of
        the ``1 + 2P`` of the parameter-shift ``grad``; validated against it.
        """
        theta = np.asarray(theta, dtype=np.float64)
        S = self.S
        psi = self.encoded.copy()
        for l in range(self.M):
            apply_rotation_layer(psi, theta[l], self.axes[l], self.n, self._bits)
            if l < self.M - 1:
                apply_entangler(psi, self._esign)
        pred = expected_value(psi, self.n, self.readout, self.qubits)
        w = -2.0 * (self.y - pred) / S
        lam = self.observable_apply(psi) * w[:, None]
        g = np.zeros((self.M, self.n), dtype=np.float64)
        for l in range(self.M - 1, -1, -1):
            g[l] += pauli_inner(lam, psi, self.axes[l], self.n).imag.sum(axis=0)
            apply_rotation_layer(psi, -theta[l], self.axes[l], self.n, self._bits)
            apply_rotation_layer(lam, -theta[l], self.axes[l], self.n, self._bits)
            if l > 0:
                apply_entangler(psi, self._esign)
                apply_entangler(lam, self._esign)
        return g


def adam_classify(clf, theta0, lr=0.05, steps=200, beta1=0.9, beta2=0.999,
                  eps=1e-8, grad_fn=None):
    """Adam on the classifier loss. Returns (theta, loss_history).

    ``grad_fn`` defaults to the parameter-shift ``clf.grad``; pass
    ``clf.grad_adjoint`` for the (much faster) reverse-mode gradient.
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    gfun = clf.grad if grad_fn is None else grad_fn
    for t in range(1, steps + 1):
        c = clf.loss(theta)
        history.append(c)
        g = gfun(theta)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1 ** t)
        vhat = v / (1.0 - beta2 ** t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
    history.append(clf.loss(theta))
    return theta, history
