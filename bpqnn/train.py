"""Training on the palindromic ansatz (SPEC section 7).

Two optimizers are provided, both driven by the exact adjoint gradient:

- ``adam_train``: Adam (SPEC D1: beta1=0.9, beta2=0.999, eps=1e-8).
- ``lbfgs_train``: L-BFGS-B oracle (SciPy), for reference optimal costs.

Budget accounting follows SPEC D2 (state preparations): a cost evaluation
costs 1; a gradient costs ``grad_cost`` (default ``2*M*n + 1``, the
parameter-shift convention, so budgets stay comparable to any shift-based
baseline). The actual adjoint work is ~2 forward passes and is reported
separately as ``circuit_evals``.
"""

from __future__ import annotations

import numpy as np

from . import costs, hamiltonian
from .grad import adjoint_gradient
from .simulator import StatevectorSimulator, default_axes


def cost_value(state, kind: str, n: int, h: float = 1.0, target=None) -> float:
    if kind == "ising":
        return float(hamiltonian.ising_energy(state, n, h=h)[0])
    if kind == "global_zero":
        return float(costs.global_cost_zero(state)[0])
    if kind == "local_zero":
        return float(costs.local_cost_zero(state, n)[0])
    if kind == "infidelity":
        return float(costs.infidelity(state, target)[0])
    raise ValueError(f"unknown cost kind: {kind!r}")


def _run_circuit(n, edges, axes, theta):
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(theta, edges, axes)
    return sim.state


def make_evals(n, M, kind="ising", h=1.0, target=None, edges=None, axes=None):
    """Return (cost_fn, grad_fn) closures for a single parameter vector."""
    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)

    def cost(theta):
        state = _run_circuit(n, edges, axes, np.asarray(theta, dtype=np.float64))
        return cost_value(state, kind, n, h, target)

    def grad(theta):
        return adjoint_gradient(
            theta, n, edges, axes, kind, h=h, target=target
        ).reshape(M, n)

    return cost, grad


def adam_train(
    theta0,
    *,
    n: int,
    kind: str = "ising",
    h: float = 1.0,
    target=None,
    edges=None,
    axes=None,
    lr: float = 0.05,
    max_steps: int = 300,
    budget: float | None = None,
    grad_cost: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    grad_clip: float | None = None,
    tol: float = 0.0,
):
    """Adam training loop. Returns a result dict with history and finals."""
    theta = np.asarray(theta0, dtype=np.float64).copy()
    M, nq = theta.shape
    if grad_cost is None:
        grad_cost = 2.0 * M * nq + 1.0
    cost_fn, grad_fn = make_evals(n, M, kind, h, target, edges, axes)

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    budget_used = 0.0
    circuit_evals = 0.0
    history = []
    for t in range(1, max_steps + 1):
        c = cost_fn(theta)
        g = grad_fn(theta)
        budget_used += 1.0 + grad_cost
        circuit_evals += 1.0 + 2.0  # one forward cost + one forward/backward pair
        gnorm = float(np.linalg.norm(g))
        history.append(
            {"step": t, "cost": float(c), "grad_norm": gnorm, "budget": budget_used}
        )
        if tol and c <= tol:
            break
        if budget is not None and budget_used >= budget:
            break
        if grad_clip is not None and gnorm > grad_clip:
            g = g * (grad_clip / gnorm)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1**t)
        vhat = v / (1.0 - beta2**t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

    final_cost = cost_fn(theta)
    return {
        "theta": theta,
        "cost": final_cost,
        "history": history,
        "budget": budget_used,
        "circuit_evals": circuit_evals,
        "steps": len(history),
    }


def layerwise_adam_train(
    theta0,
    *,
    n: int,
    kind: str = "ising",
    h: float = 1.0,
    target=None,
    edges=None,
    axes=None,
    lr: float = 0.05,
    block: int = 6,
    max_steps: int = 300,
    grow: str = "input",
    tol_stage: float = 1e-6,
    patience: int = 15,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    """Skolik-style layerwise training on the palindromic ansatz.

    The full ``(M, n)`` circuit definition is fixed throughout (same ``axes``,
    same ``edges``); "layerwise" means only a growing slice of layers is
    trainable at a time. Stage ``s`` activates layers ``[s*block, (s+1)*block)``
    while every earlier layer is frozen at its incoming values. Each stage runs
    Adam (fresh per-stage moments) until the cost improvement over a patience
    window drops below ``tol_stage`` or the stage's step budget runs out.

    Every Adam step is charged the full ``1 + grad_cost`` budget (D2) -- the
    adjoint sweep already produces the whole gradient -- so a layerwise run
    with ``max_steps`` in total consumes the same budget as a full-depth run
    of ``max_steps`` steps.
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    M, nq = theta.shape
    grad_cost = 2.0 * M * nq + 1.0
    cost_fn, grad_fn = make_evals(n, M, kind, h, target, edges, axes)

    n_stages = int(np.ceil(M / block))
    stage_cap = int(np.ceil(max_steps / n_stages))
    budget_used = 0.0
    circuit_evals = 0.0
    history = []
    global_step = 0

    for s in range(n_stages):
        if grow == "input":
            lo, hi = s * block, min((s + 1) * block, M)
        else:
            hi, lo = M - s * block, max(M - (s + 1) * block, 0)
        active = np.zeros((M, nq), dtype=bool)
        active[lo:hi, :] = True
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        best_cost = np.inf
        stall = 0

        for t in range(1, stage_cap + 1):
            global_step += 1
            c = cost_fn(theta)
            g = grad_fn(theta)
            g = g * active  # only the active slice is updated
            budget_used += 1.0 + grad_cost
            circuit_evals += 1.0 + 2.0
            history.append({
                "step": global_step, "stage": s, "cost": float(c),
                "grad_norm": float(np.linalg.norm(g)), "budget": budget_used,
            })
            if best_cost - c > tol_stage:
                best_cost = c
                stall = 0
            else:
                stall += 1
            if stall >= patience:
                break
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * g * g
            mhat = m / (1.0 - beta1 ** global_step)
            vhat = v / (1.0 - beta2 ** global_step)
            theta = theta - lr * mhat * active / (np.sqrt(vhat) + eps)

    final_cost = cost_fn(theta)
    return {
        "theta": theta,
        "cost": final_cost,
        "history": history,
        "budget": budget_used,
        "circuit_evals": circuit_evals,
        "steps": len(history),
        "stages": n_stages,
    }


def lbfgs_train(
    theta0,
    *,
    n: int,
    kind: str = "ising",
    h: float = 1.0,
    target=None,
    edges=None,
    axes=None,
    maxiter: int = 500,
):
    """L-BFGS-B oracle minimisation using the adjoint gradient."""
    from scipy.optimize import minimize

    theta0 = np.asarray(theta0, dtype=np.float64)
    M, nq = theta0.shape
    cost_fn, grad_fn = make_evals(n, M, kind, h, target, edges, axes)

    def fun(x):
        return cost_fn(x.reshape(M, nq))

    def jac(x):
        return grad_fn(x.reshape(M, nq)).reshape(-1)

    res = minimize(fun, theta0.reshape(-1), jac=jac, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return {
        "theta": res.x.reshape(M, nq),
        "cost": float(res.fun),
        "success": bool(res.success),
        "nit": int(res.nit),
    }


def train_ansatz(
    theta0,
    *,
    optimizer: str = "adam",
    n: int,
    kind: str = "ising",
    h: float = 1.0,
    target=None,
    edges=None,
    axes=None,
    **kwargs,
):
    if optimizer == "adam":
        return adam_train(theta0, n=n, kind=kind, h=h, target=target,
                          edges=edges, axes=axes, **kwargs)
    if optimizer == "lbfgs":
        return lbfgs_train(theta0, n=n, kind=kind, h=h, target=target,
                           edges=edges, axes=axes, **kwargs)
    raise ValueError(f"unknown optimizer: {optimizer!r}")
