"""Cost-Support Annealing (E13) - a law-driven mitigation for the E11 plateau.

The E1/E2/E11 story: gradient variance decays exponentially in the observable's
support radius ``r`` (`Var(dO) ~ C e^{-kr}`, E11). A fixed cost therefore has to
choose between *trainable* (small ``r``) and *faithful* (large ``r``) -- the
local-vs-global trade-off. Cost-Support Annealing is the third option: start from
``r=1`` (variance is poly, cheap to spend) and *grow the support during training*
toward the target radius ``r*``. Growth is admission-controlled by the empirical
gradient power, i.e. the measured variance is used as the control law.

All costs here are the E11 *single contiguous block* family
(`bpqnn.observables.block_cost`, left-anchored at ``start=0``):

    C_r = 1 - P(qubits 0..r-1 all 0),     r = 1..n

so ``r=1`` is a single-qubit *local* cost (blind to qubits beyond 0), ``r=n`` is
the E1 *global* cost (faithful, plateaus), and ``r*`` is the target block.
"""

from __future__ import annotations

import numpy as np

from .grad import adjoint_gradient
from .observables import block_cost
from .simulator import StatevectorSimulator, default_axes


# ---------------------------------------------------------------- schedules

class Schedule:
    """Return the support radius for optimisation step ``t`` (0-indexed)."""
    def __init__(self, r_star: int):
        if not 1 <= r_star:
            raise ValueError("r_star must be >= 1")
        self.r_star = int(r_star)

    def r(self, t: int) -> int:
        return self.r_star

    def update(self, info: dict) -> None:
        """Admission hook (var-gated); base class does nothing."""


class Linear(Schedule):
    """r_t ramps linearly 1 -> r* across the run."""
    def __init__(self, r_star: int, steps: int):
        super().__init__(r_star)
        self.steps = max(steps, 1)

    def r(self, t: int):
        if self.r_star == 1:
            return 1
        return min(self.r_star, 1 + (self.r_star - 1) * (t + 1) // self.steps)


class Staged(Schedule):
    """Each radius held for ``hold`` steps; 1 -> r* then stays."""
    def __init__(self, r_star: int, hold: int):
        super().__init__(r_star)
        self.hold = max(hold, 1)

    def r(self, t: int):
        return min(self.r_star, 1 + t // self.hold)


class VarGated(Schedule):
    """Law-driven admission control (the CSA control law).

    r starts at 1 and increments one at a time. Every ``gate_every`` steps the
    empirical gradient power at the current radius ``r`` (and the *prospective*
    power at ``r+1``, when the caller supplies it) is compared against
    ``floor_frac`` of the power measured at the very first step; r grows only
    while both signals are affordable. This implements "spend the variance
    budget where it is poly (small r); grow only when the signal survives".
    """
    def __init__(self, r_star: int, *, gate_every: int = 5, floor_frac: float = 0.25):
        super().__init__(r_star)
        self.gate_every = max(gate_every, 1)
        self.floor_frac = float(floor_frac)
        self.cur = 1
        self.base = None

    def _affordable(self, power) -> bool:
        if self.base is None or self.base <= 0.0:
            return True
        return power >= self.floor_frac * self.base

    def update(self, info: dict) -> None:
        if self.cur >= self.r_star:
            return
        pow_cur = info.get("pow_current")
        if self.cur == 1 and pow_cur is not None and self.base is None:
            self.base = pow_cur
        if pow_cur is None or not self._affordable(pow_cur):
            return
        pow_next = info.get("pow_prospective")
        if pow_next is not None and not self._affordable(pow_next):
            return
        self.cur += 1

    def r(self, t: int):
        return min(self.r_star, self.cur)


# ---------------------------------------------------------------- training

def _block_observable(r: int, n: int):
    from .observables import block_observable
    return lambda st: block_observable(st, r, n, start=0)


def block_grad(p, n, r, edges, axes) -> np.ndarray:
    return adjoint_gradient(p, n, edges, axes,
                            observable=_block_observable(r, n)).reshape(p.shape)


def window_grad(p, n, r, edges, axes) -> np.ndarray:
    """Gradient of the window-averaged cost ``C_w(r)`` (faithful, spread)."""
    from .observables import support_radius_observable
    return adjoint_gradient(p, n, edges, axes,
                            observable=lambda st: support_radius_observable(st, r, n)
                            ).reshape(p.shape)


def block_cost_of(theta, n, r, edges, axes) -> float:
    dim = 1 << n
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(np.asarray(theta, np.float64), edges, axes)
    return float(block_cost(sim.state, r, n, start=0)[0])


def _final_costs(theta, n, edges, axes, targets) -> dict:
    return {f"cost_{name}": block_cost_of(theta, n, r, edges, axes)
            for name, r in targets.items()}


def adam_block(theta0, n, r, *, steps=300, lr=0.05, edges=None, axes=None,
               record_every=1, beta1=0.9, beta2=0.999, eps=1e-8):
    """Fixed-radius Adam minimising the block cost ``C_r`` (start=0)."""
    return _train(theta0, n, r, Schedule(r), steps=steps, lr=lr,
                  edges=edges, axes=axes, record_every=record_every,
                  beta1=beta1, beta2=beta2, eps=eps)


def adam_window(theta0, n, r, *, steps=300, lr=0.05, edges=None, axes=None,
                record_every=1, beta1=0.9, beta2=0.999, eps=1e-8):
    """Fixed-radius Adam minimising the window-averaged cost ``C_w(r)``.

    ``C_w(r) = 1 - (1/W) sum_w P(qubits in window w all 0)`` is *faithful* for
    the all-zero target at any ``r >= 1`` (its windows tile the register) yet
    keeps each observable support at ``r`` -- the E13 "support-spread" arm.
    """
    return _train(theta0, n, n, Schedule(r), steps=steps, lr=lr,
                  edges=edges, axes=axes, record_every=record_every,
                  beta1=beta1, beta2=beta2, eps=eps, window=r)


def csa_adam_train(theta0, n, r_star, *, schedule="linear", steps=300, lr=0.05,
                   hold=10, gate_every=5, floor_frac=0.25, edges=None, axes=None,
                   record_every=1, beta1=0.9, beta2=0.999, eps=1e-8):
    """Cost-Support Annealing: Adam while the support radius grows 1 -> r*.

    ``schedule``: ``"linear"``, ``"staged"`` (``hold`` steps per radius), or
    ``"var_gated"`` (admission-controlled; ``gate_every`` / ``floor_frac``).
    ``targets`` = ``{"r_star": r_star}``; final costs are recorded there.
    """
    if isinstance(schedule, Schedule):
        sched = schedule
    elif schedule == "linear":
        sched = Linear(r_star, steps)
    elif schedule == "staged":
        sched = Staged(r_star, hold)
    elif schedule == "var_gated":
        sched = VarGated(r_star, gate_every=gate_every, floor_frac=floor_frac)
    else:
        raise ValueError(f"unknown CSA schedule: {schedule!r}")
    return _train(theta0, n, r_star, sched, steps=steps, lr=lr,
                  edges=edges, axes=axes, record_every=record_every,
                  beta1=beta1, beta2=beta2, eps=eps)


def _train(theta0, n, target_r, sched, *, steps=300, lr=0.05, edges=None,
           axes=None, record_every=1, beta1=0.9, beta2=0.999, eps=1e-8,
           window=None):
    theta = np.asarray(theta0, dtype=np.float64).copy()
    M, nq = theta.shape
    if axes is None:
        axes = default_axes(M)
    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]

    need_next = isinstance(sched, VarGated)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    for t in range(steps):
        r = sched.r(t)
        g = (window_grad(theta, n, r, edges, axes) if window is not None
             else block_grad(theta, n, r, edges, axes))
        pow_cur = float(np.mean(np.abs(g) ** 2))
        info = {"pow_current": pow_cur}
        if need_next and (t + 1) % sched.gate_every == 0 and r < target_r:
            gnext = block_grad(theta, n, r + 1, edges, axes)
            info["pow_prospective"] = float(np.mean(np.abs(gnext) ** 2))
        sched.update(info)
        if (t + 1) % max(record_every, 1) == 0 or t == steps - 1:
            rec = {"step": t + 1, "r": r,
                   "grad_power": pow_cur,
                   "cost_target": block_cost_of(theta, n, target_r, edges, axes)}
            history.append(rec)

        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1 ** (t + 1))
        vhat = v / (1.0 - beta2 ** (t + 1))
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

    finals = {"cost_target": block_cost_of(theta, n, target_r, edges, axes),
              "cost_1": block_cost_of(theta, n, 1, edges, axes),
              "cost_n": block_cost_of(theta, n, theta.shape[1], edges, axes),
              "final_r": sched.r(steps - 1)}
    history[-1]["cost_target"] = finals["cost_target"]
    return {"theta": theta, "history": history, "finals": finals,
            "steps": len(history)}


# ------------------------------------------------- bounded local->global mix

def mixed_grad(p, n, lam, edges, axes) -> np.ndarray:
    """Gradient of ``C(lam) = (1-lam) C_L + lam C_G`` (both target |0...0>)."""
    from .costs import global_zero_observable, local_zero_observable

    def obs(st):
        return (1.0 - lam) * local_zero_observable(st, n) + lam * global_zero_observable(st)

    return adjoint_gradient(p, n, edges, axes, observable=obs).reshape(p.shape)


def mixed_cost_of(theta, n, lam, edges, axes) -> float:
    from .costs import global_cost_zero, local_cost_zero

    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(np.asarray(theta, np.float64), edges, axes)
    cg = float(global_cost_zero(sim.state)[0])
    cl = float(local_cost_zero(sim.state, n)[0])
    return (1.0 - lam) * cl + lam * cg


def adam_mix(theta0, n, *, steps=300, lr=0.05, lam_max=1.0, schedule="linear",
             edges=None, axes=None, record_every=1, beta1=0.9, beta2=0.999,
             eps=1e-8):
    """Adam on the blended cost ``(1-lam) C_L + lam C_G`` with ``lam`` ramped.

    ``schedule``: ``"linear"`` ramps ``0 -> lam_max`` over the run;
    ``"constant"`` holds ``lam = lam_max`` throughout (the global-cost control
    for ``lam_max = 1``). Uses the bound ``C_L <= C_G <= n C_L`` (Cerezo et al.
    2021): driving the trainable ``C_L`` down also drives the faithful ``C_G``
    down, so a ramped blend reaches a low global cost without starting on the
    plateau.
    """
    from .costs import global_cost_zero, local_cost_zero

    theta = np.asarray(theta0, dtype=np.float64).copy()
    M, nq = theta.shape
    if axes is None:
        axes = default_axes(M)
    if edges is None:
        edges = [(q, q + 1) for q in range(n - 1)]

    if schedule == "linear":
        steps_i = max(int(steps), 1)
        lam_at = lambda t: lam_max * (t + 1) / steps_i  # noqa: E731
    elif schedule == "constant":
        lam_at = lambda t: lam_max  # noqa: E731
    else:
        raise ValueError(f"unknown mix schedule: {schedule!r}")

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    history = []
    for t in range(steps):
        lam = lam_at(t)
        g = mixed_grad(theta, n, lam, edges, axes)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * g * g
        mhat = m / (1.0 - beta1 ** (t + 1))
        vhat = v / (1.0 - beta2 ** (t + 1))
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)
        if (t + 1) % max(record_every, 1) == 0 or t == steps - 1:
            sim = StatevectorSimulator(n, 1)
            sim.apply_circuit(theta, edges, axes)
            history.append({"step": t + 1, "lam": float(lam),
                            "cost_L": float(local_cost_zero(sim.state, n)[0]),
                            "cost_G": float(global_cost_zero(sim.state)[0])})

    finals = dict(history[-1])
    return {"theta": theta, "history": history, "finals": finals,
            "steps": len(history)}