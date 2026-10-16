"""E0 sanity checks (SPEC section 9).

A. Own batched simulator vs qiskit.quantum_info.Statevector.
B. Parameter-shift gradients vs central finite differences, for
   - global |0...0> cost
   - local |0...0> cost
   - Ising energy (local training cost)
C. Physics invariants: Ising energy vs dense H, identity-block is exactly
   the identity, and C_L <= C_G <= n * C_L on |0...0>.
D. Adjoint (reverse-mode) gradients vs parameter-shift, for all four costs.

Run:  python experiments/e0_sanity.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.ansatz import make_edges, to_qiskit
from bpqnn.costs import global_cost_zero, infidelity, local_cost_zero
from bpqnn.grad import (
    adjoint_gradient,
    finite_difference,
    max_abs_diff,
    parameter_shift,
)
from bpqnn.hamiltonian import ising_energy, ising_ground_state, ising_hamiltonian
from bpqnn.init import identity_block_init, random_init
from bpqnn.simulator import StatevectorSimulator, default_axes

TOL_STATE = 1e-9
TOL_GRAD = 1e-6
TOL_ADJOINT = 1e-8
FD_EPS = 1e-5

_SEED = {"global": 101, "local": 202, "ising": 303}


def check_statevector(n_max: int = 6, M_max: int = 3):
    from qiskit.quantum_info import Statevector

    worst = 0.0
    rows = []
    for n in range(1, n_max + 1):
        edges = make_edges(n, "line")
        for M in range(1, M_max + 1):
            rng = np.random.default_rng(1000 * n + M)
            params = random_init(M, n, rng)
            axes = default_axes(M)
            sim = StatevectorSimulator(n, 1)
            sim.apply_circuit(params, edges=edges, axes=axes)
            qs = Statevector(to_qiskit(params, edges=edges, axes=axes)).data
            diff = float(np.max(np.abs(sim.state[0] - qs)))
            infid = 1.0 - abs(np.vdot(sim.state[0], qs)) ** 2
            worst = max(worst, diff)
            rows.append((n, M, diff, infid))
    ok = worst < TOL_STATE and all(r[3] < 1e-12 for r in rows)
    print("A. statevector vs Qiskit")
    print(f"   cases: {len(rows)}   worst |amp diff|: {worst:.3e}   tol {TOL_STATE:.0e}")
    print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def _cost_fn(n, edges, kind, h=1.0):
    def cost(p):
        p = np.asarray(p, dtype=np.float64)
        B = p.shape[0]
        M = p.shape[1]
        sim = StatevectorSimulator(n, B)
        sim.apply_circuit(p, edges=edges, axes=default_axes(M))
        if kind == "global":
            return global_cost_zero(sim.state)
        if kind == "local":
            return local_cost_zero(sim.state, n)
        if kind == "ising":
            return ising_energy(sim.state, n, h=h)
        raise ValueError(kind)

    return cost


def check_gradients(n_max: int = 5, M_max: int = 3):
    worst = 0.0
    rows = []
    for kind in ("global", "local", "ising"):
        for n in range(2, n_max + 1):
            edges = make_edges(n, "line")
            for M in range(1, M_max + 1):
                rng = np.random.default_rng(77 * n + 13 * M + _SEED[kind])
                params = random_init(M, n, rng)
                cost = _cost_fn(n, edges, kind)
                ps = parameter_shift(params, cost)
                fd = finite_difference(params, cost, eps=FD_EPS)
                err = max_abs_diff(ps, fd)
                worst = max(worst, err)
                rows.append((kind, n, M, err))
    ok = worst < TOL_GRAD
    print("B. parameter-shift vs finite differences")
    for kind in ("global", "local", "ising"):
        k = max(r[3] for r in rows if r[0] == kind)
        print(f"   {kind:>7}: worst |g_ps - g_fd| = {k:.3e}")
    print(f"   tolerance {TOL_GRAD:.0e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_physics(n_max: int = 6):
    """Ising energy vs dense H; identity-block is the identity; cost bounds."""
    ok = True
    # Ising energy consistency
    e_worst = 0.0
    for n in range(2, n_max + 1):
        h = 1.0
        E0, gs, _ = ising_ground_state(n, h)
        e_worst = max(e_worst, abs(float(ising_energy(gs[None, :], n, h)[0]) - E0))
        H = ising_hamiltonian(n, h)
        rng = np.random.default_rng(5 * n)
        params = random_init(2, n, rng)
        sim = StatevectorSimulator(n, 1)
        sim.apply_circuit(params)
        psi = sim.state[0]
        dense = float(np.real(np.conj(psi) @ (H @ psi)))
        e_worst = max(e_worst, abs(float(ising_energy(psi[None, :], n, h)[0]) - dense))
    ok &= e_worst < 1e-9

    # identity-block (odd M): circuit should be exactly I => state stays |0...0>
    ib_worst = 0.0
    for n in range(2, n_max + 1):
        for M in (3, 5):
            assert _is_palindromic(default_axes(M))
            rng = np.random.default_rng(31 * n + M)
            params = identity_block_init(M, n, rng)
            sim = StatevectorSimulator(n, 1)
            sim.apply_circuit(params, axes=default_axes(M))
            ib_worst = max(ib_worst, float(global_cost_zero(sim.state)[0]))
    ok &= ib_worst < 1e-12

    # cost bounds C_L <= C_G <= n * C_L
    bnd_worst = 0.0
    for n in range(2, n_max + 1):
        rng = np.random.default_rng(9 * n)
        params = random_init(3, n, rng, batch=16)
        sim = StatevectorSimulator(n, 16)
        sim.apply_circuit(params)
        cg = global_cost_zero(sim.state)
        cl = local_cost_zero(sim.state, n)
        bnd_worst = max(bnd_worst, float(np.max(cl - cg)), float(np.max(cg - n * cl)))
    ok &= bnd_worst < 1e-12

    print("C. physics invariants")
    print(f"   Ising energy vs dense H : worst {e_worst:.3e}")
    print(f"   identity-block cost      : worst {ib_worst:.3e}")
    print(f"   C_L<=C_G<=n*C_L          : worst violation {bnd_worst:.3e}")
    print(f"   -> {'PASS' if ok else 'FAIL'}")
    return ok


def _is_palindromic(axes) -> bool:
    return all(axes[j] == axes[len(axes) - 1 - j] for j in range(len(axes)))


_KINDS4 = ("global_zero", "local_zero", "ising", "infidelity")


def check_adjoint(n_max: int = 6, M_max: int = 4):
    """Adjoint (reverse-mode) gradient vs exact parameter-shift, all costs."""
    worst = 0.0
    rows = []
    for kind in _KINDS4:
        for n in range(2, n_max + 1):
            edges = make_edges(n, "line")
            for M in range(1, M_max + 1):
                axes = default_axes(M)
                for B in (1, 4):
                    rng = np.random.default_rng(404 * B + 7 * n + M + len(kind))
                    params = random_init(M, n, rng, batch=B)
                    target = ising_ground_state(n, 1.0)[1] if kind == "infidelity" else None

                    def cost(p, kind=kind, n=n, edges=edges, axes=axes, target=target):
                        p = np.asarray(p, dtype=np.float64)
                        single = p.ndim == 2
                        P = p[None, ...] if single else p
                        sim = StatevectorSimulator(n, P.shape[0])
                        sim.apply_circuit(P, edges=edges, axes=axes)
                        if kind == "global_zero":
                            return global_cost_zero(sim.state)
                        if kind == "local_zero":
                            return local_cost_zero(sim.state, n)
                        if kind == "ising":
                            return ising_energy(sim.state, n)
                        return infidelity(sim.state, target)

                    ga = adjoint_gradient(params, n, edges, axes, kind, target=target)
                    gps = parameter_shift(params, cost)
                    err = max_abs_diff(ga, gps)
                    worst = max(worst, err)
                    rows.append((kind, n, M, B, err))
    ok = worst < TOL_ADJOINT
    print("D. adjoint (reverse-mode) vs parameter-shift")
    for kind in _KINDS4:
        k = max(r[4] for r in rows if r[0] == kind)
        print(f"   {kind:>12}: worst |g_adj - g_ps| = {k:.3e}")
    print(f"   tolerance {TOL_ADJOINT:.0e}   -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 60)
    print("E0 sanity")
    print("=" * 60)
    a = check_statevector()
    b = check_gradients()
    c = check_physics()
    d = check_adjoint()
    print("=" * 60)
    print("E0:", "PASS" if (a and b and c and d) else "FAIL")
    return 0 if (a and b and c and d) else 1


if __name__ == "__main__":
    raise SystemExit(main())
