"""E2b: init / layerwise mitigations on the Ising protocol (SPEC section 9, E2b).

Identity-block and layerwise mitigations are measured on the *Ising* task
(energy cost ``<H>``, local) where an identity init is **not** degenerate
(SPEC section 6) -- never on the |0...0> protocol. Two parts:

  A. Gradient variance at the initial point: identity-block vs random init
     (adjoint gradients, N inits).
  B. Equal-budget training (D2): full-depth Adam from random init, full-depth
     Adam from identity init, and Skolik-style layerwise Adam from identity
     init, plus an L-BFGS-B oracle as the reachability reference. Final
     metric = normalised energy error ``(E-E0)/(E1-E0)`` (D5) and fidelity;
     success = ``1-F <= 1e-3``. Because all arms may converge to the same
     reachable floor, arms are also compared *under budget*: the first step
     at which ``rel_err <= threshold`` for thresholds {1e-1, 1e-2}.

Structural note (recorded Honest-field): with the *default* odd-M alternation
``Y,Z,Y,Z,...`` this ansatz cannot express the TFIM ground state -- even the
L-BFGS oracle floors at rel_err ~1.18 at M=17. The **period-4 folded
palindromic** schedule ``...Y,Z,Z,Y,<centre>,Y,Z,Z,Y,...`` reaches rel_err
~6e-3. Both are palindromic and identity-block compatible; E2b uses the
folded schedule (the default remains the alternation documented in SPEC 4).

Run:  python experiments/e2b_init_layerwise.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.init import identity_block_init, random_init
from bpqnn.simulator import StatevectorSimulator
from bpqnn.train import adam_train, layerwise_adam_train, lbfgs_train

N = 8
H = 1.0
M = 2 * N + 1  # odd -> identity-block constructible; ~2n depth (SPEC 14c.4)
BLOCK = 6
MAX_STEPS = 600
N_INIT_VAR = 128
SEEDS_TRAIN = 10
THRESHOLDS = (1e-1, 1e-2)
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def axes_folded(M: int) -> list:
    """Palindromic period-4 ``Y,Z,Z,Y`` schedule, valid for odd and even M."""
    base = ["Y", "Z", "Z", "Y"]
    axes = [base[j % 4] for j in range(M)]
    for j in range(M // 2):
        axes[M - 1 - j] = axes[j]
    return axes


def run_state(n, edges, axes, theta):
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(np.asarray(theta, dtype=np.float64), edges, axes)
    return sim.state


def infidelity_to_gs(state, gs) -> float:
    state = np.asarray(state).reshape(-1)
    return float(1.0 - abs(np.vdot(gs, state)) ** 2)


def main():
    RESULTS.mkdir(exist_ok=True)
    E0, gs, all_w = ising_ground_state(N, h=H)
    E1 = float(all_w[1])
    edges = [(q, q + 1) for q in range(N - 1)]
    axes = axes_folded(M)
    out = {
        "config": {"n": N, "h": H, "M": M, "block": BLOCK,
                   "max_steps": MAX_STEPS, "n_init_var": N_INIT_VAR,
                   "seeds_train": SEEDS_TRAIN, "E0": E0, "E1": E1,
                   "axes": axes},
    }

    print("=" * 78)
    print(f"E2b Ising protocol (cost = <H>, local), n={N}, h={H}, M={M}")
    print(f"axes (period-4 folded palindromic): {''.join(axes)}")
    print("=" * 78)

    # ---- part A: gradient variance at the initial point -------------------
    print("Part A: gradient variance at the initial point")
    rng = np.random.default_rng(11)
    t_id = identity_block_init(M, N, rng, batch=N_INIT_VAR)
    t_rd = random_init(M, N, rng, batch=N_INIT_VAR)
    g_id = adjoint_gradient(t_id, N, edges, axes, "ising", h=H)  # (N, M, n)
    g_rd = adjoint_gradient(t_rd, N, edges, axes, "ising", h=H)

    init_stats = {}
    for name, g in (("identity", g_id), ("random", g_rd)):
        per_layer = g.var(axis=0).mean(axis=1)
        stats = {
            "agg_var": float(per_layer.mean()),
            "var_first": float(per_layer[0]),
            "var_mid": float(per_layer[M // 2]),
            "var_last": float(per_layer[-1]),
            "mean_abs_grad": float(np.abs(g).mean()),
        }
        init_stats[name] = stats
        print(f"  {name:<9}: agg Var={stats['agg_var']:.3e}  "
              f"first={stats['var_first']:.3e} mid={stats['var_mid']:.3e} "
              f"last={stats['var_last']:.3e} ")
    out["init_variance"] = init_stats

    e_id_at = run_state(N, edges, axes, t_id[0])
    out["init_energy"] = {
        "identity": float(ising_energy(e_id_at, N, h=H)[0]),
        "E0": E0,
    }
    print(f"  identity-init energy (any seed) = {out['init_energy']['identity']:.6f}"
          f"  (E0={E0:.6f})")

    # ---- part B: equal-budget training ------------------------------------
    print("-" * 78)
    arms = ("random_full", "identity_full", "identity_layerwise")
    records = {a: [] for a in arms}
    oracle = []
    for s in range(SEEDS_TRAIN):
        rng = np.random.default_rng(1000 + s)
        t0_rd = random_init(M, N, rng)
        t0_id = identity_block_init(M, N, rng)
        results = (
            adam_train(t0_rd, n=N, kind="ising", h=H, edges=edges, axes=axes,
                       max_steps=MAX_STEPS),
            adam_train(t0_id, n=N, kind="ising", h=H, edges=edges, axes=axes,
                       max_steps=MAX_STEPS),
            layerwise_adam_train(t0_id, n=N, kind="ising", h=H, edges=edges,
                                 axes=axes, block=BLOCK, max_steps=MAX_STEPS,
                                 grow="output"),
        )
        for (name, res) in zip(arms, results):
            state = run_state(N, edges, axes, res["theta"])
            f = 1.0 - infidelity_to_gs(state, gs)
            rel_err = (res["cost"] - E0) / (E1 - E0)
            crosses = {}
            for thr in THRESHOLDS:
                hit = next((h["step"] for h in res["history"]
                            if (h["cost"] - E0) / (E1 - E0) <= thr), None)
                crosses[f"rel<={thr}"] = hit
            records[name].append({
                "seed": s, "energy": res["cost"], "rel_err": float(rel_err),
                "fidelity": float(f), "success": float(1.0 - f <= 1e-3),
                "steps": res["steps"], "budget": res["budget"],
                "cross": crosses,
            })

        res = lbfgs_train(t0_rd, n=N, kind="ising", h=H, edges=edges,
                          axes=axes, maxiter=400)
        state = run_state(N, edges, axes, res["theta"])
        oracle.append({
            "seed": s, "energy": res["cost"],
            "rel_err": (res["cost"] - E0) / (E1 - E0),
            "infidelity": infidelity_to_gs(state, gs),
        })

    out["training"] = {}
    print(f"oracle (L-BFGS-B, random init): median rel_err="
          f"{np.median([o['rel_err'] for o in oracle]):.3e}  "
          f"best={min(o['rel_err'] for o in oracle):.3e}")
    print("-" * 78)
    print(f"{'arm':<22} {'med rel_err':>12} {'med 1-F':>10} {'succ':>6} "
          f"{'steps':>6}  cross-threshold: n_seeds (median step)")
    for a in arms:
        rel = np.array([r["rel_err"] for r in records[a]])
        inf = np.array([1.0 - r["fidelity"] for r in records[a]])
        succ = np.array([r["success"] for r in records[a]])
        branches = []
        for thr in THRESHOLDS:
            hits = [r["cross"][f"rel<={thr}"] for r in records[a]]
            hits = [h for h in hits if h is not None]
            if hits:
                branches.append(f"t={thr}:{len(hits)}@{int(np.median(hits))}")
            else:
                branches.append(f"t={thr}:0")
        summary = {
            "rel_err_median": float(np.median(rel)),
            "rel_err_iqr": [float(np.percentile(rel, 25)),
                            float(np.percentile(rel, 75))],
            "infidelity_median": float(np.median(inf)),
            "success_rate": float(succ.mean()),
            "mean_steps": float(np.mean([r["steps"] for r in records[a]])),
            "cross": {f"t={thr}": [
                int(r["cross"][f"rel<={thr}"]) if r["cross"][f"rel<={thr}"]
                is not None else None for r in records[a]]
                for thr in THRESHOLDS},
        }
        out["training"][a] = {"summary": summary, "records": records[a]}
        print(f"{a:<22} {summary['rel_err_median']:>12.3e} "
              f"{summary['infidelity_median']:>10.3e} "
              f"{summary['success_rate']:>6.2f} {summary['mean_steps']:>6.1f}  "
              f"{' '.join(branches)}")
    out["training"]["oracle"] = oracle

    with open(RESULTS / "e2b_init_layerwise.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e2b_init_layerwise.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())