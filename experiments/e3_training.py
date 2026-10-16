"""E3: headline equal-budget training comparison on the Ising task
(SPEC section 9, D1-D6, D8).

Six arms, all on the period-4 folded palindromic schedule at M = 2n+1
(SPEC 14f resolves the expressivity trap):

  global_random    -- infidelity cost, random init  (baseline: expect BP harm)
  global_identity  -- infidelity cost, identity init
  local_random     -- energy cost,  random init
  local_identity   -- energy cost,  identity init
  local_zero       -- energy cost,  zero init
  layerwise_identity -- energy cost, layerwise (grow=output) from identity

Every arm shares the same Adam hyperparameters (D1), the same budget
accounting per step ``1 + (2*M*n + 1)`` (D2), and the same 600-step budget.
Reports: final energy error and infidelity (D5), success rate (infidelity
<= 1e-3), training curves vs budget with a bootstrap band over seeds (D6),
and an L-BFGS-B oracle as a reachability reference. A short *shot-noisy*
training run (sampled energy + parameter-shift gradient, 2048 shots) is
reported separately as descriptive (D8).

Run:  python experiments/e3_training.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hamiltonian import ising_energy, ising_ground_state, ising_hamiltonian_apply
from bpqnn.init import identity_block_init, random_init, zero_init
from bpqnn.simulator import StatevectorSimulator
from bpqnn.shots import estimate_ising_energy, sampled_ising_grad_cost
from bpqnn.train import adam_train, layerwise_adam_train, lbfgs_train
from experiments.e2b_init_layerwise import axes_folded, run_state

N = 8
H = 1.0
M = 2 * N + 1
BLOCK = 6
MAX_STEPS = 600
SEEDS_MAIN = 16
SEEDS_LAYERWISE = 8
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

ARMS = [
    ("global_random", "infidelity", "random", False),
    ("global_identity", "infidelity", "identity", False),
    ("local_random", "ising", "random", False),
    ("local_identity", "ising", "identity", False),
    ("local_zero", "ising", "zero", False),
    ("layerwise_identity", "ising", "identity", True),
]


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
                   "max_steps": MAX_STEPS, "E0": E0, "E1": E1,
                   "axes": axes},
        "arms": {},
    }

    print("=" * 78)
    print(f"E3 equal-budget training, Ising n={N} h={H}, M={M}, "
          f"schedule {'Y/Z,Z/Y,..'} : {''.join(axes)}")
    print("=" * 78)

    out_path = RESULTS / "e3_training.json"

    for (name, kind, init, layerwise) in ARMS:
        nseeds = SEEDS_LAYERWISE if layerwise else SEEDS_MAIN
        records = []
        for s in range(nseeds):
            rng = np.random.default_rng(2000 + s)
            t0 = {
                "random": random_init(M, N, rng),
                "identity": identity_block_init(M, N, rng),
                "zero": zero_init(M, N),
            }[init]
            if layerwise:
                res = layerwise_adam_train(
                    t0, n=N, kind=kind, h=H, target=gs, edges=edges,
                    axes=axes, block=BLOCK, max_steps=MAX_STEPS, grow="output")
            else:
                res = adam_train(t0, n=N, kind=kind, h=H, target=gs,
                                 edges=edges, axes=axes, max_steps=MAX_STEPS)
            state = run_state(N, edges, axes, res["theta"])
            inf = infidelity_to_gs(state, gs)
            E = float(ising_energy(state, N, h=H)[0])
            rel = (E - E0) / (E1 - E0)
            records.append({
                "seed": s, "energy": E, "rel_err": rel, "infidelity": inf,
                "success": float(inf <= 1e-3), "steps": res["steps"],
                "steps_history": [(h["step"], h["cost"], h["budget"]) for h in res["history"]],
            })
        med_rel = float(np.median([r["rel_err"] for r in records]))
        med_inf = float(np.median([r["infidelity"] for r in records]))
        succ = float(np.mean([r["success"] for r in records]))
        out["arms"][name] = {"kind": kind, "init": init, "layerwise": layerwise,
                             "records": records,
                             "summary": {"rel_err_median": med_rel,
                                         "infidelity_median": med_inf,
                                         "success_rate": succ}}
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"{name:<20} med rel_err={med_rel:.3e}  med 1-F={med_inf:.3e}  "
              f"success={succ:4.2f}  ({nseeds} seeds x {MAX_STEPS} steps)")

    # ---- oracle reference (reachability, not budget-honest) ---------------
    print("-" * 78)
    out["oracle"] = {}
    for kind in ("infidelity", "ising"):
        vals = []
        for s in range(8):
            rng = np.random.default_rng(3000 + s)
            t0 = random_init(M, N, rng)
            res = lbfgs_train(t0, n=N, kind=kind, h=H, target=gs,
                              edges=edges, axes=axes, maxiter=400)
            state = run_state(N, edges, axes, res["theta"])
            E = float(ising_energy(state, N, h=H)[0])
            vals.append({"seed": s, "energy": E,
                         "rel_err": (E - E0) / (E1 - E0),
                         "infidelity": infidelity_to_gs(state, gs)})
        out["oracle"][kind] = vals
        med = float(np.median([v["rel_err"] for v in vals]))
        print(f"oracle {kind:<10}: med rel_err={med:.3e}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ---- shot noise (descriptive, D8) ------------------------------------
    print("-" * 78)
    print("shot-noise section (descriptive): sampled-energy validation + one "
          "short shot-noisy training run")
    gsE = float(np.vdot(np.asarray(gs), ising_hamiltonian_apply(gs, N, h=H).reshape(-1)).real)
    est, used, se = estimate_ising_energy(gs, N, h=H, shots=4096, n_batches=20,
                                          rng=np.random.default_rng(5))
    print(f"  estimator on exact gs: E={est:.4f} +- {se:.4f}  (exact {gsE:.4f}, "
          f"shots used {used})")

    shot_train = []
    for s in range(2):
        rng = np.random.default_rng(4000 + s)
        theta = random_init(M, N, rng)
        hist = []
        for step in range(1, 21):
            from bpqnn.simulator import default_axes
            sim = StatevectorSimulator(N, 1)
            sim.apply_circuit(theta, edges, axes)
            E, used, _ = estimate_ising_energy(sim.state, N, h=H, shots=2048,
                                               n_batches=8, rng=rng)
            g = sampled_ising_grad_cost(theta, N, h=H, shots=2048, rng=rng,
                                        edges=edges, axes=axes)
            theta = theta - 0.05 * g / (np.sqrt(g * g) + 1e-8)
            hist.append({"step": step, "sampled_energy": float(E)})
        sim = StatevectorSimulator(N, 1)
        sim.apply_circuit(theta, edges, axes)
        E_ex = float(ising_energy(sim.state, N, h=H)[0])
        shot_train.append({"seed": s, "final_exact_energy": E_ex,
                           "final_rel_err": (E_ex - E0) / (E1 - E0),
                           "history": hist})
    out["shot_noise"] = {"validation": {"est": est, "se": se, "exact": gsE,
                                        "shots_used": used},
                         "sampled_run": shot_train,
                         "note": "descriptive only (2 seeds, 20 steps, 2048 "
                                 "shots); parameter-shift with sampled "
                                 "gradients"}
    med = float(np.median([r["final_rel_err"] for r in shot_train]))
    print(f"  shot-noisy trained (20 steps, 2048 shots, 2 seeds): "
          f"med rel_err={med:.3e}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")

    # ---- scaling section: regime where plateaus bite (n=10, M=21) ---------
    print("=" * 78)
    print("E3-scaling: same comparison at n=10 / M=21 (folded schedule)")
    out["scaling"] = {"config": {"n": 10, "h": 1.0, "M": 21},
                      "arms": {}}
    n10, M10 = 10, 21
    E0_10, gs10, allw10 = ising_ground_state(n10, h=H)
    E1_10 = float(allw10[1])
    edges10 = [(q, q + 1) for q in range(n10 - 1)]
    axes10 = axes_folded(M10)
    SCALE_SEEDS = 6
    SCALE_STEPS = 400
    for (name, kind, init) in (("global_random", "infidelity", "random"),
                               ("global_identity", "infidelity", "identity"),
                               ("local_random", "ising", "random"),
                               ("local_identity", "ising", "identity")):
        records = []
        for s in range(SCALE_SEEDS):
            rng = np.random.default_rng(9000 + s)
            t0 = {"random": random_init(M10, n10, rng),
                  "identity": identity_block_init(M10, n10, rng)}[init]
            res = adam_train(t0, n=n10, kind=kind, h=H, target=gs10,
                             edges=edges10, axes=axes10,
                             max_steps=SCALE_STEPS)
            state = run_state(n10, edges10, axes10, res["theta"])
            E = float(ising_energy(state, n10, h=H)[0])
            records.append({"seed": s,
                            "rel_err": (E - E0_10) / (E1_10 - E0_10),
                            "infidelity": infidelity_to_gs(state, gs10),
                            "success": float(infidelity_to_gs(state, gs10)
                                             <= 1e-3),
                            "steps": res["steps"],
                            "steps_history": [(h["step"], h["cost"],
                                               h["budget"])
                                              for h in res["history"]]})
        med_rel = float(np.median([r["rel_err"] for r in records]))
        out["scaling"]["arms"][name] = {"records": records,
                                        "rel_err_median": med_rel}
        print(f"n=10 {name:<16} med rel_err={med_rel:.3e}")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    oracle10 = []
    for s in range(4):
        rng = np.random.default_rng(9500 + s)
        t0 = random_init(M10, n10, rng)
        res = lbfgs_train(t0, n=n10, kind="ising", h=H, target=gs10,
                          edges=edges10, axes=axes10, maxiter=300)
        state = run_state(n10, edges10, axes10, res["theta"])
        E = float(ising_energy(state, n10, h=H)[0])
        oracle10.append({"rel_err": (E - E0_10) / (E1_10 - E0_10),
                         "infidelity": infidelity_to_gs(state, gs10)})
    out["scaling"]["oracle"] = oracle10
    med = float(np.median([o["rel_err"] for o in oracle10]))
    print(f"n=10 oracle ising: med rel_err={med:.3e}")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {out_path}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e3_training.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _resample(records, kind, E0, E1, grid):
    """Resample per-seed (step, cost) curves onto a common step grid, mapping
    an energy cost to rel_err = (cost - E0)/(E1 - E0)."""
    ys = []
    for r in records:
        step_arr = np.array([h[0] for h in r["steps_history"]], dtype=float)
        cost_arr = np.array([h[1] for h in r["steps_history"]], dtype=float)
        if kind == "ising":
            cost_arr = (cost_arr - E0) / (E1 - E0)
        if step_arr[-1] < grid[-1]:  # last-value fill beyond the recorded end
            step_arr = np.concatenate([step_arr, [grid[-1]]])
            cost_arr = np.concatenate([cost_arr, [cost_arr[-1]]])
        ys.append(np.interp(grid, step_arr, cost_arr))
    return np.vstack(ys)


def _add_curves(ax, arm_records, kind, E0, E1, grid, label, color):
    mat = _resample(arm_records, kind, E0, E1, grid)
    mean = mat.mean(axis=0)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(300):
        idx = rng.integers(0, mat.shape[0], mat.shape[0])
        boots.append(mat[idx].mean(axis=0))
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    ax.semilogy(grid, mean, lw=1.5, color=color, label=label)
    ax.fill_between(grid, lo, hi, alpha=0.15, color=color)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = out["config"]["n"]
    E0 = out["config"]["E0"]
    E1 = out["config"]["E1"]
    grid = np.arange(0, out["config"]["max_steps"] + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # --- left: n=8 headline arms ---
    ax = axes[0]
    colors = {"global_random": "#d62728",
              "local_random": "#1f77b4",
              "local_identity": "#2ca02c",
              "local_zero": "#ff7f0e",
              "layerwise_identity": "#9467bd"}
    for name, color in colors.items():
        arm = out["arms"][name]
        _add_curves(ax, arm["records"], arm["kind"], E0, E1, grid,
                    {"global_random": "global infidelity, random",
                     "local_random": "local energy, random",
                     "local_identity": "local energy, identity",
                     "local_zero": "local energy, zero",
                     "layerwise_identity": "local energy, layerwise"}[name],
                    color)
    ax.set_xlabel("step")
    ax.set_ylabel("rel. energy error / infidelity")
    ax.set_title(f"E3 headline (n={n}, M={2 * n + 1})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)

    # --- right: n=10 scaling comparison ---
    sc = out.get("scaling")
    if sc:
        ax2 = axes[1]
        from bpqnn.hamiltonian import ising_ground_state as _igs
        E0_10, gs10, w10 = _igs(10, h=1.0)
        E1_10 = float(w10[1])
        grid10 = np.arange(0, 401.0)
        for name, color in (("global_random", "#d62728"),
                            ("global_identity", "#e377c2"),
                            ("local_random", "#1f77b4"),
                            ("local_identity", "#2ca02c")):
            rec = sc["arms"][name]["records"]
            kind = "infidelity" if name.startswith("global") else "ising"
            _add_curves(ax2, rec, kind, E0_10, E1_10, grid10, name, color)
        ax2.set_xlabel("step")
        ax2.set_ylabel("rel. energy error / infidelity")
        ax2.set_title("E3 scaling (n=10, M=21)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e3_training.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())