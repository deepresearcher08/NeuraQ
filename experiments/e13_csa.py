"""E13: Cost-Support scheduling/spreading on the E11 support-radius axis.

E11 measured ``Var(dO) ~ C e^{-kr}`` in the observable's support radius ``r``.
E13 asks the design question E11 poses: *if trainability is set by r, how should
the optimizer place its observable support?* Three candidate policies, all on the
all-zero target (global metric = the E1 global cost ``C_n = 1 - P(0...0)``):

  * **grow the support** (CSA: anneal ``r`` from 1 to the target) -- linear,
    staged (hold-10), and variance-gated (admission from the measured gradient
    power, the law-as-control-law version);
  * **leaving it fixed** as a *single block* of radius ``r`` (E11 family).
    ``r=1`` (qubit-0 block) and ``r=n//2`` (regional) are the *unfaithful* arms
    -- they cannot see the rest of the register; ``r=n`` is the faithful global
    cost (the E1 plateau point);
  * **spreading it** (window-averaged cost ``C_w(r)``, E2a's local cost): the
    windows tile the register, so it is *faithful for the all-zero target at any
    ``r >= 1``*, while each observable's support stays small (E11's gentler
    averaged rate, 0.98-1.63 x ln2 vs the single-block 1.56-2.01 x ln2).

Predictions to test:
  (a) plateau-vs-trainability: the E1/E11 initial-variance plateau of ``C_n``
      need **not** prevent Adam from reaching the optimum -- the target is a
      reachable product state;
  (b) single-block costs are *unfaithful* at small ``r`` (they leave the rest of
      the register untouched) -- global cost stays ~``1 - 2^{-(n-r)}``;
  (c) window-averaged costs are faithful **and** low-variance, i.e. support
      *spread* is the constructive answer to the E11 trade-off;
  (d) CSA (support *growth*) is the wrong lever here -- the budget spent at
      small left-anchored blocks is wasted.

Run:  python experiments/e13_csa.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn import init
from bpqnn.csa import (
    adam_block,
    adam_window,
    block_cost_of,
    block_grad,
    csa_adam_train,
    window_grad,
)
from bpqnn.simulator import default_axes

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

NS = [12, 14]
STEPS = 250
LR = 0.05
SEEDS = 4
HOLD = 10
VAR_GATE_EVERY = 5
VAR_FLOOR = 0.25
N_INIT_VAR = 256

CSA_SCHEDULES = ["linear", "staged", "var_gated"]


def medians(vals):
    a = np.asarray(vals)
    return (float(np.median(a)), float(np.quantile(a, 0.25)),
            float(np.quantile(a, 0.75)))


def e11_kappa():
    p = RESULTS / "e11_support_radius.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("single_block_kappa", None)
        except Exception:
            return None
    return None


def init_grad_variance(n, M, gfun, r, edges, axes, rng_init=0):
    vals = []
    for s in range(N_INIT_VAR):
        th0 = init.random_init(M, n, np.random.default_rng(1000 + s))
        g = gfun(th0, n, r, edges, axes)
        vals.append(np.mean(np.abs(g) ** 2))
    return float(np.mean(vals))


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"n": NS, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                      "hold": HOLD, "var_floor": VAR_FLOOR, "n_init_var": N_INIT_VAR,
                      "e11_single_block_kappa": e11_kappa()}}
    print("=" * 94)
    print("E13: support scheduling/spreading on the E11 axis "
          "(target |0...0>, metric = global cost C_n)")
    t_all = time.time()
    results = {}

    for n in NS:
        M = n
        axes = default_axes(M)
        edges = [(q, q + 1) for q in range(n - 1)]
        r_star2 = n // 2
        results[n] = {}
        arms = {}

        def arm(key, fn):
            ts = []
            for s in range(SEEDS):
                th0 = init.random_init(M, n, np.random.default_rng(100 + s))
                ts.append(block_cost_of(fn(th0), n, n, edges, axes))
            arms[key] = medians(ts)
            return arms[key]

        print(f"n={n}  (M={M}, dim={1 << n})")
        arm("local_r1", lambda th0: adam_block(th0, n, 1, steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("regional_rstar", lambda th0: adam_block(th0, n, r_star2, steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("global_rn", lambda th0: adam_block(th0, n, n, steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("window_r2", lambda th0: adam_window(th0, n, 2, steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("window_r4", lambda th0: adam_window(th0, n, 4, steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("csa_linear", lambda th0: csa_adam_train(th0, n, n, schedule="linear", steps=STEPS, lr=LR, axes=axes)["theta"])
        arm("csa_staged", lambda th0: csa_adam_train(th0, n, n, schedule="staged", steps=STEPS, lr=LR, hold=HOLD, axes=axes)["theta"])
        res = arm("csa_var_gated", lambda th0: csa_adam_train(th0, n, n, schedule="var_gated", steps=STEPS, lr=LR, gate_every=VAR_GATE_EVERY, floor_frac=VAR_FLOOR, axes=axes)["theta"])
        arms["csa_var_gated"] = res

        # initial gradient variance (the E11-lens "why"): single-block vs spread
        var_arms = {
            "var_init_block_r1": init_grad_variance(n, M, block_grad, 1, edges, axes),
            "var_init_block_rn": init_grad_variance(n, M, block_grad, n, edges, axes),
            "var_init_window_r2": init_grad_variance(n, M, window_grad, 2, edges, axes),
            "var_init_window_r4": init_grad_variance(n, M, window_grad, 4, edges, axes),
        }
        results[n] = {"arms": {k: list(v) for k, v in arms.items()}, **var_arms}

        print(f"{'arm':>20} {'final global cost (median)':>26}")
        for k, (md, lo, hi) in arms.items():
            print(f"{k:>20} {md:>24.4f}   [{lo:.4f},{hi:.4f}]")
        for k, v in var_arms.items():
            print(f"  {k} = {v:.3e}")

    out["results"] = results

    # ------------- verdicts -------------
    vd = []
    for n in NS:
        a = results[n]["arms"]
        vd.append({
            "n": n,
            "local_r1": a["local_r1"],
            "regional_rstar": a["regional_rstar"],
            "global_rn": a["global_rn"],
            "window_r2": a["window_r2"],
            "window_r4": a["window_r4"],
            "csa_linear": a["csa_linear"],
            "csa_staged": a["csa_staged"],
            "csa_var_gated": a["csa_var_gated"],
            "var_init_block_rn": results[n]["var_init_block_rn"],
            "var_init_window_r2": results[n]["var_init_window_r2"],
            "var_ratio_block_rn_over_window_r2": (
                results[n]["var_init_block_rn"] / results[n]["var_init_window_r2"]
                if results[n]["var_init_window_r2"] else float("nan")),
        })
    out["verdict"] = vd
    print("-" * 94)
    for v in vd:
        print(v)
    with open(RESULTS / "e13_csa.json", "w") as f:
        json.dump(_plain(out), f, indent=2)
    print(f"saved {RESULTS / 'e13_csa.json'}  (total {time.time() - t_all:.1f}s)")
    return 0


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


if __name__ == "__main__":
    raise SystemExit(main())