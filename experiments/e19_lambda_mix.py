"""E19: a bounded local->global schedule (lambda-mix).

For target ``|0...0>`` define the blended cost

    C(lam) = (1 - lam) C_L + lam C_G,

with ``lam`` ramped 0 -> 1. This is principled because for these normalisations
``C_L <= C_G <= n C_L`` (Cerezo et al. 2021; verified numerically below), so
driving the trainable ``C_L`` down also drives the faithful ``C_G`` down. The
ramp starts on the trainable objective and finishes on the global one, rather
than starting on the plateau.

Two regimes are reported, because they answer different questions:

  * ``exact`` -- analytic parameter-shift gradients. E3 showed Adam's
    per-parameter normalisation masks the gradient decay here, so *all* arms
    reach a low global cost; this is the optimistic upper bound.
  * ``shots`` -- gradients estimated from a finite number of shots (the regime
    where E4/D8/E18 found the plateau actually bites). Here the tiny global
    signal is swamped by shot noise and the pure-global arm should stall while
    the local and ramped arms still descend.

Run:  python experiments/e19_lambda_mix.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.csa import adam_mix
from bpqnn.costs import global_cost_zero, local_cost_zero, single_qubit_zero_probs, zero_prob
from bpqnn.init import random_init
from bpqnn.simulator import StatevectorSimulator, default_axes

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

NS = [8, 10]
STEPS = 150
LR = 0.05
SEEDS = 4
SHOTS = 4096
R_BOUND = 256
SHIFT = np.pi / 2.0
SUCCESS = 0.1

ARMS = {
    "local": {"lam_max": 0.0, "schedule": "constant"},
    "global": {"lam_max": 1.0, "schedule": "constant"},
    "mix_ramp": {"lam_max": 1.0, "schedule": "linear"},
}


def _final(theta, n, edges, axes):
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(np.asarray(theta, np.float64), edges, axes)
    return (float(global_cost_zero(sim.state)[0]),
            float(local_cost_zero(sim.state, n)[0]))


def _lam(name, t, steps):
    if name == "local":
        return 0.0
    if name == "global":
        return 1.0
    return min(1.0, (t - 1) / max(1, steps - 1))  # linear ramp for mix_ramp


def _mix_cost_var(theta_batch, n, lam, edges, axes, shots):
    """Blended cost and its sampling variance (independent-endpoint model)."""
    sim = StatevectorSimulator(n, theta_batch.shape[0])
    sim.apply_circuit(theta_batch, edges, axes)
    p0 = zero_prob(sim.state)
    p = single_qubit_zero_probs(sim.state, n)
    cg, cl = 1.0 - p0, 1.0 - p.mean(axis=1)
    c = (1.0 - lam) * cl + lam * cg
    var_l = (p * (1.0 - p)).mean(axis=1) / n / shots
    var_g = p0 * (1.0 - p0) / shots
    var = (1.0 - lam) ** 2 * var_l + lam ** 2 * var_g
    return c, var


def _shot_mix_grad(theta, n, lam, edges, axes, shots, rng):
    M, nq = theta.shape
    shifts, index = [], []
    for l in range(M):
        for q in range(nq):
            tp = theta.copy(); tp[l, q] += SHIFT
            tm = theta.copy(); tm[l, q] -= SHIFT
            shifts.append(tp); shifts.append(tm); index.append((l, q))
    c, var = _mix_cost_var(np.array(shifts), n, lam, edges, axes, shots)
    c = c + rng.normal(0.0, np.sqrt(np.maximum(var, 0.0)))
    g = np.empty((M, nq), dtype=np.float64)
    for k, (l, q) in enumerate(index):
        g[l, q] = 0.5 * (c[2 * k] - c[2 * k + 1])
    return g


def train_shot_mix(theta0, n, name, edges, axes, shots, steps, lr, rng):
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        lam = _lam(name, t, steps)
        g = _shot_mix_grad(theta, n, lam, edges, axes, shots, rng)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * (m / (1.0 - 0.9 ** t)) / (
            np.sqrt(v / (1.0 - 0.999 ** t)) + 1e-8
        )
    return theta


def bound_check(n, seed=0):
    """Max violations of C_L <= C_G <= n C_L over random states."""
    rng = np.random.default_rng(seed)
    M = n
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(R_BOUND, M, n))
    sim = StatevectorSimulator(n, R_BOUND)
    sim.apply_circuit(theta, [(q, q + 1) for q in range(n - 1)], default_axes(M))
    cg = global_cost_zero(sim.state)
    cl = local_cost_zero(sim.state, n)
    return {
        "n": n,
        "max_violation_low": float(np.max(np.clip(cl - cg, 0.0, None))),
        "max_violation_high": float(np.max(np.clip(cg - n * cl, 0.0, None))),
        "max_ratio_global_over_local": float(np.max(cg / np.maximum(cl, 1e-12))),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"ns": NS, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                      "shots": SHOTS, "success_threshold": SUCCESS},
           "normalisation_bound": [bound_check(n, seed=1) for n in NS]}

    print("=" * 86)
    print("E19: bounded local->global schedule, target |0...0>")
    print("=" * 86)
    for b in out["normalisation_bound"]:
        print(f"  n={b['n']:>2}: C_L<=C_G violation={b['max_violation_low']:.2e}  "
              f"C_G<=n C_L violation={b['max_violation_high']:.2e}  "
              f"max C_G/C_L={b['max_ratio_global_over_local']:.2f} (<=n)")
    t_all = time.time()

    out["results"] = {}
    for n in NS:
        M = n
        edges = [(q, q + 1) for q in range(n - 1)]
        axes = default_axes(M)
        out["results"][str(n)] = {}
        for regime in ("exact", "shots"):
            out["results"][str(n)][regime] = {}
            print("-" * 86)
            print(f"n={n} (M={M})  regime={regime}")
            print(f"{'arm':>10} {'final C_G med':>14} {'IQR':>20} "
                  f"{'final C_L med':>14} {'success':>8}")
            for name, kw in ARMS.items():
                cgs, cls, succ = [], [], []
                for s in range(SEEDS):
                    rng = np.random.default_rng(100 + s)
                    th0 = random_init(M, n, rng)
                    if regime == "exact":
                        res = adam_mix(th0, n, steps=STEPS, lr=LR, edges=edges,
                                       axes=axes, **kw)
                        theta = res["theta"]
                    else:
                        theta = train_shot_mix(th0, n, name, edges, axes, SHOTS,
                                               STEPS, LR, rng)
                    cg, cl = _final(theta, n, edges, axes)
                    cgs.append(cg); cls.append(cl); succ.append(cg < SUCCESS)
                cgs = np.array(cgs)
                out["results"][str(n)][regime][name] = {
                    "final_cost_G_median": float(np.median(cgs)),
                    "final_cost_G_iqr": [float(np.percentile(cgs, 25)),
                                         float(np.percentile(cgs, 75))],
                    "final_cost_L_median": float(np.median(cls)),
                    "success_rate": float(np.mean(succ)),
                }
                r = out["results"][str(n)][regime][name]
                print(f"{name:>10} {r['final_cost_G_median']:>14.4f} "
                      f"[{r['final_cost_G_iqr'][0]:.4f},{r['final_cost_G_iqr'][1]:.4f}]"
                      f"{'':>6} {r['final_cost_L_median']:>14.4f} "
                      f"{r['success_rate']:>8.2f}")

    with open(RESULTS / "e19_lambda_mix.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 86)
    print(f"saved {RESULTS / 'e19_lambda_mix.json'}  "
          f"(total {time.time() - t_all:.1f}s)")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e19_lambda_mix.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = out["config"]["ns"]
    arms = ("local", "global", "mix_ramp")
    regimes = ("exact", "shots")
    x = np.arange(len(ns))
    w = 0.25
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, regime in zip(axs, regimes):
        for i, arm in enumerate(arms):
            vals = [out["results"][str(n)][regime][arm]["final_cost_G_median"]
                    for n in ns]
            ax.bar(x + (i - 1) * w, vals, w, label=arm)
        ax.axhline(out["config"]["success_threshold"], color="k", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([f"n={n}" for n in ns])
        ax.set_title(f"{regime} gradients")
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].set_ylabel("final global cost C_G (median)")
    axs[0].legend(fontsize=8)
    fig.suptitle("E19: ramped local->global blend stays trainable under shots")
    fig.tight_layout()
    fig.savefig(RESULTS / "e19_lambda_mix.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
