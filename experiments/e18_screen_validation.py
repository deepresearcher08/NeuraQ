"""E18: does the screen predict real (shot-limited) training outcomes?

The E16 screen labels a configuration trainable / marginal / barren from the
init-time gradient variance and its decay in width. E3 showed that with *exact*
gradients Adam's per-parameter normalisation masks the decay, so every arm
trains; the plateau only bites when gradients are estimated from finite shots
(E4/D8). This experiment therefore checks the screen's verdict against what
actually happens under a shot-limited parameter-shift Adam: a barren
configuration should fail to reduce its cost, a trainable one should succeed.

We sweep ``(n, depth, cost)`` in pure simulation, train each configuration with
a shot-noisy gradient, and score the screen's verdict as a binary predictor
(``trainable``/``marginal`` -> predict success, ``barren plateau`` -> predict
failure). Chunking matches the |0...0> protocol of E1/E2a/E15.

Run:  python experiments/e18_screen_validation.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.costs import (
    global_cost_zero,
    local_cost_zero,
    single_qubit_zero_probs,
    zero_prob,
)
from bpqnn.init import random_init
from bpqnn.screen import screen
from bpqnn.simulator import StatevectorSimulator, default_axes

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

NS = [4, 6, 8, 10]
DEPTHS = [3, 6]
COSTS = ("global", "local")
SHOTS = 2000
STEPS = 60
LR = 0.08
SEEDS = 3
SUCCESS_THRESHOLD = 0.25
SCREEN_SAMPLES = 128
SHIFT = np.pi / 2.0

_POSITIVE_VERDICTS = ("trainable", "marginal")


def exact_cost(theta, n, cost, edges, axes) -> float:
    sim = StatevectorSimulator(n, 1)
    sim.apply_circuit(np.asarray(theta, dtype=np.float64), edges, axes)
    if cost == "global":
        return float(global_cost_zero(sim.state)[0])
    return float(local_cost_zero(sim.state, n)[0])


def _batched_cost_and_var(theta_batch, n, cost, edges, axes, shots):
    sim = StatevectorSimulator(n, theta_batch.shape[0])
    sim.apply_circuit(theta_batch, edges, axes)
    if cost == "global":
        p0 = zero_prob(sim.state)
        var_est = p0 * (1.0 - p0) / shots
        return 1.0 - p0, var_est
    p = single_qubit_zero_probs(sim.state, n)
    var_est = (p * (1.0 - p)).mean(axis=1) / n / shots
    return 1.0 - p.mean(axis=1), var_est


def shot_gradient(theta, n, cost, edges, axes, shots, rng):
    M, nq = theta.shape
    shifts, index = [], []
    for l in range(M):
        for q in range(nq):
            tp = theta.copy()
            tp[l, q] += SHIFT
            tm = theta.copy()
            tm[l, q] -= SHIFT
            shifts.append(tp)
            shifts.append(tm)
            index.append((l, q))
    c, var = _batched_cost_and_var(np.array(shifts), n, cost, edges, axes, shots)
    c = c + rng.normal(0.0, np.sqrt(np.maximum(var, 0.0)))
    g = np.empty((M, nq), dtype=np.float64)
    for k, (l, q) in enumerate(index):
        g[l, q] = 0.5 * (c[2 * k] - c[2 * k + 1])
    return g


def train_shot(theta0, n, cost, edges, axes, shots, steps, lr, rng):
    theta = theta0.copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t in range(1, steps + 1):
        g = shot_gradient(theta, n, cost, edges, axes, shots, rng)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        theta = theta - lr * (m / (1.0 - 0.9 ** t)) / (
            np.sqrt(v / (1.0 - 0.999 ** t)) + 1e-8
        )
    return theta


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"ns": NS, "depths": DEPTHS, "costs": list(COSTS),
                      "shots": SHOTS, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                      "success_threshold": SUCCESS_THRESHOLD,
                      "screen_samples": SCREEN_SAMPLES}}

    print("=" * 90)
    print("E18: does the screen predict shot-limited training outcomes?")
    print("=" * 90)
    t_all = time.time()

    rows = []
    for n in NS:
        edges = [(q, q + 1) for q in range(n - 1)]
        for M in DEPTHS:
            axes = default_axes(M)
            for cost in COSTS:
                succ = []
                finals = []
                for s in range(SEEDS):
                    rng = np.random.default_rng(7000 + 100 * n + 10 * M + s)
                    th0 = random_init(M, n, rng)
                    c0 = exact_cost(th0, n, cost, edges, axes)
                    theta = train_shot(th0, n, cost, edges, axes, SHOTS, STEPS,
                                       LR, rng)
                    cf = exact_cost(theta, n, cost, edges, axes)
                    finals.append(cf)
                    succ.append(cf < SUCCESS_THRESHOLD)
                success_rate = float(np.mean(succ))
                actual_success = success_rate >= 0.5

                rep = screen(n=n, M=M, cost=cost, init="random",
                             N=SCREEN_SAMPLES, seed=0)
                verdict = rep["verdict"]
                predicted_success = verdict in _POSITIVE_VERDICTS
                rows.append({
                    "n": n, "M": M, "cost": cost,
                    "screen_verdict": verdict,
                    "screen_slope": rep["slope_per_qubit"],
                    "init_cost": c0, "final_cost_median": float(np.median(finals)),
                    "success_rate": success_rate,
                    "actual_success": actual_success,
                    "predicted_success": predicted_success,
                    "correct": bool(predicted_success == actual_success),
                })

    out["rows"] = rows
    print(f"{'n':>3} {'M':>3} {'cost':>7} {'verdict':>15} {'slope':>8} "
          f"{'final':>8} {'succ':>6} {'ok':>4}")
    for r in rows:
        slope = "-" if r["screen_slope"] is None else f"{r['screen_slope']:+.2f}"
        print(f"{r['n']:>3} {r['M']:>3} {r['cost']:>7} {r['screen_verdict']:>15} "
              f"{slope:>8} {r['final_cost_median']:>8.3f} "
              f"{r['success_rate']:>6.2f} {str(r['correct']):>4}")

    tp = sum(1 for r in rows if r["predicted_success"] and r["actual_success"])
    tn = sum(1 for r in rows if not r["predicted_success"] and not r["actual_success"])
    fp = sum(1 for r in rows if r["predicted_success"] and not r["actual_success"])
    fn = sum(1 for r in rows if not r["predicted_success"] and r["actual_success"])
    acc = (tp + tn) / len(rows) if rows else float("nan")
    out["confusion"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
                        "accuracy": acc, "n_configs": len(rows)}

    print("-" * 90)
    print(f"screen-as-predictor: accuracy={acc:.3f}  "
          f"(TP={tp} TN={tn} FP={fp} FN={fn} of {len(rows)})")
    print(f"(total {time.time() - t_all:.1f}s)")

    with open(RESULTS / "e18_screen_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULTS / 'e18_screen_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
