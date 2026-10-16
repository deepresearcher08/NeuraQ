"""E4: does shot noise make the cost-locality gap visible in training?

Exact-gradient training (E3) showed that Adam's per-parameter normalisation
masks raw-gradient decay, so mitigation did not show up in training success.
This experiment asks the question on a *sampled* pipeline (SPEC D8), where
gradients are estimated from measurements:

  global cost   -- sampled infidelity 1 - |<gs|state>|^2 (one Bernoulli
                   readout of the projector |gs><gs| per circuit execution,
                   i.e. ``shots`` per execution);
  local cost    -- sampled energy <H> (``1 + n`` measurement bases per circuit
                   execution, ``shots`` shots per base, as in D8 accounting).

Phase 1 (calibrate): for n in {6,8,10}, measures the exact raw-gradient RMS
  and the per-component sample noise RMS of the parameter-shift gradient at a
  fixed per-execution shot count. SNR = exact_rms / noise_rms. Expectation:
  local SNR stays ~O(1) while global SNR collapses with n (the E2a variance
  gap, now on the estimator).

Phase 2 (train): equal *circuit executions* per step (D2: cost=1 eval,
  gradient=2*M*n+1 evals, identical per arm); Adam (D1 hyperparameters);
  same per-seed init across arms; exact metrics recomputed after every step.
  Default config: n=10, M=21, s=512 (chosen from the phase-1 calibration so
  that local SNR ~2.3 > 1 > global SNR ~0.8 at equal per-execution shots).
  Arms:
    local_512       energy cost, 512 shots per base (works)
    local_46        energy cost, 512/11 shots per base (same *total shots*
                    per step as global_512: basis-splitting cost)
    global_512      infidelity cost, 512 shots per execution (noise-limited)
    global_4096     infidelity cost, 8x shots (SNR ~2.2, recovers)
    local_exact     adjoint control (energy)
    global_exact    adjoint control (infidelity)
  Reported by median exact rel-energy-error vs step, plus the fraction of
  seeds that achieve a 25% error reduction ("useful descent").

Run:  python experiments/e4_shot_training.py --phase calibrate --s 4096
      python experiments/e4_shot_training.py --phase train --n 10 --s 512
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.init import random_init
from bpqnn.simulator import StatevectorSimulator
from bpqnn.shots import (
    estimate_ising_energy,
    sampled_infidelity_grad_cost,
    sampled_ising_grad_cost,
)
from experiments.e2b_init_layerwise import axes_folded, run_state

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def _snr(exact, noise):
    return float(np.sqrt(np.mean(exact ** 2)) / np.sqrt(np.mean(noise ** 2)))


def calibrate(n: int, s: int, n_inits: int = 4, reps: int = 8):
    """Return per-cost SNR table at (n, s)."""
    M = 2 * n + 1
    E0, gs, w = ising_ground_state(n, h=1.0)
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = axes_folded(M)
    row = {"n": n, "M": M, "s": s, "costs": {}}
    for kind in ("ising", "infidelity"):
        exacts, noises = [], []
        for i in range(n_inits):
            rng = np.random.default_rng(7000 + i)
            t0 = random_init(M, n, rng)
            g_exact = adjoint_gradient(t0, n, edges, axes, kind=kind,
                                       h=1.0, target=gs)
            gsamp = []
            for r in range(reps):
                grng = np.random.default_rng(8000 + i * 1000 + r)
                if kind == "ising":
                    g = sampled_ising_grad_cost(t0, n, 1.0, s, grng,
                                                edges=edges, axes=axes)
                else:
                    g = sampled_infidelity_grad_cost(
                        t0, n, gs, s, grng, edges=edges, axes=axes)
                gsamp.append(g)
            gsamp = np.stack(gsamp)
            per_comp_std = gsamp.std(axis=0, ddof=1)
            exacts.append(g_exact)
            noises.append(per_comp_std)
        exact_rms = float(np.median([np.sqrt(np.mean(e ** 2)) for e in exacts]))
        noise_rms = float(np.median([np.sqrt(np.mean(v ** 2)) for v in noises]))
        row["costs"][kind] = {"exact_rms": exact_rms, "noise_rms": noise_rms,
                              "snr": _snr(np.array(exact_rms),
                                          np.array(noise_rms))}
    return row


def _rel_err(state, n, E0, E1):
    E = float(ising_energy(state, n, h=1.0)[0])
    return (E - E0) / (E1 - E0)


def train(n: int, s: int, seeds: int = 6, steps: int = 30, beta1: float = 0.9):
    """Equal-execution-budget shot-noisy Adam comparison (D1 hyperparams)."""
    M = 2 * n + 1
    E0, gs, w = ising_ground_state(n, h=1.0)
    E1 = float(w[1])
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = axes_folded(M)
    evals_per_step = 1 + 2 * M * n

    def _run(cost, shots, t0, rng):
        theta = t0.copy()
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        hist = []
        for t in range(1, steps + 1):
            if cost.startswith("exact_"):
                kind = cost[len("exact_"):]
                g = adjoint_gradient(theta, n, edges, axes, kind=kind,
                                     h=1.0, target=gs)
            else:
                grng = np.random.default_rng(rng.integers(2 ** 31 - 1))
                if cost == "ising":
                    g = sampled_ising_grad_cost(theta, n, 1.0, shots, grng,
                                                edges=edges, axes=axes)
                else:
                    g = sampled_infidelity_grad_cost(theta, n, gs, shots, grng,
                                                     edges=edges, axes=axes)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            mhat = m / (1.0 - beta1 ** t) if beta1 > 0 else g
            vhat = v / (1.0 - 0.999 ** t)
            theta = theta - 0.05 * mhat / (np.sqrt(vhat) + 1e-8)
            sim = StatevectorSimulator(n, 1)
            sim.apply_circuit(theta, edges, axes)
            hist.append({"step": t,
                         "rel_err": _rel_err(sim.state, n, E0, E1),
                         "infidelity": float(
                             1.0 - abs(np.vdot(gs, sim.state.reshape(-1))) ** 2)})
        return theta, hist

    arms = {"local_%d" % s: ("ising", s),
            "local_%d" % max(1, s // (1 + n)): ("ising",
                                                max(1, s // (1 + n))),
            "global_%d" % s: ("infidelity", s),
            "global_%d" % (8 * s): ("infidelity", 8 * s),
            "local_exact": ("exact_ising", 0),
            "global_exact": ("exact_infidelity", 0)}
    out = {"config": {"n": n, "M": M, "s": s, "steps": steps, "seeds": seeds,
                      "evals_per_step": evals_per_step, "E0": E0, "E1": E1},
           "arms": {}}
    for name, (cost, shots) in arms.items():
        records = []
        for k in range(seeds):
            rng = np.random.default_rng(9500 + k)
            t0 = random_init(M, n, rng)
            start = _rel_err(run_state(n, edges, axes, t0), n, E0, E1)
            _, hist = _run(cost, shots, t0, rng)
            end = hist[-1]["rel_err"]
            records.append({
                "seed": k, "start_rel": float(start), "end_rel": float(end),
                "improved": float(end <= 0.75 * start),
                "history": hist})
        meds = np.median(
            np.array([[h["rel_err"] for h in r["history"]]
                      for r in records]),
            axis=0)
        out["arms"][name] = {
            "cost": cost, "shots_per_exec": shots,
            "shots_per_step": float(evals_per_step * shots),
            "records": records,
            "summary": {
                "start_median": float(np.median([r["start_rel"]
                                                 for r in records])),
                "end_median": float(np.median([r["end_rel"]
                                              for r in records])),
                "improved_fraction": float(np.mean([r["improved"]
                                                    for r in records])),
                "curve": [float(x) for x in meds],
            },
        }
        print(f"{name:<12} shots/exec={shots if shots else 'exact':>6}  "
              f"start={out['arms'][name]['summary']['start_median']:.3f} "
              f"end={out['arms'][name]['summary']['end_median']:.3f} "
              f"improved={out['arms'][name]['summary']['improved_fraction']:.2f}",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("calibrate", "train"), default="calibrate")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--s", type=int, default=512)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--beta1", type=float, default=0.9)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e4_shot_training.json"
    if args.phase == "calibrate":
        rows = []
        for n in (6, 8, 10):
            rows.append(calibrate(n, args.s))
            print(f"n={n} s={args.s}: " + "  ".join(
                f"{k} SNR={v['snr']:.2f} (exact {v['exact_rms']:.1e}, "
                f"noise {v['noise_rms']:.1e})"
                for k, v in rows[-1]["costs"].items()), flush=True)
        rows.append(calibrate(8, 8 * args.s))
        print(f"n=8 s={8 * args.s}: " + "  ".join(
            f"{k} SNR={v['snr']:.2f}" for k, v in rows[-1]["costs"].items()),
            flush=True)
        data = {"calibration": rows}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"saved {path}")
    else:
        data = {}
        if path.exists():
            data = json.load(open(path))
        key = "training" if args.beta1 >= 0.9 else "training_nomom"
        data[key] = train(args.n, args.s, args.seeds, args.steps, args.beta1)
        data["latest"] = key
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        try:
            _plot(data, key)
            print(f"saved {RESULTS / ('e4_' + key + '.png')}")
        except Exception as e:  # noqa: BLE001
            print(f"(plot skipped: {e})")
    return 0


def _plot(data, key="training"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = data[key]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name, arm in tr["arms"].items():
        exact = arm["cost"].startswith("exact_")
        if exact:
            color, style = "k", "--"
        else:
            color = "#1f77b4" if arm["cost"] == "ising" else "#d62728"
            style = "-"
        ax.plot(range(1, len(arm["summary"]["curve"]) + 1),
                arm["summary"]["curve"], lw=1.5, color=color, ls=style,
                marker="o", ms=3,
                label=f"{name}" + ("" if exact else
                                  f" ({arm['shots_per_exec']} shots/exec)"))
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("step (equal circuit executions/step, D2)")
    ax.set_ylabel("median exact rel. energy error")
    ax.set_title("E4 %s: shot-noisy Adam at n=%d, s=%d"
                 % (key, tr["config"]["n"], tr["config"]["s"]))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / ("e4_" + key + ".png"), dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())