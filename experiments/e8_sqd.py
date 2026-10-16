"""E8: SQD/QSCI simulability + contrast probe (SPEC D9/E8).

Sample-based quantum diagonalisation (SQD/QSCI): sample computational-basis
bitstrings from a state, build the sub-space they span (plus, for the TFIM, all
single-bit flips, i.e. the Hamiltonian-connected configurations), project ``H``
into it, and diagonalise *classically*. No parameter gradients are used -- SQD
is a gradient-free post-processing of quantum samples.

We compare three samplers at fixed depth, on the critical (h=1) and gapped
(h=2) TFIM:

  exact_gs   -- samples from the true ground state (best possible support)
  trained    -- samples from an equal-budget trained HEA (identity init, local
                Ising cost, period-4 folded schedule, M = 2n+1)
  random     -- samples from a random HEA

For each we report the SQD relative energy error (D5 metric) vs number of
samples, the number of samples needed to reach rel-error <= 1e-2, and the
half-chain entanglement entropy of the states involved (D9: needed for the
simulability reading, though the subspace test is the actual probe).

Honest reading this experiment is set up to support: at these widths the whole
Hilbert space is classically diagonalisable, so *any* sufficiently entropic
sampler eventually recovers the ground energy once its samples cover the space
-- the interesting question is the sample efficiency of each sampler and the
entanglement structure, not a quantum advantage. Frames per the Cerezo et al.
perspective ("Yes and No"), see SPEC D9.

Run:  python experiments/e8_sqd.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.hamiltonian import ising_ground_state, ising_hamiltonian
from bpqnn.init import identity_block_init, random_init
from bpqnn.sqd import (
    half_chain_entropy,
    qsci_indices,
    rel_energy_error,
    sample_pool,
    sqd_ground_energy,
)
from bpqnn.train import adam_train, lbfgs_train
from experiments.e2b_init_layerwise import axes_folded, run_state

N_LIST = [8, 10]
H_LIST = [1.0, 2.0]
M_OF = lambda n: 2 * n + 1  # noqa: E731
TRAIN_STEPS = 400
TRAIN_LR = 0.05
SAMPLES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
TRIALS = 5
THRESHOLD = 1e-2
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SAMPLERS = ("exact_gs", "trained", "random")


def sampler_states(n, h, seed):
    """Return {sampler_name: statevector} plus training diagnostics."""
    edges = [(q, q + 1) for q in range(n - 1)]
    M = M_OF(n)
    axes = axes_folded(M)
    rng = np.random.default_rng(seed)

    E0, gs, w = ising_ground_state(n, h=h)

    theta0 = identity_block_init(M, n, rng)
    res = adam_train(theta0, n=n, kind="ising", h=h, edges=edges, axes=axes,
                     lr=TRAIN_LR, max_steps=TRAIN_STEPS)
    res = lbfgs_train(res["theta"], n=n, kind="ising", h=h, edges=edges,
                      axes=axes, maxiter=500)
    psi_trained = run_state(n, edges, axes, res["theta"])

    psi_random = run_state(n, edges, axes, random_init(M, n, rng))

    return {
        "states": {"exact_gs": gs, "trained": psi_trained, "random": psi_random},
        "E0": E0,
        "E1": float(w[1]),
        "trained_rel_error": rel_energy_error(res["cost"], E0, float(w[1])),
        "trained_infidelity": float(1.0 - abs(np.vdot(gs, psi_trained)) ** 2),
    }


def sqd_curve(state, n, H, E0, E1, seeds):
    """Mean/std SQD rel-energy error vs sample budget over independent trials."""
    dim = 1 << n
    max_s = max(SAMPLES)
    curves = {"mean": [], "std": []}
    for k in SAMPLES:
        errs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            pool = sample_pool(state, n, max_s, rng)[:k]
            idx = qsci_indices(np.unique(pool), n)
            if idx.size == dim:
                errs.append(0.0)
            else:
                Emin, _, _ = sqd_ground_energy(H, idx)
                errs.append(rel_energy_error(Emin, E0, E1))
        curves["mean"].append(float(np.mean(errs)))
        curves["std"].append(float(np.std(errs)))
    return curves


def samples_to_threshold(samples, mean, thr=THRESHOLD):
    for k, m in zip(samples, mean):
        if m <= thr:
            return int(k)
    return None


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"n": N_LIST, "h": H_LIST, "M": "2n+1",
                      "train_steps": TRAIN_STEPS, "samples": SAMPLES,
                      "trials": TRIALS, "threshold": THRESHOLD}, "cases": {}}

    print("=" * 78)
    print("E8 SQD/QSCI probe: samples -> classical subspace -> ground energy")
    print("=" * 78)

    for n in N_LIST:
        for h in H_LIST:
            key = f"n{n}_h{h:g}"
            H = ising_hamiltonian(n, h=h)
            info = sampler_states(n, h, seed=1000 * n + int(10 * h))
            E0, E1 = info["E0"], info["E1"]
            seed_base = 70000 + 100 * n + int(h)
            seeds = [seed_base + 13 * t for t in range(TRIALS)]

            case = {
                "E0": E0, "E1": E1,
                "trained_rel_error": info["trained_rel_error"],
                "trained_infidelity": info["trained_infidelity"],
                "entropies": {name: half_chain_entropy(st, n)
                              for name, st in info["states"].items()},
                "curves": {},
                "samples_to_threshold": {},
            }
            print("-" * 78)
            print(f"n={n} h={h:g}  E0={E0:+.4f}  E1={E1:+.4f}  "
                  f"trained rel-err={info['trained_rel_error']:.2e}  "
                  f"infid={info['trained_infidelity']:.2e}")
            print("  entropy S(n/2): " + "  ".join(
                f"{k}={v:.3f}" for k, v in case["entropies"].items()))
            for name in SAMPLERS:
                c = sqd_curve(info["states"][name], n, H, E0, E1, seeds)
                case["curves"][name] = {"samples": list(SAMPLES), **c}
                thr = samples_to_threshold(SAMPLES, c["mean"])
                case["samples_to_threshold"][name] = thr
                last = c["mean"][-1]
                print(f"    {name:>8}: err@8192={last:.2e}  "
                      f"samples to <= {THRESHOLD:g}: {thr}")
            out["cases"][key] = case

    with open(RESULTS / "e8_sqd.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e8_sqd.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e8_sqd.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"exact_gs": "tab:green", "trained": "tab:blue", "random": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    hs = out["config"]["h"]
    ns = out["config"]["n"]
    sam = out["config"]["samples"]

    for r, h in enumerate(hs):
        for c, n in enumerate(ns):
            ax = axes[r, c]
            case = out["cases"][f"n{n}_h{h:g}"]
            for name in ("exact_gs", "trained", "random"):
                y = case["curves"][name]["mean"]
                ax.loglog(sam, np.maximum(y, 1e-16), "o-",
                          color=colors[name], label=name)
            ax.axhline(out["config"]["threshold"], color="gray", ls=":",
                       label="1e-2")
            ax.set_xlabel("samples")
            ax.set_ylabel("SQD rel. energy error")
            ax.set_title(f"n={n}, h={h:g} "
                         f"(trained err {case['trained_rel_error']:.1e})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        ax = axes[r, 2]
        case_n8 = out["cases"][f"n8_h{h:g}"]
        case_n10 = out["cases"][f"n10_h{h:g}"]
        labels = ("exact_gs", "trained", "random")
        x = np.arange(len(labels))
        s8 = [case_n8["entropies"][k] for k in labels]
        s10 = [case_n10["entropies"][k] for k in labels]
        ax.bar(x - 0.2, s8, 0.4, label="n=8", color="steelblue")
        ax.bar(x + 0.2, s10, 0.4, label="n=10", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("half-chain entropy (base 2)")
        ax.set_title(f"entanglement, h={h:g}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e8_sqd.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
