"""E9: chi-swept MPS surrogate for the trained QNN (SPEC D9 / E4-simulability).

The other half of D9 (E8 §14l covers the entanglement half). We ask the D9.1
question directly: *does a small-bond-dimension MPS reproduce the trained QNN's
energy to within the QNN's own error?*

Procedure per ``(n, h)``:

  1. Train the equal-budget HEA (identity init + local Ising cost, Adam 400 +
     L-BFGS 500, M = 2n+1 folded -- same as E8), giving ``psi_trained`` and its
     relative energy error ``eps_Q`` against the exact ``E0``.
  2. Take two reference states -- the exact ground state and ``psi_trained`` --
     convert each to an MPS by sequential SVD, and reconstruct it at maximum
     bond dimension ``chi`` (a single-pass left-to-right compression).
  3. Sweep ``chi`` up to the exact rank ``2**floor(n/2)``. Report the compressed
     state's relative energy error ``(E_chi - E0)/(E1 - E0)``, its fidelity to
     the reference, and the truncation weight.

The surrogate test uses two thresholds:
  - **reproduce**: smallest ``chi`` with ``|E_chi - E_trained| <= eps_Q`` -- the
    MPS matches the QNN's energy to within the QNN's own error (the literal
    D9.1 statement);
  - **D5**: smallest ``chi`` reaching relative error ``<= 1e-2``.

Honest framing (D9 / Cerezo et al. perspective): the maximal bond of an exact
``n``-qubit state is only ``2**floor(n/2)`` (16 at n=8, 32 at n=10), so the
*whole* state is an MPS of modest bond dimension and the answer is expected to
be "a tiny chi suffices" -- i.e. the trained model is classically simulable in
this regime, **not** a quantum advantage. The chi-sweep quantifies *how* small
and contrasts the critical (h=1) against the gapped (h=2) point.

Run:  python experiments/e9_mps.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.init import identity_block_init
from bpqnn.mps import (
    compress,
    max_bond_dimension,
    schmidt_spectrum,
    significant_rank,
    statevector_to_mps,
    truncation_weight,
)
from bpqnn.sqd import rel_energy_error
from bpqnn.train import adam_train, lbfgs_train
from experiments.e2b_init_layerwise import axes_folded, run_state

N_LIST = [8, 10]
H_LIST = [1.0, 2.0]
M_OF = lambda n: 2 * n + 1  # noqa: E731
TRAIN_STEPS = 400
TRAIN_LR = 0.05
THRESHOLD = 1e-2
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
SAMPLERS = ("exact_gs", "trained")


def chis_for(n: int) -> list:
    """Powers of two up to the exact max bond dimension 2**floor(n/2)."""
    return [1 << k for k in range(0, n // 2 + 1)]


def trained_state(n, h, seed):
    edges = [(q, q + 1) for q in range(n - 1)]
    M = M_OF(n)
    axes = axes_folded(M)
    rng = np.random.default_rng(seed)

    E0, gs, w = ising_ground_state(n, h=h)
    E1 = float(w[1])

    res = adam_train(identity_block_init(M, n, rng), n=n, kind="ising", h=h,
                     edges=edges, axes=axes, lr=TRAIN_LR, max_steps=TRAIN_STEPS)
    res = lbfgs_train(res["theta"], n=n, kind="ising", h=h, edges=edges,
                      axes=axes, maxiter=500)
    psi = run_state(n, edges, axes, res["theta"])
    E_trained = float(ising_energy(psi, n, h=h)[0])
    return {
        "states": {"exact_gs": gs, "trained": psi},
        "E0": E0, "E1": E1, "E_trained": E_trained,
        "trained_rel_error": rel_energy_error(E_trained, E0, E1),
        "trained_infidelity": float(1.0 - abs(np.vdot(gs, psi)) ** 2),
    }


def sweep(ref_state, n, h, E0, E1, chis):
    """Compress ``ref_state`` at each chi; return per-chi diagnostics."""
    rows = []
    for chi in chis:
        mps = statevector_to_mps(ref_state, n, chi=chi)
        psi = compress(ref_state, n, chi=chi)
        E_chi = float(ising_energy(psi, n, h=h)[0])
        rows.append({
            "chi": int(chi),
            "E": E_chi,
            "rel_error": rel_energy_error(E_chi, E0, E1),
            "fidelity": float(abs(np.vdot(ref_state, psi)) ** 2),
            "bond": max_bond_dimension(mps),
            "trunc_weight": truncation_weight(ref_state, psi),
        })
    return rows


def chi_to(rows, predicate):
    for r in rows:
        if predicate(r):
            return int(r["chi"])
    return None


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"n": N_LIST, "h": H_LIST, "M": "2n+1",
                      "train_steps": TRAIN_STEPS, "threshold": THRESHOLD,
                      "chis": {n: chis_for(n) for n in N_LIST}}, "cases": {}}

    print("=" * 78)
    print("E9 chi-swept MPS surrogate: does a small-chi MPS reproduce the QNN?")
    print("=" * 78)

    for n in N_LIST:
        chis = chis_for(n)
        for h in H_LIST:
            key = f"n{n}_h{h:g}"
            info = trained_state(n, h, seed=1000 * n + int(10 * h))
            E0, E1, Et = info["E0"], info["E1"], info["E_trained"]
            eps_Q = abs(Et - E0)

            case = {
                "E0": E0, "E1": E1, "E_trained": Et,
                "trained_rel_error": info["trained_rel_error"],
                "trained_infidelity": info["trained_infidelity"],
                "chi_max_exact": 1 << (n // 2),
                "spectra": {},
                "sweeps": {},
                "thresholds": {},
            }
            for name in SAMPLERS:
                st = info["states"][name]
                spec = schmidt_spectrum(st, n)
                case["spectra"][name] = {
                    "mid_rank_significant": significant_rank(spec[(n - 1) // 2]),
                    "max_rank_significant": max(significant_rank(s) for s in spec),
                    "exact_bond": max_bond_dimension(statevector_to_mps(st, n)),
                }
                rows = sweep(st, n, h, E0, E1, chis)
                case["sweeps"][name] = rows
                if name == "trained":
                    case["thresholds"]["reproduce_chi"] = chi_to(
                        rows, lambda r: abs(r["E"] - Et) <= eps_Q)
                case["thresholds"][f"{name}_d5_chi"] = chi_to(
                    rows, lambda r: r["rel_error"] <= THRESHOLD)

            print("-" * 78)
            print(f"n={n} h={h:g}  E0={E0:+.4f}  E_trained={Et:+.4f}  "
                  f"QNN rel-err={info['trained_rel_error']:.2e}  "
                  f"infid={info['trained_infidelity']:.2e}")
            print(f"  exact bond dim = {case['chi_max_exact']}; "
                  f"significant mid-cut rank: gs="
                  f"{case['spectra']['exact_gs']['mid_rank_significant']}, "
                  f"trained={case['spectra']['trained']['mid_rank_significant']}")
            for name in SAMPLERS:
                rows = case["sweeps"][name]
                cells = "  ".join(f"chi={r['chi']}:{r['rel_error']:.1e}"
                                  for r in rows)
                print(f"    {name:>8}: {cells}")
            print(f"    thresholds: reproduce(<=QNN err) chi="
                  f"{case['thresholds']['reproduce_chi']}, "
                  f"D5<=1e-2 chi: gs="
                  f"{case['thresholds']['exact_gs_d5_chi']}, trained="
                  f"{case['thresholds']['trained_d5_chi']}")
            out["cases"][key] = case

    with open(RESULTS / "e9_mps.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e9_mps.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e9_mps.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"exact_gs": "tab:green", "trained": "tab:blue"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    hs = out["config"]["h"]
    ns = out["config"]["n"]
    thr = out["config"]["threshold"]

    for r, h in enumerate(hs):
        for c, n in enumerate(ns):
            ax = axes[r, c]
            case = out["cases"][f"n{n}_h{h:g}"]
            for name in SAMPLERS:
                rows = case["sweeps"][name]
                x = [row["chi"] for row in rows]
                y = [max(row["rel_error"], 1e-16) for row in rows]
                ax.loglog(x, y, "o-", color=colors[name], label=name)
            ax.axhline(case["trained_rel_error"], color="gray", ls="--",
                       label="QNN's own rel-err")
            ax.axhline(thr, color="black", ls=":", label="1e-2")
            ax.set_xlabel("bond dimension chi")
            ax.set_ylabel("rel. energy error")
            ax.set_title(f"n={n}, h={h:g} "
                         f"(QNN err {case['trained_rel_error']:.1e})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
        ax = axes[r, 2]
        labels = SAMPLERS
        x = np.arange(len(labels))
        r8 = [out["cases"][f"n8_h{h:g}"]["spectra"][k]["mid_rank_significant"]
              for k in labels]
        r10 = [out["cases"][f"n10_h{h:g}"]["spectra"][k]["mid_rank_significant"]
               for k in labels]
        ax.bar(x - 0.2, r8, 0.4, label="n=8", color="steelblue")
        ax.bar(x + 0.2, r10, 0.4, label="n=10", color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("significant Schmidt rank (mid cut)")
        ax.set_title(f"target entanglement rank, h={h:g}")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e9_mps.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
