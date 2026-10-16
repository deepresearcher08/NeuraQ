"""E20: perturbed identity init re-injects first/last-layer signal.

The exact identity block (Grant et al. 2019) telescopes the whole palindromic
circuit to the identity, which leaves the first and last layers with almost no
gradient signal at initialisation (E2b recorded variance ~1e-30 on those
layers). That starves the very layers that must move first. Perturbing every
angle by ``U(-eps, eps)`` keeps the circuit near identity but restores signal.

This sweeps ``eps`` on the Ising task (where an identity init is not
degenerate) and measures the per-layer gradient variance at init, plus the
init energy (which should stay near the ground energy for small ``eps``).

Run:  python experiments/e20_perturbed_init.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hamiltonian import ising_energy, ising_ground_state
from bpqnn.init import identity_block_init, perturbed_identity_init
from bpqnn.simulator import StatevectorSimulator

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N = 8
H = 1.0
M = 2 * N + 1
N_INIT = 128
EPSILONS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]


def axes_folded(M: int) -> list:
    base = ["Y", "Z", "Z", "Y"]
    axes = [base[j % 4] for j in range(M)]
    for j in range(M // 2):
        axes[M - 1 - j] = axes[j]
    return axes


def energy(theta, edges, axes) -> float:
    sim = StatevectorSimulator(N, 1)
    sim.apply_circuit(np.asarray(theta, np.float64), edges, axes)
    return float(ising_energy(sim.state, N, h=H)[0])


def main():
    RESULTS.mkdir(exist_ok=True)
    E0, _gs, _w = ising_ground_state(N, h=H)
    edges = [(q, q + 1) for q in range(N - 1)]
    axes = axes_folded(M)
    out = {"config": {"n": N, "h": H, "M": M, "n_init": N_INIT,
                      "epsilons": EPSILONS, "E0": E0, "axes": axes}}

    print("=" * 82)
    print(f"E20 perturbed identity init (Ising, n={N}, M={M}, E0={E0:.5f})")
    print("=" * 82)

    rows = []
    print(f"{'eps':>6} {'var_first':>12} {'var_mid':>12} {'var_last':>12} "
          f"{'agg var':>12} {'mean E':>10}")
    for eps in EPSILONS:
        rng = np.random.default_rng(11)
        if eps == 0.0:
            theta = identity_block_init(M, N, rng, batch=N_INIT)
        else:
            theta = perturbed_identity_init(M, N, rng, eps=eps, batch=N_INIT)
        g = adjoint_gradient(theta, N, edges, axes, "ising", h=H)  # (N_INIT,M,n)
        per_layer = g.var(axis=0).mean(axis=1)
        e_vals = [energy(theta[i], edges, axes) for i in range(min(8, N_INIT))]
        row = {
            "eps": eps,
            "var_first": float(per_layer[0]),
            "var_mid": float(per_layer[M // 2]),
            "var_last": float(per_layer[-1]),
            "agg_var": float(per_layer.mean()),
            "mean_energy": float(np.mean(e_vals)),
        }
        rows.append(row)
        print(f"{eps:>6.2f} {row['var_first']:>12.3e} {row['var_mid']:>12.3e} "
              f"{row['var_last']:>12.3e} {row['agg_var']:>12.3e} "
              f"{row['mean_energy']:>10.4f}")
    out["rows"] = rows

    with open(RESULTS / "e20_perturbed_init.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 82)
    print(f"saved {RESULTS / 'e20_perturbed_init.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e20_perturbed_init.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [r["eps"] for r in out["rows"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for key, label in (("var_first", "first layer"),
                       ("var_last", "last layer"),
                       ("var_mid", "middle layer")):
        ax1.semilogy(eps, [max(r[key], 1e-40) for r in out["rows"]], "o-",
                     label=label)
    ax1.set_xlabel("perturbation eps")
    ax1.set_ylabel("gradient variance at init")
    ax1.set_title("E20: perturbing identity restores edge-layer signal")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.plot(eps, [r["mean_energy"] for r in out["rows"]], "s-", color="darkgreen")
    ax2.axhline(out["config"]["E0"], color="k", ls=":", label="ground energy")
    ax2.set_xlabel("perturbation eps")
    ax2.set_ylabel("mean init energy <H>")
    ax2.set_title("E20: small eps keeps the init near identity")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e20_perturbed_init.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
