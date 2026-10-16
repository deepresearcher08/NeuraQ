"""Rebuild the hero figure (figures/main_result.png) from the E15 hardware data.

Combines the original n=2..8 run with the n=10 / n=12 extensions, drawing the
QPU gradient-variance points with 95% bootstrap CIs (taken from each run's
``bootstrap`` block) plus the ideal adjoint reference lines.
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIGS = ROOT / "figures"


def load():
    srcs = ["e15_qpu.json", "e15_hw_n10.json", "e15_hw_n12.json"]
    rows, gci, lci = [], [], []
    for s in srcs:
        d = json.loads((RES / s).read_text())
        b = d.get("bootstrap", {})
        for i, r in enumerate(d["rows"]):
            rows.append(r)
            gci.append(b.get("global", {}).get("var_ci", [None] * len(d["rows"]))[i])
            lci.append(b.get("local", {}).get("var_ci", [None] * len(d["rows"]))[i])
    order = np.argsort([r["n"] for r in rows])
    rows = [rows[i] for i in order]
    gci = [gci[i] for i in order]
    lci = [lci[i] for i in order]
    return rows, gci, lci


def yerr(pt, ci):
    lo = [max(p - c[0], 0.0) if c else 0.0 for p, c in zip(pt, ci)]
    hi = [c[1] - p if c else 0.0 for p, c in zip(pt, ci)]
    return np.array([lo, hi])


def main() -> int:
    rows, gci, lci = load()
    ns = [r["n"] for r in rows]
    gG = [r["global_var_hw"] for r in rows]
    gL = [r["local_var_hw"] for r in rows]
    iG = [r["global_var_ideal"] for r in rows]
    iL = [r["local_var_ideal"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.errorbar(ns, gG, yerr=yerr(gG, gci), fmt="o-", color="crimson", capsize=3,
                label="global  $C_G$  (QPU, 95% CI)")
    ax.errorbar(ns, gL, yerr=yerr(gL, lci), fmt="s-", color="navy", capsize=3,
                label="local  $C_L$  (QPU, 95% CI)")
    ax.semilogy(ns, iG, "o--", color="crimson", alpha=0.45,
                label="global (ideal adjoint)")
    ax.semilogy(ns, iL, "s--", color="navy", alpha=0.45,
                label="local (ideal adjoint)")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel(r"Var$(\partial C/\partial\theta)$  (middle layer)")
    ax.set_title("Barren plateau on real hardware (ibm_fez):\n"
                 "global gradient collapses, local survives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGS.mkdir(exist_ok=True)
    dst = FIGS / "main_result.png"
    fig.savefig(dst, dpi=150)
    plt.close(fig)
    print(f"saved {dst}  (n = {ns})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
