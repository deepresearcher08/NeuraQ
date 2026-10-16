"""Figure for E13: final global cost per support policy + initial gradient variance."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

ORDER = ["local_r1", "regional_rstar", "global_rn",
         "window_r2", "window_r4", "csa_linear", "csa_staged", "csa_var_gated"]
LABELS = ["single-block r=1\n(local, blind)", "single-block r=n/2\n(regional, blind)",
          "single-block r=n\n(global, plateau pt)", "window r=2\n(spread, faithful)",
          "window r=4\n(spread, faithful)", "CSA linear\n(grow 1->n)",
          "CSA staged\n(grow 1->n)", "CSA var-gated\n(grow 1->n)"]

data = json.loads((RESULTS / "e13_csa.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ["#c0392b", "#e67e22", "#16a085", "#2980b9", "#3498db",
          "#8e44ad", "#7f8c8d", "#9b59b6"]

for ax, n in zip(axes, data["config"]["n"]):
    a = data["results"][str(n)]["arms"]
    med = [a[k][0] for k in ORDER]
    lo = [a[k][1] for k in ORDER]
    hi = [a[k][2] for k in ORDER]
    x = np.arange(len(ORDER))
    ax.bar(x, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)],
           color=colors, capsize=3, log=True)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 2)
    ax.axhline(1.0, color="grey", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("final global cost $C_n = 1-P(0\\dots 0)$")
    ax.set_title(f"n = {n}  (M={n}, 250 Adam steps, median of {data['config']['seeds']} seeds)")
    v = data["results"][str(n)]
    ratio = v["var_init_block_rn"] / v["var_init_window_r2"]
    txt = (f"init Var: block r=n {v['var_init_block_rn']:.1e}\n"
           f"          window r=2 {v['var_init_window_r2']:.1e}\n"
           f"ratio {ratio:.1e}")
    ax.text(0.01, 0.03, txt, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(fc="white", ec="grey", alpha=0.8))

fig.suptitle("E13: scheduling vs spreading the observable support (E11 axis)")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(RESULTS / "e13_csa.png", dpi=140)
print("saved", RESULTS / "e13_csa.png")