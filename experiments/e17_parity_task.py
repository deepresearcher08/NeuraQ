"""E17: task expressivity -- a parity task the local readout cannot represent.

E6 used ``additive_grid`` (label = sign(x0 + x1 - 1)), a linearly separable
boundary that a 2-qubit local readout represents *by construction* -- so the
local cost wins trivially there and the comparison says little. Here the same
4x4 knot grid is labelled by *parity* (checkerboard),
``sign(x0 - 0.5) XOR sign(x1 - 0.5)``.

The feature map is ``RY(pi x)``, so ``<Z_q> = cos(pi x_q) =: z_q`` and parity is
``sign(z0 z1)``. A local readout (mean of ``<Z_0>, <Z_1>``) is an affine
function of ``z0, z1`` and *cannot* match ``sign(z0 z1)``; the global readout
(full-register parity) *can*. That isolates the expressivity-vs-trainability
trade-off: the global cost is the one that can fit parity, but it carries the
plateau risk (E1/E15).

Every arm is scored on a *held-out* denser grid (400 points offset from the
training knots), not on the 16 training points.

Run:  python experiments/e17_parity_task.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.classify import QNNClassifier, adam_classify, train_test_datasets
from bpqnn.init import random_init
from bpqnn.simulator import default_axes

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N = 6
M = 7
LOCAL_QUBITS = [0, 1]
STEPS = 200
LR = 0.05
SEEDS = 8
TASKS = ("additive", "parity")
READOUTS = ("global", "local")


def run_arm(readout, Xtr, ytr, Xte, yte):
    clf = QNNClassifier(N, M, Xtr, ytr, readout=readout,
                        qubits=None if readout == "global" else LOCAL_QUBITS)
    train_acc, test_acc, finals = [], [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        theta, _ = adam_classify(
            clf, random_init(M, N, rng), lr=LR, steps=STEPS,
            grad_fn=clf.grad_adjoint,
        )
        train_acc.append(clf.accuracy(theta))
        test_acc.append(clf.accuracy_on(theta, Xte, yte))
        finals.append(clf.loss(theta))
    return {
        "readout": readout,
        "median_train_acc": float(np.median(train_acc)),
        "median_test_acc": float(np.median(test_acc)),
        "test_acc": [float(x) for x in test_acc],
        "final_loss": [float(x) for x in finals],
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"n": N, "M": M, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                      "local_qubits": LOCAL_QUBITS, "axes": default_axes(M)},
           "chance_acc": 0.5}

    print("=" * 78)
    print(f"E17 task expressivity: additive vs parity, n={N}, M={M}, seeds={SEEDS}")
    print("=" * 78)

    out["arms"] = {}
    print(f"{'task':>9} {'readout':>8} {'train acc':>10} {'held-out acc':>13}")
    for task in TASKS:
        Xtr, ytr, Xte, yte = train_test_datasets(task)
        out["dataset_sizes"] = out.get("dataset_sizes", {})
        out["dataset_sizes"][task] = {"train": int(Xtr.shape[0]),
                                      "test": int(Xte.shape[0])}
        for readout in READOUTS:
            res = run_arm(readout, Xtr, ytr, Xte, yte)
            out["arms"][f"{task}_{readout}"] = res
            print(f"{task:>9} {readout:>8} {res['median_train_acc']:>10.3f} "
                  f"{res['median_test_acc']:>13.3f}")

    with open(RESULTS / "e17_parity_task.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e17_parity_task.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e17_parity_task.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(TASKS)
    x = np.arange(len(tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for off, readout, color in ((-w / 2, "global", "crimson"),
                                (w / 2, "local", "navy")):
        vals = [out["arms"][f"{t}_{readout}"]["median_test_acc"] for t in tasks]
        ax.bar(x + off, vals, w, color=color, alpha=0.85,
               label=f"{readout} readout")
    ax.axhline(out["chance_acc"], color="k", ls=":", lw=1, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(held-out)" for t in tasks])
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("E17: task expressivity — locality is not enough on parity")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "e17_parity_task.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
