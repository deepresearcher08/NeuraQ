"""E6: classification QNN (SPEC E6, Core Task 3).

Small binary classification task: 2-feature *additive* grid
(16 points, label = sign(x0 + x1 - 1)), angle-encoded into qubits 0-1 of an
n=6 single-axis HEA and trained with exact parameter-shift gradients. Two
mitigation axes are compared on the same task:

  * cost locality -- global = full-qubit parity <Z_0 ... Z_5> (acts on every
    qubit) vs local = mean of <Z_0>, <Z_1> (few-qubit observable);
  * init -- random vs identity-block (M=7 odd).

The 2x2 arm grid lets the final loss / accuracy / convergence speed (Core 3)
be attributed to cost vs init, and the loss-gradient variance at init
(global vs local) is the Core-Task-4 "why" plot. Honest caveat recorded in the
write-up: the additive task is only partially expressible by a parity readout,
so the global arm's gap mixes a smaller initial signal with reduced
expressivity; the *init* gradient-variance ratio is the clean mechanism number.

Run:  python experiments/e6_classify.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.classify import (
    QNNClassifier,
    adam_classify,
    additive_grid,
    held_out_grid,
    make_dataset,
)
from bpqnn.init import identity_block_init, random_init
from bpqnn.simulator import default_axes

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

N = 6
M = 7
LOCAL_QUBITS = [0, 1]
STEPS = 150
LR = 0.05
SEEDS = 8
N_INIT = 32

ARMS = [
    ("global_random", "global", "random"),
    ("global_identity", "global", "identity"),
    ("local_random", "local", "random"),
    ("local_identity", "local", "identity"),
]


def _init(kind, rng):
    if kind == "identity":
        return identity_block_init(M, N, rng)
    return random_init(M, N, rng)


def _classifier(readout, X, y):
    qubits = None if readout == "global" else LOCAL_QUBITS
    return QNNClassifier(N, M, X, y, readout=readout, qubits=qubits)


def init_signals(X, y):
    """Loss-gradient variance at init for global vs local readout (random init)."""
    rng = np.random.default_rng(2024)
    out = {}
    for readout in ("global", "local"):
        clf = _classifier(readout, X, y)
        grads = np.empty((N_INIT, M, N), dtype=np.float64)
        losses = np.empty(N_INIT)
        for i in range(N_INIT):
            th = random_init(M, N, rng)
            grads[i] = clf.grad(th)
            losses[i] = clf.loss(th)
        per_param = grads.var(axis=0)
        out[readout] = {
            "grad_var": float(per_param.mean()),
            "grad_var_first_layer": float(per_param[0].mean()),
            "loss_mean": float(losses.mean()),
        }
    out["ratio_global_over_local"] = (
        out["global"]["grad_var"] / out["local"]["grad_var"]
    )
    return out


def run_arm(name, readout, init_kind, X, y, Xte, yte):
    clf = _classifier(readout, X, y)
    finals, accs, test_accs, histories = [], [], [], []
    steps_to = {"1.0": [], "0.5": []}
    for s in range(SEEDS):
        rng = np.random.default_rng(1000 + s)
        th0 = _init(init_kind, rng)
        theta, hist = adam_classify(clf, th0, lr=LR, steps=STEPS)
        finals.append(clf.loss(theta))
        accs.append(clf.accuracy(theta))
        test_accs.append(clf.accuracy_on(theta, Xte, yte))
        histories.append(hist)
        for key, hits in steps_to.items():
            hit = next((i for i, c in enumerate(hist) if c <= float(key)), None)
            if hit is not None:
                hits.append(hit)
    hist = np.array(histories)  # (S, STEPS+1)
    return {
        "readout": readout,
        "init": init_kind,
        "final_loss": [float(x) for x in finals],
        "final_acc": [float(x) for x in accs],
        "median_loss": float(np.median(finals)),
        "median_acc": float(np.median(accs)),
        "median_test_acc": float(np.median(test_accs)),
        "test_acc": [float(x) for x in test_accs],
        "history_median": [float(x) for x in np.median(hist, axis=0)],
        "history_q25": [float(x) for x in np.percentile(hist, 25, axis=0)],
        "history_q75": [float(x) for x in np.percentile(hist, 75, axis=0)],
        "steps_to_1.0": float(np.median(steps_to["1.0"])) if steps_to["1.0"] else None,
        "steps_to_0.5": float(np.median(steps_to["0.5"])) if steps_to["0.5"] else None,
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    X, y = additive_grid()
    Xte, yte = make_dataset("additive", held_out_grid())
    out = {
        "config": {"n": N, "M": M, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                   "n_points": int(X.shape[0]), "n_test_points": int(Xte.shape[0]),
                   "local_qubits": LOCAL_QUBITS,
                   "task": "additive: y=sign(x0+x1-1)", "axes": default_axes(M)},
        "chance_loss": 1.0,
    }

    print("=" * 78)
    print(f"E6 classification: additive grid ({X.shape[0]} pts), n={N}, M={M}, "
          f"steps={STEPS}, seeds={SEEDS}")
    print("=" * 78)

    out["init_signals"] = init_signals(X, y)
    gs = out["init_signals"]
    print(f"init loss-gradient variance: global {gs['global']['grad_var']:.3e}  "
          f"local {gs['local']['grad_var']:.3e}  "
          f"(ratio g/l {gs['ratio_global_over_local']:.3e})")

    print("-" * 78)
    print(f"{'arm':>18} {'med loss':>10} {'train acc':>9} {'test acc':>9} "
          f"{'<=1.0':>7} {'<=0.5':>7}")
    out["arms"] = {}
    for name, readout, init_kind in ARMS:
        res = run_arm(name, readout, init_kind, X, y, Xte, yte)
        out["arms"][name] = res
        s1 = "-" if res["steps_to_1.0"] is None else f"{res['steps_to_1.0']:.0f}"
        s5 = "-" if res["steps_to_0.5"] is None else f"{res['steps_to_0.5']:.0f}"
        print(f"{name:>18} {res['median_loss']:>10.4f} "
              f"{res['median_acc']:>9.3f} {res['median_test_acc']:>9.3f} "
              f"{s1:>7} {s5:>7}")

    with open(RESULTS / "e6_classify.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e6_classify.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e6_classify.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    styles = {
        "global_random": ("C3", "-", "global cost, random init (unmitigated)"),
        "global_identity": ("C3", "--", "global cost, identity init"),
        "local_random": ("C0", "-", "local cost, random init"),
        "local_identity": ("C0", "--", "local cost, identity init (mitigated)"),
    }
    ax1.axhline(out["chance_loss"], color="k", ls=":", lw=1, label="chance")
    for name, (color, ls, label) in styles.items():
        r = out["arms"][name]
        x = np.arange(len(r["history_median"]))
        ax1.plot(x, r["history_median"], color=color, ls=ls, label=label)
        ax1.fill_between(x, r["history_q25"], r["history_q75"], color=color, alpha=0.12)
    ax1.set_xlabel("Adam step")
    ax1.set_ylabel("train loss (MSE)")
    ax1.set_title("E6: convergence, additive classification")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    gs = out["init_signals"]
    vals = [gs["global"]["grad_var"], gs["local"]["grad_var"]]
    ax2.bar(["global\n(parity)", "local\n(mean Z0,Z1)"], vals,
            color=["C3", "C0"], alpha=0.8)
    ax2.set_yscale("log")
    ax2.set_ylabel("loss-gradient variance at init")
    ax2.set_title(f"E6: why — cost locality vs init signal "
                  f"(x{gs['ratio_global_over_local']:.0f})")
    ax2.grid(True, which="both", axis="y", alpha=0.3)
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e6_classify.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
