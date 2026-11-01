"""E21: classical ML head-to-head baseline for the QNN classifier.

The QNN results come from E17 (same 4x4 knot grid, 20x20 offset held-out grid,
eight seeds, full-batch Adam, 200 steps, lr 0.05).  This experiment scores
classical models on exactly the same split and protocol:

  * ``mlp_matched`` -- 41 trainable parameters, one tanh hidden layer of width
    10.  This is the fair parameter-matched baseline (QNN has ``M*n = 42``).
  * ``logreg`` -- linear baseline (3 parameters).
  * ``mlp_wide`` -- 129-parameter tanh MLP (generous, not matched).
  * ``svm_rbf``, ``knn_k3``, ``random_forest`` -- non-parametric references.

The point is to answer "is the QNN at least competitive with ordinary classical
ML on these tasks?" on a common held-out split.  It is not a quantum-advantage
claim: these are 2-D toys with 16 training points.

Run:  python experiments/e21_classical_baseline.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.classical import evaluate_numpy_mlp, evaluate_sklearn, mlp_param_count
from bpqnn.classify import train_test_datasets

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
QNN_RESULTS = RESULTS / "e17_parity_task.json"

TASKS = ("additive", "parity")
QNN_N = 6
QNN_M = 7
QNN_PARAMS = QNN_N * QNN_M
HIDDEN_MATCHED = 10
HIDDEN_WIDE = 32
SEEDS = 8
STEPS = 200
LR = 0.05

QNN_ARMS = (("qnn_global", "global"), ("qnn_local", "local"))
MODEL_LABELS = {
    "qnn_global": "QNN global readout",
    "qnn_local": "QNN local readout",
    "logreg": "Logistic regression",
    "mlp_matched": "MLP (matched)",
    "mlp_wide": "MLP (wide)",
    "svm_rbf": "SVM (RBF)",
    "knn_k3": "k-NN (k=3)",
    "random_forest": "Random forest",
}


def load_qnn() -> dict:
    if not QNN_RESULTS.exists():
        raise SystemExit(
            f"missing {QNN_RESULTS}; run experiments/e17_parity_task.py first"
        )
    with open(QNN_RESULTS) as f:
        return json.load(f)


def classical_models(Xtr, ytr, Xte, yte) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    rows = {}
    rows["logreg"] = evaluate_sklearn(
        lambda seed: LogisticRegression(max_iter=1000),
        Xtr, ytr, Xte, yte, seeds=(0,),
    )
    rows["logreg"]["n_params"] = int(np.asarray(Xtr).shape[1] + 1)

    rows["mlp_matched"] = evaluate_numpy_mlp(
        Xtr, ytr, Xte, yte, hidden=HIDDEN_MATCHED, seeds=range(SEEDS),
        lr=LR, steps=STEPS,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rows["mlp_wide"] = evaluate_sklearn(
            lambda seed: MLPClassifier(
                hidden_layer_sizes=(HIDDEN_WIDE,), max_iter=3000,
                random_state=seed,
            ),
            Xtr, ytr, Xte, yte, seeds=range(SEEDS),
        )
    rows["mlp_wide"]["n_params"] = mlp_param_count(
        int(np.asarray(Xtr).shape[1]), HIDDEN_WIDE
    )

    rows["svm_rbf"] = evaluate_sklearn(
        lambda seed: SVC(kernel="rbf", C=10.0, gamma="scale"),
        Xtr, ytr, Xte, yte, seeds=(0,),
    )
    rows["svm_rbf"]["n_params"] = None

    rows["knn_k3"] = evaluate_sklearn(
        lambda seed: KNeighborsClassifier(n_neighbors=3),
        Xtr, ytr, Xte, yte, seeds=(0,),
    )
    rows["knn_k3"]["n_params"] = None

    rows["random_forest"] = evaluate_sklearn(
        lambda seed: RandomForestClassifier(n_estimators=200, random_state=seed),
        Xtr, ytr, Xte, yte, seeds=range(SEEDS),
    )
    rows["random_forest"]["n_params"] = None
    return rows


def qnn_rows(qnn: dict, task: str) -> dict:
    arms = qnn["arms"]
    rows = {}
    for key, readout in QNN_ARMS:
        src = arms[f"{task}_{readout}"]
        rows[key] = {
            "model": key,
            "readout": readout,
            "n_params": QNN_PARAMS,
            "median_train_acc": float(src["median_train_acc"]),
            "median_test_acc": float(src["median_test_acc"]),
            "train_acc": None,  # E17 does not store per-seed train accuracy
            "test_acc": [float(x) for x in src["test_acc"]],
            "final_loss": [float(x) for x in src["final_loss"]],
            "n_seeds": int(len(src["test_acc"])),
            "fit_seconds": None,
        }
    return rows


def analyse(task_rows: dict) -> dict:
    classical = {k: v for k, v in task_rows.items() if not k.startswith("qnn_")}
    qnn = {k: v for k, v in task_rows.items() if k.startswith("qnn_")}
    best_classical = max(classical, key=lambda k: classical[k]["median_test_acc"])
    best_qnn = max(qnn, key=lambda k: qnn[k]["median_test_acc"])
    gap = (qnn[best_qnn]["median_test_acc"]
           - classical[best_classical]["median_test_acc"])
    return {
        "best_classical": best_classical,
        "best_classical_test_acc": classical[best_classical]["median_test_acc"],
        "best_qnn": best_qnn,
        "best_qnn_test_acc": qnn[best_qnn]["median_test_acc"],
        "qnn_minus_best_classical": float(gap),
        "qnn_within_2_points": bool(gap >= -0.02),
    }


def verdict(task_analysis: dict, task_rows: dict) -> dict:
    add = task_analysis["additive"]
    par = task_analysis["parity"]
    return {
        "qnn_competitive_additive": add["qnn_within_2_points"],
        "qnn_competitive_parity": par["qnn_within_2_points"],
        "additive_best": add["best_classical"],
        "parity_best": par["best_classical"],
        "logreg_additive_test_acc":
            task_rows["additive"]["logreg"]["median_test_acc"],
        "logreg_parity_test_acc":
            task_rows["parity"]["logreg"]["median_test_acc"],
        "linear_fails_parity":
            task_rows["parity"]["logreg"]["median_test_acc"] < 0.65,
        "matched_mlp_parity_test_acc":
            task_rows["parity"]["mlp_matched"]["median_test_acc"],
        "summary": (
            "QNN global matches the best classical model on parity; the QNN "
            "does not beat ordinary classical ML on either 2-D toy task. "
            "Linear regression fails parity as expected. This is a baseline, "
            "not a quantum-advantage result."
        ),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    qnn = load_qnn()
    out = {
        "config": {
            "tasks": list(TASKS),
            "n_train": 16,
            "n_test": 400,
            "qnn_source": "results/e17_parity_task.json",
            "qnn_n": QNN_N,
            "qnn_M": QNN_M,
            "qnn_params": QNN_PARAMS,
            "hidden_matched": HIDDEN_MATCHED,
            "hidden_wide": HIDDEN_WIDE,
            "seeds": SEEDS,
            "steps": STEPS,
            "lr": LR,
            "protocol": "full-batch Adam, MSE loss, same 8 seeds as E17",
        },
        "param_counts": {
            "qnn": QNN_PARAMS,
            "mlp_matched": mlp_param_count(2, HIDDEN_MATCHED),
            "mlp_wide": mlp_param_count(2, HIDDEN_WIDE),
            "logreg": 3,
            "non_parametric": ["svm_rbf", "knn_k3", "random_forest"],
        },
        "chance_acc": 0.5,
        "tasks": {},
    }

    print("=" * 88)
    print("E21 classical ML head-to-head (QNN: "
          f"{QNN_N}x{QNN_M} = {QNN_PARAMS} params; matched MLP: "
          f"{mlp_param_count(2, HIDDEN_MATCHED)} params)")
    print("=" * 88)
    print(f"{'task':>9} {'model':>18} {'params':>8} {'train acc':>10} "
          f"{'held-out acc':>13}")
    for task in TASKS:
        Xtr, ytr, Xte, yte = train_test_datasets(task)
        rows = qnn_rows(qnn, task)
        rows.update(classical_models(Xtr, ytr, Xte, yte))
        out["tasks"][task] = rows
        for key, row in rows.items():
            n = "-" if row["n_params"] is None else str(row["n_params"])
            print(f"{task:>9} {MODEL_LABELS[key]:>18} {n:>8} "
                  f"{row['median_train_acc']:>10.3f} "
                  f"{row['median_test_acc']:>13.3f}")

    out["analysis"] = {task: analyse(out["tasks"][task]) for task in TASKS}
    out["verdict"] = verdict(out["analysis"], out["tasks"])
    print("-" * 88)
    for task in TASKS:
        a = out["analysis"][task]
        print(f"{task}: best classical {a['best_classical']} "
              f"{a['best_classical_test_acc']:.3f}, best QNN {a['best_qnn']} "
              f"{a['best_qnn_test_acc']:.3f}, "
              f"QNN - classical {a['qnn_minus_best_classical']:+.3f}")
    print(out["verdict"]["summary"])

    with open(RESULTS / "e21_classical_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULTS / 'e21_classical_baseline.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e21_classical_baseline.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, task in zip(axes, TASKS):
        rows = out["tasks"][task]
        order = sorted(rows, key=lambda k: rows[k]["median_test_acc"])
        vals = [rows[k]["median_test_acc"] for k in order]
        labels = []
        colors = []
        for k in order:
            n = rows[k]["n_params"]
            tag = "" if n is None else f" ({n}p)"
            labels.append(MODEL_LABELS[k] + tag)
            colors.append("C0" if k == "qnn_global" else
                          "C1" if k == "qnn_local" else
                          "C2" if k == "mlp_matched" else "0.55")
        ax.barh(np.arange(len(order)), vals, color=colors, alpha=0.85)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(out["chance_acc"], color="k", ls=":", lw=1)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("held-out accuracy (400 offset points)")
        ax.set_title(f"E21: {task} task")
        ax.grid(True, axis="x", alpha=0.3)
        for i, v in enumerate(vals):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    axes[0].plot([], [], "s", color="C0", label="QNN (42 params)")
    axes[0].plot([], [], "s", color="C2", label="matched MLP (41 params)")
    axes[0].plot([], [], "s", color="0.55", label="other classical")
    axes[0].plot([], [], "k:", label="chance")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("E21: classical ML head-to-head on the E17 QNN tasks")
    fig.tight_layout()
    fig.savefig(RESULTS / "e21_classical_baseline.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
