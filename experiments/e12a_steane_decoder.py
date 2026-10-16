"""E12-A: Steane [[7,1,3]] syndrome-decoder QNN vs the standard decoder.

Task: decode the logical X class of the Steane code from the *measured* 3-bit
syndrome, under data X-errors (prob p) **and** syndrome-measurement errors
(prob q per check). Decoders compared:

  * **standard** -- minimum-weight coset leader assuming an exact syndrome
    (the textbook decoder, i.e. q = 0);
  * **Bayes**    -- exact posterior over the 2**7 errors given the noisy
    syndrome (accounts for q);
  * **QNN**      -- ``bpqnn.classify.QNNClassifier`` on the measured syndrome;
  * **logreg / MLP** -- classical references (sklearn) on the same features.

Question: does the QNN beat the *standard* decoder (yes, when it exploits q) and
does it match the *Bayes-optimal* decoder (i.e. is any classical/quantum gap
absent)? Run:  python experiments/e12a_steane_decoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.classify import QNNClassifier, adam_classify
from bpqnn.init import random_init
from bpqnn.steane import bayes_predict, random_syndrome_features, standard_predict

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

P_LIST = [0.03, 0.05]
Q_LIST = [0.0, 0.05, 0.10, 0.15]
N_TRAIN = 4096
N_TEST = 40000
QNN_M = 5
QNN_SEEDS = 3
QNN_STEPS = 150


def qnn_ler(Xtr, ytr, Xte, yte, seed):
    clf = QNNClassifier(3, QNN_M, Xtr, ytr, readout="local", qubits=[0, 1, 2])
    rng = np.random.default_rng(seed)
    theta, _ = adam_classify(clf, random_init(QNN_M, 3, rng),
                             lr=0.05, steps=QNN_STEPS,
                             grad_fn=clf.grad_adjoint)
    clf_te = QNNClassifier(3, QNN_M, Xte, yte, readout="local", qubits=[0, 1, 2])
    acc = clf_te.accuracy(theta)
    return 1.0 - acc


def main():
    RESULTS.mkdir(exist_ok=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    out = {"config": {"p": P_LIST, "q": Q_LIST, "n_train": N_TRAIN,
                      "n_test": N_TEST, "qnn_M": QNN_M, "qnn_seeds": QNN_SEEDS}}
    rng = np.random.default_rng(0)

    print("=" * 84)
    print("E12-A: Steane [[7,1,3]] X-channel decoder (measurement-noise aware vs standard)")
    print(f"{'p':>5} {'q':>5} {'LER_std':>9} {'LER_bayes':>10} {'LER_QNN':>9} "
          f"{'LER_logreg':>11} {'LER_MLP':>9} {'QNN/std':>8}")
    rows = []
    for p in P_LIST:
        for q in Q_LIST:
            Xtr, Ltr = random_syndrome_features(p, q, N_TRAIN, rng)
            Xte, Lte = random_syndrome_features(p, q, N_TEST, rng)
            ytr = 1.0 - 2.0 * Ltr
            yte = 1.0 - 2.0 * Lte

            ler_std = float(np.mean(
                [standard_predict(p, s) for s in Xte] != Lte))
            ler_bay = float(np.mean(
                [bayes_predict(p, q, s) for s in Xte] != Lte))

            t0 = time.time()
            ler_qnn = min(qnn_ler(Xtr, ytr, Xte, yte, seed)
                          for seed in range(QNN_SEEDS))
            dt = time.time() - t0

            lr = LogisticRegression(max_iter=200).fit(Xtr, Ltr)
            ler_lr = float((lr.predict(Xte) != Lte).mean())
            mlp = MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300,
                                random_state=0).fit(Xtr, Ltr)
            ler_mlp = float((mlp.predict(Xte) != Lte).mean())

            rows.append({"p": p, "q": q, "ler_std": ler_std, "ler_bayes": ler_bay,
                         "ler_qnn": ler_qnn, "ler_logreg": ler_lr, "ler_mlp": ler_mlp,
                         "qnn_over_std": ler_qnn / ler_std if ler_std else float("nan")})
            print(f"{p:>5.3f} {q:>5.3f} {ler_std:>9.4f} {ler_bay:>10.4f} "
                  f"{ler_qnn:>9.4f} {ler_lr:>11.4f} {ler_mlp:>9.4f} "
                  f"{ler_qnn / ler_std:>8.3f}   ({dt:.1f}s)")
    out["rows"] = rows

    qpos = [r["qnn_over_std"] for r in rows if r["q"] > 0]
    out["verdict"] = {
        "qnn_beats_standard_when_q_pos": bool(np.mean(qpos) < 0.9),
        "mean_qnn_over_std_qpos": float(np.mean(qpos)),
        "qnn_matches_bayes": bool(max(abs(r["ler_qnn"] - r["ler_bayes"])
                                      for r in rows if r["q"] > 0) < 0.02),
    }
    print("-" * 84)
    print("verdict:", out["verdict"])
    with open(RESULTS / "e12a_steane_decoder.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULTS / 'e12a_steane_decoder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
