"""E10: decoder QNN on the repetition code (SPEC E10, combine-with-another-track).

A QNN that *learns to decode* a bit-flip repetition code: the received d-bit
word is angle-encoded into d qubits, the single-axis HEA acts, and a readout
predicts the logical bit. Two mitigation axes are compared exactly as in E6 --
cost locality (global parity readout vs local mean-<Z>) and init (random vs
identity-block) -- and the loss-gradient variance is measured vs code distance
``d`` (the same barren-plateau lens as E1/E2/E6, now on a decoding task).

Baselines: the analytic Bayes-optimal majority decoder and a classical logistic
regression on the received bits (which is optimal here -- the repetition decoder
is linear). Honest framing (SPEC section 12): this is a mechanism/transfer demo,
not a competitive decoder and not a quantum-advantage claim.

Run:  python experiments/e10_decoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.classify import QNNClassifier, adam_classify
from bpqnn.init import identity_block_init, random_init
from bpqnn.qec import (
    logreg_accuracy,
    majority_accuracy,
    repetition_samples,
    train_logreg,
)

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
D_LIST = [3, 5, 7, 9]
P_NOISE = 0.1
M_OF = lambda d: d + 2  # odd -> identity-block constructible  # noqa: E731
S_TRAIN = 96
S_TEST = 512
STEPS = 90
LR = 0.05
SEEDS = 3
N_INIT_VAR = 20
THRESHOLDS = (0.5, 0.3)
ARMS = [
    ("global_random", "global", "random"),
    ("global_identity", "global", "identity"),
    ("local_random", "local", "random"),
    ("local_identity", "local", "identity"),
]


def _make_clf(d, X, y, readout):
    qubits = None if readout == "global" else list(range(d))
    return QNNClassifier(d, M_OF(d), X, y, readout=readout, qubits=qubits)


def _init(kind, d, rng):
    M = M_OF(d)
    if kind == "identity":
        return identity_block_init(M, d, rng)
    return random_init(M, d, rng)


def init_signal(d, X, y):
    """Mean loss-gradient variance at random init, global vs local readout."""
    rng = np.random.default_rng(9000 + d)
    out = {}
    for readout in ("global", "local"):
        clf = _make_clf(d, X, y, readout)
        grads = np.empty((N_INIT_VAR, M_OF(d), d), dtype=np.float64)
        for i in range(N_INIT_VAR):
            grads[i] = clf.grad(random_init(M_OF(d), d, rng))
        out[readout] = float(grads.var(axis=0).mean())
    out["ratio_global_over_local"] = out["global"] / out["local"]
    return out


def run_arm(d, readout, init_kind, Xtr, ytr, Xte, yte):
    clf_tr = _make_clf(d, Xtr, ytr, readout)
    clf_te = _make_clf(d, Xte, yte, readout)
    test_acc, final_loss, histories = [], [], []
    steps_to = {f"{t:g}": [] for t in THRESHOLDS}
    for s in range(SEEDS):
        rng = np.random.default_rng(2000 + 37 * d + s)
        th0 = _init(init_kind, d, rng)
        theta, hist = adam_classify(clf_tr, th0, lr=LR, steps=STEPS,
                                    grad_fn=clf_tr.grad_adjoint)
        test_acc.append(clf_te.accuracy(theta))
        final_loss.append(clf_tr.loss(theta))
        histories.append(hist)
        for key, hits in steps_to.items():
            hit = next((i for i, c in enumerate(hist) if c <= float(key)), None)
            if hit is not None:
                hits.append(hit)
    hist = np.array(histories)
    return {
        "readout": readout, "init": init_kind,
        "median_test_acc": float(np.median(test_acc)),
        "test_acc": [float(a) for a in test_acc],
        "median_final_loss": float(np.median(final_loss)),
        "history_median": [float(x) for x in np.median(hist, axis=0)],
        "history_q25": [float(x) for x in np.percentile(hist, 25, axis=0)],
        "history_q75": [float(x) for x in np.percentile(hist, 75, axis=0)],
        "steps_to": {k: (float(np.median(v)) if v else None)
                     for k, v in steps_to.items()},
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {
        "config": {"d": D_LIST, "p": P_NOISE, "M": "d+2", "s_train": S_TRAIN,
                   "s_test": S_TEST, "steps": STEPS, "lr": LR, "seeds": SEEDS,
                   "n_init_var": N_INIT_VAR, "task": "repetition-code decoder",
                   "label_convention": "y=1-2b (+1 = logical 0)"},
        "cases": {},
    }

    print("=" * 78)
    print(f"E10 decoder QNN: repetition code, p={P_NOISE}, d in {D_LIST}, "
          f"{SEEDS} seeds")
    print("=" * 78)
    print(f"{'d':>3} {'bayes':>7} {'logreg':>7} "
          f"{'g_rand':>8} {'g_id':>8} {'l_rand':>8} {'l_id':>8} "
          f"{'gv_g':>9} {'gv_l':>9}")

    for d in D_LIST:
        rng_tr = np.random.default_rng(1000 + d)
        rng_te = np.random.default_rng(5000 + d)
        Xtr, ytr = repetition_samples(d, P_NOISE, S_TRAIN, rng_tr)
        Xte, yte = repetition_samples(d, P_NOISE, S_TEST, rng_te)
        bayes = majority_accuracy(d, P_NOISE)
        w = train_logreg(Xtr, ytr)
        logreg = logreg_accuracy(w, Xte, yte)

        case = {"bayes_acc": bayes, "logreg_acc": float(logreg),
                "init_signal": init_signal(d, Xtr, ytr), "arms": {}}
        accs = {}
        for name, readout, init_kind in ARMS:
            res = run_arm(d, readout, init_kind, Xtr, ytr, Xte, yte)
            case["arms"][name] = res
            accs[name] = res["median_test_acc"]
        out["cases"][f"d{d}"] = case

        gv = case["init_signal"]
        print(f"{d:>3} {bayes:>7.3f} {logreg:>7.3f} "
              f"{accs['global_random']:>8.3f} {accs['global_identity']:>8.3f} "
              f"{accs['local_random']:>8.3f} {accs['local_identity']:>8.3f} "
              f"{gv['global']:>9.2e} {gv['local']:>9.2e}")

    with open(RESULTS / "e10_decoder.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULTS / 'e10_decoder.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e10_decoder.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = out["config"]["d"]
    styles = {
        "global_random": ("C3", "-", "global, random (unmitigated)"),
        "global_identity": ("C3", "--", "global, identity"),
        "local_random": ("C0", "-", "local, random"),
        "local_identity": ("C0", "--", "local, identity (mitigated)"),
    }
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    base = [out["cases"][f"d{d}"]["bayes_acc"] for d in ds]
    logr = [out["cases"][f"d{d}"]["logreg_acc"] for d in ds]
    ax1.plot(ds, base, "k:", marker="x", label="Bayes (majority)")
    ax1.plot(ds, logr, "k-.", marker="+", label="logistic regression")
    for name, (color, ls, label) in styles.items():
        acc = [out["cases"][f"d{d}"]["arms"][name]["median_test_acc"]
               for d in ds]
        ax1.plot(ds, acc, color=color, ls=ls, marker="o", label=label)
    ax1.set_xlabel("code distance d")
    ax1.set_ylabel("test accuracy")
    ax1.set_title("E10: decoded logical bit")
    ax1.set_ylim(0.4, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=7)

    gv_g = [out["cases"][f"d{d}"]["init_signal"]["global"] for d in ds]
    gv_l = [out["cases"][f"d{d}"]["init_signal"]["local"] for d in ds]
    ax2.semilogy(ds, gv_g, "C3-o", label="global (parity)")
    ax2.semilogy(ds, gv_l, "C0-o", label="local (mean Z)")
    ax2.set_xlabel("code distance d")
    ax2.set_ylabel("loss-gradient variance at init")
    ax2.set_title("E10: why — the same locality gap")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    dmax = ds[-1]
    for name, (color, ls, label) in styles.items():
        r = out["cases"][f"d{dmax}"]["arms"][name]
        x = np.arange(len(r["history_median"]))
        ax3.plot(x, r["history_median"], color=color, ls=ls, label=name)
        ax3.fill_between(x, r["history_q25"], r["history_q75"],
                         color=color, alpha=0.12)
    ax3.set_xlabel("Adam step")
    ax3.set_ylabel("train loss (MSE)")
    ax3.set_title(f"E10: convergence, d={dmax}")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(RESULTS / "e10_decoder.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
