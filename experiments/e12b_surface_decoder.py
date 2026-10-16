"""E12-B: surface-code neural decoder vs MWPM (does anything beat the standard?).

Stim generates a rotated surface-code memory experiment with circuit-level
depolarising noise. The *standard* decoder is pymatching MWPM built from the
circuit's own detector error model (so it is well-matched, the hard case). We
train a small neural decoder (sklearn MLP) on the raw detector events and
compare the logical error rate.

Honest caveat: this is a **classical** neural decoder, not the project's QNN
(the syndrome graph has ~24-48 detectors; a statevector QNN cannot run at that
width). The question E12-B answers is narrow: *under matched circuit-level
noise, can a learned decoder beat MWPM?*

Run:  python experiments/e12b_surface_decoder.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import stim
import pymatching

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def make_circuit(d, rounds, p):
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d, rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def run_point(d, rounds, p, n_train, n_test, seed=1):
    from sklearn.neural_network import MLPClassifier
    circ = make_circuit(d, rounds, p)
    mwpm = pymatching.Matching.from_detector_error_model(
        circ.detector_error_model(decompose_errors=True))
    sampler = circ.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(n_train + n_test, separate_observables=True)
    y = obs.any(axis=1).astype(np.int64)
    dets = dets.astype(np.float32)
    tr, te = slice(0, n_train), slice(n_train, None)

    ler_mwpm = float((mwpm.decode_batch(dets[te]).any(axis=1) != y[te]).mean())
    clf = MLPClassifier(hidden_layer_sizes=(128, 128), max_iter=120,
                        early_stopping=True, n_iter_no_change=8, random_state=seed)
    clf.fit(dets[tr], y[tr])
    ler_mlp = float((clf.predict(dets[te]) != y[te]).mean())
    return {
        "d": d, "rounds": rounds, "p": p, "n_det": int(dets.shape[1]),
        "n_train": n_train, "n_test": n_test,
        "ler_mwpm": ler_mwpm, "ler_mlp": ler_mlp,
        "ratio_mlp_over_mwpm": ler_mlp / ler_mwpm if ler_mwpm else float("nan"),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"exp": "E12-B surface code", "routine": "rotated_memory_z"}}
    points = [
        (3, 3, 0.008), (3, 3, 0.012),
        (5, 5, 0.006), (5, 5, 0.010),
    ]
    N_TRAIN, N_TEST = 200_000, 200_000
    print("=" * 78)
    print("E12-B: neural decoder vs MWPM (matched circuit-level depolarising noise)")
    print(f"{'d':>2} {'rounds':>6} {'p':>6} {'#det':>5} {'LER_MWPM':>10} "
          f"{'LER_MLP':>10} {'MLP/MWPM':>9}")
    rows = []
    for d, rounds, p in points:
        r = run_point(d, rounds, p, N_TRAIN, N_TEST)
        rows.append(r)
        print(f"{d:>2} {rounds:>6} {p:>6.3f} {r['n_det']:>5} "
              f"{r['ler_mwpm']:>10.5f} {r['ler_mlp']:>10.5f} "
              f"{r['ratio_mlp_over_mwpm']:>9.3f}")
    out["points"] = rows
    best = min(rows, key=lambda r: r["ratio_mlp_over_mwpm"])
    rmin = best["ratio_mlp_over_mwpm"]
    if rmin < 0.97:
        verdict = f"MLP beats MWPM by {100 * (1 - rmin):.1f}% (best)"
    elif rmin > 1.03:
        verdict = "MWPM dominates MLP (no learned-decoder beat)"
    else:
        verdict = "tie / no beat (within noise)"
    out["verdict"] = verdict
    print("-" * 78)
    print("verdict:", out["verdict"], " (best ratio "
          f"{rmin:.3f} at d={best['d']}, p={best['p']})")
    with open(RESULTS / "e12b_surface_decoder.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {RESULTS / 'e12b_surface_decoder.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
