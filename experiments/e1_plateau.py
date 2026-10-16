"""E1: gradient-variance barren-plateau sweep (SPEC section 9, D3/D4/D7).

Global |0...0> cost, random init only. For n = 2..12 and two depth regimes
(L = O(log n) and L = O(n)) we estimate the per-parameter gradient variance
over N_init random inits (adjoint gradients, batched). We report the aggregate
variance, the per-layer profile, and the variance at fixed relative positions
(first / middle / last layer, D7). We fit log-variance vs n (OLS with a
bootstrap CI on the slope, resampling the inits) and overlay the universal
2-design (Haar) reference from ``bpqnn.reference``. A depth sweep at fixed n
shows convergence to the 2-design value.

Run:  python experiments/e1_plateau.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import random_init
from bpqnn.reference import two_design_var
from bpqnn.simulator import default_axes

N_INIT = {n: (512 if n <= 10 else 256) for n in range(2, 13)}
NS = list(range(2, 13))
DEPTH_SWEEP_N = 8
DEPTH_SWEEP_L = list(range(1, 17))
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def depth_log(n: int) -> int:
    return max(1, int(np.ceil(np.log2(n))))


def depth_linear(n: int) -> int:
    return n


def ansatz_gradients(n, M, kind="global_zero", N=512, seed=0, axes=None):
    rng = np.random.default_rng(seed)
    theta = random_init(M, n, rng, batch=N)
    edges = [(q, q + 1) for q in range(n - 1)]
    if axes is None:
        axes = default_axes(M)
    return adjoint_gradient(theta, n, edges, axes, kind)  # (N, M, n)


def variance_summary(g):
    """g: (N, M, n) -> per-parameter variance and summaries."""
    per_param = g.var(axis=0)  # (M, n)
    per_layer = per_param.mean(axis=1)  # (M,)
    M = g.shape[1]
    positions = {
        "first": float(per_layer[0]),
        "mid": float(per_layer[M // 2]),
        "last": float(per_layer[-1]),
    }
    return per_param, per_layer, float(per_param.mean()), positions


def bootstrap_slope(grads_by_n, ns, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        ys = np.empty(len(ns))
        for i, g in enumerate(grads_by_n):
            idx = rng.integers(0, len(g), len(g))
            ys[i] = np.log(g[idx].var(axis=0).mean())
        slopes[b] = np.polyfit(ns, ys, 1)[0]
    return {
        "slope": float(np.mean(slopes)),
        "ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))],
        "se": float(np.std(slopes, ddof=1)),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(2024)
    out = {"config": {"n": NS, "n_init": N_INIT, "depth_sweep_n": DEPTH_SWEEP_N}}

    print("=" * 72)
    print("E1 plateau: global |0...0> cost, random init")
    print("=" * 72)
    print(f"{'n':>3} {'regime':>7} {'M':>3} {'N':>4} {'agg Var':>11} "
          f"{'std(g)':>10} {'first':>10} {'mid':>10} {'last':>10}")
    rows = []
    grads_by_regime = {"log": [], "linear": []}
    for regime, depth_fn in (("log", depth_log), ("linear", depth_linear)):
        for n in NS:
            M = depth_fn(n)
            N = N_INIT[n]
            g = ansatz_gradients(n, M, N=N, seed=1000 * n + M)
            grads_by_regime[regime].append(g)
            per_param, per_layer, agg, pos = variance_summary(g)
            rows.append({
                "regime": regime, "n": n, "M": M, "N": N,
                "agg_var": agg, "grad_mean": float(g.mean()),
                "per_layer": [float(x) for x in per_layer],
                "pos": pos,
                "log10_var": float(np.log10(agg)),
            })
            print(f"{n:>3} {regime:>7} {M:>3} {N:>4} {agg:>11.3e} "
                  f"{np.sqrt(agg):>10.3e} {pos['first']:>10.3e} "
                  f"{pos['mid']:>10.3e} {pos['last']:>10.3e}")
    out["main"] = rows

    print("-" * 72)
    slopes = {}
    for regime in ("log", "linear"):
        ns = [r["n"] for r in rows if r["regime"] == regime]
        s = bootstrap_slope(grads_by_regime[regime], ns, seed=7)
        slopes[regime] = s
        print(f"slope(log Var vs n), L={regime:<6}: "
              f"{s['slope']:.3f}  [95% CI {s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]")
    out["slopes"] = slopes

    # 2-design reference overlay
    print("-" * 72)
    ref = []
    for n in NS:
        var, mean, se = two_design_var(n, n_samples=100000, rng=rng)
        ref.append({"n": n, "var": var, "se": se})
        print(f"2-design ref n={n:>2}: Var={var:.4e}  (log10={np.log10(var):.2f})")
    out["reference"] = ref
    ref_ns = np.array([r["n"] for r in ref])
    ref_slope = np.polyfit(ref_ns, np.log([r["var"] for r in ref]), 1)[0]
    out["reference_slope"] = float(ref_slope)
    print(f"2-design reference slope(log Var vs n) = {ref_slope:.3f}")

    # depth sweep at fixed n (fixed Y/Z alternation, averaged over seeds so the
    # curve isolates depth rather than the axis schedule)
    print("-" * 72)
    sweep = {"n": DEPTH_SWEEP_N, "L": [], "agg_var": [], "per_layer_last": []}
    axes_alt = lambda L: ["Y" if j % 2 == 0 else "Z" for j in range(L)]
    for L in DEPTH_SWEEP_L:
        N = 1024
        accs = []
        lasts = []
        for s in range(4):
            g = ansatz_gradients(DEPTH_SWEEP_N, L, N=N, seed=1234 * s + L,
                                 axes=axes_alt(L))
            _, per_layer, agg, _ = variance_summary(g)
            accs.append(agg)
            lasts.append(float(per_layer[-1]))
        agg = float(np.mean(accs))
        last = float(np.mean(lasts))
        sweep["L"].append(L)
        sweep["agg_var"].append(agg)
        sweep["per_layer_last"].append(last)
        print(f"depth sweep n={DEPTH_SWEEP_N} L={L:>2}: agg Var={agg:.3e}")
    out["depth_sweep"] = sweep

    with open(RESULTS / "e1_plateau.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 72)
    print(f"saved {RESULTS / 'e1_plateau.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e1_plateau.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ref = out["reference"]
    rn = [r["n"] for r in ref]
    rv = [r["var"] for r in ref]
    ax1.semilogy(rn, rv, "k--", label="2-design (Haar)")

    for regime, mk in (("log", "o-"), ("linear", "s-")):
        xs = [r["n"] for r in out["main"] if r["regime"] == regime]
        ys = [r["agg_var"] for r in out["main"] if r["regime"] == regime]
        s = out["slopes"][regime]
        ax1.semilogy(xs, ys, mk, label=f"ansatz L={regime} "
                     f"(slope {s['slope']:.2f})")
    ax1.set_xlabel("n (qubits)")
    ax1.set_ylabel("gradient variance (aggregate)")
    ax1.set_title("E1: gradient variance vs n, global cost")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    sw = out["depth_sweep"]
    ref_at = [r["var"] for r in ref if r["n"] == sw["n"]]
    if ref_at:
        ax2.axhline(ref_at[0], color="k", ls="--", label="2-design (Haar)")
    ax2.semilogy(sw["L"], sw["agg_var"], "o-", label="ansatz (agg)")
    ax2.semilogy(sw["L"], sw["per_layer_last"], "^:", label="last layer")
    ax2.set_xlabel("depth L")
    ax2.set_ylabel("gradient variance")
    ax2.set_title(f"E1: depth sweep (n={sw['n']})")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e1_plateau.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
