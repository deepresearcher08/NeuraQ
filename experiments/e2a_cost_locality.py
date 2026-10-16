"""E2a: cost-locality mitigation on the |0...0> protocol (SPEC section 9, E2a).

We compare the *global* cost C_G = 1 - P(0...0) with the *local* cost
C_L = 1 - (1/n) sum_i P_i(0) at fixed random init. Cerezo et al.
(arXiv:2001.00550) prove that C_G gives a barren plateau at *every* depth,
while C_L is trainable (Var ~ poly(n)) for depth L = O(log n) and only
plateaus once L = O(poly n). This experiment maps that boundary empirically:

  1. Variance vs n under two depth regimes, L = O(log n) and L = O(n), for
     both costs. Expectation: global exponential in both; local shallow
     (polynomial) in the log regime and exponential again in the linear one.
  2. A depth sweep at fixed n for the local cost, locating the depth at which
     the variance collapses (the polynomial -> exponential transition).

Random init only; identity-block / layerwise are E2b (never run here -- on the
|0...0> target an identity init has C = 0 exactly, SPEC section 6).

Run:  python experiments/e2a_cost_locality.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import random_init
from bpqnn.simulator import default_axes

N_INIT = {n: (512 if n <= 10 else 256) for n in range(2, 13)}
NS = list(range(2, 13))
DEPTH_SWEEP_NS = [8, 10]
SWEEP_N_INIT = 512
SWEEP_SEEDS = 3
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"

COSTS = ("global_zero", "local_zero")


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


def aggregate_var(g) -> float:
    return float(g.var(axis=0).mean())


def position_vars(g) -> dict:
    """Per-layer mean variance at fixed relative positions (SPEC D7)."""
    per_layer = g.var(axis=0).mean(axis=1)
    M = g.shape[1]
    return {"first": float(per_layer[0]),
            "mid": float(per_layer[M // 2]),
            "last": float(per_layer[-1])}


def bootstrap_slope(grads_by_n, ns, n_boot=500, seed=0):
    """OLS slope of log Var vs n, with a percentile bootstrap CI over inits."""
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
        "ci95": [float(np.percentile(slopes, 2.5)),
                 float(np.percentile(slopes, 97.5))],
        "se": float(np.std(slopes, ddof=1)),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {
        "config": {"n": NS, "n_init": N_INIT, "costs": list(COSTS),
                   "depth_sweep_n": DEPTH_SWEEP_NS, "sweep_n_init": SWEEP_N_INIT,
                   "sweep_seeds": SWEEP_SEEDS},
    }

    print("=" * 78)
    print("E2a cost locality: global C_G vs local C_L on |0...0>, random init")
    print("=" * 78)

    # ---- main sweep: variance vs n for each (cost, regime) -----------------
    rows = []
    grads = {(kind, regime): [] for kind in COSTS for regime in ("log", "linear")}
    print(f"{'n':>3} {'regime':>7} {'M':>3} {'N':>4} "
          f"{'C_G var':>11} {'C_L var':>11} {'C_G/C_L':>9}")
    for regime, depth_fn in (("log", depth_log), ("linear", depth_linear)):
        for n in NS:
            M = depth_fn(n)
            N = N_INIT[n]
            g_g = ansatz_gradients(n, M, "global_zero", N=N, seed=1000 * n + M)
            g_l = ansatz_gradients(n, M, "local_zero", N=N, seed=2000 * n + M)
            grads[("global_zero", regime)].append(g_g)
            grads[("local_zero", regime)].append(g_l)
            vg = aggregate_var(g_g)
            vl = aggregate_var(g_l)
            rows.append({"regime": regime, "n": n, "M": M, "N": N,
                         "global_var": vg, "local_var": vl,
                         "grad_mean_global": float(g_g.mean()),
                         "grad_mean_local": float(g_l.mean()),
                         "pos_global": position_vars(g_g),
                         "pos_local": position_vars(g_l)})
            print(f"{n:>3} {regime:>7} {M:>3} {N:>4} "
                  f"{vg:>11.3e} {vl:>11.3e} {vg / vl:>9.2e}")
    out["main"] = rows

    # ---- slopes ------------------------------------------------------------
    print("-" * 78)
    slopes = {}
    for kind in COSTS:
        for regime in ("log", "linear"):
            ns = [r["n"] for r in rows if r["regime"] == regime]
            s = bootstrap_slope(grads[(kind, regime)], ns, seed=7)
            slopes[f"{kind}/{regime}"] = s
            tag = "C_G" if kind == "global_zero" else "C_L"
            print(f"slope(log Var vs n) {tag}, L={regime:<6}: "
                  f"{s['slope']:+.3f}  [95% CI {s['ci95'][0]:+.3f}, "
                  f"{s['ci95'][1]:+.3f}]")
    out["slopes"] = slopes

    # ---- depth boundary for the local cost ---------------------------------
    print("-" * 78)
    print("local-cost depth boundary (where Var collapses):")
    boundary = []
    for n in DEPTH_SWEEP_NS:
        Ls = list(range(1, 2 * n + 1))
        varl = []
        for L in Ls:
            accs = []
            for s in range(SWEEP_SEEDS):
                g = ansatz_gradients(n, L, "local_zero", N=SWEEP_N_INIT,
                                     seed=4242 * s + 13 * L + n)
                accs.append(aggregate_var(g))
            varl.append(float(np.mean(accs)))
        varl = np.array(varl)
        v0 = varl[0]
        drop = np.where(varl < v0 / 10.0)[0]
        L_drop = int(Ls[drop[0]]) if len(drop) else None
        plateau = float(np.median(varl[-3:]))
        boundary.append({"n": n, "L": Ls, "local_var": [float(x) for x in varl],
                         "var_L1": float(v0), "plateau": plateau,
                         "L_drop_factor10": L_drop, "log2_n": depth_log(n)})
        print(f"  n={n:>2}: Var(L=1)={v0:.3e}  plateau={plateau:.3e}  "
              f"L (x1/10 drop)={L_drop}  ceil(log2 n)={depth_log(n)}")
    out["depth_boundary"] = boundary

    with open(RESULTS / "e2a_cost_locality.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e2a_cost_locality.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e2a_cost_locality.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    styles = {
        ("global_zero", "log"): ("r", "o-", "C_G  L=O(log n)"),
        ("global_zero", "linear"): ("darkred", "s--", "C_G  L=O(n)"),
        ("local_zero", "log"): ("b", "o-", "C_L  L=O(log n)"),
        ("local_zero", "linear"): ("navy", "s--", "C_L  L=O(n)"),
    }
    for (kind, regime), (color, mk, label) in styles.items():
        xs = [r["n"] for r in out["main"] if r["regime"] == regime]
        key = "global_var" if kind == "global_zero" else "local_var"
        ys = [r[key] for r in out["main"] if r["regime"] == regime]
        s = out["slopes"][f"{kind}/{regime}"]["slope"]
        ax1.semilogy(xs, ys, mk, color=color,
                     label=f"{label} (slope {s:+.2f})")
    ns = np.array(out["config"]["n"])
    ax1.semilogy(ns, 1e-2 / ns**2, ":", color="gray", label="1/n^2 guide")
    ax1.set_xlabel("n (qubits)")
    ax1.set_ylabel("gradient variance (aggregate)")
    ax1.set_title("E2a: global vs local cost, variance vs n")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    for b in out["depth_boundary"]:
        ax2.semilogy(b["L"], b["local_var"], "o-", label=f"n={b['n']}")
        ax2.axvline(b["log2_n"], color="gray", ls=":", alpha=0.6)
    ax2.set_xlabel("depth L")
    ax2.set_ylabel("C_L gradient variance")
    ax2.set_title("E2a: local-cost depth boundary (dotted = ceil(log2 n))")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e2a_cost_locality.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
