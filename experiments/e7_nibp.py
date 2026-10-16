"""E7: noise-induced barren plateaus (NIBP) (SPEC section 9, E7).

The stretch goal in the problem statement asks us to study *noise-induced*
barren plateaus: environmental decoherence alone can make the gradient vanish
exponentially in ``n`` even for a shallow, local-cost circuit that would
otherwise be trainable. This is the effect analysed by Wang et al.,
"Noise-induced barren plateaus in variational quantum algorithms"
(Nat. Commun. 12, 6961 (2021)).

We evolve the same ansatz as the rest of the study on a *density matrix* and
apply per-qubit depolarising noise after every rotation layer (channel and
convention validated against ``qiskit-aer`` in ``tests/test_e7.py``). For the
local cost ``C_L = 1 - (1/n) sum_i P_i(0)`` -- which is polynomial/trainable
without noise (E2a) -- we measure the gradient variance over random inits:

  1. vs ``n`` at several fixed noise strengths ``p``;
  2. the *ratio* Var(p) / Var(p=0) vs ``n``. This removes the noiseless
     n-dependence entirely and exposes the pure noise-induced contraction, a
     straight line in semilog-y whose slope is the NIBP rate;
  3. the fitted NIBP rate vs ``p`` -- it should grow roughly linearly with
     ``p`` (depth x noise is what accumulates the contraction);
  4. vs ``p`` at fixed ``n``.

Identity init is never used here (degenerate on the |0...0> target, SPEC
section 6).

Run:  python experiments/e7_nibp.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.init import random_init
from bpqnn.noise import parameter_shift_grads

M = 16
N_LIST = [2, 3, 4, 5, 6]
P_LIST = [0.0, 0.05, 0.10]
N_INIT = 32
P_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
P_SWEEP_N = 6
P_SWEEP_INIT = 32
KIND = "local_zero"
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def gradients(n, M, p, B, seed, kind=KIND):
    rng = np.random.default_rng(seed)
    theta = random_init(M, n, rng, batch=B)
    return parameter_shift_grads(theta, n, p=p, kind=kind)  # (B, M, n)


def aggregate_var(g) -> float:
    return float(g.var(axis=0).mean())


def bootstrap_slope(grads_by_n, ns, n_boot=500, seed=0, ref=None):
    """OLS slope of log Var vs n (or log Var/ref if ``ref`` given), bootstrap CI."""
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        ys = np.empty(len(ns))
        for i, g in enumerate(grads_by_n):
            idx = rng.integers(0, len(g), len(g))
            v = g[idx].var(axis=0).mean()
            if ref is not None:
                r0 = ref[i]
                j = rng.integers(0, len(r0), len(r0))
                v = v / r0[j].var(axis=0).mean()
            ys[i] = np.log(v)
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
        "config": {"M": M, "n": N_LIST, "p": P_LIST, "n_init": N_INIT,
                   "p_sweep": P_SWEEP, "p_sweep_n": P_SWEEP_N,
                   "p_sweep_init": P_SWEEP_INIT, "kind": KIND,
                   "noise": "per-qubit depolarising, qiskit depolarizing_error(p,1) convention"},
    }

    print("=" * 78)
    print("E7 noise-induced barren plateaus: local-cost gradient variance")
    print("=" * 78)

    # ---- main sweep: variance vs n for each p ------------------------------
    main_rows = []
    grads = {p: [] for p in P_LIST}
    print(f"{'p':>6} {'n':>3} {'var(C_L grad)':>14} {'var/var(p=0)':>13}")
    noiseless = {}
    for p in P_LIST:
        for n in N_LIST:
            g = gradients(n, M, p, N_INIT, seed=9137 * n + int(1e5 * p))
            grads[p].append(g)
            v = aggregate_var(g)
            if p == 0.0:
                noiseless[n] = v
            main_rows.append({"p": p, "n": n, "var": v,
                              "ratio_to_noiseless": v / noiseless[n] if p else 1.0})
            print(f"{p:>6.3f} {n:>3} {v:>14.3e} "
                  f"{(v / noiseless[n] if p else 1.0):>13.3e}")
    out["main"] = main_rows

    # ---- absolute-variance slope per p -------------------------------------
    print("-" * 78)
    slopes_abs = {}
    for p in P_LIST:
        s = bootstrap_slope(grads[p], N_LIST, seed=7)
        slopes_abs[p] = s
        print(f"slope(log Var vs n)        p={p:.3f}: {s['slope']:+.3f}  "
              f"[95% CI {s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]")
    out["slopes_abs"] = {f"{p}": s for p, s in slopes_abs.items()}

    # ---- NIBP rate: slope of log(Var(p)/Var(0)) vs n -----------------------
    print("-" * 78)
    print("NIBP rate = slope of log(Var_p / Var_0) vs n  (pure noise contraction):")
    g0 = grads[0.0]
    slopes_ratio = {}
    for p in P_LIST:
        if p == 0.0:
            continue
        s = bootstrap_slope(grads[p], N_LIST, seed=11, ref=g0)
        slopes_ratio[p] = s
        kk = -s["slope"] / (p * M)
        print(f"  p={p:.3f}: rate={s['slope']:+.3f}  "
              f"[95% CI {s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]  "
              f"=> rate/(p*M) = {kk:+.4f}")
    out["slopes_ratio"] = {f"{p}": s for p, s in slopes_ratio.items()}

    ps = np.array(sorted(p for p in slopes_ratio))  # decreasing
    rates = np.array([-slopes_ratio[p]["slope"] for p in ps])
    order = np.argsort(ps)
    ps, rates = ps[order], rates[order]
    k = float(np.sum(ps * rates) / np.sum(ps * ps))
    out["nibp_rate_per_p"] = {"k": k}
    print(f"through-origin fit rate = k*p  with k = {k:+.3f} "
          f"(rate/(p*M) would be k/M = {k / M:+.4f})")

    # ---- p-sweep at fixed n ------------------------------------------------
    print("-" * 78)
    p_sweep_rows = []
    for p in P_SWEEP:
        g = gradients(P_SWEEP_N, M, p, P_SWEEP_INIT, seed=555 + int(1e5 * p))
        v = aggregate_var(g)
        p_sweep_rows.append({"p": p, "var": v,
                             "ratio_to_noiseless": v / p_sweep_rows[0]["var"] if p else 1.0})
        print(f"n={P_SWEEP_N}  p={p:.3f}  var={v:.3e}")
    out["p_sweep"] = p_sweep_rows

    with open(RESULTS / "e7_nibp.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e7_nibp.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e7_nibp.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ps = out["config"]["p"]
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(ps)))

    for color, p in zip(colors, ps):
        xs = [r["n"] for r in out["main"] if r["p"] == p]
        ys = [r["var"] for r in out["main"] if r["p"] == p]
        s = out["slopes_abs"][f"{p}"]["slope"]
        ax1.semilogy(xs, ys, "o-", color=color, label=f"p={p:g} ({s:+.2f})")
    ax1.set_xlabel("n (qubits)")
    ax1.set_ylabel("local-cost gradient variance")
    ax1.set_title("(a) variance vs n, depth M=%d" % out["config"]["M"])
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8, title="slope(log Var)")

    for color, p in zip(colors, ps):
        if p == 0.0:
            continue
        xs = [r["n"] for r in out["main"] if r["p"] == p]
        ys = [r["ratio_to_noiseless"] for r in out["main"] if r["p"] == p]
        s = out["slopes_ratio"][f"{p}"]["slope"]
        ax2.semilogy(xs, ys, "o-", color=color, label=f"p={p:g} (rate {s:+.2f})")
    ax2.axhline(1.0, color="gray", ls=":")
    ax2.set_xlabel("n (qubits)")
    ax2.set_ylabel("Var(p) / Var(p=0)")
    ax2.set_title("(b) pure noise contraction (ratio to noiseless)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8, title="slope of log ratio")

    rr = sorted((float(p), -out["slopes_ratio"][p]["slope"]) for p in out["slopes_ratio"])
    xr = [p for p, _ in rr]
    yr = [r for _, r in rr]
    k = out["nibp_rate_per_p"]["k"]
    ax3.plot(xr, yr, "o-", color="crimson", label="measured rate")
    xg = np.linspace(0, max(xr), 50)
    ax3.plot(xg, k * xg, ":", color="gray", label=f"k p, k={k:+.2f}")
    ax3.set_xlabel("noise strength p")
    ax3.set_ylabel("NIBP rate  |slope of log ratio|")
    ax3.set_title("(c) NIBP rate grows ~linearly with p")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    xs = [r["p"] for r in out["p_sweep"]]
    ys = [r["var"] for r in out["p_sweep"]]
    ax4.semilogy(xs, ys, "o-", color="navy")
    ax4.set_xlabel("noise strength p")
    ax4.set_ylabel("local-cost gradient variance")
    ax4.set_title("(d) variance vs p at fixed n=%d" % out["config"]["p_sweep_n"])
    ax4.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "e7_nibp.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
