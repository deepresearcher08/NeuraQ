"""E11: support-radius law of gradient variance (SPEC section 9).

Replaces the binary local-vs-global story with a *tunable* one, using the
observable-support family in ``bpqnn.observables``. For the target ``|0...0>``
and a contiguous block of ``r`` qubits we study two costs:

  * **single block** ``C_b = 1 - P(block of r qubits all 0)``  -- ``r`` is the
    observable's support; centred block. This is the primary quantity.
  * **window-averaged** ``C_r = 1 - (1/W) sum_w P(window w all 0)`` -- the
    natural generalisation of the E2a local cost, with endpoints
    ``r=1`` = the E2a local cost and ``r=n`` = the E1 global cost.

For each ``n`` we estimate ``Var(dC/dtheta)`` (aggregate over all ``M*n``
parameters, ``N_init`` random inits, adjoint gradients) and fit the decay in
the support radius ``r``. Findings (SPEC §14o):

  * ``Var(dC_b)`` decreases **monotonically and approximately exponentially**
    in the support radius ``r``: to leading order each extra qubit of
    observable support roughly halves the gradient variance (rate ~ ln 2),
    steepening to ~1.6-2.0 ln 2 at large ``r``/``n`` (the results JSON gives
    1.56, 1.71, 2.01 and 1.96 x ln 2 for n = 6, 8, 10, 12).
  * the *averaged* cost additionally carries a ``1/W`` normalisation that
    suppresses the ``r=1,2`` points, producing a mild non-monotonic shoulder
    (reported honestly).

Run:  python experiments/e11_support_radius.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import random_init
from bpqnn.observables import block_observable, support_radius_observable
from bpqnn.simulator import default_axes

NS = [6, 8, 10, 12]
N_INIT = 512
DEPTH_SWEEP_N = 10
DEPTH_SWEEP_M = [2, 4, 8, 16]
SEEDS = 3
LN2 = float(np.log(2.0))
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def grads_by_radius(n, M, N, r_values, seed, averaged=False):
    """{r: (N, M, n) adjoint gradients} for one (n, M) circuit ensemble."""
    rng = np.random.default_rng(seed)
    theta = random_init(M, n, rng, batch=N)
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    factory = support_radius_observable if averaged else block_observable
    return {
        r: adjoint_gradient(
            theta, n, edges, axes,
            observable=lambda s, r=r: factory(s, r, n),
        )
        for r in r_values
    }


def aggregate_var(g):
    return float(g.var(axis=0).mean())


def fit_rate(r_values, variances):
    slope = np.polyfit(r_values, np.log2(variances), 1)[0]
    return -float(slope)  # in nats (expect ~ln 2)


def bootstrap_rate(grads_by_r, r_values, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    N = next(iter(grads_by_r.values())).shape[0]
    rates = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        ys = [np.log2(grads_by_r[r][idx].var(axis=0).mean()) for r in r_values]
        rates[b] = -np.polyfit(r_values, ys, 1)[0]
    return {"rate": float(np.mean(rates)),
            "ci95": [float(np.percentile(rates, 2.5)),
                     float(np.percentile(rates, 97.5))]}


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {"config": {"n": NS, "n_init": N_INIT, "depth_sweep_n": DEPTH_SWEEP_N,
                      "depth_sweep_M": DEPTH_SWEEP_M, "seeds": SEEDS}}

    print("=" * 76)
    print("E11 support-radius law: Var(dC) vs observable support radius r")
    print("=" * 76)
    print(f"{'n':>3} {'M':>3} {'Vb(r=1)':>10} {'Vb(r=n)':>10} "
          f"{'rate_b[ln2]':>11} {'rate_avg[ln2]':>13}")
    main_rows = []
    for n in NS:
        M = n
        r_values = list(range(1, n + 1))
        gb = grads_by_radius(n, M, N_INIT, r_values, seed=1000 * n + M)
        ga = grads_by_radius(n, M, N_INIT, r_values, seed=1000 * n + M, averaged=True)
        vb = {r: aggregate_var(gb[r]) for r in r_values}
        va = {r: aggregate_var(ga[r]) for r in r_values}
        fit_r = list(range(2, n + 1))
        rate_b = fit_rate(fit_r, [vb[r] for r in fit_r])
        rate_a = fit_rate(fit_r, [va[r] for r in fit_r])
        bt = bootstrap_rate({r: gb[r] for r in fit_r}, fit_r, seed=7 * n + M)
        main_rows.append({
            "n": n, "M": M, "r": r_values,
            "var_block": [vb[r] for r in r_values],
            "var_avg": [va[r] for r in r_values],
            "rate_block": rate_b, "rate_block_over_ln2": rate_b / LN2,
            "rate_avg": rate_a, "rate_avg_over_ln2": rate_a / LN2,
            "rate_block_bootstrap": bt,
        })
        print(f"{n:>3} {M:>3} {vb[1]:>10.2e} {vb[n]:>10.2e} "
              f"{rate_b / LN2:>11.3f} {rate_a / LN2:>13.3f}   "
              f"[{bt['ci95'][0]/LN2:.3f}, {bt['ci95'][1]/LN2:.3f}]")
    out["main"] = main_rows

    print("-" * 76)
    n = DEPTH_SWEEP_N
    r_values = list(range(1, n + 1))
    depth_rows = []
    print(f"depth sweep n={n}: collapsed 2^r * Var_block")
    print(f"{'M':>3} " + " ".join(f"{str(r):>8}" for r in range(2, n + 1)))
    for M in DEPTH_SWEEP_M:
        acc = np.zeros(n + 1)
        for s in range(SEEDS):
            gb = grads_by_radius(n, M, N_INIT, r_values, seed=31 * M + s)
            acc[1:] += np.array([aggregate_var(gb[r]) for r in r_values])
        var = acc / SEEDS
        depth_rows.append({
            "M": M, "var_block": [float(var[r]) for r in r_values],
            "var_block_x2r": [float(var[r] * 2.0 ** r) for r in r_values],
            "rate_over_ln2": fit_rate(list(range(2, n + 1)),
                                      [var[r] for r in range(2, n + 1)]) / LN2,
        })
        print(f"{M:>3} " + " ".join(f"{var[r] * 2.0 ** r:>8.2e}" for r in range(2, n + 1)))
    out["depth_sweep"] = {"n": n, "rows": depth_rows}

    with open(RESULTS / "e11_support_radius.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 76)
    print(f"saved {RESULTS / 'e11_support_radius.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e11_support_radius.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")

    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    for row in out["main"]:
        r = row["r"]
        ax1.semilogy(r, row["var_block"], "o-", ms=4, label=f"n={row['n']}")
    ax1.set_xlabel("observable support radius r")
    ax1.set_ylabel("Var(dC)  (single block)")
    ax1.set_title("E11: variance falls with support radius")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)

    sw = out["depth_sweep"]
    for row in sw["rows"]:
        r = list(range(1, sw["n"] + 1))
        ax2.semilogy(r, row["var_block_x2r"], "s-", ms=4, label=f"M={row['M']}")
    ax2.set_xlabel("r")
    ax2.set_ylabel("2^r * Var(dC)  (collapsed)")
    ax2.set_title(f"E11: depth collapse (n={sw['n']})")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)

    for row in out["main"]:
        r = row["r"]
        ax3.semilogy(r, [v * 2.0 ** r[r_] for r_, v in enumerate(row["var_avg"])],
                     "^-", ms=4, label=f"n={row['n']}")
    ax3.set_xlabel("r")
    ax3.set_ylabel("2^r * Var(C_r)  (window-averaged)")
    ax3.set_title("E11: averaged cost (shoulder at small r)")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(RESULTS / "e11_support_radius.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
