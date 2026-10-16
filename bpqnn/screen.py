"""Pre-flight trainability screening for the layered single-axis QNN.

Cheap diagnostic for the question a team asks before it spends shots or QPU
time on a variational model: *given this width, depth, cost and init, will the
gradient carry any signal at initialisation, or are we on a barren plateau?*

It works exactly the way the rest of the toolkit does -- the init-time
gradient variance of the chosen cost -- and then measures how fast that
variance decays as the circuit is widened by two qubits. A flat (polynomial)
decay means the model is trainable; a steep exponential decay is the
plateau signature. For a global cost it also measures the local-cost
variance on the same ansatz so the report can quantify the fix.

Everything here is pure NumPy: no shots, no hardware, no optimiser. It is the
screening layer behind E2a/E15 (see PITCH.md).
"""

from __future__ import annotations

import numpy as np

from .ansatz import make_edges
from .grad import adjoint_gradient
from .init import identity_block_init, random_init
from .simulator import default_axes

COSTS = ("global", "local")
INITS = ("random", "identity")
TOPOLOGIES = ("line", "ring", "star", "complete")

_KIND = {"global": "global_zero", "local": "local_zero"}

# Empirical thresholds on d(log Var)/dn, calibrated against the E1/E2a sweeps:
# a shallow-window global cost sits near -1.2 per qubit, a local cost near
# -0.5, and a flat/polynomial regime stays above about -0.35.
SLOPE_BARREN = -1.0
SLOPE_MARGINAL = -0.35

_VERDICT_TRAINABLE = "trainable"
_VERDICT_MARGINAL = "marginal"
_VERDICT_BARREN = "barren plateau"


def _grads(n, M, cost, init, topology, N, seed, axes):
    if cost not in COSTS:
        raise ValueError(f"unknown cost {cost!r}; pick one of {COSTS}")
    if init not in INITS:
        raise ValueError(f"unknown init {init!r}; pick one of {INITS}")
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; pick one of {TOPOLOGIES}")
    rng = np.random.default_rng(seed)
    edges = make_edges(n, topology)
    if axes is None:
        axes = default_axes(M)
    if init == "random":
        theta = random_init(M, n, rng, batch=N)
    else:
        theta = identity_block_init(M, n, rng, batch=N)
    return adjoint_gradient(theta, n, edges, axes, _KIND[cost])


def gradient_variance(
    n, M=5, cost="global", init="random", topology="line", N=256, seed=0, axes=None
):
    """Init-time gradient variance of a configuration (aggregate over params)."""
    g = _grads(n, M, cost, init, topology, N, seed, axes)
    return {
        "n": int(n),
        "M": int(M),
        "cost": cost,
        "init": init,
        "topology": topology,
        "n_samples": int(N),
        "grad_var": float(g.var(axis=0).mean()),
        "grad_mean_abs": float(np.abs(g).mean()),
        "per_layer_var": [float(x) for x in g.var(axis=0).mean(axis=1)],
    }


def classify(slope, grad_var):
    """Verdict from the width-decay slope (and an absolute fallback)."""
    if slope is None:
        if grad_var <= 0.0:
            return "degenerate (no gradient at init)"
        return "unknown"
    if slope <= SLOPE_BARREN:
        return _VERDICT_BARREN
    if slope <= SLOPE_MARGINAL:
        return _VERDICT_MARGINAL
    return _VERDICT_TRAINABLE


def recommend(cost, init, verdict):
    """One-line suggested action, or ``None`` when nothing needs changing."""
    if verdict == _VERDICT_TRAINABLE:
        return None
    if cost == "global":
        return "switch to a local cost C_L (gradient signal is polynomially large -- E2a/E15)"
    if init == "random":
        return "try identity-block init on the Ising / random-target task (not on |0...0>)"
    return "reduce depth: local costs return to a plateau at depth O(poly n) -- E2a"


def screen(
    n,
    M=5,
    cost="global",
    init="random",
    topology="line",
    N=256,
    seed=0,
    n_step=2,
    n_cap=12,
    axes=None,
):
    """Screen one configuration and return a JSON-serialisable report.

    Widens the circuit by ``n_step`` (capped at ``n_cap``) to estimate the decay
    of the gradient variance, classifies the result, and -- for a global cost --
    measures the local-cost variance as the recommended mitigation.
    """
    main = gradient_variance(n, M, cost, init, topology, N, seed, axes)

    ref = None
    slope = None
    ref_n = min(int(n) + int(n_step), int(n_cap))
    if ref_n > n:
        ref = gradient_variance(ref_n, M, cost, init, topology, N, seed + 1, axes)
        if main["grad_var"] > 0.0 and ref["grad_var"] > 0.0:
            slope = float(
                (np.log(ref["grad_var"]) - np.log(main["grad_var"])) / (ref_n - n)
            )

    verdict = classify(slope, main["grad_var"])

    mitigation = None
    if cost == "global":
        ml = gradient_variance(n, M, "local", init, topology, N, seed + 7, axes)
        ratio = (
            ml["grad_var"] / main["grad_var"] if main["grad_var"] > 0.0 else float("inf")
        )
        mitigation = {
            "cost": "local",
            "grad_var": ml["grad_var"],
            "improvement_x": ratio,
        }

    return {
        "config": main,
        "wide_ref": ref,
        "slope_per_qubit": slope,
        "verdict": verdict,
        "mitigation": mitigation,
        "recommendation": recommend(cost, init, verdict),
    }


def format_report(report):
    """Human-readable multi-line summary of a :func:`screen` report."""
    c = report["config"]
    lines = [
        "NeuraQ trainability screen",
        f"  configuration  : n={c['n']}  depth M={c['M']}  cost={c['cost']}  "
        f"init={c['init']}  topology={c['topology']}",
        f"  gradient var   : {c['grad_var']:.3e}   "
        f"(mean |dC/dtheta| = {c['grad_mean_abs']:.3e})",
    ]
    ref = report.get("wide_ref")
    if ref is not None and report["slope_per_qubit"] is not None:
        lines.append(
            f"  decay slope    : {report['slope_per_qubit']:+.2f} / qubit  "
            f"(n={c['n']} -> n={ref['n']})"
        )
    else:
        lines.append("  decay slope    : not measurable")
    lines.append(f"  verdict        : {report['verdict'].upper()}")
    mit = report.get("mitigation")
    if mit is not None:
        lines.append(
            f"  mitigation     : local cost gives {mit['grad_var']:.3e}  "
            f"({mit['improvement_x']:.1f}x more signal)"
        )
    if report["recommendation"]:
        lines.append(f"  recommendation : {report['recommendation']}")
    return "\n".join(lines)
