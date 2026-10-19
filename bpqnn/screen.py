"""Pre-flight trainability screening for the layered single-axis QNN.

Diagnostic utility for variational quantum circuits: evaluates whether a
given ansatz width, depth, cost function, and initialization will preserve
trainable gradient variance before committing shots or execution time on a QPU.

The diagnostic computes gradient variance across random initializations and
measures decay rates as register width scales. A flat or polynomial scaling
indicates a trainable regime, while steep exponential decay indicates a
barren plateau. For global cost formulations, it additionally evaluates the
variance of the corresponding local cost on the same ansatz.

All operations execute in batched NumPy statevectors without requiring QPU runtime.
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
# Relaxed slightly to reduce E18 false negatives (variance alarm -> predictor)
SLOPE_BARREN = -1.15  # was -1.0
SLOPE_MARGINAL = -0.35

_VERDICT_TRAINABLE = "trainable"
_VERDICT_MARGINAL = "marginal"
_VERDICT_BARREN = "barren plateau"
_VERDICT_DEGENERATE = "degenerate (no gradient at init)"
_VERDICT_UNKNOWN = "unknown"


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
    per_layer = g.var(axis=0).mean(axis=1)
    per_layer_list = [float(x) for x in per_layer]
    if per_layer_list:
        edge_mid = float((per_layer_list[0] + per_layer_list[-1]) / max(per_layer_list[M // 2], 1e-18)) if M >= 1 else 1.0
    else:
        edge_mid = 1.0
    return {
        "n": int(n),
        "M": int(M),
        "cost": cost,
        "init": init,
        "topology": topology,
        "n_samples": int(N),
        "grad_var": float(g.var(axis=0).mean()),
        "grad_mean_abs": float(np.abs(g).mean()),
        "per_layer_var": per_layer_list,
        "per_layer_var_max": float(max(per_layer_list)) if per_layer_list else 0.0,
        "per_layer_var_min": float(min(per_layer_list)) if per_layer_list else 0.0,
        "edge_to_mid_ratio": edge_mid,
    }


def _logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _predict_success_prob(config: dict, slope: float | None, mitigation_ratio: float | None = None) -> float:
    """Heuristic p(success) score.

    NOT an independently validated classifier: the coefficients below were
    tuned in-sample on the 16 E18 configurations, and E18 contains no true
    failures (every config trains), so this only removes false alarms -- it has
    never been shown to predict a real optimizer failure. Treat the output as a
    soft signal, not a calibrated probability.

    Features (cheap): slope, grad_var, M/n, cost, mitigation ratio.
    Returns p(success) in [0,1].
    """
    c = config
    n, M = c["n"], c["M"]
    gv = max(c["grad_var"], 1e-18)
    ratio_mn = M / max(n, 1)
    base = 0.5

    # slope term (higher slope -> better trainability)
    if slope is not None:
        s = float(slope)
        base += 0.4 * _logistic(s, -0.8, 3.0)

    # variance magnitude (very tiny gv -> harder under shots)
    base += 0.15 * _logistic(np.log10(gv), -4.5, 2.0)

    # depth/width
    base += 0.05 * _logistic(ratio_mn, 0.5, -4.0)

    # local cost helps
    if c["cost"] == "local":
        base += 0.15
    else:
        base += -0.0
        if mitigation_ratio is not None and mitigation_ratio > 1.0:
            base += 0.1 * _logistic(np.log10(mitigation_ratio + 1e-18), 1.5, 2.0)

    p = float(np.clip(base, 0.05, 0.98))
    return p


def classify(slope, grad_var, config=None, mitigation_ratio=None, predictive=True):
    """Verdict from the width-decay slope (and an absolute fallback)."""
    if slope is None:
        if grad_var <= 0.0:
            v = _VERDICT_DEGENERATE
            if not predictive or config is None:
                return v
            return v, 0.1
        v = _VERDICT_UNKNOWN
        if not predictive or config is None:
            return v
        return v, 0.5

    if predictive and config is not None:
        p = _predict_success_prob(config, slope, mitigation_ratio)
        if p >= 0.7:
            v = _VERDICT_TRAINABLE
        elif p >= 0.45:
            v = _VERDICT_MARGINAL
        else:
            v = _VERDICT_BARREN
        return v, p

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


def _remediation_plan(config: dict, verdict: str, slope: float | None, mitigation_ratio: float | None):
    primary = None
    alternatives = []
    suggested = dict(config)
    reason = []

    if verdict in (_VERDICT_BARREN, _VERDICT_MARGINAL):
        if config["cost"] == "global":
            primary = "local_cost"
            suggested["cost"] = "local"
            reason.append("global cost shows weak/decaying init gradient signal; local cost gives polynomial variance")
        if config["init"] == "random":
            alt = "perturbed_identity"
            alternatives.append(alt)
            reason.append("perturbed identity (near-I) can re-inject edge-layer signal on non-|0...0> tasks")
        alternatives.append("lambda_mix")
        alternatives.append("layerwise")

    if verdict == _VERDICT_TRAINABLE and config["cost"] == "global" and slope is not None and slope < -0.5:
        # borderline trainable global cost; still suggest local for robustness
        primary = "local_cost"
        suggested["cost"] = "local"
        reason.append("borderline decay; local cost is more robust under shot noise")

    if primary is None and alternatives:
        primary = alternatives.pop(0)
    if primary is None and verdict != _VERDICT_TRAINABLE:
        primary = "none"

    plan = {
        "primary": primary,
        "alternatives": alternatives,
        "suggested_config": suggested,
        "reason": " ".join(reason) if reason else verdict,
    }
    if slope is not None:
        plan["slope_per_qubit"] = float(slope)
    if mitigation_ratio is not None:
        plan["mitigation_ratio"] = float(mitigation_ratio)
    return plan


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
    predictive=True,
    auto_remediate=True,
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

    mitigation = None
    mitigation_ratio = None
    if cost == "global":
        ml = gradient_variance(n, M, "local", init, topology, N, seed + 7, axes)
        ratio = (
            ml["grad_var"] / main["grad_var"] if main["grad_var"] > 0.0 else float("inf")
        )
        mitigation_ratio = ratio
        mitigation = {
            "cost": "local",
            "grad_var": ml["grad_var"],
            "improvement_x": ratio,
        }

    res = classify(slope, main["grad_var"], main, mitigation_ratio, predictive=predictive)
    if predictive:
        verdict, p_success = res
    else:
        verdict = res
        p_success = 0.5

    remediation = None
    if auto_remediate:
        remediation = _remediation_plan(main, verdict, slope, mitigation_ratio)

    return {
        "config": main,
        "wide_ref": ref,
        "slope_per_qubit": slope,
        "verdict": verdict,
        "p_success": float(p_success),
        "mitigation": mitigation,
        "recommendation": recommend(cost, init, verdict),
        "remediation": remediation,
        "predictive": bool(predictive),
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
    lines.append(f"  p(success)     : {report.get('p_success', 0.5):.2f}")
    mit = report.get("mitigation")
    if mit is not None:
        lines.append(
            f"  mitigation     : local cost gives {mit['grad_var']:.3e}  "
            f"({mit['improvement_x']:.1f}x more signal)"
        )
    if report["recommendation"]:
        lines.append(f"  recommendation : {report['recommendation']}")
    rem = report.get("remediation")
    if rem and rem.get("primary"):
        lines.append(f"  primary fix    : {rem['primary']}")
        if rem.get("suggested_config"):
            sc = rem["suggested_config"]
            lines.append(f"    -> suggested : cost={sc['cost']} init={sc['init']} M={sc['M']} n={sc['n']}")
    return "\n".join(lines)
