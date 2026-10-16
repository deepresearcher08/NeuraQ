"""Reusable metrics for NeuraQ: bootstrap CIs, slopes, success rates."""

from __future__ import annotations

import numpy as np


def bootstrap_mean(vals, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    acc = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, vals.size, vals.size)
        acc[b] = vals[idx].mean()
    return {
        "mean": float(vals.mean()),
        "ci95": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "se": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
    }


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


def aggregate_var(g) -> float:
    return float(g.var(axis=0).mean())


def position_vars(g) -> dict:
    per_layer = g.var(axis=0).mean(axis=1)
    M = g.shape[1] if g.ndim == 3 else 1
    return {"first": float(per_layer[0]), "mid": float(per_layer[M // 2]), "last": float(per_layer[-1])}


def rel_energy_error(E_min, E0, E1) -> float:
    if E1 - E0 <= 0:
        return float("inf")
    return (E_min - E0) / (E1 - E0)


def success_rate(flags):
    if not flags:
        return 0.0
    return float(sum(bool(f) for f in flags) / len(flags))
