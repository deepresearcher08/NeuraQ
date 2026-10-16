"""Minimal benchmark scorer.

Runs a tiny E1/E2a style check with small N for speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.init import random_init
from bpqnn.simulator import default_axes

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def depth_log(n):
    return max(1, int(np.ceil(np.log2(n))))


def ansatz_grads(n, M, kind, N, seed):
    rng = np.random.default_rng(seed)
    theta = random_init(M, n, rng, batch=N)
    edges = [(q, q + 1) for q in range(n - 1)]
    axes = default_axes(M)
    return adjoint_gradient(theta, n, edges, axes, kind)


def bootstrap_slope(grads_by_n, ns, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        ys = np.empty(len(ns))
        for i, g in enumerate(grads_by_n):
            idx = rng.integers(0, len(g), len(g))
            ys[i] = np.log(g[idx].var(axis=0).mean())
        slopes[b] = np.polyfit(ns, ys, 1)[0]
    return {"slope": float(np.mean(slopes)), "ci95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="candidate")
    ap.add_argument("--ns", nargs="+", type=int, default=[4,6,8])
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    ns = args.ns
    out = {"name": args.name, "ns": list(ns)}
    gl, ll = [], []
    for n in ns:
        M = depth_log(n)
        gl.append(ansatz_grads(n, M, "global_zero", args.N, seed=100+n))
        ll.append(ansatz_grads(n, M, "local_zero", args.N, seed=200+n))
    out["global_log"] = bootstrap_slope(gl, ns, seed=1)
    out["local_log"] = bootstrap_slope(ll, ns, seed=2)
    p = args.out or (RES / f"bench_{args.name}.json")
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
