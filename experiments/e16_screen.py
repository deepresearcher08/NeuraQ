"""E16: trainability screening CLI.

One-shot pre-flight check for a layered single-axis QNN. Given a width, depth,
cost and init, it reports the init-time gradient variance, how fast that
variance decays with width, a trainable / marginal / barren verdict, and the
recommended mitigation (with the local-cost signal measured for comparison).

    python experiments/e16_screen.py --n 8 --cost global
    python experiments/e16_screen.py --n 8 --cost global --plot

Pure NumPy; no shots and no hardware. See PITCH.md for the product framing.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.screen import COSTS, INITS, TOPOLOGIES, format_report, gradient_variance, screen

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def main(argv=None):
    ap = argparse.ArgumentParser(description="NeuraQ trainability screening")
    ap.add_argument("--n", type=int, default=8, help="number of qubits")
    ap.add_argument("--depth", type=int, default=5, help="number of rotation layers M")
    ap.add_argument("--cost", choices=COSTS, default="global")
    ap.add_argument("--init", choices=INITS, default="random")
    ap.add_argument("--topology", choices=TOPOLOGIES, default="line")
    ap.add_argument("--samples", type=int, default=256, help="random inits to average")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-step", type=int, default=2, help="width step for the slope")
    ap.add_argument("--n-cap", type=int, default=12, help="max width used for the slope")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--plot", action="store_true", help="also write a variance-vs-n chart")
    args = ap.parse_args(argv)

    report = screen(
        n=args.n,
        M=args.depth,
        cost=args.cost,
        init=args.init,
        topology=args.topology,
        N=args.samples,
        seed=args.seed,
        n_step=args.n_step,
        n_cap=args.n_cap,
    )

    print("=" * 72)
    print(format_report(report))
    print("=" * 72)

    out = args.out or (RESULTS / "e16_screen.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved {out}")

    if args.plot:
        try:
            png = out.with_suffix(".png")
            _plot(args, png)
            print(f"saved {png}")
        except Exception as e:  # noqa: BLE001
            print(f"(plot skipped: {e})")

    return 0


def _plot(args, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({w for w in range(2, args.n_cap + 1) if w % 2 == 0} | {args.n})
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for cost, color, marker in (("global", "crimson", "o-"), ("local", "navy", "s-")):
        ys = [
            gradient_variance(w, args.depth, cost, args.init, args.topology,
                              N=args.samples, seed=args.seed)["grad_var"]
            for w in widths
        ]
        ax.semilogy(widths, ys, marker, color=color, label=f"{cost} cost")
    ax.axvline(args.n, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel("gradient variance at init")
    ax.set_title(f"E16 screen: depth M={args.depth}, {args.init} init")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
