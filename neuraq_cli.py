"""neuraq CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bpqnn import screen


def cmd_screen(argv):
    ap = argparse.ArgumentParser("neuraq screen")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--cost", choices=["global", "local"], default="global")
    ap.add_argument("--init", choices=["random", "identity"], default="random")
    ap.add_argument("--topology", choices=["line", "ring", "star", "complete"], default="line")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--auto-patch", action="store_true")
    args = ap.parse_args(argv)
    rep = screen.screen(args.n, args.depth, args.cost, args.init, args.topology, args.samples, args.seed, auto_remediate=True)
    print(screen.format_report(rep))
    if args.auto_patch:
        p = Path("results") / "neuraq_patch.json"
        p.parent.mkdir(exist_ok=True)
        with open(p, "w") as f:
            json.dump(rep["remediation"], f, indent=2)
        print(f"saved {p}")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] != "screen":
        ap = argparse.ArgumentParser(prog="neuraq")
        sub = ap.add_subparsers(dest="cmd")
        sub.add_parser("screen")
        ns = ap.parse_args(argv)
        if ns.cmd == "screen":
            return cmd_screen([])
        ap.print_help()
        return 1
    return cmd_screen(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
