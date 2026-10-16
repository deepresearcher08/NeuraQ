"""Merge the E15 hardware points into one record for the write-up.

Combines the original n=2..8 run (results/e15_qpu.json) with the budget-limited
extensions n=10 (R=48) and n=12 (R=32, reduced inits because the free-plan
runtime had only ~142 s left). Writes results/e15_qpu_extended.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> int:
    base = json.loads((RES / "e15_qpu.json").read_text())
    rows = list(base["rows"])
    for tag in ("e15_hw_n10", "e15_hw_n12"):
        p = RES / f"{tag}.json"
        if p.exists():
            rows.extend(json.loads(p.read_text())["rows"])
    rows.sort(key=lambda r: r["n"])

    ns = np.array([r["n"] for r in rows], dtype=float)
    slopes = {}
    for k in ("global_var_hw", "local_var_hw"):
        ys = np.log([r[k] for r in rows])
        slopes[k] = float(np.polyfit(ns, ys, 1)[0])

    out = {
        "config": dict(base["config"],
                       note="n=10 uses R=48; n=12 uses R=32 (free-plan budget limit). "
                            "All on ibm_fez, M=4, shots=512, seed=2026, axes=Y."),
        "rows": rows,
        "sources": ["results/e15_qpu.json", "results/e15_hw_n10.json",
                    "results/e15_hw_n12.json"],
        "slopes_log_var_per_n": slopes,
    }
    jobs_path = RES / "e15_qpu_jobs.json"
    if jobs_path.exists():
        out["jobs"] = json.loads(jobs_path.read_text())
    dst = RES / "e15_qpu_extended.json"
    dst.write_text(json.dumps(out, indent=2))
    print("n:", [r["n"] for r in rows])
    print("slopes:", slopes)
    print(f"saved {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
