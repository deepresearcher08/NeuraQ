"""Calibrate the trainability screen against E18 validation results.

The current screen uses hard thresholds (SLOPE_BARREN=-1.0, SLOPE_MARGINAL=-0.35)
and only a binary verdict. Here we look at the false negatives from E18:
- n=6,M=3,global: slope=-1.23 succ=1.0 (correctly flagged as barren but actually succeeded)
- n=8,M=3,global: slope=-1.38 succ=1.0
- n=10,M=6,global: slope=-1.24 succ=1.0

So "barren plateau" verdict (slope < -1.0) is too conservative under Adam + finite
shots at these widths; many configs still train. Also the step of n_step=2 in
screen() gives a very noisy single-slope estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "e18_screen_validation.json"


def main():
    d = json.load(open(RES))
    fn = [r for r in d["rows"] if r["predicted_success"] is False and r["actual_success"] is True]
    tp = [r for r in d["rows"] if r["predicted_success"] is True and r["actual_success"] is True]
    tn = [r for r in d["rows"] if r["predicted_success"] is False and r["actual_success"] is False]
    fp = [r for r in d["rows"] if r["predicted_success"] is True and r["actual_success"] is False]
    print("E18 confusion:", d["confusion"])
    print("FN (barren but succeeded):", len(fn))
    for f in fn[:5]:
        print("  ", f)
    # proposed rule ideas
    print("\nProposed tweaks:")
    print("1. Relax SLOPE_BARREN: e.g. -1.25 (capture extreme cases) but better to add other features.")
    print("2. Don't predict failure purely on slope< -1.0 when cost is local? (local never triggers FN here.)")
    print("3. Use predictive scoring: slope, M/n, grad_var magnitude, ratio (local/global).")
    print("4. For the minimal change: in screen(), when cost=='global', also require slope < -1.15 AND grad_var < 1e-3? Or just be less aggressive.")


if __name__ == "__main__":
    main()
