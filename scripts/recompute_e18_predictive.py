"""Re-run the E18 *prediction* side under the new predictive screen.

E18's `actual_success` comes from deterministic shot-limited Adam training, so
it is unchanged. Only the screen's verdict logic changed (predictive mode,
relaxed SLOPE_BARREN). This recomputes `predicted_success` from the stored
configs and reports old vs new confusion, writing results/e18_screen_predictive.json.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bpqnn.screen import screen

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "e18_screen_validation.json"
DST = ROOT / "results" / "e18_screen_predictive.json"

POSITIVE = ("trainable", "marginal")


def confusion(rows, key):
    tp = sum(1 for r in rows if r[key] and r["actual_success"])
    tn = sum(1 for r in rows if not r[key] and not r["actual_success"])
    fp = sum(1 for r in rows if r[key] and not r["actual_success"])
    fn = sum(1 for r in rows if not r[key] and r["actual_success"])
    n = len(rows)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / n if n else float("nan"), "n_configs": n}


def main() -> int:
    d = json.loads(SRC.read_text())
    rows = d["rows"]
    for r in rows:
        rep = screen(n=r["n"], M=r["M"], cost=r["cost"], init="random",
                     N=128, seed=0)
        r["screen_verdict_predictive"] = rep["verdict"]
        r["p_success"] = rep["p_success"]
        r["predicted_success_predictive"] = rep["verdict"] in POSITIVE

    old = confusion(rows, "predicted_success")
    new = confusion(rows, "predicted_success_predictive")

    print("OLD (legacy thresholds):", old)
    print("NEW (predictive)       :", new)

    print("\n%-3s %-3s %-7s %-16s %-6s %-6s %-5s" %
          ("n", "M", "cost", "verdict(new)", "p", "actual", "ok"))
    for r in rows:
        print("%-3d %-3d %-7s %-16s %.2f %-6s %-5s" % (
            r["n"], r["M"], r["cost"], r["screen_verdict_predictive"],
            r["p_success"], r["actual_success"],
            r["predicted_success_predictive"] == r["actual_success"]))

    out = dict(d)
    out["confusion"] = new
    out["confusion_legacy"] = old
    out["rows"] = rows
    out["note"] = ("prediction side recomputed with predictive screen; "
                   "actual_success unchanged from e18_screen_validation.json")
    DST.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
