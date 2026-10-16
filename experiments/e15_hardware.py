"""E15 -- barren plateau on real hardware.

Ported from E2a: layered HEA, random init, target |0...0>. For each init we
parameter-shift every qubit of the middle rotation layer and read both costs
off the same bitstrings.

    C_G = 1 - P(0...0)              global
    C_L = 1 - (1/n) sum_i P_i(0)    local

The gradient variance across inits is the plateau diagnostic. Runs on aer by
default; ``--backend least_busy`` (or a backend name) submits to IBM Quantum.
The token is read from a file and never printed.

    python experiments/e15_hardware.py                       # aer, noiseless
    python experiments/e15_hardware.py --noise device        # aer + device noise
    python experiments/e15_hardware.py --backend least_busy  # real QPU
    python experiments/e15_hardware.py --dry-run             # build only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.grad import adjoint_gradient
from bpqnn.hardware import build_circuit, run_gradients, sample_variance
from bpqnn.init import random_init
from bpqnn.simulator import default_axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# device + backend plumbing
# --------------------------------------------------------------------------- #
def load_token(path: str) -> str:
    """Read the token from a file, without printing it."""
    txt = pathlib.Path(path).read_text().strip()
    if not txt:
        raise ValueError(f"token file {path} is empty")
    if txt.startswith("{"):  # JSON form
        obj = json.loads(txt)
        for k in ("token", "api_key", "apikey", "IBMQ_TOKEN", "key"):
            if k in obj:
                return str(obj[k]).strip()
    if "=" in txt:  # KEY=VALUE form
        return txt.split("=", 1)[1].strip().strip('"').strip("'")
    return txt  # bare token (single line)


def load_instance(path: str | None) -> str | None:
    """Read the instance CRN from a file."""
    if not path:
        return None
    txt = pathlib.Path(path).read_text().strip()
    return txt or None


def get_service(token_file: str, channel: str | None,
                instance_file: str | None = None):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = load_token(token_file)
    instance = load_instance(instance_file)
    channels = [channel] if channel else ["ibm_quantum_platform", "ibm_cloud"]
    errs = []
    for ch in channels:
        try:
            return QiskitRuntimeService(channel=ch, token=token,
                                        instance=instance)
        except Exception as e:  # noqa: BLE001
            errs.append(f"{ch}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("could not instantiate QiskitRuntimeService -> " +
                       " | ".join(errs))


def select_backend(service, name: str):
    if name == "least_busy":
        return service.least_busy(operational=True, simulator=False,
                                  min_num_qubits=8)
    return service.backend(name)


def find_line_chain(backend, n: int):
    """A simple path of ``n`` physical qubits in the coupling graph (no SWAPs)."""
    try:
        cm = backend.coupling_map
    except Exception:  # noqa: BLE001
        return None
    if cm is None:
        return None
    adj: dict = {}
    for a, b in cm.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for s in sorted(adj):
        stack = [(s, [s], {s})]
        while stack:
            cur, path, vis = stack.pop()
            if len(path) == n:
                return path
            for nb in sorted(adj[cur], reverse=True):
                if nb not in vis:
                    stack.append((nb, path + [nb], vis | {nb}))
    return None


def aer_sampler(noise: str | None, seed: int):
    from qiskit_aer.primitives import SamplerV2
    if noise is None:
        return SamplerV2(seed=seed)
    from qiskit_aer import AerSimulator
    if noise == "device":
        from qiskit_ibm_runtime.fake_provider import FakeBrisbane
        from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

        b = FakeBrisbane()
        p1 = float(np.median([p.error for _, p in b.target["sx"].items()
                              if p is not None and p.error is not None]))
        p2 = float(np.median([p.error for _, p in b.target["ecr"].items()
                              if p is not None and p.error is not None]))
        pro = float(np.median([p.error for _, p in b.target["measure"].items()
                               if p is not None and p.error is not None]))
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["ry", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cz"])
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - pro, pro], [pro, 1 - pro]]))
        return SamplerV2.from_backend(AerSimulator(noise_model=nm), seed=seed)
    raise ValueError(f"unknown --noise {noise!r}")


def transpile_for(qc, backend, n):
    from qiskit import transpile

    layout = find_line_chain(backend, n)
    kw = dict(optimization_level=3, seed_transpiler=1234)
    if layout is not None:
        kw["initial_layout"] = layout
    tqc = transpile(qc, backend=backend, **kw)
    return tqc, layout


# --------------------------------------------------------------------------- #
def bootstrap_stats(grads, ns, n_boot=1000, seed=0):
    """Percentile bootstrap over inits: per-n variance CIs and the log-Var slope CI.

    ``grads`` is a list (one per n) of ``(R, P)`` arrays; resampling rows keeps
    the init-level correlation between the P probe components.
    """
    rng = np.random.default_rng(seed)
    R = grads[0].shape[0]
    xs = np.asarray(ns, dtype=float)
    var_boot = np.empty((n_boot, len(ns)))
    slopes = np.empty(n_boot)
    for b in range(n_boot):
        ys = np.empty(len(ns))
        for i, g in enumerate(grads):
            idx = rng.integers(0, R, R)
            var_boot[b, i] = g[idx].var(axis=0, ddof=1).mean()
            ys[i] = np.log(var_boot[b, i])
        slopes[b] = np.polyfit(xs, ys, 1)[0]
    return {
        "var_ci": [[float(np.percentile(var_boot[:, i], 2.5)),
                    float(np.percentile(var_boot[:, i], 97.5))]
                   for i in range(len(ns))],
        "slope": float(np.mean(slopes)),
        "slope_ci": [float(np.percentile(slopes, 2.5)),
                     float(np.percentile(slopes, 97.5))],
    }


def run(backend_kind: str, args) -> dict:
    NS = list(args.ns)
    M = args.M
    R = args.r
    shots = args.shots

    sampler = None
    backend = None
    layout_info = {}
    if backend_kind == "aer":
        sampler = aer_sampler(args.noise, seed=args.seed)
    else:
        service = get_service(args.token_file, args.channel,
                              args.instance_file)
        backend = select_backend(service, backend_kind)
        print(f"[hw] backend = {backend.name}  ({backend.num_qubits} qubits)")
        from qiskit_ibm_runtime import SamplerV2 as IBMSampler
        sampler = IBMSampler(mode=backend)

    rows = []
    gradsG, gradsL, ns_used = [], [], []
    for n in NS:
        edges = [(q, q + 1) for q in range(n - 1)]
        axes = ["Y"] * M if args.axes == "Y" else default_axes(M)
        l0 = M // 2
        probes = [(l0, q) for q in range(n)]
        rng = np.random.default_rng(args.seed + 7 * n + 1)
        theta = random_init(M, n, rng, batch=R)
        qc, params = build_circuit(n, M, edges, axes, measure=True)

        # ideal (adjoint) reference on the same inits
        aG = adjoint_gradient(theta, n, edges, axes, "global_zero")
        aL = adjoint_gradient(theta, n, edges, axes, "local_zero")
        vG_ideal = float(np.mean([aG[:, l0, q].var(ddof=1) for q in range(n)]))
        vL_ideal = float(np.mean([aL[:, l0, q].var(ddof=1) for q in range(n)]))

        tqc, layout = qc, None
        if backend is not None:
            tqc, layout = transpile_for(qc, backend, n)
            layout_info[n] = {"layout": layout, "depth": tqc.depth(),
                              "ops": {k: int(v) for k, v in tqc.count_ops().items()}}
        creg = tqc.cregs[0].name if tqc.cregs else "meas"

        n_circ = 2 * R * len(probes)
        print(f"[run] n={n} M={M} R={R} probes={len(probes)} shots={shots} "
              f"circuits={n_circ}")
        if args.dry_run:
            continue

        t0 = time.time()
        gG, gL = run_gradients(tqc, params, theta, probes, sampler,
                               shots=shots, creg=creg)
        dt = time.time() - t0
        vG = sample_variance(gG)
        vL = sample_variance(gL)
        gradsG.append(gG); gradsL.append(gL); ns_used.append(n)
        rows.append({
            "n": n, "M": M, "R": R, "shots": shots, "circuits": n_circ,
            "seconds": dt,
            "global_var_hw": vG, "local_var_hw": vL,
            "global_var_ideal": vG_ideal, "local_var_ideal": vL_ideal,
            "ratio_hw": vG / vL if vL > 0 else None,
            "ratio_ideal": vG_ideal / vL_ideal,
        })
        print(f"      Var_gG={vG:.3e} (ideal {vG_ideal:.3e})   "
              f"Var_gL={vL:.3e} (ideal {vL_ideal:.3e})   "
              f"G/L={vG / vL:.2e}   {dt:.1f}s")

    out = {
        "config": {"backend": backend_kind, "noise": args.noise, "ns": NS,
                   "M": M, "R": R, "shots": shots, "seed": args.seed,
                   "axes": args.axes, "probe_layer": "middle"},
        "layout_info": layout_info,
        "rows": rows,
    }
    if gradsG:
        out["bootstrap"] = {
            "global": bootstrap_stats(gradsG, ns_used, seed=args.seed),
            "local": bootstrap_stats(gradsL, ns_used, seed=args.seed + 1),
        }
    return out


def plot(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = out["rows"]
    ns = [r["n"] for r in rows]
    boot = out.get("bootstrap", {})
    fig, ax = plt.subplots(figsize=(6.4, 4.6))

    def yerr(key):
        if key not in boot:
            return None
        ci = boot[key]["var_ci"]
        pt = [r["global_var_hw"] if key == "global" else r["local_var_hw"]
              for r in rows]
        lo = [max(p - c[0], 0.0) for p, c in zip(pt, ci)]
        hi = [c[1] - p for p, c in zip(pt, ci)]
        return np.array([lo, hi])

    ax.errorbar(ns, [r["global_var_hw"] for r in rows],
                yerr=yerr("global"), fmt="o-", color="crimson", capsize=3,
                label="global  $C_G$  (QPU, 95% CI)")
    ax.errorbar(ns, [r["local_var_hw"] for r in rows],
                yerr=yerr("local"), fmt="s-", color="navy", capsize=3,
                label="local  $C_L$  (QPU, 95% CI)")
    ax.semilogy(ns, [r["global_var_ideal"] for r in rows], "o--", color="crimson",
                alpha=0.45, label="global (ideal adjoint)")
    ax.semilogy(ns, [r["local_var_ideal"] for r in rows], "s--", color="navy",
                alpha=0.45, label="local (ideal adjoint)")
    ax.set_xlabel("n (qubits)")
    ax.set_ylabel(r"Var$(\partial C/\partial\theta)$  (middle layer)")
    ax.set_title("Barren plateau on real hardware:\n"
                 "global gradient collapses, local survives")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="aer",
                    help="aer | least_busy | <IBM backend name>")
    ap.add_argument("--noise", default=None,
                    help="aer only: 'device' to apply Brisbane-magnitude noise")
    ap.add_argument("--token-file", default=str(ROOT / "ibmapi.txt"))
    ap.add_argument("--instance-file", default=str(ROOT / "ibmcrn.txt"))
    ap.add_argument("--channel", default=None)
    ap.add_argument("--shots", type=int, default=2048)
    ap.add_argument("--r", type=int, default=48, help="inits per n")
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--axes", default="Y", choices=["Y", "default"],
                    help="'Y' = all-RY HEA (genuinely entangling); 'default' = palindromic Y/Z")
    ap.add_argument("--ns", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="e15_hardware")
    ap.add_argument("--replot-from", default=None,
                    help="rebuild the figure from a stored results JSON "
                         "(no token or network needed)")
    args = ap.parse_args(argv)

    if args.replot_from:
        src = pathlib.Path(args.replot_from)
        if not src.is_absolute():
            src = ROOT / src
        out = json.loads(src.read_text())
        RESULTS.mkdir(exist_ok=True)
        plot(out, RESULTS / f"{args.tag}.png")
        print(f"[saved] {RESULTS / (args.tag + '.png')}  (from {src})")
        return 0

    RESULTS.mkdir(exist_ok=True)
    out = run(args.backend, args)
    out_path = RESULTS / f"{args.tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {out_path}")

    if out["rows"]:
        try:
            plot(out, RESULTS / f"{args.tag}.png")
            print(f"[saved] {RESULTS / (args.tag + '.png')}")
        except Exception as e:  # noqa: BLE001
            print(f"(plot skipped: {e})")
        # headline
        rows = out["rows"]
        r8 = [r for r in rows if r["n"] == max(r["n"] for r in rows)]
        if r8:
            r8 = r8[0]
            print("=" * 66)
            print(f"HEADLINE: n={r8['n']}  Var_gG={r8['global_var_hw']:.2e}  "
                  f"Var_gL={r8['local_var_hw']:.2e}  "
                  f"local/global = {1.0 / r8['ratio_hw']:.0f}x")
            slopes = {}
            for key in ("global_var_hw", "local_var_hw"):
                ys = np.log([r[key] for r in rows])
                xs = np.array([r["n"] for r in rows])
                slopes[key] = float(np.polyfit(xs, ys, 1)[0])
            bs = out.get("bootstrap", {})
            gci = bs.get("global", {}).get("slope_ci")
            lci = bs.get("local", {}).get("slope_ci")
            gtxt = "" if not gci else "[95%% CI %.2f, %.2f]" % (gci[0], gci[1])
            ltxt = "" if not lci else "[95%% CI %.2f, %.2f]" % (lci[0], lci[1])
            print(f"slope d log Var / dn:  global {slopes['global_var_hw']:+.2f} {gtxt}"
                  f"  local {slopes['local_var_hw']:+.2f} {ltxt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
