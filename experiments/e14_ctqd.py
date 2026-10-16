"""E14: CTQD -- train classically, deploy quantumly (SPEC §14r).

The plateau-mitigation arms (local costs, identity init, layerwise training,
support annealing, E13 CSA) all remain inside the Cerezo-Larocca dictum:
provably trainable losses live in polynomially-sized subspaces and are
classically simulable, so a "mitigated" QNN never offers a sampling
advantage. E14 *inverts* the protocol instead of patching it:

  - ALL optimisation is classical. The model is the IQP-style phase ansatz
    ``U(theta) = H^xn D(theta) H^xn`` with ``D = diag_x exp(i sum_g theta_g
    (-1)^{q_g(x)})`` (single-qubit ``Z`` and ``ZZ`` phase gates on a chain).
    For this family the computational-basis correlators are classically
    computable without building the state, ``<Z_a> = 2^{-n} sum_t cos(phi(t^a)
    - phi(t))`` (den-Nest / IQP classical-evaluability), and the spectral MMD
    ``sum_{|a|<=K} w_a (<Z_a>_q - <Z_a>_p)^2`` is a classical, differentiable
    Objective minimised by Adam with ZERO quantum gradient evaluations --
    no parameter shifts, no adjoint sweeps, no shots.
  - The quantum role is reserved for DEPLOYMENT: drawing bitstrings from the
    trained distribution and feeding them to the classical SQD/QSCI
    post-processor (E8).

Samplers compared on the critical TFIM at n = 8,10,12:
  exact_gs    -- samples from the true ground state (best possible support)
  ctqd        -- classically-MMD-trained phase ansatz, deployed by sampling
  hea_trained -- E8-style trained HEA (identity init + folded axes), the
                 gradient-mitigated route, for reference
  uniform_phi -- dense random-phase ansatz (complete-graph ZZ + ZZZ gates,
                 theta ~ U(-2pi,2pi)): a Haar/volume-law-like deployed state,
                 the expensive end of the frontier

Headline budget metric: per-step quantum gradient evaluations.

  hea : (2*M*n + 1) per Adam step        (E8 conventions, SPEC D2)
  ctqd: 0 -- training is pure classical

A classical-surrogate (MPS / Schmidt) frontier is reported for each deployed
distribution (SPEC D9/E9): half-chain entropy, significant Schmidt rank at the
mid cut, and exact MPS bond.

Honest boundary (also in SPEC §14r): at n <= 16 everything is classically
diagonalisable and the critical-TFIM target is area-law, so no sampling
*advantage* is claimed. The wins are (1) the framework: optimisation is
classical while the deployed sampling task is not; (2) a measured phase
diagram of how expensive the *deployed* distribution actually is; (3) an
explicit no-advantage verdict consistent with E8/E9.

Run:  python experiments/e14_ctqd.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from bpqnn.ctqd import (
    chain_zz_edges,
    classical_adam_train,
    correlator_set,
    dense_phase_signs,
    expectation_z_from_state,
    iqp_state,
    phase_signs,
    sampler_frontier,
    sample_from_theta,
    xor_table,
)
from bpqnn.hamiltonian import ising_ground_state, ising_hamiltonian
from bpqnn.init import identity_block_init
from bpqnn.sqd import (
    half_chain_entropy,
    qsci_indices,
    rel_energy_error,
    sqd_ground_energy,
)
from bpqnn.train import adam_train, lbfgs_train
from experiments.e2b_init_layerwise import axes_folded, run_state

N_LIST = [8, 10, 12]
N_FRONTIER_EXTRA = []
H = 1.0
M_OF = lambda n: 2 * n + 1  # noqa: E731
K = 3
TAU = 1.5
CTQD_SEEDS = 3
CTQD_STEPS = 200
CTQD_LR = 0.05
HEA_TRAIN_STEPS = 400
SAMPLES = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
TRIALS = 3
THRESHOLD = 1e-2
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def target_correlators(n, h, masks) -> np.ndarray:
    """``<Z_a>_{gs}`` for every mask, from the exact TFIM ground state."""
    _, gs, _ = ising_ground_state(n, h=h)
    return expectation_z_from_state(gs, n, masks)


def ctqd_trained(n, h, seed):
    """Classical MMD training of the phase ansatz. Returns theta + diagnostics."""
    G = n + len(chain_zz_edges(n))
    rng = np.random.default_rng(seed)
    signs, _ = phase_signs(n)
    cs = correlator_set(n, K, TAU)
    idx_ta = xor_table(cs["masks"], n)
    target_z = target_correlators(n, h, cs["masks"])

    runs = []
    for s in range(CTQD_SEEDS):
        theta0 = rng.uniform(-0.5, 0.5, size=G)
        res = classical_adam_train(
            theta0,
            signs=signs,
            idx_ta=idx_ta,
            weights=cs["weights"],
            target_z=target_z,
            lr=CTQD_LR,
            max_steps=CTQD_STEPS,
            rel_drop=1e-9,
        )
        runs.append(res)
    best = min(runs, key=lambda r: r["loss"])
    return {
        "theta": np.asarray(best["theta"]),
        "loss": best["loss"],
        "steps": best["steps"],
        "quantum_grad_evals": best["quantum_grad_evals"],
        "classical_correlator_sums": best["classical_correlator_sums"],
        "loss_history": [r["loss"] for r in best["history"]],
    }


def distributions(n, h, ctqd_theta):
    """Build the probability tables for every sampler at the given (n, h)."""
    dim = 1 << n
    E0, gs, w = ising_ground_state(n, h=h)
    p_gs = np.abs(gs.reshape(-1)) ** 2

    signs, _ = phase_signs(n)
    p_ctqd = np.abs(iqp_state(ctqd_theta, n, signs).reshape(-1)) ** 2

    rng = np.random.default_rng(1000 * n + int(10 * h))
    signs_dense, _ = dense_phase_signs(n, rng)
    theta_uniform = rng.uniform(-2.0 * np.pi, 2.0 * np.pi,
                                size=signs_dense.shape[0])
    p_uni = np.abs(iqp_state(theta_uniform, n, signs_dense).reshape(-1)) ** 2

    states = {"exact_gs": gs, "ctqd": iqp_state(ctqd_theta, n, signs),
              "uniform_phi": iqp_state(theta_uniform, n, signs_dense)}
    ps = {"exact_gs": p_gs, "ctqd": p_ctqd, "uniform_phi": p_uni}
    return {
        "E0": E0, "E1": float(w[1]),
        "n_target_masks": int(2 ** n),
        "states": states, "ps": ps,
    }


def sqd_curve_from_p(p, n, H, E0, E1, seeds):
    """Mean/std SQD rel-energy error vs sample budget over independent trials."""
    max_s = max(SAMPLES)
    mean, std = [], []
    for k in SAMPLES:
        errs = []
        for s in seeds:
            rng = np.random.default_rng(s)
            pool = rng.choice(p.size, size=max_s, p=p)[:k]
            idx = qsci_indices(np.unique(pool), n)
            if idx.size == (1 << n):
                errs.append(0.0)
            else:
                Emin = _lowest_energy(H, idx, E0, n)
                errs.append(rel_energy_error(Emin, E0, E1))
        mean.append(float(np.mean(errs)))
        std.append(float(np.std(errs)))
    return {"samples": list(SAMPLES), "mean": mean, "std": std}


def _lowest_energy(H, idx, E0, n):
    """Lowest eigenvalue of H restricted to idx (dense for small, eigsh else)."""
    if idx.size == (1 << n):
        return E0
    if idx.size <= 700:
        return sqd_ground_energy(H, idx)[0]
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh

    Hs = csr_matrix(H[np.ix_(idx, idx)])
    return float(eigsh(Hs, k=1, which="SA", return_eigenvectors=False, tol=0)[0])


def samples_to_threshold(samples, mean, thr=THRESHOLD):
    for k, m in zip(samples, mean):
        if m <= thr:
            return int(k)
    return None


def hea_trained_state(n, h):
    """E8-style trained HEA (identity init, folded axes, Adam + L-BFGS-B).

    ``budget_shift_evals`` is the D2 param-shift-equivalent budget spent in the
    Adam phase (each step charged ``1 + (2*M*n + 1)``); the L-BFGS-B phase adds
    its own ``nit`` gradient evaluations (reported by ``nit_bfgs``) on top.
    """
    edges = [(q, q + 1) for q in range(n - 1)]
    M = M_OF(n)
    axes = axes_folded(M)
    rng = np.random.default_rng(1000 * n + int(10 * h))
    theta0 = identity_block_init(M, n, rng)
    res_adam = adam_train(theta0, n=n, kind="ising", h=h, edges=edges,
                          axes=axes, lr=0.05, max_steps=HEA_TRAIN_STEPS)
    res_bfgs = lbfgs_train(res_adam["theta"], n=n, kind="ising", h=h,
                           edges=edges, axes=axes, maxiter=500)
    state = run_state(n, edges, axes, res_bfgs["theta"])
    return {
        "state": state,
        "energy": float(res_bfgs["cost"]),
        "budget_shift_evals": float(res_adam["budget"]),
        "budget_circuit_evals": float(res_adam["circuit_evals"]),
        "nit_bfgs": int(res_bfgs.get("nit", 0)),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    out = {
        "config": {
            "n": N_LIST,
            "h": H,
            "K": K,
            "tau": TAU,
            "ctqd_seeds": CTQD_SEEDS,
            "ctqd_steps": CTQD_STEPS,
            "ctqd_lr": CTQD_LR,
            "hea_train_steps": HEA_TRAIN_STEPS,
            "samples": SAMPLES,
            "trials": TRIALS,
            "threshold": THRESHOLD,
            "gates": "Z + ZZ chain (CTQD); complete-graph Z+ZZ+ZZZ (uniform_phi)",
        },
        "cases": {},
        "frontier": {},
    }
    print("=" * 78)
    print("E14 CTQD: train classically, deploy quantumly (SPEC §14r)")
    print("=" * 78)

    # ------------------------------------------------------------------ CTQD
    ctqd_info = {}
    for n in N_LIST:
        G = n + len(chain_zz_edges(n))
        signs, labels = phase_signs(n)
        cs = correlator_set(n, K, TAU)
        idx_ta = xor_table(cs["masks"], n)
        target_z = target_correlators(n, H, cs["masks"])
        t0 = time.time()
        info = ctqd_trained(n, H, seed=31000 + n)
        dt = time.time() - t0
        ctqd_info[n] = info
        print("-" * 78)
        print(f"n={n}: CTQD trained (classical, {G} gates, {info['steps']} steps) "
              f"in {dt:.1f}s; final MMD={info['loss']:.3e}; "
              f"quantum grad evals = {info['quantum_grad_evals']}")

    # ------------------------------------------------------------------ cases
    hea_cache = {}
    for n in N_LIST:
        Hn = ising_hamiltonian(n, h=H)
        info = ctqd_info[n]
        di = distributions(n, H, info["theta"])
        E0, E1 = di["E0"], di["E1"]

        # HEA comparator (E8 recipe) only where it is already affordable
        hea = None
        if n in (8, 10):
            hea = hea_trained_state(n, H)
            hea_cache[n] = hea

        seed_base = 81000 + 100 * n
        seeds = [seed_base + 13 * t for t in range(TRIALS)]

        sam_names = ["exact_gs", "ctqd", "uniform_phi"]
        if hea is not None:
            sam_names.append("hea_trained")

        case = {
            "E0": E0, "E1": E1,
            "ctqd_mmd": info["loss"],
            "ctqd_steps": info["steps"],
            "ctqd_classical_correlator_sums": info["classical_correlator_sums"],
            "ctqd_loss_history": info["loss_history"],
            "curves": {},
            "samples_to_threshold": {},
            "entropies": {k: half_chain_entropy(st, n)
                          for k, st in di["states"].items()},
        }
        for name in sam_names:
            if name == "hea_trained":
                p = np.abs(hea["state"].reshape(-1)) ** 2
            else:
                p = di["ps"][name]
            c = sqd_curve_from_p(p, n, Hn, E0, E1, seeds)
            case["curves"][name] = c
            thr = samples_to_threshold(c["samples"], c["mean"])
            case["samples_to_threshold"][name] = thr
            print(f"  n={n} {name:>12}: err@{c['samples'][-1]}="
                  f"{c['mean'][-1]:.2e}  samples to <={THRESHOLD:g}: {thr}")

        case["ctqd_budget_shift_evals"] = 0.0
        if hea is not None:
            case["hea_budget_shift_evals"] = hea["budget_shift_evals"]
            case["hea_budget_circuit_evals"] = hea["budget_circuit_evals"]
            case["hea_nit_bfgs"] = hea["nit_bfgs"]
            case["hea_rel_err"] = rel_energy_error(
                hea["energy"], E0, E1)
            print(f"      hea_trained final rel-err = "
                  f"{case['hea_rel_err']:.2e}")
        out["cases"][f"n{n}"] = case

    # ---------------------------------------------------------------- frontier
    print("-" * 78)
    print("Sampling-surrogate frontier (SPEC D9/E9): entropy, Schmidt rank, bond")
    for n in N_LIST + N_FRONTIER_EXTRA:
        row = {}
        if n in N_LIST:
            info = ctqd_info[n]
            signs, _ = phase_signs(n)
            row["ctqd"] = sampler_frontier(iqp_state(info["theta"], n, signs), n)
        E0, gs, _ = ising_ground_state(n, h=H)
        row["exact_gs"] = sampler_frontier(gs, n)
        rng = np.random.default_rng(7 * n)
        signs_dense, _ = dense_phase_signs(n, rng)
        th_uniform = rng.uniform(-2.0 * np.pi, 2.0 * np.pi,
                                 size=signs_dense.shape[0])
        row["uniform_phi"] = sampler_frontier(
            iqp_state(th_uniform, n, signs_dense), n)
        out["frontier"][f"n{n}"] = row
        for label, d in row.items():
            print(f"  n={n} {label:>10}: S(n/2)={d['entropy_half']:.3f}  "
                  f"sig.rank={d['significant_rank_mid']}  bond={d['exact_bond']}")

    with open(RESULTS / "e14_ctqd.json", "w") as f:
        json.dump(out, f, indent=2)
    print("-" * 78)
    print(f"saved {RESULTS / 'e14_ctqd.json'}")

    try:
        _plot(out)
        print(f"saved {RESULTS / 'e14_ctqd.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(plot skipped: {e})")
    return 0


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = out["config"]["n"]
    sam = out["config"]["samples"]
    colors = {"exact_gs": "tab:green", "ctqd": "tab:blue",
              "hea_trained": "tab:orange", "uniform_phi": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))

    # (a) CTQD MMD convergence (best classical run per n)
    ax = axes[0, 0]
    for n in ns:
        case = out["cases"][f"n{n}"]
        hist = case["ctqd_loss_history"]
        ax.semilogy(np.arange(1, len(hist) + 1), np.maximum(hist, 1e-16),
                    label=f"n={n}, final {case['ctqd_mmd']:.1e}")
    ax.set_xlabel("classical Adam step")
    ax.set_ylabel("spectral MMD (K={})".format(out["config"]["K"]))
    ax.set_title("CTQD classical training (0 quantum grad evals)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # (b),(c) SQD curves
    for (ax, n) in ((axes[0, 1], 10), (axes[0, 2], 12)):
        case = out["cases"][f"n{n}"]
        for name in ("exact_gs", "ctqd", "uniform_phi"):
            if name not in case["curves"]:
                continue
            ax.loglog(sam, np.maximum(case["curves"][name]["mean"], 1e-16),
                      "o-", color=colors[name], label=name)
        if "hea_trained" in case["curves"]:
            ax.loglog(sam, np.maximum(case["curves"]["hea_trained"]["mean"], 1e-16),
                      "o-", color=colors["hea_trained"], label="hea_trained")
        ax.axhline(out["config"]["threshold"], color="gray", ls=":", label="1e-2")
        ax.set_xlabel("samples"); ax.set_ylabel("SQD rel. energy error")
        ax.set_title(f"SQD curve, n={n}")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)

    # (d) entropy frontier
    ax = axes[1, 0]
    fn = out["frontier"]
    for label in ("exact_gs", "ctqd", "uniform_phi"):
        xs = sorted(fn, key=lambda k: int(k[1:]))
        ys = [fn[k][label]["entropy_half"] for k in xs if label in fn[k]]
        xv = [int(k[1:]) for k in xs if label in fn[k]]
        ax.plot(xv, ys, "o-", color=colors[label], label=label)
    ax.set_xlabel("n"); ax.set_ylabel("half-chain entropy (base 2)")
    ax.set_title("sampling-surrogate frontier: entanglement")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (e) samples-to-threshold grouped bars
    ax = axes[1, 1]
    cats = ["exact_gs", "ctqd", "hea_trained", "uniform_phi"]
    x = np.arange(len(ns)); width = 0.2
    for i, cat in enumerate(cats):
        vals = []
        for n in ns:
            c = out["cases"][f"n{n}"]
            v = c["samples_to_threshold"].get(cat)
            vals.append(v if v is not None else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=cat, color=colors.get(cat))
    ax.set_xticks(x); ax.set_xticklabels([f"n={n}" for n in ns])
    ax.set_ylabel("samples to reach 1e-2")
    ax.set_title("sample efficiency (SQD threshold crossing)")
    ax.legend(fontsize=7)

    # (f) training budget: quantum gradient evaluations (param-shift units)
    ax = axes[1, 2]
    hea_h8 = out["cases"]["n8"].get("hea_budget_shift_evals")
    labels = []
    vals = []
    for n in ns:
        c = out["cases"][f"n{n}"]
        if "hea_budget_shift_evals" in c:
            labels.append(f"n={n} HEA")
            vals.append(c["hea_budget_shift_evals"])
        labels.append(f"n={n} CTQD")
        vals.append(c.get("ctqd_budget_shift_evals", 0.0))
    ax.bar(np.arange(len(vals)), vals, color=["tab:orange", "tab:blue"])
    ax.set_yscale("log"); ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_ylabel("quantum gradient evals (shift units)")
    ax.set_title("training budget")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS / "e14_ctqd.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())