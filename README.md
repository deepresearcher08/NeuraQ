# NeuraQ

**Team:** NeuraQ

**Project title:** NeuraQ — measuring and mitigating barren plateaus in
variational quantum neural networks, validated on IBM Quantum hardware.
**Track:** QML, with cross-track extras (QEC decoders, SQD/QSCI, tensor-network surrogates).

---

## Summary

Deep, randomly initialised quantum neural networks have gradients that decay
**exponentially** with the number of qubits, so training stalls — the *barren
plateau*. NeuraQ measures the plateau directly (gradient variance vs. width and
depth, averaged over many random initialisations with bootstrap confidence
intervals), then evaluates the standard mitigations on equal footing: a **local
cost function**, **identity-block / perturbed-identity initialisation**,
**layerwise training**, and **support scheduling**. The headline result is a
measured, error-barred comparison of cost locality that holds across an exact
NumPy simulator, an Aer device-noise model, and one real IBM Quantum run on
`ibm_fez`. A business-facing wrap (a pre-flight *trainability screen*) is built
on the same measurement; see [`PITCH.md`](PITCH.md). NeuraQ **characterises
trainability — it does not claim quantum advantage.**

![Main result](figures/main_result.png)

*Main result (E15 / `figures/main_result.png`): on real hardware the global-cost
gradient collapses with width while the local-cost gradient survives.*

---

## Environment

- **Python 3.14.7** (tested on Windows 11; standard CPython, no GPU).
- Install everything exactly as pinned:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

- **No IBM account or token is required** for the main result or for the whole
  test suite. The token is needed only to *submit* a new hardware job. It is
  read from a file (`ibmapi.txt` + `ibmcrn.txt`) or from environment variables
  and is **never committed** (see `.gitignore`). A complete simulator-only path
  exists for every experiment.

---

## Exact commands and expected runtime

Main result (token-free, simulator only):

```bash
python experiments/e2a_cost_locality.py            # ~1–3 min
```

Rebuild the hero hardware figure from the stored QPU results (token-free):

```bash
python experiments/e15_hardware.py --replot-from results/e15_qpu.json --tag e15_qpu
```

Full test suite (checks every module against an exact/classical reference):

```bash
pytest -q                                           # ~2–3 min
```

Optional — submit a *new* hardware job (needs your own IBM token):

```bash
python experiments/e15_hardware.py --backend least_busy --shots 512 --r 48 --ns 2 4 6 8
```

Each experiment writes `results/eN_*.json` (data) and `results/eN_*.png`
(figure) and prints a summary. Every figure used in the write-up is also copied
into `figures/` (see [`WRITEUP.pdf`](WRITEUP.pdf)).

---

## Statistics and seeds

- Statistical results average over many **random initialisations**: `R = 512`
  inits per width for E2a/E11, `R = 48` for the hardware run (E15), and are
  reported with **bootstrap 95% confidence intervals** (variance ranges and
  log-linear slopes).
- **All seeds are fixed** in the scripts (e.g. `seed = 2026` for E15); reruns
  are reproducible. Experiment scripts that are cheap are fully deterministic;
  the statistical ones are deterministic given their seeds and are labelled
  with the number of runs.
- The main figure is rebuilt from the committed `results/e15_qpu.json`, so no
  network access is needed to reproduce it.

---

## The hardware run

- **Backend:** IBM Quantum `ibm_fez` (156-qubit Eagle).
- **Date:** 2026-10-13 (jobs created 23:05–23:13 IST).
- **Jobs (all `DONE`):** `db76ns2raffc73f9s62g`, `db76pn8r1c5c738js3hg`,
  `db76qfqraffc73f9sb60`, `db76rl6r6oqs7386spt0` (full list in
  [`results/e15_qpu_jobs.json`](results/e15_qpu_jobs.json)).
- **Settings:** `n ∈ {2,4,6,8}`, depth `M = 4`, `R = 48` random inits,
  `512` shots per circuit, single-axis all-RY HEA, target `|0…0>`, middle-layer
  parameter-shift gradient.
- The **noise model** used for the simulator comparison is the **FakeBrisbane**
  device model, not `ibm_fez` specifically — this is a limitation, stated in the
  write-up.

---

## What's inside

### Library — `bpqnn/`

| Module | Purpose |
|---|---|
| `simulator.py` | batched NumPy statevector simulator (single-axis layered ansatz, CZ chain) |
| `ansatz.py` | parameter shapes, entangling topologies, Qiskit export |
| `costs.py` | global / local costs, infidelity, the `\|0…0>` plateau protocol |
| `grad.py` | adjoint (reverse-mode) and parameter-shift gradients + finite-difference check |
| `init.py` | random, identity-block, and perturbed-identity initialisations |
| `observables.py` | tunable-support (block / window) readout family |
| `hamiltonian.py` | transverse-field Ising energy / ground state (dense) |
| `train.py` | Adam, layerwise Adam, and an L-BFGS-B oracle with shot accounting |
| `shots.py` / `noise.py` | shot-noise estimators / batched density-matrix simulator |
| `mps.py` / `sqd.py` | statevector→MPS, Schmidt spectrum / SQD-QSCI |
| `classify.py` | small QNN binary classifier (additive + parity tasks, held-out split) |
| `csa.py` | Cost-Support Annealing and the λ-mix local→global blend |
| `hardware.py` | circuit + counts + parameter-shift gradients for a real QPU (E15) |
| `screen.py` | pre-flight trainability screening (verdict + recommended fix) |
| `reference.py` | analytic reference states |

### Experiments — `experiments/e*.py`

| Script | Study |
|---|---|
| `e1_plateau.py` | gradient-variance barren-plateau sweep |
| `e2a_cost_locality.py` | global vs. local cost (the core mitigation) |
| `e2b_init_layerwise.py` | identity-block init and layerwise training |
| `e3_training.py` / `e4_shot_training.py` | equal-budget training; shot noise |
| `e6_classify.py` | classification QNN over cost × init arms (held-out test) |
| `e7_nibp.py` | noise-induced barren plateaus |
| `e8_sqd.py` / `e9_mps.py` | SQD/QSCI sample-efficiency; MPS surrogate |
| `e10_decoder.py` / `e12a` / `e12b` | repetition / Steane / surface-code decoders |
| `e11_support_radius.py` | gradient variance vs. observable support radius |
| `e13_csa.py` | support scheduling (Cost-Support Annealing) |
| `e14_ctqd.py` | train classically, deploy quantumly |
| `e15_hardware.py` | real-hardware plateau: global vs. local variance on IBM Quantum |
| `e16_screen.py` | trainability-screening CLI (verdict + mitigation report) |
| `e17_parity_task.py` | expressivity: a parity task the local readout cannot represent |
| `e18_screen_validation.py` | does the screen predict shot-limited training outcomes? |
| `e19_lambda_mix.py` | bounded local→global cost blend; normalisation bound |
| `e20_perturbed_init.py` | perturbed identity re-injects edge-layer gradient signal |

---

## Honesty, negative results, and limits

- **What is simulated.** Exact noiseless NumPy statevector; Qiskit Aer
  noiseless; Aer with a **FakeBrisbane** device noise model; and one real IBM
  QPU run at small scale (`n ≤ 8`).
- **No advantage claim.** NeuraQ measures trainability. It does not claim
  quantum advantage, speed-up, or a better classifier.
- **Classical baselines.** Energy tasks are compared against exact
  diagonalisation and an L-BFGS-B oracle; the classifier (E6/E17) is scored on a
  **held-out** grid. A matched-parameter-count classical MLP baseline is on the
  roadmap (see the write-up's limits section).
- **Negative / weak results reported as they are:**
  - Cost-Support Annealing (E13) is **partial**: the linear schedule helps; the
    staged and variance-gated variants do not beat the plain global cost.
  - At `n = 8` the global-cost arm **can still train** — the plateau *slows*
    training here rather than blocking it (E3).
  - The trainability screen (E16) is validated only weakly: under
    Adam-normalised shot training the "barren" verdicts were **false negatives**
    (every flagged configuration still reached a low cost). The screen is a
    gradient-*variance* alarm, not an optimiser-failure predictor at these
    widths (E18). Its thresholds were calibrated on the same sweeps it is
    judged against.
  - The local-cost slope (≈ −0.55 per qubit) is **still exponential** over
    `n = 2…8`; the claim is *relative* (global decays much faster), not
    asymptotic.
  - The hardware gap is **at least as large as simulated, with overlapping
    CIs** — not "wider on hardware". The E15 hardware variances include shot
    noise.
- **AI use.** See [`AI_USE_NOTE.md`](AI_USE_NOTE.md).

---

## References (and what NeuraQ adds)

- McClean et al., *Nat. Commun.* **9**, 4812 (2018) — barren plateaus.
- Cerezo et al., *Nat. Commun.* **12**, 1791 (2021) — cost-function-dependent
  plateaus; the bound `C_L ≤ C_G ≤ n·C_L` used in E19.
- Grant et al., *Quantum* **3**, 214 (2019) — identity-block initialisation.
- Skolik et al., arXiv:2006.14904 — layerwise learning.
- Wang et al., *Nat. Commun.* **12**, 6961 (2021) — noise-induced plateaus.
- Cerezo et al., *Nat. Commun.* **14**, 5742 (2023) — simulability.
- Jones & Gacon, arXiv:2011.02991 — adjoint differentiation.
- Huggins et al., *J. Chem. Theory Comput.* **18**, 7001 (2022) — QSCI/SQD.
- IBM Quantum SQD / Qiskit Fall Fest SQD notebook — used for the QEC/SQD extras.

**What NeuraQ adds (vs. a replication).** The mechanism results (McClean,
Cerezo, Grant, Skolik, Wang) are reproduced and extended with: (i) a
measured, bootstrap-CI comparison of cost locality **on real hardware** (E15);
(ii) a **support-radius law** relating noise-tolerance to observable support
(E11); (iii) the **λ-mix** local→global schedule and its normalisation bound
(E19); (iv) a **perturbed-identity** initialisation that restores edge-layer
signal (E20); and (v) a **trainability-screen** interface (E16/E18) with its
honest limits. Everything is measured, nothing is claimed beyond the data.
