# NeuraQ — Pitch

*A research result with a product wrapped around it.*

---

## Slide 1 — The one-liner

**Enterprises are about to spend a lot of QPU budget training quantum neural
networks that cannot train.**

NeuraQ predicts *before you train* whether a QNN will learn — and tells you how
to fix it. Under the hood it is a hardware-validated study of barren plateaus.

---

## Slide 2 — The problem (research)

Deep, randomly initialised quantum circuits have gradients that decay
**exponentially with the number of qubits**. This is the *barren plateau*: the
loss surface is flat, so training stalls.

- It is not just "small gradients". At a plateau a fixed number of gradient
  measurements carries an exponentially vanishing amount of information — more
  shots buy you nothing.
- It hits exactly the models people want to train: deep, expressive,
  global-cost variational circuits.

---

## Slide 3 — The lever (research)

A **cost function's locality** decides the plateau.

- A **global** cost (fidelity to the whole register) plateaus at *any* depth.
- A **local** cost (sum of single-qubit terms) stays trainable at depth
  `O(log n)`, and only plateaus at `O(poly n)`.

NeuraQ measures this directly — gradient variance vs. width and depth, over
many random inits — and then stress-tests the standard mitigations on equal
footing: local costs, identity-block init, layerwise training, structured
ansatze.

---

## Slide 4 — The result (the money slide)

We ran the protocol on **real IBM Quantum hardware** (156-qubit `ibm_fez`,
`n = 2…8`, ~1M shots).

| quantity | simulated (noiseless) | real QPU |
|---|---|---|
| global/local variance gap at n=8 | 24× | **35×** |
| global-cost decay (log Var / qubit) | −1.13 | **−1.20** |
| local-cost decay | −0.56 | **−0.55** |

**The gap is *wider* on hardware than in simulation.** Noise shrinks the signal
but not the decay slopes, and it punishes the global cost roughly twice as hard
as the local one. The mitigation matters *more* on today's noisy devices, not
less.

---

## Slide 5 — Why it matters (business turn)

Every quantum-ML pilot faces the same hidden cost: it is easy to pick a
circuit, a cost, and an optimiser that will never train — and you only find out
after burning expensive QPU hours.

**Trainability is a pre-flight check.** It belongs in the design loop, before
training, before deployment, before the QPU bill.

---

## Slide 6 — The product: NeuraQ Screen

A dependency-light CLI (today) / API (next) that answers one question:

> *Given this width, depth, cost and init — will this QNN train?*

It returns a verdict and a concrete fix.

```
$ python experiments/e16_screen.py --n 8 --cost global

NeuraQ trainability screen
  configuration  : n=8  depth M=5  cost=global  init=random  topology=line
  gradient var   : 3.318e-05   (mean |dC/dtheta| = 1.726e-03)
  decay slope    : -1.17 / qubit  (n=8 -> n=10)
  verdict        : BARREN PLATEAU
  mitigation     : local cost gives 9.987e-04  (30.1x more signal)
  recommendation : switch to a local cost C_L
```

No shots, no hardware, no optimiser — seconds, not QPU hours.

---

## Slide 7 — Who pays

- **Quantum-ML teams** validating an ansatz before committing hardware.
- **Cloud / hardware providers** (IBM Quantum, AWS Braket, IonQ…) wanting fewer
  wasted jobs and better first-run success.
- **Consultancies & integrators** needing a standard, defensible pre-flight
  step they can put their name on.
- **Vertical pilots** — finance (risk/portfolio), chemistry, logistics — where
  the ansatz choice is the make-or-break decision.

---

## Slide 8 — The value

- **Saves QPU time** — avoid training runs that provably go nowhere.
- **De-risks pilots** — a screening report is an artifact you can show a
  stakeholder before the spend.
- **Backend-agnostic** — the diagnostic is about the *model*, not the vendor.
- **Hardware-validated** — the recommendation is backed by real-device data
  (E15), not just simulation.

---

## Slide 9 — Scalability roadmap

1. **Now** — CLI screening on the NumPy simulator (this repo).
2. **Next** — hosted API + dashboard; plug into cloud quantum jobs as a
   pre-flight gate; support arbitrary ansatz/cost/init configs.
3. **Then** — domain packs (finance / chemistry / logistics) with tuned
   defaults and reporting.
4. **Beyond** — tensor-network surrogates (MPS groundwork already in E9) to push
   the screen past the `n ≈ 16` statevector limit and onto larger circuits.

Each stage reuses the same core measurement; only the interface and the scope
grow.

---

## Slide 10 — Why us / the moat

- **Hardware-validated principle**: we did the QPU work (E15), not just the
  theory — the global/local gap on real silicon is *measured*.
- **A novel quantitative law**: the E11 support-radius law of gradient variance
  is our own contribution, not a reproduction.
- **Reproducible & light**: the whole toolkit runs on batched NumPy; Qiskit is
  only needed for the hardware arm.
- **Honest framing**: we state the boundaries (no quantum-advantage claim;
  characterisation, not hype), which is exactly what makes a screening tool
  trustworthy.

---

## Slide 11 — The ask

- Pair NeuraQ Screen with a real pilot workload and show the QPU-hours saved.
- Extend the screen to a backend-agnostic API and a hosted dashboard.
- Publish the E11 law + E15 hardware result as the credibility anchor.

**Research that measures the plateau. Product that avoids it.**
