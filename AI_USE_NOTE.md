# AI-use note

Coding agents were used for scaffolding, boilerplate, docstrings, running the test suite and helping 
with the mathematical framework.

**One bug was flagged and fixed.** The classifier's MSE gradient was
written with the wrong sign, `-2(y - p)`. With the angle-encoded cost and the
parameter-shift estimate, the correct gradient is
`mean(-2(y - base) * 0.5 * (C(theta + pi/2) - C(theta - pi/2)))`. The sign error
had flipped an earlier "local cost worse" conclusion. It was caught by comparing
the analytic gradient against finite differences (agreement ~2e-10), where the
sign was the only discrepancy.
