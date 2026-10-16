# AI-use note

Coding agents (the opencode CLI) gave **mild assistance** — scaffolding
modules, boilerplate, docstrings, and running the test suite. The ideas,
experiments and analysis are the team's, and every result was verified by
running the code in this repository.

**One bug the team flagged and had fixed.** The classifier's MSE gradient was
written with the wrong sign, `-2(y - p)`. With the angle-encoded cost and the
parameter-shift estimate, the correct gradient is
`mean(-2(y - base) * 0.5 * (C(theta + pi/2) - C(theta - pi/2)))`. The sign error
had flipped an earlier "local cost worse" conclusion. It was caught by comparing
the analytic gradient against finite differences (agreement ~2e-10), where the
sign was the only discrepancy.
