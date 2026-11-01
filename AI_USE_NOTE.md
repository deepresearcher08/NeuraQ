# AI Assistance Note

- **Tools used:** GitHub Copilot and LLM coding assistants (used for syntax completion and test scaffolding).
- AI helped set up the framework and experiments; we verified it with three independent gradient checks and 127 tests.
- **Identified error:** During gradient derivation for the binary classifier, an initial suggestion yielded an inverted sign for the mean-squared-error parameter-shift chain rule (`-2(y - p)` rather than accounting for the outer loss derivative relative to expectation values). This flipped the initial optimization trajectory and made local cost appear inferior.
- **Verification check:** We caught this bug by writing a numerical finite-difference sanity check (`tests/test_e6.py`), which showed an exact sign reversal with a magnitude discrepancy of only `~2e-10`. Once corrected, local cost optimization converged properly.
