"""Utilities for memory/formatting."""

from __future__ import annotations

import numpy as np


def estimate_mem(n, M, batch, dtype=np.complex128):
    bytes_per_elem = np.dtype(dtype).itemsize
    state = batch * (1 << n) * bytes_per_elem
    return {"state_bytes": state, "state_mb": state / (1024 ** 2)}
