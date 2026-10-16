"""bpqnn: barren-plateau-resistant QNN toolkit.

Batched NumPy statevector simulator plus the pieces needed to characterise
barren plateaus and mitigations (see SPEC.md).
"""

from .simulator import StatevectorSimulator
from . import ansatz, costs, grad, init, hamiltonian, reference, train

__all__ = [
    "StatevectorSimulator",
    "ansatz",
    "costs",
    "grad",
    "init",
    "hamiltonian",
    "reference",
    "train",
]
