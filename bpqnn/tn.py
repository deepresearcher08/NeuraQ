"""Optional tensor-network utilities (quimb/itensor fallback)."""

from __future__ import annotations

HAS_QUIMB = False
try:
    import quimb  # type: ignore
    import quimb.tensor as qtn  # type: ignore
    HAS_QUIMB = True
except Exception:
    HAS_QUIMB = False


def supports():
    return {"quimb": HAS_QUIMB}


def mps_from_vector(psi, n, chi=None, cutoff=None):
    if not HAS_QUIMB:
        from .mps import statevector_to_mps, mps_to_statevector, compress
        return compress(psi, n, chi=chi)
    p = qtn.MPS_from_dense(psi.reshape([2] * n), cutoff=cutoff, max_bond=chi)
    return p.to_dense().reshape(-1)


def dmrg_ground(n, h=1.0, chi=16, max_sweeps=10):
    if not HAS_QUIMB:
        raise ImportError("quimb not installed")
    H = qtn.MPO_ham_ising(n, j=-1.0, h=h, cyclic=False)
    psi = qtn.MPS_rand_state(n, 2, chi=chi)
    dmrg = qtn.DMRG2(H, psi, max_sweeps=max_sweeps, bond_dims=[chi])
    E = dmrg.energy
    return {"E": float(E), "chi": int(chi)}
