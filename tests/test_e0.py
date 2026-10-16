from experiments.e0_sanity import (
    check_adjoint,
    check_gradients,
    check_physics,
    check_statevector,
)


def test_statevector_matches_qiskit():
    assert check_statevector(n_max=5, M_max=3)


def test_parameter_shift_matches_finite_difference():
    assert check_gradients(n_max=4, M_max=2)


def test_adjoint_matches_parameter_shift():
    assert check_adjoint(n_max=4, M_max=3)


def test_physics_invariants():
    assert check_physics(n_max=5)
