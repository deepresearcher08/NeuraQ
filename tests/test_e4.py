import json
import pathlib

import pytest

_RESULT = json.load(open(
    pathlib.Path(__file__).resolve().parents[1]
    / "results" / "e4_shot_training.json"))


def _snr(row, kind):
    return row["costs"][kind]["snr"]


def _by_n(cal, n, s):
    return next(r for r in cal if r["n"] == n and r["s"] == s)


@pytest.mark.skipif(not _RESULT.get("calibration"),
                    reason="calibration section missing")
def test_e4_calibration_local_snr_beats_global():
    """Local energy estimator has higher SNR than global infidelity at every
    calibrated row; both fall with n at fixed shots and rise with shots."""
    cal = _RESULT["calibration"]
    for row in cal:
        assert _snr(row, "ising") > _snr(row, "infidelity")
    assert _snr(_by_n(cal, 6, 4096), "ising") > _snr(_by_n(cal, 8, 4096), "ising")
    assert _snr(_by_n(cal, 8, 4096), "ising") > _snr(_by_n(cal, 10, 4096), "ising")
    assert _snr(_by_n(cal, 8, 32768), "ising") > _snr(_by_n(cal, 8, 4096), "ising")


@pytest.mark.skipif(not _RESULT.get("training_nomom"),
                    reason="no-momentum run missing")
def test_e4_local_256_matches_exact_and_beats_global():
    """n=8, s=256, beta1=0: noisy local ties exact local and beats global at
    8x the shots (the ~8x shot-efficiency win)."""
    arms = _RESULT["training_nomom"]["arms"]
    local = arms["local_256"]["summary"]["end_median"]
    local_exact = arms["local_exact"]["summary"]["end_median"]
    global_exact = arms["global_exact"]["summary"]["end_median"]
    global_2048 = arms["global_2048"]["summary"]["end_median"]
    assert local < global_2048 < global_exact
    assert local < 1.5 * local_exact


@pytest.mark.skipif(not _RESULT.get("training_n10"),
                    reason="n=10 run missing")
def test_e4_momentum_local_beats_global_equal_shots():
    """n=10, s=512, beta1=0.9: noisy local beats noisy global at equal shots."""
    arms = _RESULT["training_n10"]["arms"]
    assert (arms["local_512"]["summary"]["end_median"]
            < arms["global_512"]["summary"]["end_median"])
