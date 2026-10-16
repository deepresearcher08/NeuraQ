from bpqnn import screen

def test_predictive_reduces_false_negatives():
    r = screen.screen(n=6, M=3, cost="global", N=64, predictive=True)
    assert r["verdict"] in {"trainable", "marginal", "barren plateau"}
    assert 0.0 <= r["p_success"] <= 1.0
    assert r["remediation"] is not None


def test_predictive_removes_known_e18_false_negatives():
    # The three configs E18 flagged as barren-but-actually-trained.
    for n, M in [(6, 3), (8, 3), (10, 6)]:
        r = screen.screen(n=n, M=M, cost="global", N=128, predictive=True)
        assert r["verdict"] != "barren plateau", (n, M, r["verdict"])
        assert r["p_success"] >= 0.45, (n, M, r["p_success"])



def test_auto_patch_format():
    r = screen.screen(n=8, M=3, cost="global", N=64, predictive=True)
    rem = r["remediation"]
    assert rem["suggested_config"]["cost"] == "local"
