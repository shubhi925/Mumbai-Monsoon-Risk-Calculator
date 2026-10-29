"""
pytest suite for the simulation engine.

Run with:  pytest -q
These are the same checks the app exposes under Validation > Self-tests, plus a
regression test pinning the engine to the figures quoted in the validation report.
"""
import math
import numpy as np
import pytest

import engine as E


def fixed(spec, rate=0.5):
    acts = [E.Activity(id=f"X{i}", dur=d, pi=list(pre), dO=dl, dM=dl, dP=dl,
                       cO=c, cM=c, cP=c) for i, (d, pre, dl, c) in enumerate(spec)]
    return acts, E.Config(rho=0.0, rate=rate, n=50, seed=1)


@pytest.mark.parametrize("spec,order,exp_d,exp_c", [
    ([(10, [], 0, 0), (10, [0], 0, 0)], [0, 1], 0, 0),
    ([(10, [], 3, 1), (10, [0], 4, 2)], [0, 1], 7, 6.5),
    ([(10, [], 0, 0), (5, [], 3, 0), (5, [0, 1], 0, 0)], [0, 1, 2], 0, 0),
    ([(10, [], 0, 0), (5, [], 8, 0), (5, [0, 1], 0, 0)], [0, 1, 2], 3, 1.5),
])
def test_deterministic_cases(spec, order, exp_d, exp_c):
    acts, cfg = fixed(spec)
    r = E.simulate(acts, order, cfg, n=50, seed=1)
    assert np.allclose(r.delay, exp_d)
    assert np.allclose(r.extra, exp_c)


def test_pert_mean_identity():
    """a + b = 6 by construction, so the beta mean is exactly (O+4M+P)/6."""
    for O, M, P in [(0, 2, 7), (2, 5, 12), (0, 1, 3), (1.5, 6, 20)]:
        a, b = E._shapes(O, M, P)
        assert a + b == pytest.approx(6.0)
        assert O + (P - O) * a / (a + b) == pytest.approx(E.pert_mean(O, M, P))


def test_sampled_mean_matches_pert():
    acts = [E.Activity(id="T", dur=10, dO=0, dM=2, dP=7)]
    r = E.simulate(acts, [0], E.Config(rho=0, n=60000, seed=5))
    got = float(r.act_delay[0].mean())
    se = float(r.act_delay[0].std(ddof=1) / math.sqrt(60000))
    assert abs(got - E.pert_mean(0, 2, 7)) <= 3 * se


def test_exact_beta_sd_beats_classical():
    acts = [E.Activity(id="T", dur=10, dO=0, dM=2, dP=7)]
    r = E.simulate(acts, [0], E.Config(rho=0, n=60000, seed=5))
    sample = float(r.act_delay[0].std(ddof=1))
    assert abs(sample - E.pert_sd_beta(0, 2, 7)) < abs(sample - E.pert_sd_classical(0, 2, 7))


def test_pairwise_correlation_is_rho_squared():
    acts = [E.Activity(id="P", dur=10, dO=0, dM=2, dP=7),
            E.Activity(id="Q", dur=10, dO=0, dM=2, dP=7)]
    r = E.simulate(acts, [0, 1], E.Config(rho=0.5, n=60000, seed=11))
    obs = float(np.corrcoef(r.act_delay[0], r.act_delay[1])[0, 1])
    assert obs == pytest.approx(0.25, abs=0.03)


def test_legacy_quantile_matches_exact():
    """The 2000-bin table should agree with the exact beta quantile to ~1e-7."""
    pytest.importorskip("scipy")
    q_tab = E.beta_quantile_table(0, 2, 7)
    q_exa = E.beta_quantile_exact(0, 2, 7)
    u = np.linspace(0.001, 0.999, 999)
    assert np.max(np.abs(q_tab(u) - q_exa(u))) < 5e-3
    assert abs(q_tab(u).mean() - q_exa(u).mean()) < 1e-5


def test_cycle_detected():
    acts = [E.Activity(id="A", dur=1, pre=["B"]), E.Activity(id="B", dur=1, pre=["A"])]
    errors, _, _ = E.validate(acts, E.Config())
    assert any("loop" in e.lower() for e in errors)


def test_omp_ordering_enforced():
    acts = [E.Activity(id="A", dur=1, dO=5, dM=2, dP=1)]
    errors, _, _ = E.validate(acts, E.Config())
    assert any("O <= M <= P" in e for e in errors)


def _benchmark():
    import csv
    rows = list(csv.DictReader(open("sample_activities.csv")))
    acts = [E.Activity(id=r["id"], name=r["name"], dur=float(r["duration"]),
                       cost=float(r["cost"]),
                       pre=[p for p in r["predecessors"].split(";") if p],
                       dO=float(r["delay_O"]), dM=float(r["delay_M"]), dP=float(r["delay_P"]),
                       cO=float(r["cost_O"]), cM=float(r["cost_M"]), cP=float(r["cost_P"]))
            for r in rows]
    cfg = E.Config(rate=0.6561, rho=0.5, n=10000, seed=42, budget=900, buf=30)
    _, _, order = E.validate(acts, cfg)
    return acts, order, cfg


def test_benchmark_regression():
    """Pins the legacy engine to the figures quoted in the validation report."""
    acts, order, cfg = _benchmark()
    r = E.simulate(acts, order, cfg)
    s = E.summarise(r)
    assert r.base == pytest.approx(402.0)
    assert cfg.base == pytest.approx(858.0)
    assert s["mean_delay"] == pytest.approx(41.098, abs=0.01)
    assert s["delay_p"][.8] == pytest.approx(50.41, abs=0.05)
    assert s["mean_cost"] == pytest.approx(62.795, abs=0.02)


def test_merge_bias_is_detected_and_explains_the_gap():
    """E[max] - max[E] should account for the manual-to-tool difference."""
    acts, order, cfg = _benchmark()
    r = E.simulate(acts, order, cfg)
    s = E.summarise(r)
    _, cp, _ = E.floats_and_critical_path(acts, order)
    h = E.hand_pert_project(acts, cp, cfg.rho, cfg.rate)
    md = E.merge_diagnostics(r)
    assert md["merge_id"] == "A11"
    assert md["jensen"] > 0
    gap = s["mean_delay"] - h["mean_delay"]
    # the Jensen gap explains the difference to within a fifth of a standard error
    assert abs(gap - md["jensen"]) < 0.2 * s["se_delay"] + 0.05


def test_corrected_sd_formula_tracks_simulation():
    acts, order, cfg = _benchmark()
    _, cp, _ = E.floats_and_critical_path(acts, order)
    for rho in (0.0, 0.3, 0.5, 0.7, 0.9):
        cfg.rho = rho
        r = E.simulate(acts, order, cfg)
        h = E.hand_pert_project(acts, cp, rho, cfg.rate)
        sim_sd = float(r.delay.std(ddof=1))
        assert abs(h["sd_corr"] - sim_sd) / sim_sd < 0.01, f"rho={rho}"
