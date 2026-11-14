"""Density checks against the synthetic models."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from synthetic import WASTE, make_model

from qaqc.stats import weighted_mean, weighted_quantile


def _used(m):
    t = m.truth
    return t[(t["selected"] & ~t["is_air"]).to_numpy()]


@pytest.mark.parametrize("kind", ["regular", "sub_blocked", "percentage"])
def test_domain_table(run_checks, kind):
    m = make_model(kind)
    ctx = run_checks(m)
    t = _used(m)
    rows = {r["domain"]: r for r in ctx.bundle["density"]["by_domain"]}
    assert set(rows) == set(t["domain"])
    for dom, r in rows.items():
        d = t[t["domain"] == dom]
        sg = d["sg"].astype(np.float32).astype(np.float64)
        assert r["blocks"] == len(d)
        assert r["missing"] == int(sg.isna().sum())
        assert r["mean_volume_weighted"] == pytest.approx(weighted_mean(sg, d["vol"]), rel=1e-9)
        assert r["p50"] == pytest.approx(weighted_quantile(sg, d["vol"], 0.5), rel=1e-9)
        assert r["min"] == pytest.approx(np.nanmin(sg))
        assert r["is_constant"] is False


def test_domain_ranges_override(run_checks):
    m = make_model("regular")
    base = run_checks(m).bundle["density"]["anomalies"]
    # OVB density is 2.0 +/- 0.1, so a 2.5 lower limit flags every OVB block with a value
    ctx = run_checks(m, settings={"density": {"domain_ranges": {"OVB": [2.5, 3.0]}}})
    t = _used(m)
    ovb = t[t["domain"] == "OVB"]
    ovb_low_extra = int(ovb["sg"].between(1.5, 5.5).sum())  # were inside the default range
    assert ctx.bundle["density"]["anomalies"]["low"] == base["low"] + ovb_low_extra


def test_waste_left_out_when_asked(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, settings={"density": {"check_waste": False}})
    t = _used(m)
    t = t[~t["domain"].isin(WASTE)]
    lo, hi = 1.5, 5.5
    expected = int((t["sg"].isna() | (t["sg"] < lo) | (t["sg"] > hi)).sum())
    assert ctx.flags.counts()["FLAG_SG_ANOMALY"] == expected


def test_density_vs_grade(run_checks):
    m = make_model("regular", inject=False, n_parents=(20, 20, 8))
    ctx = run_checks(m)
    result = ctx.bundle["density"]
    t = _used(m)
    got = {(r["element"], r["domain"]): r for r in result["density_vs_grade"]}
    sul = t[t["domain"] == "SUL"]
    sg = sul["sg"].astype(np.float32).astype(np.float64).to_numpy()
    cu = sul["cu"].astype(np.float32).astype(np.float64).to_numpy()
    r = got[("cu", "SUL")]
    assert r["pairs"] == len(sul)
    assert r["pearson"] == pytest.approx(np.corrcoef(sg, cu)[0, 1], abs=1e-9)
    assert r["spearman"] == pytest.approx(sps.spearmanr(sg, cu).statistic, abs=1e-9)
    # gold is only estimated in SUL: other domains are skipped for too few pairs
    skipped = {(s["element"], s["domain"]) for s in result["density_vs_grade_skipped"]}
    assert ("au", "OX") in skipped
    assert ("au", "SUL") in got
