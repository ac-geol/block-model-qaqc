"""Grade checks against the synthetic models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import KIND_STYLES
from synthetic import make_model

from qaqc.context import jsonable
from qaqc.modules import grades
from qaqc.stats import weighted_mean


def _run(run_checks, m, settings=None):
    ctx = run_checks(m, modules=("geometry",), settings=settings)
    grades.run(ctx)
    return ctx


@pytest.mark.parametrize(("kind", "style"), KIND_STYLES)
def test_flag_counts_and_applicability(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = _run(run_checks, m)
    b = ctx.bundle["grades"]
    for name, n in b["flags"].items():
        assert n == m.expected["flags"][name], name
        assert ctx.flags.counts()[name] == n
    for el, expected in m.expected["applicable_domains"].items():
        assert sorted(b["elements"][el]["applicable_domains"]) == sorted(expected)
    json.dumps(jsonable(b))


def test_not_estimated_domain_is_reported_not_flagged(run_checks):
    ctx = _run(run_checks, make_model("regular"))
    au = ctx.bundle["grades"]["elements"]["au"]
    assert "OX" in au["not_estimated_domains"]
    row = next(r for r in au["by_domain"] if r["domain"] == "OX")
    assert row["status"] == grades.NOT_ESTIMATED


def test_unit_conversion_of_thresholds(run_checks):
    ctx = _run(run_checks, make_model("regular"), {"elements": {"cu": {"unit": "ppm"}}})
    th = ctx.bundle["grades"]["elements"]["cu"]["thresholds"]
    assert th["ceiling"] == pytest.approx(346000)
    assert th["diagnostic_cutoff"] == pytest.approx(5000)
    assert th["unit"] == "ppm"


def test_unit_likely_wrong_warning(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, modules=("geometry",))
    ctx.bm["cu"] = ctx.bm["cu"] * 1e4  # ppm values read as percent
    grades.run(ctx)
    assert ctx.bundle["grades"]["elements"]["cu"]["units"]["likely_wrong"]
    assert any("likely wrong" in w.message for w in ctx.warnings)

    ctx = run_checks(m, modules=("geometry",), settings={"elements": {"cu": {"unit": "ppm"}}})
    ctx.bm["cu"] = ctx.bm["cu"] * 1e-3  # tiny values read as ppm
    grades.run(ctx)
    assert ctx.bundle["grades"]["elements"]["cu"]["units"]["likely_wrong"]

    ctx = _run(run_checks, m)
    assert not ctx.bundle["grades"]["elements"]["cu"]["units"]["likely_wrong"]


def test_null_thresholds_skip_checks(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m, {"elements": {"cu": {"ceiling": None, "diagnostic_cutoff": None}}})
    cu = ctx.bundle["grades"]["elements"]["cu"]
    assert {s["check"] for s in cu["skipped_checks"]} == {"FLAG_CEILING_CU", "FLAG_BLEED_CU"}
    assert ctx.bundle["grades"]["flags"]["FLAG_CEILING_CU"] == 0
    assert ctx.bundle["grades"]["flags"]["FLAG_BLEED_CU"] == 0
    assert ctx.bundle["grades"]["flags"]["FLAG_NEG_CU"] == m.expected["flags"]["FLAG_NEG_CU"]
    assert ctx.bundle["grades"]["elements"]["au"]["skipped_checks"] == []


def test_element_without_config_runs_without_thresholds(run_checks):
    ctx = run_checks(make_model("regular"), modules=("geometry",))
    ctx.cfg.elements.items.pop("au")
    ctx.schema.elements["au"].configured = False
    grades.run(ctx)
    au = ctx.bundle["grades"]["elements"]["au"]
    assert au["thresholds"]["ceiling"] is None
    assert len(au["skipped_checks"]) == 2
    assert ctx.bundle["grades"]["flags"]["FLAG_NEG_AU"] >= 0
    assert au["by_domain"]


def test_contained_metal_hand_made(run_checks):
    ctx = run_checks(make_model("regular"), modules=("geometry",))
    n = int(ctx.analysis_mask.sum())
    ctx.bm["_TONNES"] = 1000.0
    ctx.bm["cu"] = 1.0  # 1 % of 1000 t = 10 t
    ctx.bm["au"] = 31.1035  # 31.1035 g/t x 1000 t = 1000 oz
    grades.run(ctx)
    els = ctx.bundle["grades"]["elements"]
    assert els["cu"]["metal_unit"] == "t"
    assert sum(r["contained_metal"] for r in els["cu"]["by_domain"]) == pytest.approx(10.0 * n)
    assert els["au"]["metal_unit"] == "oz"
    assert sum(r["contained_metal"] for r in els["au"]["by_domain"]) == pytest.approx(1000.0 * n)

    # copper held as ppm is still reported in tonnes of metal
    ctx.schema.elements["cu"].unit = "ppm"
    ctx.bm["cu"] = 10000.0
    grades.run(ctx)
    cu = ctx.bundle["grades"]["elements"]["cu"]
    assert cu["metal_unit"] == "t"
    assert sum(r["contained_metal"] for r in cu["by_domain"]) == pytest.approx(10.0 * n)


def test_domain_table_weighted_stats(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m)
    t = ctx.bm
    rows = ctx.bundle["grades"]["elements"]["cu"]["by_domain"]
    sul = next(r for r in rows if r["domain"] == "SUL")
    sel = ctx.analysis_mask & (t["domain"] == "SUL").to_numpy()
    g = t["cu"].to_numpy(dtype=float)[sel]
    w = t["_TONNES"].to_numpy(dtype=float)[sel]
    assert sul["blocks"] == int(sel.sum())
    assert sul["unestimated"] == int(np.isnan(g).sum())
    assert sul["mean_weighted"] == pytest.approx(weighted_mean(g, w))
    assert sul["mean_naive"] == pytest.approx(np.nanmean(g))
    assert sul["tonnes"] == pytest.approx(np.nansum(w))
