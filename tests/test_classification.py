"""Classification checks (spec 7.5) against the synthetic models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import KIND_STYLES
from synthetic import make_model

from qaqc.modules import classification


@pytest.mark.parametrize(("kind", "style"), KIND_STYLES)
def test_flag_and_unmapped_counts(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = run_checks(m, modules=("geometry",))
    classification.run(ctx)
    res = ctx.bundle["classification"]
    assert (
        res["flags"]["FLAG_UNCLASS_MINERALIZED"] == m.expected["flags"]["FLAG_UNCLASS_MINERALIZED"]
    )
    assert (
        ctx.flags.counts()["FLAG_UNCLASS_MINERALIZED"] == res["flags"]["FLAG_UNCLASS_MINERALIZED"]
    )
    got = {r["code"]: r["blocks"] for r in res["unmapped_codes"]}
    assert got == m.expected["unmapped_codes"]
    assert any("not in the rank map" in w.message for w in ctx.warnings)
    json.dumps(ctx.bundle["classification"])


@pytest.mark.parametrize("kind", ["regular", "sub_blocked", "percentage"])
def test_inventory_adds_up(run_checks, kind):
    m = make_model(kind)
    ctx = run_checks(m, modules=("geometry",))
    classification.run(ctx)
    res = ctx.bundle["classification"]
    geo = ctx.bundle["geometry"]
    assert sum(r["tonnes"] for r in res["inventory"]) == pytest.approx(geo["total_tonnes"])
    assert sum(r["volume_m3"] for r in res["inventory"]) == pytest.approx(geo["total_volume_m3"])
    assert sum(r["blocks"] for r in res["inventory"]) == geo["blocks_analysed"]
    assert sum(r["tonnes"] for r in res["by_class"]) == pytest.approx(geo["total_tonnes"])
    # shares of each domain add to 100
    per_dom: dict[str, float] = {}
    for r in res["inventory"]:
        per_dom[r["domain"]] = per_dom.get(r["domain"], 0.0) + r["pct_of_domain"]
    assert all(v == pytest.approx(100.0) for v in per_dom.values())


def test_inventory_grade_and_metal(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, modules=("geometry",))
    classification.run(ctx)
    bm = ctx.bm
    row = next(r for r in ctx.bundle["classification"]["inventory"] if r["rank"] == 3)
    sel = ctx.analysis_mask & (bm["domain"] == row["domain"]).to_numpy()
    from qaqc.domains import class_ranks

    ranks, _ = class_ranks(bm, ctx.cfg)
    sel &= ranks == 3
    t = bm["_TONNES"].to_numpy()[sel]
    g = bm["cu"].to_numpy(dtype=np.float64)[sel]
    ok = np.isfinite(g) & np.isfinite(t)
    assert row["grade"]["cu"] == pytest.approx((t[ok] * g[ok]).sum() / t[ok].sum())
    assert row["metal"]["cu"] == pytest.approx((t[ok] * g[ok]).sum() / 100.0)


def _hand_model(run_checks, resource_cutoff, include_waste=False):
    """Four-block model with hand-set values, run through geometry first."""
    m = make_model("regular", inject=False, n_parents=(8, 8, 6))
    ctx = run_checks(
        m,
        modules=("geometry",),
        settings={
            "reporting": {
                "resource_cutoff": resource_cutoff,
                "include_waste_in_statement": include_waste,
            }
        },
    )
    bm = ctx.bm
    bm["_AIR"] = True
    idx = bm.index[:5]
    bm.loc[idx, "_AIR"] = False
    bm["domain"] = bm["domain"].astype(object)
    bm["classif"] = bm["classif"].astype(object)
    # domain, class code, copper (pct), gold, tonnes
    rows = [
        ("SUL", "1", 1.0, 2.0, 100.0),
        ("SUL", "2", 0.6, 1.0, 300.0),
        ("SUL", "2", 0.2, 5.0, 100.0),  # below the 0.3 % cutoff
        ("OX", "3", 0.4, 0.0, 200.0),
        ("OVB", "0", 0.5, 0.0, 1000.0),  # waste
    ]
    for i, (d, c, cu, au, t) in zip(idx, rows, strict=True):
        bm.loc[i, ["domain", "classif", "cu", "au", "_TONNES"]] = [d, c, cu, au, t]
        bm.loc[i, "_VOL"] = t / 2.0
    return ctx


def test_resource_statement_hand_checked(run_checks):
    ctx = _hand_model(run_checks, {"element": "cu", "value": 0.3})
    classification.run(ctx)
    st = ctx.bundle["classification"]["resource_statement"]
    assert st["cutoff_in_column_unit"] == pytest.approx(0.3)
    rows = {(r["domain"], r["rank"]): r for r in st["rows"]}
    assert set(rows) == {("SUL", 3), ("SUL", 2), ("OX", 1)}  # waste and low grade left out
    assert rows[("SUL", 2)]["tonnes"] == 300.0
    assert rows[("SUL", 2)]["grade"]["cu"] == pytest.approx(0.6)
    assert rows[("SUL", 2)]["metal"]["cu"] == pytest.approx(300 * 0.6 / 100)
    by_class = {r["rank"]: r for r in st["by_class"]}
    assert by_class[3]["tonnes"] == 100.0
    assert by_class[1]["metal"]["cu"] == pytest.approx(0.8)
    tot = st["total"]
    assert tot["tonnes"] == pytest.approx(600.0)
    assert tot["grade"]["cu"] == pytest.approx((100 * 1.0 + 300 * 0.6 + 200 * 0.4) / 600)
    assert tot["metal"]["cu"] == pytest.approx(1.0 + 1.8 + 0.8)
    # gold: pct canonical, gpt column; ounces = t * g / 31.1035
    assert tot["metal"]["au"] == pytest.approx((100 * 2.0 + 300 * 1.0) / 31.1035)


def test_resource_statement_waste_and_unit_conversion(run_checks):
    ctx = _hand_model(run_checks, {"element": "cu", "value": 0.3}, include_waste=True)
    classification.run(ctx)
    st = ctx.bundle["classification"]["resource_statement"]
    assert st["total"]["tonnes"] == pytest.approx(1600.0)
    # cutoff is in canonical units: 0.3 pct is 3000 ppm in a ppm column
    ctx2 = _hand_model(run_checks, {"element": "cu", "value": 0.3})
    ctx2.schema.elements["cu"].unit = "ppm"
    ctx2.bm["cu"] = ctx2.bm["cu"] * 10_000
    classification.run(ctx2)
    st2 = ctx2.bundle["classification"]["resource_statement"]
    assert st2["cutoff_in_column_unit"] == pytest.approx(3000.0)
    assert st2["total"]["tonnes"] == pytest.approx(600.0)
    # metal stays in tonnes of metal (canonical unit), DECISIONS.md 9
    assert st2["total"]["metal"]["cu"] == pytest.approx(3.6)


def test_no_statement_by_default(run_checks):
    ctx = run_checks(make_model("regular"), modules=("geometry",))
    classification.run(ctx)
    assert ctx.bundle["classification"]["resource_statement"] is None


def test_null_cutoff_elements_do_not_count(run_checks):
    m = make_model("regular")
    base = run_checks(m, modules=("geometry",))
    classification.run(base)
    n_both = base.bundle["classification"]["unclassified_mineralized"]["blocks"]
    ctx = run_checks(
        m,
        modules=("geometry",),
        settings={"elements": {"au": {"diagnostic_cutoff": None}}},
    )
    classification.run(ctx)
    res = ctx.bundle["classification"]["unclassified_mineralized"]
    assert "au" in res["elements_without_cutoff"]
    assert res["blocks"] <= n_both


def test_no_classification_column(run_checks):
    ctx = run_checks(make_model("regular"), modules=("geometry",))
    ctx.bm.drop(columns=["classif"], inplace=True)
    classification.run(ctx)
    assert ctx.bundle["classification"]["skipped"]
    assert any(w.severity == "error" for w in ctx.warnings)
