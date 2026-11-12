"""Estimation support checks (spec 7.6) against the synthetic models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import KIND_STYLES
from synthetic import make_model

from qaqc.modules import support


@pytest.mark.parametrize(("kind", "style"), KIND_STYLES)
def test_flag_and_condition_counts(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = run_checks(m, modules=("geometry",))
    support.run(ctx)
    res = ctx.bundle["support"]
    low = res["low_support"]
    assert (
        res["flags"]["FLAG_LOW_SUPPORT_HIGH_CLASS"]
        == m.expected["flags"]["FLAG_LOW_SUPPORT_HIGH_CLASS"]
    )
    assert ctx.flags.counts()["FLAG_LOW_SUPPORT_HIGH_CLASS"] == low["blocks"]
    assert low["by_condition"] == m.expected["low_support_by_condition"]
    assert low["missing_by_condition"] == m.expected["low_support_missing_by_condition"]
    assert "kriging" in res["kriging_variance_note"].lower()
    json.dumps(res)


def test_leapfrog_uses_element_scoped_source(run_checks):
    ctx = run_checks(make_model("regular", style="leapfrog"), modules=("geometry",))
    support.run(ctx)
    src = ctx.bundle["support"]["source"]
    assert src["element"] == "cu"
    assert set(src["by_detail"].values()) == {"element:cu"}
    assert src["columns"]["nsamples"] == "OK Cu: NS"


def test_global_metadata_preferred(run_checks):
    ctx = run_checks(make_model("regular", style="datamine"), modules=("geometry",))
    support.run(ctx)
    src = ctx.bundle["support"]["source"]
    assert src["by_detail"]["pass_num"] == "global"
    assert src["element"] is None or src["element"] in ctx.schema.elements


def test_explicit_element_without_columns_warns(run_checks):
    ctx = run_checks(
        make_model("regular", style="leapfrog"),
        modules=("geometry",),
        settings={"support": {"element": "zn"}},
    )
    support.run(ctx)
    assert ctx.bundle["support"].get("skipped")
    assert any("support.element" in w.message for w in ctx.warnings)


def test_skipped_without_metadata(run_checks):
    ctx = run_checks(make_model("regular", style="isatis"), modules=("geometry",))
    ctx.schema.global_metadata.clear()
    for es in ctx.schema.elements.values():
        es.metadata.clear()
    support.run(ctx)
    assert ctx.bundle["support"]["skipped"]
    assert ctx.flags.counts()["FLAG_LOW_SUPPORT_HIGH_CLASS"] == 0
    assert any(w.severity == "info" and w.module == "support" for w in ctx.warnings)


def test_pass_breakdown_and_stats(run_checks):
    m = make_model("regular", style="datamine")
    ctx = run_checks(m, modules=("geometry",))
    support.run(ctx)
    res = ctx.bundle["support"]
    geo_t = ctx.bundle["geometry"]["total_tonnes"]
    by_class = res["pass_breakdown"]["by_class"]
    assert sum(r["tonnes"] for r in by_class) == pytest.approx(geo_t)
    shares: dict[int, float] = {}
    for r in by_class:
        shares[r["rank"]] = shares.get(r["rank"], 0.0) + r["pct_of_row"]
    assert all(v == pytest.approx(100.0) for v in shares.values())
    by_dom = res["pass_breakdown"]["by_domain"]
    assert sum(r["tonnes"] for r in by_dom) == pytest.approx(geo_t)
    stats = {s["rank"]: s for s in res["stats_by_class"]}
    assert set(stats) <= {0, 1, 2, 3}
    for s in stats.values():
        d = s["details"]["nholes"]
        if d["blocks_with_value"]:
            assert d["min"] <= d["p10"] <= d["p50"] <= d["p90"] <= d["max"]
    # unweighted-safe check: min and max over a class equal the column's
    bm = ctx.bm
    from qaqc.domains import class_ranks

    ranks, _ = class_ranks(bm, ctx.cfg)
    sel = ctx.analysis_mask & (ranks == 3) & np.isfinite(bm["_TONNES"].to_numpy())
    col = ctx.bundle["support"]["source"]["columns"]["nholes"]
    assert stats[3]["details"]["nholes"]["max"] == np.nanmax(
        bm[_internal(ctx, col)].to_numpy()[sel]
    )


def _internal(ctx, file_col):
    from qaqc.io import internal_columns

    inv = {v: k for k, v in internal_columns(ctx.schema).items()}
    return inv[file_col]


def test_null_threshold_skipped(run_checks):
    m = make_model("regular", style="datamine")
    ctx = run_checks(
        m,
        modules=("geometry",),
        settings={
            "support": {
                "thresholds_by_rank": {
                    3: {"nholes_min": 3, "max_pass": None},
                    2: {"nholes_min": 2, "max_pass": None},
                }
            }
        },
    )
    support.run(ctx)
    by = ctx.bundle["support"]["low_support"]["by_condition"]
    assert set(by) == {"nholes"}
    assert by["nholes"] == m.expected["low_support_by_condition"]["nholes"]
