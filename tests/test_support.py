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


def test_models_without_kriging_details_run_cleanly(tmp_path):
    """Many models carry pass, samples and holes but no KV, KE or slope of regression."""
    import pandas as pd

    from qaqc import pipeline
    from qaqc.config import from_dict

    m = make_model("sub_blocked", style="datamine")
    m.write(tmp_path)
    csv = tmp_path / "block_model.csv"
    pd.read_csv(csv, dtype=str, keep_default_na=False).drop(columns=["KV", "KE", "SOR"]).to_csv(
        csv, index=False
    )
    cfg = from_dict(m.config_dict(), base_dir=tmp_path)
    ctx = pipeline.prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx)
    files = pipeline.export.write_outputs(ctx, report_path=pipeline.write_report(ctx))

    sup = ctx.bundle["support"]
    assert not sup.get("skipped")
    assert {"slope_reg", "kriging_eff", "kriging_var"} <= set(sup["source"]["missing_details"])
    assert set(sup["low_support"]["by_condition"]) == {"nholes", "nsamples", "pass_num"}
    skipped = {(s["condition"], s["reason"]) for s in sup["skipped_checks"]}
    assert ("slope_reg", "detail not in model") in skipped
    assert ("kriging_eff", "detail not in model") in skipped
    assert not [w for w in ctx.warnings if w.severity == "error" and w.module == "support"]
    assert not [w for w in ctx.warnings if "could not be drawn" in w.message]
    assert "support_box" in ctx.figures or any(k.startswith("support") for k in ctx.figures)
    assert "BM_QAQC_Report.html" in files


def test_null_threshold_skipped(run_checks):
    m = make_model("regular", style="datamine")
    ctx = run_checks(
        m,
        modules=("geometry",),
        settings={
            "support": {
                "measured": {"nholes_min": 3, "max_pass": None},
                "indicated": {"nholes_min": 2, "max_pass": None},
            }
        },
    )
    support.run(ctx)
    by = ctx.bundle["support"]["low_support"]["by_condition"]
    assert set(by) == {"nholes"}
    assert by["nholes"] == m.expected["low_support_by_condition"]["nholes"]
