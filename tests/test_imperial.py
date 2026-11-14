"""Imperial projects: feet, short tons per cubic foot, short tons and troy ounces per short ton."""

from __future__ import annotations

import dataclasses
import io as stdio

import numpy as np
import pandas as pd
import pytest
from synthetic import make_model

from qaqc import pipeline, report, units

T_M3_TO_ST_FT3 = 0.0312142799  # 1 t/m3 = 62.428 lb/ft3 = 0.0312 short tons/ft3
OPT = units.factor("opt", "gpt")  # 34.2857 g/t per oz/ton


@pytest.fixture
def imperial(run_checks):
    """A clean model rewritten as imperial: density in short tons/ft3, gold in oz/ton.

    The coordinates and block sizes are read as feet (the numbers stay the same).
    """
    m = make_model("regular", inject=False)
    frame = pd.read_csv(stdio.StringIO(m.csv), dtype=str, keep_default_na=False)
    sg = pd.to_numeric(frame["SG"], errors="coerce")
    frame["SG"] = (sg * T_M3_TO_ST_FT3).map(lambda v: "" if np.isnan(v) else f"{v:.6f}")
    au = pd.to_numeric(frame["AU_GPT"], errors="coerce") / OPT
    frame["AU_GPT"] = au.map(lambda v: "" if np.isnan(v) else f"{v:.8f}")
    frame = frame.rename(columns={"AU_GPT": "AU_OPT"})
    m2 = dataclasses.replace(m, csv=frame.to_csv(index=False, lineterminator="\n"))
    settings = {
        "units": "imperial",
        "elements": {
            "cu": {"unit": "pct", "ceiling": 34.6, "cutoff": 0.5},
            "au": {"unit": "opt", "ceiling": 15.0, "cutoff": 0.5 / OPT},
        },
    }
    ctx = run_checks(
        m2, modules=("geometry", "density", "grades", "classification"), settings=settings
    )
    return ctx, frame


def test_tonnage_is_volume_times_density(imperial):
    ctx, frame = imperial
    bm = ctx.bm
    use = ctx.analysis_mask
    expected = np.nansum(bm["_VOL"].to_numpy()[use] * bm["sg"].to_numpy(dtype=np.float64)[use])
    assert ctx.bundle["geometry"]["total_tonnes"] == pytest.approx(expected, rel=1e-9)
    assert ctx.bundle["units"]["tonnage"] == "short tons"
    assert ctx.bundle["units"]["volume"] == "ft³"


def test_default_imperial_density_range_flags_nothing(imperial):
    ctx, _ = imperial
    assert ctx.cfg.density_range == (0.045, 0.175)
    assert ctx.flags.counts()["FLAG_SG_ANOMALY"] == 0


def test_gold_ounces_are_short_tons_times_oz_per_ton(imperial):
    ctx, _ = imperial
    au = ctx.schema.elements["au"]
    assert au.unit == "opt" and au.metal_unit == "oz"
    assert ctx.schema.elements["cu"].metal_unit == "short tons"
    bm = ctx.bm
    use = ctx.analysis_mask & np.isfinite(bm["au"].to_numpy(dtype=np.float64))
    tons = bm["_TONNES"].to_numpy()[use]
    opt = bm["au"].to_numpy(dtype=np.float64)[use]
    rows = ctx.bundle["grades"]["elements"]["au"]["by_domain"]
    total = sum(r["contained_metal"] or 0.0 for r in rows)
    assert total == pytest.approx(np.nansum(tons * opt), rel=1e-6)


def test_same_flags_as_the_metric_model(imperial, run_checks):
    ctx, _ = imperial
    metric = run_checks(
        make_model("regular", inject=False),
        modules=("geometry", "density", "grades", "classification"),
    )
    assert ctx.flags.counts() == metric.flags.counts()


def test_report_uses_imperial_labels(imperial):
    ctx, _ = imperial
    pipeline.run_checks(ctx, ["support"])
    text = report.write_report(ctx).read_text(encoding="utf-8")
    for label in ("Short tons", "ft³", "short tons/ft³", "oz/ton"):
        assert label in text, label
