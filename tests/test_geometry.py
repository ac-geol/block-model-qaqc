"""Grid set-up and geometry checks against the synthetic models (exact counts)."""

from __future__ import annotations

import dataclasses
import io as stdio

import numpy as np
import pandas as pd
import pytest
from conftest import KIND_STYLES
from synthetic import make_model

from qaqc.grid import GridError, RotationNotSupported

GEOMETRY_FLAGS = (
    "FLAG_DUP_CENTROID",
    "FLAG_OFF_GRID",
    "FLAG_PARENT_OVERFILL",
    "FLAG_PCT_ANOMALY",
    "FLAG_SG_ANOMALY",
)


def _with_csv(m, frame: pd.DataFrame):
    """Copy of a synthetic model with its CSV replaced (no header lines)."""
    return dataclasses.replace(
        m, csv=frame.to_csv(index=False, lineterminator="\n"), header_lines=[]
    )


def _frame(m) -> pd.DataFrame:
    return pd.read_csv(
        stdio.StringIO(m.csv),
        comment=None,
        skiprows=len(m.header_lines),
        dtype=str,
        keep_default_na=False,
    )


def _messages(ctx, severity=None):
    return [w.message for w in ctx.warnings if severity is None or w.severity == severity]


@pytest.mark.parametrize("kind,style", KIND_STYLES)
def test_flag_counts(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = run_checks(m)
    counts = ctx.flags.counts()
    for flag in GEOMETRY_FLAGS:
        if flag in m.expected["flags"]:
            assert counts[flag] == m.expected["flags"][flag], flag
    assert counts["FLAG_DUP_CENTROID"] == 2 * m.injected["duplicate"]
    sg = ctx.bundle["density"]["anomalies"]
    assert {k: sg[k] for k in ("missing", "low", "high")} == m.expected["sg_breakdown"]
    if kind == "sub_blocked":
        assert (
            ctx.bundle["geometry"]["overfill"]["parent_cells"]
            == m.expected["overfill_parent_count"]
        )
    assert ctx.grid.model_type == kind
    assert ctx.grid.origin == pytest.approx(m.grid["origin"], abs=1e-9)
    assert ctx.grid.parent_size == pytest.approx(m.grid["parent_size"])


@pytest.mark.parametrize("kind,style", KIND_STYLES)
def test_clean_models_have_no_flags(run_checks, kind, style):
    ctx = run_checks(make_model(kind, style=style, inject=False))
    assert {f: n for f, n in ctx.flags.counts().items() if n} == {}


@pytest.mark.parametrize("style", ["generic", "leapfrog", "datamine"])
def test_subblock_grid_check(run_checks, style):
    m = make_model("sub_blocked", style=style)
    ctx = run_checks(m, settings={"model": {"subblock_grid_check": True}})
    expected = m.expected["flags_with_subblock_grid_check"]["FLAG_OFF_GRID"]
    assert ctx.flags.counts()["FLAG_OFF_GRID"] == expected
    assert expected == m.injected["subblock_crossing"] + m.injected["subblock_misaligned"]


@pytest.mark.parametrize("grid_check", [False, True])
def test_precision_half_metre_subblocks(run_checks, grid_check):
    """0.5 m sub-blocks at northing 7,000,000.25 give no false off-grid or duplicate flags."""
    m = make_model(
        "sub_blocked", origin=(500000.0, 7000000.25, 0.0), parent_size=(1.0, 1.0, 1.0), inject=False
    )
    ctx = run_checks(m, settings={"model": {"subblock_grid_check": grid_check}})
    counts = ctx.flags.counts()
    assert counts["FLAG_OFF_GRID"] == 0
    assert counts["FLAG_DUP_CENTROID"] == 0
    assert counts["FLAG_PARENT_OVERFILL"] == 0
    assert ctx.grid.origin == pytest.approx((500000.0, 7000000.25, 0.0), abs=1e-9)
    assert ctx.grid.min_subblock_size == (0.5, 0.5, 0.5)


# --- rotation ----------------------------------------------------------------------------------


def test_rotated_model_in_config_stops(run_checks):
    with pytest.raises(RotationNotSupported, match="not supported yet"):
        run_checks(make_model("regular", rotation_deg=30.0))


def test_rotated_model_in_header_stops(run_checks):
    with pytest.raises(RotationNotSupported, match="file header"):
        run_checks(make_model("regular", style="leapfrog", rotation_deg=30.0))


def test_dip_in_header_stops(run_checks):
    m = make_model("regular", style="leapfrog")
    m = dataclasses.replace(m, csv=m.csv.replace("# rotation: 0.0, 0, 0", "# rotation: 0, 10, 0"))
    with pytest.raises(RotationNotSupported, match="dip"):
        run_checks(m)


def test_coords_already_local_skip_rotation(run_checks):
    m = make_model("regular")
    ctx = run_checks(
        m, settings={"model": {"rotation": {"azimuth_deg": 30.0}, "coords_are_local": True}}
    )
    assert ctx.flags.counts()["FLAG_OFF_GRID"] == m.expected["flags"]["FLAG_OFF_GRID"]


# --- grid values -------------------------------------------------------------------------------


def test_origin_ignores_a_stray_block(run_checks):
    m = make_model("regular")
    frame = _frame(m)
    stray = frame.iloc[[0]].copy()
    stray["X"] = str(float(stray["X"].iloc[0]) - 1003.0)
    m2 = _with_csv(m, pd.concat([frame, stray], ignore_index=True))
    ctx = run_checks(m2)
    assert ctx.grid.origin == pytest.approx(m.grid["origin"])
    assert ctx.flags.counts()["FLAG_OFF_GRID"] == m.expected["flags"]["FLAG_OFF_GRID"] + 1


def test_wrong_origin_is_reported(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, settings={"model": {"grid_origin": [500003.0, 7000000.0, 0.0]}})
    assert any("off the model grid" in msg for msg in _messages(ctx, "error"))
    assert ctx.grid.sources["origin"] == "config"


def test_parent_size_mismatch_warning(run_checks):
    ctx = run_checks(make_model("regular"), settings={"model": {"parent_cell_size": [10, 10, 10]}})
    assert any("does not match" in msg for msg in _messages(ctx, "warning"))
    assert ctx.grid.parent_size == (10.0, 10.0, 10.0)


def test_percentage_column_with_varying_sizes_needs_model_type(run_checks):
    m = make_model("sub_blocked", inject=False)
    frame = _frame(m)
    frame["PCT"] = "1"
    m2 = _with_csv(m, frame)
    with pytest.raises(GridError, match="model_type"):
        run_checks(m2)
    ctx = run_checks(m2, settings={"model": {"model_type": "sub_blocked"}})
    assert ctx.grid.model_type == "sub_blocked"


def test_percent_scale_worked_out(run_checks):
    m = make_model("percentage", pct_scale="percent")
    ctx = run_checks(m, settings={"model": {"pct_scale": "auto"}})
    assert ctx.grid.pct_scale == "percent"
    assert any("read as percent" in msg for msg in _messages(ctx, "warning"))
    assert ctx.flags.counts()["FLAG_PCT_ANOMALY"] == m.expected["flags"]["FLAG_PCT_ANOMALY"]


def test_blocks_without_coordinates(run_checks):
    m = make_model("regular", inject=False)
    frame = _frame(m)
    frame.loc[5, "X"] = ""
    ctx = run_checks(_with_csv(m, frame))
    assert ctx.bundle["geometry"]["blocks_without_coordinates"] == 1
    assert any("no X, Y or Z" in msg for msg in _messages(ctx, "error"))


# --- volume, tonnes and totals -----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["regular", "sub_blocked", "percentage"])
def test_volume_and_tonnes(run_checks, kind):
    m = make_model(kind)
    ctx = run_checks(m)
    t = m.truth
    use = (t["selected"] & ~t["is_air"]).to_numpy()
    geo = ctx.bundle["geometry"]
    assert geo["total_volume"] == pytest.approx(t.loc[use, "vol"].sum(), rel=1e-12)
    tonnes = (t["vol"] * t["sg"].astype(np.float32).astype(np.float64))[use]
    assert geo["total_tonnes"] == pytest.approx(np.nansum(tonnes), rel=1e-9)
    assert geo["blocks_without_tonnes"] == int(t.loc[use, "sg"].isna().sum())
    assert geo["blocks_analysed"] == int(use.sum())


def test_no_density_column(run_checks):
    m = make_model("regular", inject=False)
    ctx = run_checks(_with_csv(m, _frame(m).drop(columns=["SG"])))
    assert ctx.bm["_TONNES"].isna().all()
    assert ctx.bundle["density"]["skipped"] == "no density column"
    assert len(_messages(ctx, "error")) == 2


def test_wireframe_volumes(run_checks):
    m = make_model("sub_blocked")
    t = m.truth
    ox = float(t.loc[t["selected"] & (t["domain"] == "OX"), "vol"].sum())
    ctx = run_checks(m, settings={"target_wireframe_volumes": {"OX": ox * 1.02, "SUL": 1000.0}})
    rows = {r["domain"]: r for r in ctx.bundle["geometry"]["wireframe_volumes"]}
    assert rows["OX"]["model_volume"] == pytest.approx(ox)
    assert rows["OX"]["difference_pct"] == pytest.approx((1 / 1.02 - 1) * 100)
    assert rows["SUL"]["wireframe_volume"] == 1000.0


# --- sub-block estimation support --------------------------------------------------------------


def _expected_support_flags(m, declared: str, tol: float = 1e-4) -> int:
    t = m.truth
    t = t[t["selected"] & ~t["is_air"]]
    bad = pd.Series(False, index=t.index)
    for el in ("cu", "au"):
        v = t[el].astype(np.float32).astype(np.float64)
        grp = t.assign(v=v).groupby(["I", "J", "K", "domain"])["v"]
        n = grp.transform("count")
        sd = grp.transform(
            lambda s: (
                np.sqrt(np.mean((s.dropna() - s.dropna().mean()) ** 2))
                if s.notna().sum()
                else np.nan
            )
        )
        varying = (n >= 2) & (sd > tol)
        same = (n >= 2) & ~(sd > tol)
        bad |= varying if declared == "parent" else same
    return int(bad.sum())


@pytest.mark.parametrize("support", ["parent", "subblock"])
def test_subblock_support_observed(run_checks, support):
    ctx = run_checks(make_model("sub_blocked", subblock_support=support, inject=False))
    info = ctx.bundle["geometry"]["subblock_support"]
    assert info["elements"]["cu"]["observed_support"] == support
    assert ctx.flags.counts()["FLAG_SUBBLOCK_SUPPORT_MISMATCH"] == 0  # nothing declared


@pytest.mark.parametrize("support", ["parent", "subblock"])
@pytest.mark.parametrize("declared", ["parent", "subblock"])
def test_subblock_support_mismatch_flag(run_checks, support, declared):
    m = make_model("sub_blocked", subblock_support=support)
    ctx = run_checks(m, settings={"model": {"expected_estimation_support": declared}})
    assert ctx.flags.counts()["FLAG_SUBBLOCK_SUPPORT_MISMATCH"] == _expected_support_flags(
        m, declared
    )
    if support != declared:
        assert ctx.flags.counts()["FLAG_SUBBLOCK_SUPPORT_MISMATCH"] > 0
