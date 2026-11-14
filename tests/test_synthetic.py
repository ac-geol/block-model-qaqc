"""Tests that the synthetic generator plants exactly what it says it plants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import DEFAULT_PLANTS, KINDS, make_model

from qaqc.grid import world_to_local

P = DEFAULT_PLANTS


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rotation", [0.0, 30.0])
def test_planted_counts(kind, rotation):
    m = make_model(kind, rotation_deg=rotation)
    f = m.expected["flags"]
    assert f["FLAG_DUP_CENTROID"] == 2 * P["duplicates"]
    if kind == "sub_blocked":
        assert f["FLAG_OFF_GRID"] == P["subblock_crossing"]
        assert m.expected["flags_with_subblock_grid_check"]["FLAG_OFF_GRID"] == (
            P["subblock_crossing"] + P["subblock_misaligned"]
        )
        # duplicated full parents and parents given an extra sub-block: 2 rows each
        assert f["FLAG_PARENT_OVERFILL"] == 2 * (P["duplicates"] + P["overfill"])
        assert m.expected["overfill_parent_count"] == P["duplicates"] + P["overfill"]
    else:
        assert f["FLAG_OFF_GRID"] == P["off_grid"]
    if kind == "percentage":
        assert f["FLAG_PCT_ANOMALY"] == (
            2 * P["duplicates"] + P["pct_over_1"] + P["pct_negative"] + 2 * P["pct_sum_over_1"]
        )
    sg = m.expected["sg_breakdown"]
    assert sg == {"missing": P["sg_missing"] + 3 + 1, "low": P["sg_low"], "high": P["sg_high"]}
    assert f["FLAG_NEG_CU"] == P["cu_negative"]
    assert f["FLAG_CEILING_CU"] == P["cu_ceiling"]
    assert f["FLAG_UNEST_CU"] == P["cu_unestimated"] + 7 + 1  # blanks + placeholders + junk
    assert f["FLAG_BLEED_CU"] == P["cu_bleed"]
    assert f["FLAG_UNEST_AU"] == P["au_unestimated"]
    assert f["FLAG_BLEED_AU"] == 0
    assert m.expected["applicable_domains"] == {"cu": ["OX", "SUL"], "au": ["SUL"]}
    assert m.expected["unmapped_codes"] == {"PROV": P["unmapped_code"]}
    assert f["FLAG_UNCLASS_MINERALIZED"] >= P["unclassified_mineralized"]
    assert m.expected["low_support_by_condition"] == {
        c: P["low_support_measured_each"] + P["low_support_indicated_each"]
        for c in ("nholes", "nsamples", "slope_reg", "kriging_eff", "pass_num")
    }
    assert m.expected["low_support_missing_by_condition"]["nsamples"] == 2
    assert m.expected["sentinel_replacements"] == {"CU_PCT": 7, "SG": 3, "NS": 2}
    assert m.expected["coerced_values"] == {"CU_PCT": 1, "SG": 1}


@pytest.mark.parametrize("kind", KINDS)
def test_clean_model_has_no_problems(kind):
    m = make_model(kind, inject=False, rotation_deg=25.0)
    assert m.expected["flags"]["FLAG_DUP_CENTROID"] == 0
    assert m.expected["flags"]["FLAG_OFF_GRID"] == 0
    for name, count in m.expected["flags"].items():
        assert count == 0, name


@pytest.mark.parametrize("kind", KINDS)
def test_planted_bias_is_ten_percent(kind):
    m = make_model(kind, inject=False)
    t = m.truth
    for dom, factor in (("SUL", 1.10), ("OX", 1.0)):
        s = t[(t["domain"] == dom) & t["selected"]]
        w = s["vol"] * s["sg"]
        bm = np.sum(w * s["cu"]) / np.sum(w)
        nn = np.sum(w * s["cu_nn"]) / np.sum(w)
        assert bm / nn == pytest.approx(factor, rel=1e-9)


def test_isatis_selection_and_percent_scale():
    m = make_model("percentage", style="isatis", pct_scale="percent")
    assert m.expected["excluded_rows"]["selection"] == int((~m.truth["selected"]).sum()) > 0
    body = m.csv.splitlines()[len(m.header_lines) :]
    frame = pd.read_csv(pd.io.common.StringIO("\n".join(body)))
    assert frame["Proportion"].max() > 1.05


@pytest.mark.parametrize("rotation", [0.0, 30.0, 135.0])
def test_world_coordinates_rotate_back(rotation):
    origin = (500000.0, 7000000.0, 0.0)
    m = make_model("regular", rotation_deg=rotation, origin=origin, inject=False)
    xl, yl = world_to_local(m.truth["xw"], m.truth["yw"], rotation, pivot=origin[:2])
    assert np.max(np.abs(xl - origin[0] - m.truth["xl"])) < 1e-6
    assert np.max(np.abs(yl - origin[1] - m.truth["yl"])) < 1e-6


def test_repeatable_with_seed():
    a = make_model("sub_blocked", seed=7)
    b = make_model("sub_blocked", seed=7)
    c = make_model("sub_blocked", seed=8)
    assert a.csv == b.csv
    assert a.csv != c.csv


def test_precision_model_builds():
    """0.5 m sub-blocks at northing 7,000,000.25 (used for the false-positive test in M1)."""
    m = make_model(
        "sub_blocked", origin=(500000.0, 7000000.25, 0.0), parent_size=(1.0, 1.0, 1.0), inject=False
    )
    assert m.truth["sx"].min() == 0.5
    assert m.expected["flags"]["FLAG_DUP_CENTROID"] == 0
    assert m.expected["flags"]["FLAG_PARENT_OVERFILL"] == 0


def test_subblock_support_modes():
    for mode, varies in (("parent", False), ("subblock", True)):
        t = make_model("sub_blocked", subblock_support=mode, inject=False).truth
        multi = t[t["multi"] & t["domain"].isin(["OX", "SUL"])]
        sd = multi.groupby(["I", "J", "K", "domain"])["cu"].std()
        assert (sd > 1e-4).any() == varies
