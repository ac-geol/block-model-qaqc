"""Composites, smoothing and swaths (spec 7.7) against the synthetic models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import KIND_STYLES
from synthetic import make_model

from qaqc import config, schema
from qaqc.composites_io import resolve
from qaqc.context import jsonable
from qaqc.modules import composites


def _run(run_checks, m, **kwargs):
    ctx = run_checks(m, modules=("geometry",), **kwargs)
    composites.run(ctx)
    return ctx


def _detectable_mismatches(m) -> int:
    """Planted domain mismatches the check can see.

    In a percentage model a composite only counts as a mismatch when its domain is not among
    the rows of the parent cell it sits in (DECISIONS.md). A planted mismatch in a cell that
    holds both OX and SUL rows cannot be seen.
    """
    c = m.composites_truth
    planted = c[c["true_domain"].notna() & (c["true_domain"] != c["label"])]
    if m.kind != "percentage":
        return len(planted)
    t = m.truth[m.truth["selected"] & ~m.truth["is_air"]]
    dx, dy, dz = m.grid["parent_size"]
    cells = t.groupby(["I", "J", "K"])["domain"].agg(set).to_dict()
    n = 0
    for xl, yl, zl, label in zip(
        planted["xl"], planted["yl"], planted["zl"], planted["label"], strict=True
    ):
        key = (int(xl // dx), int(yl // dy), int(zl // dz))
        n += label not in cells[key]
    return n


@pytest.mark.parametrize(("kind", "style"), KIND_STYLES)
def test_flags_match_expectations(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = _run(run_checks, m)
    exp = m.expected["composites"]
    flags = ctx.bundle["composites"]["flags"]
    assert flags["COMP_OUTSIDE_MODEL"] == exp["COMP_OUTSIDE_MODEL"]
    assert flags["COMP_DOMAIN_MISMATCH"] == _detectable_mismatches(m)
    if kind != "percentage":
        assert flags["COMP_DOMAIN_MISMATCH"] == exp["COMP_DOMAIN_MISMATCH"]
        # the very same composites are flagged, not just the same number
        truth = m.composites_truth
        out = truth["true_domain"].isna().to_numpy()
        assert (ctx.comp_flags.get("COMP_OUTSIDE_MODEL") == out).all()
        mm = ~out & (truth["true_domain"] != truth["label"]).to_numpy()
        assert (ctx.comp_flags.get("COMP_DOMAIN_MISMATCH") == mm).all()
    for el in ("cu", "au"):
        name = f"FLAG_ABOVE_COMP_MAX_{el.upper()}"
        assert flags[name] == m.expected["flags"][name]
        assert ctx.flags.counts()[name] == m.expected["flags"][name]


def test_percentage_mismatch_rule_is_documented_deviation():
    """Under the proposed percentage rule at least one planted mismatch is invisible."""
    m = make_model("percentage")
    assert _detectable_mismatches(m) < m.expected["composites"]["COMP_DOMAIN_MISMATCH"]


def test_mismatch_table_and_matrix(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m)
    b = ctx.bundle["composites"]
    table = ctx.tables["composite_domain_mismatch"]
    assert len(table) == b["outside_count"] + b["mismatch_count"]
    assert {"x", "y", "z", "hole_id", "composite_domain", "block_domain", "cu", "au"} <= set(
        table.columns
    )
    assert table["COMP_OUTSIDE_MODEL"].sum() == b["outside_count"]
    assert set(table.loc[table["COMP_OUTSIDE_MODEL"] == 1, "block_domain"]) == {"(outside model)"}
    mism = table[table["COMP_DOMAIN_MISMATCH"] == 1]
    assert (mism["composite_domain"] != mism["block_domain"]).all()
    matrix = {(r["composite_domain"], r["block_domain"]): r["count"] for r in b["domain_matrix"]}
    assert sum(matrix.values()) == b["rows"]
    # planted swaps: OX <-> SUL
    assert sum(v for (c, bd), v in matrix.items() if c != bd and bd != "(outside model)") == 4
    assert matrix[("SUL", "(outside model)")] == 3


@pytest.mark.parametrize("kind", ["regular", "sub_blocked", "percentage"])
def test_cu_bias_in_sul_detected(run_checks, kind):
    m = make_model(kind, style="datamine" if kind != "percentage" else "generic")
    ctx = _run(run_checks, m)
    rows = {(r["domain"], r["element"]): r for r in ctx.bundle["composites"]["bias_table"]}
    sul = rows[("SUL", "cu")]
    assert sul["status"] in ("warn", "fail")
    assert sul["diff_vs_nn_pct"] == pytest.approx(10.0, abs=1.5)
    assert rows[("OX", "cu")]["status"] == "ok"
    assert rows[("SUL", "au")]["status"] == "ok"
    assert sul["composite_count"] > 100
    assert sul["variance_ratio_bm_nn"] is not None and sul["variance_ratio_bm_nn"] < 1.5
    msgs = [w for w in ctx.warnings if w.module == "composites" and "CU in SUL" in w.message]
    assert msgs


def test_bias_status_follows_thresholds(run_checks):
    m = make_model("regular")
    ctx = _run(
        run_checks, m, settings={"composites": {"bias_warn_pct": 20.0, "bias_fail_pct": 30.0}}
    )
    rows = {(r["domain"], r["element"]): r for r in ctx.bundle["composites"]["bias_table"]}
    assert rows[("SUL", "cu")]["status"] == "ok"
    ctx = _run(
        run_checks, m, settings={"composites": {"bias_warn_pct": 5.0, "bias_fail_pct": 50.0}}
    )
    rows = {(r["domain"], r["element"]): r for r in ctx.bundle["composites"]["bias_table"]}
    assert rows[("SUL", "cu")]["status"] == "warn"


def test_uncapped_composites_raise_error_warning(run_checks):
    m = make_model("regular", composites_capped=False)
    ctx = _run(run_checks, m)
    errors = [w for w in ctx.warnings if w.module == "composites" and w.severity == "error"]
    assert any("composites uncapped: bias results may be misleading" in w.message for w in errors)
    b = ctx.bundle["composites"]
    assert b["uncapped_elements"] == ["au", "cu"]
    assert b["capping"]["cu"]["method"] == "uncapped"
    for el in ("cu", "au"):
        name = f"FLAG_ABOVE_COMP_MAX_{el.upper()}"
        assert b["flags"][name] == m.expected["flags"][name]


def test_capped_composites_have_no_uncapped_warning(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m)
    assert not [w for w in ctx.warnings if "uncapped" in w.message]
    cap = ctx.bundle["composites"]["capping"]
    assert cap["cu"]["method"] == "capped_column"
    assert cap["cu"]["n_capped"] >= 3  # the 3 planted 25 % outliers are capped at 5 %
    assert cap["cu"]["max_used"] <= 8.0


def test_config_caps_used_when_no_capped_column(run_checks):
    m = make_model("regular", composites_capped=False)
    caps = {"cu": {"OX": 5.0, "SUL": 8.0}, "au": {"OX": 5.0, "SUL": 20.0}}
    ctx = _run(run_checks, m, settings={"composites": {"caps": caps}})
    b = ctx.bundle["composites"]
    assert b["capping"]["cu"]["method"] == "config_caps"
    assert b["uncapped_elements"] == []
    assert not [w for w in ctx.warnings if "uncapped" in w.message]
    truth = m.composites_truth
    want = np.minimum(truth["cu"], truth["label"].map(caps["cu"]))  # same as the capped column
    np.testing.assert_allclose(ctx.comps["_USED_cu"].to_numpy(dtype=np.float64), want, rtol=1e-6)
    # blocks above the highest *capped* composite: the 25 % outliers no longer set the maximum
    uncapped = _run(run_checks, make_model("regular", composites_capped=False))
    assert (
        b["flags"]["FLAG_ABOVE_COMP_MAX_CU"]
        > uncapped.bundle["composites"]["flags"]["FLAG_ABOVE_COMP_MAX_CU"]
    )
    assert ctx.comps["_USED_cu"].max() <= 8.0


def test_no_composites_file_skips(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m, settings={"paths": {"composites_csv": None}})
    b = ctx.bundle["composites"]
    assert b["loaded"] is False
    assert "skipped" in b
    assert ctx.comps is None and ctx.comp_flags is None
    assert any(w.severity == "info" and w.module == "composites" for w in ctx.warnings)
    assert "composite_domain_mismatch" not in ctx.tables


def test_nn_uses_model_column_when_present(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m)
    b = ctx.bundle["composites"]
    assert b["nn_source"]["cu"].startswith("model column")
    got = ctx.bm["_NN_CU"]
    assert got.dtype == np.float32
    assert np.array_equal(got.to_numpy(), ctx.bm["cu_nn"].to_numpy(), equal_nan=True)


def test_nn_computed_without_model_column(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, modules=("geometry",))
    model_nn = ctx.bm["cu_nn"].to_numpy().copy()
    ctx.bm = ctx.bm.drop(columns=["cu_nn", "au_nn"])
    ctx.schema.elements["cu"].nn_col = None
    ctx.schema.elements["au"].nn_col = None
    composites.run(ctx)
    b = ctx.bundle["composites"]
    assert "computed" in b["nn_source"]["cu"]
    got = ctx.bm["_NN_CU"]
    assert got.dtype == np.float32
    air = ~ctx.analysis_mask
    assert np.isnan(got.to_numpy()[air]).all()
    # the generator builds its NN column the same way, so nearly every block agrees
    both = ~air & np.isfinite(model_nn)
    close = np.isclose(got.to_numpy()[both], model_nn[both], rtol=1e-5)
    assert close.mean() > 0.97
    # blocks of domains with no composites have no NN value
    waste = ctx.bm["domain"].isin(["0", "OVB"]).to_numpy()
    assert np.isnan(got.to_numpy()[waste]).all()


def test_grade_tonnage_curves(run_checks):
    steps = 12
    m = make_model("regular")
    ctx = _run(run_checks, m, settings={"reporting": {"gt_curve_steps": steps}})
    gt = ctx.bundle["composites"]["gt_curves"]
    assert {"OX|cu", "SUL|cu", "(all mineralized)|cu", "SUL|au"} <= set(gt)
    assert "OX|au" not in gt  # gold is only estimated in SUL
    for key, c in gt.items():
        n = len(c["cutoffs"])
        assert 2 <= n <= steps + 1
        assert all(len(c[k]) == n for k in ("bm_tonnes", "bm_grade", "nn_tonnes", "nn_grade"))
        assert np.all(np.diff(c["cutoffs"]) > 0), key
        for t in ("bm_tonnes", "nn_tonnes"):
            assert np.all(np.diff(c[t]) <= 1e-9), (key, t)  # non-increasing with cutoff
        # the grade above a higher cutoff is never lower
        assert np.all(np.diff(c["bm_grade"]) >= -1e-9), key
        assert c["bm_tonnes"][0] == pytest.approx(c["nn_tonnes"][0])  # lowest cutoff = all


def test_grade_tonnage_matches_hand_calculation(run_checks):
    m = make_model("regular", inject=False)
    ctx = _run(run_checks, m)
    c = ctx.bundle["composites"]["gt_curves"]["SUL|cu"]
    bm = ctx.bm
    sel = (bm["domain"] == "SUL").to_numpy() & ctx.analysis_mask
    g = bm["cu"].to_numpy(dtype=np.float64)[sel]
    nn = bm["_NN_CU"].to_numpy(dtype=np.float64)[sel]
    t = bm["_TONNES"].to_numpy()[sel]
    i = len(c["cutoffs"]) // 2
    cut = c["cutoffs"][i]
    above = g >= cut
    assert c["bm_tonnes"][i] == pytest.approx(t[above].sum())
    assert c["bm_grade"][i] == pytest.approx((t[above] * g[above]).sum() / t[above].sum())
    above = nn >= cut
    assert c["nn_tonnes"][i] == pytest.approx(t[above].sum())


@pytest.mark.parametrize("kind", ["regular", "sub_blocked", "percentage"])
def test_swath_tonnes_sum_to_domain_tonnes(run_checks, kind):
    m = make_model(kind)
    ctx = _run(run_checks, m)
    swaths = ctx.bundle["composites"]["swaths"]
    bm = ctx.bm
    dom = bm["domain"].astype(object).to_numpy()
    tonnes = bm["_TONNES"].to_numpy(dtype=np.float64)
    ok = ctx.analysis_mask
    assert {f"SUL|au|{a}" for a in "xyz"} <= set(swaths)
    assert not any(k.startswith("OX|au") for k in swaths)
    for key, bins in swaths.items():
        d, _el, axis = key.split("|")
        total = np.nansum(tonnes[ok & (dom == d)])
        assert sum(b["tonnes"] for b in bins) == pytest.approx(total, rel=1e-9)
        # bins are contiguous and as wide as parent size x multiplier
        a = "xyz".index(axis)
        width = m.grid["parent_size"][a] * (2, 2, 1)[a]
        assert all(b["bin_to"] - b["bin_from"] == pytest.approx(width) for b in bins)
        assert all(
            bins[i]["bin_to"] == pytest.approx(bins[i + 1]["bin_from"])
            for i in range(len(bins) - 1)
        )
        for b in bins:
            assert b["low_support"] == (b["tonnes"] < 0.005 * total)


def test_swath_composite_counts_and_low_support(run_checks):
    m = make_model("regular")
    ctx = _run(run_checks, m, settings={"swath": {"min_bin_tonnage_pct": 20.0}})
    b = ctx.bundle["composites"]
    bins = b["swaths"]["SUL|cu|x"]
    # composites with a SUL label that sit inside the model's SUL x range
    c = ctx.comps
    sel = (c["domain"] == "SUL").to_numpy() & np.isfinite(c["_USED_cu"].to_numpy())
    x = c["x"].to_numpy()[sel]
    lo, hi = bins[0]["bin_from"], bins[-1]["bin_to"]
    assert sum(b_["comp_count"] for b_ in bins) == int(((x >= lo) & (x < hi)).sum())
    assert all(x_["low_support"] for x_ in bins)  # no x bin holds 20 % of the domain
    # low-support bins are still reported (greyed out by the report, not dropped)
    assert all(x["tonnes"] >= 0 for x in bins)


def test_bundle_is_json_safe(run_checks):
    m = make_model("sub_blocked", style="leapfrog")
    ctx = _run(run_checks, m)
    text = json.dumps(jsonable(ctx.bundle["composites"]), allow_nan=False)
    assert json.loads(text)["loaded"] is True
    assert ctx.bundle["composites"]["input"]["sha256"]


@pytest.mark.parametrize("style", ["generic", "datamine", "isatis", "leapfrog"])
def test_composite_columns_resolve(style):
    m = make_model("regular", style=style)
    cfg = config.from_dict(m.config_dict(), base_dir=".")
    cs = resolve(schema.prescan(m.composites_csv), cfg, block_elements={"cu", "au"})
    cols = m.composite_columns
    for key in ("x", "y", "z", "domain", "length", "hole_id"):
        assert cs.structural[key] == cols[key]
    assert cs.structural["from_depth"] == cols["from"]
    assert cs.structural["to_depth"] == cols["to"]
    for el in ("cu", "au"):
        assert cs.elements[el].raw == cols[el]
        assert cs.elements[el].capped == cols[f"{el}_cap"]


def test_composites_without_capped_columns_and_without_length(tmp_path):
    m = make_model("regular", composites_capped=False)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    cs = resolve(schema.prescan(m.composites_csv), cfg, block_elements={"cu", "au"})
    assert cs.elements["cu"].capped is None
    assert cs.elements["cu"].raw == "CU_PCT"


def test_missing_required_composite_column_is_an_error(tmp_path):
    m = make_model("regular")
    text = m.composites_csv.replace("DOMAIN", "LITHO", 1)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    with pytest.raises(KeyError, match="domain"):
        resolve(schema.prescan(text), cfg, block_elements={"cu", "au"})


def test_length_weighting_and_unit_conversion(run_checks, tmp_path):
    """A composites file in ppm is converted to the model's pct; lengths weight the mean."""
    m = make_model("regular", inject=False)
    head, *rows = m.composites_csv.splitlines()
    cols = head.split(",")
    i_cu = cols.index("CU_PCT")
    new_head = head.replace("CU_PCT", "CU_PPM")
    out = [new_head]
    for r in rows:
        f = r.split(",")
        f[i_cu] = repr(float(f[i_cu]) * 10_000.0) if f[i_cu] else ""
        out.append(",".join(f))
    m.composites_csv = "\n".join(out) + "\n"
    ctx = _run(run_checks, m)
    u = ctx.bundle["composites"]["units"]["cu"]
    assert (u["file_unit"], u["model_unit"]) == ("ppm", "pct")
    truth = m.composites_truth
    np.testing.assert_allclose(
        ctx.comps["cu"].to_numpy(dtype=np.float64), truth["cu"].to_numpy(), rtol=1e-5
    )
    assert ctx.bundle["composites"]["length_source"] == "column 'LENGTH'"
