"""Shared helpers: unit conversion, contained metal, class ranks, applicable domains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import make_model

from qaqc import config, units
from qaqc.domains import applicable_domains, class_ranks, waste_mask, weights


def test_unit_factors():
    assert units.factor("pct", "ppm") == 10_000
    assert units.factor("ppm", "gpt") == 1
    assert units.factor("gpt", "ppb") == 1_000
    assert units.convert(0.5, "pct", "ppm") == 5_000
    assert units.convert(None, "pct", "ppm") is None
    with pytest.raises(ValueError):
        units.factor("pct", "lbs")


def test_ounces_per_ton():
    # 1 troy oz per short ton = 31.1034768 g / 0.90718474 t = 34.2857 g/t
    assert units.factor("opt", "gpt") == pytest.approx(34.2857142857)
    assert units.convert(1.0, "gpt", "opt") == pytest.approx(1 / 34.2857142857)
    assert units.convert(0.5, "opt", "ppb") == pytest.approx(17142.857142857)


def test_contained_metal():
    # base metal: tonnes (or short tons) of metal, whatever the grade unit
    assert units.contained_metal([100.0], [2.0], "pct", "pct")[0] == pytest.approx(2.0)
    assert units.contained_metal([100.0], [20_000.0], "ppm", "pct")[0] == pytest.approx(2.0)
    # precious metal in troy ounces
    assert units.contained_metal([31.1034768], [1.0], "gpt", "gpt")[0] == pytest.approx(1.0)
    assert units.contained_metal([31.1034768], [1000.0], "ppb", "gpt")[0] == pytest.approx(1.0)
    # imperial: 1,000 short tons at 0.25 oz/ton = 250 oz
    short = units.TONNES_PER_SHORT_TON
    assert units.contained_metal([1000.0], [0.25], "opt", "gpt", short)[0] == pytest.approx(250)
    # g/t grades on short-ton tonnage: 1,000 short tons = 907.18 t at 1 g/t = 29.17 oz
    assert units.contained_metal([1000.0], [1.0], "gpt", "gpt", short)[0] == pytest.approx(
        907.18474 / 31.1034768
    )
    assert np.isnan(units.contained_metal([1.0], [np.nan], "pct", "pct")[0])
    assert units.metal_unit("pct") == "t" and units.metal_unit("gpt") == "oz"
    assert units.metal_unit("pct", "imperial") == "short tons"


def _cfg(**extra):
    return config.from_dict({"project_name": "t", "paths": {"block_model_csv": "x.csv"}, **extra})


def test_class_ranks_and_unmapped():
    bm = pd.DataFrame({"classif": pd.Categorical(["1", "2", "3", "PROV", None, "PROV"])})
    ranks, unmapped = class_ranks(bm, _cfg())
    assert ranks.tolist() == [3, 2, 1, 0, 0, 0]
    assert unmapped == {"PROV": 2}


def test_class_ranks_without_column():
    ranks, unmapped = class_ranks(pd.DataFrame({"x": [1, 2]}), _cfg())
    assert ranks.tolist() == [0, 0] and unmapped == {}


def test_waste_mask():
    bm = pd.DataFrame({"domain": pd.Categorical(["OX", "0", "OVB", None])})
    assert waste_mask(bm, _cfg()).tolist() == [False, True, True, False]


def test_applicable_domains_and_weights(run_checks):
    m = make_model("regular", seed=1)
    ctx = run_checks(m, modules=("geometry",))
    assert applicable_domains(ctx, "cu") == m.expected["applicable_domains"]["cu"]
    assert applicable_domains(ctx, "au") == m.expected["applicable_domains"]["au"]
    w, kind = weights(ctx)
    assert kind == "tonnes" and len(w) == len(ctx.bm)
    cu = ctx.schema.elements["cu"]
    assert units.element_threshold(ctx.cfg, cu, "ceiling") == pytest.approx(34.6)
    ppm = cu.__class__(**{**cu.__dict__, "unit": "ppm"})
    assert units.element_threshold(ctx.cfg, ppm, "cutoff") == pytest.approx(5000)


def test_weights_need_geometry(run_checks):
    ctx = run_checks(make_model("regular", seed=1), modules=())
    with pytest.raises(RuntimeError, match="geometry"):
        weights(ctx)
