"""Reading the block model: placeholders, text in number columns, types, cache."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from conftest import KIND_STYLES, deep_update
from synthetic import make_model

from qaqc import config, io, schema


def _read(tmp_path, text, use_cache=False, **settings):
    path = tmp_path / "bm.csv"
    path.write_text(text, encoding="utf-8")
    data = deep_update(
        {
            "project_name": "test",
            "paths": {"block_model_csv": str(path), "cache_dir": str(tmp_path / "cache")},
        },
        settings,
    )
    cfg = config.from_dict(data)
    resolved = schema.resolve_file(path, cfg)
    return io.read_block_model(cfg, resolved, use_cache=use_cache)


@pytest.mark.parametrize("kind,style", KIND_STYLES)
def test_placeholder_and_text_counts(run_checks, kind, style):
    m = make_model(kind, style=style)
    ctx = run_checks(m, modules=())
    ingest = ctx.bundle["ingest"]
    assert ingest["sentinel_replacements"] == m.expected["sentinel_replacements"]
    assert ingest["coerced_values"] == m.expected["coerced_values"]
    assert ingest["excluded_rows"] == m.expected["excluded_rows"]
    assert ctx.bundle["inputs"]["block_model"]["rows"] == len(m.truth)


def test_cleaned_values_match_truth(run_checks):
    m = make_model("sub_blocked", style="leapfrog")
    ctx = run_checks(m, modules=())
    t = m.truth.loc[ctx.bm["_ROW"].to_numpy()]
    np.testing.assert_allclose(ctx.bm["x"], t["xw"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ctx.bm["cu"], t["cu"].astype(np.float32), equal_nan=True)
    np.testing.assert_allclose(ctx.bm["sg"], t["sg"].astype(np.float32), equal_nan=True)
    assert (ctx.bm["domain"].astype(object).to_numpy() == t["domain"].to_numpy()).all()


def test_column_types(run_checks):
    ctx = run_checks(make_model("sub_blocked"), modules=())
    bm = ctx.bm
    for c in ("x", "y", "z", "dx", "dy", "dz"):
        assert bm[c].dtype == np.float64, c
    for c in ("cu", "au", "cu_nn", "sg", "nsamples"):
        assert bm[c].dtype == np.float32, c
    assert isinstance(bm["domain"].dtype, pd.CategoricalDtype)
    assert isinstance(bm["classif"].dtype, pd.CategoricalDtype)
    assert bm["_ROW"].dtype == np.int64


HANDMADE = """X,Y,Z,DOMAIN,CU,SG,CLASS
1.5,1.5,-99,1.0,NA,2.5,1
2.5,1.5,-99,1,N/A,null,MEAS
3.5,1.5,-99, ox ,outside,-999,-
4.5,1.5,-99,OX,1e31,2.6,
5.5,1.5,-99,OX,-99.0000000001,abc,2.0
"""


def test_placeholder_rules(tmp_path):
    bm, report = _read(tmp_path, HANDMADE)
    # NA, N/A, outside (any case), 1e31 (beyond the limit) and -99 (within 1e-9) are placeholders
    assert report["sentinel_replacements"] == {"CU": 5, "SG": 2, "CLASS": 1}
    assert report["coerced_values"] == {"SG": 1}
    assert bm["cu"].isna().all()
    # an elevation of -99 m is a real place: coordinates are not checked against the list
    assert (bm["z"] == -99.0).all()
    assert list(bm["domain"].cat.categories) == ["1", "OX"]
    assert bm["domain"].astype(object).tolist() == ["1", "1", "OX", "OX", "OX"]
    assert bm["classif"].astype(object).tolist()[2:] == [np.nan, np.nan, "2"]


def test_blank_and_comment_lines_above_header(tmp_path):
    text = (
        "# exported 2026\n\n# units: m\nX,Y,Z,DOMAIN,CU\n1.5,1.5,1.5,OX,0.5\n2.5,1.5,1.5,OX,0.7\n"
    )
    bm, report = _read(tmp_path, text)
    assert report["rows_read"] == 2
    assert bm["cu"].tolist() == pytest.approx([0.5, 0.7])


def test_mostly_text_column_reported(tmp_path):
    text = "X;Y;Z;DOMAIN;CU\n1.5;1.5;1.5;OX;0,5\n2.5;1.5;1.5;OX;0,7\n3.5;1.5;1.5;OX;1\n"
    _, report = _read(tmp_path, text)
    assert report["mostly_text_columns"] == ["CU"]


def test_cache_is_reused_and_rebuilt(tmp_path):
    bm1, r1 = _read(tmp_path, HANDMADE, use_cache=True)
    bm2, r2 = _read(tmp_path, HANDMADE, use_cache=True)
    assert not r1["from_cache"] and r2["from_cache"]
    pd.testing.assert_frame_equal(bm1, bm2)
    assert r2["sentinel_replacements"] == r1["sentinel_replacements"]
    # different placeholder settings -> new read
    _, r3 = _read(tmp_path, HANDMADE, use_cache=True, sentinels={"numeric": [-999]})
    assert not r3["from_cache"]
    # changed file -> new read
    _, r4 = _read(tmp_path, HANDMADE.replace("2.6", "2.7"), use_cache=True)
    assert not r4["from_cache"]


def test_pandas_reader_gives_same_result(tmp_path, monkeypatch):
    m = make_model("percentage", style="leapfrog")
    bm_fast, r_fast = _read(tmp_path, m.csv)

    def broken(*args, **kwargs):
        raise pa.ArrowInvalid("simulated failure")
        yield  # pragma: no cover

    monkeypatch.setattr(io, "_batches_pyarrow", broken)
    bm_slow, r_slow = _read(tmp_path, m.csv)
    assert r_slow["reader"] == "pandas" and r_fast["reader"] == "pyarrow"
    pd.testing.assert_frame_equal(bm_fast, bm_slow)
    assert r_slow["sentinel_replacements"] == r_fast["sentinel_replacements"]
    assert r_slow["coerced_values"] == r_fast["coerced_values"]


ISATIS_NO_XYZ = """# [ISATIS. ASCII format]
# structure=grid, x_unit="m", y_unit="m", z_unit="m";
# sorting=+X +Y +Z ;
# x0= 105.0 , y0= 205.0 , z0= 52.5 ;
# dx= 10 , dy= 10 , dz= 5 ;
# nx= 3 , ny= 2 , nz= 2 ;
# theta= 0 , phi= 0 , psi= 0
Domain,Kriging Cu,Density
"""


def test_centres_rebuilt_from_isatis_header(tmp_path):
    text = ISATIS_NO_XYZ + "".join(f"OX,{r / 10},2.5\n" for r in range(12))
    bm, _ = _read(tmp_path, text)
    # row 4: i = 1, j = 1, k = 0
    assert (bm.at[4, "x"], bm.at[4, "y"], bm.at[4, "z"]) == (115.0, 215.0, 52.5)
    # row 11: i = 2, j = 1, k = 1
    assert (bm.at[11, "x"], bm.at[11, "y"], bm.at[11, "z"]) == (125.0, 215.0, 57.5)


def test_centres_from_header_need_matching_row_count(tmp_path):
    text = ISATIS_NO_XYZ + "".join(f"OX,{r / 10},2.5\n" for r in range(11))
    with pytest.raises(io.ReadError, match="12 blocks"):
        _read(tmp_path, text)


def test_missing_file(tmp_path):
    cfg = config.from_dict({"project_name": "t", "paths": {"block_model_csv": "nope.csv"}})
    with pytest.raises(io.ReadError, match="not found"):
        io.read_block_model(cfg, None)  # type: ignore[arg-type]
