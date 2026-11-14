"""Flag export and audit bundle tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from synthetic import make_model

from qaqc import export


def _strict_load(path):
    def bad(token):
        raise ValueError(f"non-strict JSON constant {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


@pytest.fixture
def ctx(run_checks):
    c = run_checks(make_model("regular"))
    assert c.flags.any().any(), "synthetic model should have flagged blocks"
    return c


def test_flagged_csv(ctx):
    files = export.write_outputs(ctx)
    assert files[export.BLOCKS_CSV].parent == ctx.cfg.paths.output_dir
    df = pd.read_csv(files[export.BLOCKS_CSV], dtype={"DOMAIN": str})
    mask = ctx.flags.any()
    assert len(df) == int(mask.sum())
    assert df["_ROW"].tolist() == ctx.bm["_ROW"].to_numpy()[mask].tolist()
    assert {"X", "Y", "Z", "DOMAIN", "CLASS_RANK", "FLAG_COUNT", "FLAG_REASONS"} <= set(df.columns)
    assert set(df["CLASS_RANK"]) <= {0, 1, 2, 3, 4}
    names = {
        0: "Unclassified",
        1: "Inferred",
        2: "Indicated",
        3: "Measured",
        4: "Exploration Target",
    }
    assert (df["CLASS"] == df["CLASS_RANK"].map(names)).all()
    np.testing.assert_allclose(df["X"], ctx.bm["x"].to_numpy()[mask])
    for name, n in ctx.flags.counts().items():
        if n:
            assert set(df[name].unique()) <= {0, 1}
            assert int(df[name].sum()) == n
        else:
            assert name not in df.columns
    flag_cols = [c for c in df.columns if c.startswith("FLAG_") and c.isupper()]
    flag_cols = [c for c in flag_cols if c not in ("FLAG_COUNT", "FLAG_REASONS")]
    assert (df[flag_cols].sum(axis=1) == df["FLAG_COUNT"]).all()
    assert (df["FLAG_COUNT"] >= 1).all()
    for _, row in df.head(50).iterrows():
        expected = {c.removeprefix("FLAG_") for c in flag_cols if row[c] == 1}
        assert set(row["FLAG_REASONS"].split(";")) == expected


def test_full_model_flags(run_checks):
    m = make_model("regular")
    ctx = run_checks(m, settings={"export": {"full_model_flags": True}})
    files = export.write_outputs(ctx)
    df = pd.read_parquet(files[export.FULL_FLAGS])
    assert len(df) == len(ctx.bm)
    assert df["_ROW"].tolist() == ctx.bm["_ROW"].tolist()
    for name, n in ctx.flags.counts().items():
        assert int(df[name].sum()) == n


def test_no_full_model_by_default(ctx):
    files = export.write_outputs(ctx)
    assert export.FULL_FLAGS not in files
    assert not (ctx.cfg.paths.output_dir / export.FULL_FLAGS).exists()


def test_tables_written(ctx):
    ctx.tables["my_table"] = pd.DataFrame({"a": [1, 2], "b": [np.nan, 3.5]})
    files = export.write_outputs(ctx)
    got = pd.read_csv(files["my_table.csv"])
    assert got["a"].tolist() == [1, 2]
    assert got["b"].isna().tolist() == [True, False]


def test_bundle_strict_json(ctx):
    ctx.bundle["geometry"]["nan_probe"] = float("nan")
    report = ctx.cfg.paths.output_dir / "BM_QAQC_Report.html"
    files = export.write_outputs(ctx, report_path=report)
    b = _strict_load(files[export.BUNDLE_JSON])
    assert b["schema_version"] == export.SCHEMA_VERSION
    assert b["geometry"]["nan_probe"] is None
    for key in ("metadata", "schema", "grid", "warnings", "ingestion", "geometry", "density"):
        assert key in b
    for key in ("grades", "classification", "support"):
        assert key not in b  # modules not run
    md = b["metadata"]
    assert md["project_name"] == ctx.cfg.project_name
    assert md["inputs"][0]["role"] == "block_model" and md["inputs"][0]["sha256"]
    assert md["config_snapshot"] and md["qaqc_version"] and "export" in md["timings_s"]
    fl = b["flags"]
    assert fl["total_flagged"] == int(ctx.flags.any().sum())
    cat = {c["name"]: c for c in fl["catalog"]}
    assert {n: c["count"] for n, c in cat.items()} == ctx.flags.counts()
    assert all(c["description"] and c["leapfrog_filter"] == f"{n} = 1" for n, c in cat.items())
    assert set(fl["exports"]) == set(files)
    assert "BM_QAQC_Report.html" in fl["exports"]
    assert files["BM_QAQC_Report.html"] == report


def test_filter_template(run_checks):
    ctx = run_checks(
        make_model("regular"), settings={"report": {"leapfrog_filter_template": "[{flag}] == 1"}}
    )
    cat = export.flag_catalog(ctx.flags, ctx.cfg.report.leapfrog_filter_template, len(ctx.bm))
    assert all(c["leapfrog_filter"] == f"[{c['name']}] == 1" for c in cat)


def test_locked_file_message(ctx, monkeypatch):
    def locked(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(export, "_open_for_write", locked)
    with pytest.raises(export.ExportError, match=r"flagged_blocks_leapfrog\.csv.*Close it"):
        export.write_outputs(ctx)


@pytest.mark.parametrize("kind", ["sub_blocked", "percentage"])
def test_other_model_types(run_checks, kind):
    ctx = run_checks(make_model(kind))
    files = export.write_outputs(ctx)
    df = pd.read_csv(files[export.BLOCKS_CSV])
    assert len(df) == int(ctx.flags.any().sum())
    _strict_load(files[export.BUNDLE_JSON])
