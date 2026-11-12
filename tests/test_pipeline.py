"""Whole run: pipeline order and the `python -m qaqc run` command."""

from __future__ import annotations

import json

import pytest
from synthetic import make_model

from qaqc import config, pipeline
from qaqc.__main__ import main
from qaqc.context import prepare
from qaqc.demo import make_demo_project


@pytest.fixture
def project(tmp_path):
    return make_demo_project(tmp_path / "demo", kind="sub_blocked", style="datamine")


def test_run_project_writes_outputs(project):
    ctx, files = pipeline.run_project(project, use_cache=False)
    for key in pipeline.MODULES:
        assert key in ctx.bundle, key
    assert "flagged_blocks_leapfrog.csv" in files
    assert "composite_domain_mismatch.csv" in files
    bundle = json.loads(files["audit_bundle.json"].read_text(encoding="utf-8"))
    for key in ("geometry", "density", "grades", "classification", "support", "composites"):
        assert key in bundle
    roles = [i["role"] for i in bundle["metadata"]["inputs"]]
    assert roles == ["block_model", "composites"]


def test_run_checks_order_and_geometry_first(tmp_path):
    m = make_model("regular", seed=3)
    m.write(tmp_path)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    ctx = prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx, ["grades"])  # geometry is added automatically
    assert "geometry" in ctx.bundle and "grades" in ctx.bundle
    with pytest.raises(ValueError, match="Unknown"):
        pipeline.run_checks(ctx, ["nope"])


def test_cli_run(project, capsys):
    assert main(["run", str(project), "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "blocks checked" in out and "FLAG_DUP_CENTROID" in out
    assert "audit_bundle.json" in out


def test_cli_run_bad_config(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("project_name: x\npaths: {block_model_csv: x.csv}\nmodle: {}\n")
    assert main(["run", str(bad)]) == 2
    assert "not recognised" in capsys.readouterr().err
