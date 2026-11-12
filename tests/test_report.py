"""HTML report and figures (spec 11 and 12)."""

from __future__ import annotations

import html as htmllib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import deep_update
from synthetic import make_model

from qaqc import config, figures, pipeline, report
from qaqc.context import AuditContext, prepare
from qaqc.demo import make_demo_project
from qaqc.export import ExportError

PLOTLY_BANNER = "plotly.js v"
HOSTILE = "<script>alert(1)</script>"
KINDS = [("regular", "datamine"), ("sub_blocked", "generic"), ("percentage", "leapfrog")]
META_COLUMNS = ["SVOL", "NSAMP", "NHOLES", "KV", "KE", "SOR", "MINDIST", "AVEDIST"]


def _run(
    folder: Path,
    kind: str = "regular",
    style: str = "datamine",
    settings: dict[str, Any] | None = None,
    edit_model: Any = None,
) -> tuple[AuditContext, Path]:
    """Write a synthetic project, run every check and write the report."""
    m = make_model(kind, style=style)
    m.write(folder)
    if edit_model:
        edit_model(m, folder)
    cfg = config.from_dict(deep_update(m.config_dict(), settings or {}), base_dir=folder)
    ctx = prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx)
    return ctx, report.write_report(ctx)


@pytest.fixture(scope="module")
def reports(tmp_path_factory) -> dict[str, tuple[AuditContext, str]]:
    out = {}
    for kind, style in KINDS:
        ctx, path = _run(tmp_path_factory.mktemp(kind), kind, style)
        out[kind] = (ctx, path.read_text(encoding="utf-8"))
    return out


@pytest.mark.parametrize("kind", [k for k, _ in KINDS])
def test_report_has_every_section(reports, kind):
    ctx, text = reports[kind]
    path = Path(ctx.cfg.paths.output_dir) / report.REPORT_NAME
    assert path.is_file()
    assert text.lstrip().startswith("<!DOCTYPE html>") and text.rstrip().endswith("</html>")
    for title in report.TABS.values():
        assert f"<h2>{htmllib.escape(title)}</h2>" in text
    assert "Status summary" in text and "Flag register" in text
    assert text.count("<section") == text.count("</section>") == len(report.TABS)


@pytest.mark.parametrize("kind", [k for k, _ in KINDS])
def test_inline_plotly_once_and_offline(reports, kind):
    _, text = reports[kind]
    assert text.count(PLOTLY_BANNER) == 1
    assert not re.search(r"<script[^>]+src=", text)
    page = text[text.index("<body>") :]  # the plotly.js library text itself mentions web addresses
    assert 'href="http' not in page and "<link" not in text and "@import" not in text


def test_every_figure_is_drawn_from_json(reports):
    ctx, text = reports["sub_blocked"]
    blocks = re.findall(
        r'<script type="application/json" id="data-(fig\d+)">(.*?)</script>', text, re.S
    )
    assert len(blocks) >= len(["density_box", "classification_bar", "support_box"])
    for _, payload in blocks:
        fig = json.loads(payload)
        assert "data" in fig and "layout" in fig
    for name in ("density_box", "classification_bar", "support_box", "cdf_cu"):
        assert name in ctx.figures
    assert any(n.startswith("swath_") for n in ctx.figures)
    assert any(n.startswith("gt_") for n in ctx.figures)
    # one selector option per swath and per grade-tonnage figure, nothing drawn until chosen
    assert text.count('data-selector="swath"') == 1 and text.count('data-selector="gt"') == 1


def test_cdn_mode_has_no_embedded_plotly(tmp_path):
    _, path = _run(tmp_path, settings={"report": {"plotly_js": "cdn"}})
    text = path.read_text(encoding="utf-8")
    assert PLOTLY_BANNER not in text
    assert (
        len(re.findall(r'<script src="https://cdn\.plot\.ly/plotly-[\d.]+\.min\.js">', text)) == 1
    )
    assert len(text) < 1_000_000


def test_downsample_is_bounded_stratified_and_repeatable():
    labels = np.array(["A"] * 9000 + ["B"] * 900 + ["C"] * 100)
    a = figures.downsample_indices(labels, 500, seed=1)
    assert len(a) <= 500 and len(set(a)) == len(a)
    assert np.array_equal(a, figures.downsample_indices(labels, 500, seed=1))
    assert {"A", "B", "C"} <= set(labels[a])
    assert (labels[a] == "C").sum() >= 10  # a small domain stays visible
    assert len(figures.downsample_indices(labels, 20000)) == len(labels)  # nothing to cut


def test_scatter_honours_max_scatter_points(tmp_path):
    ctx, _ = _run(
        tmp_path, "sub_blocked", "datamine", settings={"report": {"max_scatter_points": 300}}
    )
    fig = ctx.figures["density_vs_grade"]
    n = sum(len(t.x) for t in fig.data)
    assert 0 < n <= 300
    ctx2, _ = _run(tmp_path / "b", "sub_blocked", "datamine")
    assert sum(len(t.x) for t in ctx2.figures["density_vs_grade"].data) > n


def test_no_composites_renders_with_skipped_notes(tmp_path):
    ctx, path = _run(tmp_path, settings={"paths": {"composites_csv": None}})
    text = path.read_text(encoding="utf-8")
    assert "Skipped because no composites file configured" in text
    assert not any(n.startswith(("swath_", "gt_")) for n in ctx.figures)
    assert "density_box" in ctx.figures


def _drop_columns(model, folder: Path) -> None:
    path = folder / "block_model.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.drop(columns=["RESCAT", *META_COLUMNS])
    df.to_csv(path, index=False)


def test_missing_classification_and_metadata_render(tmp_path):
    ctx, path = _run(
        tmp_path,
        settings={"paths": {"composites_csv": None}},
        edit_model=_drop_columns,
    )
    text = path.read_text(encoding="utf-8")
    assert "Skipped because no classification column" in text
    assert "Skipped because no estimation detail columns" in text
    assert "classification_bar" not in ctx.figures and "support_box" not in ctx.figures
    assert report.TABS["support"] in text


def test_modules_not_run_still_render(tmp_path):
    m = make_model("regular", style="datamine")
    m.write(tmp_path)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    ctx = prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx, ["geometry"])
    text = report.write_report(ctx).read_text(encoding="utf-8")
    assert text.count("Not run: this check was not part of this run.") >= 4
    assert "Flag register" in text


def _hostile_domain(model, folder: Path) -> None:
    for name, col in (("block_model.csv", "KZONE"), ("composites.csv", None)):
        path = folder / name
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        cols = [col] if col else [c for c in df.columns if c.upper() in ("DOMAIN", "KZONE", "ZONE")]
        for c in cols:
            df[c] = df[c].replace({"SUL": HOSTILE})
        df.to_csv(path, index=False)


def test_text_from_the_data_is_escaped(tmp_path):
    ctx, path = _run(tmp_path, "regular", "datamine", edit_model=_hostile_domain)
    ctx.warn("warning", "<b>module</b>", HOSTILE)
    text = report.render(ctx)
    lowered = text.lower()
    assert "<script>alert(1)" not in lowered
    assert "<b>module</b>" not in lowered
    assert "&lt;script&gt;alert(1)" in lowered
    # figure data is JSON inside script tags: it must not contain markup either
    for _, payload in re.findall(r'id="data-(fig\d+)">(.*?)</script>', text, re.S):
        assert "<" not in payload
        json.loads(payload)
    assert path.read_text(encoding="utf-8").lower().count("<script>alert(1)") == 0


def test_project_name_is_escaped(tmp_path):
    ctx, _ = _run(tmp_path, settings={"project_name": "A & B <i>mine</i>"})
    text = report.render(ctx)
    assert "A &amp; B &lt;i&gt;mine&lt;/i&gt;" in text
    assert "<i>mine</i>" not in text


def test_locked_file_gives_plain_message(tmp_path, monkeypatch):
    m = make_model("regular", style="datamine")
    m.write(tmp_path)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    ctx = prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx, ["geometry"])

    def locked(*args, **kwargs):
        raise PermissionError

    monkeypatch.setattr(report, "open", locked, raising=False)
    with pytest.raises(ExportError, match="open in a browser"):
        report.write_report(ctx)


def test_flag_register_does_not_need_export(reports):
    ctx, text = reports["regular"]
    out = Path(ctx.cfg.paths.output_dir)
    assert not (out / "audit_bundle.json").exists()  # export has not run
    for name in ctx.flags.info():
        assert name in text
    assert "FLAG_DUP_CENTROID = 1" in text  # default Leapfrog filter template


def test_pipeline_writes_report_before_export(tmp_path):
    project = make_demo_project(tmp_path / "demo", kind="sub_blocked", style="datamine")
    ctx, files = pipeline.run_project(project, use_cache=False)
    assert report.REPORT_NAME in files
    bundle = json.loads(files["audit_bundle.json"].read_text(encoding="utf-8"))
    assert "figures" in bundle["metadata"]["timings_s"]
    text = files[report.REPORT_NAME].read_text(encoding="utf-8")
    # the uncapped-composites badge and the bias table come from the composite results
    assert "Composites" in text and "Model against composites and nearest neighbour" in text


def test_uncapped_composites_show_red_badge(tmp_path):
    m = make_model("regular", style="datamine", composites_capped=False)
    m.write(tmp_path)
    cfg = config.from_dict(m.config_dict(), base_dir=tmp_path)
    ctx = prepare(cfg, use_cache=False)
    pipeline.run_checks(ctx)
    text = report.write_report(ctx).read_text(encoding="utf-8")
    assert '<div class="badge bad">Composites are uncapped' in text
