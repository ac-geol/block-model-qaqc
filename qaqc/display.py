"""Tables and messages for the notebook (``notebooks/block_model_qaqc.ipynb``).

The notebook only calls the functions here and in the check modules; it holds no analysis
logic. Every function returns a pandas DataFrame (or a styled one) built from results the
checks have already worked out, so the same numbers appear in the notebook and, later, in the
report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from qaqc import config, schema
from qaqc.context import prepare
from qaqc.grid import GridError
from qaqc.io import ReadError

if TYPE_CHECKING:
    from pandas.io.formats.style import Styler

    from qaqc.context import AuditContext

# Problems a user can fix by editing the project YAML or the input files.
USER_ERRORS = (
    config.ConfigError,
    schema.SchemaError,
    schema.MissingColumnError,
    GridError,
    ReadError,
    PermissionError,
    FileNotFoundError,
)

SEVERITY_COLOURS = {"error": "#f8d7da", "warning": "#fff3cd", "info": ""}
SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


class StopRun(Exception):
    """A problem the user needs to fix. In Jupyter it shows as a short message, not a
    traceback."""

    def __init__(self, step: str, message: str) -> None:
        super().__init__(message)
        self.step = step
        self.message = message

    def _render_traceback_(self) -> list[str]:  # used by IPython / Jupyter instead of a traceback
        return [
            f"STOPPED at step '{self.step}':",
            "",
            *self.message.strip().splitlines(),
            "",
            "Fix the problem (usually in the project YAML file), then run the notebook again "
            "from the top.",
        ]


def setup_notebook() -> None:
    """Notebook display settings.

    The checks' messages are shown in tables, so the log is silenced (otherwise each message
    would also appear as red text under the cell). Real failures still raise exceptions.
    """
    logging.getLogger("qaqc").setLevel(logging.CRITICAL)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_colwidth", 120)


def _message(err: BaseException) -> str:
    # KeyError wraps its message in quotes; take the plain text instead.
    if isinstance(err, KeyError) and err.args:
        return str(err.args[0])
    return str(err)


# ---------------------------------------------------------------------------------------------
# steps (each wraps a package call so user errors show as a short message)
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path) -> config.ProjectConfig:
    """Load and check the project YAML file."""
    try:
        return config.load(path)
    except USER_ERRORS as err:
        raise StopRun("settings", _message(err)) from None


def match_columns(cfg: config.ProjectConfig) -> schema.ResolvedSchema:
    """Match the block model columns to standard names (does not read the data)."""
    try:
        return schema.resolve_file(cfg.paths.block_model_csv, cfg)
    except USER_ERRORS as err:
        raise StopRun("column matching", _message(err)) from None


def read_model(cfg: config.ProjectConfig, use_cache: bool = True) -> AuditContext:
    """Read and clean the block model and set up the grid."""
    try:
        return prepare(cfg, use_cache=use_cache)
    except USER_ERRORS as err:
        raise StopRun("reading the block model", _message(err)) from None


# ---------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------


def _table(rows: list[tuple[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Item", "Value"]).set_index("Item")


def _path_status(p: Path | None) -> str:
    if p is None:
        return "not given"
    return f"{p}  ({'found' if Path(p).exists() else 'NOT FOUND'})"


def config_summary(cfg: config.ProjectConfig) -> pd.DataFrame:
    """The main settings, and whether the input files exist."""
    p = cfg.paths
    m = cfg.model
    return _table(
        [
            ("Project", cfg.project_name),
            ("Software", cfg.source_software),
            ("Block model CSV", _path_status(p.block_model_csv)),
            ("Composites CSV", _path_status(p.composites_csv)),
            ("Output folder", str(p.output_dir)),
            ("Model type", m.model_type),
            ("Parent cell size", m.parent_cell_size or "from file"),
            ("Grid origin", m.grid_origin or "from file"),
            ("Elements configured", ", ".join(cfg.elements.items) or "none"),
            ("Column overrides", len(cfg.column_overrides)),
        ]
    )


def unused_columns(s: schema.ResolvedSchema) -> pd.DataFrame:
    """Columns in the file that no check uses, with the reason."""
    return pd.DataFrame(
        [
            {"Column in file": c, "Why it is not used": s.unmatched_reasons.get(c, "")}
            for c in s.unmatched_columns
        ],
        columns=["Column in file", "Why it is not used"],
    )


def schema_notes(s: schema.ResolvedSchema) -> pd.DataFrame:
    """Plain-language notes from column matching and the file header."""
    notes = list(s.notes) + list(s.header.parsed.get("notes", []))
    return pd.DataFrame({"Note": notes})


def file_summary(s: schema.ResolvedSchema) -> pd.DataFrame:
    """What was learned from the top of the file."""
    h = s.header
    grid_values = {k: v for k, v in h.parsed.items() if k != "notes"}
    return _table(
        [
            ("Software", f"{s.software or 'not recognised'} ({h.detection_reason})"),
            ("Header lines above the column names", h.skiprows),
            ("Grid values in the header", grid_values or "none"),
        ]
    )


def reading_summary(ctx: AuditContext) -> pd.DataFrame:
    """Rows read, rows left out and values cleaned up."""
    ing = ctx.bundle["ingest"]
    inp = ctx.bundle["inputs"]["block_model"]
    exc = ing["excluded_rows"]
    sentinels = ing["sentinel_replacements"]
    coerced = ing["coerced_values"]
    return _table(
        [
            ("Rows read", f"{inp['rows']:,}"),
            ("Left out: selection = 0", f"{exc.get('selection', 0):,}"),
            ("Air blocks (grid checks only)", f"{exc.get('air', 0):,}"),
            ("Rows analysed", f"{ing['rows_analysed']:,}"),
            (
                "Placeholder values treated as missing",
                ", ".join(f"{c}: {n:,}" for c, n in sentinels.items()) or "none",
            ),
            (
                "Values that are not numbers",
                ", ".join(f"{c}: {n:,}" for c, n in coerced.items()) or "none",
            ),
            ("Read from cache", "yes" if ing["from_cache"] else "no"),
        ]
    )


def grid_summary(ctx: AuditContext) -> pd.DataFrame:
    """Model grid values and where each came from."""
    g = ctx.grid
    src = g.sources

    def fmt(t: tuple[float, ...] | None) -> str:
        if t is None:
            return "-"
        return ", ".join(f"{v:,.3f}".rstrip("0").rstrip(".") for v in t)

    rows = [
        ("Model type", g.model_type, src.get("model_type", "")),
        ("Parent cell size (X, Y, Z)", fmt(g.parent_size), src.get("parent_size", "")),
        ("Model corner (X, Y, Z)", fmt(g.origin), src.get("origin", "")),
        ("Rotation (degrees)", f"{g.azimuth_deg:g}", src.get("azimuth", "")),
    ]
    if g.min_subblock_size is not None:
        rows.append(("Smallest sub-block", fmt(g.min_subblock_size), ""))
    if g.pct_scale is not None:
        rows.append(("Percentage scale", g.pct_scale, src.get("pct_scale", "")))
    return pd.DataFrame(rows, columns=["Item", "Value", "Where it came from"]).set_index("Item")


def flag_table(ctx: AuditContext, module: str | None = None) -> pd.DataFrame:
    """Flagged blocks per check, with what each flag means."""
    counts = ctx.flags.counts()
    n = max(ctx.flags.n_rows, 1)
    rows = [
        {
            "Flag": name,
            "Blocks flagged": counts[name],
            "% of blocks": round(counts[name] / n * 100, 3),
            "Meaning": info.description,
            "Check": info.module,
        }
        for name, info in ctx.flags.info().items()
        if module is None or info.module == module
    ]
    return pd.DataFrame(
        rows, columns=["Flag", "Blocks flagged", "% of blocks", "Meaning", "Check"]
    ).set_index("Flag")


def geometry_summary(ctx: AuditContext) -> pd.DataFrame:
    """Totals and grid check results."""
    r = ctx.bundle["geometry"]
    box = r.get("bounding_box", {})
    rows: list[tuple[str, Any]] = [
        ("Model type", r["model_type"]),
        ("Parent cells", f"{r['parent_cells']:,}"),
        ("Blocks analysed", f"{r['blocks_analysed']:,}"),
        ("Total volume (m³)", f"{r['total_volume_m3']:,.0f}"),
        ("Total tonnes", f"{r['total_tonnes']:,.0f}"),
        ("Blocks with no tonnes", f"{r['blocks_without_tonnes']:,}"),
        ("Blocks with no coordinates", f"{r['blocks_without_coordinates']:,}"),
    ]
    for axis in ("x", "y", "z"):
        if box.get(axis):
            lo, hi = box[axis]
            rows.append((f"Extent {axis.upper()}", f"{lo:,.1f} to {hi:,.1f}"))
    if "overfill" in r:
        o = r["overfill"]
        rows.append(("Overfilled parent cells", f"{o['parent_cells']:,} ({o['blocks']:,} blocks)"))
    if "percentage" in r:
        p = r["percentage"]
        rows.append(("Percentage out of 0-100%", f"{p['rows_out_of_range']:,} rows"))
        rows.append(("Parent cells over 100%", f"{p['parent_cells_over_100pct']:,}"))
    return _table(rows)


def subblock_support_table(ctx: AuditContext) -> pd.DataFrame:
    """Sub-blocked models: do sub-blocks in one parent cell carry different grades?"""
    sup = ctx.bundle["geometry"].get("subblock_support")
    if not sup:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(sup["elements"], orient="index")
    df.index.name = "Element"
    return df


def wireframe_volume_table(ctx: AuditContext) -> pd.DataFrame:
    """Model volume against wireframe volume per domain (if targets are set)."""
    return pd.DataFrame(ctx.bundle["geometry"].get("wireframe_volumes", []))


def density_summary(ctx: AuditContext) -> pd.DataFrame:
    """Missing and out-of-range density counts."""
    r = ctx.bundle["density"]
    if "skipped" in r:
        return _table([("Skipped", r["skipped"])])
    a = r["anomalies"]
    return _table(
        [
            ("Blocks checked", f"{a['blocks_checked']:,}"),
            ("Waste checked too", "yes" if a["waste_checked"] else "no"),
            ("Missing density", f"{a['missing']:,}"),
            ("Below the allowed range", f"{a['low']:,}"),
            ("Above the allowed range", f"{a['high']:,}"),
        ]
    )


def density_by_domain(ctx: AuditContext) -> pd.DataFrame:
    """Volume-weighted density statistics per domain."""
    rows = ctx.bundle["density"].get("by_domain", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("domain").round(3)


def density_vs_grade(ctx: AuditContext) -> pd.DataFrame:
    """Correlation between density and grade per element and domain."""
    rows = ctx.bundle["density"].get("density_vs_grade", [])
    return pd.DataFrame(rows).round(3) if rows else pd.DataFrame()


def _records(rows: list[dict[str, Any]] | None, index: str | None = None) -> pd.DataFrame:
    """Records (nested dicts allowed) as a table; ``grade.cu`` becomes ``cu grade``."""
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    df.columns = [
        f"{c.split('.', 1)[1]} {c.split('.', 1)[0]}" if "." in c else c for c in df.columns
    ]
    df = df.round(3)
    return df.set_index(index) if index and index in df.columns else df


def skipped_note(ctx: AuditContext, module: str) -> pd.DataFrame:
    """Why a module did not run (empty table if it ran)."""
    r = ctx.bundle.get(module, {})
    if r.get("skipped"):
        return _table([("Skipped", r["skipped"])])
    return pd.DataFrame()


# --- grades ----------------------------------------------------------------------------------


def grades_summary(ctx: AuditContext) -> pd.DataFrame:
    """One row per element: column, unit, where it is estimated and the flag counts."""
    rows = []
    for el, r in ctx.bundle.get("grades", {}).get("elements", {}).items():
        u, f = r["units"], r["flags"]
        rows.append(
            {
                "Element": el.upper(),
                "Column": r["column"],
                "Unit": f"{u['unit']} ({u['unit_source']})"
                + (" - LIKELY WRONG" if u.get("likely_wrong") else ""),
                "Estimated in": ", ".join(r["applicable_domains"]) or "none",
                "Not estimated in": ", ".join(r["not_estimated_domains"]) or "-",
                "Negative": f["NEG"],
                "Above ceiling": f["CEILING"],
                "Unestimated": f["UNEST"],
                "Waste bleed": f["BLEED"],
                "Checks skipped": ", ".join(s["check"] for s in r["skipped_checks"]) or "-",
            }
        )
    return pd.DataFrame(rows).set_index("Element") if rows else pd.DataFrame()


def grades_by_domain(ctx: AuditContext, element: str) -> pd.DataFrame:
    """Grade statistics by domain for one element (tonnes-weighted)."""
    r = ctx.bundle.get("grades", {}).get("elements", {}).get(element.lower())
    return _records(r["by_domain"] if r else None, "domain")


# --- classification --------------------------------------------------------------------------


def classification_summary(ctx: AuditContext) -> pd.DataFrame:
    """Tonnes, grade and metal by class."""
    return _records(ctx.bundle.get("classification", {}).get("by_class"), "class")


def classification_inventory(ctx: AuditContext) -> pd.DataFrame:
    """Domain x class: volume, tonnes, share of the domain, grades and metal."""
    return _records(ctx.bundle.get("classification", {}).get("inventory"))


def unmapped_codes(ctx: AuditContext) -> pd.DataFrame:
    """Class codes not in the rank map (treated as unclassified)."""
    return _records(ctx.bundle.get("classification", {}).get("unmapped_codes"), "code")


def resource_statement(ctx: AuditContext) -> pd.DataFrame:
    """The reproduced resource statement (by class, plus total), if a cutoff is set."""
    rs = ctx.bundle.get("classification", {}).get("resource_statement")
    if not rs:
        return _table([("Resource statement", "not set (reporting: resource_cutoff)")])
    if rs.get("skipped"):
        return _table([("Resource statement", f"skipped: {rs['skipped']}")])
    rows = list(rs["by_class"])
    if rs.get("total"):
        rows.append({"class": "Total", **rs["total"]})
    return _records(rows, "class")


# --- support ---------------------------------------------------------------------------------


def support_low(ctx: AuditContext) -> pd.DataFrame:
    """Measured / Indicated blocks failing each support condition."""
    r = ctx.bundle.get("support", {})
    if r.get("skipped"):
        return skipped_note(ctx, "support")
    rows = [
        {
            "Class": info["class"],
            "Condition": cond,
            "Limit": c["limit"],
            "Blocks failing": c["failed"],
            "Blocks with no value": c["missing"],
        }
        for info in r.get("low_support", {}).get("by_rank", {}).values()
        for cond, c in info["conditions"].items()
    ]
    return pd.DataFrame(rows)


def support_stats(ctx: AuditContext) -> pd.DataFrame:
    """Estimation details (samples, holes, distances, slope ...) by class."""
    rows = [
        {"Class": s["class"], "Detail": d, **v}
        for s in ctx.bundle.get("support", {}).get("stats_by_class", [])
        for d, v in s["details"].items()
    ]
    return pd.DataFrame(rows).round(3).set_index(["Class", "Detail"]) if rows else pd.DataFrame()


def support_passes(ctx: AuditContext) -> pd.DataFrame:
    """Tonnes by class and search pass."""
    pb = ctx.bundle.get("support", {}).get("pass_breakdown")
    return _records(pb.get("by_class") if pb else None)


# --- composites ------------------------------------------------------------------------------


def composites_summary(ctx: AuditContext) -> pd.DataFrame:
    """Composites read, capping, and how many sit outside the model or in the wrong domain."""
    r = ctx.bundle.get("composites", {})
    if not r.get("loaded"):
        return _table([("Composites", r.get("skipped", "not run"))])
    capping = ", ".join(f"{el.upper()}: {c['method']}" for el, c in r["capping"].items())
    return _table(
        [
            ("Composites read", f"{r['input']['rows']:,}"),
            ("Capping", capping or "-"),
            ("Outside the model", f"{r['outside_count']:,}"),
            ("Domain differs from the block", f"{r['mismatch_count']:,}"),
            ("No coordinates", f"{r['no_coordinates_count']:,}"),
            ("Nearest-neighbour benchmark", str(r.get("nn_source"))),
        ]
    )


def bias_table(ctx: AuditContext) -> Styler | pd.DataFrame:
    """Block model vs composites vs nearest-neighbour, coloured by status."""
    df = _records(ctx.bundle.get("composites", {}).get("bias_table"))
    if df.empty:
        return df
    colours = {"fail": SEVERITY_COLOURS["error"], "warn": SEVERITY_COLOURS["warning"]}

    def colour(row: pd.Series) -> list[str]:
        bg = colours.get(row["status"], "")
        return [f"background-color: {bg}" if bg else "" for _ in row]

    return df.style.apply(colour, axis=1).format(precision=3).hide(axis="index")


def domain_matrix(ctx: AuditContext) -> pd.DataFrame:
    """Composite domain (rows) against the domain of the block it sits in (columns)."""
    rows = ctx.bundle.get("composites", {}).get("domain_matrix")
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .pivot_table(index="composite_domain", columns="block_domain", values="count", fill_value=0)
        .astype(int)
    )


def write_outputs(ctx: AuditContext) -> pd.DataFrame:
    """Write the HTML report and the export files; list what was written."""
    from qaqc import export, report  # noqa: PLC0415

    try:
        report_path = report.write_report(ctx)
        files = export.write_outputs(ctx, report_path=report_path)
    except (export.ExportError, *USER_ERRORS) as err:
        raise StopRun("writing the outputs", _message(err)) from None
    return pd.DataFrame(
        [{"File": name, "Folder": str(Path(p).parent)} for name, p in files.items()]
    ).set_index("File")


def warnings_table(ctx: AuditContext, module: str | None = None) -> Styler | pd.DataFrame:
    """Messages from the run, worst first, coloured by severity."""
    rows = [w.to_dict() for w in ctx.warnings if module is None or w.module == module]
    df = pd.DataFrame(rows, columns=["severity", "module", "message"])
    if df.empty:
        return df
    df = df.sort_values("severity", key=lambda s: s.map(SEVERITY_ORDER), kind="stable")
    df = df.reset_index(drop=True)
    df.columns = ["Severity", "Step", "Message"]

    def colour(row: pd.Series) -> list[str]:
        bg = SEVERITY_COLOURS.get(row["Severity"], "")
        return [f"background-color: {bg}" if bg else "" for _ in row]

    return (
        df.style.apply(colour, axis=1)
        .set_properties(subset=["Message"], **{"text-align": "left", "white-space": "pre-wrap"})
        .hide(axis="index")
    )


def timings(ctx: AuditContext) -> pd.DataFrame:
    """Seconds taken by each step."""
    t = ctx.bundle.get("timings", {})
    return pd.DataFrame({"Seconds": t}).rename_axis("Step")
