"""The HTML report (spec 12).

:func:`write_report` draws the figures, collects every table from ``ctx.bundle`` and writes one
self-contained file, ``BM_QAQC_Report.html``, into ``paths.output_dir``. It does not need the
export step to have run: the flag table and the Leapfrog filters are built here from
``ctx.flags``.

The page is built from simple "blocks" (heading, note, table, figure, selector, ...) that the
template ``templates/report.html.j2`` turns into HTML. All text goes through Jinja's
autoescaping. Only the plotly figure data (made here, with ``<``, ``>`` and ``&`` escaped) and
the plotly.js library are marked safe.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, PackageLoader
from markupsafe import Markup
from plotly import offline

import qaqc
from qaqc import figures
from qaqc.export import ExportError, flag_catalog

if TYPE_CHECKING:
    from qaqc.context import AuditContext

logger = logging.getLogger(__name__)

REPORT_NAME = "BM_QAQC_Report.html"
DASH = "–"
CDN_URL = "https://cdn.plot.ly/plotly-{version}.min.js"

# Tab id -> heading (the section headings of the report)
TABS: dict[str, str] = {
    "overview": "Overview and column matching",
    "geometry": "Geometry and density",
    "classification": "Classification",
    "grades": "Grade distributions",
    "support": "Estimation and support",
    "composites": "Composite validation and swaths",
    "flags": "Flag register",
}

Block = dict[str, Any]


# --------------------------------------------------------------------------- formatting


def _ok(x: Any) -> bool:
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fnum(x: Any, dp: int = 2) -> str:
    """Number with thousands separators and ``dp`` decimals (an en dash if missing)."""
    return f"{float(x):,.{dp}f}" if _ok(x) else DASH


def fsig(x: Any, sig: int = 4) -> str:
    """Number with ``sig`` significant figures (good for grades)."""
    if not _ok(x):
        return DASH
    v = float(x)
    if v != 0 and (abs(v) >= 1e7 or abs(v) < 1e-3):
        return f"{v:.{sig - 1}e}"
    return f"{v:,.{sig}g}" if abs(v) < 10**sig else f"{v:,.0f}"


def fint(x: Any) -> str:
    """Whole number with thousands separators."""
    return f"{round(float(x)):,}" if _ok(x) else DASH


def fpct(x: Any, dp: int = 1) -> str:
    """A number that is already a percentage."""
    return f"{float(x):.{dp}f}%" if _ok(x) else DASH


def c(text: Any, cls: str | None = None) -> dict[str, Any]:
    """A table cell with a style class (``ok``, ``warn``, ``bad`` or ``info``)."""
    return {"text": str(text), "cls": cls}


def _count_cell(n: int) -> dict[str, Any]:
    return c(fint(n), "ok" if n == 0 else "warn")


# --------------------------------------------------------------------------- block makers


def h3(text: str) -> Block:
    return {"type": "h3", "text": text}


def h4(text: str) -> Block:
    return {"type": "h4", "text": text}


def note(text: str) -> Block:
    return {"type": "note", "text": text}


def skipped(text: str) -> Block:
    return {"type": "skipped", "text": text}


def table(columns: list[str], rows: list[list[Any]], caption: str | None = None) -> Block | None:
    """A sortable table, or None when there are no rows."""
    if not rows:
        return None
    return {"type": "table", "columns": columns, "rows": rows, "caption": caption}


def kv(rows: list[tuple[str, Any]]) -> Block:
    """A two-column "setting / value" table."""
    return {"type": "kv", "rows": rows}


def kpis(items: list[dict[str, Any]]) -> Block:
    return {"type": "kpis", "items": items}


def details(summary: str, text: str) -> Block:
    return {"type": "details", "summary": summary, "text": text}


class _Page:
    """Collects the figure JSON and hands out figure ids while the blocks are built."""

    def __init__(self, ctx: AuditContext) -> None:
        self.ctx = ctx
        self.json: dict[str, str] = {}

    def _store(self, fig: Any) -> str:
        fid = f"fig{len(self.json) + 1}"
        text = fig.to_json()
        # nothing in the data can end the <script> block or be read as markup
        self.json[fid] = (
            text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        )
        return fid

    def figure(self, name: str, alt: str | None = None) -> Block | None:
        fig = self.ctx.figures.get(name)
        if fig is None:
            return None
        return {"type": "figure", "id": self._store(fig), "alt": alt or name}

    def selector(self, prefix: str, label: str) -> Block | None:
        """A drop-down that draws one of the figures whose name starts with ``prefix``."""
        names = sorted(n for n in self.ctx.figures if n.startswith(prefix))
        if not names:
            return None
        options = []
        for n in names:
            parts = n[len(prefix) :].rpartition("|")
            options.append(
                {
                    "id": self._store(self.ctx.figures[n]),
                    "label": f"{parts[0]} | {parts[2].upper()}",
                }
            )
        return {"type": "selector", "key": prefix.rstrip("_"), "label": label, "options": options}


def _blocks(*items: Block | list[Block | None] | None) -> list[Block]:
    out: list[Block] = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, list):
            out.extend(b for b in it if b is not None)
        else:
            out.append(it)
    return out


def _skip_reason(ctx: AuditContext, key: str) -> str | None:
    """Why a module has no results ("not run" or the reason it gave), or None if it has."""
    res = ctx.bundle.get(key)
    if not res:
        return "Not run: this check was not part of this run."
    if key == "composites" and not res.get("loaded"):
        return f"Skipped because {res.get('skipped', 'no composites were loaded')}."
    if res.get("skipped"):
        return f"Skipped because {res['skipped']}."
    return None


# --------------------------------------------------------------------------- overview


def _status_badges(ctx: AuditContext) -> list[dict[str, str]]:
    """Headline red / amber / green items for the top of the report."""
    out: list[dict[str, str]] = []
    comp = ctx.bundle.get("composites") or {}
    uncapped = comp.get("uncapped_elements") or []
    if uncapped:
        out.append(
            {
                "cls": "bad",
                "text": "Composites are uncapped ("
                + ", ".join(e.upper() for e in uncapped)
                + "): the bias results may be misleading.",
            }
        )
    bias = [b for b in comp.get("bias_table") or [] if b.get("domain") != "(all mineralized)"]
    for status, cls, word in (
        ("fail", "bad", "differs a lot from"),
        ("warn", "warn", "differs from"),
    ):
        rows = [b for b in bias if b.get("status") == status]
        if rows:
            names = ", ".join(f"{b['element'].upper()} in {b['domain']}" for b in rows[:6])
            more = f" and {len(rows) - 6} more" if len(rows) > 6 else ""
            out.append(
                {
                    "cls": cls,
                    "text": f"Model grade {word} the nearest-neighbour benchmark: {names}{more}.",
                }
            )
    for key, label in (
        ("geometry", "Geometry"),
        ("density", "Density"),
        ("grades", "Grades"),
        ("classification", "Classification"),
        ("support", "Estimation support"),
        ("composites", "Composites"),
    ):
        reason = _skip_reason(ctx, key)
        if reason:
            out.append({"cls": "warn", "text": f"{label}: {reason}"})
    for el, info in ((ctx.bundle.get("grades") or {}).get("elements") or {}).items():
        if (info.get("units") or {}).get("likely_wrong"):
            out.append({"cls": "warn", "text": f"The unit of {el.upper()} looks wrong."})
    unclass = (ctx.bundle.get("classification") or {}).get("unclassified_mineralized") or {}
    if unclass.get("blocks"):
        out.append(
            {
                "cls": "warn",
                "text": f"{fint(unclass['blocks'])} unclassified blocks are mineralized.",
            }
        )
    low = (ctx.bundle.get("support") or {}).get("low_support") or {}
    if low.get("blocks"):
        out.append(
            {
                "cls": "warn",
                "text": f"{fint(low['blocks'])} Measured or Indicated blocks fail a support limit.",
            }
        )
    n_err = sum(1 for w in ctx.warnings if w.severity == "error")
    if n_err:
        out.append(
            {
                "cls": "bad",
                "text": f"{n_err} error message(s): some results are missing or not trustworthy.",
            }
        )
    if not out:
        out.append(
            {"cls": "ok", "text": "No headline problems found. Read the sections below for detail."}
        )
    return out


def _overview(ctx: AuditContext, page: _Page) -> list[Block]:
    b = ctx.bundle
    inputs = (b.get("inputs") or {}).get("block_model") or {}
    geo = b.get("geometry") or {}
    n_rows = len(ctx.bm)
    flagged = int(ctx.flags.any().sum())
    ingest = b.get("ingest") or {}
    sev = {s: sum(1 for w in ctx.warnings if w.severity == s) for s in ("error", "warning", "info")}
    items = [
        {"label": "Blocks read", "value": fint(inputs.get("rows", n_rows))},
        {"label": "Blocks checked", "value": fint(ingest.get("rows_analysed", n_rows))},
        {"label": "Volume (m³)", "value": fsig(geo.get("total_volume_m3"), 4)},
        {"label": "Tonnes", "value": fsig(geo.get("total_tonnes"), 4)},
        {
            "label": "Blocks flagged",
            "value": f"{fint(flagged)} ({fpct(100.0 * flagged / n_rows if n_rows else 0, 2)})",
        },
        {"label": "Errors", "value": str(sev["error"]), "cls": "bad" if sev["error"] else "ok"},
        {
            "label": "Warnings",
            "value": str(sev["warning"]),
            "cls": "warn" if sev["warning"] else "ok",
        },
        {"label": "Notes", "value": str(sev["info"])},
    ]
    blocks: list[Block | None] = [
        h3("Status summary"),
        {"type": "badges", "items": _status_badges(ctx)},
        kpis(items),
    ]

    # inputs
    in_rows = [
        [
            "Block model",
            inputs.get("path", DASH),
            fint(inputs.get("rows")),
            inputs.get("sha256") or DASH,
        ]
    ]
    comp_in = (b.get("composites") or {}).get("input") or {}
    if comp_in.get("path"):
        in_rows.append(
            [
                "Composites",
                comp_in.get("path"),
                fint(comp_in.get("rows")),
                comp_in.get("sha256") or DASH,
            ]
        )
    blocks += [
        h3("Input files"),
        note("The SHA-256 fingerprint identifies the exact file that was checked."),
        table(["Input", "File", "Rows", "SHA-256"], in_rows),
    ]

    # warnings
    order = {"error": 0, "warning": 1, "info": 2}
    names = {"error": "Error", "warning": "Warning", "info": "Note"}
    cls = {"error": "bad", "warning": "warn", "info": "info"}
    warn_rows = [
        [c(names[w.severity], cls[w.severity]), w.module, w.message]
        for w in sorted(ctx.warnings, key=lambda w: order[w.severity])
    ]
    blocks += [
        h3("Warnings and notes"),
        note(
            "Errors mean a result is missing or cannot be trusted. Warnings need a look. "
            "Notes are for information."
        ),
        table(["Level", "Check", "Message"], warn_rows) or note("There are no warnings."),
    ]

    # column matching
    schema = b.get("schema") or ctx.schema.to_bundle()
    blocks += [
        h3("How the file's columns were matched"),
        note(
            "Check that each grade, domain and coordinate column is the one you expect. "
            "If not, set it under column_overrides in the project file."
        ),
        table(
            ["What it is", "Column in file", "How it was found"],
            [list(map(str, r)) for r in ctx.schema.describe().itertuples(index=False)],
        ),
    ]
    reasons = schema.get("unmatched_reasons") or {}
    unmatched = schema.get("unmatched_columns") or []
    blocks += [
        h4("Columns that were not used"),
        table(["Column in file", "Why"], [[col, reasons.get(col, DASH)] for col in unmatched]),
    ]
    software = schema.get("software") or DASH
    detected = schema.get("software_detected") or DASH
    blocks.append(
        kv(
            [
                ("Software (used for column names)", software),
                ("Software (detected from the file)", detected),
            ]
        )
    )
    if schema.get("header_raw"):
        blocks.append(details("File header lines (as read)", str(schema["header_raw"])))

    # grid
    grid = b.get("grid") or {}
    src = grid.get("sources") or {}

    def vec(v: Any) -> str:
        return ", ".join(fsig(x, 8) for x in v) if v else DASH

    grid_rows = [
        ["Model type", grid.get("model_type", DASH), src.get("model_type", DASH)],
        [
            "Parent cell size (DX, DY, DZ)",
            vec(grid.get("parent_size")),
            src.get("parent_size", DASH),
        ],
        ["Model corner (X0, Y0, Z0)", vec(grid.get("origin")), src.get("origin", DASH)],
        [
            "Rotation (azimuth, degrees)",
            fsig(grid.get("azimuth_deg"), 6),
            src.get("azimuth_deg", src.get("rotation", DASH)),
        ],
    ]
    if grid.get("pct_scale"):
        grid_rows.append(["Percentage scale", grid["pct_scale"], src.get("pct_scale", DASH)])
    if grid.get("min_subblock_size"):
        grid_rows.append(["Smallest sub-block", vec(grid["min_subblock_size"]), DASH])
    blocks += [
        h3("Model grid"),
        note(
            "Where each grid value came from. A wrong origin or cell size shows up later "
            "as many off-grid blocks."
        ),
        table(["Setting", "Value", "Where it came from"], grid_rows),
    ]
    for n in grid.get("notes") or []:
        blocks.append(note(str(n)))

    # reading
    excl = ingest.get("excluded_rows") or {}
    read_rows: list[list[Any]] = [
        ["Reader", ingest.get("reader") or DASH],
        ["Read from the saved copy (cache)", "yes" if ingest.get("from_cache") else "no"],
        ["Air blocks (grid checks only)", fint(excl.get("air", 0))],
        ["Blocks switched off by a selection column", fint(excl.get("selection", 0))],
    ]
    blocks += [h3("Reading the file"), kv([(a, b_) for a, b_ in read_rows])]
    blocks.append(
        table(
            ["Column", "Placeholder values treated as missing"],
            [[k, fint(v)] for k, v in (ingest.get("sentinel_replacements") or {}).items()],
        )
    )
    blocks.append(
        table(
            ["Column", "Values that are not numbers (treated as missing)"],
            [[k, fint(v)] for k, v in (ingest.get("coerced_values") or {}).items()],
        )
    )

    # timings
    blocks += [
        h3("How long each step took"),
        table(["Step", "Seconds"], [[k, fnum(v, 2)] for k, v in (b.get("timings") or {}).items()]),
    ]
    blocks.append(
        details(
            "Project settings used (for the record)",
            json.dumps(b.get("config") or ctx.cfg.snapshot(), indent=2, default=str),
        )
    )
    return _blocks(blocks)


# --------------------------------------------------------------------------- geometry & density


def _geometry(ctx: AuditContext, page: _Page) -> list[Block]:
    out: list[Block | None] = [h3("Geometry")]
    reason = _skip_reason(ctx, "geometry")
    if reason:
        out.append(skipped(reason))
    else:
        g = ctx.bundle["geometry"]
        info = ctx.flags.info()
        box = g.get("bounding_box") or {}
        out.append(
            kv(
                [
                    ("Model type", g.get("model_type", DASH)),
                    ("Blocks checked", fint(g.get("blocks_analysed"))),
                    ("Parent cells", fint(g.get("parent_cells"))),
                    ("Total volume (m³)", fnum(g.get("total_volume_m3"), 0)),
                    ("Total tonnes", fnum(g.get("total_tonnes"), 0)),
                    ("Blocks without tonnes", fint(g.get("blocks_without_tonnes"))),
                    ("Blocks without coordinates", fint(g.get("blocks_without_coordinates"))),
                ]
                + [
                    (f"Extent along {a.upper()} (m)", f"{fnum(v[0], 1)} to {fnum(v[1], 1)}")
                    for a, v in box.items()
                    if v
                ]
            )
        )
        rows = [
            [
                name.removeprefix("FLAG_"),
                info[name].description if name in info else DASH,
                _count_cell(int(n)),
            ]
            for name, n in (g.get("flags") or {}).items()
        ]
        out += [
            note("Each check below counts the blocks it flags. Zero is what you want to see."),
            table(["Check", "What it looks for", "Blocks flagged"], rows),
        ]
        if g.get("overfill"):
            o = g["overfill"]
            out.append(
                note(
                    "Parent cells overfilled by their sub-blocks: "
                    f"{fint(o['parent_cells'])} ({fint(o['blocks'])} blocks)."
                )
            )
        if g.get("percentage"):
            p = g["percentage"]
            out.append(
                kv(
                    [
                        (
                            "Rows with a percentage below 0 or above 100%",
                            fint(p["rows_out_of_range"]),
                        ),
                        (
                            "Parent cells whose percentages add up to more than 100%",
                            fint(p["parent_cells_over_100pct"]),
                        ),
                        (
                            "Blocks in those parent cells",
                            fint(p["blocks_in_parent_cells_over_100pct"]),
                        ),
                        ("Rows with no percentage value", fint(p["rows_missing_pct"])),
                    ]
                )
            )
        sub = (g.get("subblock_support") or {}).get("elements") or {}
        if sub:
            rows = [
                [
                    el.upper(),
                    fint(v.get("groups_with_several_subblocks")),
                    fint(v.get("groups_with_different_grades")),
                    fpct(100 * v["share_with_different_grades"])
                    if _ok(v.get("share_with_different_grades"))
                    else DASH,
                    v.get("observed_support") or DASH,
                ]
                for el, v in sub.items()
            ]
            expected = g["subblock_support"].get("expected")
            out += [
                h4("Sub-block estimation support"),
                note(
                    "Were sub-blocks estimated as one parent cell, or one by one? "
                    + (
                        f"The project file says: {expected}."
                        if expected
                        else "The project file does not say, so nothing is flagged."
                    )
                ),
                table(
                    [
                        "Element",
                        "Groups with several sub-blocks",
                        "Of which grades differ",
                        "Share that differ",
                        "Looks like",
                    ],
                    rows,
                ),
            ]
        wf = [
            [
                r["domain"],
                fnum(r["model_volume_m3"], 0),
                fnum(r["wireframe_volume_m3"], 0),
                fnum(r["difference_m3"], 0),
                fpct(r["difference_pct"]),
            ]
            for r in g.get("wireframe_volumes") or []
        ]
        out += [
            h4("Model volume against wireframe volume") if wf else None,
            table(
                [
                    "Domain",
                    "Model volume (m³)",
                    "Wireframe volume (m³)",
                    "Difference (m³)",
                    "Difference",
                ],
                wf,
            ),
        ]

    out.append(h3("Density"))
    reason = _skip_reason(ctx, "density")
    if reason:
        out.append(skipped(reason))
        return _blocks(out)
    d = ctx.bundle["density"]
    a = d.get("anomalies") or {}
    out += [
        note(
            "Density is checked against the allowed range for each domain. "
            "Look for domains with a wide spread or a fixed value."
        ),
        kv(
            [
                ("Blocks checked", fint(a.get("blocks_checked"))),
                ("Density missing", fint(a.get("missing"))),
                ("Density too low", fint(a.get("low"))),
                ("Density too high", fint(a.get("high"))),
            ]
        ),
        table(
            [
                "Domain",
                "Blocks",
                "Missing",
                "Min",
                "P10",
                "Median",
                "Mean (by volume)",
                "P90",
                "Max",
                "SD",
                "Constant?",
            ],
            [
                [
                    r["domain"],
                    fint(r["blocks"]),
                    fint(r["missing"]),
                    fnum(r["min"], 3),
                    fnum(r["p10"], 3),
                    fnum(r["p50"], 3),
                    fnum(r["mean_volume_weighted"], 3),
                    fnum(r["p90"], 3),
                    fnum(r["max"], 3),
                    fnum(r["sd"], 3),
                    c("yes", "info") if r.get("is_constant") else "no",
                ]
                for r in d.get("by_domain") or []
            ],
        ),
        page.figure("density_box"),
        page.figure("density_vs_grade"),
    ]
    corr = [
        [
            r["element"].upper(),
            r["domain"],
            fint(r["pairs"]),
            fnum(r["pearson"], 2),
            fnum(r["spearman"], 2),
        ]
        for r in d.get("density_vs_grade") or []
    ]
    out += [
        h4("Density against grade: correlation") if corr else None,
        table(["Element", "Domain", "Blocks", "Pearson", "Spearman"], corr),
    ]
    skipped_pairs = d.get("density_vs_grade_skipped") or []
    if skipped_pairs:
        out.append(
            details(
                f"{len(skipped_pairs)} element and domain pair(s) not tested",
                "\n".join(
                    f"{s['element'].upper()} in {s['domain']}: {s['reason']}" for s in skipped_pairs
                ),
            )
        )
    return _blocks(out)


# --------------------------------------------------------------------------- classification


def _grade_columns(units: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for el, u in units.items():
        cols += [
            f"{el.upper()} grade ({u.get('grade', '')})",
            f"{el.upper()} metal ({u.get('metal', '')})",
        ]
    return cols


def _grade_cells(row: dict[str, Any], units: dict[str, Any]) -> list[str]:
    cells: list[str] = []
    for el in units:
        cells += [fsig((row.get("grade") or {}).get(el)), fsig((row.get("metal") or {}).get(el), 5)]
    return cells


def _classification(ctx: AuditContext, page: _Page) -> list[Block]:
    reason = _skip_reason(ctx, "classification")
    if reason:
        return [skipped(reason)]
    cl = ctx.bundle["classification"]
    units = cl.get("units") or {}
    labels = ctx.cfg.classification.rank_labels
    by_rank: dict[int, list[str]] = {}
    for code, rank in ctx.cfg.classification.rank_map.items():
        by_rank.setdefault(int(rank), []).append(str(code))
    rank_rows = [
        [labels.get(r, str(r)), ", ".join(sorted(codes))]
        for r, codes in sorted(by_rank.items(), reverse=True)
    ]
    basis = (
        "tonnes" if cl.get("weight_basis") == "tonnes" else "volume (there is no density column)"
    )
    inv = cl.get("inventory") or []
    inv_rows = [
        [
            r["domain"] + (" (waste)" if r.get("is_waste") else ""),
            r["class"],
            fint(r["blocks"]),
            fnum(r.get("tonnes"), 0),
            fnum(r["volume_m3"], 0),
            fpct(r.get("pct_of_domain")),
        ]
        + _grade_cells(r, units)
        for r in inv
    ]
    un = cl.get("unclassified_mineralized") or {}
    metal_txt = ", ".join(f"{e.upper()}: {u.get('metal')}" for e, u in units.items())
    cut_txt = ", ".join(
        f"{e.upper()} {fsig(v)}" for e, v in (un.get("cutoffs_in_column_units") or {}).items()
    )
    out: list[Block | None] = [
        h3("Class codes"),
        note(
            "How the codes in the file were turned into Measured, Indicated, Inferred "
            "or Unclassified."
        ),
        table(["Class", "Codes in the project file"], rank_rows),
        h4("Codes that are not in the list") if cl.get("unmapped_codes") else None,
        table(
            ["Code", "Blocks", "Tonnes"],
            [
                [u["code"], fint(u["blocks"]), fnum(u.get("tonnes"), 0)]
                for u in cl.get("unmapped_codes") or []
            ],
        ),
        h3("Inventory by domain and class"),
        note(f"Averages are weighted by {basis}. Metal is in {metal_txt}."),
        table(
            ["Domain", "Class", "Blocks", "Tonnes", "Volume (m³)", "Share of domain"]
            + _grade_columns(units),
            inv_rows,
        ),
        page.figure("classification_bar"),
        h3("Unclassified blocks that look mineralized"),
        kv(
            [
                ("Blocks", c(fint(un.get("blocks", 0)), "ok" if not un.get("blocks") else "warn")),
                ("Tonnes", fnum(un.get("tonnes"), 0)),
                ("Grade limits used (at or above)", cut_txt or DASH),
                (
                    "Elements with no limit set (not checked)",
                    ", ".join(e.upper() for e in un.get("elements_without_cutoff") or []) or DASH,
                ),
            ]
        ),
    ]
    st = cl.get("resource_statement")
    if st:
        out.append(h3("Resource statement check"))
        if st.get("skipped"):
            out.append(skipped(f"Skipped because {st['skipped']}."))
        else:
            el = st["element"].upper()
            out.append(
                note(
                    f"Everything at or above {fsig(st['cutoff_canonical'])} "
                    f"{st['canonical_unit']} {el} (in the file's unit: "
                    f"{fsig(st['cutoff_in_column_unit'])} {st['column_unit']}); "
                    + ("waste included." if st.get("waste_included") else "waste left out.")
                )
            )
            rows = list(st.get("by_class") or [])
            if st.get("total"):
                rows.append({**st["total"], "class": "Total"})
            out.append(
                table(
                    ["Class", "Blocks", "Tonnes"] + _grade_columns(units),
                    [
                        [r["class"], fint(r["blocks"]), fnum(r.get("tonnes"), 0)]
                        + _grade_cells(r, units)
                        for r in rows
                    ],
                )
            )
    return _blocks(out)


# --------------------------------------------------------------------------- grades


def _grades(ctx: AuditContext, page: _Page) -> list[Block]:
    reason = _skip_reason(ctx, "grades")
    if reason:
        return [skipped(reason)]
    g = ctx.bundle["grades"]
    elements = g.get("elements") or {}
    if not elements:
        return [skipped("No grade columns were found in the block model.")]
    basis = "tonnes" if g.get("weights") == "tonnes" else "volume (there is no density column)"
    out: list[Block | None] = [
        note(f"Averages and percentiles are weighted by {basis}. Air blocks are left out.")
    ]
    for el, e in elements.items():
        u = e.get("units") or {}
        th = e.get("thresholds") or {}
        unit = th.get("unit", u.get("unit", ""))
        rows = []
        for r in e.get("by_domain") or []:
            status_cls = {"estimated": "ok", "waste": "info"}.get(r["status"], "warn")
            rows.append(
                [
                    r["domain"],
                    c(r["status"], status_cls),
                    fint(r["blocks"]),
                    f"{fint(r['unestimated'])} ({fpct(r['unestimated_pct'])})",
                    fnum(r.get("tonnes"), 0),
                    fsig(r["mean_weighted"]),
                    fsig(r["mean_naive"]),
                    fsig(r["sd_weighted"]),
                    fnum(r["cv_weighted"], 2),
                    fsig(r["min"]),
                    fsig(r["p10"]),
                    fsig(r["p50"]),
                    fsig(r["p90"]),
                    fsig(r["p99"]),
                    fsig(r["max"]),
                    fsig(r.get("contained_metal"), 5),
                ]
            )
        counts = e.get("flags") or {}
        flag_rows = [
            ["Grade below zero", _count_cell(counts.get("NEG", 0))],
            [
                f"Grade above the ceiling ({fsig(th.get('ceiling'))} {unit})"
                if th.get("ceiling") is not None
                else "Grade above the ceiling (no ceiling set)",
                _count_cell(counts.get("CEILING", 0)),
            ],
            ["Grade missing where the element is estimated", _count_cell(counts.get("UNEST", 0))],
            [
                f"Waste grade above {fsig(th.get('diagnostic_cutoff'))} {unit}"
                if th.get("diagnostic_cutoff") is not None
                else "Waste grade above the diagnostic cutoff (none set)",
                _count_cell(counts.get("BLEED", 0)),
            ],
        ]
        metal = e.get("metal_unit")
        out += [
            h3(f"{el.upper()}"),
            kv(
                [
                    ("Column in file", e.get("column", DASH)),
                    (
                        "Unit",
                        f"{unit} ({u.get('unit_source', DASH)})"
                        + ("  - LOOKS WRONG, check the unit" if u.get("likely_wrong") else ""),
                    ),
                    ("Estimated in", ", ".join(e.get("applicable_domains") or []) or DASH),
                    (
                        "In the file but not estimated",
                        ", ".join(e.get("not_estimated_domains") or []) or DASH,
                    ),
                ]
            ),
            table(["Check", "Blocks flagged"], flag_rows),
            table(
                [
                    "Domain",
                    "Status",
                    "Blocks",
                    "Not estimated",
                    "Tonnes",
                    "Mean (weighted)",
                    "Mean (simple)",
                    "SD",
                    "CV",
                    "Min",
                    "P10",
                    "Median",
                    "P90",
                    "P99",
                    "Max",
                    f"Contained metal ({metal})" if metal else "Contained metal",
                ],
                rows,
            ),
            page.figure(f"cdf_{el}"),
        ]
        skipped_checks = e.get("skipped_checks") or []
        if skipped_checks:
            out.append(
                details(
                    "Checks not done for " + el.upper(),
                    "\n".join(f"{s['check']}: {s['reason']}" for s in skipped_checks),
                )
            )
    return _blocks(out)


# --------------------------------------------------------------------------- support


def _support(ctx: AuditContext, page: _Page) -> list[Block]:
    reason = _skip_reason(ctx, "support")
    if reason:
        return [skipped(reason)]
    s = ctx.bundle["support"]
    src = s.get("source") or {}
    cols = src.get("columns") or {}
    by_detail = src.get("by_detail") or {}
    out: list[Block | None] = [
        h3("Where the estimation details come from"),
        note("These are the columns used for the number of samples, search pass and so on."),
        table(
            ["Detail", "Column in file", "Taken from"],
            [
                [
                    k,
                    cols.get(k) or DASH,
                    v.replace("element:", "grade column of ").replace("global", "whole model"),
                ]
                for k, v in by_detail.items()
            ],
        ),
    ]
    if src.get("missing_details"):
        out.append(note("Not in the file: " + ", ".join(src["missing_details"]) + "."))
    out.append(page.figure("support_box"))

    stats_rows = []
    for cls_row in s.get("stats_by_class") or []:
        for detail, d in (cls_row.get("details") or {}).items():
            stats_rows.append(
                [
                    cls_row["class"],
                    detail,
                    fint(d.get("blocks_with_value")),
                    fsig(d.get("min")),
                    fsig(d.get("p10")),
                    fsig(d.get("p50")),
                    fsig(d.get("mean")),
                    fsig(d.get("p90")),
                    fsig(d.get("max")),
                ]
            )
    out += [
        h3("Estimation details by class"),
        table(
            [
                "Class",
                "Detail",
                "Blocks with a value",
                "Min",
                "P10",
                "Median",
                "Mean",
                "P90",
                "Max",
            ],
            stats_rows,
        ),
    ]
    pb = s.get("pass_breakdown")
    if pb:
        out.append(h3("Search pass"))
        for key, first, title in (
            ("by_class", "class", "By class"),
            ("by_domain", "domain", "By domain"),
        ):
            rows = [
                [
                    r[first],
                    r["pass"] if r["pass"] is not None else "no value",
                    fint(r["blocks"]),
                    fnum(r.get("tonnes"), 0),
                    fpct(r.get("pct_of_row")),
                ]
                for r in pb.get(key) or []
            ]
            out += [
                h4(title),
                table([first.capitalize(), "Pass", "Blocks", "Tonnes", "Share of row total"], rows),
            ]
    low = s.get("low_support") or {}
    thr_rows = []
    for info in (low.get("by_rank") or {}).values():
        for cond, v in (info.get("conditions") or {}).items():
            thr_rows.append(
                [
                    info["class"],
                    cond,
                    fsig(v["limit"]),
                    _count_cell(v["failed"]),
                    fint(v["missing"]),
                ]
            )
    out += [
        h3("Blocks with too little support for their class"),
        note(
            low.get("note", "")
            or "A block with no value for a detail does not fail that condition."
        ),
        kv(
            [
                (
                    "Blocks failing at least one condition",
                    c(fint(low.get("blocks", 0)), "ok" if not low.get("blocks") else "warn"),
                ),
                ("Tonnes", fnum(low.get("tonnes"), 0)),
            ]
        ),
        table(["Class", "Condition", "Limit", "Blocks failing", "Blocks with no value"], thr_rows),
    ]
    sk = s.get("skipped_checks") or []
    if sk:
        out.append(
            details(
                "Conditions not tested",
                "\n".join(f"{x['rank']}: {x['condition']}: {x['reason']}" for x in sk),
            )
        )
    if s.get("kriging_variance_note"):
        out.append(note(s["kriging_variance_note"]))
    return _blocks(out)


# --------------------------------------------------------------------------- composites


def _composites(ctx: AuditContext, page: _Page) -> list[Block]:
    reason = _skip_reason(ctx, "composites")
    if reason:
        return [skipped(reason)]
    cp = ctx.bundle["composites"]
    cap_rows = []
    for el, info in (cp.get("capping") or {}).items():
        method = info.get("method")
        text, cls = {
            "capped_column": ("Capped column found", "ok"),
            "config_caps": ("Caps from the project file", "ok"),
            "uncapped": ("UNCAPPED", "bad"),
        }.get(method, (str(method), "info"))
        cap_rows.append(
            [
                el.upper(),
                c(text, cls),
                info.get("raw_column") or DASH,
                info.get("capped_column") or DASH,
                fint(info.get("n_capped")),
                fsig(info.get("max_used")),
            ]
        )
    out: list[Block | None] = [
        h3("Capping"),
        note(
            "Bias checks should use capped composites. If the composites are not capped, "
            "high grades make the benchmark look too high."
        ),
        kv(
            [
                ("Composites read", fint(cp.get("rows"))),
                ("Weighted by", "length" if cp.get("length_source") else "equal weight"),
            ]
        ),
        table(
            [
                "Element",
                "Capping",
                "Raw grade column",
                "Capped grade column",
                "Composites capped",
                "Highest grade used",
            ],
            cap_rows,
        ),
    ]
    if cp.get("elements_without_composites"):
        out.append(
            note(
                "No composite grades for: "
                + ", ".join(e.upper() for e in cp["elements_without_composites"])
                + "."
            )
        )

    matrix = cp.get("domain_matrix") or []
    comp_doms = sorted({str(m["composite_domain"]) for m in matrix})
    block_doms = sorted({str(m["block_domain"]) for m in matrix})
    lookup = {(str(m["composite_domain"]), str(m["block_domain"])): m["count"] for m in matrix}
    mat_rows = []
    for cd in comp_doms:
        row: list[Any] = [cd]
        for bd in block_doms:
            n = lookup.get((cd, bd), 0)
            row.append(c(fint(n), "ok") if cd == bd and n else (c(fint(n), "warn") if n else "0"))
        mat_rows.append(row)
    out += [
        h3("Do composite domains agree with the model?"),
        note(
            "Each row is the composite's own domain. Each column is the domain of the block "
            "it sits in. Counts off the diagonal are mismatches."
        ),
        kv(
            [
                (
                    "Composites outside the model",
                    c(
                        fint(cp.get("outside_count", 0)),
                        "ok" if not cp.get("outside_count") else "warn",
                    ),
                ),
                (
                    "Composites in a different domain from their block",
                    c(
                        fint(cp.get("mismatch_count", 0)),
                        "ok" if not cp.get("mismatch_count") else "warn",
                    ),
                ),
                ("Composites with no coordinates", fint(cp.get("no_coordinates_count", 0))),
            ]
        ),
        table(["Composite domain \\ block domain"] + block_doms, mat_rows),
    ]

    bias_rows = []
    for r in cp.get("bias_table") or []:
        status = r.get("status")
        cls = {"ok": "ok", "warn": "warn", "fail": "bad"}.get(status, "info")
        label = {"ok": "OK", "warn": "Check", "fail": "Large difference"}.get(status, DASH)
        bias_rows.append(
            [
                r["domain"],
                r["element"].upper(),
                fint(r.get("composite_count")),
                fint(r.get("blocks")),
                fnum(r.get("tonnes"), 0),
                fsig(r.get("composite_mean")),
                fsig(r.get("nn_mean")),
                fsig(r.get("bm_mean")),
                fpct(r.get("diff_vs_composites_pct")),
                fpct(r.get("diff_vs_nn_pct")),
                fnum(r.get("variance_ratio_bm_nn"), 2),
                c(label, cls),
            ]
        )
    warn_pct, fail_pct = ctx.cfg.composites.bias_warn_pct, ctx.cfg.composites.bias_fail_pct
    nn_txt = "; ".join(f"{el.upper()}: {src}" for el, src in (cp.get("nn_source") or {}).items())
    out += [
        h3("Model against composites and nearest neighbour"),
        note(
            "The status compares the model mean with the nearest-neighbour mean: "
            f"more than {warn_pct:g}% apart is 'Check', "
            f"more than {fail_pct:g}% is 'Large difference'. "
            f"Nearest-neighbour source: {nn_txt or DASH}."
        ),
        table(
            [
                "Domain",
                "Element",
                "Composites",
                "Blocks",
                "Tonnes",
                "Composite mean",
                "Nearest-neighbour mean",
                "Model mean",
                "Model vs composites",
                "Model vs nearest neighbour",
                "Variance ratio (model / NN)",
                "Status",
            ],
            bias_rows,
        ),
    ]
    above = cp.get("above_comp_max") or {}
    out += [
        h4("Blocks above the highest composite in their domain") if above else None,
        table(
            ["Element", "Blocks"],
            [[el.upper(), _count_cell(v["count"])] for el, v in above.items()],
        ),
    ]
    out += [
        h3("Grade-tonnage curves"),
        note(
            "Tonnes and average grade above each cutoff, model against nearest neighbour. "
            "The model should be smoother (fewer tonnes at high cutoffs) but not far apart "
            "overall."
        ),
        page.selector("gt_", "Domain and element"),
        h3("Swath plots"),
        note(
            "Average grade along the model X, Y and Z axes. Grey bars are tonnes; bins with "
            "very little tonnage are drawn in lighter grey. Hover over the red line for the "
            "composite count."
        ),
        page.selector("swath_", "Domain and element"),
    ]
    return _blocks(out)


# --------------------------------------------------------------------------- flags


def _flags(ctx: AuditContext, page: _Page) -> list[Block]:
    template = ctx.cfg.report.leapfrog_filter_template
    cat = flag_catalog(ctx.flags, template, len(ctx.bm))

    def rows_of(items: list[dict[str, Any]]) -> list[list[Any]]:
        return [
            [
                r["name"],
                r["module"],
                r["description"],
                _count_cell(r["count"]),
                f"{r['pct']:.3f}%",
                r["leapfrog_filter"],
            ]
            for r in items
        ]

    cols = ["Flag", "Check", "What it means", "Blocks", "Share of model", "Leapfrog filter"]
    out: list[Block | None] = [
        h3("Block flags"),
        note(
            "Flagged blocks are exported as points for Leapfrog. Paste a filter expression "
            "into Leapfrog to show one flag at a time."
        ),
        table(cols, rows_of(cat)),
    ]
    if ctx.comp_flags is not None:
        ccat = flag_catalog(ctx.comp_flags, template, ctx.comp_flags.n_rows)
        out += [
            h3("Composite flags"),
            table(
                ["Flag", "Check", "What it means", "Composites", "Share", "Leapfrog filter"],
                rows_of(ccat),
            ),
        ]
    names = ["flagged_blocks_leapfrog.csv"]
    if ctx.comp_flags is not None:
        names.append("flagged_composites.csv")
    names += [f"{stem}.csv" for stem in ctx.tables]
    if ctx.cfg.export.full_model_flags:
        names.append("model_flags.parquet")
    names.append("audit_bundle.json")
    out += [
        h3("Files written next to this report"),
        note(f"Folder: {ctx.cfg.paths.output_dir}"),
        table(["File"], [[n] for n in names]),
    ]
    return _blocks(out)


# --------------------------------------------------------------------------- page


def _plotly_script(mode: str) -> Markup:
    if mode == "cdn":
        url = CDN_URL.format(version=offline.get_plotlyjs_version())
        return Markup('<script src="{}"></script>').format(url)
    return Markup("<script>{}</script>").format(Markup(offline.get_plotlyjs()))


def _write(path: Path, text: str) -> None:
    try:
        with open(path, "wb") as f:
            f.write(text.encode("utf-8"))
    except PermissionError:
        raise ExportError(
            f"Cannot write '{path}'. The file is probably open in a browser or another program. "
            "Close it and run the report again."
        ) from None


def render(ctx: AuditContext) -> str:
    """The report as HTML text, drawn from ``ctx.bundle`` and the figures in ``ctx.figures``."""
    page = _Page(ctx)
    builders = {
        "overview": _overview,
        "geometry": _geometry,
        "classification": _classification,
        "grades": _grades,
        "support": _support,
        "composites": _composites,
        "flags": _flags,
    }
    tabs = []
    for key, title in TABS.items():
        try:
            blocks = builders[key](ctx, page)
        except Exception as exc:  # noqa: BLE001 - one broken section must not lose the report
            logger.exception("Report section %s failed", key)
            ctx.warn("error", "report", f"The section '{title}' could not be built ({exc}).")
            blocks = [skipped(f"This section could not be built ({exc}).")]
        tabs.append({"id": key, "title": title, "blocks": blocks})

    env = Environment(
        loader=PackageLoader("qaqc", "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    n_flag_total = int(ctx.flags.any().sum())
    return env.get_template("report.html.j2").render(
        title=f"Block model QA/QC report: {ctx.cfg.project_name}",
        project=ctx.cfg.project_name,
        run_time=dt.datetime.now().astimezone().strftime("%d %B %Y, %H:%M %Z"),
        version=qaqc.__version__,
        n_flagged=n_flag_total,
        tabs=tabs,
        figure_json=page.json,
        plotly_script=_plotly_script(ctx.cfg.report.plotly_js),
    )


def write_report(ctx: AuditContext) -> Path:
    """Build the figures and write ``BM_QAQC_Report.html`` into ``paths.output_dir``.

    Returns the path of the report.

    Raises:
        ExportError: If the file cannot be written, for example because it is open in a browser.
    """
    out_dir = Path(ctx.cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx.figures.clear()
    with ctx.timed("figures"):
        figures.build_figures(ctx)
    path = out_dir / REPORT_NAME
    _write(path, render(ctx))
    logger.info("Wrote report %s", path)
    return path


__all__ = ["REPORT_NAME", "TABS", "render", "write_report"]
