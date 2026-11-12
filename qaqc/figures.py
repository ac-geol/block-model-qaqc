"""Figures for the report (spec 11).

Every figure is drawn from the summaries the check modules put in ``ctx.bundle``. The only
exceptions are the density-versus-grade scatter and the grade CDFs, which read block (and
composite) rows. The scatter is cut down to ``report.max_scatter_points`` points (stratified by
domain, fixed random seed) and every CDF is reduced to about 200 points.

:func:`build_figures` returns plotly figures keyed by name:

* ``density_box``, ``density_vs_grade``: density by domain; density against each grade;
* ``classification_bar``: tonnes by domain, split by class;
* ``support_box``: estimation details by class;
* ``cdf_<el>``: grade distribution by domain for one element;
* ``gt_<domain>|<el>``: grade-tonnage curves, block model against nearest-neighbour;
* ``swath_<domain>|<el>``: three stacked swath plots (model X, Y and Z).

A figure that cannot be drawn (module not run, nothing to show) is simply left out.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qaqc.domains import applicable_domains, domain_labels, weights
from qaqc.stats import clean_pairs

if TYPE_CHECKING:
    from qaqc.context import AuditContext

logger = logging.getLogger(__name__)

PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#8C6D31",
    "#7F7F7F",
)
CLASS_COLOURS = {3: "#0072B2", 2: "#56B4E9", 1: "#E69F00", 0: "#9A9A9A"}
BM_COLOUR = "#0072B2"
COMP_COLOUR = "#D55E00"
NN_COLOUR = "#009E73"
GREY = "rgba(110,110,110,0.35)"
GREY_LIGHT = "rgba(190,190,190,0.25)"
CDF_POINTS = 200
AXIS_NAMES = {"x": "Model X (m)", "y": "Model Y (m)", "z": "Model Z (m)"}

# estimation detail -> axis title, in the order they are drawn (pass number is a table, not a box)
SUPPORT_DETAILS = {
    "nsamples": "Number of samples",
    "nholes": "Number of holes",
    "avg_dist": "Average distance to samples",
    "min_dist": "Distance to nearest sample",
    "kriging_var": "Kriging variance",
    "kriging_std": "Kriging standard deviation",
    "kriging_eff": "Kriging efficiency",
    "slope_reg": "Regression slope",
}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _f(x: Any) -> float | None:
    """A finite float, or None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "x": 0.01, "font": {"size": 15}} if title else None,
        height=height,
        margin={"l": 60, "r": 30, "t": 60 if title else 30, "b": 50},
        font={"family": "Segoe UI, Helvetica, Arial, sans-serif", "size": 12},
        legend={"orientation": "h", "y": -0.18},
        hoverlabel={"namelength": -1},
    )
    return fig


def unit_label(ctx: AuditContext, element: str) -> str:
    """Grade unit as shown on axes, for example ``pct`` or ``g/t``."""
    es = ctx.schema.elements.get(element)
    if es is None:
        return ""
    return {"gpt": "g/t", "pct": "%"}.get(es.unit, es.unit)


def domain_colours(ctx: AuditContext) -> dict[str, str]:
    """One colour per domain code, the same in every figure."""
    names: list[str] = []
    if "domain" in ctx.bm.columns:
        col = ctx.bm["domain"]
        names = [str(v) for v in (col.cat.categories if hasattr(col, "cat") else col.unique())]
    for row in ctx.bundle.get("density", {}).get("by_domain", []) or []:
        names.append(str(row.get("domain")))
    names = sorted(set(names))
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


def downsample_indices(labels: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    """Positions of at most ``max_points`` rows, spread over the groups in ``labels``.

    Every group keeps a share of the points: a small base amount (so a small domain is still
    visible) plus its proportional share of the rest. The same inputs always give the same rows.
    """
    n = len(labels)
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    uniq, inverse, counts = np.unique(
        np.asarray(labels).astype(str), return_inverse=True, return_counts=True
    )
    base = np.minimum(counts, max_points // (4 * len(uniq)))
    spare = max_points - int(base.sum())
    extra_room = counts - base
    share = np.floor(spare * extra_room / max(int(extra_room.sum()), 1)).astype(np.int64)
    quota = np.minimum(base + share, counts)
    keep = []
    for g in range(len(uniq)):
        rows = np.flatnonzero(inverse == g)
        keep.append(rng.choice(rows, size=int(quota[g]), replace=False))
    return np.sort(np.concatenate(keep))


def _cdf_points(
    values: np.ndarray, w: np.ndarray | None, n: int = CDF_POINTS
) -> tuple[np.ndarray, np.ndarray] | None:
    """~``n`` points (grade, share of weight below) of a weighted distribution, grades > 0."""
    v, ww = clean_pairs(values, w)
    keep = v > 0
    v, ww = v[keep], ww[keep]
    if v.size == 0:
        return None
    order = np.argsort(v, kind="mergesort")
    v, ww = v[order], ww[order]
    cum = np.cumsum(ww) / np.sum(ww)
    probs = np.linspace(0.0, 1.0, n)
    idx = np.clip(np.searchsorted(cum, probs, side="left"), 0, v.size - 1)
    return v[idx], probs


# ---------------------------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------------------------


def density_box(ctx: AuditContext) -> go.Figure | None:
    """Density by domain as box plots from the precomputed summary."""
    rows = (ctx.bundle.get("density") or {}).get("by_domain") or []
    colours = domain_colours(ctx)
    fig = go.Figure()
    for r in rows:
        vals = {
            k: _f(r.get(k)) for k in ("min", "p10", "p50", "p90", "max", "mean_volume_weighted")
        }
        if vals["p50"] is None:
            continue
        name = str(r["domain"])
        fig.add_trace(
            go.Box(
                x=[name],
                q1=[vals["p10"]],
                median=[vals["p50"]],
                q3=[vals["p90"]],
                lowerfence=[vals["min"]],
                upperfence=[vals["max"]],
                mean=[vals["mean_volume_weighted"]],
                name=name,
                marker_color=colours.get(name, PALETTE[0]),
                showlegend=False,
            )
        )
    if not fig.data:
        return None
    fig.update_yaxes(title="Density (SG)")
    fig.update_xaxes(title="Domain")
    fig.update_layout(boxmode="overlay")
    return _base_layout(
        fig, "Density by domain (box: 10th to 90th percentile; whiskers: min to max)"
    )


def density_vs_grade(ctx: AuditContext) -> go.Figure | None:
    """Density against each element's grade, one panel per element, downsampled by domain."""
    bm = ctx.bm
    if "sg" not in bm.columns or not ctx.schema.elements:
        return None
    elements = [e for e in ctx.schema.elements if e in bm.columns]
    if not elements:
        return None
    max_points = ctx.cfg.report.max_scatter_points
    per_panel = max(1, max_points // len(elements))
    sg = bm["sg"].to_numpy(dtype=np.float64)
    dom = domain_labels(bm)
    analysed = ctx.analysis_mask
    colours = domain_colours(ctx)
    ncols = min(3, len(elements))
    nrows = math.ceil(len(elements) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"{e.upper()} ({unit_label(ctx, e)})" for e in elements],
        horizontal_spacing=0.08,
        vertical_spacing=0.16 if nrows > 1 else 0.1,
    )
    seen: set[str] = set()
    drawn = 0
    not_shown = 0  # zero or negative grades cannot sit on a log axis
    for n, el in enumerate(elements):
        grade = bm[el].to_numpy(dtype=np.float64)
        valid = analysed & np.isfinite(sg) & np.isfinite(grade)
        with np.errstate(invalid="ignore"):
            positive = grade > 0
        not_shown += int((valid & ~positive).sum())
        cand = np.flatnonzero(valid & positive)
        if not cand.size:
            continue
        keep = cand[downsample_indices(dom[cand], per_panel, seed=n)]
        r, c = divmod(n, ncols)
        for code in sorted(set(dom[keep])):
            m = keep[dom[keep] == code]
            fig.add_trace(
                go.Scatter(
                    x=grade[m],
                    y=sg[m],
                    mode="markers",
                    name=str(code),
                    legendgroup=str(code),
                    showlegend=str(code) not in seen,
                    marker={"size": 4, "opacity": 0.6, "color": colours.get(str(code))},
                    hovertemplate=(
                        f"{code}<br>{el.upper()}: %{{x:.4g}}<br>SG: %{{y:.3f}}<extra></extra>"
                    ),
                ),
                row=r + 1,
                col=c + 1,
            )
            seen.add(str(code))
            drawn += int(m.size)
        fig.update_xaxes(
            title_text=f"{el.upper()} grade, log scale", type="log", row=r + 1, col=c + 1
        )
        fig.update_yaxes(title_text="Density (SG)" if c == 0 else None, row=r + 1, col=c + 1)
    if not drawn:
        return None
    title = "Density against grade (grade on a log scale)"
    notes = []
    if drawn >= per_panel:
        notes.append(f"at most {per_panel:,} blocks per panel are drawn")
    if not_shown:
        notes.append(f"{not_shown:,} zero or negative grade(s) not shown")
    if notes:
        title += "<br><sub>" + "; ".join(notes) + "</sub>"
    return _base_layout(fig, title, height=380 * nrows + 60)


# ---------------------------------------------------------------------------------------------
# classification and support
# ---------------------------------------------------------------------------------------------


def classification_bar(ctx: AuditContext) -> go.Figure | None:
    """Stacked horizontal bars of tonnes (or volume) by domain, split by class."""
    cls = ctx.bundle.get("classification") or {}
    inv = cls.get("inventory") or []
    if not inv:
        return None
    use_tonnes = all(r.get("tonnes") is not None for r in inv)
    key = "tonnes" if use_tonnes else "volume_m3"
    domains = sorted({str(r["domain"]) for r in inv})
    ranks = sorted({int(r["rank"]) for r in inv}, reverse=True)
    fig = go.Figure()
    for rank in ranks:
        label = next(str(r["class"]) for r in inv if int(r["rank"]) == rank)
        by_dom = {str(r["domain"]): _f(r.get(key)) or 0.0 for r in inv if int(r["rank"]) == rank}
        fig.add_trace(
            go.Bar(
                y=domains,
                x=[by_dom.get(d, 0.0) for d in domains],
                orientation="h",
                name=label,
                marker_color=CLASS_COLOURS.get(rank, "#9A9A9A"),
                hovertemplate="%{y}<br>" + label + ": %{x:,.0f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed", title="Domain", type="category")
    fig.update_xaxes(title="Tonnes" if use_tonnes else "Volume (m³)", tickformat=",.3s")
    return _base_layout(fig, "Classification by domain", height=max(300, 60 * len(domains) + 160))


def support_box(ctx: AuditContext) -> go.Figure | None:
    """Estimation details by class: one panel per detail found in the model."""
    stats = (ctx.bundle.get("support") or {}).get("stats_by_class") or []
    if not stats:
        return None
    present = [
        d
        for d in SUPPORT_DETAILS
        if any(_f((s.get("details", {}).get(d) or {}).get("p50")) is not None for s in stats)
    ]
    if not present:
        return None
    ncols = 2 if len(present) > 1 else 1
    nrows = math.ceil(len(present) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[SUPPORT_DETAILS[d] for d in present],
        vertical_spacing=0.14 if nrows > 1 else 0.1,
    )
    for n, detail in enumerate(present):
        r, c = divmod(n, ncols)
        for s in stats:
            d = s.get("details", {}).get(detail) or {}
            p50 = _f(d.get("p50"))
            if p50 is None:
                continue
            rank = int(s["rank"])
            fig.add_trace(
                go.Box(
                    x=[str(s["class"])],
                    q1=[_f(d.get("p10"))],
                    median=[p50],
                    q3=[_f(d.get("p90"))],
                    lowerfence=[_f(d.get("min"))],
                    upperfence=[_f(d.get("max"))],
                    mean=[_f(d.get("mean"))],
                    name=str(s["class"]),
                    marker_color=CLASS_COLOURS.get(rank, "#9A9A9A"),
                    showlegend=False,
                ),
                row=r + 1,
                col=c + 1,
            )
    if not fig.data:
        return None
    fig.update_layout(boxmode="overlay")
    return _base_layout(
        fig,
        "Estimation details by class (box: 10th to 90th percentile; whiskers: min to max)",
        height=330 * nrows + 70,
    )


# ---------------------------------------------------------------------------------------------
# grade distributions
# ---------------------------------------------------------------------------------------------


def grade_cdfs(ctx: AuditContext) -> dict[str, go.Figure]:
    """One CDF figure per element (``cdf_<el>``), by domain, with composites if loaded."""
    out: dict[str, go.Figure] = {}
    if "_VOL" not in ctx.bm.columns or not ctx.bundle.get("grades", {}).get("elements"):
        return out
    bm = ctx.bm
    w, basis = weights(ctx)
    dom = domain_labels(bm)
    analysed = ctx.analysis_mask
    colours = domain_colours(ctx)
    comps = ctx.comps
    cdom = domain_labels(comps) if comps is not None else None
    clen = (
        comps["length"].to_numpy(dtype=np.float64)
        if comps is not None and "length" in comps.columns
        else None
    )
    for el in ctx.bundle["grades"]["elements"]:
        if el not in bm.columns:
            continue
        grade = bm[el].to_numpy(dtype=np.float64)
        domains = ctx.bundle["grades"]["elements"][el].get(
            "applicable_domains"
        ) or applicable_domains(ctx, el)
        fig = go.Figure()
        for d in domains:
            pts = _cdf_points(grade[analysed & (dom == d)], w[analysed & (dom == d)])
            if pts is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=pts[0],
                    y=pts[1],
                    mode="lines",
                    name=f"{d} (model)",
                    legendgroup=str(d),
                    line={"color": colours.get(str(d)), "width": 2},
                )
            )
            if comps is not None and f"_USED_{el}" in comps.columns:
                m = cdom == d
                cw = clen[m] if clen is not None else None
                cpts = _cdf_points(comps[f"_USED_{el}"].to_numpy(dtype=np.float64)[m], cw)
                if cpts is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=cpts[0],
                            y=cpts[1],
                            mode="lines",
                            name=f"{d} (composites)",
                            legendgroup=str(d),
                            line={"color": colours.get(str(d)), "width": 2, "dash": "dash"},
                        )
                    )
        if not fig.data:
            continue
        fig.update_xaxes(title=f"{el.upper()} grade ({unit_label(ctx, el)}), log scale", type="log")
        fig.update_yaxes(title="Share of material below this grade", tickformat=".0%", range=[0, 1])
        weight_text = "tonnes" if basis == "tonnes" else "volume"
        extra = "; composites: length-weighted, capped" if comps is not None else ""
        out[f"cdf_{el}"] = _base_layout(
            fig,
            f"{el.upper()} cumulative distribution (model: weighted by {weight_text}{extra})",
        )
    return out


# ---------------------------------------------------------------------------------------------
# composites: grade-tonnage curves and swaths
# ---------------------------------------------------------------------------------------------


def gt_figures(ctx: AuditContext) -> dict[str, go.Figure]:
    """Grade-tonnage curves, model against nearest-neighbour: ``gt_<domain>|<el>``."""
    out: dict[str, go.Figure] = {}
    curves = (ctx.bundle.get("composites") or {}).get("gt_curves") or {}
    for key, c in curves.items():
        domain, _, el = key.rpartition("|")
        cut = c.get("cutoffs") or []
        if not cut:
            continue
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        for name, field, colour, dash, secondary in (
            ("Model tonnes", "bm_tonnes", BM_COLOUR, "solid", False),
            ("Nearest-neighbour tonnes", "nn_tonnes", NN_COLOUR, "solid", False),
            ("Model grade", "bm_grade", BM_COLOUR, "dot", True),
            ("Nearest-neighbour grade", "nn_grade", NN_COLOUR, "dot", True),
        ):
            fig.add_trace(
                go.Scatter(
                    x=cut,
                    y=c.get(field),
                    mode="lines",
                    name=name,
                    line={"color": colour, "dash": dash, "width": 2},
                ),
                secondary_y=secondary,
            )
        unit = unit_label(ctx, el)
        fig.update_xaxes(title=f"{el.upper()} cutoff grade ({unit})")
        fig.update_yaxes(title_text="Tonnes above cutoff", tickformat=",.3s", secondary_y=False)
        fig.update_yaxes(
            title_text=f"Mean {el.upper()} grade above cutoff ({unit})",
            showgrid=False,
            secondary_y=True,
        )
        out[f"gt_{key}"] = _base_layout(
            fig, f"{domain}: {el.upper()} grade-tonnage (solid: tonnes, dotted: grade)"
        )
    return out


def swath_figures(ctx: AuditContext) -> dict[str, go.Figure]:
    """Swath plots, three stacked rows (model X, Y, Z): ``swath_<domain>|<el>``."""
    out: dict[str, go.Figure] = {}
    swaths = (ctx.bundle.get("composites") or {}).get("swaths") or {}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for key, bins in swaths.items():
        name, _, axis = key.rpartition("|")  # "<domain>|<el>", "<axis>"
        grouped.setdefault(name, {})[axis] = bins
    for name, by_axis in grouped.items():
        _, _, el = name.rpartition("|")
        unit = unit_label(ctx, el)
        fig = make_subplots(
            rows=3,
            cols=1,
            specs=[[{"secondary_y": True}]] * 3,
            vertical_spacing=0.09,
            subplot_titles=[AXIS_NAMES[a] for a in ("x", "y", "z")],
        )
        for row, axis in enumerate(("x", "y", "z"), start=1):
            bins = by_axis.get(axis) or []
            if not bins:
                continue
            first = row == 1
            centre = [b["centre"] for b in bins]
            low = [bool(b.get("low_support")) for b in bins]
            fig.add_trace(
                go.Bar(
                    x=centre,
                    y=[b["tonnes"] for b in bins],
                    width=[b["bin_to"] - b["bin_from"] for b in bins],
                    marker_color=[GREY_LIGHT if x else GREY for x in low],
                    name="Tonnes (light grey: very few tonnes)",
                    legendgroup="tonnes",
                    showlegend=first,
                    hovertemplate="Tonnes: %{y:,.0f}<extra></extra>",
                ),
                row=row,
                col=1,
                secondary_y=False,
            )
            for label, field, colour, dash in (
                ("Model grade", "bm_grade", BM_COLOUR, "solid"),
                ("Composite mean", "comp_mean", COMP_COLOUR, "dash"),
                ("Nearest neighbour", "nn_grade", NN_COLOUR, "dot"),
            ):
                custom = [b["comp_count"] for b in bins] if field == "comp_mean" else None
                hover = f"{label}: %{{y:.4g}}"
                if custom:
                    hover += "<br>Composites: %{customdata}"
                fig.add_trace(
                    go.Scatter(
                        x=centre,
                        y=[b.get(field) for b in bins],
                        mode="lines",
                        name=label,
                        legendgroup=field,
                        showlegend=first,
                        line={"color": colour, "dash": dash, "width": 2},
                        customdata=custom,
                        hovertemplate=hover + "<extra></extra>",
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )
            if any(low):
                fig.add_trace(
                    go.Scatter(
                        x=[b["centre"] for b in bins if b.get("low_support")],
                        y=[b.get("bm_grade") for b in bins if b.get("low_support")],
                        mode="markers",
                        name="Model grade (low-tonnage bin)",
                        legendgroup="low",
                        showlegend=first,
                        marker={
                            "color": "rgba(0,0,0,0)",
                            "line": {"color": "#777", "width": 1.5},
                            "size": 7,
                        },
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                    secondary_y=True,
                )
            fig.update_yaxes(
                title_text="Tonnes", tickformat=".2s", secondary_y=False, row=row, col=1
            )
            fig.update_yaxes(
                title_text=f"{el.upper()} ({unit})",
                showgrid=False,
                secondary_y=True,
                row=row,
                col=1,
            )
        if not fig.data:
            continue
        domain = name.rpartition("|")[0]
        out[f"swath_{name}"] = _base_layout(fig, f"{domain}: {el.upper()} swaths", height=900)
    return out


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------


def build_figures(ctx: AuditContext) -> dict[str, go.Figure]:
    """Build every figure the available results allow and store them in ``ctx.figures``.

    A figure that fails to draw is skipped with a warning; the rest of the report is unaffected.
    """
    builders: list[tuple[str, Callable[[AuditContext], Any]]] = [
        ("density_box", density_box),
        ("density_vs_grade", density_vs_grade),
        ("classification_bar", classification_bar),
        ("support_box", support_box),
        ("grade CDFs", grade_cdfs),
        ("grade-tonnage curves", gt_figures),
        ("swaths", swath_figures),
    ]
    for label, builder in builders:
        try:
            result = builder(ctx)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not stop the report
            logger.exception("Figure %s failed", label)
            ctx.warn("warning", "report", f"The figure '{label}' could not be drawn ({exc}).")
            continue
        if isinstance(result, go.Figure):
            ctx.figures[label] = result
        elif isinstance(result, dict):
            ctx.figures.update(result)
    return ctx.figures


def figure_names(prefix: str, figures: Iterable[str]) -> list[str]:
    """Names of the figures that start with ``prefix``, in order."""
    return [n for n in figures if n.startswith(prefix)]
