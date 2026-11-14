"""Figures for the report (spec 11).

Every figure is drawn from the summaries the check modules put in ``ctx.bundle``. The only
exceptions are the density-versus-grade scatter and the grade CDFs, which read block rows. The
scatter is cut down to ``report.max_scatter_points`` points (stratified by domain, fixed random
seed) and every CDF is reduced to about 200 points.

:func:`build_figures` returns plotly figures keyed by name:

* ``density_box``, ``density_vs_grade``: density by domain; density against each grade;
* ``classification_bar``: tonnage by domain, split by class;
* ``support_box``: estimation details by class;
* ``cdf_<el>``: grade distribution by domain for one element.

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

from qaqc.config import CLASS_ORDER
from qaqc.domains import applicable_domains, domain_labels, weights
from qaqc.stats import clean_pairs
from qaqc.units import UNIT_LABELS, measure_units

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
CLASS_COLOURS = {3: "#0072B2", 2: "#56B4E9", 1: "#E69F00", 4: "#CC79A7", 0: "#9A9A9A"}
CDF_POINTS = 200

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
    return UNIT_LABELS.get(es.unit, es.unit)


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
    fig.update_yaxes(title=f"Density ({measure_units(ctx.cfg).density})")
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
                        f"{code}<br>{el.upper()}: %{{x:.4g}}<br>Density: %{{y:.4g}}<extra></extra>"
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
        fig.update_yaxes(
            title_text=f"Density ({measure_units(ctx.cfg).density})" if c == 0 else None,
            row=r + 1,
            col=c + 1,
        )
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
    key = "tonnes" if use_tonnes else "volume"
    domains = sorted({str(r["domain"]) for r in inv})
    present = {int(r["rank"]) for r in inv}
    ranks = [r for r in CLASS_ORDER if r in present]
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
    mu = measure_units(ctx.cfg)
    fig.update_xaxes(
        title=mu.tonnage_word if use_tonnes else f"Volume ({mu.volume})", tickformat=",.3s"
    )
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
    """One CDF figure per element (``cdf_<el>``), by domain."""
    out: dict[str, go.Figure] = {}
    if "_VOL" not in ctx.bm.columns or not ctx.bundle.get("grades", {}).get("elements"):
        return out
    bm = ctx.bm
    w, basis = weights(ctx)
    dom = domain_labels(bm)
    analysed = ctx.analysis_mask
    colours = domain_colours(ctx)
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
                    name=str(d),
                    legendgroup=str(d),
                    line={"color": colours.get(str(d)), "width": 2},
                )
            )
        if not fig.data:
            continue
        fig.update_xaxes(title=f"{el.upper()} grade ({unit_label(ctx, el)}), log scale", type="log")
        fig.update_yaxes(title="Share of material below this grade", tickformat=".0%", range=[0, 1])
        weight_text = "tonnage" if basis == "tonnes" else "volume"
        out[f"cdf_{el}"] = _base_layout(
            fig, f"{el.upper()} cumulative distribution (weighted by {weight_text})"
        )
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
