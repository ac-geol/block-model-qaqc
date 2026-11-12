"""Export of flags, tables and the audit bundle (spec 9 and 10).

One entry point, :func:`write_outputs`, writes everything into ``paths.output_dir``:

* ``flagged_blocks_leapfrog.csv``: blocks with at least one flag, as points for Leapfrog;
* ``flagged_composites.csv``: composites with at least one flag (only if composite flags exist);
* one ``<stem>.csv`` for every table in ``ctx.tables`` (for example ``composite_domain_mismatch``);
* ``model_flags.parquet``: every block and every flag (only if ``export.full_model_flags``);
* ``audit_bundle.json``: all results in one machine-readable file.

The HTML report is written by the report step. Pass its path as ``report_path`` so it is listed
in the bundle and in the returned files.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

import qaqc
from qaqc.context import jsonable
from qaqc.domains import class_ranks
from qaqc.flags import FlagRegistry

if TYPE_CHECKING:
    from qaqc.context import AuditContext

logger = logging.getLogger(__name__)

BLOCKS_CSV = "flagged_blocks_leapfrog.csv"
COMPOSITES_CSV = "flagged_composites.csv"
FULL_FLAGS = "model_flags.parquet"
BUNDLE_JSON = "audit_bundle.json"
SCHEMA_VERSION = 1

# Standard column name in ``ctx.bm`` -> name in the export file
_BLOCK_COLUMNS = {
    "x": "X",
    "y": "Y",
    "z": "Z",
    "dx": "DX",
    "dy": "DY",
    "dz": "DZ",
    "domain": "DOMAIN",
    "classif": "CLASSIF",
}
_PACKAGES = ("pandas", "numpy", "scipy", "pyarrow", "pydantic", "plotly", "jinja2", "pyyaml")


class ExportError(RuntimeError):
    """An output file could not be written (plain-language message)."""


# --------------------------------------------------------------------------- file helpers


def _open_for_write(path: Path) -> IO[bytes]:
    return open(path, "wb")  # noqa: SIM115 - closed by the caller


def _write_file(path: Path, writer: Callable[[IO[bytes]], None]) -> None:
    """Open ``path`` for writing and call ``writer``; turn a locked file into a clear error."""
    try:
        with _open_for_write(path) as f:
            writer(f)
    except PermissionError:
        raise ExportError(
            f"Cannot write '{path}'. The file is probably open in Excel or another program. "
            "Close it and run the export again."
        ) from None


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame as CSV with pyarrow (fast for millions of rows); missing = empty."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    _write_file(path, lambda f: pv.write_csv(table, f))


# --------------------------------------------------------------------------- flag tables


def flag_reasons(active: pd.DataFrame) -> pd.Series:
    """Text like ``DUP_CENTROID;OFF_GRID`` per row from 0/1 flag columns (no row-wise apply)."""
    reasons = np.full(len(active), "", dtype=object)
    for col in active.columns:
        label = col.removeprefix("FLAG_") + ";"
        reasons = reasons + np.where(active[col].to_numpy().astype(bool), label, "")
    return pd.Series(reasons, index=active.index, dtype=object).str.rstrip(";")


def _flag_columns(flags: FlagRegistry, rows: np.ndarray) -> pd.DataFrame:
    """0/1 columns for flags with a hit, plus FLAG_COUNT and FLAG_REASONS."""
    active = flags.to_frame(rows)
    out = active.copy()
    out["FLAG_COUNT"] = active.sum(axis=1).astype(np.int32) if len(active.columns) else 0
    out["FLAG_REASONS"] = flag_reasons(active)
    return out


def flagged_blocks(ctx: AuditContext, only_flagged: bool = True) -> pd.DataFrame:
    """Table of flagged blocks for Leapfrog (spec 9). ``only_flagged=False`` keeps all rows."""
    bm = ctx.bm
    rows = np.flatnonzero(ctx.flags.any()) if only_flagged else np.arange(len(bm))
    data: dict[str, Any] = {}
    if "_ROW" in bm.columns:
        data["_ROW"] = bm["_ROW"].to_numpy()[rows]
    for std, name in _BLOCK_COLUMNS.items():
        if std in bm.columns:
            col = bm[std].to_numpy()[rows] if std not in ("domain", "classif") else None
            if col is None:
                col = bm[std].iloc[rows].astype(object).to_numpy()
            data[name] = col
    if "classif" in bm.columns:
        data["CLASS_RANK"] = class_ranks(bm, ctx.cfg)[0][rows]
    for el, es in ctx.schema.elements.items():
        if el in bm.columns:
            data[es.primary] = bm[el].to_numpy()[rows]
    df = pd.DataFrame(data)
    return pd.concat([df, _flag_columns(ctx.flags, rows)], axis=1)


def flagged_composites(ctx: AuditContext) -> pd.DataFrame | None:
    """Table of flagged composites, or None if there are no composite flags."""
    if ctx.comps is None or ctx.comp_flags is None:
        return None
    rows = np.flatnonzero(ctx.comp_flags.any())
    comps = ctx.comps
    data: dict[str, Any] = {}
    if "_ROW" in comps.columns:
        data["_ROW"] = comps["_ROW"].to_numpy()[rows]
    else:
        data["_ROW"] = rows
    for std, name in _BLOCK_COLUMNS.items():
        if std in comps.columns and std not in ("dx", "dy", "dz", "classif"):
            col = comps[std].iloc[rows]
            data[name] = col.astype(object).to_numpy() if std == "domain" else col.to_numpy()
    df = pd.DataFrame(data)
    return pd.concat([df, _flag_columns(ctx.comp_flags, rows)], axis=1)


# --------------------------------------------------------------------------- bundle


def flag_catalog(flags: FlagRegistry, template: str, n_rows: int) -> list[dict[str, Any]]:
    """One record per registered flag: description, module, count, share and Leapfrog filter."""
    counts = flags.counts()
    out = []
    for name, info in flags.info().items():
        n = counts[name]
        out.append(
            {
                "name": name,
                "description": info.description,
                "module": info.module,
                "count": n,
                "pct": round(100.0 * n / n_rows, 6) if n_rows else 0.0,
                "leapfrog_filter": template.format(flag=name),
            }
        )
    return out


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_bundle(ctx: AuditContext, files: dict[str, Path]) -> dict[str, Any]:
    """Assemble the audit bundle (spec 10) from whatever the modules have put in ``ctx.bundle``.

    Modules that did not run simply do not appear. The result is plain Python (no numpy, no NaN).
    """
    raw = ctx.bundle
    template = ctx.cfg.report.leapfrog_filter_template
    n_rows = len(ctx.bm)
    n_flagged = int(ctx.flags.any().sum())
    inputs = [{"role": role, **info} for role, info in raw.get("inputs", {}).items()]
    schema = raw.get("schema") or ctx.schema.to_bundle()

    reserved = {"config", "schema", "inputs", "ingest", "timings", "grid", "flags", "metadata"}
    bundle: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "project_name": ctx.cfg.project_name,
            "run_timestamp": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "qaqc_version": qaqc.__version__,
            "package_versions": _package_versions(),
            "config_snapshot": raw.get("config") or ctx.cfg.snapshot(),
            "source_software": schema.get("software"),
            "source_software_detected": schema.get("software_detected"),
            "inputs": inputs,
            "timings_s": raw.get("timings", {}),
        },
        "schema": schema,
        "grid": raw.get("grid", {}),
        "warnings": [w.to_dict() for w in ctx.warnings],
        "ingestion": raw.get("ingest", {}),
    }
    for key, value in raw.items():
        if key not in reserved:
            bundle[key] = value
    flags: dict[str, Any] = {
        "catalog": flag_catalog(ctx.flags, template, n_rows),
        "total_flagged": n_flagged,
        "pct_flagged": round(100.0 * n_flagged / n_rows, 6) if n_rows else 0.0,
        "exports": {name: str(path) for name, path in files.items()},
    }
    if ctx.comp_flags is not None:
        flags["composite_catalog"] = flag_catalog(ctx.comp_flags, template, ctx.comp_flags.n_rows)
        flags["composites_flagged"] = int(ctx.comp_flags.any().sum())
    bundle["flags"] = flags
    return jsonable(bundle)


def _dump_json(path: Path, bundle: dict[str, Any]) -> None:
    text = json.dumps(bundle, indent=2, allow_nan=False, default=str)
    _write_file(path, lambda f: f.write(text.encode("utf-8")))


# --------------------------------------------------------------------------- entry point


def write_outputs(ctx: AuditContext, report_path: str | Path | None = None) -> dict[str, Path]:
    """Write every output file for this run and return them.

    Args:
        ctx: The audit context after the check modules have run.
        report_path: The HTML report, if the report step has written it. It is listed in the
            audit bundle and in the result; this function does not write it.

    Returns:
        File name -> full path, for every file written (the audit bundle is last).

    Raises:
        ExportError: If a file cannot be written, for example because it is open in Excel.
    """
    out_dir = Path(ctx.cfg.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    with ctx.timed("export"):
        path = out_dir / BLOCKS_CSV
        _write_csv(path, flagged_blocks(ctx))
        files[BLOCKS_CSV] = path

        comp = flagged_composites(ctx)
        if comp is not None:
            path = out_dir / COMPOSITES_CSV
            _write_csv(path, comp)
            files[COMPOSITES_CSV] = path

        for stem, table in ctx.tables.items():
            path = out_dir / f"{stem}.csv"
            _write_csv(path, table)
            files[path.name] = path

        if ctx.cfg.export.full_model_flags:
            frame = ctx.flags.to_frame(np.arange(len(ctx.bm)), only_active=False)
            if "_ROW" in ctx.bm.columns:
                frame.insert(0, "_ROW", ctx.bm["_ROW"].to_numpy())
            path = out_dir / FULL_FLAGS
            table = pa.Table.from_pandas(frame, preserve_index=False)
            _write_file(path, lambda f: pq.write_table(table, f))
            files[FULL_FLAGS] = path

    if report_path is not None:
        rp = Path(report_path)
        files[rp.name] = rp

    bundle_path = out_dir / BUNDLE_JSON
    files[BUNDLE_JSON] = bundle_path
    bundle = build_bundle(ctx, files)
    ctx.bundle["flags"] = bundle["flags"]
    _dump_json(bundle_path, bundle)
    logger.info("Wrote %d output file(s) to %s", len(files), out_dir)
    return files
