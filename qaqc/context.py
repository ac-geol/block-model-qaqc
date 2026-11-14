"""Shared run context (spec 3, "Shared context") and the steps that build it.

Every check module exposes ``run(ctx)`` and writes only to its own key in ``ctx.bundle``, to
``ctx.flags``, ``ctx.figures``, ``ctx.tables`` and ``ctx.warnings``. Modules run in this
order: geometry, density, grades, classification, support (geometry first: it adds ``_VOL``
and ``_TONNES``).
"""

from __future__ import annotations

import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from qaqc import io, schema
from qaqc.config import ProjectConfig
from qaqc.flags import FlagRegistry
from qaqc.grid import GridInfo, setup_grid
from qaqc.schema import ResolvedSchema
from qaqc.units import measure_units

logger = logging.getLogger(__name__)

Severity = Literal["info", "warning", "error"]


@dataclass
class AuditWarning:
    """A message for the report. ``error`` means a result is missing or not trustworthy."""

    severity: Severity
    module: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"severity": self.severity, "module": self.module, "message": self.message}


@dataclass
class AuditContext:
    """Everything the check modules share."""

    cfg: ProjectConfig
    schema: ResolvedSchema
    grid: GridInfo
    bm: pd.DataFrame
    flags: FlagRegistry
    bundle: dict[str, Any] = field(default_factory=dict)
    figures: dict[str, Any] = field(default_factory=dict)
    warnings: list[AuditWarning] = field(default_factory=list)
    # Tables to write as CSV files by the export step: file stem -> DataFrame
    # Modules add to it; only export writes files.
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def warn(self, severity: Severity, module: str, message: str) -> None:
        """Add a message for the report (also written to the log)."""
        self.warnings.append(AuditWarning(severity, module, message))
        level = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}
        logger.log(level[severity], "[%s] %s", module, message)

    @property
    def analysis_mask(self) -> np.ndarray:
        """Rows used for volume, tonnage, statistics and the non-grid checks (not air)."""
        return ~self.bm["_AIR"].to_numpy(dtype=bool)

    @contextmanager
    def timed(self, name: str):
        """Record how long a step takes, in ``bundle['timings']``."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.bundle.setdefault("timings", {})[name] = round(time.perf_counter() - t0, 3)


def jsonable(obj: Any) -> Any:
    """Turn numpy / pandas values into plain Python so ``json.dumps`` accepts them.

    NaN and infinity become None.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return jsonable(obj.to_dict(orient="records"))
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def prepare(
    cfg: ProjectConfig,
    aliases: schema.Aliases | None = None,
    use_cache: bool = True,
) -> AuditContext:
    """Match columns, read and clean the block model, and set up the grid.

    This is everything that has to happen before the check modules run.
    """
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    resolved = schema.resolve_file(cfg.paths.block_model_csv, cfg, aliases)
    timings["column_matching"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    bm, report = io.read_block_model(cfg, resolved, use_cache=use_cache)
    bm, excluded = io.apply_exclusions(bm, cfg)
    timings["reading"] = round(time.perf_counter() - t0, 3)

    ctx = AuditContext(
        cfg=cfg,
        schema=resolved,
        grid=GridInfo.empty(),
        bm=bm,
        flags=FlagRegistry(len(bm)),
    )
    ctx.bundle["timings"] = timings
    ctx.bundle["config"] = cfg.snapshot()
    mu = measure_units(cfg)
    ctx.bundle["units"] = {
        "system": mu.system,
        "length": mu.length,
        "volume": mu.volume,
        "tonnage": mu.tonnage,
        "density": mu.density,
        "note": "Result keys named 'tonnes' hold tonnage in the unit above.",
    }
    ctx.bundle["schema"] = resolved.to_bundle()
    ctx.bundle["inputs"] = {
        "block_model": {
            "path": report.get("path", str(cfg.paths.block_model_csv)),
            "sha256": report.get("sha256"),
            "rows": report["rows_read"],
            "columns": list(report.get("columns", {}).values()),
        }
    }
    ctx.bundle["ingest"] = {
        "reader": report.get("reader"),
        "from_cache": report.get("from_cache", False),
        "sentinel_replacements": report.get("sentinel_replacements", {}),
        "coerced_values": report.get("coerced_values", {}),
        "excluded_rows": excluded,
        "rows_analysed": int(len(bm) - excluded["air"]),
    }
    for note in resolved.notes:
        ctx.warn("info", "columns", note)
    for col, n in report.get("sentinel_replacements", {}).items():
        ctx.warn("info", "reading", f"{n:,} placeholder value(s) in '{col}' treated as missing.")
    for col, n in report.get("coerced_values", {}).items():
        ctx.warn(
            "warning", "reading", f"{n:,} value(s) in '{col}' are not numbers; treated as missing."
        )
    for col in report.get("mostly_text_columns", []):
        ctx.warn(
            "error",
            "reading",
            f"Most values in '{col}' could not be read as numbers. Check the column and the "
            "decimal separator (a comma as decimal separator is not supported).",
        )
    if excluded["selection"]:
        ctx.warn(
            "info",
            "reading",
            f"{excluded['selection']:,} block(s) with selection = 0 left out of all checks.",
        )
    if excluded["air"]:
        ctx.warn(
            "info",
            "reading",
            f"{excluded['air']:,} air block(s) used in the grid checks only (not in volumes, "
            "tonnes or statistics).",
        )

    with ctx.timed("grid_setup"):
        setup_grid(ctx)
    return ctx
