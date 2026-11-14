"""The whole run in order: read, check, export (and, once built, the report).

Used by the command line (``python -m qaqc run``) and available to scripts. The notebook runs
the same steps one cell at a time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType

from qaqc import config, export
from qaqc.context import AuditContext, prepare
from qaqc.modules import classification, density, geometry, grades, support

logger = logging.getLogger(__name__)

# Order matters: geometry adds block volumes and tonnes that the later modules use.
MODULES: dict[str, ModuleType] = {
    "geometry": geometry,
    "density": density,
    "grades": grades,
    "classification": classification,
    "support": support,
}


def run_checks(ctx: AuditContext, modules: list[str] | None = None) -> AuditContext:
    """Run the check modules (all of them, in order, unless ``modules`` names some)."""
    names = list(MODULES) if modules is None else modules
    unknown = [n for n in names if n not in MODULES]
    if unknown:
        raise ValueError(f"Unknown check module(s): {unknown}. Known: {list(MODULES)}")
    if names and names[0] != "geometry" and "_VOL" not in ctx.bm.columns:
        names = ["geometry", *names]
    for name in MODULES:  # keep the fixed order whatever order was asked for
        if name in names:
            logger.info("Running %s checks", name)
            MODULES[name].run(ctx)
    return ctx


def write_report(ctx: AuditContext) -> Path | None:
    """Write the HTML report if the report step is available (milestone 4)."""
    try:
        from qaqc import report  # noqa: PLC0415 (optional until the report is built)
    except ImportError:
        return None
    return report.write_report(ctx)


def run_project(config_path: str | Path, use_cache: bool = True) -> tuple[AuditContext, dict]:
    """Load the settings, run every check and write every output.

    Returns the run context and the output files (name -> path).
    """
    cfg = config.load(config_path)
    ctx = prepare(cfg, use_cache=use_cache)
    run_checks(ctx)
    report_path = write_report(ctx)
    files = export.write_outputs(ctx, report_path=report_path)
    return ctx, files
