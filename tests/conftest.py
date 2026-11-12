"""Shared test helpers."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from synthetic import SyntheticModel

from qaqc import config
from qaqc.context import AuditContext, prepare
from qaqc.modules import classification, density, geometry, grades, support

MODULES = {
    "geometry": geometry,
    "density": density,
    "grades": grades,
    "classification": classification,
    "support": support,
}

# (kind, style) pairs the generator supports
KIND_STYLES = [
    (k, s)
    for k, s in itertools.product(
        ("regular", "sub_blocked", "percentage"), ("generic", "datamine", "isatis", "leapfrog")
    )
    if not (s == "isatis" and k == "sub_blocked")
]


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Copy of ``base`` with nested values from ``extra`` written over it."""
    out = dict(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@pytest.fixture
def run_checks(tmp_path):
    """Write a synthetic model to a temporary folder, read it and run check modules.

    Usage: ``ctx = run_checks(model, modules=("geometry",), settings={"model": {...}})``.
    ``settings`` are merged into the model's matching project configuration.
    """
    counter = itertools.count()

    def _run(
        m: SyntheticModel,
        modules: tuple[str, ...] = ("geometry", "density"),
        settings: dict[str, Any] | None = None,
        use_cache: bool = False,
        folder_name: str | None = None,
    ) -> AuditContext:
        folder = tmp_path / (folder_name or f"{m.kind}_{m.style}_{next(counter)}")
        if not (folder / "block_model.csv").exists():
            m.write(folder)
        data = deep_update(m.config_dict(), settings or {})
        cfg = config.from_dict(data, base_dir=folder)
        ctx = prepare(cfg, use_cache=use_cache)
        for name in modules:
            MODULES[name].run(ctx)
        return ctx

    return _run
