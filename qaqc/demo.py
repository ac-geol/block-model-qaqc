"""A small made-up project for trying the notebook without a real block model.

It uses the synthetic model generator in ``tests/synthetic.py``, so it needs this repository
(it is not part of an installed package). The model has known problems planted in it, so most
checks find something to report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"


def make_demo_project(
    folder: str | Path,
    kind: str = "sub_blocked",
    style: str = "datamine",
    seed: int = 0,
) -> Path:
    """Write a demo block model, composites and project YAML into ``folder``.

    Args:
        folder: Where to write the files (created if needed; existing demo files are replaced).
        kind: ``regular``, ``sub_blocked`` or ``percentage``.
        style: Column style: ``generic``, ``datamine``, ``isatis`` or ``leapfrog``.
        seed: Random seed (the same seed gives the same model).

    Returns:
        Path of the project YAML file.
    """
    if not (TESTS_DIR / "synthetic.py").is_file():
        raise FileNotFoundError(
            f"The demo needs tests/synthetic.py from the project repository ({TESTS_DIR})."
        )
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    import synthetic  # noqa: PLC0415 (only available inside the repository)

    folder = Path(folder)
    m = synthetic.make_model(kind, style=style, seed=seed)
    m.write(folder)
    data = m.config_dict()
    data["project_name"] = f"Demo: made-up {kind.replace('_', '-')} model ({style} style)"
    cfg_path = folder / "demo_project.yaml"
    header = (
        "# Demo project written by qaqc.demo.make_demo_project. Made-up data with planted\n"
        "# problems. It is replaced each time the demo runs, so edit a copy instead.\n"
    )
    cfg_path.write_text(header + yaml.safe_dump(_plain(data), sort_keys=False), encoding="utf-8")
    return cfg_path


def _plain(obj):
    """Tuples to lists so the YAML stays readable."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj
