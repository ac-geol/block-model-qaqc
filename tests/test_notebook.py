"""The notebook runs from start to finish in demo mode, and user errors show as short messages."""

from __future__ import annotations

from pathlib import Path

import pytest

from qaqc import display as show

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "block_model_qaqc.ipynb"


def test_notebook_runs_in_demo_mode():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    settings = next(c for c in nb.cells if c.cell_type == "code")
    assert "DEMO = True" in settings.source, "the notebook should ship in demo mode"

    client = nbclient.NotebookClient(
        nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": NOTEBOOK.parent}}
    )
    client.execute()  # raises CellExecutionError if any cell fails

    text = "".join(
        str(out.get("data", {}).get("text/plain", "")) + str(out.get("text", ""))
        for cell in nb.cells
        if cell.cell_type == "code"
        for out in cell.get("outputs", [])
    )
    assert "FLAG_DUP_CENTROID" in text
    assert "FLAG_SG_ANOMALY" in text


def test_missing_config_stops_with_short_message(tmp_path):
    with pytest.raises(show.StopRun) as info:
        show.load_config(tmp_path / "missing.yaml")
    lines = info.value._render_traceback_()
    assert lines[0] == "STOPPED at step 'settings':"
    assert any("not found" in line for line in lines)
