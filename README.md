# Block model QA/QC

Automated checks for mineral resource block models exported from Datamine, Isatis and
Leapfrog. The checks cover grid and geometry, density, grades, classification, estimation
support and composite validation. The outputs are an offline HTML report, a flagged-block CSV
for Leapfrog and a machine-readable audit file.

The full specification is in `block_model_qaqc_spec.md`. Choices made along the way are in
`DECISIONS.md`.

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Settings file, column recognition, header reading, synthetic test models | done |
| M1 | Reading the data, placeholder clean-up, grid set-up, geometry, density | done, awaiting review |
| M2 | Grades, units, classification, resource statement, estimation support | |
| M3 | Composites, bias, grade-tonnage curves, swaths | |
| M4 | Flag export, figures, HTML report, audit file | |
| M5 | Notebook, command line run, real model test | notebook started (grows with each milestone) |

## Setup

Works on Windows, macOS and Linux. Python 3.11 or newer is needed. The commands below call
the project's own Python directly, so there is nothing to "activate" (this avoids the
PowerShell script-permission error that activation can cause on work PCs).

**Windows (PowerShell or Command Prompt)**

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev,notebook]"
```

**macOS / Linux**

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebook]"
```

(`notebook` installs JupyterLab for the notebook below; leave it out if you use VS Code.)

`uv` is a small installer ([install instructions](https://docs.astral.sh/uv/)); on Windows,
`winget install --id=astral-sh.uv -e` also works. No `uv`? Use conda instead:
`conda env create -f environment.yml`, then `conda activate bm_qaqc`.

Below, `python` means the project's Python: `.venv\Scripts\python.exe` on Windows,
`.venv/bin/python` on macOS and Linux (or plain `python` inside the conda environment).

Tips for Windows:
- In project YAML files, write paths with forward slashes (`C:/Projects/model.csv`) or in
  single quotes if you use backslashes (see the note in `config/project_example.yaml`).
- Close the CSV in Excel before running; Windows will not let two programs use it.
- Keep the project and your data outside synced folders such as OneDrive if you can; syncing
  can slow reading and lock the cache files.
- Exports with decimal commas (for example `0,5`) are not supported yet.

## Running the checks: the notebook

The easiest way to run everything is the notebook `notebooks/block_model_qaqc.ipynb`. It runs
each step in order, with a short note on what it does and what to check, and shows the results
as tables.

**Open it in JupyterLab:**

```powershell
.venv\Scripts\jupyter-lab.exe notebooks/block_model_qaqc.ipynb
```

(macOS / Linux: `.venv/bin/jupyter-lab notebooks/block_model_qaqc.ipynb`.) A browser tab
opens. Use **Run → Run All Cells**.

**Or open it in VS Code:** open the file, click **Select Kernel** at the top right, choose the
`.venv` Python, then **Run All**. VS Code needs its Python and Jupyter extensions.

The notebook starts in **demo mode**: it makes a small made-up model with known problems, so
you can see every step working before using real data. To use your own model, copy
`config/project_example.yaml`, set the paths in it, then in Step 0 of the notebook set
`DEMO = False` and `CONFIG_PATH` to your file. If a step finds a problem with the settings or
the file, it stops with a short message saying what to fix.

## Command line and Python (for scripting)

Check how the columns of a block model CSV are recognised:

```
python -m qaqc schema path/to/block_model.csv --config config/project_example.yaml
```

Run the checks built so far (grid, geometry and density) from Python:

```python
from qaqc import config
from qaqc.context import prepare
from qaqc.modules import density, geometry

ctx = prepare(config.load("config/my_project.yaml"))
geometry.run(ctx)
density.run(ctx)
ctx.flags.counts()          # number of flagged blocks per check
ctx.bundle["geometry"]      # grid, volume and tonnage results
[w.message for w in ctx.warnings]
```

Rotated block models are not supported yet. The run stops with a message if one is found.

Time the checks on a large made-up model (about 5 million blocks):

```
python scripts/benchmark.py qaqc_benchmark
```

(The folder name can be anything; it is created if needed and holds about 400 MB.)

Run the tests:

```
python -m pytest
```

## Project settings

Copy `config/project_example.yaml` for each project. Every setting is explained in the file.
A misspelled setting is reported as an error rather than ignored.
