# Block model QA/QC

A high-level check of mineral resource block models exported from Datamine, Isatis and
Leapfrog. It looks at the model grid and block geometry, density, grades, classification and
the estimation details. The results are an offline HTML report, a file of flagged blocks to
view in Leapfrog, and an audit file. Metric and imperial models are supported.

**New to this? Start with [GETTING_STARTED.md](GETTING_STARTED.md).** It is a step-by-step
guide for Windows that needs no programming knowledge.

The original specification is in `block_model_qaqc_spec.md`. Changes from it, and the reasons,
are in `DECISIONS.md`.

## What is checked

| Check | What it looks for |
|---|---|
| Grid and geometry | Duplicate blocks, blocks off the grid, sub-blocks overfilling their parent block, percentages above 100%; total volume and tonnage |
| Density | Missing density, or density outside the allowed range |
| Grades | Negative grades, grades above a ceiling, missing grades, grade in waste above a cutoff; statistics and contained metal by domain |
| Classification | Codes not recognised, unclassified mineralised blocks; tonnage, grade and metal by class; the resource statement rebuilt above a cutoff |
| Estimation details | Measured and Indicated blocks that fail your classification criteria (search pass, samples, holes, slope of regression, kriging efficiency) |

Composite checks (bias against composites, grade-tonnage curves, swath plots) were removed in
November 2026 to keep the tool a simple, high-level check.

## Status

| Milestone | Content | State |
|---|---|---|
| M0 | Settings file, column recognition, header reading, synthetic test models | done |
| M1 | Reading the data, placeholder clean-up, grid set-up, geometry, density | done |
| M2 | Grades, units, classification, resource statement, estimation support | done |
| M3 | Composites, bias, grade-tonnage curves, swaths | removed (Nov 2026) |
| M4 | Flag export, figures, HTML report, audit file | done |
| M5 | Notebook, command line run, real model test | real model test to do |

## Outputs

Written to the output folder (`paths: output_dir` in the settings file):

| File | What it is |
|---|---|
| `BM_QAQC_Report.html` | Full report with tables and charts. Opens in any browser, works offline. |
| `flagged_blocks_leapfrog.csv` | Every flagged block, to import into Leapfrog as points. |
| `audit_bundle.json` | Every result in one machine-readable file. |
| `model_flags.parquet` | Every block and every flag (only with `export: full_model_flags: true`). |

## Settings

- `config/project_example.yaml` has the settings most projects need, with a short note on
  each. Copy it for each project.
- `config/advanced_settings_reference.yaml` lists every other setting with its default.
- A misspelled setting is reported, not ignored. A setting from an older version gets a
  message saying what it is now called.

## Setup

Windows: follow [GETTING_STARTED.md](GETTING_STARTED.md). In short, install `uv`, then
double-click `setup_windows.bat`.

From a terminal on any system (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev,notebook]"
```

On Windows the Python path is `.venv\Scripts\python.exe` instead of `.venv/bin/python`. Below,
`python` means that path. The `dev` extra adds the test tools. No `uv`? Use conda instead:
`conda env create -f environment.yml`.

## Running the checks

**Windows, no terminal.** Drag your settings file onto `run_qaqc.bat`. `run_demo.bat` tries
the tool on a made-up model, and `run_notebook.bat` opens the notebook.

### The notebook

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

### Command line and Python (for scripting)

Run everything and write all the outputs in one go:

```
python -m qaqc run config/my_project.yaml --open
```

It prints the errors and warnings, the flagged-block counts and the files written. `--open`
opens the report when the run finishes. Add `--no-cache` to read the CSV again, and `-v` for
detailed progress. `python -m qaqc demo --open` runs everything on a made-up model.

Check how the columns of a block model CSV are recognised (fast; reads only the top):

```
python -m qaqc schema path/to/block_model.csv --config config/project_example.yaml
```

From Python:

```python
from qaqc import pipeline

ctx, files = pipeline.run_project("config/my_project.yaml")
ctx.flags.counts()          # number of flagged blocks per check
ctx.bundle["grades"]        # results of one check (also in audit_bundle.json)
[w.message for w in ctx.warnings]
```

Rotated block models are not supported yet. The run stops with a message if one is found.

Time the whole run on a large made-up model (about 5 million blocks):

```
python scripts/benchmark.py qaqc_benchmark
```

(The folder name can be anything. It is created if needed and holds about 400 MB.)

Run the tests:

```
python -m pytest
```
