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
| M5 | Notebook, command line run, real model test | |

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

or with conda: `conda env create -f environment.yml`.

## Use (so far)

Check how the columns of a block model CSV are recognised:

```bash
.venv/bin/python -m qaqc schema path/to/block_model.csv --config config/project_example.yaml
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

```bash
.venv/bin/python scripts/benchmark.py /tmp/qaqc_benchmark
```

Run the tests:

```bash
.venv/bin/python -m pytest
```

## Project settings

Copy `config/project_example.yaml` for each project. Every setting is explained in the file.
A misspelled setting is reported as an error rather than ignored.
