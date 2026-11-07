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
| M0 | Settings file, column recognition, header reading, synthetic test models | done, awaiting review |
| M1 | Reading the data, placeholder clean-up, grid and rotation, geometry, density | next |
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

Run the tests:

```bash
.venv/bin/python -m pytest
```

## Project settings

Copy `config/project_example.yaml` for each project. Every setting is explained in the file.
A misspelled setting is reported as an error rather than ignored.
