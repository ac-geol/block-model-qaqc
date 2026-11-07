"""Command line interface.

python -m qaqc schema <file.csv> [--config project.yaml] [--software datamine|isatis|leapfrog]
    Show which column was matched to what, for review.

python -m qaqc run <project.yaml>
    Run all checks and write the outputs (available from milestone 5).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from qaqc import config, schema


def _schema_command(args: argparse.Namespace) -> int:
    if args.config:
        cfg = config.load(args.config)
    else:
        cfg = config.from_dict(
            {"project_name": "column check", "paths": {"block_model_csv": str(args.csv)}}
        )
    if args.software:
        cfg = cfg.model_copy(update={"source_software": args.software})
    try:
        s = schema.resolve_file(args.csv, cfg)
    except (schema.SchemaError, schema.MissingColumnError) as err:
        print(f"\nProblem matching columns in {args.csv}:\n{err}\n", file=sys.stderr)
        return 2

    h = s.header
    print(f"\nFile: {args.csv}")
    print(f"Software: {s.software or 'not recognised'} (detected: {h.detection_reason})")
    print(f"Header lines above the column names: {h.skiprows}")
    if h.parsed:
        grid_values = {k: v for k, v in h.parsed.items() if k != "notes"}
        print(f"Grid values read from the header: {grid_values}")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("\nColumns matched:\n")
        print(s.describe().to_string(index=False))
    if s.unmatched_columns:
        print("\nColumns not used:")
        for col in s.unmatched_columns:
            print(f"  - {col}: {s.unmatched_reasons.get(col, '')}")
    notes = list(s.notes) + list(h.parsed.get("notes", []))
    if notes:
        print("\nNotes:")
        for note in notes:
            print(f"  - {note}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qaqc",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show detailed log messages")
    sub = parser.add_subparsers(dest="command", required=True)

    p_schema = sub.add_parser("schema", help="show how the columns of a CSV file are matched")
    p_schema.add_argument("csv", type=Path)
    p_schema.add_argument("--config", type=Path, help="project YAML (for overrides and elements)")
    p_schema.add_argument("--software", choices=list(schema.SOFTWARE))

    p_run = sub.add_parser("run", help="run all checks (available from milestone 5)")
    p_run.add_argument("config", type=Path)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.command == "schema":
        return _schema_command(args)
    print("The full run is not built yet (milestone 5).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
