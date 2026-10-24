"""Command line entry point for field studies and reproducible artifacts."""

import argparse
import json
from pathlib import Path

from . import __version__
from .experiments import CASES, ExperimentConfig, load_config, run


def parser():
    root = argparse.ArgumentParser(prog="emlab", description="Reproducible computational electromagnetics")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list available physical studies")
    single = commands.add_parser("run", help="solve one study and export a report")
    choice = single.add_mutually_exclusive_group(required=True)
    choice.add_argument("--case", choices=CASES)
    choice.add_argument("--config", type=Path)
    single.add_argument("--scale", type=int)
    single.add_argument("--seed", type=int)
    single.add_argument("--output", required=True, type=Path)
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.command == "list":
            for case, (_, _, description) in CASES.items():
                print(f"{case:28} {description}")
        elif args.command == "run":
            from .reporting import write_report

            if args.config:
                if args.scale is not None or args.seed is not None:
                    raise ValueError("config file cannot be combined with scale or seed overrides")
                config = load_config(args.config)
            else:
                config = ExperimentConfig(
                    args.case, 1 if args.scale is None else args.scale, 42 if args.seed is None else args.seed
                )
            result = run(config, print)
            report = write_report(result, args.output)
            print(json.dumps(result["summary"], indent=2))
            print("Report: " + str(report.resolve()))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        root.error(str(error))
    return 0
