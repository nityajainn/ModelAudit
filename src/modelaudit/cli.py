"""Public configuration-driven CLI with explicit completion/outcome exit codes."""

import argparse
import csv
from importlib.metadata import version
import logging
from pathlib import Path

from .audit_config import load_audit_config
from .logging import configure_logging
from .runner import run_audit


def main(argv=None):
    parser = argparse.ArgumentParser(prog="modelaudit", description="Run offline model checks and create review reports.", allow_abbrev=False)
    parser.add_argument("--version", action="version", version=f"ModelAudit {version('modelaudit')}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    check = subcommands.add_parser("check-config", help="Validate configuration without auditing data.", allow_abbrev=False)
    check.add_argument("--config", type=Path, required=True)
    run = subcommands.add_parser("run", help="Run configured checks into a new output directory.", allow_abbrev=False)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "check-config":
            config = load_audit_config(args.config)
            print(f"Configuration valid: {', '.join(config['checks'])}. Input data has not been audited.")
            return 0
        configure_logging(logging.ERROR if args.quiet else logging.INFO)
        report = run_audit(args.config, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Audit could not complete: {exc}\n")
    decision = report["decision"]
    print(f"Audit complete: {decision['status'].upper()}; report={args.output.resolve() / 'summary.html'}")
    return decision["exit_code"]
