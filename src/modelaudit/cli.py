"""Public configuration-driven CLI with explicit completion/outcome exit codes."""

import argparse
import csv
from importlib.metadata import version
import logging
import subprocess
from pathlib import Path

from .audit_config import load_audit_config
from .experiment_suite import run_experiments
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
    experiments = subcommands.add_parser("experiments", help="Run controlled defects on copies; preserve source artifacts.", allow_abbrev=False)
    experiments.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    experiments.add_argument("--models", type=Path, default=Path("models/baseline"))
    experiments.add_argument("--output", type=Path, required=True)
    experiments.add_argument("--quiet", action="store_true")
    experiments.add_argument("--mode", choices=("saved", "fitted"), default="saved")
    experiments.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "check-config":
            config = load_audit_config(args.config)
            print(f"Configuration valid: {', '.join(config['checks'])}. Input data has not been audited.")
            return 0
        configure_logging(logging.ERROR if args.quiet else logging.INFO)
        if args.command == "experiments":
            if args.mode == "fitted":
                from .cli_experiments import run_cli_experiments
                summary = run_cli_experiments(args.output, args.seed)
            else:
                summary = run_experiments(args.prepared, args.models, args.output)
            matched = sum(case["expectations_met"] for case in summary["scenarios"])
            print(f"Experiments complete: {matched}/{len(summary['scenarios'])} scenarios matched expectations; "
                  f"report={args.output.resolve() / ('summary.json' if args.mode == 'fitted' else 'summary.html')}")
            return summary["exit_code"]
        report = run_audit(args.config, args.output)
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as exc:
        label = "Experiments" if args.command == "experiments" else "Audit"
        parser.exit(2, f"{label} could not complete: {exc}\n")
    decision = report["decision"]
    print(f"Audit complete: {decision['status'].upper()}; report={args.output.resolve() / 'summary.html'}")
    return decision["exit_code"]
