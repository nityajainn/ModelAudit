"""Phase 5 snapshot comparison command; orchestration remains Phase 6."""

import argparse
import csv
from pathlib import Path

from modelaudit.drift_audit import audit_snapshot_files


def main():
    parser = argparse.ArgumentParser(description="Compare explicitly prepared offline snapshots.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/phase5-drift-audit.json"))
    parser.add_argument("--numeric", nargs="+", default=[])
    parser.add_argument("--categorical", nargs="+", default=[])
    parser.add_argument("--probability")
    parser.add_argument("--target")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-rows", type=int, default=30)
    parser.add_argument("--minimum-class-rows", type=int, default=10)
    parser.add_argument("--minimum-predicted-positives", type=int, default=10)
    parser.add_argument("--maximum-categories", type=int, default=100)
    parser.add_argument("--n-bins", type=int, default=10)
    args = vars(parser.parse_args())
    try:
        result = audit_snapshot_files(**args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Drift comparison failed: {exc}\n")
    comparison = result["comparison"]
    print(f"Compared {comparison['reference_rows']} reference and {comparison['current_rows']} current rows; "
          f"report={args['output']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
