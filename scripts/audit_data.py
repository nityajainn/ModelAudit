"""Phase 2 development command; full validation CLI remains Phase 6."""

import argparse
import csv
from pathlib import Path

from modelaudit.data_audit import audit_prepared


def main():
    parser = argparse.ArgumentParser(description="Audit verified Phase 1 prepared data.")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path,
                        default=Path("reports/generated/phase2-data-audit.json"))
    args = parser.parse_args()
    try:
        result = audit_prepared(args.prepared, args.output)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Data audit failed: {exc}\n")
    print(f"Data audit: {result['status']}; counts={result['counts']}; report={args.output}")
    return 1 if result["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
