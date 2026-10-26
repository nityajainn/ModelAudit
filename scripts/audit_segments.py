"""Phase 4 development command; full audit orchestration remains Phase 6."""

import argparse
import csv
from pathlib import Path

from modelaudit.segment_audit import audit_baseline_segments


def main():
    parser = argparse.ArgumentParser(description="Compare model performance across categorical segments.")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--models", type=Path, default=Path("models/baseline"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/phase4-segment-audit.json"))
    parser.add_argument("--columns", nargs="+", default=["Contract"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-rows", type=int, default=30)
    parser.add_argument("--minimum-class-rows", type=int, default=10)
    parser.add_argument("--minimum-predicted-positives", type=int, default=10)
    args = parser.parse_args()
    try:
        result = audit_baseline_segments(args.prepared, args.models, args.output,
                                         columns=args.columns, threshold=args.threshold,
                                         minimum_rows=args.minimum_rows,
                                         minimum_class_rows=args.minimum_class_rows,
                                         minimum_predicted_positives=args.minimum_predicted_positives)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Segment evaluation failed: {exc}\n")
    print(f"Evaluated segments for {len(result['models'])} models; report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
