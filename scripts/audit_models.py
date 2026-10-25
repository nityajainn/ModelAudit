"""Phase 3 development entry point; the full validation CLI remains Phase 6."""

import argparse
import csv
from pathlib import Path

from modelaudit.model_audit import audit_baseline_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate verified Phase 1 baseline predictions.")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--models", type=Path, default=Path("models/baseline"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/phase3-model-audit.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    try:
        report = audit_baseline_predictions(args.prepared, args.models, args.output,
                                            threshold=args.threshold, n_bins=args.bins)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Model evaluation failed: {exc}\n")
    print(f"Evaluated {len(report['models'])} saved prediction arrays; report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
