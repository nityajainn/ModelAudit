"""Train the fixed Phase 1 baselines into a new local output directory."""

import argparse
import csv
from pathlib import Path

from modelaudit.baselines import train_baselines


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate two baseline pipelines.")
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        summary = train_baselines(args.prepared, args.output, seed=args.seed)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Training failed: {exc}\n")
    for name, result in summary["models"].items():
        print(f"{name}: reference F1={result['metrics']['f1']:.6f}")
    print(f"Artifacts and provenance saved in {args.output}.")


if __name__ == "__main__":
    main()
