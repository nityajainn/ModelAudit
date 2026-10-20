"""Phase 1 development entry point; not the future validation CLI."""

import argparse
import csv
from pathlib import Path

from modelaudit.dataset import download_source, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned IBM Telco sample.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw/telco.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("Choose a new output directory; existing data is protected.")
        raw = download_source(args.raw)
        manifest = prepare_dataset(raw, args.output, seed=args.seed)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(f"Prepared {manifest['files']['train.csv']['rows']} training and "
          f"{manifest['files']['reference.csv']['rows']} reference rows in {args.output}.")


if __name__ == "__main__":
    main()
