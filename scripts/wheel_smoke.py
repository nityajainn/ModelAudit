"""Run installed ModelAudit on synthetic inputs from a fresh directory."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sysconfig
from pathlib import Path

import joblib
import yaml

import modelaudit
from modelaudit.baselines import build_pipeline
from modelaudit.cli_experiments import assess_report, scenario_inputs, synthetic_frame
from modelaudit.dataset import CATEGORICAL, FEATURES, IDENTIFIER, NUMERIC, TARGET


def installed_command():
    dist = importlib.metadata.distribution("modelaudit")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    installed = Path(dist.locate_file("modelaudit/__init__.py")).resolve()
    if (
        "archive_info" not in direct
        or not direct.get("url", "").endswith(".whl")
        or Path(modelaudit.__file__).resolve() != installed
        or not installed.is_relative_to(Path(sysconfig.get_path("purelib")).resolve())
    ):
        raise RuntimeError("Expected the installed wheel in this interpreter's environment")
    launcher = Path(sysconfig.get_path("scripts")) / (
        "modelaudit.exe" if os.name == "nt" else "modelaudit"
    )
    if not launcher.is_file():
        raise RuntimeError(f"Console command missing at {launcher}")
    return launcher


def smoke(output):
    launcher = installed_command()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    train, reference = synthetic_frame(42, "train"), synthetic_frame(43, "reference")
    train.to_csv(output / "train.csv", index=False)
    model = build_pipeline("logistic_regression")
    model.fit(train[FEATURES], train[TARGET])
    model_path = output / "model.joblib"
    joblib.dump(model, model_path)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    summary = {}
    for case in ("clean", "contamination", "missing"):
        ref, current = scenario_inputs(train, reference, "clean" if case == "missing" else case)
        ref.to_csv(output / "reference.csv", index=False)
        checks = {
            "data": {},
            "evaluation": {"n_bins": 1, "thresholds": [0.5]},
            "segments": {"columns": ["Contract"], "minimum_rows": 50},
        }
        gates = [
            {
                "name": "overall F1",
                "path": ["evaluation", "models", "fitted", "classification", "f1"],
                "minimum": 0.85,
            },
            {
                "name": "annual recall",
                "path": [
                    "segments",
                    "models",
                    "fitted",
                    "segments",
                    0,
                    "groups",
                    1,
                    "metrics",
                    "recall",
                ],
                "minimum": 0.8,
            },
        ]
        inputs = {"train": "train.csv", "reference": "reference.csv", "model": "model.joblib"}
        if current is not None:
            current.to_csv(output / "current.csv", index=False)
            inputs["current"] = "current.csv"
            checks["drift"] = {"minimum_rows": 50}
            gates.append(
                {
                    "name": "tenure shift",
                    "path": ["drift", "comparison", "features", "tenure", "ks_distance"],
                    "maximum": 0.2,
                }
            )
        if case == "missing":
            inputs["reference"] = "absent.csv"
        config = {
            "version": 2,
            "schema": {
                "numeric": NUMERIC,
                "categorical": CATEGORICAL,
                "target": TARGET,
                "identifier": IDENTIFIER,
            },
            "inputs": inputs,
            "model": {"trusted": True, "sha256": digest, "features": FEATURES},
            "current_labeled": current is not None,
            "checks": checks,
            "gates": gates,
        }
        config_path = output / f"{case}.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = subprocess.run(
            [
                str(launcher),
                "run",
                "--config",
                str(config_path),
                "--output",
                str(output / case),
                "--quiet",
            ],
            cwd=output,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if case == "missing":
            if (
                result.returncode != 2
                or "absent.csv" not in result.stderr
                or (output / case / "manifest.json").exists()
            ):
                raise RuntimeError(
                    f"Wrong missing-input behavior: {result.returncode} {result.stderr}"
                )
        else:
            report = json.loads((output / case / "report.json").read_text(encoding="utf-8"))
            failed = [
                row["description"]
                for row in assess_report(case, report, result.returncode)
                if not row["met"]
            ]
            if failed:
                raise RuntimeError(f"{case} failed: {failed}")
        summary[case] = result.returncode
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    smoke(parser.parse_args().output)
