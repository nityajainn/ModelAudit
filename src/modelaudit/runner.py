"""Sequential audit orchestration with exclusive outputs and a completion manifest."""

import logging
from pathlib import Path

from .artifacts import fingerprint, run_context, write_json
from .audit_config import load_audit_config
from .data_audit import audit_prepared
from .data_checks import CheckPolicy
from .decisions import summarize_results
from .drift_audit import audit_snapshot_files
from .model_audit import BASELINE_NAMES, audit_baseline_predictions
from .reporting import render_reports
from .segment_audit import audit_baseline_segments


def _input_paths(config, config_path):
    if config["version"] == 2:
        return {"configuration":config_path, **{f"inputs/{k}":Path(v) for k,v in config["inputs"].items()}}
    paths = {"configuration": config_path}
    if "prepared" in config["inputs"]:
        paths.update({f"prepared/{name}": Path(config["inputs"]["prepared"]) / name
                      for name in ("manifest.json", "train.csv", "reference.csv")})
    if "models" in config["inputs"]:
        paths["models/summary.json"] = Path(config["inputs"]["models"]) / "summary.json"
        paths.update({f"models/{name}_probabilities.npy": Path(config["inputs"]["models"]) / f"{name}_probabilities.npy"
                      for name in BASELINE_NAMES})
    if "drift" in config["checks"]:
        paths.update({f"drift/{name}": Path(config["checks"]["drift"][name]) for name in ("reference", "current")})
    return paths


def run_audit(config_path, output) -> dict:
    """Do not reuse a directory. Only manifest.json marks a completed run.

    On failure, retain partial evidence in the new directory and mark it failed
    where possible. Never remove outputs or retry into a partially written run.
    """
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    first_config_hash = fingerprint(config_path)
    config = load_audit_config(config_path)
    paths = _input_paths(config, config_path)
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Required input is not a regular file: {path}")
    before = {name: fingerprint(path) for name, path in paths.items()}
    if before["configuration"] != first_config_hash:
        raise ValueError("Configuration changed during loading; no output was created.")
    for protected in (config["inputs"].values() if config["version"] == 1 else []):
        if output.is_relative_to(Path(protected)):
            raise ValueError("Output cannot be inside the prepared data or model directory.")
    output.mkdir(parents=True, exist_ok=False)
    results, files, coverage = {}, [], {}
    logger = logging.getLogger("modelaudit")
    try:
        if config["version"] == 2:
            from .raw_audit import run_raw_checks
            results, coverage = run_raw_checks(config)
            for name, result in results.items():
                destination = output / f"{name}.json"
                write_json(destination, result)
                files.append(destination.name)
        for name, options in (config["checks"].items() if config["version"] == 1 else []):
            logger.info("Running %s checks", name)
            destination = output / f"{name}.json"
            if name == "data":
                result = audit_prepared(config["inputs"]["prepared"], destination, policy=CheckPolicy(**options))
            elif name == "evaluation":
                result = audit_baseline_predictions(config["inputs"]["prepared"], config["inputs"]["models"], destination, **options)
            elif name == "segments":
                result = audit_baseline_segments(config["inputs"]["prepared"], config["inputs"]["models"], destination, **options)
            else:
                result = audit_snapshot_files(output=destination, **options)
            results[name] = result
            files.append(destination.name)
        if before != {name: fingerprint(path) for name, path in paths.items()}:
            raise ValueError("An input changed across the audit; no completion manifest was written.")
        report = {"format_version": 1, "context": run_context(), "configuration": config,
                  "inputs": before, "results": results, "decision": summarize_results(results, config["gates"]),
                  "model_deserialization_performed": config["version"] == 2 and config["model"] is not None,
                  "coverage": coverage}
        markdown, html = render_reports(report)
        write_json(output / "report.json", report)
        files.append("report.json")
        for filename, content in (("summary.md", markdown), ("summary.html", html)):
            with (output / filename).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            files.append(filename)
        write_json(output / "manifest.json", {"format_version": 1, "status": "complete",
                   "outcome": report["decision"]["status"], "exit_code": report["decision"]["exit_code"],
                   "files": {name: fingerprint(output / name) for name in files}})
        return report
    except Exception as exc:
        try:
            write_json(output / "failure.json", {"format_version": 1, "status": "failed",
                       "error_type": type(exc).__name__, "message": str(exc), "completed_checks": list(results)})
        except OSError:
            pass  # A missing completion manifest still identifies an incomplete run.
        raise
