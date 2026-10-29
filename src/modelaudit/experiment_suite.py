"""Run controlled scenarios against copies of verified Phase 1 artifacts."""

import logging
from pathlib import Path
import shutil
import tempfile

from .artifacts import fingerprint, run_context, write_json
from .dataset import load_partitions
from .experiment_reports import render_experiments
from .model_audit import _load_prediction_inputs, _verify_prediction_inputs, audit_baseline_predictions
from .scenarios import build_scenarios


def _checksum_case(paths):
    """Corrupt only a fresh temporary copy; unrelated errors are not successes."""
    with tempfile.TemporaryDirectory(prefix="modelaudit-checksum-") as directory:
        root = Path(directory)
        for key, source in paths.items():
            target = root / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        with (root / "models/logistic_regression_probabilities.npy").open("ab") as stream:
            stream.write(b"controlled-checksum-defect")
        destination = root / "unexpected-report.json"
        rejected, reason = False, None
        try:
            audit_baseline_predictions(root / "prepared", root / "models", destination)
        except ValueError as exc:
            if str(exc) != "Probability checksum mismatch for logistic_regression.":
                raise
            rejected, reason = True, str(exc)
        report_written = destination.exists()
    met = rejected and not report_written
    return {"id": "corrupt_checksum", "mutation": "Append bytes to a copied probability array without changing its copied summary.",
            "parameters": {"copied_input_files": len(paths), "original_files_modified": False},
            "evidence": {"rejected": rejected, "reason": reason, "successful_report_written": report_written},
            "expectations": [{"description": "The exact checksum mismatch is rejected before a successful report is written.", "met": met}],
            "expectations_met": met, "audit_outcome": "rejected" if rejected else "accepted",
            "results": {"integrity_check": {"rejected": rejected, "reason": reason}}}


def run_experiments(prepared, models, output):
    """Create a new aggregate-only experiment bundle and completion manifest.

    Exit 0/1 describes whether expectations matched, never model acceptability.
    Unexpected failures retain partial evidence without a completion manifest.
    """
    prepared, models, output = Path(prepared).resolve(), Path(models).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    if output.is_relative_to(prepared) or output.is_relative_to(models):
        raise ValueError("Experiment output cannot be inside prepared data or model directories.")
    logger = logging.getLogger("modelaudit")
    logger.info("Verifying inputs and evaluating controlled copies")
    reference, predictions, paths, before = _load_prediction_inputs(prepared, models)
    train, second_reference, _ = load_partitions(prepared)
    if not reference.equals(second_reference):
        raise ValueError("Reference data changed while loading experiment inputs.")
    cases = build_scenarios(train, reference, predictions)
    _verify_prediction_inputs(paths, before)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cases").mkdir()
    files, summaries = [], []
    try:
        cases.append(_checksum_case(paths))
        for case in cases:
            logger.info("Recording %s: expectations %s", case["id"], "matched" if case["expectations_met"] else "not matched")
            filename = f"cases/{case['id']}.json"
            write_json(output / filename, case)
            files.append(filename)
            summaries.append({key: value for key, value in case.items() if key not in ("results", "decision")})
            summaries[-1]["file"] = filename
            summaries[-1]["gates"] = case.get("decision", {}).get("gates", [])
        _verify_prediction_inputs(paths, before)
        all_met = all(case["expectations_met"] for case in cases)
        summary = {"format_version": 1, "protocol_version": 1, "context": run_context(),
                   "inputs": before, "training_rows": len(train), "reference_rows": len(reference),
                   "source_inputs_unchanged": True, "model_deserialization_performed": False,
                   "all_expectations_met": all_met, "exit_code": 0 if all_met else 1,
                   "scenarios": summaries,
                   "limitations": ["Controlled copied-data experiments are not real production incidents.",
                                   "Expectation success is not approval of the injected data or model.",
                                   "Feature mutations do not simulate a fitted model response; saved scores are manipulated explicitly.",
                                   "No training, causal claim, population-level fairness conclusion or deployment approval is produced."]}
        markdown, html = render_experiments(summary)
        write_json(output / "summary.json", summary)
        files.append("summary.json")
        for name, content in (("summary.md", markdown), ("summary.html", html)):
            with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            files.append(name)
        write_json(output / "manifest.json", {"format_version": 1, "status": "complete",
                   "all_expectations_met": all_met, "exit_code": summary["exit_code"],
                   "files": {name: fingerprint(output / name) for name in files}})
        return summary
    except Exception as exc:
        try:
            write_json(output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                       "message": str(exc), "recorded_scenarios": [case["id"] for case in summaries]})
        except OSError:
            pass
        raise
