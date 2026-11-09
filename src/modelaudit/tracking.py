"""Record verified audit snapshots in a local MLflow experiment."""

import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

from .decisions import summarize_results

EXPERIMENT = "ModelAudit audits"


class TrackingError(ValueError):
    """Tracking failed while the source report remains available."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _json(payload):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number {value}")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed

    return json.loads(
        payload, object_pairs_hook=_object, parse_constant=invalid, parse_float=number
    )


def read_report(path):
    """Read an immutable byte snapshot and validate its audit envelope."""
    payload = path.read_bytes()
    report = _json(payload)
    if (
        not isinstance(report, dict)
        or type(report.get("format_version")) is not int
        or report["format_version"] != 1
    ):
        raise ValueError("Expected a version 1 audit report or failure marker")
    if report.get("status") == "failed":
        if not all(
            isinstance(report.get(key), str) and report[key] for key in ("error_type", "message")
        ):
            raise ValueError("Malformed failure marker")
        return payload, report
    for key in ("context", "configuration", "inputs", "results", "decision"):
        if not isinstance(report.get(key), dict):
            raise TypeError(f"Missing audit field {key}")
    configuration = report["configuration"]
    if (
        type(configuration.get("version")) is not int
        or configuration.get("version") not in (1, 2)
        or not isinstance(configuration.get("checks"), dict)
    ):
        raise ValueError("Unsupported audit configuration")
    if set(configuration["checks"]) != set(report["results"]) or not report["results"]:
        raise ValueError("Enabled checks and reported results differ")
    if not set(configuration["checks"]) <= {"data", "evaluation", "segments", "drift"}:
        raise ValueError("Unknown check family")
    if not all(isinstance(value, dict) for value in configuration["checks"].values()):
        raise ValueError("Invalid check options")
    if not all(isinstance(value, dict) for value in report["results"].values()):
        raise ValueError("Invalid check results")
    if not isinstance(configuration.get("gates"), list):
        raise TypeError("Invalid gate configuration")
    if not all(
        isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
        for value in report["inputs"].values()
    ):
        raise ValueError("Invalid input fingerprint")
    if report["decision"] != summarize_results(report["results"], configuration["gates"]):
        raise ValueError("Audit decision does not agree with recorded evidence")
    manifest = _json((path.parent / "manifest.json").read_bytes())
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "complete"
        or not isinstance(manifest.get("files"), dict)
        or manifest["files"].get(path.name) != hashlib.sha256(payload).hexdigest()
        or manifest.get("outcome") != report["decision"]["status"]
        or manifest.get("exit_code") != report["decision"]["exit_code"]
    ):
        raise ValueError("Completion manifest does not match the report")
    return payload, report


def _fields(report, digest):
    tags = {
        "modelaudit.report_sha256": digest,
        "modelaudit.audit_id": digest,
        "modelaudit.execution": "error" if report.get("status") == "failed" else "complete",
    }
    params, metrics = {}, {}
    if report.get("status") == "failed":
        tags.update({"modelaudit.audit_exit": "2", "modelaudit.error_type": report["error_type"]})
        return tags, params, metrics

    tags.update(
        {
            "modelaudit.verdict": report["decision"]["status"],
            "modelaudit.audit_exit": str(report["decision"]["exit_code"]),
        }
    )
    context = report["context"]
    for name, value in context.get("packages", {}).items():
        if name == "modelaudit":
            tags["modelaudit.version"] = str(value)
    if isinstance(context.get("git"), dict) and context["git"].get("commit"):
        tags["modelaudit.source_commit"] = str(context["git"]["commit"])
    for index, (role, value) in enumerate(sorted(report["inputs"].items())):
        tags[f"input.{index}.role"] = role
        tags[f"input.{index}.sha256"] = value
    params["configuration.version"] = str(report["configuration"]["version"])
    for family, options in report["configuration"]["checks"].items():
        for name, value in options.items():
            if type(value) in (bool, int, float) and (
                type(value) is not float or math.isfinite(value)
            ):
                params[f"{family}.{name}"] = str(value)
            elif isinstance(value, list) and all(
                type(item) in (int, float) and math.isfinite(item) for item in value
            ):
                params[f"{family}.{name}"] = json.dumps(value)
    params["configuration.gates"] = json.dumps(report["configuration"]["gates"], sort_keys=True)

    def measurement(key, value):
        if type(value) in (int, float) and math.isfinite(value):
            metrics[key] = float(value)

    for index, (name, model) in enumerate(
        sorted(report["results"].get("evaluation", {}).get("models", {}).items())
    ):
        tags[f"model.{index}.name"] = name
        metadata = report["results"]["evaluation"].get("model_metadata", {}).get(name, {})
        for field in ("pipeline", "classifier", "seed"):
            if field in metadata:
                params[f"model.{index}.{field}"] = str(metadata[field])
        for key, value in metadata.get("parameters", {}).items():
            if value is None or type(value) in (bool, int, float, str):
                params[f"model.{index}.parameter.{key}"] = str(value)
        for section in ("classification", "ranking", "probability_scores", "support"):
            for key, value in model.get(section, {}).items():
                measurement(f"model.{index}.reference.{key}", value)
    for finding in report["results"].get("data", {}).get("findings", []):
        if finding.get("check") == "identifier_overlap" and finding.get("dataset") == "cross_split":
            measurement("reference.overlapping_id_rows", finding["evidence"].get("reference_rows"))
    comparison = report["results"].get("drift", {}).get("comparison", {})
    performance = comparison.get("performance", {})
    if performance.get("status") == "measured":
        for population in ("reference", "current"):
            for name, value in performance[population]["metrics"].items():
                measurement(f"comparison.{population}.{name}", value)
    return tags, params, metrics


def track_report(report_path, store):
    """Create one verified run per invocation; retrying preserves earlier attempts."""
    report_path = Path(report_path).resolve()
    try:
        payload, report = read_report(report_path)
        digest = hashlib.sha256(payload).hexdigest()
        tags, params, metrics = _fields(report, digest)
    except (KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed audit evidence") from exc
    if "://" in str(store) or str(store).startswith(("//", "\\\\")):
        raise ValueError("Choose a local directory for the tracking store")
    store = Path(store).resolve()
    if store.is_relative_to(report_path.parent):
        raise ValueError("Choose a tracking store outside the audit output directory")
    try:
        from mlflow import MlflowClient
        from mlflow.exceptions import MlflowException
    except ImportError as exc:
        raise TrackingError(
            'Install the optional tracking extra with pip install "modelaudit[tracking]"'
        ) from exc

    store.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (store / "modelaudit.db").as_posix()
    artifact_uri = (store / "artifacts").as_uri()
    run_id = None
    client = None
    try:
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(EXPERIMENT, artifact_location=artifact_uri)
            except MlflowException:
                experiment = client.get_experiment_by_name(EXPERIMENT)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
            experiment = client.get_experiment(experiment_id)
        if experiment.lifecycle_stage != "active" or experiment.artifact_location != artifact_uri:
            raise ValueError(
                "Existing experiment has a different lifecycle or artifact destination"
            )
        run_id = client.create_run(experiment.experiment_id, tags=tags).info.run_id
        for name, value in params.items():
            client.log_param(run_id, name, value)
        for name, value in metrics.items():
            client.log_metric(run_id, name, value)
        with tempfile.TemporaryDirectory(prefix="modelaudit-track-") as directory:
            artifact = Path(directory) / "report.json"
            artifact.write_bytes(payload)
            client.log_artifact(run_id, str(artifact))
            restored = Path(directory) / "download"
            restored.mkdir()
            downloaded = Path(client.download_artifacts(run_id, "report.json", str(restored)))
            if hashlib.sha256(downloaded.read_bytes()).hexdigest() != digest:
                raise ValueError("Retrieved report bytes differ")
        saved = client.get_run(run_id).data
        if (
            saved.params != params
            or saved.metrics != metrics
            or any(saved.tags.get(k) != v for k, v in tags.items())
        ):
            raise ValueError("Retrieved tracking fields differ")
        client.set_terminated(run_id, status="FINISHED")
        if client.get_run(run_id).info.status != "FINISHED":
            raise ValueError("Run completion was not recorded")
    except Exception as exc:
        failed = ""
        if client is not None and run_id is not None:
            try:
                client.set_terminated(run_id, status="FAILED")
            except Exception:  # noqa: BLE001 - preserve the original failure if the store is unavailable.
                failed = " The store could not record the failed status."
        raise TrackingError(
            f"Tracking failed (run {run_id or 'not created'}): {exc}.{failed}"
        ) from exc
    return {
        "run_id": run_id,
        "audit_id": digest,
        "report_sha256": digest,
        "tracking_uri": uri,
        "audit_verdict": tags.get("modelaudit.verdict"),
        "audit_exit": int(tags["modelaudit.audit_exit"]),
        "status": "FINISHED",
    }
