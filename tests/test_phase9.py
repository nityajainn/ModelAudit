"""Tracking acceptance tests: exact artifacts, truthful outcomes and failed writes."""

import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from modelaudit.cli import main
from modelaudit.decisions import summarize_results
from modelaudit.model_metadata import describe_model
from modelaudit.tracking import EXPERIMENT, TrackingError, read_report, track_report

mlflow = pytest.importorskip("mlflow")
MlflowClient = mlflow.MlflowClient


def report_at(directory, outcome="pass", metadata=True):
    directory.mkdir()
    results = {
        "evaluation": {
            "models": {
                "fitted": {
                    "classification": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
                    "ranking": {"pr_auc": 0.91, "average_precision": 0.92, "roc_auc": None},
                    "probability_scores": {"brier_score": 0.08},
                    "support": {"rows": 20},
                }
            }
        },
        "data": {
            "findings": [
                {
                    "check": "identifier_overlap",
                    "dataset": "cross_split",
                    "status": "fail" if outcome == "fail" else "pass",
                    "evidence": {"reference_rows": 2 if outcome == "fail" else 0},
                }
            ]
        },
        "drift": {
            "comparison": {
                "performance": {
                    "status": "measured",
                    "reference": {"metrics": {"f1": 0.85}},
                    "current": {"metrics": {"f1": 0.75, "roc_auc": None}},
                }
            }
        },
    }
    if metadata:
        results["evaluation"]["model_metadata"] = {
            "fitted": {
                "pipeline": "Pipeline",
                "classifier": "LogisticRegression",
                "seed": 42,
                "parameters": {"random_state": 42, "C": 1.0, "class_weight": None},
            }
        }
    if outcome == "review":
        results["evaluation"]["warnings"] = ["Small reference sample"]
    config = {
        "version": 2,
        "checks": {
            "evaluation": {"threshold": 0.5, "thresholds": [0.3, 0.5]},
            "data": {},
            "drift": {},
        },
        "gates": [],
    }
    report = {
        "format_version": 1,
        "context": {"packages": {"modelaudit": "0.1.0"}, "git": {"commit": "a" * 40}},
        "configuration": config,
        "inputs": {"reference": "b" * 64},
        "results": results,
        "decision": summarize_results(results, []),
    }
    path = directory / "report.json"
    save_report(path, report)
    return path


def save_report(path, report):
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()},
                "outcome": report["decision"]["status"],
                "exit_code": report["decision"]["exit_code"],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("outcome", ["pass", "fail", "review"])
def test_round_trip_and_repeat_preserve_source(tmp_path, outcome):
    path = report_at(tmp_path / "audit", outcome)
    before = {p.name: p.read_bytes() for p in path.parent.iterdir()}
    global_uri = mlflow.get_tracking_uri()
    first = track_report(path, tmp_path / "store")
    second = track_report(path, tmp_path / "store")
    assert first["run_id"] != second["run_id"]
    assert (
        first["audit_id"] == second["audit_id"] == hashlib.sha256(before["report.json"]).hexdigest()
    )
    assert first["audit_verdict"] == outcome
    assert first["audit_exit"] == (0 if outcome == "pass" else 1)
    client = MlflowClient(tracking_uri=first["tracking_uri"])
    run = client.get_run(first["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["model.0.reference.f1"] == 0.85
    assert "model.0.reference.roc_auc" not in run.data.metrics
    assert run.data.metrics["comparison.current.f1"] == 0.75
    assert run.data.metrics["reference.overlapping_id_rows"] == (2 if outcome == "fail" else 0)
    assert run.data.params["model.0.parameter.random_state"] == "42"
    assert run.data.params["evaluation.thresholds"] == "[0.3, 0.5]"
    assert run.data.tags["input.0.sha256"] == "b" * 64
    assert (
        Path(client.download_artifacts(first["run_id"], "report.json")).read_bytes()
        == before["report.json"]
    )
    assert before == {p.name: p.read_bytes() for p in path.parent.iterdir()}
    assert mlflow.get_tracking_uri() == global_uri


def test_legacy_report_and_cli_failure_verdict_still_track_successfully(tmp_path, capsys):
    path = report_at(tmp_path / "audit", "fail", metadata=False)
    assert main(["track", "--report", str(path), "--store", str(tmp_path / "store")]) == 0
    assert "audit_exit=1" in capsys.readouterr().out
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "store/modelaudit.db").as_posix())
    run = client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])[0]
    assert not any(key.startswith("model.0.parameter.") for key in run.data.params)


def test_failure_marker_is_execution_error_not_audit_failure(tmp_path):
    directory = tmp_path / "audit"
    directory.mkdir()
    path = directory / "failure.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "status": "failed",
                "error_type": "ValueError",
                "message": "Checksum mismatch",
                "completed_checks": [],
            }
        ),
        encoding="utf-8",
    )
    result = track_report(path, tmp_path / "store")
    assert result["audit_exit"] == 2
    assert result["audit_verdict"] is None
    run = MlflowClient(tracking_uri=result["tracking_uri"]).get_run(result["run_id"])
    assert run.data.tags["modelaudit.execution"] == "error"
    assert run.data.metrics == {}


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.update(format_version=True),
        lambda r: r.update(context=None),
        lambda r: r["configuration"].update(version=True),
        lambda r: r["configuration"].update(checks={"unknown": {}}),
        lambda r: r["configuration"]["checks"].update(data=[]),
        lambda r: r["results"].update(data=[]),
        lambda r: r["configuration"].update(gates={}),
        lambda r: r["inputs"].update(reference="not-a-hash"),
        lambda r: r["decision"].update(status="pass", exit_code=2),
        lambda r: r["context"].update(packages=None),
        lambda r: r["configuration"].update(gates=[{}]),
    ],
)
def test_invalid_envelopes_do_not_create_store(tmp_path, change):
    path = report_at(tmp_path / "audit")
    report = json.loads(path.read_text(encoding="utf-8"))
    change(report)
    save_report(path, report)
    with pytest.raises(ValueError):
        track_report(path, tmp_path / "store")
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "payload",
    [
        '{"format_version":1,"format_version":1}',
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":1e999}',
        '{"format_version":1,"status":"failed"}',
        "[]",
        "{",
    ],
)
def test_invalid_json_is_rejected(tmp_path, payload):
    path = tmp_path / "report.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        read_report(path)


def test_completion_manifest_is_required_and_bound_to_bytes(tmp_path):
    path = report_at(tmp_path / "audit")
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="manifest"):
        track_report(path, tmp_path / "store")
    (path.parent / "manifest.json").unlink()
    with pytest.raises(OSError):
        track_report(path, tmp_path / "store")
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize(
    "destination", ["https://example.invalid/store", "//server/share", "\\\\server\\share"]
)
def test_remote_stores_rejected(tmp_path, destination):
    path = report_at(tmp_path / "audit")
    with pytest.raises(ValueError, match="local directory"):
        track_report(path, destination)


def test_store_cannot_change_audit_output(tmp_path):
    path = report_at(tmp_path / "audit")
    with pytest.raises(ValueError, match="outside"):
        track_report(path, path.parent / "store")


def test_missing_extra_has_actionable_cli_error_without_store(tmp_path, capsys):
    path = report_at(tmp_path / "audit")
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("optional dependency absent")
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=missing), pytest.raises(SystemExit) as exc:
        main(["track", "--report", str(path), "--store", str(tmp_path / "store")])
    assert exc.value.code == 2
    assert "modelaudit[tracking]" in capsys.readouterr().err
    assert not (tmp_path / "store").exists()


@pytest.mark.parametrize("fault", ["log", "artifact", "fields", "completion", "failed_status"])
def test_partial_writes_never_report_success(tmp_path, fault):
    path = report_at(tmp_path / "audit")
    original_get = MlflowClient.get_run
    original_stop = MlflowClient.set_terminated

    def get(client, run_id):
        run = original_get(client, run_id)
        if fault == "fields":
            run.data.params["unrequested"] = "changed"
        if fault == "completion" and run.info.status == "FINISHED":
            return SimpleNamespace(info=SimpleNamespace(status="RUNNING"), data=run.data)
        return run

    def stop(client, run_id, status="FINISHED", **kwargs):
        if fault == "failed_status":
            raise OSError("unavailable store")
        return original_stop(client, run_id, status=status, **kwargs)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{}")
    with (
        patch.object(MlflowClient, "get_run", get),
        patch.object(MlflowClient, "set_terminated", stop),
    ):
        if fault in ("log", "failed_status"):
            with (
                patch.object(MlflowClient, "log_param", side_effect=OSError("write failed")),
                pytest.raises(TrackingError) as exc,
            ):
                track_report(path, tmp_path / "store")
        elif fault == "artifact":
            with (
                patch.object(MlflowClient, "download_artifacts", return_value=str(corrupt)),
                pytest.raises(TrackingError) as exc,
            ):
                track_report(path, tmp_path / "store")
        else:
            with pytest.raises(TrackingError) as exc:
                track_report(path, tmp_path / "store")
    client = MlflowClient(tracking_uri="sqlite:///" + (tmp_path / "store/modelaudit.db").as_posix())
    run = client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])[0]
    assert run.info.status == ("RUNNING" if fault == "failed_status" else "FAILED")
    if fault == "failed_status":
        assert "could not record the failed status" in str(exc.value)


def test_existing_experiment_cannot_redirect_artifacts(tmp_path):
    path = report_at(tmp_path / "audit")
    first = track_report(path, tmp_path / "store")
    with (
        patch.object(
            MlflowClient,
            "get_experiment_by_name",
            return_value=SimpleNamespace(
                lifecycle_stage="active", artifact_location="https://example.invalid"
            ),
        ),
        pytest.raises(TrackingError, match="different lifecycle"),
    ):
        track_report(path, tmp_path / "store")
    client = MlflowClient(tracking_uri=first["tracking_uri"])
    assert len(client.search_runs([client.get_experiment_by_name(EXPERIMENT).experiment_id])) == 1


def test_model_metadata_only_records_available_simple_values():
    class Estimator:
        def get_params(self, deep):
            assert deep is False
            return {
                "random_state": 17,
                "name": "binary",
                "none": None,
                "object": object(),
                "nan": float("nan"),
            }

    assert describe_model(object())["parameters"] == {}
    metadata = describe_model(SimpleNamespace(named_steps={"classifier": Estimator()}))
    assert metadata["classifier"] == "Estimator"
    assert metadata["parameters"] == {"random_state": 17, "name": "binary", "none": None}
