# Local MLflow tracking (Phase 9)

Tracking records an existing offline audit. It does not rerun checks, fit a model,
or approve deployment.

## Install and record

Install the integration in the same environment as ModelAudit:

```powershell
python -m pip install -e ".[tracking]"
modelaudit track --report reports/generated/my-audit/report.json --store reports/generated/tracking
```

Use an existing completed audit report. Tracking returns 0 only after verifying
the stored report and searchable fields. It returns 2 on invalid input, missing
dependencies or a store failure. An audit verdict of fail or review can still be
recorded successfully; the underlying audit exit remains a separate tag.

The local store contains modelaudit.db (SQLite) and artifacts/. The experiment is
"ModelAudit audits". The default store is reports/generated/tracking relative to
the working directory. Remote URIs and stores inside the audit output directory
are rejected. An explicit MLflow client leaves the global tracking URI unchanged.
There is no remote server integration, account, autologging or registry.

## Recorded fields

| Field | Content |
| --- | --- |
| Audit identity | SHA-256 of the exact report bytes |
| Tags | Execution state, audit exit, available verdict, report hash, available tool version/source commit, input roles and hashes |
| Parameters | Configuration version/gates, numeric check settings, threshold lists, available estimator classes and simple model parameters |
| Metrics | Available reference classification/ranking/probability/support values, reference/current labeled comparisons, cross-split ID overlap |
| Artifact | Exact source report bytes as report.json |

Metric names use model indices with names in model.N.name tags. Examples are
model.0.reference.f1, model.0.reference.pr_auc_trapezoidal,
comparison.current.recall and reference.overlapping_id_rows.
Undefined scores remain absent, never replaced with zero. The full report retains
all findings, metadata and limitations beyond these selected searchable fields.

New fitted-model reports describe the already loaded checksum-verified estimator.
Saved-score reports copy available seed and parameters from the verified baseline
summary. Tracking never deserializes models. Older reports are accepted without
inventing missing training details.

## Validation and failure behavior

Both configuration routes produce version 1 report envelopes. Completed reports
require an adjacent matching completion manifest. Validation checks the envelope,
enabled check families, hash syntax and recomputed decision. Duplicate JSON keys,
nonfinite numbers, contradictory outcomes and changed report bytes are rejected.
Hashes establish consistency, not a digital signature or independent reproduction.

A runner failure.json marker can also be recorded, with execution=error and audit
exit=2, no verdict and no fabricated metrics. These markers lack a completion
manifest; only their shape is checked. Errors before output creation may leave no
marker.

Each invocation creates a new run ID and uses the real recording time. Repeating
a report preserves its SHA-256 audit identity. FINISHED requires artifact download,
hash comparison and parameter/metric/tag readback. A handled partial failure
attempts FAILED and reports the run ID. A process crash or unavailable store may
leave RUNNING; inspect it before retrying. Prior runs are not overwritten.

## Local viewer

Use the absolute SQLite URI printed by the tracking command:

```text
mlflow server --backend-store-uri "sqlite:///C:/absolute/path/tracking/modelaudit.db" --host 127.0.0.1 --port 5000
```

For POSIX absolute paths use four slashes: sqlite:////absolute/path/...
Open http://127.0.0.1:5000 and choose ModelAudit audits. Stop the server when done.
No server starts automatically. Reports may contain local paths and audit evidence;
keep the store private and outside Git. Moving it requires accounting for absolute
artifact locations recorded in SQLite.

The optional dependency is mlflow>=3.7,<4. The exact tested versions are captured
in the ignored verification inventory. See [testing](TESTING_CI.md) and
[release notes](RELEASE_NOTES.md).
