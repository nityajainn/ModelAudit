# Raw fitted-model audits (configuration version 2)

Version 1 remains the saved-artifact workflow. Version 2 closes the Phase 2–6
scope gaps found in the October 30 review. It audits raw CSVs before prediction,
never refits during an audit, and uses the same metric, segment, drift, decision,
and report engines. Neither workflow modifies its inputs or existing reports.

## Configuration

The root keys are `version: 2`, `inputs`, `schema`, `checks`, optional `model`,
optional `current_labeled` (default false), and optional `gates`.

- `inputs.train` and `inputs.reference`: required CSV paths.
- `inputs.current`: optional CSV path; its labels are used only when
  `current_labeled: true`. Missing current data is explicitly unavailable.
- `schema`: lists `numeric` and `categorical`, one numeric 0/1 `target`, and an
  optional string `identifier`. Roles must be distinct. Row order is preserved.
- `inputs.model`: required for evaluation, segments or prediction drift.
- `model`: `trusted: true`, the exact lowercase `sha256`, optional
  `positive_label: 1`, and optional `features` (defaults to all schema predictors).
  Explicit features must exclude identifier and target. Unselected extra audit
  columns are never passed to the model.
- `checks`: data/evaluation/segments/drift option mappings. Metric and support
  options match version 1; segment columns may be any declared categorical
  features. Drift uses schema feature roles and fitted predictions automatically;
  reference/current paths belong in `inputs`, not inside drift options.
- `checks.data.prohibited_columns`: optional distinct names. Presence fails,
  including on unlabeled current data. Extra raw columns warn and are inspected
  for target-copy associations when their types support the heuristic.
- Enabled evaluation/segment/drift checks must share one decision threshold.
- Gates retain version 1's bounded numeric result paths. The fitted model key is
  `fitted`; for example `[evaluation, models, fitted, classification, f1]`.

Joblib deserialization can execute code. Only mark a model trusted when its
source is trusted. A checksum verifies bytes, not whether their source is safe.
The loader checks the exact byte buffer that it deserializes, validates binary
classes, selects the positive class by label, validates both probability columns
and their row sums, and calls prediction once per required population.

Use `configs/raw-audit.yaml` as a template: its intentionally false trust flag
and zero fingerprint must be replaced after verifying your own local model.
Compute the fingerprint with `Get-FileHash -Algorithm SHA256 <model path>` and
use its lowercase value. No new data download or dependency installation is needed.

## Findings, partial coverage and outputs

Missing/duplicate/blank headers, invalid numeric values, target problems and
contamination become structured findings where parsing is possible. Ragged or
unparseable CSV is an execution error. Predictor defects block inference for that
population; independent feature drift continues. A missing current target never
becomes an invented performance measurement. Reports record each family's
coverage and the reason for each blocked calculation.

Both versions retain ModelAudit's existing exit contract: 0 pass, 1 review or
failure, 2 execution error. This intentionally differs from ModelGate's warning
exit convention. A completion manifest marks a finished audit, not a passing one.
Input/config/model fingerprints are checked again before completion; partial
outputs and failure.json remain after an execution error. Always use a new folder.

Segments include metric-specific best values/tied groups, signed per-group
best differences, and support reasons. For losses such as Brier and log loss,
lower is better. Unsupported groups cannot define a best value; a single eligible
group does not establish parity.

Categorical drift retains total variation and additionally reports base-2
Jensen–Shannon distance. Its configurable `smoothing` defaults to 1e-6:
`(1-epsilon)*observed_fraction + epsilon/K` over shared categorical support,
including missing values. Use the named metric when choosing gate thresholds;
these distances are not interchangeable. No p-values or significance claims are made.

## Full CLI experiment protocol

`modelaudit experiments --mode fitted --output <new directory> --seed 42`
creates synthetic training/reference populations of 1,000 rows each with
disjoint IDs and balanced binary labels. Contract groups contain 900 and 100
rows, balanced by outcome. A signed continuous tenure signal separates classes;
other features vary independently. These are invented rows, not customer data.

The two existing baseline builders fit only clean training predictors. Each
model is serialized once and the same hash-pinned artifact is reused in five
ordinary `python -m modelaudit run` subprocesses, each with a 120-second timeout.

| Scenario | Mutation and expected evidence |
| --- | --- |
| Clean | Labeled current is reference copy; perfect reference/group metrics, zero drift/performance changes, no review findings, pass/0 |
| Contamination | Replace 20 complete reference rows with training rows; exact identifier/predictor overlap counts, failure/1 |
| Target leakage | Add an audit-only target copy to reference; heuristic warning, unchanged model F1, review/1 |
| Feature drift | Remove current labels and add 100 to tenure; tenure KS 1, all other feature distances 0, measured prediction drift, no labeled performance, failure/1 |
| Segment degradation | Reflect only One year tenure around 30; that group's recall 0, other group's recall 1, overall F1 .90, overall/best recall differences -.90/-1, failure/1 |

Fixed demonstration gates are minimum overall F1 .85, minimum annual-group recall
.8 and maximum tenure KS .2. Group minimum size is 50. ModelAudit uses a failure
gate for this drift example and a review exit for leakage; these differ from
ModelGate's warning exits and are checked explicitly. No result-based tuning is used.

The assessor reads existing report measurements rather than recomputing them.
It verifies finding identities/counts, exact gate identities/statuses, group
metrics and support, final verdict, actual process exit, inference provenance,
input/config/model fingerprints and completion/output hashes. Missing, duplicate,
blocked or wrong targeted evidence cannot count as successful detection.

The experiment exit is 0 only when all expected observations and verdicts match,
1 for completed but unmet expectations, 2 for execution errors. Ten audit reports,
logs, synthetic inputs, two models, ten configs, summary.json and a final output
manifest stay inside the new ignored experiment directory. Repeated runs compare
input bytes and measured results; timestamps, absolute paths and model bytes may
legitimately differ. These are regression controls, not estimates of real-world
accuracy, drift-detection power, fairness or production readiness.
