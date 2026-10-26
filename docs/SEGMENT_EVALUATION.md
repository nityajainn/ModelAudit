# Phase 4: segment performance

## Acceptance criteria

- Compare observed categorical groups at one shared threshold using Phase 3's
  classification, ranking and probability-score definitions.
- Retain support, confusion counts, undefined reasons and low-support warnings.
  Compare each group with the overall population and report supported-group ranges.
- Keep all missing/blank segment rows in an explicit null group and check each
  column independently. Reject ambiguous labels, bad alignment and excessive
  cardinality rather than silently dropping observations.
- Use verified Phase 1 predictions and reference-data hashes; preserve all inputs,
  avoid model deserialization, and refuse to overwrite reports.
- Pass the existing regressions and new defect fixtures, then record actual Contract
  comparisons and repeatability. No fairness conclusion or significance claim is made.

## Development command

From the ModelAudit directory after Phase 1 preparation and training:

```powershell
.venv\Scripts\python scripts/audit_segments.py
```

Default grouping is `Contract`, as specified in the dataset plan. Options include
`--prepared data/prepared`, `--models models/baseline`,
`--output reports/generated/phase4-segment-audit.json`, `--columns Contract`,
`--threshold 0.5`, `--minimum-rows 30`, `--minimum-class-rows 10`, and
`--minimum-predicted-positives 10`. Multiple categorical predictor columns can be
supplied after `--columns`. Identifiers, the target, numeric and unknown columns
are rejected. Reports stay local and ignored; repeats require a new filename.
Exit code 0 means computation completed (warnings may exist), and 2 means a handled
input/artifact/output error. This is not a release gate or the planned Phase 6 CLI.

## In-memory API

```python
from modelaudit.segments import evaluate_segments

report = evaluate_segments(
    labels, probabilities, categorical_frame, threshold=0.5,
    minimum_rows=30, minimum_class_rows=10,
    minimum_predicted_positives=10, maximum_groups=50,
)
```

Labels and P(label=1) probabilities follow the Phase 3 contract. Segment columns
are unique nonempty string names; values must be strings or missing scalars.
Blank/whitespace-only strings and missing values form one group with `value: null`
and `is_missing: true`. Other strings retain spelling, including surrounding spaces;
no sentinel string is reserved. Only observed categories appear; unused categorical
levels do not create empty groups. Columns and nonmissing categories sort
lexicographically; the missing group comes last. Each column partitions every row.
The API uses positional alignment, not pandas indexes, and cannot verify that a
caller aligned unrelated inputs correctly. No intersections between columns are built.
More than `maximum_groups` (default 50, allowed 1–1000) causes an error. Support
thresholds must be positive integers. Inputs are never changed or fitted.

## Measurements and comparisons

Every group retains row count, positive/negative support, prevalence, predicted
positive count, population fraction and confusion matrix. The metrics are accuracy,
balanced accuracy, precision, recall, specificity, F1, ROC AUC, trapezoidal PR AUC,
average precision, Brier score and log loss. Undefined values remain null with the
Phase 3 reason; single-class groups are not assigned invented ranking scores.
See [Phase 3 definitions](MODEL_EVALUATION.md). Segment-specific calibration and
threshold sweeps are intentionally absent; one threshold is shared by all groups.

Support defaults are review rules, not estimated uncertainty or confidence intervals:

| Metric | Required support for a comparison |
| --- | --- |
| Every metric | At least 30 rows and a defined measurement |
| Accuracy, Brier score, log loss | No additional support rule |
| Precision | At least 10 predicted positives |
| Recall | At least 10 actual positives |
| Specificity | At least 10 actual negatives |
| Balanced accuracy, F1, ROC AUC, PR AUC, average precision | At least 10 actual positives and 10 actual negatives |

Equality meets a threshold. Scores below these limits are still shown with warnings,
but the affected comparison is null and explains why. Zero predicted positives,
missing classes and insufficient rows are handled separately. The F1 comparison
rule is deliberately conservative: F1 can be defined even when the comparison
policy withholds it. Both group and overall measurements must be eligible for a
`difference_from_overall` value. It equals group minus overall, without rounding.
For Brier/log loss, lower values are better; for the other metrics, higher values
are better. The direction is included explicitly in the report.

For each column and metric, `supported_ranges` includes the minimum, maximum and
max-minus-min gap among eligible groups, plus eligible/excluded group counts.
A range requires at least two eligible groups; otherwise it is null with a reason.
Unsupported groups are not ranked or treated as zero. Group-to-overall comparisons
are not independent because the overall population includes each group.

## Artifact verification and limits

Phases 3 and 4 share the same internal verified-input loader and post-read hash check.
The existing Phase 3 public function and output shape remain unchanged. The adapter
binds two fixed probability arrays to the exact preparation manifest and partition
metadata, checks file hashes, loads arrays with `allow_pickle=False`, and never
loads `.joblib` files. Reports include all six input hashes, actual UTC creation
time, runtime versions and Git commit/dirty state. Hashes are integrity checks,
not authentication of a manifest replaced together with its inputs.

Outputs contain group labels and aggregates, not customer IDs or individual
predictions. Group labels may themselves be sensitive: generated outputs stay local.
Differences are descriptive on this fixed sample, with no significance testing,
confidence intervals, causal or fairness conclusions. Distinct prevalence and small
positive counts can affect the comparisons. Passing support thresholds is not proof
of reliability. Threshold selection, model training, drift (Phase 5), full CLI
orchestration (Phase 6), and scenario experiments (Phase 7) remain separate work.
