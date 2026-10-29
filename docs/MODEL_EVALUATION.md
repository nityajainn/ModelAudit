# Phase 3: model evaluation

The original phase protocol below remains applicable to its saved-artifact API.
The October 30 scope corrections add the [raw fitted-model workflow](RAW_AUDITS.md);
see the [correction record](SCOPE_CORRECTIONS.md) for current capabilities.

## Acceptance criteria

- Evaluate aligned binary labels and positive-class probabilities without changing
  inputs, fitting/recalibrating models, or choosing a threshold.
- Report fixed-threshold confusion counts, accuracy, balanced accuracy, precision,
  recall, specificity, F1, ROC AUC, trapezoidal PR AUC and average precision.
- Report binary Brier score, clipped log loss, uniform calibration-bin support and
  gaps, expected calibration error (ECE), and sparse-bin warnings.
- Compare a declared increasing threshold grid; retain undefined values as JSON
  null with explicit reasons. Single-class samples must not produce invented scores.
- Verify saved prediction hashes and their binding to exact prepared-data bytes
  before evaluation. Preserve inputs and existing outputs; emit local provenance.
- Pass hand-calculated, regression and integration tests, then measure the existing
  Telco baseline predictions and verify repeatability before requesting publication.

## Development command

After the Phase 1 preparation and baseline training, run from this repository:

```powershell
.venv\Scripts\python scripts/audit_models.py
```

Defaults: `--prepared data/prepared`, `--models models/baseline`,
`--output reports/generated/phase3-model-audit.json`, `--threshold 0.5`, `--bins 10`.
A repeat needs a new output filename. Reports remain ignored by Git. The command
returns 0 when evaluation succeeds (including review warnings) and 2 on handled
input/artifact/output errors. It is not a model-release gate. The complete CLI and
configuration schema remain Phase 6 work.

The adapter reads the two fixed Phase 1 `.npy` probability files with
`allow_pickle=False`, never the serialized `.joblib` models. It checks the summary's
preparation-manifest hash, partition metadata, and probability hashes; the prepared
loader also verifies exact partition bytes and counts. This binds positional
predictions to the recorded reference row order. Hash checks detect accidental
changes, not coordinated replacement of both arrays and their manifests. Model
files are neither loaded nor certified by this phase. No datasets are downloaded.

## In-memory interface

```python
from modelaudit.evaluation import evaluate_binary

result = evaluate_binary(
    labels, probabilities, threshold=0.5,
    thresholds=(0.1, 0.3, 0.5, 0.7, 0.9), n_bins=10, minimum_bin_rows=20,
)
```

Arrays must be nonempty, aligned, one-dimensional real numerical inputs; labels
are 0/1 and probabilities are finite in [0, 1]. Boolean, string, complex and object
arrays are rejected rather than silently coerced. The function cannot infer whether
caller-supplied rows are correctly paired; the artifact adapter verifies the Phase 1
pairing. No sample weights, multiclass labels, or missing observations are supported.
Invalid inputs raise ValueError. Inputs are copied and remain unchanged.

## Definitions and undefined cases

A prediction is positive when `probability >= threshold`, including equality.
Confusion matrices use `[[TN, FP], [FN, TP]]`. Precision is TP/(TP+FP), recall is
TP/(TP+FN), specificity is TN/(TN+FP), accuracy is (TP+TN)/N, balanced accuracy is
the mean of recall and specificity, and F1 is 2TP/(2TP+FP+FN).
A zero denominator produces null with a reason; F1 is zero when actual positives
are all missed, and null when there are neither actual nor predicted positives.
Balanced accuracy requires both actual classes. Support and prevalence are always
reported. Single-class samples retain supported classification/probability scores;
this project's policy withholds all three ranking scores and explains that choice.

ROC AUC measures ranking of positives versus negatives. Average precision weights
precision by recall increments; trapezoidal PR AUC linearly integrates the PR curve.
They are separate metrics and may differ, particularly for ties. See the official
[average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

Binary Brier score is mean((probability - label)^2), on [0, 1]. Log loss is mean
negative log probability of the observed class. Only log-loss computation clips
probabilities to [epsilon, 1-epsilon], with epsilon equal to float64 machine epsilon
(recorded in output), so predictions of exactly zero/one remain finite. Classification,
ranking, Brier score and calibration use the original probabilities. Brier and log
loss reflect more than calibration alone; see the official
[calibration guidance](https://scikit-learn.org/stable/modules/calibration.html).

## Calibration and thresholds

Calibration uses 1–100 uniform-width bins covering [0, 1]. Bins are left-inclusive
and right-exclusive, except the last bin includes 1. Internal boundary values go
to the bin on their right. Every bin remains in the output, including empty bins
with null averages and gaps. Each occupied bin records count, mean probability,
observed positive fraction and absolute difference. ECE is the sum of these gaps
weighted by count/N; no empty-bin contribution is added. Occupied bins with fewer
than 20 rows trigger a support warning by default. This is descriptive evidence,
not recalibration or a confidence interval. ECE depends on bin choice and sample size.

The default analysis grid is 0.1 through 0.9 in steps of 0.1. Custom grids must have
1–201 strictly increasing distinct values in [0, 1]. The separately supplied fixed
threshold need not belong to the grid. Each row reports classification scores and
predicted-positive support. No optimum or new threshold is selected. Choosing a
threshold after inspecting these reference results would require independent
validation to estimate its performance without selection bias.

## Scope and provenance

Reports contain aggregates only. Artifact reports record actual UTC creation time,
runtime packages, current Git commit/dirty state, and hashes of all six inputs.
The stored models and prepared files are not modified. Exact repeated results are
checked for the recorded environment, not promised across dependency versions.
No model training, uncertainty estimation, segment comparisons (Phase 4), drift
(Phase 5), deployment decisions, or public validation CLI is introduced here.
