# Baseline workflow

Phase 1 trains two fixed models on the same training partition and evaluates them
on the untouched reference partition. It does not select a winning model, tune
parameters, or make a release decision.

## Reproduce a run

From this repository's root, after installing the package:

```powershell
.venv\Scripts\python scripts/prepare_data.py
.venv\Scripts\python scripts/train_baselines.py
```

Preparation downloads only the pinned source if it is absent. An existing source
must match its checksum. Review [dataset provenance](DATASET.md) before downloading.
Training is offline and verifies the prepared file hashes before fitting.

Each output directory must be new, including when a previous run failed partway.
For a repeat using the same source and seed:

```powershell
.venv\Scripts\python scripts/prepare_data.py --output data/prepared-repeat
.venv\Scripts\python scripts/train_baselines.py --prepared data/prepared-repeat --output models/repeat
```

The scripts are development commands, not the Phase 6 `modelaudit validate` CLI.
Success exits with 0; handled input or filesystem failures exit with 2. These are
not PASS/WARNING/FAIL validation exit codes. Use `--help` for path/seed options.
Keep custom outputs inside ignored `data/` and `models/` directories.

## Preprocessing and model settings

Each classifier owns a separate fitted [scikit-learn pipeline](https://scikit-learn.org/stable/modules/compose.html).
Training-only numerical median imputation is followed by standard scaling.
Categorical missing values receive a reserved token, followed by one-hot encoding.
Unseen categories encode as all zeros for that feature. Empty numerical training
columns use the imputer's zero fallback; this preserves shape but is not evidence
that the column is useful. Later validation should flag insufficient evidence.

- Logistic Regression: L-BFGS, maximum 2,000 iterations, seed 42.
- Random Forest: 100 trees, one worker, seed 42, otherwise library defaults.
- Both models exclude `customerID` and `Churn` from the 19 predictors.
- Both evaluate `P(Churn=1) >= 0.50` without threshold optimization.

Complete resolved classifier parameters and package versions are recorded in each
local summary. Version constraints permit development updates; the tested-version
file records the actual environment used for the documented experiment.

## Metrics and artifacts

Precision, recall, F1, and the confusion matrix use the fixed threshold. The matrix
is ordered `[[TN, FP], [FN, TP]]`. Precision is null with an explicit reason when
no positives are predicted. Both observed target classes are required for this
baseline evaluation. The reference prevalence is reported for context.

`pr_auc_trapezoidal` integrates the precision–recall curve; `average_precision`
uses recall-increment weighting. They are deliberately separate measurements.
Phase 3 now evaluates calibration, further binary metrics, and threshold comparisons
from these saved predictions; see [model evaluation](MODEL_EVALUATION.md). Phase 4 adds
[segment comparisons](SEGMENT_EVALUATION.md). Phase 5 adds explicitly prepared
[snapshot comparisons](DRIFT_ANALYSIS.md); full orchestration and configurable
quality gates remain later-phase work. A random split is not a time series.

Each run creates two fitted `.joblib` pipelines, corresponding `.npy` reference
probabilities, and `summary.json`. Probability order follows the ID-sorted reference
CSV. The summary records data/artifact hashes, settings, warnings, UTC creation time,
runtime versions, and the source Git commit/dirty state. Artifact reload is checked
against the probabilities computed before serialization (`atol=1e-12`, `rtol=0`).

Only load joblib files you trust: deserialization can execute Python code. The
training code reloads only its own newly created artifacts. Tests use invented
fixtures and do not download the dataset. Hashes detect accidental changes and do
not authenticate an attacker-controlled manifest or artifact.

Cross-platform bit-for-bit determinism is not promised. Repeatability results apply
to the documented environment, split, and settings; timestamps and artifact hashes
need not be identical across runs. Reported model metrics do not establish temporal
generalization, fairness, causal effects, or production suitability.
