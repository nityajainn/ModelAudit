# ModelAudit

An incremental project for examining data quality and model behavior before
a release. Its intended scope is offline evaluation of scikit-learn binary
classifiers, with explicit evidence and limitations.

**Current stage: Phase 7 - published baseline, with scope corrections verified locally; push approval pending.** The package includes configuration
loading, logging, reproducible Telco preparation, two baseline pipelines, data-quality
checks, binary model metrics, calibration measurements, threshold comparisons, and
segment comparisons with support warnings, feature/prediction distribution comparisons,
and supported labeled performance changes. A public command now runs the configured
checks and creates JSON, Markdown and offline HTML reports. A reproducible experiment
suite now exercises deliberate defects on copies of the verified inputs.

## Local setup

Python 3.11 or newer is required. From this project's directory on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

The import package is `modelaudit`. `configs/foundation.yaml` is a minimal
example; it is not yet a model-validation configuration.

## Prepare and train

First review [dataset provenance and limitations](docs/DATASET.md). Raw and prepared
data, fitted models, and generated reports stay local and are excluded from Git.

```powershell
.venv\Scripts\python scripts/prepare_data.py
.venv\Scripts\python scripts/train_baselines.py
```

Each run uses a new output directory; existing outputs are never overwritten.
The seed-42 split contains 5,634 training and 1,409 reference rows. On 2026-10-21,
reference F1 was **0.5727** for Logistic Regression and **0.5501** for Random Forest
at threshold 0.50. Repeat runs produced identical probabilities and metrics.
These are educational baseline measurements, not deployment approval.

See [workflow and model settings](docs/BASELINES.md) and
[full measured results](docs/EXPERIMENTS.md). For the recorded dependency versions:

```powershell
.venv\Scripts\python -m pip install -e . -c constraints-python314.txt
```

The recorded environment is Python 3.14.6 on Windows 11. Other Python versions
within the declared range have not yet been tested in this project.

## Inspect data quality

After preparing the Phase 1 data:

```powershell
.venv\Scripts\python scripts/audit_data.py
```

The command checks schema, missing values, duplicates, cross-split overlap, and
limited target-leakage heuristics. It writes aggregate findings and provenance to
`reports/generated/phase2-data-audit.json`; choose a new `--output` filename to
repeat it. Data and existing reports are never overwritten.

The 2026-10-25 audit produced **91 passing findings, 3 warnings, and no failures**.
Warnings identified 27 repeated predictor rows in training, 4 in reference data,
and 10 reference rows matching training predictors. Customer IDs were disjoint.
Shared predictors are not proof of leakage. The same findings repeated exactly;
all 69 tests available at that phase passed. See [check definitions and limitations](docs/DATA_VALIDATION.md)
and [recorded results](docs/EXPERIMENTS.md).

## Evaluate saved model predictions

After preparing data and training the Phase 1 baselines:

```powershell
.venv\Scripts\python scripts/audit_models.py
```

This evaluates the verified saved probability arrays, without loading or retraining
models. It reports binary classification/ranking scores, Brier score, log loss,
calibration bins, and a descriptive threshold sweep. The default decision threshold
remains 0.50; no new threshold is selected. Choose a fresh `--output` filename for
repeat reports. Outputs stay local under `reports/generated/`.

On 2026-10-26, ROC AUC was **0.8430** for Logistic Regression and **0.8214** for
Random Forest on the 1,409 reference rows. Logistic Regression had one sparse
calibration-bin warning (eight rows in [0.8, 0.9)). All **104 tests available at Phase 3 passed**, and
repeat evaluations matched exactly. See [definitions and limitations](docs/MODEL_EVALUATION.md)
and [the measured comparisons](docs/EXPERIMENTS.md).

## Compare customer segments

```powershell
.venv\Scripts\python scripts/audit_segments.py
```

The default compares the three `Contract` groups at the same 0.50 threshold, using
verified saved predictions. Each group retains its scores, support counts and
warnings. Differences from overall performance and group ranges exclude metrics
with insufficient support. Use `--columns` for other categorical predictors and a
fresh `--output` path to repeat a report. Generated reports stay local.

The 2026-10-26 reference data contains 780 Month-to-month, 295 One year and 334 Two
year customers. The Two year group has only nine positive cases, below the default
ten-positive comparison requirement. The report also flags very few predicted
positives in the annual-contract groups. All **137 tests passed**; repeated reports
matched exactly and the Phase 3 measurements remained unchanged.
See [support rules and limitations](docs/SEGMENT_EVALUATION.md) and
[the measured segment results](docs/EXPERIMENTS.md).

## Compare offline snapshots

Phase 5 compares explicitly declared feature columns, an optional probability column,
and optional binary labels in two CSV snapshots. Predictions and labels must share
their snapshot's row order. Numerical distributions use empirical CDF distance;
categorical distributions and probability histograms use total variation. Missing
values, unseen categories, unsupported performance differences and absent labels
remain explicit. No fitting, model deserialization, threshold tuning or release
approval occurs.

```powershell
.venv\Scripts\python scripts/audit_drift.py --reference data/reference_snapshot.csv --current data/current_snapshot.csv --numeric tenure MonthlyCharges TotalCharges --categorical Contract --probability probability --target Churn --output reports/generated/drift-review.json
```

Prepare these snapshot files explicitly; they are not downloaded or inferred by the
command. Omit prediction/label options for feature-only analysis. Both CSVs must
contain only the declared columns; a declared target may be absent. Existing inputs
and reports are never overwritten. Reports stay local and may include category labels.

On 2026-10-28, all **176 tests passed**, including 39 Phase 5 tests. Repeated audits
of the prepared random split and reference-versus-itself checks agreed exactly.
All 19 saved data/model artifacts were unchanged. These checks verify behavior,
not real production drift. See [input contracts and limits](docs/DRIFT_ANALYSIS.md)
and [the recorded evidence](docs/EXPERIMENTS.md). Phase 5 is published.

## Run one configured audit

After installing the package and creating the Phase 1 data/model artifacts:

```powershell
.venv\Scripts\modelaudit check-config --config configs/audit.yaml
.venv\Scripts\modelaudit run --config configs/audit.yaml --output reports/generated/my-audit
```

For an existing installation, rerun `python -m pip install -e .` in the virtual
environment to register the new command. `python -m modelaudit` is equivalent.
The example runs data quality, evaluation and segment checks. Add a configured
drift check for your own explicitly prepared snapshots; `configs/drift.yaml`
illustrates the required inputs. Paths in YAML resolve relative to that file.

Open `summary.html` in the new output folder for readable results. `report.json`
contains the full evidence, and the final `manifest.json` records completion and
output hashes. Every run needs a fresh folder. Existing inputs and reports are
never overwritten. Configured numeric gates retain unavailable values and support
warnings instead of turning them into passes.

Exit 0 means the configured checks passed; exit 1 means a completed audit needs
review or failed a check; exit 2 means the audit could not complete. These outcomes
are not deployment approval. The recorded baseline audit completed with **16
review items and no data failures**. All **209 tests passed**, repeated results
matched, and all 44 pre-existing data/model/report artifacts remained unchanged.
See [configuration, gate and completion rules](docs/CLI_REPORTS.md) and
[actual Phase 6 results](docs/EXPERIMENTS.md). Phase 6 is published.

## Exercise controlled defects

```powershell
.venv\Scripts\modelaudit experiments --prepared data/prepared --models models/baseline --output reports/generated/my-experiments
```

The command runs an unchanged control and deliberate missingness, duplicates,
identifier overlap, target-copy leakage, missing-column, feature-shift, unseen-category,
probability-inversion, probability-permutation, sparse-cohort and checksum scenarios.
Source data/models are preserved; no model is retrained or deserialized. Generated
evidence and summaries stay in a new ignored output directory.

On 2026-10-30, **12 of 12 scenarios matched their expected observations** in two
runs, with byte-identical case evidence. This means the experiments behaved as
specified, not that the corrupted models/data passed their audits. In particular,
scrambling prediction-to-row alignment left distribution distances at zero while
ROC AUC fell from 0.843 to 0.516 for Logistic Regression and from 0.821 to 0.521 for
Random Forest. Labeled comparisons exposed what marginal drift checks could not.

All **238 tests passed**, including 29 new Phase 7 tests. All 69 pre-existing
data/model/report files were unchanged. See the [scenario protocol and limits](docs/CONTROLLED_EXPERIMENTS.md)
and [recorded results](docs/EXPERIMENTS.md). The initial Phase 7 implementation is published. The October 30 review identified
scope gaps; [the correction record](docs/SCOPE_CORRECTIONS.md) describes the local
follow-up. Phases 8 (CI/package verification) and 9 (MLflow) remain planned. This is an educational
offline toolkit, not a production-readiness certification.

## Fitted-model audits and full CLI controls

Version 2 configuration audits raw training/reference/current CSV snapshots and
uses an explicitly trusted, checksum-pinned fitted binary model. Invalid data
produces findings; blocked prediction-dependent checks remain visible while
independent feature checks continue. No model is fitted during an ordinary audit.
See [the version 2 contract](docs/RAW_AUDITS.md).

The additional synthetic protocol fits two clean laboratory models once each,
then runs five scenarios per model through the ordinary audit CLI:

```powershell
.venv\Scripts\modelaudit experiments --mode fitted --output reports/generated/my-fitted-controls
```

It tests a clean control, contamination, an audit-only target copy, unlabeled
feature drift, and a group-specific degradation hidden by overall F1. These are
invented rows, not new measurements on real customers. The original saved-score
protocol remains available with `--mode saved` and now checks gate verdicts too.

## Development

Follow [the phase plan](docs/PHASE_PLAN.md) in order. Each completed phase
receives tests, recorded evidence, and a commit made when the work occurs.
See the [build log](docs/BUILD_LOG.md) and [architecture](docs/ARCHITECTURE.md).

This implementation follows the previously established phase roadmap with
new foundation code and its own history. Development dates in the build log
record actual work; commit timestamps record when commits are made.

Future results will support human review, not automatically approve a model
or guarantee the absence of leakage, bias, or production failures. Datasets,
models, credentials, and generated reports remain outside Git.
