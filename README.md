# ModelAudit

An incremental project for examining data quality and model behavior before
a release. Its intended scope is offline evaluation of scikit-learn binary
classifiers, with explicit evidence and limitations.

**Current stage: Phase 2 — data validation.** The package includes configuration
loading, logging, reproducible Telco preparation, two baseline pipelines, and
read-only data-quality checks. Model auditing, full reporting, and the public
validation CLI remain later-phase work.

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
all 69 tests passed. See [check definitions and limitations](docs/DATA_VALIDATION.md)
and [recorded results](docs/EXPERIMENTS.md).

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
