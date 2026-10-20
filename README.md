# ModelAudit

An incremental project for examining data quality and model behavior before
a release. Its intended scope is offline evaluation of scikit-learn binary
classifiers, with explicit evidence and limitations.

**Current stage: Phase 0 — foundation.** Configuration loading, package-scoped
logging, and foundation tests are implemented. Model training, validation,
reporting, and the public CLI are planned, not currently available.

## Local setup

Python 3.11 or newer is required. From this project's directory on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m unittest discover -s tests -v
```

The import package is `modelaudit`. `configs/foundation.yaml` is a minimal
example; it is not yet a model-validation configuration.

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
