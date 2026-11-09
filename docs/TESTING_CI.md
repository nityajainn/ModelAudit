# Phase 8 verification

The Verify ModelAudit workflow checks pushes to main and phase branches and pull requests targeting main. The Linux jobs use Ubuntu 24.04 with Python 3.11 and 3.12. The Windows job uses Windows 2025 with Python 3.12. Each test job installs the project with its optional verify and tracking dependencies, checks dependency consistency, and runs the full suite on synthetic fixtures. Jobs have read-only repository access. The workflow pins its actions to reviewed release commits.

## Coverage and code checks

Linux/Python 3.12 collects coverage for the application and its CLI child processes. The gate uses integer counts to require at least 90% line coverage and at least 90% combined line and branch coverage. It also requires every statement in the module CLI entry point to execute. Tests and release helpers are outside the application measurement. GitHub retains JUnit and coverage results for seven days.

Ruff checks syntax and Pyflakes rules across the repository. The Phase 8 and new Phase 9 Python files also have a format check and run the default lint rules. This scope preserves the formatting of previously published phases.

From the checkout on Windows, choose a fresh evidence directory for each run. Run each command after the preceding command succeeds.

```powershell
python -m pip install -e ".[verify,tracking]"
python -m pip check
$evidence = New-Item -ItemType Directory reports/generated/local-ci-run
$env:COVERAGE_FILE = Join-Path $evidence.FullName '.coverage'
$env:COVERAGE_RCFILE = (Resolve-Path pyproject.toml).Path
python -m coverage run -m pytest -q -p no:cacheprovider "--junitxml=$evidence/junit.xml"
python -m coverage combine
python -m coverage json --fail-under=0 -o "$evidence/coverage.json"
python -m coverage xml --fail-under=0 -o "$evidence/coverage.xml"
python scripts/check_release.py coverage "$evidence/coverage.json"
python -m coverage report
```

## Distribution and console checks

Build from the project root with `python -m build`. The isolated backend creates a source archive and builds the wheel from that archive. Run `python scripts/check_release.py archives dist` to check the package modules, metadata, license and console command. The source archive also includes the public examples, tests, development scripts and documentation. Unexpected archive files, traversal paths and links fail verification.

Install the resulting wheel in a fresh virtual environment. Copy `scripts/wheel_smoke.py` to a temporary directory outside the checkout. From that directory run the **wheel environment's Python** with `-I wheel_smoke.py <fresh-output-directory>`. The check verifies the installed distribution and invokes its actual console launcher in that environment's scripts directory.

| Installed command case | Required evidence |
| --- | --- |
| Clean inputs | Pass, exit 0, correct group metrics, zero drift and labeled changes |
| 20 contaminated rows | Fail, exit 1, exact identifier and predictor overlap counts |
| Missing reference CSV | Execution error, exit 2, missing filename in the error and no completed report |

The package job uploads the synthetic reports and their compact summary. Local test and package evidence belongs under the ignored `reports/generated/` directory. The test matrix and package workflow can be observed on GitHub after the approved push. Phase 9 covers MLflow.

## Phase 9 integration checks

The matrix installs the tracking extra and tests real SQLite runs, exact artifacts,
unchanged source reports, retries, invalid reports and handled write/readback failures.
The package job checks a core-only wheel, installs its tracking extra, then repeats
the smoke with --tracking in a new directory. It records clean and contamination
audits and repeats the clean report: distinct run IDs, identical report hashes.

Phase 8 correction 0c099a7 passed all five hosted jobs on main and phase/8-testing-ci.
Phase 9 hosted checks await an approved push. Stores and generated evidence stay out of Git.
