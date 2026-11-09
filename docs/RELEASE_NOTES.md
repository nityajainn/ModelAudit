# Phase 9 release candidate

Phase 9 adds optional local MLflow recording with exact artifact verification,
searchable outcomes, repeat runs and available model metadata. Core offline
commands still work without MLflow.

This candidate awaits publication approval. No release tag, GitHub release,
package-index upload or deployment is created. Version 0.1.0 remains unchanged;
phase boundaries are recorded by real Git commits and phase branches.

## Reproduce from a fresh checkout

1. Check out the reviewed ModelAudit commit in a new directory.
2. Create a Python 3.11+ environment and install ".[verify,tracking]".
3. Run the test and coverage checks in TESTING_CI.md.
4. Build the source archive and wheel, then run the archive checks.
5. Install the wheel in another fresh environment. Run wheel_smoke.py outside the
   checkout with that environment's Python and a fresh output directory.
6. Install its tracking extra, repeat with --tracking and a different output
   directory, then inspect summary.json.

The smoke requires clean=0, contamination=1 and missing=2. Its three tracking runs
record clean, contamination and repeated clean, with exact artifact readback.
It uses invented data and a clean fitted baseline through ordinary CLI commands.
No real customer evidence is claimed.

## Limits

- Tracking records evidence; it does not change audit decisions.
- Hash consistency is not proof of authenticity or independent reproduction.
- Missing older training metadata remains missing.
- No remote server, model registry, automatic training or deployment is added.
- Hosted Phase 9 CI results are available only after an approved push.

## Local verification on 2026-11-09

Windows/Python 3.12.14: 307 tests and 266 subtests passed. Line coverage was 97.11%;
combined line/branch coverage was 96.35%. Archive, lint, format and dependency checks
passed. A fresh installed wheel passed the core audit and three tracking round trips.
MLflow 3.18.0 / SQLAlchemy 2.1.8 produced one upstream deprecation warning.
Run IDs and report hashes are recorded in BUILD_LOG.md. Hosted Phase 9 CI is pending.
