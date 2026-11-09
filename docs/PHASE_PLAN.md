# Phase plan

This order follows the existing ModelGate roadmap. Later phases are planned
scope, not completed capabilities. No future completion dates are assigned.

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 | Package, safe configuration loading, logging, docs, tests | Implemented |
| 1 | Reproducible data preparation and baseline models | Published |
| 2 | Schema, missingness, duplicates, overlap and leakage heuristics | Published |
| 3 | Binary model metrics, calibration and threshold analysis | Published |
| 4 | Segment performance comparisons and support warnings | Published |
| 5 | Feature/prediction drift and labeled performance changes | Published |
| 6 | Full configuration, CLI, orchestration and reports | Published |
| 7 | Controlled defective scenarios and recorded experiments | Published with scope corrections at `59a6db6` |
| 8 | Automated testing, coverage gates, CI and package verification | Published; all hosted checks passed at 0c099a7 |
| 9 | MLflow experiment tracking and integration | Implemented and verified locally on 2026-11-09; awaiting push approval |

The original Phase 0–7 commits and phase branches remain published unchanged.
The October 30 review found scope gaps; see [the correction record](SCOPE_CORRECTIONS.md).
The Phase 7 correction was published on October 30. See [Phase 8 verification](TESTING_CI.md) for the subsequent test and package workflow.

For each phase: define acceptance criteria, implement its scope, run tests,
record actual findings, then commit locally and request explicit push approval. Use phase branches for
subsequent work and preserve real commit timestamps. Dataset choice,
provenance, and licensing must be resolved before Phase 1 acquisition.
