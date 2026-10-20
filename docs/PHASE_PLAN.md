# Phase plan

This order follows the existing ModelGate roadmap. Later phases are planned
scope, not completed capabilities. No future completion dates are assigned.

| Phase | Deliverable | Status |
| --- | --- | --- |
| 0 | Package, safe configuration loading, logging, docs, tests | Implemented |
| 1 | Reproducible data preparation and baseline models | Verified locally; awaiting push approval |
| 2 | Schema, missingness, duplicates, overlap and leakage heuristics | Planned |
| 3 | Binary model metrics, calibration and threshold analysis | Planned |
| 4 | Segment performance comparisons and support warnings | Planned |
| 5 | Feature/prediction drift and labeled performance changes | Planned |
| 6 | Full configuration, CLI, orchestration and reports | Planned |
| 7 | Controlled defective scenarios and recorded experiments | Planned |

For each phase: define acceptance criteria, implement its scope, run tests,
record actual findings, then commit locally and request explicit push approval. Use phase branches for
subsequent work and preserve real commit timestamps. Dataset choice,
provenance, and licensing must be resolved before Phase 1 acquisition.
