# Build log

## 2026-10-17 — Foundation draft

Created the initial package, YAML configuration loader, logger, tests, and
phase roadmap in a separate local folder. Installation and testing did not
proceed during that session. No Git history or remote publication was created.

## 2026-10-21 — Phase 0 verified; awaiting push approval

- Finalized the ModelAudit name, MIT license, and project instructions.
- Saved the requirement to obtain explicit user approval before every phase push.
- Installed the new package in its own isolated environment with Python 3.14.6
  and PyYAML 6.0.3; no existing project environment was changed.
- Passed all nine foundation tests in 0.156 seconds. Coverage includes valid
  mappings, invalid roots, duplicate/nonstring keys, malformed YAML, unsafe
  Python tags, missing files, unchanged input files, and repeatable logging
  without modifying the root logger.
- Verified dependency consistency with pip check.
- Confirmed nityajainn/ModelAudit is the new, empty public destination.
- Phase 0 contains no training, dataset download, validation CLI, or later-phase
  functionality. Phases 1–7 remain planned.
- Prepare a new root commit attributed to nityajainn using the actual commit
  time. No inherited commits, invented dates, or co-author trailers.

Publication requires separate user approval and is not claimed by this log.

## 2026-10-21 — Phase 0 publication confirmed

After explicit user approval, root commit `099c7f3` was published on main and
phase/0-foundation. Both remote branch hashes and all 13 files were verified.
GitHub attributed the commit to nityajainn only.

## 2026-10-21 — Phase 1 data and baselines

- Started phase/1-data-baselines from the clean published Phase 0 commit.
- Reviewed the pinned IBM source and repository license before acquisition;
  documented the dataset-license distinction and kept data out of Git.
- Added new acquisition, parsing, split, provenance, baseline training, and
  development-command implementations without importing another project's code
  or commits. All work is isolated in this repository.
- Passed 36 tests: the nine existing tests plus 27 Phase 1 tests. The complete
  run took 5.323 seconds. Dependency consistency checks passed.
- Downloaded and checksum-verified the 970,457-byte source. Prepared 5,634 training
  and 1,409 reference rows twice using seed 42 and stratification.
- Trained separate Logistic Regression and Random Forest pipelines twice.
  Recorded actual measurements in EXPERIMENTS.md. No training warnings occurred.
- Verified identical prepared CSVs, exact repeated probabilities and metrics,
  save/reload agreement, and preserved preparation hashes.
- Recorded the new runtime versions in constraints-python314.txt. Dependencies
  were installed only in this project's isolated environment.
- Phase 2 and later remain unimplemented. Phase 1 publication requires separate
  explicit user approval; no Phase 1 push is claimed here.

## 2026-10-22 — Phase 1 publication confirmed

After explicit user approval, commit `c038412` was published on main and
phase/1-data-baselines. Phase 0 remained at `099c7f3`. Remote branches and all
23 published file hashes were verified; GitHub listed only nityajainn as contributor.
The actual October 21 commit timestamp was preserved. Original-project file and
remote checks confirmed no changes.

## 2026-10-25 — Phase 2 data validation verified locally

- Started `codex/phase-2-data-validation` from published Phase 1, in this independent
  repository. Recorded preservation fingerprints of the separate original project.
- Defined acceptance criteria in DATA_VALIDATION.md and implemented read-only schema,
  missingness, duplicate, split-overlap and limited leakage-association checks.
- Added structured aggregate findings with explicit skipped states and documented
  heuristic limitations. Added an adapter and development command for verified
  Phase 1 partitions, with input hashes, runtime provenance and overwrite protection.
- Passed all 69 tests with warnings treated as errors in 13.280 seconds, including
  33 new Phase 2 tests. Dependency consistency checks passed without upgrades.
- Audited the prepared Telco partitions twice: 91 passing findings, three review
  warnings, zero failures, zero skips. Findings repeated exactly, and all prepared
  files were unchanged. Actual results and warnings are in EXPERIMENTS.md.
- Kept raw data, prepared data, stored models and generated reports outside Git.
  No original-project code or commits were imported. Verified the nityajainn account
  and the intended repository identity before preparing the new local commit.
- Phase 3 and later remain planned. Phase 2 publication requires separate explicit
  user approval; this log does not claim a Phase 2 push.

## 2026-10-25 — Phase 2 publication confirmed

After explicit user approval, `0fb3353` was published on main and
phase/2-data-validation. Earlier phase branches remained unchanged. Remote hashes
and all 28 published files matched locally; GitHub listed only nityajainn as
contributor. All 13,069 original-project files and remotes remained unchanged.

## 2026-10-26 — Phase 3 model evaluation verified locally

- Started `codex/phase-3-model-evaluation` from published Phase 2 in this independent
  repository. Recorded original-project file fingerprints before implementation.
- Defined acceptance criteria and metric conventions in MODEL_EVALUATION.md.
  Added binary classification/ranking metrics, probability scores, supported
  calibration bins and threshold comparisons with explicit undefined outcomes.
- Added a hash-verifying adapter for the two existing Phase 1 probability arrays
  and a local development command. No models are deserialized or retrained.
- Passed 104 tests in 29.734 seconds with warnings treated as errors. The 35 new
  tests include a decimal-boundary regression and corrupt/empty artifact handling.
  Dependency consistency checks passed without upgrades.
- Evaluated both baseline arrays twice. Results matched exactly and prior Phase 1
  measurements remained equal. All 19 saved data/model artifacts were unchanged.
  Recorded actual scores, threshold comparisons and the sparse-bin warning in
  EXPERIMENTS.md. Generated reports stay local and ignored by Git.
- Verified the nityajainn account and intended repository identity before preparing
  the local commit. No existing history was imported, rewritten or backdated.
- Phase 4 and later remain planned. Phase 3 publication requires separate explicit
  user approval; no Phase 3 push is claimed here.

## 2026-10-26 — Phase 3 publication confirmed

After explicit user approval, `7cd6198` was published on main and
phase/3-model-evaluation. Earlier branches and timestamps were preserved. All 33
published files matched locally, and GitHub listed only nityajainn as contributor.
The original project and its remotes were unchanged by that publication.

## 2026-10-26 — Phase 4 segment evaluation verified locally

- Started `codex/phase-4-segment-evaluation` from published Phase 3 in this independent
  checkout. Captured the current original-project baseline before implementation.
- Defined acceptance criteria in SEGMENT_EVALUATION.md. Implemented independent
  categorical comparisons with group/overall metrics, missing-group accounting,
  metric-specific support rules, signed differences and supported-group ranges.
- Shared the existing verified-prediction loader with Phase 3 and added the segment
  adapter/development command. No model deserialization or retraining occurs.
- Passed 137 tests in 24.391 seconds with warnings treated as errors, including 33
  new Phase 4 tests. Dependency checks passed without installing or updating packages.
- Evaluated Contract groups twice: all results matched. Preserved unsupported
  comparisons and warnings for nine positive Two year cases and too few predicted
  positives in annual-contract groups. Actual measurements are in EXPERIMENTS.md.
- The full real Phase 3 report remained unchanged apart from runtime context. All
  19 saved data/model artifacts retained their hashes. Reports remain ignored.
- Verified all 13,112 original-project files (including Git state) against this
  phase's starting snapshot with no changes. Original remotes remain unchanged.
- Verified the nityajainn account and destination identity. No inherited commits,
  history rewrites, altered dates, or co-author trailers were introduced.
- Phases 5–7 remain planned. Phase 4 requires separate explicit push approval;
  no Phase 4 publication is claimed by this log.

## 2026-10-28 — Phase 4 publication rechecked

Confirmed the prior approved Phase 4 publication: main and
phase/4-segment-evaluation both point to `afa76e2`. Phase 0–3 remote branches remain
at their existing commits, with no tags. The second account and repository identity
were verified as nityajainn and repository 1427414883. No remote state was changed.

## 2026-10-28 — Phase 5 drift analysis verified locally

- Started `codex/phase-5-drift-analysis` from published Phase 4 in the independent
  ModelAudit-Lab checkout. Captured a fresh baseline of 13,112 original-project files
  including its Git state. Its working tree was clean and origin remained ModelGate.
- Defined acceptance criteria in DRIFT_ANALYSIS.md. Implemented read-only numerical,
  categorical and probability distribution comparisons, explicit missing/new-category
  accounting, and labeled performance deltas using existing metric/support rules.
- Added a strict CSV adapter and development command with same-row prediction/label
  alignment, hash checks, provenance and exclusive report creation. No model loading,
  threshold tuning, dependency changes, or edits to prior phase implementations.
- Passed 176 tests in 31.222 seconds with warnings treated as errors, including
  39 new Phase 5 tests. Dependency consistency checks passed. A row-permutation test
  uses a numerical tolerance for ordinary last-bit floating-point reduction changes.
- Repeated all-19-feature split comparisons and both saved-model identity checks.
  Outputs matched exactly apart from runtime context; all 19 saved data/model files
  were unchanged. Actual results and limits are recorded in EXPERIMENTS.md.
- These are offline split/identity checks, not production-period observations.
  Generated snapshots/reports remain local and ignored. Phases 6–7 remain planned.
- Rechecked all 13,112 original-project files against the starting baseline with
  no changed paths. The original remotes and published Phase 0–4 branch hashes
  remain unchanged. Verified nityajainn authentication without switching accounts.
- Phase 5 requires separate explicit push approval. No Phase 5 publication is
  claimed, and no commit timestamp is fabricated or overridden.
