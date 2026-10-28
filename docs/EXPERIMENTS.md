# Phase 1 experiments

## Measured on 2026-10-21 (Asia/Calcutta)

Verified pinned source: 970,457 bytes; 7,043 rows and 21 columns.
Class counts: 5,174 No and 1,869 Yes.
TotalCharges has 11 blank numerical cells; these remain missing until training-only imputation.

The seed-42 stratified split has 5,634 training rows (1,495 positives) and 1,409 reference rows (374 positives).

| Metric | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Precision | 0.63517915 | 0.62372881 |
| Recall | 0.52139037 | 0.49197861 |
| F1 | 0.57268722 | 0.55007474 |
| PR-AUC (trapezoidal) | 0.65965537 | 0.60849891 |
| Average precision | 0.66038218 | 0.60655711 |

Both models use threshold 0.50 and 46 transformed features.

Confusion matrices (`[[TN, FP], [FN, TP]]`):

- Logistic Regression: `[[923, 112], [179, 195]]`.
- Random Forest: `[[924, 111], [190, 184]]`.

## Repeatability and provenance

- Two independent preparation runs produced identical CSV hashes and split membership.
- Two training runs produced identical metrics for both models.
- Maximum absolute difference between repeated probabilities was 0.0 for both models.
- Each saved pipeline reproduced its in-memory probabilities after reload at
  `rtol=0, atol=1e-12`.
- No training warnings were emitted in either run.
- Prepared dataset hashes remained valid after training.

Training CSV SHA-256: `54b450a3742f15c9dc6cfb5f43dca14ffcb86d26af302489791996dcf46a6667`.

Reference CSV SHA-256: `59a0aefabe95f2a822ac256205825487dcaee9dc0e65c0f6369a212d667377e4`.

Local evidence is retained at `reports/generated/phase1_reproducibility.json`
and in the ignored preparation and training output directories. Run summaries
contain actual UTC times; the work date above uses Asia/Calcutta.

These runs occurred before the Phase 1 commit. Their recorded Git parent is
`099c7f3b5e8dd47ef2d3c4bb6a29e0c1323e6fe3` with `dirty=true`,
which accurately represents testing the uncommitted Phase 1 implementation.
The Phase 0 parent is not claimed to contain this training code.

Environment: Python 3.14.6 on Windows 11.
Tested dependency versions are pinned in `constraints-python314.txt`.

## Interpretation

These are descriptive single-split baselines, not a release gate, temporal
forecast, or fairness claim. No threshold or parameter tuning was performed.
Calibration, segment behavior, drift, and controlled defective scenarios remain
later-phase work. Reproducibility here is within one environment, not a promise
of bitwise agreement on every platform or future library version.

## 2026-10-25 — Phase 2 data-quality audit

Ran the new checks twice against the existing seed-42 Phase 1 partitions. No data
was downloaded, repaired, resplit, or used to refit the stored baseline models.
All findings, policies and input hashes matched exactly between runs; context
creation times differ by design. All four prepared artifact files were unchanged.

| Measurement | Training | Reference |
| --- | ---: | ---: |
| Rows | 5,634 | 1,409 |
| Missing TotalCharges | 10 (0.1775%) | 1 (0.0710%) |
| Exact duplicate rows including ID and target | 0 | 0 |
| Duplicate predictor rows beyond first | 27 | 4 |
| Reference IDs also present in training | — | 0 |
| Reference predictor rows also present in training | — | 10 (0.7097%) |

The report contains **91 pass, 3 warn, 0 fail, and 0 skipped** findings. The warnings
are predictor duplication in each partition and predictor overlap across them.
All required columns and values satisfied the declared schema; the limited target
association heuristics did not flag a predictor. Neither that result nor distinct
customer IDs proves absence of leakage. Equal predictors can describe different
customers, so the warnings are preserved for review rather than silently deduplicated.

The full suite passed **69 tests in 13.280 seconds**, with warnings treated as
errors. This includes 33 Phase 2 tests covering malformed schema/types, missingness
boundaries, duplicates, overlap counts, exact/inverted target copies, numerical and
categorical association, small/sparse/constant inputs, numerical precision, JSON,
input preservation, command behavior, checksums and overwrite protection. The
previous 36 tests still pass. Dependency consistency checks passed; no dependency
versions changed from Phase 1.

Local ignored evidence:
- `reports/generated/phase2-data-audit-20261025.json`
- `reports/generated/phase2-data-audit-20261025-repeat.json`
- `reports/generated/phase2-verification-20261025.json`

Reports record Python 3.14.6 and the runtime package versions, and honestly record
parent commit `c038412` with a dirty working tree because verification preceded
the Phase 2 commit. This entry records local verification, not remote publication.

## 2026-10-26 — Phase 3 model evaluation

Evaluated the existing saved Phase 1 probabilities twice on the same 1,409
reference rows: 374 positives and 1,035 negatives (26.5436% prevalence). No model
was deserialized, refitted, recalibrated, or selected. All 19 existing data/model
artifacts retained their hashes. Results excluding creation context matched exactly.

| Metric | Logistic Regression | Random Forest |
| --- | ---: | ---: |
| Accuracy at 0.50 | 0.793471 | 0.786373 |
| Balanced accuracy at 0.50 | 0.706589 | 0.692366 |
| Precision at 0.50 | 0.635179 | 0.623729 |
| Recall at 0.50 | 0.521390 | 0.491979 |
| Specificity at 0.50 | 0.891787 | 0.892754 |
| F1 at 0.50 | 0.572687 | 0.550075 |
| ROC AUC | 0.843035 | 0.821429 |
| PR AUC (trapezoidal) | 0.659655 | 0.608499 |
| Average precision | 0.660382 | 0.606557 |
| Brier score | 0.137061 | 0.146849 |
| Log loss | 0.418178 | 0.490599 |
| ECE (10 uniform bins) | 0.029187 | 0.035922 |

Fixed-threshold confusion matrices, precision, recall, F1, PR AUC and average
precision match the original Phase 1 summary exactly. The lower probability losses
and ECE here are sample measurements, not certification of generalization or
calibration. Phase 2's predictor-overlap warnings remain applicable.

Calibration-bin counts, from [0, 0.1) through [0.9, 1]:

- Logistic Regression: 539, 193, 143, 115, 112, 99, 117, 83, 8, 0.
- Random Forest: 500, 215, 165, 123, 111, 85, 70, 57, 57, 26.

Logistic Regression has eight observations in [0.8, 0.9), below the default
20-row support threshold; the report preserves that warning. Its final bin is empty
and retains null means/gaps. Random Forest has no occupied bin below the threshold.

The grid compares thresholds 0.1 through 0.9. Selected descriptive rows are below;
these are not recommendations or optimized settings.

| Model | Threshold | Precision | Recall | F1 | Predicted positive rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.30 | 0.539326 | 0.770053 | 0.634361 | 534 |
| Logistic Regression | 0.50 | 0.635179 | 0.521390 | 0.572687 | 307 |
| Logistic Regression | 0.70 | 0.835165 | 0.203209 | 0.326882 | 91 |
| Random Forest | 0.30 | 0.519849 | 0.735294 | 0.609081 | 529 |
| Random Forest | 0.50 | 0.623729 | 0.491979 | 0.550075 | 295 |
| Random Forest | 0.70 | 0.700000 | 0.262032 | 0.381323 | 140 |

All **104 tests passed in 29.734 seconds** with warnings treated as errors, including
35 new Phase 3 tests. Tests cover hand calculations, ties, perfect/inverted scores,
single-class/undefined metrics, decimal calibration boundaries and their neighboring
floating-point values, empty/sparse bins, threshold semantics, strict input/JSON,
artifact corruption and row-order binding, input preservation and overwrite protection.
Dependency consistency checks passed without version changes.

Final ignored local evidence:
- `reports/generated/phase3-model-audit-20261026-verified.json`
- `reports/generated/phase3-model-audit-20261026-verified-repeat.json`
- `reports/generated/phase3-verification-20261026-final.json`

Reports record Python 3.14.6 and actual runtime versions. Their context honestly
records parent commit `0fb3353` with a dirty working tree, because measurements were
made before the Phase 3 commit. This records local verification, not publication.

## 2026-10-26 — Phase 4 Contract comparisons

Evaluated both saved models twice at threshold 0.50, grouping the same 1,409
reference rows by Contract. The groups contain 780 Month-to-month customers (329
positives), 295 One year customers (36 positives), and 334 Two year customers
(9 positives). There are no missing Contract values. Each grouping retains every
row and reproduces the overall confusion counts and support.

| Model | Contract | Predicted positives | Precision | Recall | F1 | Supported F1 difference from overall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Month-to-month | 307 | 0.635179 | 0.592705 | 0.613208 | 0.040520 |
| Logistic Regression | One year | 0 | null | 0.000000 | 0.000000 | -0.572687 |
| Logistic Regression | Two year | 0 | null | 0.000000 | 0.000000 | null |
| Random Forest | Month-to-month | 292 | 0.626712 | 0.556231 | 0.589372 | 0.039297 |
| Random Forest | One year | 3 | 0.333333 | 0.027778 | 0.051282 | -0.498793 |
| Random Forest | Two year | 0 | null | 0.000000 | 0.000000 | null |

All groups meet the 30-row requirement, but Two year has only nine actual positives,
below the ten-positive requirement for recall, F1, balanced-accuracy and ranking
comparisons. Its defined raw measurements remain visible; affected comparisons
are null with explanations. This is a support limitation, not a claim that the
measurements are zero or that the group performed acceptably.

At 0.50, Logistic Regression predicts no positives for either annual-contract group,
so their precision is undefined. Random Forest predicts only three positives for
One year; its measured precision is 1/3, but that comparison is withheld because
it needs ten predicted positives. It predicts none for Two year. Month-to-month
has no support warnings for either model. There are six support-warning entries
across the six model/group combinations, distinct from metric exclusion reasons.

Supported F1 ranges use Month-to-month and One year only; Two year is excluded:

- Logistic Regression: minimum 0.000000, maximum 0.613208, gap 0.613208.
- Random Forest: minimum 0.051282, maximum 0.589372, gap 0.538090.

These gaps are descriptive sample differences, not significance, causation or
fairness conclusions. The overall comparator includes each group. No model,
threshold, or data split was changed in response to these results; the Phase 2
predictor-overlap warnings remain relevant.

All **137 tests passed in 24.391 seconds** with warnings treated as errors, including
33 new Phase 4 tests. Coverage includes hand-calculated scores/deltas/ranges, support
threshold boundaries and denominator-specific exclusions, missing and single-class
groups, no predicted positives, independent columns, alignment, cardinality limits,
input preservation, integrity failures and command behavior. Dependency consistency
checks passed. The shared loader left the full real Phase 3 report unchanged
apart from creation context. All 19 data/model artifacts were hash-identical before
and after both repeat segment runs and the Phase 3 regression run.

Final ignored local evidence:
- `reports/generated/phase4-segment-audit-20261026.json`
- `reports/generated/phase4-segment-audit-20261026-repeat.json`
- `reports/generated/phase4-phase3-regression-20261026.json`
- `reports/generated/phase4-verification-20261026.json`

Reports record actual runtime versions and UTC creation times. Their Git context
honestly records parent `7cd6198` with a dirty working tree because measurement
preceded the Phase 4 commit. This entry records local verification, not publication.

## 2026-10-28 — Phase 5 snapshot comparison verification

Compared the unchanged 5,634-row training partition (reference argument) against
the 1,409-row held-out partition (current argument) across all 19 predictors.
These names specify comparison direction, not time order: both are from the same
seed-42 random stratified split. No production-period data is available here.

| Numeric predictor | Reference/current observed rows | Empirical CDF distance |
| --- | ---: | ---: |
| tenure | 5,634 / 1,409 | 0.010146749193089777 |
| MonthlyCharges | 5,634 / 1,409 | 0.029838733855812538 |
| TotalCharges | 5,624 / 1,408 | 0.020842089098668026 |

TotalCharges has ten missing training observations and one missing held-out
observation. Its missing-fraction difference is -0.0010652146692253991, measured
separately from its nonmissing distribution. Other numeric predictors have no
missing values. All observed feature supports meet the default 30-row rule.

Categorical total variation ranges from 0.004521493628489526 (Contract) to
0.0279950155612545 (StreamingTV). There are no unseen categorical values in the
held-out split. These are descriptive sample differences, without a drift cutoff,
significance claim or approval decision. Phase 2's predictor-overlap warnings
remain relevant; this comparison does not establish independent populations.

For each saved baseline, exported a local snapshot containing the verified held-out
features, labels and its aligned saved probability array. Compared that snapshot
with itself at threshold 0.50: all 19 feature distances, prediction KS distance,
prediction-histogram total variation and all 11 supported performance differences
are exactly zero for both models. This is an identity sanity check, not evidence
that production behavior is stable. Nothing was retrained or deserialized.

Ran the feature comparison and both model checks twice. Results match exactly
after excluding runtime context. All 19 pre-existing data/model files retain their
original SHA-256 values. Source manifests and prediction checksums were verified
before exporting; local snapshot CSVs and generated reports remain ignored by Git.

All **176 tests passed in 31.222 seconds** with warnings treated as errors, including
39 new Phase 5 tests. Coverage includes hand-computed distances, an independent
SciPy statistic cross-check, ties, unequal sizes, all-missing cases, category unions,
support boundaries, score directions, absent/partial labels, decimal probability
bins, invalid types/schema, input immutability, strict JSON, CSV integrity and command
behavior. A fixture demonstrates that identical marginal probability distributions
can coexist with a large performance loss. It is a unit-test fixture, not a recorded
real-data defect experiment. Phase 7 controlled experiments remain planned.
Dependency checks passed without package changes.

Ignored local evidence:

- `reports/generated/phase5-features-20261028.json` and its `-repeat.json` counterpart
- `reports/generated/phase5-logistic_regression-self-20261028.json` and repeat
- `reports/generated/phase5-random_forest-self-20261028.json` and repeat
- `reports/generated/phase5-verification-20261028.json`
- `reports/generated/phase5-snapshots-20261028/` (the four source CSV snapshots)

Reports retain actual UTC runtime timestamps and parent commit `afa76e2` with a
dirty working tree: verification preceded the new commit. Phase 5 is verified
locally and awaits separate explicit push approval.

## 2026-10-29 — Phase 6 configured command and reports

Registered the installed `modelaudit` command in the independent checkout's existing
virtual environment. The initial no-build-isolation installation could not find
setuptools in that environment; a normal isolated build registered the command
successfully, without adding/upgrading runtime dependencies. Dependency checks passed.

Ran the shipped baseline configuration once and a local all-checks configuration
twice. All three runs used the installed command from a different working directory,
verifying that input paths resolve from the configuration and not the shell location.
The all-checks configuration combines the unchanged baseline artifacts with the
existing Phase 5 Logistic Regression reference snapshot compared with itself.
It is a workflow/identity check, not a new production-period experiment. No metric
gate thresholds were invented for these recorded runs (`gates: []`).

| Run | Enabled checks | Outcome | Exit | Review locations | Data failures | Output files hashed |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline | data, evaluation, segments | review | 1 | 16 | 0 | 6 |
| all | data, evaluation, segments, drift | review | 1 | 16 | 0 | 7 |
| all-repeat | data, evaluation, segments, drift | review | 1 | 16 | 0 | 7 |

The 16 review items comprise three data-quality warnings, two evaluation warning/
undefined-evidence locations, and eleven segment warning/undefined-evidence locations.
These include the existing predictor duplicates/overlap, sparse calibration support,
unsupported precision and small-group support. Locations can share a cause, so 16
does not mean 16 distinct defects. Drift identity comparisons add no review items.
Exit 1 correctly reports the need for review; all checks and report writes completed.

Every output hash matched its completion manifest. The two all-check reports match
exactly after excluding actual runtime context. Full Phase 3, Phase 4 and Phase 5
adapter outputs also match their prior saved reports after excluding context.
All 44 pre-existing files under data/, models/ and reports/generated/ retained their
hashes, including the 19 saved data/model artifacts. No models were deserialized.

All **209 tests passed in 73.465 seconds** with warnings treated as errors, including
33 new Phase 6 tests. Tests cover strict schemas, defaults/path resolution, duplicate
and unsafe YAML, optional checks, gate bounds and unavailable evidence, all-check
integration, output hashes, input/config changes, protected output locations,
partial failures, repeatability, report escaping, and command exit codes. HTML
structure was parsed to verify tables and absence of scripts/remote assets;
Markdown label escaping prevents injected links/images/HTML. Installed command
version and dependency consistency checks also passed.

Ignored local evidence:

- `reports/generated/phase6-baseline-20261029/`
- `reports/generated/phase6-all-20261029/`
- `reports/generated/phase6-all-repeat-20261029/`
- `reports/generated/phase6-inputs-20261029/all-checks.yaml`
- `reports/generated/phase6-verification-20261029.json`

Run directories include their full JSON evidence, Markdown/HTML summaries and
completion manifests. Provenance records real UTC times and parent `1cc99e8` with
a dirty working tree because verification preceded the Phase 6 commit. Phase 7
controlled experiments remain planned; Phase 6 publication awaits explicit approval.
