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
