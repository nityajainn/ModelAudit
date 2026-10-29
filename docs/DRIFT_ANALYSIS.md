# Phase 5 acceptance criteria and comparison contract

The original phase protocol below remains applicable to its saved-artifact API.
The October 30 scope corrections add the [raw fitted-model workflow](RAW_AUDITS.md);
see the [correction record](SCOPE_CORRECTIONS.md) for current capabilities.

Compare two explicitly named offline snapshots without fitting, repairing, or
overwriting inputs. Feature drift must work without predictions or labels;
prediction drift must work without labels. Compare labeled binary performance
only when both snapshots have complete aligned labels and probabilities.

## Inputs and row alignment

`drift.compare_snapshots(reference, current, numeric=..., categorical=...,
probability=..., target=...)` accepts nonempty pandas tables. Roles are explicit,
unique and disjoint. Columns must match the declared roles, except that a declared
target may be absent in either snapshot. Exclude identifiers from predictor roles.
Predictions and labels must occupy the same rows as features in their own table;
tables can have different lengths. Index labels do not join the two populations.
The caller must establish population meaning, model identity, positive-class
semantics and row alignment before exporting snapshots. The tool cannot prove
these from values alone. Comparing different models confounds model and data change.

Numerical features allow missing values but reject strings, booleans, infinities
and integers that cannot be represented exactly as float64. Categorical features
allow strings and missing values; blank strings become missing, other spelling is
preserved. Missing is distinct from a literal string such as `__MISSING__`.
Probabilities must be finite numbers in [0,1]; labels must be complete numeric 0/1.
Partial labels fail explicitly; there is no silent dropping or imputation.

## Measurements

- Numeric features: two-sample empirical CDF distance (KS statistic), computed on
  nonmissing observations. It is the maximum absolute difference of the two
  empirical CDFs, between 0 and 1. Ties are evaluated together. Missing fractions
  and their signed current-minus-reference difference are separate measurements.
  An all-missing side has a null distance with a reason, never a zero drift claim.
- Categorical features: total variation, `0.5 * sum(abs(p_current-p_reference))`,
  across the union of observed categories including a distinct missing category.
  Report counts/fractions and new-category counts/fraction; a new missing value
  affects missingness, but is not an unseen nonmissing category. No smoothing,
  reference-vocabulary truncation, or silent dropping of categories occurs.
- Predictions: KS distance, mean-probability difference, predicted-positive-rate
  difference at one shared threshold, and total variation of fixed uniform bins.
  Bin boundaries use integer division; bins are left-inclusive/right-exclusive,
  with 1 included in the last bin. Empty bins remain present. Histogram distance
  depends on binning and complements the unbinned distance.
- Labeled performance: reuse Phase 3 metrics and Phase 4 support rules. Deltas are
  current minus reference, and are null with reasons when either side lacks the
  metric's support. Raw defined scores remain visible. Probability losses are
  lower-is-better; classification/ranking metrics are higher-is-better. Label
  prevalence change is descriptive. Missing labels cause an explicit skipped
  performance section while feature/prediction measurements remain available.

Defaults are 30 rows, 10 actual observations per class, 10 predicted positives,
10 probability bins and at most 100 categories per feature across both snapshots
(including missing). Support equality is sufficient. Numeric support counts use
nonmissing observations. Low support warns; distances remain descriptive and
are not converted into a pass/fail result. Category limits stop the audit instead
of silently truncating output. These policy values are not statistical guarantees.

The KS definition follows the [SciPy reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html).
We calculate only the empirical distance, without continuous-distribution p-values,
multiple-testing claims, or statistical-significance labels. Marginal comparisons
cannot detect every joint/conditional change. Drift does not prove performance
loss or its cause; absence of measured drift does not prove safety. No concept-
drift conclusion, model selection, retraining, threshold tuning or release gate
is produced. Overlapping populations are allowed for sanity checks, but cannot
be treated as independent periods. No time ordering is inferred from file names.

## File adapter and development command

`drift_audit.audit_snapshot_files` reads two CSV files with strict unique headers
and row widths, preserves categorical text, and parses declared numeric roles.
Predictions travel in the same CSV as their associated observations. Hash both
files before and after reading/evaluation, record provenance, and exclusively
create a new aggregate JSON report. No model is deserialized. Hashes detect
accidental changes; they do not authenticate coordinated input replacement.
Reports contain category labels (which may be sensitive), counts and measurements,
not raw records or prediction arrays. Keep reports local and ignored.

```powershell
.venv\Scripts\python scripts/audit_drift.py --reference data/reference_snapshot.csv --current data/current_snapshot.csv --numeric tenure MonthlyCharges TotalCharges --categorical Contract --probability probability --target Churn --output reports/generated/drift-review.json
```

This is a development command for user-prepared snapshots, not the Phase 6 public
orchestrator. Exit 0 means a report was produced, including warnings/skips; exit 2
means invalid inputs or a filesystem error. Use a new output path for every run.

## Verification requirements

Hand-check equal, shifted, tied, unequal-size and missing distributions; unseen
categories and literal missing markers; probability boundaries; score direction
and support exclusions; unlabeled/single-class cases; strict schema/invalid inputs;
input immutability, strict JSON, hash checks and exclusive output. Compare a real
prepared training/reference split descriptively and saved reference predictions
against themselves as a zero-change sanity check. Neither is a production-period
experiment. Controlled defective scenario experiments remain Phase 7.
