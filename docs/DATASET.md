# Phase 1 dataset

The development example uses the IBM-hosted Telco Customer Churn CSV, pinned to
revision `d5371f5d83a446ad5673cbcca3b814b926491f8a`.

- [Pinned CSV](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/d5371f5d83a446ad5673cbcca3b814b926491f8a/data/Telco-Customer-Churn.csv)
- Expected SHA-256: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- [Source README](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/README.md)
- [Source repository license](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/d5371f5d83a446ad5673cbcca3b814b926491f8a/LICENSE)

The pinned repository license is Apache 2.0. Its README describes the code pattern
license; it does not establish a separate, dataset-specific redistribution grant.
ModelAudit's MIT license applies to its code and does not relicense the CSV.
This project publishes references and acquisition code, not the dataset or trained
artifacts. Local-only storage is not itself a license grant. The source and license
were reviewed on 2026-10-21 before this implementation's acquisition.

## Schema and split

Exactly 21 columns are required. `customerID` must be unique and nonmissing;
`Churn` is encoded as No=0, Yes=1. Both classes must appear in each partition.
Neither the target nor identifier is a predictor. `Contract` remains available
as a predictor and future segmentation column and must not be missing.

`tenure`, `MonthlyCharges`, and `TotalCharges` are numerical. Blank numerical
cells remain missing until training-only median imputation. Other predictor
columns, including `SeniorCitizen`, are categorical strings. Blank categorical
cells use a reserved missing token during training; source values equal to that
token are rejected. Nonblank category spelling is preserved. Full categorical
domain and numerical-range validation are outside this phase.

Rows are sorted by customer ID before a seed-42 stratified 80/20 train/reference
split and sorted again within each partition. No learned transformation runs
before splitting. Header duplicates, extra/missing columns, ragged rows, invalid
targets, duplicate IDs, and nonfinite/malformed numerical values are rejected.

Prepared CSV hashes, counts, settings, source hash, runtime versions, actual UTC
timestamps, and Git state are recorded locally. Training verifies partition hashes,
counts, labels, and identifier disjointness. These checks detect accidental changes;
the manifest is not a signed guarantee against someone modifying both data and hashes.

## Limits

A single random split is educational evaluation, not temporal validation,
causal evidence, calibration assurance, or a deployment recommendation. No threshold
or hyperparameter search uses the reference set. Repeated reference-set tuning would
bias future conclusions. Each run requires a fresh output directory; partial failed
runs are preserved for inspection and are not silently overwritten.
