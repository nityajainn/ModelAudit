# Scope correction record through Phase 7

Date: 2026-10-30. These corrections follow the user's request to close the gaps
found by the phase-by-phase audit. They were developed only in ModelAudit-Lab.
The published Phase 0-7 commits retain their original identities and timestamps.
This follow-up does not claim implementation of Phase 8 or Phase 9.

## Corrections

| Phase | Corrected scope |
| --- | --- |
| 0-1 | Existing foundation, preparation and baseline training retained and regression-tested |
| 2 | Optional labeled/unlabeled current checks; explicit prohibited columns; raw CSV schema defects retained as findings; audit-only extra target-copy checks |
| 3 | Trusted, hash-bound fitted inference with binary class/positive-column validation, both-column probability validation and preserved row order |
| 4 | Metric-specific best groups, ties, signed differences and support reasons; loss direction handled separately |
| 5 | Smoothed base-2 Jensen-Shannon distance alongside existing total variation; independent feature drift survives blocked predictions in raw mode |
| 6 | Version 2 raw-data/model configuration, ordinary CLI integration, partial/blocked coverage, model provenance and final input verification |
| 7 | Saved-score verdict regression fixed; clean plus four defective scenarios through ten actual CLI processes using two fitted models |

Version 1, its development commands, existing models and saved predictions remain
available. Existing generated evidence was not overwritten. New reports live in
`reports/generated/phase7-scope-final-20261030/` and remain ignored by Git.

## Verification

- All 267 tests passed with warnings treated as errors in 122.560 seconds.
  This includes all 238 earlier tests and 29 added regression tests.
- The tests run the full ten-audit fitted-model protocol twice and compare CSV
  bytes, results and decisions. They verify output hashes, model reuse, isolated
  mutations, wrong/missing/duplicate/blocked evidence and process failures.
- A dedicated fault-injection regression makes decisions and gates incorrectly
  report pass. The saved-score suite now detects the mismatch.
- Installed console commands were run from outside the checkout. The existing
  trusted Telco Logistic Regression model produced F1 0.5726872246696035, zero
  data failures and a review verdict with existing limitations retained.
- Installed saved-score experiments: 12/12 scenarios matched their updated
  measurement and verdict expectations.
- Installed fitted-model experiments: 10/10 audits matched 196 assertions.
  Clean passes; contamination fails; target-copy leakage needs review; feature
  drift and the local segment defect fail their configured gates, for both models.
- In the segment defect, overall F1 stays .90 while annual-contract recall is 0;
  the other group's recall remains 1. Recall differences are -.90 from overall
  and -1 from the best group. The affected confusion matrix is [[0,50],[50,0]].
- Dependency consistency passed; no dependencies were added or upgraded.

The tests ran locally on Python 3.14.6/Windows. Multi-version CI, coverage gates,
package build/installed-wheel checks belong to Phase 8 and remain unimplemented.

## Explicit differences and boundaries

These are equivalent development capabilities, not copied implementation or
rewritten history. ModelAudit retains exit 1 for review as well as failure;
ModelGate permits warnings with exit 0. The synthetic drift example uses an
explicit failing gate. Both choices are documented and asserted by the controls.

Serialized model loading requires explicit trust and an exact SHA256; checksums
do not establish that an unknown model is safe. Audits never train models. Only
the synthetic experiment generator trains its two clean laboratory models.
Synthetic results are software controls, not real-world accuracy/fairness claims.

The full roadmap now includes Phase 8 (testing, CI and package verification) and
Phase 9 (MLflow integration) as planned. Initial Phase 7 publication is recorded
accurately. These corrections require a separate push approval; no publication
of this follow-up is claimed here.
