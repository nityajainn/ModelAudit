# Phase 7 controlled experiment protocols

The original saved-score protocol below is retained. The fitted-model CLI
protocol added after the scope review is described in [RAW_AUDITS.md](RAW_AUDITS.md).
It adds five scenarios across two models, using ten ordinary audit subprocesses.
Both protocols now verify expected decisions as well as measurements.

## Purpose and acceptance criteria

Exercise the implemented checks on an unchanged control and explicit, reproducible
defects. Use the verified Phase 1 Telco partitions and saved probability arrays.
Mutate only independent in-memory copies, except for the checksum exercise, which
uses a temporary copy of the six required source artifacts. Never retrain, load
joblib models, rewrite input manifests to legitimize corruption, or change sources.

Record transformations, parameters, full results, expected observations and actual
outcomes. Run the suite twice and compare measurements excluding runtime context.
Verify source hashes before/after and hash every generated report. Preserve prior
reports. A successful experiment means its specified observation occurred, not
that the injected data/model is acceptable. Unexpected errors must fail the run,
not be counted as a successful detector response.

## Fixed scenarios

All selection uses the verified CSV row order. No random generator or invented
date is used. Let N be reference rows and K be min(25, N); identifier overlap also
limits K to training rows. Missingness selects ceil(0.30*N) currently observed
MonthlyCharges rows and requires enough observations. No row values/IDs are
included in published aggregate results.

| ID | Transformation | Expected observation |
| --- | --- | --- |
| control | No change; each reference prediction snapshot compared with itself | Zero feature/prediction distances and zero defined performance deltas; existing baseline warnings retained |
| missingness | Set selected MonthlyCharges values to missing in a reference copy | Missing count increases by the selected count and the missingness warning triggers |
| duplicate_rows | Append first K reference rows to a copy | Exact duplicate rows warn; repeated identifiers fail |
| identifier_overlap | Replace first K reference IDs with first K training IDs | Cross-split ID overlap fails with K overlapping rows |
| target_copy | Add simulated_target_copy equal to Churn to both copied partitions; declare it numeric | Target-copy leakage signals in both partitions |
| missing_column | Remove Contract from a reference copy | Schema failure explicitly names Contract |
| numeric_shift | Add 100 to each observed reference MonthlyCharges value | Nonzero empirical CDF distance; missingness unchanged |
| unseen_category | Replace every reference Contract with simulated-new-contract | Total variation 1 and unseen-category fraction 1 |
| inverted_probabilities | Replace each saved probability with 1-p; features/labels unchanged | Current ROC AUC plus original ROC AUC equals 1 (ties handled); record measured score change |
| permuted_probabilities | Rotate each probability vector by max(1, floor(N/3)) rows; keep labels in place | Prediction CDF and histogram distances remain exactly zero; record labeled performance changes |
| sparse_segment | Select first positive and up to 30 first negatives into one artificial cohort | One-positive support warning and unavailable supported ROC-AUC difference |
| corrupt_checksum | Append bytes to the copied Logistic Regression .npy without updating its copied summary | Exact probability-checksum mismatch is rejected, with no successful model report |

Control input must pass the Phase 2 structural requirements, with warnings allowed.
It must contain both classes, valid paired predictions, and no pre-existing
simulated_target_copy feature or simulated-new-contract level. These prerequisites
are checked before writing the experiment directory.

The data-quality cases call the Phase 2 in-memory API deliberately: malformed
copies cannot pass the stricter Phase 1 artifact loader. This does not bypass or
change integrity checks on real inputs. Feature-only cases do not make predictions
on altered features and do not claim a model response to them. Probability cases
are explicit manipulations of saved scores, not outputs from newly fitted models.
The sparse cohort is selected using labels solely to exercise support handling;
it is artificial and cannot support population-level fairness conclusions.

## Demonstration gates and interpretation

Numeric-shift and unseen-category cases use maximum distance 0.10 as explicit
demonstration bounds. Prediction cases use minimum supported ROC-AUC difference
-0.05; the permutation case also uses maximum prediction CDF distance 0.10. These
are fixed experimental bounds, not recommended production thresholds. A group
support gate requests the sparse cohort's supported ROC-AUC difference; it must
be unavailable. Phase 6 decisions retain all warnings and undefined values.

The inversion identity is an algebraic expectation; whether an audit fails its
performance gate depends on original quality and support. The permutation case
does not promise a particular deterioration for every input. Its guaranteed
distribution-level blind spot is separated from the actual labeled results.
Likewise, the numeric shift must change the empirical distribution, but there is
no universal guarantee that every input exceeds the demonstration bound.
The inversion check uses floating-point tolerance; extreme scores that collapse
to additional ties under subtraction can legitimately leave that expectation unmet.

Each scenario records `expectations` with `met` booleans, `expectations_met`, and
`audit_outcome` separately. Most intentional defects should yield review/fail or
rejection. A suite whose expectations all hold is not a clean-data or model-safety
certificate. An accepted corrupt file is an unmet expectation, not a success.

## Run and preserve evidence

```powershell
.venv\Scripts\modelaudit experiments --prepared data/prepared --models models/baseline --output reports/generated/my-experiments
```

`python -m modelaudit experiments` is equivalent. The existing editable installation
uses this new subcommand without dependency changes. `--quiet` suppresses progress
logging, retaining the final outcome. Exit 0 means all experiment expectations
were met; exit 1 means a completed suite has an unmet expectation; exit 2 means
invalid inputs or an execution failure. These meanings differ from the ordinary
`run` command's model-audit decision and are printed explicitly.

Output must be a new directory outside prepared/model directories. It contains
one JSON evidence file per scenario, summary.json, summary.md, summary.html and a
final manifest.json with output hashes. Summary JSON links scenario IDs to their
evidence filenames and records real runtime/Git provenance and source hashes.
Neither raw data nor arrays/models are exported. HTML is self-contained and
escaped. The manifest's completion status is distinct from expectation success.
Failed runs retain partial evidence and failure.json where possible; without a
valid completion manifest, treat a run as incomplete. Never overwrite a previous
run to retry it.

Checksums detect accidental changes, not coordinated replacement. This finite
scenario suite cannot establish sensitivity/specificity on real incidents, cover
all joint/conditional drift, establish causal effects, or prove absence of leakage
or bias. Source-data limitations and prior overlap warnings remain relevant.
Phase 7 does not complete the full roadmap: Phase 8 testing/CI/package verification
and Phase 9 MLflow remain planned. These controls do not establish production readiness.

For the positive-class score convention see [scikit-learn ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
For background on leakage beyond this deliberately obvious target-copy scenario,
see [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls).
