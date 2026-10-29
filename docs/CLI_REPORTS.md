# Phase 6 configuration, command-line workflow and reports

## Acceptance criteria

One versioned configuration must select and configure the existing Phase 2–5
checks, reject ambiguous/unknown options before creating output, run checks in a
fixed order, and produce machine-readable evidence plus readable offline reports.
Inputs must remain unchanged. Existing outputs must never be reused or overwritten.
Incomplete runs must not be mistaken for complete ones. Gate results must never
convert missing or unsupported values to zero or pass. Test installed and module
commands, failure/review exit codes, escaping, hashes and repeatability.

## Run a configured audit

After installing the package, preparing data and training the Phase 1 baselines:

```powershell
.venv\Scripts\python -m pip install -e .
.venv\Scripts\modelaudit check-config --config configs/audit.yaml
.venv\Scripts\modelaudit run --config configs/audit.yaml --output reports/generated/my-audit
```

`python -m modelaudit` provides the same commands. `--version` and `--help` describe
the installed command. `--quiet` on `run` suppresses progress logging, but retains
the final outcome/error. Logging stays scoped to ModelAudit.

`check-config` validates schema and resolves paths; it does not read or approve
input datasets. `run` requires all input files. Neither command downloads data,
fits models, loads serialized estimators, selects thresholds, or pushes anything.
The existing development scripts continue to work independently.

## Configuration version 1

Top-level keys are exactly `version`, optional `inputs`, required `checks` and
optional `gates`. Unknown keys, duplicate YAML keys, unsafe YAML tags/merge keys,
nonfinite numbers, invalid types and unused input paths are rejected. Version must
be integer 1, not boolean. At least one check must be enabled. Omit a check to
disable it; an empty mapping `{}` enables its documented defaults. Null and boolean
check settings are invalid. Configuration does not support environment expansion,
dynamic imports, external plugins or executable expressions.

All input paths are relative to the YAML file's directory; absolute paths also work.
The command's `--output` path is relative to the shell's working directory. Prepared
and model inputs are required only for checks that use them. Run output cannot be
inside those data/model directories and must be an entirely new directory.

Checks always execute in this order:

| Check | Inputs | Options and defaults |
| --- | --- | --- |
| data | inputs.prepared | missing_fraction=0.05, correlation_threshold=0.98, minimum_pairs=20, maximum_categories=20, minimum_category_rows=5 |
| evaluation | inputs.prepared, inputs.models | threshold=0.5, thresholds=[0.1,...,0.9], n_bins=10, minimum_bin_rows=20 |
| segments | inputs.prepared, inputs.models | columns=[Contract], threshold=0.5, minimum_rows=30, minimum_class_rows=10, minimum_predicted_positives=10, maximum_groups=50 |
| drift | reference/current CSV paths within this check | numeric=[], categorical=[], probability=null, target=null, threshold=0.5, minimum_rows=30, minimum_class_rows=10, minimum_predicted_positives=10, maximum_categories=100, n_bins=10 |

The data check uses the Phase 1 Telco schema. Segment columns must be categorical
Telco predictors. Drift uses explicitly declared CSV roles from Phase 5; at least
one feature or probability role is needed. Labels, when present, must be complete.
`configs/drift.yaml` illustrates snapshot inputs that the user must prepare.
Both example configs are valid schemas; only audit.yaml can run directly on existing
Phase 1 artifacts. No imaginary production snapshots are included.

Bounds follow the existing engines: fractions/thresholds in [0,1], correlation
threshold in (0,1], data integer policies at least 2, other support counts positive,
bin count 1–100, category/group limits 1–1000, and threshold lists nonempty, strictly
increasing and at most 201 entries. The normalized configuration, including all
defaults and resolved paths, is recorded in report.json.

## Gates and outcomes

`gates` is a list of named rules over numeric result fields. `path` is a literal
list of mapping keys and nonnegative list indices rooted at an enabled check.
There is no expression language or evaluation of code. Supply inclusive `minimum`,
`maximum`, or both; names must be unique and minimum cannot exceed maximum.

For example, this is a syntax illustration with a user-chosen bound, not a recommended
release criterion:

```yaml
gates:
  - name: Required ROC AUC
    path: [evaluation, models, logistic_regression, ranking, roc_auc]
    minimum: 0.8
```

The referenced value must be a finite numeric scalar; missing paths, nulls,
booleans, strings and containers produce `unavailable`, never a passing value.
Gates may address support counts or supported differences. Always interpret raw
metrics with their support warnings; gates do not erase those warnings. A bad
path whose check is disabled is a configuration error; an absent field under an
enabled check is reported as unavailable evidence. No default metric gates are
asserted by the shipped examples.

| Outcome / event | Exit code | Meaning |
| --- | ---: | --- |
| pass | 0 | Configured checks completed, no review items/data failures, and all configured gates passed |
| review | 1 | Checks completed, but warnings/skips/undefined evidence or unavailable gates need attention |
| fail | 1 | Checks completed with data failures or a failed configured bound; review evidence remains visible |
| execution/configuration error | 2 | No completed audit; invalid inputs, configuration, usage or filesystem failure |

Failure takes precedence over review, which takes precedence over pass. Each warning
or undefined-result location is retained; review-item counts are locations, not
counts of unique root causes. Disabled checks are listed only as absent, not passed.
Passing configured checks is not deployment approval or evidence of no leakage,
bias, drift or production risk. Unit-test failure fixtures remain distinct from
the Phase 7 experiment suite.

## Output and integrity contract

A successful run directory contains the enabled `data.json`, `evaluation.json`,
`segments.json`, `drift.json` files, `report.json`, `summary.md`, `summary.html`, and
`manifest.json`. The final manifest marks `status: complete`, records the outcome
and exit code, and hashes every other output. Verify these hashes before consuming
a stored run. Completion does not imply a pass outcome.

The runner fingerprints the configuration and all relevant input files before and
after the full check sequence, in addition to each adapter's checks. It records
actual runtime/Git context and never deserializes a model. These hashes catch
accidental changes, not coordinated malicious replacement. Existing files and
directories are not overwritten, even if they contain a failed earlier run.

If a check or report write fails, partial files are retained in the newly created
directory and failure.json is written when possible. Without a readable completion
manifest whose hashes match, treat the directory as incomplete. An interruption or
disk error can prevent even the failure marker. Use a new output path to retry;
the tool never cleans up or reuses an old run.

HTML is self-contained, uses no scripts or remote assets, and escapes all dynamic
content. Markdown also escapes labels that could become links, images or HTML.
Summaries show selected metrics, support restrictions and every review/failure
item; the JSON files retain full calibration bins, threshold sweeps and comparisons.
Six significant digits are shown for readability, with full precision retained in
JSON. Unavailable values remain labeled. Reports can contain local paths/category
names and are ignored by Git; review them before sharing.

The installed entry point follows the [Python packaging guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/).
Command parsing uses [Python argparse](https://docs.python.org/3.14/library/argparse.html).
No dependency additions or upgrades are needed for this phase.

## Phase 7 experiment subcommand

`modelaudit experiments --prepared data/prepared --models models/baseline --output
reports/generated/my-experiments` runs the fixed controlled scenario suite on
copies. It does not change the version 1 audit configuration schema or ordinary
`run` results. Its exit code 0 means every specified experiment expectation was
met, 1 means an unmet expectation in a completed suite, and 2 means execution/input
failure. This is distinct from model-audit pass/review/fail outcomes. See
[the controlled experiment protocol](CONTROLLED_EXPERIMENTS.md) before interpreting
the generated summary and per-case evidence.
