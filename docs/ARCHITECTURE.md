# Foundation architecture

The `src/modelaudit` package is installed before tests are run, keeping imports
independent of the current directory. Phase 0 has two small modules:

- `config.load_config`: reads one UTF-8 YAML document using a SafeLoader
  subclass; rejects empty/nonmapping roots, duplicate keys, nonstring keys,
  and unsupported object tags. It does not validate a complete audit schema.
- `logging.configure_logging`: configures only the `modelaudit` logger.
  Repeated startup calls do not add duplicate console handlers. Application
  startup must configure logging before launching concurrent workers.

Phase 0 required only PyYAML. Phase 1 adds NumPy, pandas, scikit-learn, and joblib.
Tests use standard-library unittest with the project's runtime dependencies.
Phase 6 exposes the `modelaudit` entry point and `python -m modelaudit`. Preparation,
measurements, policy checks and reporting remain separate modules.

## Phase 1 boundaries

- `artifacts.py`: streaming SHA-256, exclusive JSON outputs, runtime/Git provenance.
- `dataset.py`: pinned acquisition, strict parsing, ID-sorted stratified splitting,
  descriptive audit, and verification of prepared partitions.
- `baselines.py`: separate fitted pipelines, fixed baseline metrics, serialization
  checks, and provenance summaries.
- `scripts/prepare_data.py` and `scripts/train_baselines.py`: small development
  entry points with overwrite protection, not the future validation CLI.

Only deterministic parsing precedes the split. Learned imputation, scaling, and
category vocabularies are fitted exclusively on training rows. Each pipeline
retains preprocessing with its classifier. The reference split is only evaluated.
Future validation checks can consume the preserved IDs, target, Contract column,
and saved predictions without changing this preparation or training stage.

## Phase 2 boundaries

- `data_checks.py`: immutable schema/policy declarations and a read-only two-table
  audit. Each finding separates the check, dataset, status, explanation, and
  aggregate evidence. Invalid data yields failures/skips without automatic repair.
- `data_audit.py`: adapts the existing verified Telco partitions to those checks,
  verifies input fingerprints before and after, and writes an exclusive JSON report
  with actual runtime/Git provenance.
- `scripts/audit_data.py`: development entry point with explicit exit statuses;
  it does not implement the planned Phase 6 public validation CLI.

No new dependencies, fitted transforms, or model inference are introduced.
The general in-memory checks can inspect defective data that the stricter Phase 1
loader would reject. Correlation/category heuristics are review signals, not
causal claims or guarantees that leakage is absent. Reports contain counts and
column names, not raw records. See DATA_VALIDATION.md for thresholds and limits.

## Phase 3 boundaries

- `evaluation.py`: validates aligned binary arrays, computes fixed-threshold and
  ranking metrics, probability scores, uniform calibration bins and a descriptive
  threshold grid. Undefined scores have null values and reasons; no fitting or
  threshold selection occurs.
- `model_audit.py`: verifies Phase 1 summary/partition/prediction hashes and evaluates
  only the two fixed probability arrays. It uses `allow_pickle=False`, never loads
  joblib models, rechecks inputs before writing, and creates an exclusive report.
- `scripts/audit_models.py`: development entry point with path, fixed-threshold and
  bin-count options. Full validation configuration and orchestration remain Phase 6.

No dependencies were added or changed. Existing Phase 1 metrics and artifacts remain
unchanged. Calibration bins use integer-boundary division so decimal boundaries
follow the documented left-inclusive convention; empty bins remain visible.
MODEL_EVALUATION.md defines the metrics, single-class policy and support warnings.

## Phase 4 boundaries

- `segments.py`: evaluates independent categorical groups with the Phase 3 metric
  engine. It records missing groups, exact support, metric-specific comparison
  exclusions, signed differences from overall performance, and supported ranges.
- `segment_audit.py`: evaluates categorical predictor columns on the same verified
  reference rows and saved arrays, defaulting to Contract. It excludes IDs, target
  and numeric columns from the artifact-level grouping options.
- `scripts/audit_segments.py`: local development command with column, threshold
  and support settings; it does not implement full Phase 6 orchestration.

Phase 3's existing artifact checks are shared through internal input-loading and
post-read verification helpers in `model_audit.py`. Its public interface and report
shape remain unchanged; regression tests and the real Phase 3 report agree exactly.
Neither path deserializes models. No dependency versions or training code changed.
Support policies are documented in SEGMENT_EVALUATION.md; there is no threshold
optimization, significance test, intersectional grouping, or fairness decision.

## Phase 5 boundaries

- `drift.py`: strict snapshot roles, numeric empirical CDF distance, categorical
  total variation, explicit missing/new-category accounting, probability distances
  and supported labeled score changes. Tables have their own positional row order;
  index labels do not join populations. No fitting or mutation occurs.
- `drift_audit.py`: reads strict CSV snapshots with predictions/labels in the same
  rows as features. It checks before/after hashes and creates exclusive aggregate
  reports with runtime provenance, without loading models.
- `scripts/audit_drift.py`: development command requiring two named snapshot files
  and explicit column roles. Full configuration/orchestration remain Phase 6.

Labeled comparisons reuse Phase 3 metric definitions and Phase 4 support helpers.
Existing modules, dependencies and saved data/model artifacts are unchanged. Feature
comparisons do not require labels or predictions; absent labels skip performance
measurement, while incomplete/invalid labels fail rather than silently drop rows.
DRIFT_ANALYSIS.md defines distance semantics, directions, support and limitations.
No production chronology, independence or model identity is inferred from a CSV.

## Phase 6 boundaries

- `audit_config.py`: validates version 1 audit schemas on top of the existing safe
  YAML loader, expands defaults and resolves input paths relative to the YAML file.
  Only declared check options are supported; no dynamic imports or expressions.
- `runner.py`: runs enabled data/evaluation/segment/drift adapters sequentially,
  checking configuration and input hashes across the full sequence. Each run owns
  a new output directory. The completion manifest is written last; failures retain
  partial evidence and a failure marker where possible.
- `decisions.py`: collects review/failure locations and evaluates inclusive scalar
  bounds through literal result paths. Undefined/non-numeric/missing measurements
  produce unavailable gates. Failure overrides review, which overrides pass.
- `reporting.py`: renders a summary of measured results, support and all review
  items as escaped Markdown and self-contained HTML. Full evidence remains JSON.
- `cli.py` / `__main__.py`: public `check-config` and `run` commands, help/version,
  package-scoped progress logging and explicit exit codes. The installed command
  is registered in pyproject.toml; no runtime dependencies were added or upgraded.

Phase 2–5 measurement engines and existing development scripts remain unchanged.
The foundation YAML loader retains its generic behavior; its docstring now points
to the separate schema validator. Configuration/gate semantics, output completion
rules and limits are in CLI_REPORTS.md. A complete audit may require review or fail;
only the final valid manifest distinguishes completion from a partial run.

## Phase 7 boundaries

- `scenarios.py`: builds eleven deterministic in-memory scenarios from verified
  Telco tables and paired probability vectors. Copies isolate each transformation.
  It records the unchanged control, data defects, feature shifts, score corruption
  and an artificial small cohort with explicit expected observations and evidence.
- `experiment_suite.py`: verifies the source artifacts, runs the scenarios and a
  checksum rejection exercise on a temporary copy, checks source hashes again,
  and writes aggregate evidence plus a final completion manifest exclusively.
- `experiment_reports.py`: renders escaped Markdown and self-contained HTML,
  keeping expectation success separate from each scenario's audit outcome.
- `cli.py`: adds the `experiments` subcommand to the existing entry point without
  changing ordinary `run` or `check-config` semantics. No dependency changes.

Existing data, metric, drift, support and gate engines are reused without edits.
Malformed data copies intentionally use the in-memory data-check API; real source
artifacts still pass the strict Phase 1 loader and prediction/hash pairing checks.
No model is fitted or deserialized, no source manifest is rewritten, and altered
features are not assigned fabricated model responses. Score transformations are
labeled as manipulations. The protocol and limitations are documented in
CONTROLLED_EXPERIMENTS.md; measurements are in EXPERIMENTS.md.
