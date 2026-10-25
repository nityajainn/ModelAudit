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
No CLI entry point is advertised before its Phase 6 implementation. Future
modules will separate preparation, measurements, policy checks, and reporting.

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
