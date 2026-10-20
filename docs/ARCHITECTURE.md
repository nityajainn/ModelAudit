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
