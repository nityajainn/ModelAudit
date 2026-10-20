# Foundation architecture

The `src/modelaudit` package is installed before tests are run, keeping imports
independent of the current directory. Phase 0 has two small modules:

- `config.load_config`: reads one UTF-8 YAML document using a SafeLoader
  subclass; rejects empty/nonmapping roots, duplicate keys, nonstring keys,
  and unsupported object tags. It does not validate a complete audit schema.
- `logging.configure_logging`: configures only the `modelaudit` logger.
  Repeated startup calls do not add duplicate console handlers. Application
  startup must configure logging before launching concurrent workers.

PyYAML is the sole runtime dependency. Tests use the Python standard library.
No CLI entry point is advertised before its Phase 6 implementation. Future
modules will separate preparation, measurements, policy checks, and reporting.
