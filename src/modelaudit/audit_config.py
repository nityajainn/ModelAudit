"""Versioned audit configuration; validate every option before creating outputs."""

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

from .config import ConfigError, load_config
from .data_checks import CheckPolicy
from .dataset import CATEGORICAL
from .drift import _positive_integer, _roles
from .evaluation import DEFAULT_THRESHOLDS, _threshold, _vector

CHECK_ORDER = ("data", "evaluation", "segments", "drift")
DEFAULTS = {
    "data": asdict(CheckPolicy()),
    "evaluation": {"threshold": 0.5, "thresholds": list(DEFAULT_THRESHOLDS),
                   "n_bins": 10, "minimum_bin_rows": 20},
    "segments": {"columns": ["Contract"], "threshold": 0.5, "minimum_rows": 30,
                 "minimum_class_rows": 10, "minimum_predicted_positives": 10, "maximum_groups": 50},
    "drift": {"numeric": [], "categorical": [], "probability": None, "target": None,
              "threshold": 0.5, "minimum_rows": 30, "minimum_class_rows": 10,
              "minimum_predicted_positives": 10, "maximum_categories": 100, "n_bins": 10},
}


def _mapping(value, allowed, name):
    if not isinstance(value, dict) or any(key not in allowed for key in value):
        raise ConfigError(f"{name} must be a mapping with only these keys: {', '.join(sorted(allowed))}.")
    return value


def _path(value, base, name):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ConfigError(f"{name} must be a nonempty filesystem path.")
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def _finite(value, name):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigError(f"{name} must be a finite number, not a boolean or string.")
    return float(value)


def _options(name, supplied, base):
    allowed = set(DEFAULTS[name]) | ({"reference", "current"} if name == "drift" else set())
    _mapping(supplied, allowed, f"checks.{name}")
    options = {**deepcopy(DEFAULTS[name]), **deepcopy(supplied)}
    if name == "data":
        return asdict(CheckPolicy(**options))
    options["threshold"] = _threshold(options["threshold"])
    for key in ("minimum_rows", "minimum_class_rows", "minimum_predicted_positives", "minimum_bin_rows"):
        if key in options:
            options[key] = _positive_integer(options[key], key)
    for key, maximum in (("n_bins", 100), ("maximum_groups", 1000), ("maximum_categories", 1000)):
        if key in options:
            options[key] = _positive_integer(options[key], key, maximum)
    if name == "evaluation":
        if not isinstance(options["thresholds"], list):
            raise ConfigError("thresholds must be a list.")
        for value in options["thresholds"]:
            _threshold(value)
        grid = _vector(options["thresholds"], "Threshold grid")
        if (len(grid) > 201 or ((grid < 0) | (grid > 1)).any()
                or any(right <= left for left, right in zip(grid, grid[1:]))):
            raise ConfigError("thresholds must be strictly increasing values in [0,1], at most 201.")
        options["thresholds"] = grid.tolist()
    if name == "segments":
        columns = options["columns"]
        if (not isinstance(columns, list) or not columns
                or not all(isinstance(item, str) and item in CATEGORICAL for item in columns)
                or len(set(columns)) != len(columns)):
            raise ConfigError("segments.columns must list distinct categorical Telco predictors.")
    if name == "drift":
        _roles(options["numeric"], options["categorical"], options["probability"], options["target"])
        for key in ("reference", "current"):
            options[key] = _path(options.get(key), base, f"drift.{key}")
    return options


def load_audit_config(path) -> dict:
    """Resolve input paths relative to YAML, reject unknown keys and expand defaults.

    No input files or models are loaded here. No shell/environment interpolation,
    dynamic imports, plugins or executable expressions are supported.
    """
    path = Path(path).resolve()
    raw = load_config(path)
    _mapping(raw, {"version", "inputs", "checks", "gates"}, "Configuration")
    if type(raw.get("version")) is not int or raw["version"] != 1:
        raise ConfigError("version must be the integer 1.")
    checks = _mapping(raw.get("checks"), set(CHECK_ORDER), "checks")
    if not checks:
        raise ConfigError("Enable at least one check with an options mapping ({} uses defaults).")
    inputs = _mapping(raw.get("inputs", {}), {"prepared", "models"}, "inputs")
    needed = set()
    if set(checks) & {"data", "evaluation", "segments"}:
        needed.add("prepared")
    if set(checks) & {"evaluation", "segments"}:
        needed.add("models")
    if set(inputs) != needed:
        raise ConfigError(f"inputs must contain exactly the required paths: {', '.join(sorted(needed)) or '(none)' }.")
    try:
        normalized = {name: _options(name, checks[name], path.parent) for name in CHECK_ORDER if name in checks}
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigError(str(exc)) from exc
    gates = raw.get("gates", [])
    if not isinstance(gates, list):
        raise ConfigError("gates must be a list.")
    clean_gates, names = [], set()
    for gate in gates:
        _mapping(gate, {"name", "path", "minimum", "maximum"}, "gate")
        name, parts = gate.get("name"), gate.get("path")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ConfigError("Gate names must be unique nonempty strings.")
        names.add(name)
        if (not isinstance(parts, list) or len(parts) < 2
                or not all((isinstance(part, str) and bool(part)) or (type(part) is int and part >= 0) for part in parts)
                or parts[0] not in normalized):
            raise ConfigError("Gate path must begin with an enabled check and contain keys/nonnegative list indices.")
        bounds = {key: _finite(gate[key], f"gate.{key}") for key in ("minimum", "maximum") if key in gate}
        if not bounds or bounds.get("minimum", -math.inf) > bounds.get("maximum", math.inf):
            raise ConfigError("Each gate needs minimum and/or maximum, with minimum <= maximum.")
        clean_gates.append({"name": name, "path": parts.copy(), **bounds})
    return {"version": 1, "inputs": {key: _path(value, path.parent, f"inputs.{key}") for key, value in inputs.items()},
            "checks": normalized, "gates": clean_gates}
