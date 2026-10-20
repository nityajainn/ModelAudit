"""Read configuration without constructing arbitrary Python objects."""

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The supplied file is not a supported configuration document."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigError("Configuration keys must be strings.")
        if key in result:
            raise ConfigError(f"Duplicate configuration key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one nonempty YAML mapping; full schema validation is Phase 6.

    YAML merge keys and duplicate keys are intentionally rejected.
    File access errors remain OSError so callers can distinguish them.
    """
    try:
        with Path(path).open(encoding="utf-8") as stream:
            result = yaml.load(stream, Loader=_UniqueKeyLoader)
    except (yaml.YAMLError, UnicodeError) as exc:
        raise ConfigError("Cannot read a valid UTF-8 YAML document.") from exc
    if not isinstance(result, dict) or not result:
        raise ConfigError("Configuration must be a nonempty mapping.")
    return result
