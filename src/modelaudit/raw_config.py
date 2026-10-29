"""Version 2 raw-snapshot audit configuration, separate from legacy artifact mode."""

from copy import deepcopy
from dataclasses import asdict
import math
import re

from .config import ConfigError
from .data_checks import DataSchema


def normalize_raw_config(raw, path):
    from .audit_config import _mapping, _path, _options, _finite, CHECK_ORDER
    _mapping(raw, {'version','inputs','schema','model','checks','gates','current_labeled'}, 'Configuration')
    roles = _mapping(raw.get('schema'), {'numeric','categorical','target','identifier'}, 'schema')
    try:
        schema = DataSchema(**roles)
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc
    labeled = raw.get('current_labeled', False)
    if type(labeled) is not bool:
        raise ConfigError('current_labeled must be a boolean.')
    inputs = _mapping(raw.get('inputs'), {'train','reference','current','model'}, 'inputs')
    if not {'train','reference'} <= set(inputs):
        raise ConfigError('Raw audits require train and reference CSV paths.')
    if labeled and 'current' not in inputs:
        raise ConfigError('current_labeled requires current input.')
    checks = _mapping(raw.get('checks'), set(CHECK_ORDER), 'checks')
    if not checks:
        raise ConfigError('Enable at least one check.')
    normalized = {}
    try:
        for name in CHECK_ORDER:
            if name not in checks:
                continue
            supplied = deepcopy(checks[name])
            if name == 'drift':
                _mapping(supplied, {'threshold','minimum_rows','minimum_class_rows','minimum_predicted_positives','maximum_categories','n_bins','smoothing'}, 'drift')
                normalized[name] = _options(name, {**supplied, 'numeric':list(schema.numeric),
                    'categorical':list(schema.categorical), 'reference':'unused', 'current':'unused'}, path.parent)
                for key in ('numeric','categorical','probability','target','reference','current'):
                    normalized[name].pop(key)
            elif name == 'segments':
                _mapping(supplied, {'columns','threshold','minimum_rows','minimum_class_rows','minimum_predicted_positives','maximum_groups'}, 'segments')
                columns = supplied.get('columns', list(schema.categorical[:1]))
                if (not isinstance(columns,list) or not columns or not all(isinstance(c,str) and c in schema.categorical for c in columns)
                        or len(set(columns)) != len(columns)):
                    raise ConfigError('Segment columns must be distinct declared categorical columns.')
                normalized[name] = _options(name, {**supplied,'columns':['Contract']}, path.parent)
                normalized[name]['columns'] = columns
            else:
                normalized[name] = _options(name, supplied, path.parent)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ConfigError(str(exc)) from exc
    needs_model = bool(set(checks) & {'evaluation','segments','drift'})
    model = raw.get('model')
    if needs_model:
        if 'model' not in inputs:
            raise ConfigError('Enabled model checks require inputs.model.')
        _mapping(model, {'sha256','trusted','positive_label','features'}, 'model')
        if model.get('trusted') is not True:
            raise ConfigError('Set model.trusted: true only for a local model you trust; joblib can execute code.')
        if not isinstance(model.get('sha256'),str) or not re.fullmatch('[0-9a-f]{64}', model['sha256']):
            raise ConfigError('model.sha256 must be a lowercase SHA256 fingerprint.')
        if type(model.get('positive_label',1)) is not int or model.get('positive_label',1) != 1:
            raise ConfigError('The numeric binary target uses positive_label=1.')
        features = model.get('features', list(schema.features))
        if not isinstance(features,list) or not features or not all(isinstance(f,str) for f in features) or len(set(features)) != len(features) or any(f not in schema.features for f in features):
            raise ConfigError('model.features must be distinct declared predictors, excluding IDs and target.')
        model = {**model, 'positive_label':1, 'features':features}
    elif model is not None or 'model' in inputs:
        raise ConfigError('Unused model configuration is not accepted.')
    thresholds = {o['threshold'] for o in normalized.values() if 'threshold' in o}
    if len(thresholds)>1:
        raise ConfigError('Model, segment and drift checks must use one common decision threshold.')
    gates = raw.get('gates', [])
    if not isinstance(gates,list):
        raise ConfigError('gates must be a list.')
    clean, names = [], set()
    for gate in gates:
        _mapping(gate, {'name','path','minimum','maximum'}, 'gate')
        name, parts = gate.get('name'), gate.get('path')
        if not isinstance(name,str) or not name.strip() or name in names:
            raise ConfigError('Gate names must be unique nonempty strings.')
        names.add(name)
        if (not isinstance(parts,list) or len(parts)<2 or not all((isinstance(p,str) and p) or (type(p) is int and p>=0) for p in parts)
                or parts[0] not in normalized):
            raise ConfigError('Gate paths must begin at an enabled check.')
        bounds = {k:_finite(gate[k],k) for k in ('minimum','maximum') if k in gate}
        if not bounds or bounds.get('minimum',-math.inf)>bounds.get('maximum',math.inf):
            raise ConfigError('Gate bounds are required and must be ordered.')
        clean.append({'name':name,'path':parts.copy(),**bounds})
    return {'version':2, 'schema':{k:list(v) if isinstance(v,tuple) else v for k,v in asdict(schema).items()}, 'current_labeled':labeled,
            'inputs':{k:_path(v,path.parent,k) for k,v in inputs.items()}, 'model':model,
            'checks':normalized, 'gates':clean}
