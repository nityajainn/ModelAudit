"""Audit raw CSV snapshots before inference, retaining independent blocked evidence."""

import csv
from io import BytesIO
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data_checks import audit_data, CheckPolicy, DataSchema
from .drift import compare_snapshots
from .evaluation import evaluate_binary
from .predictions import positive_probabilities
from .segments import evaluate_segments


def read_snapshot(path, schema):
    """Keep order, duplicate headers and malformed values for the data auditor."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or not rows[0]:
        raise ValueError('CSV requires a header.')
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError('CSV rows must have the same width as the header.')
    frame = pd.DataFrame(rows[1:], columns=rows[0])
    if frame.columns.has_duplicates:
        return frame
    for column in frame:
        values = frame[column].map(lambda x: np.nan if not x.strip() else x)
        if column in schema.numeric or column == schema.target:
            try:
                values = pd.to_numeric(values, errors='raise')
            except (ValueError, TypeError):
                pass  # The auditor reports invalid types; do not silently drop/coerce bad cells.
        elif column not in schema.columns:
            try:
                values = pd.to_numeric(values, errors='raise')
            except (ValueError, TypeError):
                pass
        frame[column] = values
    return frame


def _blocked(reason):
    return {'status':'skipped', 'reason':reason}


def _features(frame, schema, names):
    if frame.empty or frame.columns.has_duplicates or not set(names)<=set(frame.columns):
        return None, 'Predictors require nonempty rows and all selected columns with unique headers.'
    copy = frame[names].copy(deep=True)
    for name in names:
        values = copy[name].dropna()
        if name in schema.numeric:
            if values.dtype.kind not in 'fiu' or not np.isfinite(values.to_numpy(dtype=float)).all():
                return None, f'Predictor {name} requires finite numbers or missing values.'
        elif not values.map(lambda v:isinstance(v,str)).all():
            return None, f'Predictor {name} requires strings or missing values.'
    return copy, None


def _drift(reference, current, schema, predictions, labeled, options):
    comparison = {'features':{}, 'predictions':_blocked('Predictions unavailable.'),
                  'performance':_blocked('Labeled current data and paired predictions are required.'),
                  'reference_rows':len(reference), 'current_rows':len(current) if current is not None else None,
                  'difference_direction':'current_minus_reference', 'policy':options}
    for kind, names in (('numeric',schema.numeric), ('categorical',schema.categorical)):
        for name in names:
            try:
                if (current is None or reference.columns.has_duplicates or current.columns.has_duplicates
                        or name not in reference or name not in current):
                    raise ValueError('Feature requires both snapshots and unique, present columns.')
                result = compare_snapshots(reference[[name]],current[[name]], **{kind:[name]}, **options)
                comparison['features'][name] = result['features'][name]
            except ValueError as exc:
                comparison['features'][name] = {**_blocked(str(exc)), 'kind':kind,
                                               'missingness':{'fraction_difference':None}}
    if current is not None and 'reference' in predictions and 'current' in predictions:
        first = pd.DataFrame({'probability':predictions['reference']})
        second = pd.DataFrame({'probability':predictions['current']})
        measured = compare_snapshots(first,second,probability='probability',**options)
        comparison['predictions'] = measured['predictions']
        if labeled and schema.target in reference and schema.target in current:
            first['label'] = reference[schema.target].to_numpy()
            second['label'] = current[schema.target].to_numpy()
            try:
                measured = compare_snapshots(first,second,probability='probability',target='label',**options)
                comparison['performance'] = measured['performance']
            except ValueError as exc:
                comparison['performance'] = _blocked(str(exc))
    return comparison


def run_raw_checks(config):
    schema = DataSchema(**config['schema'])
    tables = {name:read_snapshot(path,schema) for name,path in config['inputs'].items() if name!='model'}
    results, predictions, blocked = {}, {}, {}
    if 'data' in config['checks']:
        results['data'] = audit_data(tables['train'],tables['reference'],schema,
            CheckPolicy(**config['checks']['data']),current=tables.get('current'),
            current_labeled=config['current_labeled'],inspect_extra_columns=True)
    if config['model'] is not None:
        # Deserialize precisely the bytes whose hash was checked, not a second read of the path.
        payload = Path(config['inputs']['model']).read_bytes()
        if sha256(payload).hexdigest() != config['model']['sha256']:
            raise ValueError('Fitted model checksum mismatch; no deserialization performed.')
        model = joblib.load(BytesIO(payload))
        for name in ('reference','current'):
            if name not in tables:
                continue
            features, reason = _features(tables[name],schema,config['model']['features'])
            if reason:
                blocked[name] = reason
            else:
                predictions[name] = positive_probabilities(model,features)
    for name in ('evaluation','segments'):
        if name not in config['checks']:
            continue
        options = config['checks'][name].copy()
        frame = tables['reference']
        result = {'models':{}}
        reason = blocked.get('reference')
        if schema.target not in frame or frame.columns.has_duplicates:
            reason = 'A unique reference target column is required.'
        if reason is None:
            try:
                labels = frame[schema.target].to_numpy()
                if name=='evaluation':
                    result['models']['fitted'] = evaluate_binary(labels,predictions['reference'],**options)
                else:
                    columns = options.pop('columns')
                    if not set(columns)<=set(frame.columns):
                        raise ValueError('A configured segment column is missing.')
                    result['models']['fitted'] = evaluate_segments(labels,predictions['reference'],frame[columns],**options)
            except ValueError as exc:
                reason = str(exc)
        if reason:
            result.update(_blocked(reason))
        results[name] = result
    if 'drift' in config['checks']:
        results['drift'] = {'comparison':_drift(tables['reference'],tables.get('current'),schema,
            predictions,config['current_labeled'],config['checks']['drift'])}
    coverage = {}
    for name,result in results.items():
        coverage[name] = {'state': 'blocked' if result.get('status')=='skipped' else 'evaluated'}
        if name=='data':
            coverage[name].update(evaluated=sum(result['counts'][s] for s in ('pass','warn','fail')),
                                  blocked=result['counts']['skipped'])
        if name=='drift':
            pieces = list(result['comparison']['features'].values()) + [result['comparison'][n] for n in ('predictions','performance')]
            coverage[name].update(evaluated=sum(p.get('status')!='skipped' for p in pieces),
                                  blocked=sum(p.get('status')=='skipped' for p in pieces))
            if coverage[name]['blocked']:
                coverage[name]['state'] = 'partial' if coverage[name]['evaluated'] else 'blocked'
    return results, coverage
