"""Read-only data-quality checks with explicit evidence and heuristic limits."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype


@dataclass(frozen=True)
class DataSchema:
    """Numeric predictors, string categories, numeric 0/1 target, optional string ID."""

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    identifier: str | None = None

    def __post_init__(self):
        for field in ("numeric", "categorical"):
            values = getattr(self, field)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"{field} must be a list or tuple of column names.")
            object.__setattr__(self, field, tuple(values))
        names = self.columns
        if (not all(isinstance(name, str) and name.strip() for name in names)
                or len(set(names)) != len(names) or not self.features):
            raise ValueError("Schema roles require unique nonempty names and at least one predictor.")

    @property
    def features(self):
        return self.numeric + self.categorical

    @property
    def columns(self):
        return self.features + (self.target,) + (() if self.identifier is None else (self.identifier,))


@dataclass(frozen=True)
class CheckPolicy:
    missing_fraction: float = 0.05
    correlation_threshold: float = 0.98
    minimum_pairs: int = 20
    maximum_categories: int = 20
    minimum_category_rows: int = 5

    def __post_init__(self):
        for name, lower, upper in (("missing_fraction", 0, 1),
                                   ("correlation_threshold", 0, 1)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not lower <= value <= upper
                    or (name == "correlation_threshold" and value == 0)):
                raise ValueError(f"Invalid {name}.")
            object.__setattr__(self, name, float(value))
        for name in ("minimum_pairs", "maximum_categories", "minimum_category_rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two.")


def _numeric(series):
    return (is_numeric_dtype(series.dtype) and not is_bool_dtype(series.dtype)
            and not is_complex_dtype(series.dtype))


def _strings(series):
    return bool(series.dropna().map(lambda value: isinstance(value, str)).all())


def _normalize(table):
    # Whitespace-only strings count as missing. Never coerce numerical strings.
    result = table.copy(deep=True)
    for name in result.columns:
        blanks = result[name].map(lambda value: isinstance(value, str) and not value.strip())
        if blanks.any():
            result[name] = result[name].mask(blanks)
    return result


def audit_data(train: pd.DataFrame, reference: pd.DataFrame, schema: DataSchema,
               policy: CheckPolicy | None = None) -> dict:
    """Inspect two tables without changing them or fitting a model.

    Findings contain column names and aggregate counts, never source row values.
    Invalid data creates fail/skipped findings. Invalid API arguments raise errors.
    """
    if not isinstance(schema, DataSchema):
        raise TypeError("schema must be a DataSchema.")
    policy = CheckPolicy() if policy is None else policy
    if not isinstance(policy, CheckPolicy):
        raise TypeError("policy must be a CheckPolicy.")
    findings = []

    def add(check, dataset, status, message, **evidence):
        findings.append(dict(check=check, dataset=dataset, status=status,
                             message=message, evidence=evidence))

    tables, valid_columns = {}, {}
    for name, original in (("train", train), ("reference", reference)):
        if not isinstance(original, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame.")
        if not all(isinstance(column, str) and column.strip() for column in original.columns):
            raise ValueError("Input column labels must be nonempty strings.")
        if original.columns.has_duplicates:
            add("schema", name, "fail", "Duplicate column labels prevent unambiguous checks.",
                duplicate_columns=sorted(set(original.columns[original.columns.duplicated()])))
            add("data_checks", name, "skipped", "Checks require unique column labels.")
            continue
        table = _normalize(original)
        tables[name] = table
        missing = sorted(set(schema.columns) - set(table.columns))
        extra = sorted(set(table.columns) - set(schema.columns))
        invalid_types, nonfinite, usable = [], {}, set()
        for column in schema.columns:
            if column not in table:
                continue
            series = table[column]
            numeric = column in schema.numeric or column == schema.target
            if not (_numeric(series) if numeric else _strings(series)):
                invalid_types.append(column)
                continue
            if numeric:
                count = int((~np.isfinite(series.dropna().to_numpy(dtype=float))).sum())
                if count:
                    nonfinite[column] = count
                    continue
            usable.add(column)
        valid_columns[name] = usable
        add("schema", name, "fail" if missing or extra or invalid_types or nonfinite else "pass",
            "Expected columns, scalar types and finite numerical values checked.",
            missing_columns=missing, unexpected_columns=extra,
            invalid_type_columns=invalid_types, nonfinite_counts=nonfinite)
        add("row_count", name, "pass" if len(table) else "fail",
            "A nonempty dataset is required.", rows=len(table))
        for column in schema.columns:
            if column not in table:
                continue
            count = int(table[column].isna().sum())
            fraction = count / len(table) if len(table) else None
            required = column in (schema.target, schema.identifier)
            status = ("skipped" if fraction is None else "fail" if required and count else
                      "warn" if fraction > policy.missing_fraction else "pass")
            add("missingness", name, status, "Missing values counted after blank-string normalization.",
                column=column, count=count, fraction=fraction, required=required,
                warning_threshold=policy.missing_fraction)
        target_ok = False
        if schema.target in usable and len(table):
            labels = table[schema.target]
            invalid = int((labels.notna() & ~labels.isin([0, 1])).sum())
            classes = int(labels.dropna().nunique())
            target_ok = not invalid and not labels.isna().any() and classes == 2
            add("target", name, "pass" if target_ok else "fail",
                "Target must be nonmissing numeric 0/1 with both classes present.",
                invalid_labels=invalid, missing=int(labels.isna().sum()), classes=classes)
        else:
            add("target", name, "skipped", "Target requires a nonempty finite numerical column.")
        if len(table) and set(schema.columns).issubset(usable) and not extra:
            add("duplicate_rows", name, "warn" if table.duplicated().any() else "pass",
                "Exact duplicate rows counted beyond their first occurrence.",
                count=int(table.duplicated().sum()))
        else:
            add("duplicate_rows", name, "skipped", "Requires a nonempty table with valid schema.")
        if len(table) and set(schema.features).issubset(usable):
            count = int(table.duplicated(subset=list(schema.features)).sum())
            add("duplicate_predictors", name, "warn" if count else "pass",
                "Predictor duplicates exclude target and identifier; they are not proof of leakage.",
                count=count)
        else:
            add("duplicate_predictors", name, "skipped", "Requires valid predictors and nonempty data.")
        if schema.identifier is not None:
            if schema.identifier in usable and len(table):
                ids = table[schema.identifier].dropna()
                count = int(ids.duplicated().sum())
                add("duplicate_identifiers", name, "fail" if count else "pass",
                    "Repeated nonmissing identifiers violate the declared unique-ID role.", count=count)
            else:
                add("duplicate_identifiers", name, "skipped", "Requires valid nonempty identifiers.")

        for column in schema.features:
            if not target_ok or column not in usable:
                add("leakage_heuristic", name, "skipped", "Requires a valid predictor and binary target.",
                    column=column)
                continue
            pairs = table[[column, schema.target]].dropna()
            x, y = pairs[column], pairs[schema.target]
            signals, correlation = [], None
            if column in schema.numeric and len(pairs) >= 2 and y.nunique() == 2:
                values, labels = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
                if np.array_equal(values, labels):
                    signals.append("target_copy")
                if np.array_equal(values, 1 - labels):
                    signals.append("target_inverse")
            if len(pairs) < policy.minimum_pairs or x.nunique() < 2 or y.nunique() < 2:
                status = "warn" if signals else "skipped"
                reason = "Insufficient paired rows or variation for association heuristics."
            else:
                reason = "Association is a review signal; it cannot establish the origin of a feature."
                assessed = True
                if column in schema.numeric:
                    # Scale first to avoid overflow for very large finite input values.
                    values = x.to_numpy(dtype=float)
                    scale = float(np.max(np.abs(values)))
                    scaled = values / scale if scale else values
                    assessed = bool(np.unique(scaled).size >= 2)
                    correlation_value = (float(np.corrcoef(scaled, y.to_numpy(dtype=float))[0, 1])
                                         if assessed else None)
                    if correlation_value is not None and math.isfinite(correlation_value):
                        correlation = correlation_value
                        if abs(correlation) >= policy.correlation_threshold:
                            signals.append("high_target_correlation")
                else:
                    groups = pairs.groupby(column, observed=True)[schema.target].agg(["size", "nunique"])
                    assessed = (2 <= len(groups) <= policy.maximum_categories
                                and int(groups["size"].min()) >= policy.minimum_category_rows)
                    if assessed and groups["nunique"].eq(1).all():
                        signals.append("category_determines_target")
                status = "warn" if signals else "pass" if assessed else "skipped"
                if not assessed:
                    reason = "Category support/cardinality or numerical precision prevents this heuristic."
            add("leakage_heuristic", name, status, reason, column=column,
                paired_rows=len(pairs), signals=signals, correlation=correlation)

    def available(columns):
        return all(name in tables and len(tables[name])
                   and set(columns).issubset(valid_columns[name]) for name in ("train", "reference"))

    if schema.identifier is None:
        add("identifier_overlap", "cross_split", "skipped", "No identifier role was declared.")
    elif available([schema.identifier]):
        left = tables["train"][schema.identifier].dropna()
        right = tables["reference"][schema.identifier].dropna()
        count = int(right.isin(left).sum())
        add("identifier_overlap", "cross_split", "fail" if count else "pass",
            "Reference rows whose nonmissing identifiers occur in training.",
            reference_rows=count, distinct_identifiers=int(right[right.isin(left)].nunique()))
    else:
        add("identifier_overlap", "cross_split", "skipped", "Requires usable nonempty identifier columns.")
    if available(schema.features):
        columns = list(schema.features)
        # Deduplicating the training side prevents a many-to-many join expansion.
        marker = "_match"
        while marker in columns:
            marker += "_"
        known = tables["train"][columns].astype(object).drop_duplicates()
        known[marker] = 1
        matched = tables["reference"][columns].astype(object).merge(
            known, how="left", on=columns)
        count = int(matched[marker].notna().sum())
        add("predictor_overlap", "cross_split", "warn" if count else "pass",
            "Reference rows matching training predictors, including matching missing values. "
            "Different customers can legitimately share predictors.", reference_rows=count,
            reference_fraction=count / len(tables["reference"]))
    else:
        add("predictor_overlap", "cross_split", "skipped", "Requires valid predictors in both nonempty tables.")
    counts = {status: sum(item["status"] == status for item in findings)
              for status in ("pass", "warn", "fail", "skipped")}
    status = "fail" if counts["fail"] else "warn" if counts["warn"] else "pass"
    return {"format_version": 1, "status": status, "counts": counts,
            "schema": {"numeric": list(schema.numeric), "categorical": list(schema.categorical),
                       "target": schema.target, "identifier": schema.identifier},
            "policy": {key: getattr(policy, key) for key in policy.__dataclass_fields__},
            "findings": findings,
            "limitations": ["Passing checks do not establish absence of leakage.",
                            "No temporal, provenance, causal or post-outcome feature analysis is performed.",
                            "Target-association heuristics require human review; legitimate predictors can trigger them."]}
