"""Capture available estimator descriptions without inventing training provenance."""

import math


def describe_model(model):
    classifier = getattr(model, "named_steps", {}).get("classifier", model)
    parameters = classifier.get_params(deep=False) if hasattr(classifier, "get_params") else {}
    return {
        "pipeline": type(model).__name__,
        "classifier": type(classifier).__name__,
        "parameters": {
            key: value
            for key, value in parameters.items()
            if value is None
            or type(value) in (bool, int, str)
            or type(value) is float
            and math.isfinite(value)
        },
    }
